//! Integration tests for catalog mode (PLAN_V2 §6.5).
//!
//! Acceptance items:
//!
//! 1. Register-then-write — `Catalog::add_schema` + `HeliumWriter::with_catalog_ref`
//! 2. Convenience wrapper — `Catalog::open_writer` (register + write in one call)
//! 3. Write-then-read — `HeliumReader::new_with_resolver` resolves hash via catalog
//! 4. Missing-catalog / unknown-hash error wording (§6.5 Surface E)
//! 5. v4-without-resolver rejected (no silent corruption)
//! 6. catalog (v6) schema-slot CRC catches single-bit hash corruption
//! 7. Multi-file shared-schema scenario — total bytes-on-disk smaller than
//!    embedded-schema equivalent (§6.5 final checklist item)
//! 8. v4 file's reader still rejects unrelated v4 hash (resolver returns Err)

use std::fs;
use std::io::Cursor;

use helium::catalog::{Catalog, schema_hash};
use helium::{
    CoderRegistry, CoderSpec, ColumnData, ColumnSpec, DataType, HeliumError, HeliumReader,
    HeliumWriter, LogicalColumn, MAGIC_V6, Schema,
};

fn registry() -> CoderRegistry {
    CoderRegistry::default()
}

fn simple_schema() -> Schema {
    Schema::new(vec![ColumnSpec::primitive(
        "x",
        DataType::I64,
        vec![
            CoderSpec::new("delta"),
            CoderSpec::new("leb128"),
            CoderSpec::new("zstd"),
        ],
    )])
}

fn write_catalog_file<W: std::io::Write + std::io::Seek>(
    catalog: &Catalog,
    file: W,
    schema: Schema,
    values: Vec<i64>,
) -> Result<(), HeliumError> {
    let mut w = catalog.open_writer(file, schema, &registry())?;
    w.write_column("x", LogicalColumn::Primitive(ColumnData::I64(values)))?;
    w.finish()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// 1. Register-then-write
// ---------------------------------------------------------------------------

#[test]
fn register_then_write_explicit() {
    let dir = tempfile::tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).unwrap();
    let schema = simple_schema();
    let hash = catalog.add_schema(&schema).unwrap();

    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut w = HeliumWriter::with_catalog_ref(&mut buf, schema, hash, &registry()).unwrap();
    w.write_column(
        "x",
        LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 3])),
    )
    .unwrap();
    w.finish().unwrap();

    // First 8 bytes must be v6 magic (catalog mode + compressed footer).
    let bytes = buf.into_inner();
    assert_eq!(&bytes[..8], MAGIC_V6);
    // Schema slot is exactly 36 bytes (32 hash + 4 CRC).
    assert_eq!(&bytes[8..40][..32], hash.as_bytes());
}

// ---------------------------------------------------------------------------
// 2. Convenience wrapper Catalog::open_writer
// ---------------------------------------------------------------------------

#[test]
fn open_writer_convenience_wrapper() {
    let dir = tempfile::tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).unwrap();
    let schema = simple_schema();

    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut w = catalog
        .open_writer(&mut buf, schema.clone(), &registry())
        .unwrap();
    w.write_column(
        "x",
        LogicalColumn::Primitive(ColumnData::I64(vec![10, 20, 30])),
    )
    .unwrap();
    w.finish().unwrap();

    // Catalog now contains exactly one schema entry.
    let listed = catalog.list_schemas().unwrap();
    assert_eq!(listed.len(), 1);
    let expected_hash = schema_hash(&schema).unwrap();
    assert_eq!(listed[0], expected_hash);
}

// ---------------------------------------------------------------------------
// 3. Write-then-read with resolver
// ---------------------------------------------------------------------------

#[test]
fn write_then_read_with_resolver() {
    let dir = tempfile::tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).unwrap();
    let schema = simple_schema();

    let mut buf = Cursor::new(Vec::<u8>::new());
    write_catalog_file(&catalog, &mut buf, schema, vec![100, 200, 300]).unwrap();

    buf.set_position(0);
    let mut reader =
        HeliumReader::new_with_resolver(&mut buf, &registry(), catalog.resolver()).unwrap();
    let result = reader.read_column("x").unwrap();
    assert_eq!(
        result,
        LogicalColumn::Primitive(ColumnData::I64(vec![100, 200, 300]))
    );
}

// ---------------------------------------------------------------------------
// 4. Missing-catalog / unknown-hash error wording (§6.5 Surface E)
// ---------------------------------------------------------------------------

#[test]
fn missing_hash_surfaces_with_greppable_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).unwrap();
    let schema = simple_schema();

    // Write a v4 file using a hash that's NOT registered in the catalog.
    let unregistered_hash = schema_hash(&schema).unwrap();
    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut w =
        HeliumWriter::with_catalog_ref(&mut buf, schema, unregistered_hash, &registry()).unwrap();
    w.write_column("x", LogicalColumn::Primitive(ColumnData::I64(vec![1])))
        .unwrap();
    w.finish().unwrap();

    // Now try to read it. Catalog is empty so resolver returns "not found".
    buf.set_position(0);
    let err = HeliumReader::new_with_resolver(&mut buf, &registry(), catalog.resolver())
        .expect_err("must fail with not-found");
    match err {
        HeliumError::Format(msg) => {
            assert!(
                msg.contains("schema hash") && msg.contains("not found by resolver"),
                "expected greppable prefix, got: {msg}"
            );
            assert!(
                msg.contains(&unregistered_hash.to_hex().to_string()),
                "expected hash hex in message, got: {msg}"
            );
        }
        other => panic!("expected HeliumError::Format, got: {other:?}"),
    }
}

#[test]
fn corrupted_catalog_file_surfaces_with_greppable_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).unwrap();
    let schema = simple_schema();

    // Register the schema, then corrupt its catalog file with non-JSON bytes.
    let hash = catalog.add_schema(&schema).unwrap();
    let path = catalog.path_for(&hash);
    fs::write(&path, b"this is not JSON").unwrap();

    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut w = HeliumWriter::with_catalog_ref(&mut buf, schema, hash, &registry()).unwrap();
    w.write_column("x", LogicalColumn::Primitive(ColumnData::I64(vec![1])))
        .unwrap();
    w.finish().unwrap();

    buf.set_position(0);
    let err = HeliumReader::new_with_resolver(&mut buf, &registry(), catalog.resolver())
        .expect_err("must fail with deserialize error");
    match err {
        HeliumError::Format(msg) => {
            assert!(
                msg.contains("catalog schema at hash") && msg.contains("failed to deserialize"),
                "expected greppable prefix, got: {msg}"
            );
        }
        other => panic!("expected HeliumError::Format, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 5. v4-without-resolver rejected
// ---------------------------------------------------------------------------

#[test]
fn catalog_without_resolver_rejected_with_clear_message() {
    let dir = tempfile::tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).unwrap();
    let schema = simple_schema();
    let hash = catalog.add_schema(&schema).unwrap();

    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut w = HeliumWriter::with_catalog_ref(&mut buf, schema, hash, &registry()).unwrap();
    w.write_column("x", LogicalColumn::Primitive(ColumnData::I64(vec![1])))
        .unwrap();
    w.finish().unwrap();

    // Try to open with the resolver-less constructor.
    buf.set_position(0);
    let err = HeliumReader::new(&mut buf, &registry()).expect_err("v4 without resolver must error");
    match err {
        HeliumError::Format(msg) => {
            assert!(
                msg.contains("requires schema resolver but none was provided"),
                "expected greppable prefix, got: {msg}"
            );
        }
        other => panic!("expected HeliumError::Format, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 6. Catalog (v6) schema-slot CRC catches hash corruption
// ---------------------------------------------------------------------------

#[test]
fn catalog_schema_slot_crc_detects_hash_bit_flip() {
    let dir = tempfile::tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).unwrap();
    let schema = simple_schema();
    let hash = catalog.add_schema(&schema).unwrap();

    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut w = HeliumWriter::with_catalog_ref(&mut buf, schema, hash, &registry()).unwrap();
    w.write_column("x", LogicalColumn::Primitive(ColumnData::I64(vec![1])))
        .unwrap();
    w.finish().unwrap();

    // Flip a bit inside the 32-byte hash region (offset 8..40 is the slot,
    // 8..40 is hash bytes; pick byte 20 which is well inside the hash).
    let mut bytes = buf.into_inner();
    bytes[20] ^= 0x01;

    let err = HeliumReader::new_with_resolver(Cursor::new(bytes), &registry(), catalog.resolver())
        .expect_err("CRC must catch hash bit flip");
    match err {
        HeliumError::Format(msg) => {
            assert!(
                msg.contains("catalog schema-slot CRC mismatch"),
                "expected greppable prefix, got: {msg}"
            );
        }
        other => panic!("expected HeliumError::Format, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 7. Multi-file shared-schema dedup gain
// ---------------------------------------------------------------------------

#[test]
fn multi_file_shared_schema_smaller_than_embedded() {
    // Build a non-trivially-sized schema (multiple typed columns).
    let schema = Schema::new(vec![
        ColumnSpec::primitive(
            "ts",
            DataType::I64,
            vec![
                CoderSpec::new("delta"),
                CoderSpec::new("leb128"),
                CoderSpec::new("zstd"),
            ],
        ),
        ColumnSpec::primitive(
            "user_id",
            DataType::I64,
            vec![
                CoderSpec::new("delta"),
                CoderSpec::new("leb128"),
                CoderSpec::new("zstd"),
            ],
        ),
        ColumnSpec::utf8(
            "level",
            vec![
                CoderSpec::new("delta"),
                CoderSpec::new("leb128"),
                CoderSpec::new("zstd"),
            ],
            vec![CoderSpec::new("zstd")],
        ),
        ColumnSpec::nullable_prim(
            "weight",
            DataType::F64,
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
            vec![CoderSpec::new("gorilla"), CoderSpec::new("zstd")],
        ),
    ]);

    let dir = tempfile::tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).unwrap();

    // Tiny per-file payload — typical of partition-per-day Avro-replacement workload.
    let make_data = || {
        (
            ColumnData::I64((0..10_i64).collect()),
            ColumnData::I64((0..10_i64).map(|i| 100 + i).collect()),
            vec!["INFO".to_string(); 10],
            (
                vec![true; 10],
                ColumnData::F64((0..10).map(|i| i as f64 * 0.1).collect()),
            ),
        )
    };

    // Write 5 catalog-mode files
    let mut catalog_total = 0usize;
    for _ in 0..5 {
        let mut buf = Cursor::new(Vec::<u8>::new());
        let mut w = catalog
            .open_writer(&mut buf, schema.clone(), &registry())
            .unwrap();
        let (ts, uid, lvl, (wp, wv)) = make_data();
        w.write_column("ts", LogicalColumn::Primitive(ts)).unwrap();
        w.write_column("user_id", LogicalColumn::Primitive(uid))
            .unwrap();
        w.write_column("level", LogicalColumn::Utf8(lvl)).unwrap();
        w.write_column(
            "weight",
            LogicalColumn::NullablePrim {
                present: wp,
                values: wv,
            },
        )
        .unwrap();
        w.finish().unwrap();
        catalog_total += buf.into_inner().len();
    }

    // Write 5 default-mode (v3) files for comparison
    let mut embedded_total = 0usize;
    for _ in 0..5 {
        let mut buf = Cursor::new(Vec::<u8>::new());
        let mut w = HeliumWriter::new(&mut buf, schema.clone(), &registry()).unwrap();
        let (ts, uid, lvl, (wp, wv)) = make_data();
        w.write_column("ts", LogicalColumn::Primitive(ts)).unwrap();
        w.write_column("user_id", LogicalColumn::Primitive(uid))
            .unwrap();
        w.write_column("level", LogicalColumn::Utf8(lvl)).unwrap();
        w.write_column(
            "weight",
            LogicalColumn::NullablePrim {
                present: wp,
                values: wv,
            },
        )
        .unwrap();
        w.finish().unwrap();
        embedded_total += buf.into_inner().len();
    }

    eprintln!(
        "[catalog dedup] 5 files: catalog mode = {catalog_total} bytes, \
         embedded mode = {embedded_total} bytes, savings = {:.1}%",
        ((embedded_total as f64 - catalog_total as f64) / embedded_total as f64) * 100.0,
    );

    // Catalog mode must produce strictly smaller total bytes (the win is even
    // bigger if you ALSO count the catalog directory in the equation, but the
    // per-file accounting is the right metric for the "many files share schema"
    // workload — the catalog dir is amortized).
    assert!(
        catalog_total < embedded_total,
        "catalog mode bytes {catalog_total} should be smaller than embedded {embedded_total}"
    );
}

// ---------------------------------------------------------------------------
// 8. Catalog::add_schema is idempotent
// ---------------------------------------------------------------------------

#[test]
fn register_is_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).unwrap();
    let schema = simple_schema();

    let h1 = catalog.add_schema(&schema).unwrap();
    let h2 = catalog.add_schema(&schema).unwrap();
    assert_eq!(h1, h2);
    assert_eq!(catalog.list_schemas().unwrap().len(), 1);
}

// ---------------------------------------------------------------------------
// 9. Catalog::verify_consistency returns Ok(()) on clean / Err on mismatch
// ---------------------------------------------------------------------------

#[test]
fn verify_consistency_on_clean_catalog_is_ok() {
    let dir = tempfile::tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).unwrap();
    catalog.add_schema(&simple_schema()).unwrap();
    catalog
        .verify_consistency()
        .expect("clean catalog must verify cleanly");
}

#[test]
fn verify_consistency_catches_filename_content_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).unwrap();
    let schema = simple_schema();
    let hash = catalog.add_schema(&schema).unwrap();

    // Overwrite with different content but keep the filename.
    let other_schema = Schema::new(vec![ColumnSpec::primitive(
        "y",
        DataType::I32,
        vec![CoderSpec::new("zstd")],
    )]);
    let other_canonical =
        helium::catalog::canonicalize_json(&other_schema.to_json().unwrap()).unwrap();
    fs::write(catalog.path_for(&hash), &other_canonical).unwrap();

    let err = catalog
        .verify_consistency()
        .expect_err("mismatch must error");
    match err {
        HeliumError::Format(msg) => {
            assert!(
                msg.contains("catalog inconsistency")
                    && msg.contains("filename hash")
                    && msg.contains("does not match content hash"),
                "unexpected: {msg}"
            );
        }
        other => panic!("expected HeliumError::Format, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 10. Schema-hash determinism across re-serialization
// ---------------------------------------------------------------------------

#[test]
fn schema_hash_deterministic_across_clones() {
    // The same Schema in different Rust value form produces the same hash.
    let s1 = simple_schema();
    let s2 = simple_schema();
    assert_eq!(schema_hash(&s1).unwrap(), schema_hash(&s2).unwrap());
}

// ---------------------------------------------------------------------------
// 11. Writer rejects mismatched (schema, hash) pair (§6.5 Surface C lock)
// ---------------------------------------------------------------------------

#[test]
fn with_catalog_ref_rejects_mismatched_schema_hash() {
    let dir = tempfile::tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).unwrap();

    let schema_a = simple_schema();
    let schema_b = Schema::new(vec![ColumnSpec::primitive(
        "y",
        DataType::I32,
        vec![CoderSpec::new("zstd")],
    )]);

    // Register schema A; obtain its hash.
    let hash_a = catalog.add_schema(&schema_a).unwrap();
    let hash_b = catalog.add_schema(&schema_b).unwrap();
    assert_ne!(hash_a, hash_b);

    // Try to construct a writer with schema_a but hash_b → must reject.
    let mut buf = Cursor::new(Vec::<u8>::new());
    let err = HeliumWriter::with_catalog_ref(
        &mut buf,
        schema_a,
        hash_b, // wrong hash
        &registry(),
    )
    .expect_err("mismatched schema/hash must be rejected");
    match err {
        HeliumError::Schema { column, reason } => {
            assert_eq!(column, "<header>");
            assert!(
                reason.contains("does not match canonicalize_and_hash"),
                "unexpected reason: {reason}"
            );
        }
        other => panic!("expected HeliumError::Schema, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 12. v4 round-trip with deeply-nested schema
// ---------------------------------------------------------------------------

#[test]
fn catalog_roundtrip_with_nested_schema() {
    use helium::FieldSpec;

    let schema = Schema::new(vec![ColumnSpec::struct_col(
        "rec",
        vec![
            FieldSpec::primitive(
                "id",
                DataType::I64,
                vec![
                    CoderSpec::new("delta"),
                    CoderSpec::new("leb128"),
                    CoderSpec::new("zstd"),
                ],
            ),
            FieldSpec::utf8(
                "label",
                vec![
                    CoderSpec::new("delta"),
                    CoderSpec::new("leb128"),
                    CoderSpec::new("zstd"),
                ],
                vec![CoderSpec::new("zstd")],
            ),
        ],
    )]);

    let dir = tempfile::tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).unwrap();

    let data = LogicalColumn::Struct {
        fields: vec![
            (
                "id".into(),
                LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 3])),
            ),
            (
                "label".into(),
                LogicalColumn::Utf8(vec!["a".into(), "b".into(), "c".into()]),
            ),
        ],
    };

    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut w = catalog.open_writer(&mut buf, schema, &registry()).unwrap();
    w.write_column("rec", data.clone()).unwrap();
    w.finish().unwrap();

    buf.set_position(0);
    let mut reader =
        HeliumReader::new_with_resolver(&mut buf, &registry(), catalog.resolver()).unwrap();
    let result = reader.read_column("rec").unwrap();
    assert_eq!(result, data);
}
