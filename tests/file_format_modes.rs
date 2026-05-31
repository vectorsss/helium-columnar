//! File format — the two storage modes (self-contained vs catalog) and the
//! zstd-compressed footer.
//!
//! The 8-byte header is `b"HELIUM"` + version (1) + flags. Self-contained files
//! have flags `0x00`; catalog (external-schema) files set bit 0 → flags `0x01`.
//!
//! Acceptance items:
//!
//! 1. Round-trip basic: write via `HeliumWriter::new` → self-contained header
//!    at both ends → open with `HeliumReader::new` → data matches.
//! 2. Multi-stripe round-trip: 3 stripes via `finish_stripe()` → all readable.
//! 3. Footer-compression effective: wide schema (50 columns × 3 stripes)
//!    → footer_len (on-disk compressed bytes) < raw JSON length;
//!    require at least 30% smaller.
//! 4. Footer CRC catches tampering: flip a byte inside the footer region
//!    → expect `HeliumError::Corrupted` mentioning "footer CRC32C mismatch".
//! 5. Catalog round-trip: `Catalog::open_writer` → external-schema header
//!    → readable via `HeliumReader::new_with_resolver`.

use std::io::{Cursor, Seek, SeekFrom};

use helium::{
    CoderRegistry, CoderSpec, ColumnData, ColumnSpec, DataType, HeliumError, HeliumReader,
    HeliumWriter, LogicalColumn, Schema, catalog::Catalog,
};

/// 8-byte header of a self-contained file: `HELIUM` + version 1 + flags 0.
const HEADER_SELF_CONTAINED: &[u8; 8] = b"HELIUM\x01\x00";
/// 8-byte header of a catalog (external-schema) file: flags bit 0 set.
const HEADER_CATALOG: &[u8; 8] = b"HELIUM\x01\x01";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn zstd() -> CoderSpec {
    CoderSpec::new("zstd")
}
fn delta_leb_zstd() -> Vec<CoderSpec> {
    vec![CoderSpec::new("delta"), CoderSpec::new("leb128"), zstd()]
}
fn registry() -> CoderRegistry {
    CoderRegistry::default()
}

/// Write a simple single-column file and return the on-disk bytes.
fn write_simple_self_contained() -> Vec<u8> {
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "ts",
        DataType::I64,
        delta_leb_zstd(),
    )]);
    let reg = registry();
    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut w = HeliumWriter::new(&mut buf, schema, &reg).expect("writer");
    let values: Vec<i64> = (0..1_000).map(|i| 1_700_000_000_i64 + i * 30).collect();
    w.write_column("ts", LogicalColumn::Primitive(ColumnData::I64(values)))
        .expect("write");
    w.finish().expect("finish");
    buf.into_inner()
}

// ---------------------------------------------------------------------------
// 1. Round-trip basic
// ---------------------------------------------------------------------------

#[test]
fn self_contained_header_at_both_ends() {
    let bytes = write_simple_self_contained();
    assert!(bytes.len() >= 16, "file too small");
    assert_eq!(&bytes[..8], HEADER_SELF_CONTAINED, "start header must be self-contained");
    assert_eq!(&bytes[bytes.len() - 8..], HEADER_SELF_CONTAINED, "end header must be self-contained");
}

#[test]
fn self_contained_basic_round_trip() {
    let bytes = write_simple_self_contained();
    let reg = registry();
    let mut reader = HeliumReader::new(Cursor::new(bytes), &reg).expect("reader");
    let result = reader.read_column("ts").expect("read");
    let LogicalColumn::Primitive(ColumnData::I64(values)) = result else {
        panic!("expected I64 primitive");
    };
    let expected: Vec<i64> = (0..1_000).map(|i| 1_700_000_000_i64 + i * 30).collect();
    assert_eq!(values, expected);
}

// ---------------------------------------------------------------------------
// 2. Multi-stripe round-trip
// ---------------------------------------------------------------------------

#[test]
fn self_contained_multi_stripe_round_trip() {
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "x",
        DataType::I64,
        delta_leb_zstd(),
    )]);
    let reg = registry();
    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut w = HeliumWriter::new(&mut buf, schema, &reg).expect("writer");

    // Stripe 1
    w.write_column(
        "x",
        LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 3])),
    )
    .expect("s1");
    w.finish_stripe().expect("stripe 1");
    // Stripe 2
    w.write_column(
        "x",
        LogicalColumn::Primitive(ColumnData::I64(vec![4, 5, 6, 7])),
    )
    .expect("s2");
    w.finish_stripe().expect("stripe 2");
    // Stripe 3
    w.write_column("x", LogicalColumn::Primitive(ColumnData::I64(vec![8, 9])))
        .expect("s3");
    w.finish().expect("finish");

    let bytes = buf.into_inner();
    assert_eq!(&bytes[..8], HEADER_SELF_CONTAINED);
    assert_eq!(&bytes[bytes.len() - 8..], HEADER_SELF_CONTAINED);

    let mut r = HeliumReader::new(Cursor::new(bytes), &reg).expect("reader");
    assert_eq!(r.stripe_count(), 3);
    let all = r.read_column("x").expect("read");
    assert_eq!(
        all,
        LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]))
    );
}

// ---------------------------------------------------------------------------
// 3. Footer-compression effective: wide schema, ≥30% smaller on disk
// ---------------------------------------------------------------------------

#[test]
fn footer_compressed_smaller_than_raw_json() {
    // Build a wide schema: 50 primitive columns (10 each of I32, I64, F32, F64, U32).
    let mut cols = Vec::new();
    for idx in 0..10usize {
        cols.push(ColumnSpec::primitive(
            format!("col_i32_{idx:02}"),
            DataType::I32,
            delta_leb_zstd(),
        ));
    }
    for idx in 0..10usize {
        cols.push(ColumnSpec::primitive(
            format!("col_i64_{idx:02}"),
            DataType::I64,
            delta_leb_zstd(),
        ));
    }
    for idx in 0..10usize {
        cols.push(ColumnSpec::primitive(
            format!("col_u32_{idx:02}"),
            DataType::U32,
            delta_leb_zstd(),
        ));
    }
    for idx in 0..10usize {
        cols.push(ColumnSpec::primitive(
            format!("col_f32_{idx:02}"),
            DataType::F32,
            vec![CoderSpec::new("gorilla"), zstd()],
        ));
    }
    for idx in 0..10usize {
        cols.push(ColumnSpec::primitive(
            format!("col_f64_{idx:02}"),
            DataType::F64,
            vec![CoderSpec::new("gorilla"), zstd()],
        ));
    }
    assert_eq!(cols.len(), 50);
    let schema = Schema::new(cols);
    let reg = registry();

    // Write 3 stripes. Use an owned Cursor so finish() returns the Cursor.
    let inner_buf = Cursor::new(Vec::<u8>::new());
    let mut w = HeliumWriter::new(inner_buf, schema.clone(), &reg).expect("writer");
    for stripe in 0..3 {
        for col in schema.columns.iter() {
            let n = 100_usize;
            let base = stripe as i64 * 100;
            let data = match col.logical_type {
                helium::LogicalType::Primitive {
                    data_type: DataType::I32,
                } => LogicalColumn::Primitive(ColumnData::I32(
                    (0..n).map(|i| (base + i as i64) as i32).collect(),
                )),
                helium::LogicalType::Primitive {
                    data_type: DataType::I64,
                } => LogicalColumn::Primitive(ColumnData::I64(
                    (0..n).map(|i| base + i as i64).collect(),
                )),
                helium::LogicalType::Primitive {
                    data_type: DataType::U32,
                } => LogicalColumn::Primitive(ColumnData::U32(
                    (0..n).map(|i| (base + i as i64) as u32).collect(),
                )),
                helium::LogicalType::Primitive {
                    data_type: DataType::F32,
                } => LogicalColumn::Primitive(ColumnData::F32(
                    (0..n).map(|i| (base as f32) + i as f32 * 0.1).collect(),
                )),
                helium::LogicalType::Primitive {
                    data_type: DataType::F64,
                } => LogicalColumn::Primitive(ColumnData::F64(
                    (0..n).map(|i| (base as f64) + i as f64 * 0.1).collect(),
                )),
                _ => unreachable!("only primitives in this test"),
            };
            w.write_column(&col.name, data).expect("write");
        }
        if stripe < 2 {
            w.finish_stripe().expect("stripe");
        }
    }
    let file_bytes = w.finish().expect("finish").into_inner();

    // Extract the on-disk footer length from the file.
    // Trailer is 20 bytes: [footer_len(8) | footer_crc(4) | magic(8)].
    let file_len = file_bytes.len();
    let footer_len_on_disk = u64::from_le_bytes(
        file_bytes[file_len - 20..file_len - 12]
            .try_into()
            .expect("footer_len slice"),
    ) as usize;

    // Compute what the raw uncompressed footer JSON would be. We can do this
    // by building an equivalent self-contained file (raw footer) and measuring its footer
    // JSON length. Alternatively, we measure the compressed bytes vs the
    // decompressed bytes.
    let compressed_footer = &file_bytes[file_len - 20 - footer_len_on_disk..file_len - 20];
    let raw_footer_json =
        zstd::decode_all(compressed_footer).expect("zstd decompress footer for measurement");
    let raw_len = raw_footer_json.len();
    let compressed_len = footer_len_on_disk;

    let savings_pct = ((raw_len as f64 - compressed_len as f64) / raw_len as f64) * 100.0;
    eprintln!(
        "[footer compression] wide-50col × 3 stripes: \
         raw_footer={raw_len} bytes, compressed={compressed_len} bytes, \
         savings={savings_pct:.1}%"
    );

    // Require at least 30% savings (design requirement).
    assert!(
        savings_pct >= 30.0,
        "footer compression must be at least 30% smaller than raw JSON; \
         raw={raw_len}, compressed={compressed_len}, savings={savings_pct:.1}%"
    );
}

// ---------------------------------------------------------------------------
// 4. Footer CRC catches tampering
// ---------------------------------------------------------------------------

#[test]
fn footer_crc_catches_byte_flip() {
    let mut bytes = write_simple_self_contained();
    let file_len = bytes.len();

    // The trailer is 20 bytes from the end: [footer_len(8) | footer_crc(4) | magic(8)].
    // The footer body sits immediately before the trailer.
    let footer_len = u64::from_le_bytes(
        bytes[file_len - 20..file_len - 12]
            .try_into()
            .expect("footer_len slice"),
    ) as usize;

    // Flip one byte inside the compressed footer bytes.
    let footer_start = file_len - 20 - footer_len;
    bytes[footer_start] ^= 0xAB;

    let reg = registry();
    let err = HeliumReader::new(Cursor::new(bytes), &reg).expect_err("tampered footer must error");
    match err {
        HeliumError::Corrupted { reason, .. } => {
            assert!(
                reason.contains("footer CRC32C mismatch"),
                "expected 'footer CRC32C mismatch' in reason; got: {reason}"
            );
        }
        other => panic!("expected HeliumError::Corrupted, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 5. Catalog (external-schema) round-trip
// ---------------------------------------------------------------------------

#[test]
fn catalog_round_trip() {
    let dir = tempfile::tempdir().expect("tempdir");
    let catalog = Catalog::open(dir.path()).expect("catalog open");
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "x",
        DataType::I64,
        delta_leb_zstd(),
    )]);

    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut w = catalog
        .open_writer(&mut buf, schema.clone(), &registry())
        .expect("open_writer");
    w.write_column(
        "x",
        LogicalColumn::Primitive(ColumnData::I64(vec![10, 20, 30])),
    )
    .expect("write");
    w.finish().expect("finish");

    let bytes = buf.get_ref();
    // File must use the external-schema (catalog) header.
    assert_eq!(&bytes[..8], HEADER_CATALOG, "catalog writer must set the external-schema flag");
    assert_eq!(&bytes[bytes.len() - 8..], HEADER_CATALOG, "end header must match");

    buf.seek(SeekFrom::Start(0)).expect("rewind");
    let mut reader =
        HeliumReader::new_with_resolver(&mut buf, &registry(), catalog.resolver()).expect("reader");
    let result = reader.read_column("x").expect("read");
    assert_eq!(
        result,
        LogicalColumn::Primitive(ColumnData::I64(vec![10, 20, 30]))
    );
}

#[test]
fn header_flags_distinguish_modes() {
    // Both modes share the magic + version; only the flags byte (index 7)
    // differs: bit 0 set = external schema (catalog mode).
    assert_eq!(&HEADER_SELF_CONTAINED[..7], &HEADER_CATALOG[..7]);
    assert_eq!(HEADER_SELF_CONTAINED[7], 0x00);
    assert_eq!(HEADER_CATALOG[7], 0x01);
}
