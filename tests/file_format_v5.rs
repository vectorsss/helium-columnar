//! File format v5/v6 — zstd-compressed footer.
//!
//! Acceptance items:
//!
//! 1. Round-trip basic: write via `HeliumWriter::new` → file starts/ends
//!    with `MAGIC_V5` → open with `HeliumReader::new` → data matches.
//! 2. Multi-stripe round-trip: 3 stripes via `finish_stripe()` → all readable.
//! 3. Footer-compression effective: wide schema (50 columns × 3 stripes)
//!    → v5 footer_len (on-disk compressed bytes) < raw JSON length;
//!    require at least 30% smaller.
//! 4. Footer CRC catches tampering: flip a byte inside the footer region
//!    → expect `HeliumError::Corrupted` mentioning "footer CRC32C mismatch".
//! 5. Catalog v6 round-trip: `Catalog::open_writer` → file starts with
//!    `MAGIC_V6` → readable via `HeliumReader::new_with_resolver`.

use std::io::{Cursor, Seek, SeekFrom};

use helium::{
    CoderRegistry, CoderSpec, ColumnData, ColumnSpec, DataType, HeliumError, HeliumReader,
    HeliumWriter, LogicalColumn, MAGIC_V5, MAGIC_V6, Schema, catalog::Catalog,
};

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
fn write_simple_v5() -> Vec<u8> {
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
fn v5_writer_emits_v5_magic_at_both_ends() {
    let bytes = write_simple_v5();
    assert!(bytes.len() >= 16, "file too small");
    assert_eq!(&bytes[..8], MAGIC_V5, "start magic must be v5");
    assert_eq!(&bytes[bytes.len() - 8..], MAGIC_V5, "end magic must be v5");
}

#[test]
fn v5_basic_round_trip() {
    let bytes = write_simple_v5();
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
fn v5_multi_stripe_round_trip() {
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
    assert_eq!(&bytes[..8], MAGIC_V5);
    assert_eq!(&bytes[bytes.len() - 8..], MAGIC_V5);

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
fn v5_footer_compressed_smaller_than_raw_json() {
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

    // Write 3 stripes via v5. Use an owned Cursor so finish() returns the Cursor.
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
    let v5_bytes = w.finish().expect("finish").into_inner();

    // Extract the on-disk footer length from the v5 file.
    // Trailer is 20 bytes: [footer_len(8) | footer_crc(4) | magic(8)].
    let file_len = v5_bytes.len();
    let footer_len_on_disk = u64::from_le_bytes(
        v5_bytes[file_len - 20..file_len - 12]
            .try_into()
            .expect("footer_len slice"),
    ) as usize;

    // Compute what the raw uncompressed footer JSON would be. We can do this
    // by building an equivalent v3 file (raw footer) and measuring its footer
    // JSON length. Alternatively, we measure the compressed bytes vs the
    // decompressed bytes.
    let compressed_footer = &v5_bytes[file_len - 20 - footer_len_on_disk..file_len - 20];
    let raw_footer_json =
        zstd::decode_all(compressed_footer).expect("zstd decompress footer for measurement");
    let raw_len = raw_footer_json.len();
    let compressed_len = footer_len_on_disk;

    let savings_pct = ((raw_len as f64 - compressed_len as f64) / raw_len as f64) * 100.0;
    eprintln!(
        "[v5 footer compression] wide-50col × 3 stripes: \
         raw_footer={raw_len} bytes, compressed={compressed_len} bytes, \
         savings={savings_pct:.1}%"
    );

    // Require at least 30% savings (design requirement).
    assert!(
        savings_pct >= 30.0,
        "v5 footer compression must be at least 30% smaller than raw JSON; \
         raw={raw_len}, compressed={compressed_len}, savings={savings_pct:.1}%"
    );
}

// ---------------------------------------------------------------------------
// 4. Footer CRC catches tampering
// ---------------------------------------------------------------------------

#[test]
fn v5_footer_crc_catches_byte_flip() {
    let mut bytes = write_simple_v5();
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
// 5. Catalog v6 round-trip
// ---------------------------------------------------------------------------

#[test]
fn v6_catalog_round_trip() {
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
    // File must use v6 magic (catalog mode + compressed footer).
    assert_eq!(&bytes[..8], MAGIC_V6, "catalog writer must emit v6");
    assert_eq!(&bytes[bytes.len() - 8..], MAGIC_V6, "end magic must be v6");

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
fn v6_magic_byte_is_six() {
    assert_eq!(MAGIC_V6[7], 0x06);
    assert_ne!(MAGIC_V6, MAGIC_V5);
    assert_eq!(MAGIC_V5[7], 0x05);
}
