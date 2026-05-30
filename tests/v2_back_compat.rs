//! v2 → v3 back-compat consolidation pass (§5.6).
//!
//! Goal: prove that **every remaining v2 `LogicalType` variant remains readable
//! by the v3 reader**, and that v2-shaped schema JSON still parses through
//! `Schema::from_json`. This is a single grep-able boundary:
//!
//! ```text
//! cargo test --test v2_back_compat
//! ```
//!
//! Coverage:
//!
//! - v2 leaf types (`Primitive`, `Utf8`, `Binary`) — full round-trip
//! - v2 deprecated-for-write but kept-readable types (`ArrayOf`,
//!   `ArrayOfUtf8`, `NullablePrim`, `NullableUtf8`, `NullableBinary`)
//! - hand-crafted v2 schema JSON parses into `LogicalType::*` v2 variants
//!   exactly (no silent v2→v3 upgrade)
//! - dual-format isolation: v2-only file and v3-only file can coexist; the
//!   v3 writer never silently rewrites v2 schemas
//! - `HeliumWriter` is a fresh-file-only constructor — there is no API for
//!   opening an existing file in write mode (verified by inspection of
//!   `src/file.rs`: `HeliumWriter::new` is the only constructor, and it
//!   always writes the magic header at offset 0)
//!
//! Note: the legacy v2 dict variants have been fully removed and replaced by
//! the v3 `Dictionary { inner }` type. Tests for dict-encoded columns live
//! below and in `tests/file_format.rs`.
//!
//! Carry-over from §5.5 review:
//!
//! - `union_rejects_256_variants` — exercises the 255-variant cap that
//!   `validate_nested_type` enforces in `src/schema.rs`.

use std::io::Cursor;

use helium::{
    CoderRegistry, CoderSpec, ColumnData, ColumnSpec, DataType, FieldSpec, HeliumReader,
    HeliumWriter, LogicalColumn, LogicalType, Schema,
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
fn zstd_only() -> Vec<CoderSpec> {
    vec![zstd()]
}
/// U8→Bytes conversion before zstd (for `present` bitmaps).
fn present_coders() -> Vec<CoderSpec> {
    vec![CoderSpec::new("leb128"), zstd()]
}
/// U32→Bytes conversion before zstd (for indices).
fn u32_coders() -> Vec<CoderSpec> {
    vec![CoderSpec::new("delta"), CoderSpec::new("leb128"), zstd()]
}
fn registry() -> CoderRegistry {
    CoderRegistry::default()
}

/// Byte-level subsequence search. Used in `dual_format_v2_and_v3_files_coexist`
/// after decompressing the schema header bytes — the on-disk schema bytes
/// are zstd-compressed in v3 (PLAN_V2 §6.4), so a raw byte search would
/// never match the `"kind":"..."` tokens.
fn contains_subseq(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

/// Extract and decompress the schema JSON bytes from a v3/v5 `.he` file.
///
/// File layout: `magic(8) | schema_len(4 LE) | <zstd-compressed schema> | body | ...`.
/// Returns the raw (uncompressed) JSON bytes. Accepts both v3 and v5 magic
/// since both use zstd-compressed schema headers.
fn decompress_schema_from_v3_file(bytes: &[u8]) -> Vec<u8> {
    let magic = &bytes[..8];
    assert!(
        magic == b"HELIUM\x00\x03" || magic == b"HELIUM\x00\x05",
        "expected v3 or v5 magic, got: {magic:02x?}"
    );
    let schema_len =
        u32::from_le_bytes(bytes[8..12].try_into().expect("schema_len slice")) as usize;
    let compressed = &bytes[12..12 + schema_len];
    zstd::decode_all(compressed).expect("zstd decompress schema")
}

/// Single-stripe round-trip helper. Writes the column, reads it back, returns the result.
fn roundtrip(spec: ColumnSpec, data: LogicalColumn) -> LogicalColumn {
    let name = spec.name.clone();
    let schema = Schema::new(vec![spec]);
    let reg = registry();
    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut writer = HeliumWriter::new(&mut buf, schema, &reg).expect("writer");
    writer.write_column(&name, data).expect("write");
    writer.finish().expect("finish");
    buf.set_position(0);
    HeliumReader::new(&mut buf, &reg)
        .expect("reader")
        .read_column(&name)
        .expect("read")
}

// ---------------------------------------------------------------------------
// v2 leaf types — Primitive, Utf8, Binary
// ---------------------------------------------------------------------------

#[test]
fn v2_primitive_i64_roundtrip() {
    let spec = ColumnSpec::primitive("ts", DataType::I64, delta_leb_zstd());
    let values: Vec<i64> = (1_700_000_000..1_700_000_100).collect();
    let data = LogicalColumn::Primitive(ColumnData::I64(values.clone()));
    let result = roundtrip(spec, data);
    assert_eq!(result, LogicalColumn::Primitive(ColumnData::I64(values)));
}

#[test]
fn v2_primitive_f64_roundtrip() {
    let spec = ColumnSpec::primitive(
        "temp",
        DataType::F64,
        vec![CoderSpec::new("gorilla"), zstd()],
    );
    let values: Vec<f64> = (0..50).map(|i| 20.0 + (i as f64) * 0.1).collect();
    let data = LogicalColumn::Primitive(ColumnData::F64(values.clone()));
    let result = roundtrip(spec, data);
    assert_eq!(result, LogicalColumn::Primitive(ColumnData::F64(values)));
}

#[test]
fn v2_utf8_roundtrip() {
    let spec = ColumnSpec::utf8("name", delta_leb_zstd(), zstd_only());
    let values: Vec<String> = (0..100).map(|i| format!("user_{i}")).collect();
    let data = LogicalColumn::Utf8(values.clone());
    let result = roundtrip(spec, data);
    assert_eq!(result, LogicalColumn::Utf8(values));
}

#[test]
fn v2_binary_roundtrip() {
    let spec = ColumnSpec::binary("payload", delta_leb_zstd(), zstd_only());
    let blobs: Vec<Vec<u8>> = (0..20).map(|i| vec![i as u8; (i + 1) as usize]).collect();
    let data = LogicalColumn::Binary(blobs.clone());
    let result = roundtrip(spec, data);
    assert_eq!(result, LogicalColumn::Binary(blobs));
}

// ---------------------------------------------------------------------------
// v2 deprecated-for-write types — ArrayOf, ArrayOfUtf8, NullablePrim,
// NullableUtf8, NullableBinary — all kept readable
// ---------------------------------------------------------------------------

#[test]
fn v2_array_of_roundtrip() {
    let spec = ColumnSpec::array_of("tags", DataType::I32, delta_leb_zstd(), delta_leb_zstd());
    let offsets: Vec<u32> = vec![0, 3, 3, 5, 7];
    let values = ColumnData::I32(vec![1, 2, 3, 4, 5, 6, 7]);
    let data = LogicalColumn::ArrayOf {
        offsets: offsets.clone(),
        values: values.clone(),
    };
    let result = roundtrip(spec, data);
    let LogicalColumn::ArrayOf {
        offsets: ro,
        values: rv,
    } = result
    else {
        panic!("expected ArrayOf, got {result:?}");
    };
    assert_eq!(ro, offsets);
    assert_eq!(rv, values);
}

#[test]
fn v2_array_of_utf8_roundtrip() {
    let spec = ColumnSpec::array_of_utf8("words", delta_leb_zstd(), delta_leb_zstd(), zstd_only());
    let offsets: Vec<u32> = vec![0, 2, 3, 5];
    let strings: Vec<String> = vec![
        "hello".into(),
        "world".into(),
        "foo".into(),
        "bar".into(),
        "baz".into(),
    ];
    let data = LogicalColumn::ArrayOfUtf8 {
        offsets: offsets.clone(),
        strings: strings.clone(),
    };
    let result = roundtrip(spec, data);
    let LogicalColumn::ArrayOfUtf8 {
        offsets: ro,
        strings: rs,
    } = result
    else {
        panic!("expected ArrayOfUtf8");
    };
    assert_eq!(ro, offsets);
    assert_eq!(rs, strings);
}

#[test]
fn v2_nullable_prim_roundtrip() {
    let spec = ColumnSpec::nullable_prim("v", DataType::I32, present_coders(), delta_leb_zstd());
    let present = vec![true, false, true, false, true, true];
    let values = ColumnData::I32(vec![10, 30, 50, 60]);
    let data = LogicalColumn::NullablePrim {
        present: present.clone(),
        values: values.clone(),
    };
    let result = roundtrip(spec, data);
    let LogicalColumn::NullablePrim {
        present: rp,
        values: rv,
    } = result
    else {
        panic!("expected NullablePrim");
    };
    assert_eq!(rp, present);
    assert_eq!(rv, values);
}

#[test]
fn v2_nullable_utf8_roundtrip() {
    let spec = ColumnSpec::nullable_utf8("s", present_coders(), delta_leb_zstd(), zstd_only());
    let present = vec![false, true, true, false, true];
    let strings = vec!["hello".to_string(), "world".to_string(), "rust".to_string()];
    let data = LogicalColumn::NullableUtf8 {
        present: present.clone(),
        strings: strings.clone(),
    };
    let result = roundtrip(spec, data);
    let LogicalColumn::NullableUtf8 {
        present: rp,
        strings: rs,
    } = result
    else {
        panic!("expected NullableUtf8");
    };
    assert_eq!(rp, present);
    assert_eq!(rs, strings);
}

#[test]
fn v2_nullable_binary_roundtrip() {
    let spec = ColumnSpec::nullable_binary("b", present_coders(), delta_leb_zstd(), zstd_only());
    let present = vec![true, false, true];
    let blobs: Vec<Vec<u8>> = vec![vec![0xff, 0xfe], vec![0x00, 0x01, 0x02]];
    let data = LogicalColumn::NullableBinary {
        present: present.clone(),
        blobs: blobs.clone(),
    };
    let result = roundtrip(spec, data);
    let LogicalColumn::NullableBinary {
        present: rp,
        blobs: rb,
    } = result
    else {
        panic!("expected NullableBinary");
    };
    assert_eq!(rp, present);
    assert_eq!(rb, blobs);
}

// ---------------------------------------------------------------------------
// Dictionary (v3) round-trip — proves dict_encode_utf8 / dict_encode_primitive
// return the correct Dictionary{inner} shape and survive the full file round-trip.
// ---------------------------------------------------------------------------

#[test]
fn dict_utf8_roundtrip() {
    // Dictionary { inner: Utf8 } — single stripe
    let raw: Vec<String> = vec![
        "INFO".into(),
        "WARN".into(),
        "INFO".into(),
        "ERROR".into(),
        "INFO".into(),
        "WARN".into(),
        "INFO".into(),
    ];
    let encoded = LogicalColumn::dict_encode_utf8(raw.clone());
    let spec = ColumnSpec::new(
        "level",
        LogicalType::Dictionary {
            inner: Box::new(LogicalType::Utf8),
        },
        vec![delta_leb_zstd(), zstd_only(), u32_coders()],
    );
    let result = roundtrip(spec, encoded);
    let materialized = result.materialize_dict_utf8().expect("materialize");
    assert_eq!(materialized, raw);
}

#[test]
fn dict_prim_roundtrip() {
    // Dictionary { inner: Primitive(I32) } — single stripe
    let raw_values = ColumnData::I32(vec![100, 200, 100, 300, 200, 100, 100]);
    let encoded = LogicalColumn::dict_encode_primitive(raw_values.clone()).expect("dict_encode");
    let LogicalColumn::Dictionary {
        ref dictionary,
        ref indices,
    } = encoded
    else {
        panic!("expected Dictionary from dict_encode_primitive");
    };
    let dict_row_count = dictionary.row_count();
    let idx_clone = indices.clone();

    let spec = ColumnSpec::new(
        "status",
        LogicalType::Dictionary {
            inner: Box::new(LogicalType::Primitive {
                data_type: DataType::I32,
            }),
        },
        vec![delta_leb_zstd(), u32_coders()],
    );
    let result = roundtrip(spec, encoded);
    let LogicalColumn::Dictionary {
        dictionary: rd,
        indices: ri,
    } = result
    else {
        panic!("expected Dictionary from reader");
    };
    assert_eq!(rd.row_count(), dict_row_count);
    assert_eq!(ri, idx_clone);
}

#[test]
fn dict_multi_stripe_via_read_column_at_stripe() {
    // Multi-stripe dict columns require `read_column_at_stripe` because each
    // stripe may have its own dictionary. The v3 reader preserves this constraint.
    let spec = ColumnSpec::new(
        "level",
        LogicalType::Dictionary {
            inner: Box::new(LogicalType::Utf8),
        },
        vec![delta_leb_zstd(), zstd_only(), u32_coders()],
    );
    let schema = Schema::new(vec![spec]);
    let reg = registry();

    let s1_raw: Vec<String> = vec!["INFO".into(), "WARN".into(), "INFO".into()];
    let s2_raw: Vec<String> = vec!["ERROR".into(), "ERROR".into(), "DEBUG".into()];
    let s1 = LogicalColumn::dict_encode_utf8(s1_raw.clone());
    let s2 = LogicalColumn::dict_encode_utf8(s2_raw.clone());

    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut writer = HeliumWriter::new(&mut buf, schema, &reg).expect("writer");
    writer.write_column("level", s1).expect("s1");
    writer.finish_stripe().expect("finish_stripe");
    writer.write_column("level", s2).expect("s2");
    writer.finish().expect("finish");

    buf.set_position(0);
    let mut reader = HeliumReader::new(&mut buf, &reg).expect("reader");
    assert_eq!(reader.stripe_count(), 2);

    // read_column should error for dict columns in multi-stripe
    let concat_err = reader.read_column("level");
    assert!(
        concat_err.is_err(),
        "read_column should fail for dict in multi-stripe"
    );

    // read_column_at_stripe is the correct API
    let r1 = reader.read_column_at_stripe("level", 0).expect("stripe 0");
    let r2 = reader.read_column_at_stripe("level", 1).expect("stripe 1");
    assert_eq!(r1.materialize_dict_utf8().unwrap(), s1_raw);
    assert_eq!(r2.materialize_dict_utf8().unwrap(), s2_raw);
}

// ---------------------------------------------------------------------------
// v2 schema JSON parsing — hand-crafted shapes deserialize without
// silent v2→v3 upgrade
// ---------------------------------------------------------------------------

#[test]
fn v2_array_of_schema_json_deserializes_as_array_of() {
    let json = r#"{"version":1,"columns":[{"name":"x","logical_type":{"kind":"array_of","data_type":"i32"},"encodings":[[{"id":"zstd"}],[{"id":"zstd"}]]}]}"#;
    let schema = Schema::from_json(json.as_bytes()).expect("v2 ArrayOf JSON should parse");
    assert!(matches!(
        schema.columns[0].logical_type,
        LogicalType::ArrayOf {
            data_type: DataType::I32
        }
    ));
}

#[test]
fn v2_array_of_utf8_schema_json_deserializes_unchanged() {
    let json = r#"{"version":1,"columns":[{"name":"x","logical_type":{"kind":"array_of_utf8"},"encodings":[[{"id":"zstd"}],[{"id":"zstd"}],[{"id":"zstd"}]]}]}"#;
    let schema = Schema::from_json(json.as_bytes()).expect("v2 ArrayOfUtf8 JSON should parse");
    assert!(matches!(
        schema.columns[0].logical_type,
        LogicalType::ArrayOfUtf8
    ));
}

#[test]
fn v2_nullable_prim_schema_json_deserializes_unchanged() {
    let json = r#"{"version":1,"columns":[{"name":"x","logical_type":{"kind":"nullable_prim","data_type":"i32"},"encodings":[[{"id":"leb128"},{"id":"zstd"}],[{"id":"zstd"}]]}]}"#;
    let schema = Schema::from_json(json.as_bytes()).expect("v2 NullablePrim JSON should parse");
    assert!(matches!(
        schema.columns[0].logical_type,
        LogicalType::NullablePrim {
            data_type: DataType::I32
        }
    ));
}

#[test]
fn v2_nullable_utf8_schema_json_deserializes_unchanged() {
    let json = r#"{"version":1,"columns":[{"name":"x","logical_type":{"kind":"nullable_utf8"},"encodings":[[{"id":"leb128"},{"id":"zstd"}],[{"id":"zstd"}],[{"id":"zstd"}]]}]}"#;
    let schema = Schema::from_json(json.as_bytes()).expect("v2 NullableUtf8 JSON should parse");
    assert!(matches!(
        schema.columns[0].logical_type,
        LogicalType::NullableUtf8
    ));
}

#[test]
fn v2_nullable_binary_schema_json_deserializes_unchanged() {
    let json = r#"{"version":1,"columns":[{"name":"x","logical_type":{"kind":"nullable_binary"},"encodings":[[{"id":"leb128"},{"id":"zstd"}],[{"id":"zstd"}],[{"id":"zstd"}]]}]}"#;
    let schema = Schema::from_json(json.as_bytes()).expect("v2 NullableBinary JSON should parse");
    assert!(matches!(
        schema.columns[0].logical_type,
        LogicalType::NullableBinary
    ));
}

// ---------------------------------------------------------------------------
// Combined v2 schema — production-shape mix of dict + nullable + array
// + utf8 + primitive
// ---------------------------------------------------------------------------

#[test]
fn v2_combined_production_shape_roundtrip() {
    // Mimics a realistic pre-v3 schema that combined every v2 nullable/array/dict shape
    // in one file. Verifies all variants survive together through the v3 reader.
    let schema = Schema::new(vec![
        // Timestamps: monotonic, delta-encoded.
        ColumnSpec::primitive("ts", DataType::I64, delta_leb_zstd()),
        // Dict-encoded log level (Dictionary { inner: Utf8 }).
        ColumnSpec::new(
            "level",
            LogicalType::Dictionary {
                inner: Box::new(LogicalType::Utf8),
            },
            vec![delta_leb_zstd(), zstd_only(), u32_coders()],
        ),
        // Nullable user message.
        ColumnSpec::nullable_utf8("message", present_coders(), delta_leb_zstd(), zstd_only()),
        // Array of i32 tags.
        ColumnSpec::array_of("tags", DataType::I32, delta_leb_zstd(), delta_leb_zstd()),
        // Nullable measurement.
        ColumnSpec::nullable_prim(
            "weight",
            DataType::F64,
            present_coders(),
            vec![CoderSpec::new("gorilla"), zstd()],
        ),
    ]);
    let reg = registry();

    let n = 200usize;
    let ts: Vec<i64> = (0..n).map(|i| 1_700_000_000 + i as i64 * 30).collect();
    let levels: Vec<String> = (0..n)
        .map(|i| {
            match i % 4 {
                0 => "INFO",
                1 => "WARN",
                2 => "ERROR",
                _ => "DEBUG",
            }
            .into()
        })
        .collect();
    let level_col = LogicalColumn::dict_encode_utf8(levels.clone());

    let msg_present: Vec<bool> = (0..n).map(|i| i % 3 != 0).collect();
    let msg_strings: Vec<String> = msg_present
        .iter()
        .enumerate()
        .filter(|&(_, &p)| p)
        .map(|(i, _)| format!("event {i}"))
        .collect();
    let msg_col = LogicalColumn::NullableUtf8 {
        present: msg_present.clone(),
        strings: msg_strings.clone(),
    };

    // 200 rows of variable-length tag arrays (0..3 tags per row).
    let mut tag_offsets: Vec<u32> = vec![0];
    let mut tag_values: Vec<i32> = Vec::new();
    for i in 0..n {
        let count = (i % 4) as i32;
        for j in 0..count {
            tag_values.push(j + (i as i32) * 10);
        }
        tag_offsets.push(tag_values.len() as u32);
    }
    let tags_col = LogicalColumn::ArrayOf {
        offsets: tag_offsets.clone(),
        values: ColumnData::I32(tag_values.clone()),
    };

    let weight_present: Vec<bool> = (0..n).map(|i| i % 5 != 0).collect();
    let weight_values: Vec<f64> = weight_present
        .iter()
        .enumerate()
        .filter(|&(_, &p)| p)
        .map(|(i, _)| 50.0 + (i as f64) * 0.7)
        .collect();
    let weight_col = LogicalColumn::NullablePrim {
        present: weight_present.clone(),
        values: ColumnData::F64(weight_values.clone()),
    };

    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut writer = HeliumWriter::new(&mut buf, schema, &reg).expect("writer");
    writer
        .write_column("ts", LogicalColumn::Primitive(ColumnData::I64(ts.clone())))
        .expect("ts");
    writer.write_column("level", level_col).expect("level");
    writer.write_column("message", msg_col).expect("message");
    writer.write_column("tags", tags_col).expect("tags");
    writer.write_column("weight", weight_col).expect("weight");
    writer.finish().expect("finish");

    buf.set_position(0);
    let mut reader = HeliumReader::new(&mut buf, &reg).expect("reader");

    // ts
    assert_eq!(
        reader.read_column("ts").unwrap(),
        LogicalColumn::Primitive(ColumnData::I64(ts))
    );
    // level — single-stripe so read_column works for dict
    let level_back = reader.read_column("level").unwrap();
    assert_eq!(level_back.materialize_dict_utf8().unwrap(), levels);
    // message
    let LogicalColumn::NullableUtf8 {
        present: mp,
        strings: ms,
    } = reader.read_column("message").unwrap()
    else {
        panic!("message");
    };
    assert_eq!(mp, msg_present);
    assert_eq!(ms, msg_strings);
    // tags
    let LogicalColumn::ArrayOf {
        offsets: to,
        values: tv,
    } = reader.read_column("tags").unwrap()
    else {
        panic!("tags");
    };
    assert_eq!(to, tag_offsets);
    assert_eq!(tv, ColumnData::I32(tag_values));
    // weight
    let LogicalColumn::NullablePrim {
        present: wp,
        values: wv,
    } = reader.read_column("weight").unwrap()
    else {
        panic!("weight");
    };
    assert_eq!(wp, weight_present);
    assert_eq!(wv, ColumnData::F64(weight_values));
}

// ---------------------------------------------------------------------------
// Dual-format isolation: a v2-only file and a v3-only file written
// independently by the v3 writer must each round-trip cleanly.
// ---------------------------------------------------------------------------

#[test]
fn dual_format_v2_and_v3_files_coexist() {
    let reg = registry();

    // ---- File A: v2-only schema ----
    let v2_schema = Schema::new(vec![ColumnSpec::nullable_prim(
        "x",
        DataType::I32,
        present_coders(),
        delta_leb_zstd(),
    )]);
    let mut buf_a = Cursor::new(Vec::<u8>::new());
    let mut wa = HeliumWriter::new(&mut buf_a, v2_schema, &reg).expect("v2 writer");
    wa.write_column(
        "x",
        LogicalColumn::NullablePrim {
            present: vec![true, false, true],
            values: ColumnData::I32(vec![1, 3]),
        },
    )
    .expect("v2 write");
    wa.finish().expect("v2 finish");

    // Verify the file's embedded SCHEMA JSON uses the v2 kind tag, NOT v3.
    // Schema bytes are zstd-compressed in v3 file format — decompress first.
    let v2_file_bytes = buf_a.get_ref();
    let v2_schema_json = decompress_schema_from_v3_file(v2_file_bytes);
    assert!(
        contains_subseq(&v2_schema_json, b"\"kind\":\"nullable_prim\""),
        "v2-vocabulary schema should keep v2 kind tag; got: {}",
        String::from_utf8_lossy(&v2_schema_json)
    );
    assert!(
        !contains_subseq(&v2_schema_json, b"\"kind\":\"nullable\""),
        "v2-vocabulary schema should NOT auto-upgrade to v3 vocabulary"
    );

    // ---- File B: v3-only schema (Nullable wrapper) ----
    let v3_schema = Schema::new(vec![ColumnSpec::nullable(
        "x",
        LogicalType::Primitive {
            data_type: DataType::I32,
        },
        vec![present_coders(), delta_leb_zstd()],
    )]);
    let mut buf_b = Cursor::new(Vec::<u8>::new());
    let mut wb = HeliumWriter::new(&mut buf_b, v3_schema, &reg).expect("v3 writer");
    wb.write_column(
        "x",
        LogicalColumn::Nullable {
            present: vec![true, false, true],
            value: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![1, 3]))),
        },
    )
    .expect("v3 write");
    wb.finish().expect("v3 finish");

    let v3_file_bytes = buf_b.get_ref();
    let v3_schema_json = decompress_schema_from_v3_file(v3_file_bytes);
    assert!(
        contains_subseq(&v3_schema_json, b"\"kind\":\"nullable\""),
        "v3-vocabulary schema should use new kind tag; got: {}",
        String::from_utf8_lossy(&v3_schema_json)
    );
    assert!(
        !contains_subseq(&v3_schema_json, b"\"kind\":\"nullable_prim\""),
        "v3-vocabulary schema should NOT use v2 tag"
    );

    // ---- Read both back; assertions match the variant the schema requested ----
    buf_a.set_position(0);
    let read_a = HeliumReader::new(&mut buf_a, &reg)
        .expect("ra")
        .read_column("x")
        .unwrap();
    assert!(
        matches!(read_a, LogicalColumn::NullablePrim { .. }),
        "v2 file → v2 variant"
    );

    buf_b.set_position(0);
    let read_b = HeliumReader::new(&mut buf_b, &reg)
        .expect("rb")
        .read_column("x")
        .unwrap();
    assert!(
        matches!(read_b, LogicalColumn::Nullable { .. }),
        "v3 file → v3 variant"
    );
}

// ---------------------------------------------------------------------------
// Carry-over from §5.5 review: 256-variant Union rejection
// ---------------------------------------------------------------------------

#[test]
fn union_rejects_256_variants() {
    // 256 variants exceeds the U8 tag's capacity (max 255). Validation must reject.
    let variants: Vec<(String, LogicalType)> = (0..256)
        .map(|i| {
            (
                format!("v{i}"),
                LogicalType::Primitive {
                    data_type: DataType::I32,
                },
            )
        })
        .collect();
    // Encodings: 1 (tag) + 256 (one per primitive variant) = 257.
    let encodings: Vec<Vec<CoderSpec>> = (0..257).map(|_| vec![zstd()]).collect();
    let spec = ColumnSpec::union("u", variants, encodings);
    let err = Schema::new(vec![spec])
        .validate()
        .expect_err("256 variants must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("255") || msg.contains("256") || msg.contains("U8"),
        "unexpected error: {msg}"
    );
}

#[test]
fn union_accepts_255_variants() {
    // The boundary: 255 variants is the max allowed.
    // (We don't write/read; just validate the schema — 255 variants is large
    // and fully encoding/decoding adds nothing on top of validate semantics.)
    let variants: Vec<(String, LogicalType)> = (0..255)
        .map(|i| {
            (
                format!("v{i}"),
                LogicalType::Primitive {
                    data_type: DataType::I32,
                },
            )
        })
        .collect();
    let encodings: Vec<Vec<CoderSpec>> = (0..256).map(|_| vec![zstd()]).collect();
    let spec = ColumnSpec::union("u", variants, encodings);
    Schema::new(vec![spec])
        .validate()
        .expect("255 variants should pass");
}

// ---------------------------------------------------------------------------
// Sanity: writing a file is fresh-file-only — there is no API to open an
// existing file and append/modify schema in place. This is a behavioral
// assertion enforced by the absence of an alternate constructor; we
// document it here for the back-compat audit trail.
// ---------------------------------------------------------------------------

#[test]
fn writer_starts_with_magic_at_offset_zero() {
    // Tautological-looking, but it locks down the "no read-modify-write" path:
    // every HeliumWriter::new call writes the magic bytes at offset 0.
    // (Current writer emits v5 magic; see PLAN_V2 §6.4 + footer-compression update.)
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "x",
        DataType::I32,
        delta_leb_zstd(),
    )]);
    let reg = registry();
    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut w = HeliumWriter::new(&mut buf, schema, &reg).expect("writer");
    w.write_column(
        "x",
        LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3])),
    )
    .expect("write");
    w.finish().expect("finish");

    // First 8 bytes must be the current writer's magic (v5 = v3 + compressed footer).
    let bytes = buf.get_ref();
    assert!(bytes.len() >= 8);
    assert_eq!(&bytes[..8], b"HELIUM\x00\x05");
}

// ---------------------------------------------------------------------------
// Spot-check: a schema with both v2 and v3 variants in different columns
// validates, writes, and reads. Confirms the reader's dispatch table
// handles a heterogeneous schema (real-world incremental migration shape).
// ---------------------------------------------------------------------------

#[test]
fn mixed_v2_and_v3_columns_in_one_schema() {
    let schema = Schema::new(vec![
        // v2: NullablePrim
        ColumnSpec::nullable_prim(
            "old_score",
            DataType::I32,
            present_coders(),
            delta_leb_zstd(),
        ),
        // v3: Nullable(Primitive)
        ColumnSpec::nullable(
            "new_score",
            LogicalType::Primitive {
                data_type: DataType::I32,
            },
            vec![present_coders(), delta_leb_zstd()],
        ),
        // v3: Struct
        ColumnSpec::struct_col(
            "rec",
            vec![FieldSpec::primitive("id", DataType::I64, delta_leb_zstd())],
        ),
    ]);
    let reg = registry();

    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut w = HeliumWriter::new(&mut buf, schema, &reg).expect("writer");
    w.write_column(
        "old_score",
        LogicalColumn::NullablePrim {
            present: vec![true, false, true],
            values: ColumnData::I32(vec![100, 200]),
        },
    )
    .expect("old_score");
    w.write_column(
        "new_score",
        LogicalColumn::Nullable {
            present: vec![true, false, true],
            value: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![10, 30]))),
        },
    )
    .expect("new_score");
    w.write_column(
        "rec",
        LogicalColumn::Struct {
            fields: vec![(
                "id".to_string(),
                LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 3])),
            )],
        },
    )
    .expect("rec");
    w.finish().expect("finish");

    buf.set_position(0);
    let mut r = HeliumReader::new(&mut buf, &reg).expect("reader");
    assert!(matches!(
        r.read_column("old_score").unwrap(),
        LogicalColumn::NullablePrim { .. }
    ));
    assert!(matches!(
        r.read_column("new_score").unwrap(),
        LogicalColumn::Nullable { .. }
    ));
    assert!(matches!(
        r.read_column("rec").unwrap(),
        LogicalColumn::Struct { .. }
    ));
}
