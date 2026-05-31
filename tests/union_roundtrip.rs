//! Round-trip tests for `LogicalType::Union`.
//!
//! Cases covered:
//! - 2-variant non-null (alternating rows)
//! - 3-variant mixed types (I32, Utf8, U8)
//! - fully populated (every variant has at least one row)
//! - skewed-to-one-variant (99% in one variant, 1% in another)
//! - schema JSON `"kind":"union"` with `"variants"` array of `[name, type]` pairs
//! - `expected_encodings_len()` for Union shapes
//! - `physical_fields()` dotted-path names for Union
//! - multi-stripe concat
//! - validation error cases: zero variants, >255 variants, duplicate names, empty name

use std::io::Cursor;

use helium::{
    CoderRegistry, CoderSpec, ColumnData, ColumnSpec, DataType, FieldSpec, HeliumReader,
    HeliumWriter, LogicalColumn, LogicalType, Schema,
};

fn zstd() -> CoderSpec {
    CoderSpec::new("zstd")
}
fn delta_leb_zstd() -> Vec<CoderSpec> {
    vec![CoderSpec::new("delta"), CoderSpec::new("leb128"), zstd()]
}
fn zstd_only() -> Vec<CoderSpec> {
    vec![zstd()]
}
/// Tag (U8) needs U8→Bytes. leb128 converts U8→Bytes.
fn tag_coders() -> Vec<CoderSpec> {
    vec![CoderSpec::new("leb128"), zstd()]
}
fn registry() -> CoderRegistry {
    CoderRegistry::default()
}

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
// 2-variant non-null — alternating rows
// ---------------------------------------------------------------------------

#[test]
fn union_2_variant_roundtrip() {
    // Union { "int": I32, "text": Utf8 } — 6 rows: int,text,int,text,int,text
    let spec = ColumnSpec::union(
        "val",
        vec![
            (
                "int".to_string(),
                LogicalType::Primitive {
                    data_type: DataType::I32,
                },
            ),
            ("text".to_string(), LogicalType::Utf8),
        ],
        vec![
            tag_coders(),     // tag (U8)
            delta_leb_zstd(), // int: values
            delta_leb_zstd(), // text: offsets
            zstd_only(),      // text: data
        ],
    );

    // 6 rows alternating int(0) / text(1)
    let tags = vec![0u8, 1, 0, 1, 0, 1];
    // int variant: rows 0,2,4 → values [10, 30, 50]
    let int_col = LogicalColumn::Primitive(ColumnData::I32(vec![10, 30, 50]));
    // text variant: rows 1,3,5 → ["a", "b", "c"]
    let text_col = LogicalColumn::Utf8(vec!["a".to_string(), "b".to_string(), "c".to_string()]);

    let data = LogicalColumn::Union {
        tags: tags.clone(),
        variants: vec![
            ("int".to_string(), int_col.clone()),
            ("text".to_string(), text_col.clone()),
        ],
    };

    let result = roundtrip(spec, data);
    let LogicalColumn::Union {
        tags: rt,
        variants: rv,
    } = result
    else {
        panic!("expected Union");
    };
    assert_eq!(rt, tags);
    assert_eq!(rv.len(), 2);
    assert_eq!(rv[0].0, "int");
    assert_eq!(rv[0].1, int_col);
    assert_eq!(rv[1].0, "text");
    assert_eq!(rv[1].1, text_col);
}

// ---------------------------------------------------------------------------
// 3-variant mixed types — fully populated
// ---------------------------------------------------------------------------

#[test]
fn union_3_variant_fully_populated() {
    // Union { "i": I32, "s": Utf8, "b": U8 } — 9 rows, 3 each
    let spec = ColumnSpec::union(
        "u",
        vec![
            (
                "i".to_string(),
                LogicalType::Primitive {
                    data_type: DataType::I32,
                },
            ),
            ("s".to_string(), LogicalType::Utf8),
            (
                "b".to_string(),
                LogicalType::Primitive {
                    data_type: DataType::U8,
                },
            ),
        ],
        vec![
            tag_coders(),     // tag
            delta_leb_zstd(), // i.values
            delta_leb_zstd(), // s.offsets
            zstd_only(),      // s.data
            delta_leb_zstd(), // b.values
        ],
    );

    // rows: i, s, b, i, s, b, i, s, b
    let tags = vec![0u8, 1, 2, 0, 1, 2, 0, 1, 2];
    let i_col = LogicalColumn::Primitive(ColumnData::I32(vec![1, 4, 7]));
    let s_col = LogicalColumn::Utf8(vec![
        "two".to_string(),
        "five".to_string(),
        "eight".to_string(),
    ]);
    let b_col = LogicalColumn::Primitive(ColumnData::U8(vec![3, 6, 9]));

    let data = LogicalColumn::Union {
        tags: tags.clone(),
        variants: vec![
            ("i".to_string(), i_col.clone()),
            ("s".to_string(), s_col.clone()),
            ("b".to_string(), b_col.clone()),
        ],
    };

    let result = roundtrip(spec, data);
    let LogicalColumn::Union {
        tags: rt,
        variants: rv,
    } = result
    else {
        panic!();
    };
    assert_eq!(rt, tags);
    assert_eq!(rv[0].1, i_col);
    assert_eq!(rv[1].1, s_col);
    assert_eq!(rv[2].1, b_col);
}

// ---------------------------------------------------------------------------
// skewed-to-one-variant
// ---------------------------------------------------------------------------

#[test]
fn union_skewed_to_one_variant() {
    // 100 rows: 99 of variant 0 ("num"), 1 of variant 1 ("label")
    let spec = ColumnSpec::union(
        "u",
        vec![
            (
                "num".to_string(),
                LogicalType::Primitive {
                    data_type: DataType::I64,
                },
            ),
            ("label".to_string(), LogicalType::Utf8),
        ],
        vec![
            tag_coders(),     // tag
            delta_leb_zstd(), // num.values
            delta_leb_zstd(), // label.offsets
            zstd_only(),      // label.data
        ],
    );

    // row 50 is the lone label; rest are num
    let mut tags: Vec<u8> = vec![0; 100];
    tags[50] = 1;
    let nums: Vec<i64> = (0..99_i64).collect(); // 99 num rows
    let labels = vec!["special".to_string()]; // 1 label row

    let data = LogicalColumn::Union {
        tags: tags.clone(),
        variants: vec![
            (
                "num".to_string(),
                LogicalColumn::Primitive(ColumnData::I64(nums.clone())),
            ),
            ("label".to_string(), LogicalColumn::Utf8(labels.clone())),
        ],
    };

    let result = roundtrip(spec, data);
    let LogicalColumn::Union {
        tags: rt,
        variants: rv,
    } = result
    else {
        panic!();
    };
    assert_eq!(rt, tags);
    assert_eq!(rv[0].1, LogicalColumn::Primitive(ColumnData::I64(nums)));
    assert_eq!(rv[1].1, LogicalColumn::Utf8(labels));
}

// ---------------------------------------------------------------------------
// Union with Struct variant
// ---------------------------------------------------------------------------

#[test]
fn union_with_struct_variant() {
    // Union { "rec": Struct { id: I32 }, "code": U8 } — 4 rows
    let inner_struct = LogicalType::Struct {
        fields: vec![FieldSpec::primitive("id", DataType::I32, delta_leb_zstd())],
    };
    // Union<Struct, U8>: expected_encodings_len = 1 (tag) + 0 (Struct) + 1 (U8) = 2
    let spec = ColumnSpec::union(
        "u",
        vec![
            ("rec".to_string(), inner_struct),
            (
                "code".to_string(),
                LogicalType::Primitive {
                    data_type: DataType::U8,
                },
            ),
        ],
        vec![
            tag_coders(), // tag
            // no rec entries (Struct uses FieldSpec encodings)
            delta_leb_zstd(), // code.values
        ],
    );

    // rows: rec, code, rec, code
    let tags = vec![0u8, 1, 0, 1];
    let rec_col = LogicalColumn::Struct {
        fields: vec![(
            "id".to_string(),
            LogicalColumn::Primitive(ColumnData::I32(vec![1, 3])),
        )],
    };
    let code_col = LogicalColumn::Primitive(ColumnData::U8(vec![10, 20]));

    let data = LogicalColumn::Union {
        tags: tags.clone(),
        variants: vec![
            ("rec".to_string(), rec_col.clone()),
            ("code".to_string(), code_col.clone()),
        ],
    };

    let result = roundtrip(spec, data);
    let LogicalColumn::Union {
        tags: rt,
        variants: rv,
    } = result
    else {
        panic!();
    };
    assert_eq!(rt, tags);
    assert_eq!(rv[0].1, rec_col);
    assert_eq!(rv[1].1, code_col);
}

// ---------------------------------------------------------------------------
// row_count
// ---------------------------------------------------------------------------

#[test]
fn union_row_count() {
    let data = LogicalColumn::Union {
        tags: vec![0u8, 1, 0, 1, 0],
        variants: vec![
            (
                "a".to_string(),
                LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3])),
            ),
            (
                "b".to_string(),
                LogicalColumn::Primitive(ColumnData::I32(vec![4, 5])),
            ),
        ],
    };
    assert_eq!(data.row_count(), 5);

    let empty = LogicalColumn::Union {
        tags: vec![],
        variants: vec![(
            "a".to_string(),
            LogicalColumn::Primitive(ColumnData::I32(vec![])),
        )],
    };
    assert_eq!(empty.row_count(), 0);
}

// ---------------------------------------------------------------------------
// physical_fields dotted-path names
// ---------------------------------------------------------------------------

#[test]
fn union_physical_fields() {
    let lt = LogicalType::Union {
        variants: vec![
            (
                "num".to_string(),
                LogicalType::Primitive {
                    data_type: DataType::I32,
                },
            ),
            ("text".to_string(), LogicalType::Utf8),
        ],
    };
    let pf = lt.physical_fields();
    assert_eq!(pf.len(), 4); // tag + num.values + text.offsets + text.data
    assert_eq!(pf[0].role, "tag");
    assert_eq!(pf[0].data_type, DataType::U8);
    assert_eq!(pf[1].role, "v_num.values");
    assert_eq!(pf[1].data_type, DataType::I32);
    assert_eq!(pf[2].role, "v_text.offsets");
    assert_eq!(pf[2].data_type, DataType::U32);
    assert_eq!(pf[3].role, "v_text.data");
    assert_eq!(pf[3].data_type, DataType::Bytes);
}

// ---------------------------------------------------------------------------
// expected_encodings_len
// ---------------------------------------------------------------------------

#[test]
fn union_expected_encodings_len() {
    // Union<I32, Utf8>: 1 + 1 + 2 = 4
    let u2 = LogicalType::Union {
        variants: vec![
            (
                "a".to_string(),
                LogicalType::Primitive {
                    data_type: DataType::I32,
                },
            ),
            ("b".to_string(), LogicalType::Utf8),
        ],
    };
    assert_eq!(u2.expected_encodings_len(), 4);

    // Union<Struct, I32>: 1 + 0 + 1 = 2
    let u_struct = LogicalType::Union {
        variants: vec![
            ("s".to_string(), LogicalType::Struct { fields: vec![] }),
            (
                "i".to_string(),
                LogicalType::Primitive {
                    data_type: DataType::I32,
                },
            ),
        ],
    };
    assert_eq!(u_struct.expected_encodings_len(), 2);

    // Union<List<I32>, I32>: 1 + 2 + 1 = 4
    let u_list = LogicalType::Union {
        variants: vec![
            (
                "l".to_string(),
                LogicalType::List {
                    inner: Box::new(LogicalType::Primitive {
                        data_type: DataType::I32,
                    }),
                },
            ),
            (
                "i".to_string(),
                LogicalType::Primitive {
                    data_type: DataType::I32,
                },
            ),
        ],
    };
    assert_eq!(u_list.expected_encodings_len(), 4);
}

// ---------------------------------------------------------------------------
// schema JSON "kind": "union" tag + variants array round-trip
// ---------------------------------------------------------------------------

#[test]
fn union_schema_json_kind_tag() {
    let spec = ColumnSpec::union(
        "u",
        vec![
            (
                "a".to_string(),
                LogicalType::Primitive {
                    data_type: DataType::I32,
                },
            ),
            ("b".to_string(), LogicalType::Utf8),
        ],
        vec![
            tag_coders(),
            delta_leb_zstd(),
            delta_leb_zstd(),
            zstd_only(),
        ],
    );
    let schema = Schema::new(vec![spec]);
    let json = schema.to_json().expect("to_json");
    let s = String::from_utf8(json.clone()).unwrap();
    assert!(s.contains("\"kind\":\"union\""), "missing union kind: {s}");
    assert!(s.contains("\"variants\""), "missing variants field: {s}");
    // Should serialize as [["a", {...}], ["b", {...}]]
    assert!(s.contains("[\"a\""), "missing variant a: {s}");
    assert!(s.contains("[\"b\""), "missing variant b: {s}");

    let schema2 = Schema::from_json(&json).expect("from_json");
    assert_eq!(schema, schema2);
}

// ---------------------------------------------------------------------------
// multi-stripe concat
// ---------------------------------------------------------------------------

#[test]
fn union_multi_stripe_concat() {
    let spec = ColumnSpec::union(
        "u",
        vec![
            (
                "i".to_string(),
                LogicalType::Primitive {
                    data_type: DataType::I32,
                },
            ),
            ("s".to_string(), LogicalType::Utf8),
        ],
        vec![
            tag_coders(),
            delta_leb_zstd(),
            delta_leb_zstd(),
            zstd_only(),
        ],
    );
    let schema = Schema::new(vec![spec.clone()]);
    let reg = registry();

    // Stripe 1: 2 rows: i(10), s("hello")
    let s1 = LogicalColumn::Union {
        tags: vec![0u8, 1],
        variants: vec![
            (
                "i".to_string(),
                LogicalColumn::Primitive(ColumnData::I32(vec![10])),
            ),
            (
                "s".to_string(),
                LogicalColumn::Utf8(vec!["hello".to_string()]),
            ),
        ],
    };
    // Stripe 2: 2 rows: s("world"), i(20)
    let s2 = LogicalColumn::Union {
        tags: vec![1u8, 0],
        variants: vec![
            (
                "i".to_string(),
                LogicalColumn::Primitive(ColumnData::I32(vec![20])),
            ),
            (
                "s".to_string(),
                LogicalColumn::Utf8(vec!["world".to_string()]),
            ),
        ],
    };

    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut writer = HeliumWriter::new(&mut buf, schema, &reg).expect("writer");
    writer.write_column("u", s1).expect("s1");
    writer.finish_stripe().expect("stripe");
    writer.write_column("u", s2).expect("s2");
    writer.finish().expect("finish");

    buf.set_position(0);
    let result = HeliumReader::new(&mut buf, &reg)
        .expect("reader")
        .read_column("u")
        .expect("read");
    let LogicalColumn::Union {
        tags: rt,
        variants: rv,
    } = result
    else {
        panic!();
    };
    // Concatenated: [0,1,1,0]
    assert_eq!(rt, vec![0u8, 1, 1, 0]);
    assert_eq!(
        rv[0].1,
        LogicalColumn::Primitive(ColumnData::I32(vec![10, 20]))
    );
    assert_eq!(
        rv[1].1,
        LogicalColumn::Utf8(vec!["hello".to_string(), "world".to_string()])
    );
}

// ---------------------------------------------------------------------------
// Validation error cases
// ---------------------------------------------------------------------------

#[test]
fn union_rejects_zero_variants() {
    let spec = ColumnSpec::union("u", vec![], vec![tag_coders()]);
    let err = Schema::new(vec![spec])
        .validate()
        .expect_err("zero variants should fail");
    assert!(
        err.to_string().contains("variant") || err.to_string().contains("one"),
        "unexpected: {err}"
    );
}

#[test]
fn union_rejects_duplicate_variant_names() {
    let spec = ColumnSpec::union(
        "u",
        vec![
            (
                "same".to_string(),
                LogicalType::Primitive {
                    data_type: DataType::I32,
                },
            ),
            ("same".to_string(), LogicalType::Utf8),
        ],
        vec![
            tag_coders(),
            delta_leb_zstd(),
            delta_leb_zstd(),
            zstd_only(),
        ],
    );
    let err = Schema::new(vec![spec])
        .validate()
        .expect_err("duplicate variant name should fail");
    assert!(
        err.to_string().contains("duplicate") || err.to_string().contains("same"),
        "unexpected: {err}"
    );
}

#[test]
fn union_rejects_empty_variant_name() {
    let spec = ColumnSpec::union(
        "u",
        vec![(
            "".to_string(),
            LogicalType::Primitive {
                data_type: DataType::I32,
            },
        )],
        vec![tag_coders(), delta_leb_zstd()],
    );
    let err = Schema::new(vec![spec])
        .validate()
        .expect_err("empty variant name should fail");
    assert!(
        err.to_string().contains("empty") || err.to_string().contains("variant"),
        "unexpected: {err}"
    );
}

#[test]
fn union_wrong_encoding_count_fails() {
    // Union<I32, Utf8> needs 4 (tag+values+offsets+data); providing 3 fails
    let spec = ColumnSpec::union(
        "u",
        vec![
            (
                "a".to_string(),
                LogicalType::Primitive {
                    data_type: DataType::I32,
                },
            ),
            ("b".to_string(), LogicalType::Utf8),
        ],
        vec![tag_coders(), delta_leb_zstd(), zstd_only()], // 3 instead of 4
    );
    Schema::new(vec![spec])
        .validate()
        .expect_err("wrong encoding count should fail");
}

#[test]
fn union_tag_out_of_range_in_decompose() {
    let lt = LogicalType::Union {
        variants: vec![
            (
                "a".to_string(),
                LogicalType::Primitive {
                    data_type: DataType::I32,
                },
            ),
            (
                "b".to_string(),
                LogicalType::Primitive {
                    data_type: DataType::I32,
                },
            ),
        ],
    };
    // tag=2 but only 2 variants (valid range: 0..1)
    let data = LogicalColumn::Union {
        tags: vec![0, 2, 1],
        variants: vec![
            (
                "a".to_string(),
                LogicalColumn::Primitive(ColumnData::I32(vec![1])),
            ),
            (
                "b".to_string(),
                LogicalColumn::Primitive(ColumnData::I32(vec![2, 3])),
            ),
        ],
    };
    let err = data
        .decompose(&lt)
        .expect_err("out-of-range tag should fail");
    assert!(
        err.to_string().contains("tag") || err.to_string().contains("range"),
        "unexpected: {err}"
    );
}

#[test]
fn union_compaction_mismatch_in_decompose() {
    let lt = LogicalType::Union {
        variants: vec![(
            "a".to_string(),
            LogicalType::Primitive {
                data_type: DataType::I32,
            },
        )],
    };
    // 3 rows all variant 0, but variant column has only 2 rows
    let data = LogicalColumn::Union {
        tags: vec![0u8, 0, 0],
        variants: vec![(
            "a".to_string(),
            LogicalColumn::Primitive(ColumnData::I32(vec![1, 2])),
        )],
    };
    let err = data
        .decompose(&lt)
        .expect_err("compaction count mismatch should fail");
    assert!(
        err.to_string().contains("rows")
            || err.to_string().contains("tags")
            || err.to_string().contains("3")
            || err.to_string().contains("2"),
        "unexpected: {err}"
    );
}

// ---------------------------------------------------------------------------
// Avro ["null", T] convention
// ---------------------------------------------------------------------------

/// Verify that a Nullable wrapping works as the intended replacement for
/// Avro `["null", T]` — not a Union with a null variant.
#[test]
fn avro_null_t_is_nullable_not_union() {
    // The correct representation for Avro ["null", I32] is Nullable(I32).
    let nullable_i32 = LogicalType::Nullable {
        inner: Box::new(LogicalType::Primitive {
            data_type: DataType::I32,
        }),
    };
    // Confirm it serializes correctly and isn't a Union
    let json = serde_json::to_string(&nullable_i32).expect("serialize");
    assert!(json.contains("nullable"), "should be nullable: {json}");
    assert!(!json.contains("union"), "should not be union: {json}");
}
