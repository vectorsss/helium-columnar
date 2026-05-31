//! Round-trip tests for `LogicalType::Nullable` plus carry-over
//! regression tests.

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
/// Present bitmap (U8) needs U8→Bytes. leb128 converts U8→Bytes, then zstd compresses.
fn present_coders() -> Vec<CoderSpec> {
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
// Nullable(Primitive) — all-null, all-present, mixed
// ---------------------------------------------------------------------------

#[test]
fn nullable_primitive_all_present() {
    let spec = ColumnSpec::nullable(
        "v",
        LogicalType::Primitive {
            data_type: DataType::I32,
        },
        vec![present_coders(), delta_leb_zstd()],
    );
    let present = vec![true, true, true, true];
    let values = ColumnData::I32(vec![10, 20, 30, 40]);
    let data = LogicalColumn::Nullable {
        present: present.clone(),
        value: Box::new(LogicalColumn::Primitive(values.clone())),
    };
    let result = roundtrip(spec, data);
    let LogicalColumn::Nullable {
        present: rp,
        value: rv,
    } = result
    else {
        panic!("expected Nullable");
    };
    assert_eq!(rp, present);
    assert_eq!(*rv, LogicalColumn::Primitive(values));
}

#[test]
fn nullable_primitive_all_null() {
    let spec = ColumnSpec::nullable(
        "v",
        LogicalType::Primitive {
            data_type: DataType::I64,
        },
        vec![present_coders(), delta_leb_zstd()],
    );
    let present = vec![false, false, false];
    let data = LogicalColumn::Nullable {
        present: present.clone(),
        value: Box::new(LogicalColumn::Primitive(ColumnData::I64(vec![]))),
    };
    let result = roundtrip(spec, data);
    let LogicalColumn::Nullable {
        present: rp,
        value: rv,
    } = result
    else {
        panic!();
    };
    assert_eq!(rp, present);
    assert_eq!(*rv, LogicalColumn::Primitive(ColumnData::I64(vec![])));
}

#[test]
fn nullable_primitive_mixed() {
    // 5 rows: None, Some(1), None, Some(2), Some(3)
    let spec = ColumnSpec::nullable(
        "v",
        LogicalType::Primitive {
            data_type: DataType::I32,
        },
        vec![present_coders(), delta_leb_zstd()],
    );
    let present = vec![false, true, false, true, true];
    let values = ColumnData::I32(vec![1, 2, 3]); // compacted: only non-null
    let data = LogicalColumn::Nullable {
        present: present.clone(),
        value: Box::new(LogicalColumn::Primitive(values.clone())),
    };
    let result = roundtrip(spec, data);
    let LogicalColumn::Nullable {
        present: rp,
        value: rv,
    } = result
    else {
        panic!();
    };
    assert_eq!(rp, present);
    assert_eq!(*rv, LogicalColumn::Primitive(values));
}

// ---------------------------------------------------------------------------
// Nullable(Struct)
// ---------------------------------------------------------------------------

#[test]
fn nullable_struct_roundtrip() {
    // Nullable<Struct { id: I32, label: Utf8 }>
    // 3 rows: Some({id:1,label:"a"}), None, Some({id:2,label:"b"})
    let inner_struct = LogicalType::Struct {
        fields: vec![
            FieldSpec::primitive("id", DataType::I32, delta_leb_zstd()),
            FieldSpec::utf8("label", delta_leb_zstd(), zstd_only()),
        ],
    };
    // Nullable<Struct>: expected_encodings_len = 1 (present) + 0 (Struct) = 1
    let spec = ColumnSpec::nullable("rec", inner_struct, vec![present_coders()]);

    let present = vec![true, false, true];
    let inner = LogicalColumn::Struct {
        fields: vec![
            (
                "id".to_string(),
                LogicalColumn::Primitive(ColumnData::I32(vec![1, 2])),
            ),
            (
                "label".to_string(),
                LogicalColumn::Utf8(vec!["a".to_string(), "b".to_string()]),
            ),
        ],
    };
    let data = LogicalColumn::Nullable {
        present: present.clone(),
        value: Box::new(inner),
    };

    let result = roundtrip(spec, data);
    let LogicalColumn::Nullable {
        present: rp,
        value: rv,
    } = result
    else {
        panic!();
    };
    assert_eq!(rp, present);
    let LogicalColumn::Struct { fields } = *rv else {
        panic!("expected inner Struct");
    };
    assert_eq!(
        fields[0].1,
        LogicalColumn::Primitive(ColumnData::I32(vec![1, 2]))
    );
    assert_eq!(
        fields[1].1,
        LogicalColumn::Utf8(vec!["a".to_string(), "b".to_string()])
    );
}

// ---------------------------------------------------------------------------
// Struct { f: Nullable(T) }
// ---------------------------------------------------------------------------

#[test]
fn struct_with_nullable_field() {
    // Struct { id: I32, score: Nullable(I32) }
    let spec = ColumnSpec::struct_col(
        "row",
        vec![
            FieldSpec::primitive("id", DataType::I32, delta_leb_zstd()),
            FieldSpec::nullable(
                "score",
                LogicalType::Primitive {
                    data_type: DataType::I32,
                },
                vec![present_coders(), delta_leb_zstd()],
            ),
        ],
    );

    let ids = ColumnData::I32(vec![1, 2, 3, 4]);
    let present = vec![true, false, true, false];
    let scores = ColumnData::I32(vec![100, 300]); // compacted

    let data = LogicalColumn::Struct {
        fields: vec![
            ("id".to_string(), LogicalColumn::Primitive(ids.clone())),
            (
                "score".to_string(),
                LogicalColumn::Nullable {
                    present: present.clone(),
                    value: Box::new(LogicalColumn::Primitive(scores.clone())),
                },
            ),
        ],
    };

    let result = roundtrip(spec, data);
    let LogicalColumn::Struct { fields } = result else {
        panic!();
    };
    assert_eq!(fields[0].1, LogicalColumn::Primitive(ids));
    let LogicalColumn::Nullable {
        present: rp,
        value: rv,
    } = &fields[1].1
    else {
        panic!();
    };
    assert_eq!(*rp, present);
    assert_eq!(**rv, LogicalColumn::Primitive(scores));
}

// ---------------------------------------------------------------------------
// Nullable(List<T>)
// ---------------------------------------------------------------------------

#[test]
fn nullable_list_roundtrip() {
    // Nullable<List<I32>>: 4 rows: None, Some([1,2]), Some([]), Some([3])
    let inner_list = LogicalType::List {
        inner: Box::new(LogicalType::Primitive {
            data_type: DataType::I32,
        }),
    };
    // Nullable<List<Primitive>>: expected_encodings_len = 1 + (1+1) = 3
    let spec = ColumnSpec::nullable(
        "items",
        inner_list,
        vec![
            present_coders(), // present bitmap (U8→Bytes)
            delta_leb_zstd(), // list offsets
            delta_leb_zstd(), // values
        ],
    );

    let present = vec![false, true, true, true];
    // 3 non-null rows: [1,2], [], [3] → offsets [0,2,2,3], flat [1,2,3]
    let inner_data = LogicalColumn::List {
        offsets: vec![0, 2, 2, 3],
        values: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3]))),
    };
    let data = LogicalColumn::Nullable {
        present: present.clone(),
        value: Box::new(inner_data),
    };

    let result = roundtrip(spec, data);
    let LogicalColumn::Nullable {
        present: rp,
        value: rv,
    } = result
    else {
        panic!();
    };
    assert_eq!(rp, present);
    let LogicalColumn::List {
        offsets: ro,
        values: rvv,
    } = *rv
    else {
        panic!();
    };
    assert_eq!(ro, vec![0u32, 2, 2, 3]);
    assert_eq!(
        *rvv,
        LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3]))
    );
}

// ---------------------------------------------------------------------------
// List<Nullable<T>>
// ---------------------------------------------------------------------------

#[test]
fn list_of_nullable_roundtrip() {
    // List<Nullable<I32>>: 2 outer rows
    // outer offsets = [0, 3, 5] (3+2=5 inner elements)
    // inner: 5 Nullable<I32> elements: present=[T,F,T,F,T], values=[1,3,5]

    let inner_nullable = LogicalType::Nullable {
        inner: Box::new(LogicalType::Primitive {
            data_type: DataType::I32,
        }),
    };
    // List<Nullable<Primitive>>: expected_encodings_len = 1 + (1+1) = 3
    let spec = ColumnSpec::list(
        "items",
        inner_nullable,
        vec![
            delta_leb_zstd(), // outer offsets
            present_coders(), // inner present (U8→Bytes)
            delta_leb_zstd(), // inner values (compacted)
        ],
    );

    let outer_offsets = vec![0u32, 3, 5];
    let inner_data = LogicalColumn::Nullable {
        present: vec![true, false, true, false, true],
        value: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![1, 3, 5]))),
    };
    let data = LogicalColumn::List {
        offsets: outer_offsets.clone(),
        values: Box::new(inner_data),
    };

    let result = roundtrip(spec, data);
    let LogicalColumn::List {
        offsets: ro,
        values: rv,
    } = result
    else {
        panic!();
    };
    assert_eq!(ro, outer_offsets);
    let LogicalColumn::Nullable {
        present: rp,
        value: rvv,
    } = *rv
    else {
        panic!();
    };
    assert_eq!(rp, vec![true, false, true, false, true]);
    assert_eq!(
        *rvv,
        LogicalColumn::Primitive(ColumnData::I32(vec![1, 3, 5]))
    );
}

// ---------------------------------------------------------------------------
// schema JSON "kind": "nullable" tag round-trip
// ---------------------------------------------------------------------------

#[test]
fn nullable_schema_json_kind_tag() {
    let spec = ColumnSpec::nullable(
        "v",
        LogicalType::Primitive {
            data_type: DataType::I32,
        },
        vec![present_coders(), delta_leb_zstd()],
    );
    let schema = Schema::new(vec![spec]);
    let json = schema.to_json().expect("to_json");
    let s = String::from_utf8(json.clone()).unwrap();
    assert!(
        s.contains("\"kind\":\"nullable\""),
        "missing nullable kind: {s}"
    );
    assert!(s.contains("\"inner\""), "missing inner field: {s}");
    let schema2 = Schema::from_json(&json).expect("from_json");
    assert_eq!(schema, schema2);
}

// ---------------------------------------------------------------------------
// expected_encodings_len for Nullable
// ---------------------------------------------------------------------------

#[test]
fn nullable_expected_encodings_len() {
    // Nullable<Primitive>: 1 + 1 = 2
    assert_eq!(
        LogicalType::Nullable {
            inner: Box::new(LogicalType::Primitive {
                data_type: DataType::I32
            })
        }
        .expected_encodings_len(),
        2
    );
    // Nullable<Utf8>: 1 + 2 = 3
    assert_eq!(
        LogicalType::Nullable {
            inner: Box::new(LogicalType::Utf8)
        }
        .expected_encodings_len(),
        3
    );
    // Nullable<Struct>: 1 + 0 = 1
    assert_eq!(
        LogicalType::Nullable {
            inner: Box::new(LogicalType::Struct { fields: vec![] })
        }
        .expected_encodings_len(),
        1
    );
    // Nullable<List<Primitive>>: 1 + (1+1) = 3
    assert_eq!(
        LogicalType::Nullable {
            inner: Box::new(LogicalType::List {
                inner: Box::new(LogicalType::Primitive {
                    data_type: DataType::I32
                })
            })
        }
        .expected_encodings_len(),
        3
    );
    // Nullable<Nullable<Primitive>>: 1 + (1+1) = 3
    assert_eq!(
        LogicalType::Nullable {
            inner: Box::new(LogicalType::Nullable {
                inner: Box::new(LogicalType::Primitive {
                    data_type: DataType::I32
                })
            })
        }
        .expected_encodings_len(),
        3
    );
}

// ---------------------------------------------------------------------------
// physical_fields for Nullable
// ---------------------------------------------------------------------------

#[test]
fn nullable_physical_fields() {
    let lt = LogicalType::Nullable {
        inner: Box::new(LogicalType::Primitive {
            data_type: DataType::I32,
        }),
    };
    let pf = lt.physical_fields();
    assert_eq!(pf.len(), 2);
    assert_eq!(pf[0].role, "present");
    assert_eq!(pf[0].data_type, DataType::U8);
    assert_eq!(pf[1].role, "item.values");
    assert_eq!(pf[1].data_type, DataType::I32);
}

#[test]
fn nullable_struct_physical_fields() {
    // Nullable<Struct { a: I32, b: Utf8 }>
    let lt = LogicalType::Nullable {
        inner: Box::new(LogicalType::Struct {
            fields: vec![
                FieldSpec::primitive("a", DataType::I32, vec![]),
                FieldSpec::utf8("b", vec![], vec![]),
            ],
        }),
    };
    let pf = lt.physical_fields();
    assert_eq!(pf.len(), 4); // present + a.values + b.offsets + b.data
    assert_eq!(pf[0].role, "present");
    assert_eq!(pf[1].role, "item.a.values");
    assert_eq!(pf[2].role, "item.b.offsets");
    assert_eq!(pf[3].role, "item.b.data");
}

// ---------------------------------------------------------------------------
// multi-stripe concat
// ---------------------------------------------------------------------------

#[test]
fn nullable_multi_stripe_concat() {
    let spec = ColumnSpec::nullable(
        "v",
        LogicalType::Primitive {
            data_type: DataType::I32,
        },
        vec![present_coders(), delta_leb_zstd()],
    );
    let schema = Schema::new(vec![spec.clone()]);
    let reg = registry();

    let s1 = LogicalColumn::Nullable {
        present: vec![true, false],
        value: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![1]))),
    };
    let s2 = LogicalColumn::Nullable {
        present: vec![true, true],
        value: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![2, 3]))),
    };

    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut writer = HeliumWriter::new(&mut buf, schema, &reg).expect("writer");
    writer.write_column("v", s1).expect("s1");
    writer.finish_stripe().expect("stripe");
    writer.write_column("v", s2).expect("s2");
    writer.finish().expect("finish");

    buf.set_position(0);
    let result = HeliumReader::new(&mut buf, &reg)
        .expect("reader")
        .read_column("v")
        .expect("read");

    let LogicalColumn::Nullable {
        present: rp,
        value: rv,
    } = result
    else {
        panic!();
    };
    assert_eq!(rp, vec![true, false, true, true]);
    assert_eq!(
        *rv,
        LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3]))
    );
}

// ---------------------------------------------------------------------------
// Carry-over #1 regression: duplicate field name via validate_nested_type
// ---------------------------------------------------------------------------

#[test]
fn validate_rejects_duplicate_field_in_nullable_struct() {
    let spec = ColumnSpec::nullable(
        "rec",
        LogicalType::Struct {
            fields: vec![
                FieldSpec::primitive("x", DataType::I32, delta_leb_zstd()),
                FieldSpec::primitive("x", DataType::I64, delta_leb_zstd()),
            ],
        },
        vec![present_coders()],
    );
    let err = Schema::new(vec![spec]).validate().expect_err("should fail");
    assert!(
        err.to_string().contains("duplicate") || err.to_string().contains("field"),
        "unexpected: {err}"
    );
}

#[test]
fn validate_rejects_duplicate_field_top_level_struct() {
    let spec = ColumnSpec::struct_col(
        "row",
        vec![
            FieldSpec::primitive("a", DataType::I32, delta_leb_zstd()),
            FieldSpec::utf8("a", delta_leb_zstd(), zstd_only()),
        ],
    );
    let err = Schema::new(vec![spec]).validate().expect_err("should fail");
    assert!(
        err.to_string().contains("duplicate") || err.to_string().contains("field"),
        "unexpected: {err}"
    );
}

// ---------------------------------------------------------------------------
// Carry-over #2 regression: depth cap
// ---------------------------------------------------------------------------

fn build_nested_list(n: usize) -> LogicalType {
    let mut inner = LogicalType::Primitive {
        data_type: DataType::I32,
    };
    for _ in 0..n {
        inner = LogicalType::List {
            inner: Box::new(inner),
        };
    }
    inner
}

fn encodings_for_nested_list(n: usize) -> Vec<Vec<CoderSpec>> {
    // n offsets columns + 1 values column
    (0..=n).map(|_| vec![CoderSpec::new("zstd")]).collect()
}

#[test]
fn depth_cap_64_level_list_passes() {
    let lt = build_nested_list(64);
    let encodings = encodings_for_nested_list(64);
    let spec = ColumnSpec::new("x", lt, encodings);
    Schema::new(vec![spec])
        .validate()
        .expect("64-deep should pass");
}

#[test]
fn depth_cap_65_level_list_fails() {
    let lt = build_nested_list(65);
    let encodings = encodings_for_nested_list(65);
    let spec = ColumnSpec::new("x", lt, encodings);
    let err = Schema::new(vec![spec])
        .validate()
        .expect_err("65-deep should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("depth") || msg.contains("64"),
        "expected depth-cap error, got: {msg}"
    );
}

#[test]
fn depth_cap_nullable_nesting() {
    let mut inner = LogicalType::Primitive {
        data_type: DataType::I32,
    };
    for _ in 0..65 {
        inner = LogicalType::Nullable {
            inner: Box::new(inner),
        };
    }
    // 65 Nullable wrappers: encodings_len = 65 * 1 + 1 (leaf) = 66
    let encodings: Vec<Vec<CoderSpec>> = (0..66).map(|_| vec![zstd()]).collect();
    let spec = ColumnSpec::new("x", inner, encodings);
    let err = Schema::new(vec![spec])
        .validate()
        .expect_err("65-deep Nullable should fail");
    assert!(
        err.to_string().contains("depth") || err.to_string().contains("64"),
        "got: {}",
        err
    );
}

// ---------------------------------------------------------------------------
// Validation error cases
// ---------------------------------------------------------------------------

#[test]
fn nullable_wrong_encoding_count_fails() {
    // Nullable<Primitive> needs 2 (present + values); providing 3 fails.
    let spec = ColumnSpec::nullable(
        "v",
        LogicalType::Primitive {
            data_type: DataType::I32,
        },
        vec![present_coders(), delta_leb_zstd(), delta_leb_zstd()],
    );
    Schema::new(vec![spec])
        .validate()
        .expect_err("should fail with wrong count");
}

#[test]
fn nullable_present_value_count_mismatch_in_decompose() {
    let lt = LogicalType::Nullable {
        inner: Box::new(LogicalType::Primitive {
            data_type: DataType::I32,
        }),
    };
    // present has 2 true bits but values has 3 entries
    let data = LogicalColumn::Nullable {
        present: vec![true, false, true],
        value: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3]))),
    };
    let err = data.decompose(&lt).expect_err("should fail");
    assert!(
        err.to_string().contains("present") || err.to_string().contains("count"),
        "unexpected: {err}"
    );
}

// ---------------------------------------------------------------------------
// Carry-over #2: depth cap for Map
// ---------------------------------------------------------------------------

/// Build a 65-deep Map<Utf8, Map<Utf8, ...Map<Utf8, I32>...>> structure.
fn build_nested_map(n: usize) -> LogicalType {
    let mut inner = LogicalType::Primitive {
        data_type: DataType::I32,
    };
    for _ in 0..n {
        inner = LogicalType::Map {
            key: Box::new(LogicalType::Utf8),
            value: Box::new(inner),
        };
    }
    inner
}

/// Flat encodings for an n-deep Map<Utf8, ...> nesting.
fn encodings_for_nested_map(n: usize) -> Vec<Vec<CoderSpec>> {
    // Each Map contributes: 1 (offsets) + 2 (Utf8 key) = 3 encodings
    // plus 1 for the leaf I32 values
    (0..n * 3 + 1)
        .map(|_| vec![CoderSpec::new("zstd")])
        .collect()
}

#[test]
fn depth_cap_65_level_map_fails() {
    let lt = build_nested_map(65);
    let encodings = encodings_for_nested_map(65);
    let spec = ColumnSpec::new("x", lt, encodings);
    let err = Schema::new(vec![spec])
        .validate()
        .expect_err("65-deep Map should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("depth") || msg.contains("64"),
        "expected depth-cap error, got: {msg}"
    );
}
