//! Integration tests for helium-optimizer.
//!
//! Covers:
//! - Scalar column types: Primitive(I32/I64/F32/F64), Utf8, Binary
//! - recursive nested: Struct, List<Primitive>, List<Utf8>, Map<Utf8,Primitive>,
//!   Nullable<Struct>, deep nesting (5 levels)
//! - Semantic types: Decimal128, Date32 (Date{Days}), Date64 (Date{Millis}), Datetime
//! - Edge cases: all-null Nullable, empty Struct, single-row List, all-same-value column

use helium::optimizer::{Optimizer, measure_encoding};
use helium::{
    CoderRegistry, CoderSpec, ColumnData, ColumnSpec, DataType, DateUnit, FieldSpec, HeliumReader,
    HeliumWriter, LogicalColumn, LogicalType, Schema, TimeUnit,
};

// ---------------------------------------------------------------------------
// Test utilities
// ---------------------------------------------------------------------------

fn default_registry() -> CoderRegistry {
    CoderRegistry::default()
}

/// Write and read back a Schema + LogicalColumn to verify round-trip correctness.
fn roundtrip_spec(spec: &ColumnSpec, lc: LogicalColumn) -> LogicalColumn {
    let registry = default_registry();
    let schema = Schema::new(vec![spec.clone()]);
    let mut buf = Vec::<u8>::new();
    let mut writer = HeliumWriter::new(std::io::Cursor::new(&mut buf), schema, &registry)
        .expect("HeliumWriter::new");
    writer.write_column(&spec.name, lc).expect("write_column");
    writer.finish().expect("finish");

    let cursor = std::io::Cursor::new(buf);
    let registry = default_registry();
    let mut reader = HeliumReader::new(cursor, &registry).expect("HeliumReader::new");
    reader.read_column(&spec.name).expect("read_column")
}

// ---------------------------------------------------------------------------
// Flat type tests
// ---------------------------------------------------------------------------

#[test]
fn optimize_primitive_i32() {
    let values: Vec<i32> = (0..1000).map(|i| i * 7).collect();
    let lc = LogicalColumn::Primitive(ColumnData::I32(values.clone()));
    let lt = LogicalType::Primitive {
        data_type: DataType::I32,
    };

    let schema = Optimizer::new()
        .optimize(vec![("col".into(), lt, lc)])
        .expect("optimize");
    assert_eq!(schema.columns.len(), 1);

    // encodings length must match expected_encodings_len for Primitive(I32) = 1
    assert_eq!(schema.columns[0].encodings.len(), 1);

    // round-trip correctness
    let spec = &schema.columns[0];
    let lc2 = LogicalColumn::Primitive(ColumnData::I32(values.clone()));
    let result = roundtrip_spec(spec, lc2);
    assert_eq!(result, LogicalColumn::Primitive(ColumnData::I32(values)));
}

#[test]
fn optimize_primitive_i64_monotonic() {
    // Monotonically increasing — should favour delta candidates
    let values: Vec<i64> = (0..1000).map(|i| i * 1000).collect();
    let lc = LogicalColumn::Primitive(ColumnData::I64(values.clone()));
    let lt = LogicalType::Primitive {
        data_type: DataType::I64,
    };

    let schema = Optimizer::new()
        .optimize(vec![("ts".into(), lt, lc)])
        .expect("optimize");
    let spec = &schema.columns[0];
    let lc2 = LogicalColumn::Primitive(ColumnData::I64(values.clone()));
    let result = roundtrip_spec(spec, lc2);
    assert_eq!(result, LogicalColumn::Primitive(ColumnData::I64(values)));
}

#[test]
fn optimize_primitive_f64() {
    let values: Vec<f64> = (0..500).map(|i| i as f64 * 1.5).collect();
    let lc = LogicalColumn::Primitive(ColumnData::F64(values.clone()));
    let lt = LogicalType::Primitive {
        data_type: DataType::F64,
    };

    let schema = Optimizer::new()
        .optimize(vec![("fp".into(), lt, lc)])
        .expect("optimize");
    let spec = &schema.columns[0];
    let lc2 = LogicalColumn::Primitive(ColumnData::F64(values.clone()));
    let result = roundtrip_spec(spec, lc2);
    // F64 round-trip (gorilla or pcodec)
    if let LogicalColumn::Primitive(ColumnData::F64(v)) = result {
        assert_eq!(v.len(), 500);
    } else {
        panic!("unexpected column type");
    }
}

#[test]
fn optimize_utf8() {
    let strings: Vec<String> = (0..200).map(|i| format!("value_{i:04}")).collect();
    let lc = LogicalColumn::Utf8(strings.clone());
    let lt = LogicalType::Utf8;

    let schema = Optimizer::new()
        .optimize(vec![("s".into(), lt, lc)])
        .expect("optimize");
    let spec = &schema.columns[0];
    // Utf8 = 2 encodings (offsets + data)
    assert_eq!(spec.encodings.len(), 2);
    let result = roundtrip_spec(spec, LogicalColumn::Utf8(strings.clone()));
    assert_eq!(result, LogicalColumn::Utf8(strings));
}

#[test]
fn optimize_binary() {
    let blobs: Vec<Vec<u8>> = (0..100u8).map(|i| vec![i, i + 1, i + 2]).collect();
    let lc = LogicalColumn::Binary(blobs.clone());
    let lt = LogicalType::Binary;

    let schema = Optimizer::new()
        .optimize(vec![("b".into(), lt, lc)])
        .expect("optimize");
    let spec = &schema.columns[0];
    assert_eq!(spec.encodings.len(), 2);
    let result = roundtrip_spec(spec, LogicalColumn::Binary(blobs.clone()));
    assert_eq!(result, LogicalColumn::Binary(blobs));
}

// ---------------------------------------------------------------------------
// Nullable column types
// ---------------------------------------------------------------------------

#[test]
fn optimize_nullable_prim_i32() {
    let present: Vec<bool> = (0..100).map(|i| i % 3 != 0).collect();
    let values: Vec<i32> = present
        .iter()
        .filter(|&&p| p)
        .enumerate()
        .map(|(i, _)| i as i32)
        .collect();
    let lc = LogicalColumn::Nullable {
        present: present.clone(),
        value: Box::new(LogicalColumn::Primitive(ColumnData::I32(values.clone()))),
    };
    let lt = LogicalType::Nullable {
        inner: Box::new(LogicalType::Primitive {
            data_type: DataType::I32,
        }),
    };

    let schema = Optimizer::new()
        .optimize(vec![("n".into(), lt, lc)])
        .expect("optimize");
    let spec = &schema.columns[0];
    // Nullable(Primitive) = 2 encodings (present + values)
    assert_eq!(spec.encodings.len(), 2);
    let lc2 = LogicalColumn::Nullable {
        present: present.clone(),
        value: Box::new(LogicalColumn::Primitive(ColumnData::I32(values.clone()))),
    };
    let result = roundtrip_spec(spec, lc2);
    assert_eq!(
        result,
        LogicalColumn::Nullable {
            present,
            value: Box::new(LogicalColumn::Primitive(ColumnData::I32(values))),
        }
    );
}

#[test]
fn optimize_flat_nullable_utf8() {
    let present: Vec<bool> = (0..50).map(|i| i % 4 != 0).collect();
    let strings: Vec<String> = present
        .iter()
        .filter(|&&p| p)
        .enumerate()
        .map(|(i, _)| format!("s{i}"))
        .collect();
    let lc = LogicalColumn::Nullable {
        present: present.clone(),
        value: Box::new(LogicalColumn::Utf8(strings.clone())),
    };
    let lt = LogicalType::Nullable {
        inner: Box::new(LogicalType::Utf8),
    };

    let schema = Optimizer::new()
        .optimize(vec![("ns".into(), lt, lc)])
        .expect("optimize");
    let spec = &schema.columns[0];
    assert_eq!(spec.encodings.len(), 3); // present + offsets + data
    let lc2 = LogicalColumn::Nullable {
        present: present.clone(),
        value: Box::new(LogicalColumn::Utf8(strings.clone())),
    };
    let result = roundtrip_spec(spec, lc2);
    assert_eq!(
        result,
        LogicalColumn::Nullable {
            present,
            value: Box::new(LogicalColumn::Utf8(strings)),
        }
    );
}

#[test]
fn optimize_dict_prim_i32() {
    // Low cardinality: 5 distinct values repeated
    let values: Vec<i32> = (0..500).map(|i| i % 5).collect();
    let dict_col =
        LogicalColumn::dict_encode_primitive(ColumnData::I32(values.clone())).expect("dict_encode");
    let lt = LogicalType::Dictionary {
        inner: Box::new(LogicalType::Primitive {
            data_type: DataType::I32,
        }),
    };

    let schema = Optimizer::new()
        .optimize(vec![("d".into(), lt, dict_col)])
        .expect("optimize");
    let spec = &schema.columns[0];
    assert_eq!(spec.encodings.len(), 2); // inner(Primitive(I32)) + indices

    // verify round-trip
    let dict_col2 =
        LogicalColumn::dict_encode_primitive(ColumnData::I32(values.clone())).expect("dict_encode");
    let result = roundtrip_spec(spec, dict_col2);
    // Read back as Dictionary and materialize to verify
    if let LogicalColumn::Dictionary {
        dictionary,
        indices,
    } = result
    {
        if let LogicalColumn::Primitive(ColumnData::I32(dict)) = *dictionary {
            let decoded: Vec<i32> = indices.iter().map(|&i| dict[i as usize]).collect();
            assert_eq!(decoded, values);
        } else {
            panic!("unexpected dictionary inner type");
        }
    } else {
        panic!("unexpected column type");
    }
}

// ---------------------------------------------------------------------------
// recursive Struct
// ---------------------------------------------------------------------------

#[test]
fn optimize_struct_flat() {
    // Struct with two primitive fields
    let n = 200;
    let fields_lc = vec![
        (
            "x".to_string(),
            LogicalColumn::Primitive(ColumnData::I32((0..n).collect())),
        ),
        (
            "y".to_string(),
            LogicalColumn::Primitive(ColumnData::F64((0..n).map(|i| i as f64 * 0.1).collect())),
        ),
    ];
    let lc = LogicalColumn::Struct {
        fields: fields_lc.clone(),
    };

    // skeleton: Struct with empty encodings in FieldSpec
    let lt = LogicalType::Struct {
        fields: vec![
            FieldSpec::new(
                "x",
                LogicalType::Primitive {
                    data_type: DataType::I32,
                },
                vec![],
            ),
            FieldSpec::new(
                "y",
                LogicalType::Primitive {
                    data_type: DataType::F64,
                },
                vec![],
            ),
        ],
    };

    let schema = Optimizer::new()
        .optimize(vec![("s".into(), lt, lc)])
        .expect("optimize");
    let spec = &schema.columns[0];
    // Struct has empty top-level encodings; leaf encodings are in FieldSpec
    assert!(spec.encodings.is_empty());
    if let LogicalType::Struct { fields } = &spec.logical_type {
        assert_eq!(fields.len(), 2);
        // Each field should have 1 encoding (for Primitive)
        assert_eq!(fields[0].encodings.len(), 1);
        assert_eq!(fields[1].encodings.len(), 1);
    } else {
        panic!("expected Struct");
    }

    // round-trip
    let lc2 = LogicalColumn::Struct { fields: fields_lc };
    let result = roundtrip_spec(spec, lc2);
    if let LogicalColumn::Struct { fields } = result {
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].0, "x");
        assert_eq!(fields[1].0, "y");
    } else {
        panic!("expected Struct");
    }
}

#[test]
fn optimize_struct_empty_fields() {
    // Edge case: empty struct
    let lc = LogicalColumn::Struct { fields: vec![] };
    let lt = LogicalType::Struct { fields: vec![] };

    let schema = Optimizer::new()
        .optimize(vec![("empty".into(), lt, lc)])
        .expect("optimize");
    let spec = &schema.columns[0];
    assert!(spec.encodings.is_empty());
    if let LogicalType::Struct { fields } = &spec.logical_type {
        assert!(fields.is_empty());
    }
}

// ---------------------------------------------------------------------------
// recursive List
// ---------------------------------------------------------------------------

#[test]
fn optimize_list_primitive_i32() {
    // List<I32>: 3 rows with varying lengths
    let offsets = vec![0u32, 3, 5, 8];
    let items: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let lc = LogicalColumn::List {
        offsets: offsets.clone(),
        values: Box::new(LogicalColumn::Primitive(ColumnData::I32(items.clone()))),
    };
    let lt = LogicalType::List {
        inner: Box::new(LogicalType::Primitive {
            data_type: DataType::I32,
        }),
    };

    let schema = Optimizer::new()
        .optimize(vec![("li".into(), lt, lc)])
        .expect("optimize");
    let spec = &schema.columns[0];
    // List<Primitive(I32)>: 2 encodings (offsets + values)
    assert_eq!(spec.encodings.len(), 2);

    // round-trip
    let lc2 = LogicalColumn::List {
        offsets: offsets.clone(),
        values: Box::new(LogicalColumn::Primitive(ColumnData::I32(items.clone()))),
    };
    let result = roundtrip_spec(spec, lc2);
    if let LogicalColumn::List {
        offsets: off2,
        values: vals2,
    } = result
    {
        assert_eq!(off2, offsets);
        assert_eq!(*vals2, LogicalColumn::Primitive(ColumnData::I32(items)));
    } else {
        panic!("expected List");
    }
}

#[test]
fn optimize_list_utf8() {
    // List<Utf8>: variable-length string lists per row
    let offsets = vec![0u32, 2, 3, 5];
    let strings = vec![
        "a".to_string(),
        "bb".to_string(),
        "ccc".to_string(),
        "d".to_string(),
        "ee".to_string(),
    ];
    let lc = LogicalColumn::List {
        offsets: offsets.clone(),
        values: Box::new(LogicalColumn::Utf8(strings.clone())),
    };
    let lt = LogicalType::List {
        inner: Box::new(LogicalType::Utf8),
    };

    let schema = Optimizer::new()
        .optimize(vec![("ls".into(), lt, lc)])
        .expect("optimize");
    let spec = &schema.columns[0];
    // List<Utf8>: 3 encodings (list_offsets + str_offsets + str_data)
    assert_eq!(spec.encodings.len(), 3);
}

#[test]
fn optimize_list_single_row() {
    // Edge case: single-row list
    let offsets = vec![0u32, 3];
    let items = vec![10i64, 20, 30];
    let lc = LogicalColumn::List {
        offsets: offsets.clone(),
        values: Box::new(LogicalColumn::Primitive(ColumnData::I64(items.clone()))),
    };
    let lt = LogicalType::List {
        inner: Box::new(LogicalType::Primitive {
            data_type: DataType::I64,
        }),
    };

    let schema = Optimizer::new()
        .optimize(vec![("l1".into(), lt, lc)])
        .expect("optimize");
    let spec = &schema.columns[0];
    assert_eq!(spec.encodings.len(), 2);
}

// ---------------------------------------------------------------------------
// recursive Map
// ---------------------------------------------------------------------------

#[test]
fn optimize_map_utf8_to_primitive() {
    // Map<Utf8, I32>: 3 rows
    let offsets = vec![0u32, 2, 3, 5];
    let keys = LogicalColumn::Utf8(vec![
        "a".into(),
        "b".into(),
        "c".into(),
        "d".into(),
        "e".into(),
    ]);
    let values = LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3, 4, 5]));
    let lc = LogicalColumn::Map {
        offsets: offsets.clone(),
        keys: Box::new(keys),
        values: Box::new(values),
    };
    let lt = LogicalType::Map {
        key: Box::new(LogicalType::Utf8),
        value: Box::new(LogicalType::Primitive {
            data_type: DataType::I32,
        }),
    };

    let schema = Optimizer::new()
        .optimize(vec![("m".into(), lt, lc)])
        .expect("optimize");
    let spec = &schema.columns[0];
    // Map<Utf8, I32>: offsets + 2(Utf8 key) + 1(I32 value) = 4
    assert_eq!(spec.encodings.len(), 4);
}

// ---------------------------------------------------------------------------
// recursive Nullable
// ---------------------------------------------------------------------------

#[test]
fn optimize_nullable_primitive() {
    // Nullable<I32>
    let present: Vec<bool> = (0..100).map(|i| i % 5 != 0).collect();
    let values: Vec<i32> = present
        .iter()
        .filter(|&&p| p)
        .enumerate()
        .map(|(i, _)| i as i32 * 3)
        .collect();
    let lc = LogicalColumn::Nullable {
        present: present.clone(),
        value: Box::new(LogicalColumn::Primitive(ColumnData::I32(values.clone()))),
    };
    let lt = LogicalType::Nullable {
        inner: Box::new(LogicalType::Primitive {
            data_type: DataType::I32,
        }),
    };

    let schema = Optimizer::new()
        .optimize(vec![("ni".into(), lt, lc)])
        .expect("optimize");
    let spec = &schema.columns[0];
    // Nullable<Primitive(I32)>: 2 encodings (present + values)
    assert_eq!(spec.encodings.len(), 2);

    // round-trip
    let lc2 = LogicalColumn::Nullable {
        present: present.clone(),
        value: Box::new(LogicalColumn::Primitive(ColumnData::I32(values.clone()))),
    };
    let result = roundtrip_spec(spec, lc2);
    if let LogicalColumn::Nullable {
        present: p2,
        value: vals2,
    } = result
    {
        assert_eq!(p2, present);
        assert_eq!(*vals2, LogicalColumn::Primitive(ColumnData::I32(values)));
    } else {
        panic!("expected Nullable");
    }
}

#[test]
fn optimize_recursive_nullable_utf8() {
    // recursive Nullable<Utf8>
    let present: Vec<bool> = (0..50).map(|i| i % 3 != 0).collect();
    let strings: Vec<String> = present
        .iter()
        .filter(|&&p| p)
        .enumerate()
        .map(|(i, _)| format!("item_{i}"))
        .collect();
    let lc = LogicalColumn::Nullable {
        present: present.clone(),
        value: Box::new(LogicalColumn::Utf8(strings.clone())),
    };
    let lt = LogicalType::Nullable {
        inner: Box::new(LogicalType::Utf8),
    };

    let schema = Optimizer::new()
        .optimize(vec![("ns".into(), lt, lc)])
        .expect("optimize");
    let spec = &schema.columns[0];
    // Nullable<Utf8>: 3 encodings (present + str_offsets + str_data)
    assert_eq!(spec.encodings.len(), 3);
}

#[test]
fn optimize_all_null_nullable() {
    // Edge case: all-null Nullable
    let present = vec![false; 100];
    let lc = LogicalColumn::Nullable {
        present: present.clone(),
        value: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![]))),
    };
    let lt = LogicalType::Nullable {
        inner: Box::new(LogicalType::Primitive {
            data_type: DataType::I32,
        }),
    };

    let schema = Optimizer::new()
        .optimize(vec![("allnull".into(), lt, lc)])
        .expect("optimize");
    let spec = &schema.columns[0];
    assert_eq!(spec.encodings.len(), 2);

    // round-trip
    let lc2 = LogicalColumn::Nullable {
        present: present.clone(),
        value: Box::new(LogicalColumn::Primitive(ColumnData::I32(vec![]))),
    };
    let result = roundtrip_spec(spec, lc2);
    if let LogicalColumn::Nullable {
        present: p2,
        value: vals2,
    } = result
    {
        assert_eq!(p2, present);
        assert_eq!(*vals2, LogicalColumn::Primitive(ColumnData::I32(vec![])));
    }
}

// ---------------------------------------------------------------------------
// recursive Struct with nested types
// ---------------------------------------------------------------------------

#[test]
fn optimize_nullable_struct() {
    // Nullable<Struct<x: I32, y: Utf8>>
    let present = vec![true, false, true, true, false];
    let struct_count = present.iter().filter(|&&p| p).count();
    let struct_fields = vec![
        (
            "x".to_string(),
            LogicalColumn::Primitive(ColumnData::I32((0..struct_count as i32).collect())),
        ),
        (
            "y".to_string(),
            LogicalColumn::Utf8((0..struct_count).map(|i| format!("y{i}")).collect()),
        ),
    ];
    let inner_lc = LogicalColumn::Struct {
        fields: struct_fields,
    };
    let lc = LogicalColumn::Nullable {
        present: present.clone(),
        value: Box::new(inner_lc),
    };
    let lt = LogicalType::Nullable {
        inner: Box::new(LogicalType::Struct {
            fields: vec![
                FieldSpec::new(
                    "x",
                    LogicalType::Primitive {
                        data_type: DataType::I32,
                    },
                    vec![],
                ),
                FieldSpec::new("y", LogicalType::Utf8, vec![]),
            ],
        }),
    };

    let schema = Optimizer::new()
        .optimize(vec![("ns".into(), lt, lc)])
        .expect("optimize");
    let spec = &schema.columns[0];
    // Nullable<Struct>: 1 encoding (present) + Struct contributes 0 (its fields have their own)
    assert_eq!(spec.encodings.len(), 1);
    if let LogicalType::Nullable { inner } = &spec.logical_type {
        if let LogicalType::Struct { fields } = inner.as_ref() {
            assert_eq!(fields.len(), 2);
            assert_eq!(fields[0].encodings.len(), 1); // Primitive: 1
            assert_eq!(fields[1].encodings.len(), 2); // Utf8: 2 (offsets + data)
        } else {
            panic!("expected inner Struct");
        }
    } else {
        panic!("expected Nullable");
    }
}

// ---------------------------------------------------------------------------
// recursive Union
// ---------------------------------------------------------------------------

#[test]
fn optimize_union_two_variants() {
    // Union<a: I32, b: Utf8>
    let n = 50usize;
    let tags: Vec<u8> = (0..n as u8).map(|i| i % 2).collect();
    let a_count = tags.iter().filter(|&&t| t == 0).count();
    let b_count = n - a_count;
    let a_data = LogicalColumn::Primitive(ColumnData::I32((0..a_count as i32).collect()));
    let b_data = LogicalColumn::Utf8((0..b_count).map(|i| format!("b{i}")).collect());
    let lc = LogicalColumn::Union {
        tags: tags.clone(),
        variants: vec![("a".to_string(), a_data), ("b".to_string(), b_data)],
    };
    let lt = LogicalType::Union {
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

    let schema = Optimizer::new()
        .optimize(vec![("u".into(), lt, lc)])
        .expect("optimize");
    let spec = &schema.columns[0];
    // Union<I32, Utf8>: 1 (tag) + 1 (I32 values) + 2 (Utf8: offsets + data) = 4
    assert_eq!(spec.encodings.len(), 4);
}

// ---------------------------------------------------------------------------
// Deep nesting (5 levels)
// ---------------------------------------------------------------------------

#[test]
fn optimize_deep_nesting_5_levels() {
    // Struct < List < Nullable < Map < Utf8, I64 > > >
    // Level 0: Struct { field: List<...> }
    // Level 1: List<Nullable<...>>
    // Level 2: Nullable<Map<...>>
    // Level 3: Map<Utf8, I64>
    // Level 4: leaves (Utf8 keys, I64 values)

    // Build sample data from the inside out
    // Map<Utf8, I64>: 4 entries
    let map_keys = LogicalColumn::Utf8(vec!["k1".into(), "k2".into(), "k3".into(), "k4".into()]);
    let map_values = LogicalColumn::Primitive(ColumnData::I64(vec![100, 200, 300, 400]));

    // Nullable<Map>: 2 non-null, 1 null → present=[true,false,true]
    // present=3 rows, value=2 maps (flattened)
    // The nullable wraps one "row" = one map entry sequence.
    // For simplicity: 1 present row, 1 nullable value (the one map above)
    let nullable_lc = LogicalColumn::Nullable {
        present: vec![true],
        value: Box::new(LogicalColumn::Map {
            offsets: vec![0u32, 4], // one row with 4 entries
            keys: Box::new(map_keys),
            values: Box::new(map_values),
        }),
    };

    // List<Nullable<Map>>: offsets over the nullable rows
    let list_lc = LogicalColumn::List {
        offsets: vec![0u32, 1], // one list element containing 1 nullable
        values: Box::new(nullable_lc),
    };

    // Struct { "field": List<...> }
    let struct_lc = LogicalColumn::Struct {
        fields: vec![("field".to_string(), list_lc)],
    };

    // Build the type (5 levels)
    let lt = LogicalType::Struct {
        fields: vec![FieldSpec::new(
            "field",
            LogicalType::List {
                inner: Box::new(LogicalType::Nullable {
                    inner: Box::new(LogicalType::Map {
                        key: Box::new(LogicalType::Utf8),
                        value: Box::new(LogicalType::Primitive {
                            data_type: DataType::I64,
                        }),
                    }),
                }),
            },
            vec![],
        )],
    };

    let schema = Optimizer::new()
        .optimize(vec![("deep".into(), lt, struct_lc)])
        .expect("optimize deep nesting");

    let spec = &schema.columns[0];
    // Struct top-level: empty encodings
    assert!(spec.encodings.is_empty());

    // Verify the inner structure has encodings filled in
    if let LogicalType::Struct { fields } = &spec.logical_type {
        assert_eq!(fields.len(), 1);
        let field = &fields[0];
        // List<Nullable<Map<Utf8, I64>>>:
        // list_offsets(1) + nullable_present(1) + map_offsets(1) + utf8_offsets(1) + utf8_data(1) + i64_values(1) = 6
        assert_eq!(
            field.encodings.len(),
            6,
            "List<Nullable<Map<Utf8,I64>>> should have 6 encoding vectors"
        );
    } else {
        panic!("expected Struct");
    }
}

// ---------------------------------------------------------------------------
// All-same-value column (high-repetition — should benefit from RLE or dict)
// ---------------------------------------------------------------------------

#[test]
fn optimize_all_same_value_i32() {
    // All values identical — RLE should win for Primitive
    let values: Vec<i32> = vec![42i32; 1000];
    let lc = LogicalColumn::Primitive(ColumnData::I32(values.clone()));
    let lt = LogicalType::Primitive {
        data_type: DataType::I32,
    };

    let schema = Optimizer::new()
        .optimize(vec![("const".into(), lt, lc)])
        .expect("optimize");
    let spec = &schema.columns[0];
    assert_eq!(spec.encodings.len(), 1);
    // round-trip
    let lc2 = LogicalColumn::Primitive(ColumnData::I32(values.clone()));
    let result = roundtrip_spec(spec, lc2);
    assert_eq!(result, LogicalColumn::Primitive(ColumnData::I32(values)));
}

// ---------------------------------------------------------------------------
// measure_encoding (promoted encode_one_column)
// ---------------------------------------------------------------------------

#[test]
fn measure_encoding_basic() {
    let registry = default_registry();
    let spec = ColumnSpec::primitive(
        "col",
        DataType::I32,
        vec![
            CoderSpec::new("delta"),
            CoderSpec::new("leb128"),
            CoderSpec::new("zstd"),
        ],
    );
    let values: Vec<i32> = (0..1000).collect();
    let lc = LogicalColumn::Primitive(ColumnData::I32(values));
    let size = measure_encoding(&spec, lc, &registry).expect("measure_encoding");
    assert!(size > 0, "encoded size should be > 0");
}

// ---------------------------------------------------------------------------
// Multi-column schema
// ---------------------------------------------------------------------------

#[test]
fn optimize_multi_column_schema() {
    let columns = vec![
        (
            "id".to_string(),
            LogicalType::Primitive {
                data_type: DataType::I64,
            },
            LogicalColumn::Primitive(ColumnData::I64((0..500i64).collect())),
        ),
        (
            "name".to_string(),
            LogicalType::Utf8,
            LogicalColumn::Utf8((0..500).map(|i| format!("user_{i}")).collect()),
        ),
        (
            "score".to_string(),
            LogicalType::Primitive {
                data_type: DataType::F64,
            },
            LogicalColumn::Primitive(ColumnData::F64((0..500).map(|i| i as f64 / 10.0).collect())),
        ),
    ];

    let schema = Optimizer::new()
        .optimize(columns)
        .expect("optimize multi-column");
    assert_eq!(schema.columns.len(), 3);
    assert_eq!(schema.columns[0].encodings.len(), 1); // I64
    assert_eq!(schema.columns[1].encodings.len(), 2); // Utf8
    assert_eq!(schema.columns[2].encodings.len(), 1); // F64
}

// ---------------------------------------------------------------------------
// Different terminals
// ---------------------------------------------------------------------------

#[test]
fn optimize_with_lz4_terminal() {
    // The optimizer may pick lz4 or pcodec — both are valid.
    // Check that the result is usable (valid schema, correct encoding count).
    let values: Vec<i32> = (0..500).collect();
    let lc = LogicalColumn::Primitive(ColumnData::I32(values.clone()));
    let lt = LogicalType::Primitive {
        data_type: DataType::I32,
    };

    let schema = Optimizer::with_terminal("lz4")
        .optimize(vec![("col".into(), lt, lc.clone())])
        .expect("optimize with lz4");
    let spec = &schema.columns[0];
    assert_eq!(
        spec.encodings.len(),
        1,
        "Primitive(I32) needs 1 encoding vector"
    );
    // Round-trip verification
    let result = roundtrip_spec(spec, lc);
    assert_eq!(result, LogicalColumn::Primitive(ColumnData::I32(values)));
}

#[test]
fn optimize_with_snappy_terminal() {
    // The optimizer may pick snappy or pcodec — both are valid.
    let values: Vec<i32> = (0..500).collect();
    let lc = LogicalColumn::Primitive(ColumnData::I32(values.clone()));
    let lt = LogicalType::Primitive {
        data_type: DataType::I32,
    };

    let schema = Optimizer::with_terminal("snappy")
        .optimize(vec![("col".into(), lt, lc.clone())])
        .expect("optimize with snappy");
    let spec = &schema.columns[0];
    assert_eq!(spec.encodings.len(), 1);
    // Round-trip verification
    let result = roundtrip_spec(spec, lc);
    assert_eq!(result, LogicalColumn::Primitive(ColumnData::I32(values)));
}

// ---------------------------------------------------------------------------
// Parity with flat example behavior: optimizer picks an encoding that is valid
// ---------------------------------------------------------------------------

#[test]
fn optimizer_result_is_valid_schema() {
    // Build a realistic schema and verify Schema::validate() passes
    let columns = vec![
        (
            "ts".to_string(),
            LogicalType::Primitive {
                data_type: DataType::I64,
            },
            LogicalColumn::Primitive(ColumnData::I64(
                (0..1000i64).map(|i| 1_700_000_000 + i * 1000).collect(),
            )),
        ),
        (
            "tags".to_string(),
            LogicalType::List {
                inner: Box::new(LogicalType::Utf8),
            },
            {
                let offsets: Vec<u32> = (0..=1000u32).collect();
                let items: Vec<String> = (0..1000).map(|i| format!("tag{}", i % 10)).collect();
                LogicalColumn::List {
                    offsets,
                    values: Box::new(LogicalColumn::Utf8(items)),
                }
            },
        ),
    ];

    let schema = Optimizer::new().optimize(columns).expect("optimize");
    schema.validate().expect("schema should be valid");
}

// ---------------------------------------------------------------------------
// Semantic types: Decimal128, Date32, Date64, Datetime
// ---------------------------------------------------------------------------

#[test]
fn optimize_decimal128() {
    // Decimal128 → 2 physical leaves: high: I64, low: I64
    let values: Vec<i128> = vec![
        100_0000i128, // 100.0000 at scale 4
        -50_0000i128, // -50.0000
        999_9999i128, // 999.9999
        0i128,
    ];
    let lc = LogicalColumn::Decimal128 {
        values: values.clone(),
    };
    let lt = LogicalType::Decimal128 {
        precision: 38,
        scale: 4,
    };

    let schema = Optimizer::new()
        .optimize(vec![("amount".into(), lt, lc)])
        .expect("optimizer must succeed for Decimal128");
    assert_eq!(schema.columns.len(), 1);
    let spec = &schema.columns[0];
    // Decimal128 → 2 encodings (high leaf + low leaf)
    assert_eq!(
        spec.encodings.len(),
        2,
        "Decimal128 must have 2 encoding vectors"
    );
    assert!(
        !spec.encodings[0].is_empty(),
        "high encoding must be non-empty"
    );
    assert!(
        !spec.encodings[1].is_empty(),
        "low encoding must be non-empty"
    );
}

#[test]
fn optimize_decimal128_round_trip() {
    // End-to-end: optimized schema actually writes + reads back correctly.
    let values: Vec<i128> = (0i128..50)
        .map(|i| i * 123_456_789 - 25 * 123_456_789)
        .collect();
    let lc = LogicalColumn::Decimal128 {
        values: values.clone(),
    };
    let lt = LogicalType::Decimal128 {
        precision: 38,
        scale: 6,
    };

    let schema = Optimizer::new()
        .optimize(vec![("dec".into(), lt, lc.clone())])
        .expect("optimize Decimal128");
    let spec = &schema.columns[0];
    assert_eq!(
        spec.encodings.len(),
        2,
        "Decimal128 must produce 2 encoding vectors"
    );

    let result = roundtrip_spec(spec, lc);
    assert_eq!(
        result,
        LogicalColumn::Decimal128 { values },
        "Decimal128 round-trip mismatch"
    );
}

#[test]
fn optimize_date32() {
    // Date{Days} → 1 physical leaf: values: I32
    let values: Vec<i32> = (0..100).map(|i| i * 7).collect(); // weekly dates
    let lc = LogicalColumn::Date32 {
        values: values.clone(),
    };
    let lt = LogicalType::Date {
        unit: DateUnit::Days,
    };

    let schema = Optimizer::new()
        .optimize(vec![("event_date".into(), lt, lc.clone())])
        .expect("optimizer must succeed for Date32");
    let spec = &schema.columns[0];
    assert_eq!(
        spec.encodings.len(),
        1,
        "Date32 must have 1 encoding vector"
    );
    assert!(
        !spec.encodings[0].is_empty(),
        "date32 encoding must be non-empty"
    );

    let result = roundtrip_spec(spec, lc);
    assert_eq!(
        result,
        LogicalColumn::Date32 { values },
        "Date32 round-trip mismatch"
    );
}

#[test]
fn optimize_date64() {
    // Date{Millis} → 1 physical leaf: values: I64
    let base = 1_700_000_000_000i64; // arbitrary ms since epoch
    let values: Vec<i64> = (0..100).map(|i| base + i * 86_400_000).collect(); // daily steps
    let lc = LogicalColumn::Date64 {
        values: values.clone(),
    };
    let lt = LogicalType::Date {
        unit: DateUnit::Millis,
    };

    let schema = Optimizer::new()
        .optimize(vec![("event_date_ms".into(), lt, lc.clone())])
        .expect("optimizer must succeed for Date64");
    let spec = &schema.columns[0];
    assert_eq!(
        spec.encodings.len(),
        1,
        "Date64 must have 1 encoding vector"
    );
    assert!(
        !spec.encodings[0].is_empty(),
        "date64 encoding must be non-empty"
    );

    let result = roundtrip_spec(spec, lc);
    assert_eq!(
        result,
        LogicalColumn::Date64 { values },
        "Date64 round-trip mismatch"
    );
}

#[test]
fn optimize_datetime_millis() {
    // Datetime{Millis, None} → 1 physical leaf: values: I64
    let base = 1_700_000_000_000i64;
    let values: Vec<i64> = (0..200).map(|i| base + i * 1000).collect();
    let lc = LogicalColumn::Datetime {
        values: values.clone(),
    };
    let lt = LogicalType::Datetime {
        unit: TimeUnit::Millis,
        timezone: None,
    };

    let schema = Optimizer::new()
        .optimize(vec![("ts".into(), lt, lc.clone())])
        .expect("optimizer must succeed for Datetime");
    let spec = &schema.columns[0];
    assert_eq!(
        spec.encodings.len(),
        1,
        "Datetime must have 1 encoding vector"
    );
    assert!(
        !spec.encodings[0].is_empty(),
        "datetime encoding must be non-empty"
    );

    let result = roundtrip_spec(spec, lc);
    assert_eq!(
        result,
        LogicalColumn::Datetime { values },
        "Datetime round-trip mismatch"
    );
}

#[test]
fn optimize_datetime_with_timezone() {
    // Datetime{Micros, Some("UTC")} → 1 encoding; timezone is schema metadata only.
    let base = 1_700_000_000_000_000i64; // microseconds
    let values: Vec<i64> = (0..50).map(|i| base + i * 1_000_000).collect();
    let lc = LogicalColumn::Datetime {
        values: values.clone(),
    };
    let lt = LogicalType::Datetime {
        unit: TimeUnit::Micros,
        timezone: Some("UTC".into()),
    };

    let schema = Optimizer::new()
        .optimize(vec![("ts_utc".into(), lt, lc.clone())])
        .expect("optimizer must succeed for Datetime with timezone");
    let spec = &schema.columns[0];
    assert_eq!(
        spec.encodings.len(),
        1,
        "Datetime must have 1 encoding vector"
    );

    // Verify timezone is preserved in the returned LogicalType
    if let LogicalType::Datetime { unit, timezone } = &spec.logical_type {
        assert_eq!(*unit, TimeUnit::Micros);
        assert_eq!(timezone.as_deref(), Some("UTC"));
    } else {
        panic!("expected Datetime logical type");
    }

    let result = roundtrip_spec(spec, lc);
    assert_eq!(
        result,
        LogicalColumn::Datetime { values },
        "Datetime(UTC) round-trip mismatch"
    );
}

// ---------------------------------------------------------------------------
// Global zstd level (OptimizerConfig.zstd_level)
// ---------------------------------------------------------------------------

/// Collect every `zstd` CoderSpec across a column's (flattened) encodings.
fn zstd_specs(spec: &ColumnSpec) -> Vec<&CoderSpec> {
    spec.encodings
        .iter()
        .flatten()
        .filter(|c| c.id == "zstd")
        .collect()
}

#[test]
fn optimize_zstd_level_default_is_parameter_free() {
    // A Utf8 column's data leaf terminates in zstd.
    let lt = LogicalType::Utf8;
    let lc = LogicalColumn::Utf8((0..500).map(|i| format!("row-{i}")).collect());

    let schema = Optimizer::new()
        .optimize(vec![("s".into(), lt, lc)])
        .unwrap();

    let specs = zstd_specs(&schema.columns[0]);
    assert!(!specs.is_empty(), "expected at least one zstd terminal");
    for c in &specs {
        assert!(
            c.params.get("level").is_none(),
            "default optimizer should leave zstd parameter-free (coder default 3)"
        );
    }
}

#[test]
fn optimize_zstd_level_global_override() {
    let lt = LogicalType::Utf8;
    let lc = LogicalColumn::Utf8((0..500).map(|i| format!("row-{i}")).collect());

    let schema = Optimizer::new()
        .with_zstd_level(19)
        .optimize(vec![("s".into(), lt, lc)])
        .unwrap();

    let specs = zstd_specs(&schema.columns[0]);
    assert!(!specs.is_empty(), "expected at least one zstd terminal");
    for c in &specs {
        assert_eq!(
            c.params.get("level").and_then(|v| v.as_i64()),
            Some(19),
            "global zstd_level must stamp every zstd terminal"
        );
    }
}
