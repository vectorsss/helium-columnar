//! Integration tests for the Arrow bridge (`read_record_batch`, schema
//! conversion, and round-trip through Helium `.he` files).
//!
//! Gated on the `arrow` feature — skipped entirely when the feature is absent.

#![cfg(feature = "arrow")]

use std::io::Cursor;

use arrow::array::{Array, Int32Array, Int64Array, ListArray, StringArray, StructArray};

use helium::arrow::{schema_from_arrow, schema_to_arrow};
use helium::core::coder::DataType as HDataType;
use helium::{
    CoderRegistry, ColumnData, ColumnSpec, HeliumReader, HeliumWriter, LogicalColumn, LogicalType,
    Schema,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_registry() -> CoderRegistry {
    CoderRegistry::default()
}

fn write_read_back(schema: Schema, columns: Vec<(&str, LogicalColumn)>) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    let registry = make_registry();
    {
        let cursor = Cursor::new(&mut buf);
        let mut writer = HeliumWriter::new(cursor, schema, &registry).unwrap();
        for (name, col) in columns {
            writer.write_column(name, col).unwrap();
        }
        writer.finish().unwrap();
    }
    buf
}

// ---------------------------------------------------------------------------
// Test 1: read_record_batch basic — primitive + utf8 + nullable
// ---------------------------------------------------------------------------

#[test]
fn record_batch_basic_columns() {
    let schema = Schema::new(vec![
        ColumnSpec::primitive(
            "id",
            HDataType::I32,
            vec![
                helium::CoderSpec::new("delta"),
                helium::CoderSpec::new("leb128"),
                helium::CoderSpec::new("zstd"),
            ],
        ),
        ColumnSpec::utf8(
            "name",
            vec![
                helium::CoderSpec::new("delta"),
                helium::CoderSpec::new("leb128"),
                helium::CoderSpec::new("zstd"),
            ],
            vec![helium::CoderSpec::new("zstd")],
        ),
        ColumnSpec::nullable(
            "score",
            LogicalType::Primitive {
                data_type: HDataType::I64,
            },
            vec![
                vec![
                    helium::CoderSpec::new("leb128"),
                    helium::CoderSpec::new("zstd"),
                ],
                vec![
                    helium::CoderSpec::new("delta"),
                    helium::CoderSpec::new("leb128"),
                    helium::CoderSpec::new("zstd"),
                ],
            ],
        ),
    ]);

    let id_col = LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3]));
    let name_col = LogicalColumn::Utf8(vec!["Alice".into(), "Bob".into(), "Carol".into()]);
    let score_col = LogicalColumn::Nullable {
        present: vec![true, false, true],
        value: Box::new(LogicalColumn::Primitive(ColumnData::I64(vec![100, 300]))),
    };

    let buf = write_read_back(
        schema,
        vec![("id", id_col), ("name", name_col), ("score", score_col)],
    );

    let mut reader = HeliumReader::new(Cursor::new(&buf), &make_registry()).unwrap();
    assert_eq!(reader.stripe_count(), 1);

    let batch = reader.read_record_batch(0).unwrap();

    // Column count
    assert_eq!(batch.num_columns(), 3, "expected 3 columns in record batch");

    // Row count
    assert_eq!(batch.num_rows(), 3, "expected 3 rows");

    // Column 0: id (Int32, no nulls)
    let id_arr = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(id_arr.value(0), 1);
    assert_eq!(id_arr.value(1), 2);
    assert_eq!(id_arr.value(2), 3);
    assert_eq!(id_arr.null_count(), 0);

    // Column 1: name (Utf8)
    let name_arr = batch
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(name_arr.value(0), "Alice");
    assert_eq!(name_arr.value(1), "Bob");
    assert_eq!(name_arr.value(2), "Carol");

    // Column 2: score (Nullable<Int64>)
    let score_arr = batch
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(score_arr.null_count(), 1);
    assert_eq!(score_arr.value(0), 100);
    assert!(score_arr.is_null(1));
    assert_eq!(score_arr.value(2), 300);
}

// ---------------------------------------------------------------------------
// Test 2: multi-stripe — read_record_batch on each stripe independently
// ---------------------------------------------------------------------------

#[test]
fn record_batch_multi_stripe() {
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "val",
        HDataType::I32,
        vec![
            helium::CoderSpec::new("delta"),
            helium::CoderSpec::new("leb128"),
            helium::CoderSpec::new("zstd"),
        ],
    )]);

    let mut buf: Vec<u8> = Vec::new();
    {
        let cursor = Cursor::new(&mut buf);
        let mut writer = HeliumWriter::new(cursor, schema, &make_registry()).unwrap();
        writer
            .write_column(
                "val",
                LogicalColumn::Primitive(ColumnData::I32(vec![1, 2, 3])),
            )
            .unwrap();
        writer.finish_stripe().unwrap();
        writer
            .write_column("val", LogicalColumn::Primitive(ColumnData::I32(vec![4, 5])))
            .unwrap();
        writer.finish().unwrap();
    }

    let mut reader = HeliumReader::new(Cursor::new(&buf), &make_registry()).unwrap();
    assert_eq!(reader.stripe_count(), 2);

    let batch0 = reader.read_record_batch(0).unwrap();
    assert_eq!(batch0.num_rows(), 3);
    let arr0 = batch0
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(arr0.value(2), 3);

    let batch1 = reader.read_record_batch(1).unwrap();
    assert_eq!(batch1.num_rows(), 2);
    let arr1 = batch1
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(arr1.value(0), 4);
    assert_eq!(arr1.value(1), 5);
}

// ---------------------------------------------------------------------------
// Test 3: struct + list columns in a record batch
// ---------------------------------------------------------------------------

#[test]
fn record_batch_struct_and_list() {
    use helium::core::schema::FieldSpec;

    let schema = Schema::new(vec![
        ColumnSpec::struct_col(
            "person",
            vec![
                FieldSpec::primitive(
                    "age",
                    HDataType::I32,
                    vec![
                        helium::CoderSpec::new("delta"),
                        helium::CoderSpec::new("leb128"),
                        helium::CoderSpec::new("zstd"),
                    ],
                ),
                FieldSpec::utf8(
                    "city",
                    vec![
                        helium::CoderSpec::new("delta"),
                        helium::CoderSpec::new("leb128"),
                        helium::CoderSpec::new("zstd"),
                    ],
                    vec![helium::CoderSpec::new("zstd")],
                ),
            ],
        ),
        ColumnSpec::list(
            "tags",
            LogicalType::Utf8,
            vec![
                vec![
                    helium::CoderSpec::new("delta"),
                    helium::CoderSpec::new("leb128"),
                    helium::CoderSpec::new("zstd"),
                ],
                vec![
                    helium::CoderSpec::new("delta"),
                    helium::CoderSpec::new("leb128"),
                    helium::CoderSpec::new("zstd"),
                ],
                vec![helium::CoderSpec::new("zstd")],
            ],
        ),
    ]);

    let person_col = LogicalColumn::Struct {
        fields: vec![
            (
                "age".into(),
                LogicalColumn::Primitive(ColumnData::I32(vec![25, 30])),
            ),
            (
                "city".into(),
                LogicalColumn::Utf8(vec!["NYC".into(), "LA".into()]),
            ),
        ],
    };
    let tags_col = LogicalColumn::List {
        offsets: vec![0, 2, 2],
        values: Box::new(LogicalColumn::Utf8(vec!["rust".into(), "arrow".into()])),
    };

    let buf = write_read_back(schema, vec![("person", person_col), ("tags", tags_col)]);

    let mut reader = HeliumReader::new(Cursor::new(&buf), &make_registry()).unwrap();
    let batch = reader.read_record_batch(0).unwrap();

    assert_eq!(batch.num_columns(), 2);
    assert_eq!(batch.num_rows(), 2);

    // struct column
    let struct_arr = batch
        .column(0)
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    let age_arr = struct_arr
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(age_arr.value(0), 25);
    assert_eq!(age_arr.value(1), 30);

    // list column
    let list_arr = batch
        .column(1)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    assert_eq!(list_arr.len(), 2);
    assert_eq!(list_arr.value_length(0), 2); // row 0 has ["rust", "arrow"]
    assert_eq!(list_arr.value_length(1), 0); // row 1 is empty
}

// ---------------------------------------------------------------------------
// Test 4: schema_to_arrow / schema_from_arrow round-trip
// ---------------------------------------------------------------------------

#[test]
fn schema_roundtrip_via_arrow() {
    let schema = Schema::new(vec![
        ColumnSpec::primitive(
            "a",
            HDataType::I32,
            vec![
                helium::CoderSpec::new("delta"),
                helium::CoderSpec::new("leb128"),
                helium::CoderSpec::new("zstd"),
            ],
        ),
        ColumnSpec::utf8(
            "b",
            vec![
                helium::CoderSpec::new("delta"),
                helium::CoderSpec::new("leb128"),
                helium::CoderSpec::new("zstd"),
            ],
            vec![helium::CoderSpec::new("zstd")],
        ),
        ColumnSpec::nullable(
            "c",
            LogicalType::Primitive {
                data_type: HDataType::F64,
            },
            vec![
                vec![
                    helium::CoderSpec::new("leb128"),
                    helium::CoderSpec::new("zstd"),
                ],
                vec![
                    helium::CoderSpec::new("gorilla"),
                    helium::CoderSpec::new("zstd"),
                ],
            ],
        ),
    ]);

    let arrow_schema = schema_to_arrow(&schema);
    let back = schema_from_arrow(&arrow_schema).unwrap();

    // Compare column count and logical types (not encodings — those are rebuilt from defaults)
    assert_eq!(schema.columns.len(), back.columns.len());
    for (orig, rebuilt) in schema.columns.iter().zip(back.columns.iter()) {
        assert_eq!(orig.name, rebuilt.name);
        assert_eq!(
            orig.logical_type, rebuilt.logical_type,
            "logical type mismatch for column '{}'",
            orig.name
        );
    }
}

// ---------------------------------------------------------------------------
// Test 5: stripe index out of range returns an error
// ---------------------------------------------------------------------------

#[test]
fn record_batch_stripe_out_of_range() {
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "x",
        HDataType::I32,
        vec![
            helium::CoderSpec::new("delta"),
            helium::CoderSpec::new("leb128"),
            helium::CoderSpec::new("zstd"),
        ],
    )]);
    let buf = write_read_back(
        schema,
        vec![("x", LogicalColumn::Primitive(ColumnData::I32(vec![1])))],
    );

    let mut reader = HeliumReader::new(Cursor::new(&buf), &make_registry()).unwrap();
    let err = reader.read_record_batch(99);
    assert!(err.is_err(), "expected error for out-of-range stripe index");
}
