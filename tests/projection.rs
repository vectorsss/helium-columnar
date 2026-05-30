//! Column projection ("slice"): write a new `.he` file containing only a
//! subset of an existing file's columns.
//!
//! Exercises `Schema::project` (schema subset) + `HeliumReader::project_to`
//! (data subset → new file), across single/multi-stripe files, column
//! reordering, a `Dictionary` column, and the error paths.

use std::io::{Cursor, Read, Seek, Write};

use helium::{
    CoderRegistry, CoderSpec, ColumnData, ColumnSpec, DataType, FieldSpec, HeliumError,
    HeliumReader, HeliumWriter, LogicalColumn, LogicalType, Schema,
};

/// The OLD projection path (decode each kept column, re-encode through a fresh
/// writer) — exactly what `project_to` did before it became zero-copy. Used to
/// prove the zero-copy path is equivalent. Public-API only.
fn project_decode_reencode<R: Read + Seek, W: Write + Seek>(
    reader: &mut HeliumReader<R>,
    columns: &[&str],
    dst: W,
    registry: &CoderRegistry,
) -> W {
    let subset = reader.schema().project(columns).unwrap();
    let mut writer = HeliumWriter::new(dst, subset, registry).unwrap();
    let stripe_count = reader.stripe_count();
    for s_idx in 0..stripe_count {
        for &name in columns {
            let col = reader.read_column_at_stripe(name, s_idx).unwrap();
            writer.write_column(name, col).unwrap();
        }
        if s_idx + 1 < stripe_count {
            writer.finish_stripe().unwrap();
        }
    }
    writer.finish().unwrap()
}

fn registry() -> CoderRegistry {
    CoderRegistry::default()
}

fn i64_pipe() -> Vec<CoderSpec> {
    vec![
        CoderSpec::new("delta"),
        CoderSpec::new("leb128"),
        CoderSpec::new("zstd"),
    ]
}

/// A 4-column schema: ts (I64), id (I64), name (Utf8), label (Dictionary<Utf8>).
fn source_schema() -> Schema {
    Schema::new(vec![
        ColumnSpec::primitive("ts", DataType::I64, i64_pipe()),
        ColumnSpec::primitive("id", DataType::I64, i64_pipe()),
        ColumnSpec::utf8(
            "name",
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
            vec![CoderSpec::new("zstd")],
        ),
        ColumnSpec::new(
            "label",
            LogicalType::Dictionary {
                inner: Box::new(LogicalType::Utf8),
            },
            vec![
                vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")], // dict offsets (U32)
                vec![CoderSpec::new("zstd")],                           // dict data (Bytes)
                vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")], // indices (U32)
            ],
        ),
    ])
}

fn ts(n: usize) -> Vec<i64> {
    (0..n).map(|i| 1_700_000_000 + i as i64).collect()
}
fn ids(n: usize) -> Vec<i64> {
    (0..n).map(|i| (i as i64) * 3).collect()
}
fn names(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("row_{}", i % 7)).collect()
}
fn label_col(n: usize) -> LogicalColumn {
    // Low-cardinality dict over {"a","b","c"}.
    let dict = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let indices: Vec<u32> = (0..n).map(|i| (i % 3) as u32).collect();
    LogicalColumn::Dictionary {
        dictionary: Box::new(LogicalColumn::Utf8(dict)),
        indices,
    }
}

/// Write `n` rows of the source schema into one stripe and return the bytes.
fn write_single_stripe(n: usize) -> Vec<u8> {
    let mut buf = Vec::new();
    {
        let mut w = HeliumWriter::new(Cursor::new(&mut buf), source_schema(), &registry()).unwrap();
        w.write_column("ts", LogicalColumn::Primitive(ColumnData::I64(ts(n))))
            .unwrap();
        w.write_column("id", LogicalColumn::Primitive(ColumnData::I64(ids(n))))
            .unwrap();
        w.write_column("name", LogicalColumn::Utf8(names(n)))
            .unwrap();
        w.write_column("label", label_col(n)).unwrap();
        w.finish().unwrap();
    }
    buf
}

/// Write `stripes` stripes of `per` rows each.
fn write_multi_stripe(stripes: usize, per: usize) -> Vec<u8> {
    let mut buf = Vec::new();
    {
        let mut w = HeliumWriter::new(Cursor::new(&mut buf), source_schema(), &registry()).unwrap();
        for s in 0..stripes {
            let base = s * per;
            w.write_column(
                "ts",
                LogicalColumn::Primitive(ColumnData::I64(
                    (0..per)
                        .map(|i| 1_700_000_000 + (base + i) as i64)
                        .collect(),
                )),
            )
            .unwrap();
            w.write_column(
                "id",
                LogicalColumn::Primitive(ColumnData::I64(
                    (0..per).map(|i| (base + i) as i64 * 3).collect(),
                )),
            )
            .unwrap();
            w.write_column(
                "name",
                LogicalColumn::Utf8(
                    (0..per)
                        .map(|i| format!("row_{}", (base + i) % 7))
                        .collect(),
                ),
            )
            .unwrap();
            w.write_column("label", label_col(per)).unwrap();
            if s + 1 < stripes {
                w.finish_stripe().unwrap();
            }
        }
        w.finish().unwrap();
    }
    buf
}

// ---------------------------------------------------------------------------
// Schema::project
// ---------------------------------------------------------------------------

#[test]
fn schema_project_subset_and_order() {
    let schema = source_schema();
    // Reorder + subset: name, ts (drop id, label).
    let projected = schema.project(&["name", "ts"]).unwrap();
    let cols: Vec<&str> = projected.columns.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(
        cols,
        vec!["name", "ts"],
        "projection preserves requested order"
    );
    // Encodings preserved from the source spec.
    assert_eq!(
        projected.column("ts").unwrap().encodings,
        schema.column("ts").unwrap().encodings
    );
}

#[test]
fn schema_project_errors() {
    let schema = source_schema();
    assert!(matches!(
        schema.project(&["ts", "nope"]),
        Err(HeliumError::Schema { .. })
    ));
    assert!(matches!(
        schema.project(&["ts", "ts"]),
        Err(HeliumError::Schema { .. })
    ));
}

// ---------------------------------------------------------------------------
// project_to — single stripe
// ---------------------------------------------------------------------------

#[test]
fn project_single_stripe_subset() {
    let n = 300;
    let src = write_single_stripe(n);

    let mut reader = HeliumReader::new(Cursor::new(src), &registry()).unwrap();
    let mut out = Vec::new();
    reader
        .project_to(&["id", "name"], Cursor::new(&mut out), &registry())
        .unwrap();

    // New file has exactly the two projected columns, with original data.
    let mut r2 = HeliumReader::new(Cursor::new(out), &registry()).unwrap();
    let cols: Vec<String> = r2.column_names().map(|s| s.to_string()).collect();
    assert_eq!(cols, vec!["id".to_string(), "name".to_string()]);
    assert_eq!(r2.row_count(), n as u64);
    assert_eq!(
        r2.read_column("id").unwrap(),
        LogicalColumn::Primitive(ColumnData::I64(ids(n)))
    );
    assert_eq!(
        r2.read_column("name").unwrap(),
        LogicalColumn::Utf8(names(n))
    );
    // Dropped columns are absent.
    assert!(r2.read_column("ts").is_err());
}

#[test]
fn project_single_stripe_includes_dictionary() {
    let n = 120;
    let src = write_single_stripe(n);
    let mut reader = HeliumReader::new(Cursor::new(src), &registry()).unwrap();
    let mut out = Vec::new();
    reader
        .project_to(&["label", "ts"], Cursor::new(&mut out), &registry())
        .unwrap();

    let mut r2 = HeliumReader::new(Cursor::new(out), &registry()).unwrap();
    assert_eq!(r2.read_column("label").unwrap(), label_col(n));
    assert_eq!(
        r2.read_column("ts").unwrap(),
        LogicalColumn::Primitive(ColumnData::I64(ts(n)))
    );
}

// ---------------------------------------------------------------------------
// project_to — multi stripe
// ---------------------------------------------------------------------------

#[test]
fn project_multi_stripe_preserves_stripes_and_data() {
    let (stripes, per) = (4, 50);
    let src = write_multi_stripe(stripes, per);
    let mut reader = HeliumReader::new(Cursor::new(src), &registry()).unwrap();
    assert_eq!(reader.stripe_count(), stripes);

    let mut out = Vec::new();
    reader
        .project_to(&["id", "label"], Cursor::new(&mut out), &registry())
        .unwrap();

    let mut r2 = HeliumReader::new(Cursor::new(out), &registry()).unwrap();
    // Stripe structure preserved.
    assert_eq!(r2.stripe_count(), stripes);
    assert_eq!(r2.row_count(), (stripes * per) as u64);
    let cols: Vec<String> = r2.column_names().map(|s| s.to_string()).collect();
    assert_eq!(cols, vec!["id".to_string(), "label".to_string()]);

    // Data matches per stripe (id is global index * 3; label is the per-stripe dict).
    for s in 0..stripes {
        let base = s * per;
        let got_id = r2.read_column_at_stripe("id", s).unwrap();
        assert_eq!(
            got_id,
            LogicalColumn::Primitive(ColumnData::I64(
                (0..per).map(|i| (base + i) as i64 * 3).collect()
            ))
        );
        // Dictionary column: must be read per stripe.
        assert_eq!(
            r2.read_column_at_stripe("label", s).unwrap(),
            label_col(per)
        );
    }
}

// ---------------------------------------------------------------------------
// project_to — error paths
// ---------------------------------------------------------------------------

#[test]
fn project_to_errors() {
    let src = write_single_stripe(10);
    let mut reader = HeliumReader::new(Cursor::new(src), &registry()).unwrap();

    // Missing column.
    let mut o1 = Vec::new();
    assert!(
        reader
            .project_to(&["nope"], Cursor::new(&mut o1), &registry())
            .is_err()
    );

    // Empty projection.
    let mut o2 = Vec::new();
    assert!(
        reader
            .project_to(&[], Cursor::new(&mut o2), &registry())
            .is_err()
    );

    // Duplicate column.
    let mut o3 = Vec::new();
    assert!(
        reader
            .project_to(&["ts", "ts"], Cursor::new(&mut o3), &registry())
            .is_err()
    );
}

// ---------------------------------------------------------------------------
// Zero-copy property: per-leaf stats are carried over verbatim from the source
// footer (not recomputed). min/max/null_count must match the source exactly.
// ---------------------------------------------------------------------------

#[test]
fn project_preserves_source_stats() {
    let n = 256;
    let src = write_single_stripe(n);
    let mut reader = HeliumReader::new(Cursor::new(src), &registry()).unwrap();

    // Source stats for "id" (a primitive leaf with min/max).
    let src_stats = reader.stripe_column_stats(0, "id").unwrap();

    let mut out = Vec::new();
    reader
        .project_to(&["id", "ts"], Cursor::new(&mut out), &registry())
        .unwrap();

    let r2 = HeliumReader::new(Cursor::new(out), &registry()).unwrap();
    let out_stats = r2.stripe_column_stats(0, "id").unwrap();
    assert_eq!(out_stats.len(), src_stats.len());
    assert_eq!(out_stats[0].min, src_stats[0].min, "min carried over");
    assert_eq!(out_stats[0].max, src_stats[0].max, "max carried over");
    assert_eq!(
        out_stats[0].null_count, src_stats[0].null_count,
        "null_count carried over"
    );
}

// ---------------------------------------------------------------------------
// Cross-version: slicing a v6 (catalog-mode) source produces a self-contained
// v5 output (raw byte copy is version-independent).
// ---------------------------------------------------------------------------

#[test]
fn project_from_v6_catalog_source() {
    use helium::catalog::Catalog;

    let n = 80;
    let s = label_col(n); // a Dictionary column
    let dir = tempfile::tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).unwrap();

    // Write a v6 catalog-mode source with two columns.
    let schema = Schema::new(vec![
        ColumnSpec::primitive("ts", DataType::I64, i64_pipe()),
        ColumnSpec::new(
            "label",
            LogicalType::Dictionary {
                inner: Box::new(LogicalType::Utf8),
            },
            vec![
                vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
                vec![CoderSpec::new("zstd")],
                vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
            ],
        ),
    ]);
    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut w = catalog.open_writer(&mut buf, schema, &registry()).unwrap();
    w.write_column("ts", LogicalColumn::Primitive(ColumnData::I64(ts(n))))
        .unwrap();
    w.write_column("label", s.clone()).unwrap();
    w.finish().unwrap();

    // Open via resolver and slice "label" → a self-contained v5 file.
    let bytes = buf.into_inner();
    let mut reader =
        HeliumReader::new_with_resolver(Cursor::new(bytes), &registry(), catalog.resolver())
            .unwrap();
    let mut out = Vec::new();
    reader
        .project_to(&["label"], Cursor::new(&mut out), &registry())
        .unwrap();

    // Output is plain v5 — readable WITHOUT a resolver.
    let mut r2 = HeliumReader::new(Cursor::new(out), &registry()).unwrap();
    assert_eq!(r2.version_str(), "v5");
    assert_eq!(r2.read_column("label").unwrap(), s);
}

// ---------------------------------------------------------------------------
// Equivalence: zero-copy `project_to` must produce the SAME result as the old
// decode→re-encode path, over a multi-type, multi-stripe file with reordering.
// ---------------------------------------------------------------------------

#[test]
fn zero_copy_equivalent_to_decode_reencode() {
    let (stripes, per) = (3, 64);
    let src = write_multi_stripe(stripes, per);
    // Subset includes a Dictionary column ("label"), a Utf8 ("name"), a
    // primitive ("id"), and is reordered relative to the source schema.
    let kept = ["label", "id", "name"];

    // (a) zero-copy
    let mut r_zc = HeliumReader::new(Cursor::new(src.clone()), &registry()).unwrap();
    let mut out_zc = Vec::new();
    r_zc.project_to(&kept, Cursor::new(&mut out_zc), &registry())
        .unwrap();

    // (b) decode → re-encode
    let mut r_re = HeliumReader::new(Cursor::new(src.clone()), &registry()).unwrap();
    let mut out_re = Vec::new();
    project_decode_reencode(&mut r_re, &kept, Cursor::new(&mut out_re), &registry());

    // The two outputs must read back identically (per stripe, per column —
    // read_column_at_stripe handles the Dictionary column too), and both must
    // match the source.
    let mut a = HeliumReader::new(Cursor::new(out_zc.clone()), &registry()).unwrap();
    let mut b = HeliumReader::new(Cursor::new(out_re.clone()), &registry()).unwrap();
    let mut s = HeliumReader::new(Cursor::new(src), &registry()).unwrap();
    assert_eq!(a.stripe_count(), b.stripe_count());
    assert_eq!(a.row_count(), b.row_count());
    for si in 0..a.stripe_count() {
        for &col in &kept {
            let from_zc = a.read_column_at_stripe(col, si).unwrap();
            let from_re = b.read_column_at_stripe(col, si).unwrap();
            let from_src = s.read_column_at_stripe(col, si).unwrap();
            assert_eq!(
                from_zc, from_re,
                "stripe {si} col '{col}': zero-copy != decode-reencode"
            );
            assert_eq!(
                from_zc, from_src,
                "stripe {si} col '{col}': zero-copy != source"
            );
        }
    }

    // Bonus: deterministic coders mean the two methods produce byte-identical
    // files (verbatim copy == re-encode of the same data with the same coders).
    assert_eq!(
        out_zc, out_re,
        "zero-copy and decode-reencode should be byte-identical for deterministic coders"
    );
}

// ---------------------------------------------------------------------------
// Nested-type equivalence: zero-copy vs decode→re-encode over a file with
// List / Struct / Map / Nullable columns (incl. a Struct nesting a List and a
// Nullable), multi-stripe, sliced with reordering.
// ---------------------------------------------------------------------------

fn present_pipe() -> Vec<CoderSpec> {
    vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")]
}
fn zstd_only() -> Vec<CoderSpec> {
    vec![CoderSpec::new("zstd")]
}

/// A schema mixing every nested kind. `combo` is a Struct that itself nests a
/// List and a Nullable.
fn nested_schema() -> Schema {
    Schema::new(vec![
        ColumnSpec::primitive("id", DataType::I64, i64_pipe()),
        // List<I32>: [offsets, values]
        ColumnSpec::list(
            "tags",
            LogicalType::Primitive {
                data_type: DataType::I32,
            },
            vec![i64_pipe(), i64_pipe()],
        ),
        // Struct { a: I64, b: Utf8 } — fields carry their own encodings.
        ColumnSpec::struct_col(
            "meta",
            vec![
                FieldSpec::primitive("a", DataType::I64, i64_pipe()),
                FieldSpec::utf8("b", i64_pipe(), zstd_only()),
            ],
        ),
        // Map<Utf8, I32>: [offsets, key.offsets, key.data, value]
        ColumnSpec::map(
            "attrs",
            LogicalType::Utf8,
            LogicalType::Primitive {
                data_type: DataType::I32,
            },
            vec![i64_pipe(), i64_pipe(), zstd_only(), i64_pipe()],
        ),
        // Nullable<I64>: [present, values]
        ColumnSpec::nullable(
            "opt",
            LogicalType::Primitive {
                data_type: DataType::I64,
            },
            vec![present_pipe(), i64_pipe()],
        ),
        // Struct { lst: List<I64>, fl: Nullable<I64> } — deep nesting.
        ColumnSpec::struct_col(
            "combo",
            vec![
                FieldSpec::list(
                    "lst",
                    LogicalType::Primitive {
                        data_type: DataType::I64,
                    },
                    vec![i64_pipe(), i64_pipe()],
                ),
                FieldSpec::nullable(
                    "fl",
                    LogicalType::Primitive {
                        data_type: DataType::I64,
                    },
                    vec![present_pipe(), i64_pipe()],
                ),
            ],
        ),
    ])
}

fn nested_columns(per: usize) -> Vec<(&'static str, LogicalColumn)> {
    let p = per as u32;
    // tags: 2 items/row
    let tags_offsets: Vec<u32> = (0..=p).map(|i| i * 2).collect();
    let tags_values: Vec<i32> = (0..(per * 2)).map(|x| x as i32).collect();
    // meta
    let meta_a: Vec<i64> = (0..per as i64).collect();
    let meta_b: Vec<String> = (0..per).map(|i| format!("m{}", i % 5)).collect();
    // attrs: 1 entry/row
    let attr_offsets: Vec<u32> = (0..=p).collect();
    let attr_keys: Vec<String> = (0..per).map(|i| format!("k{}", i % 4)).collect();
    let attr_vals: Vec<i32> = (0..per as i32).collect();
    // opt: even rows present
    let opt_present: Vec<bool> = (0..per).map(|i| i % 2 == 0).collect();
    let opt_vals: Vec<i64> = (0..per).filter(|i| i % 2 == 0).map(|i| i as i64).collect();
    // combo.lst: 1 item/row
    let lst_offsets: Vec<u32> = (0..=p).collect();
    let lst_values: Vec<i64> = (0..per as i64).map(|x| x * 10).collect();
    // combo.fl: every 3rd row present
    let fl_present: Vec<bool> = (0..per).map(|i| i % 3 == 0).collect();
    let fl_vals: Vec<i64> = (0..per)
        .filter(|i| i % 3 == 0)
        .map(|i| (i * 100) as i64)
        .collect();

    vec![
        (
            "id",
            LogicalColumn::Primitive(ColumnData::I64((0..per as i64).collect())),
        ),
        (
            "tags",
            LogicalColumn::List {
                offsets: tags_offsets,
                values: Box::new(LogicalColumn::Primitive(ColumnData::I32(tags_values))),
            },
        ),
        (
            "meta",
            LogicalColumn::Struct {
                fields: vec![
                    (
                        "a".into(),
                        LogicalColumn::Primitive(ColumnData::I64(meta_a)),
                    ),
                    ("b".into(), LogicalColumn::Utf8(meta_b)),
                ],
            },
        ),
        (
            "attrs",
            LogicalColumn::Map {
                offsets: attr_offsets,
                keys: Box::new(LogicalColumn::Utf8(attr_keys)),
                values: Box::new(LogicalColumn::Primitive(ColumnData::I32(attr_vals))),
            },
        ),
        (
            "opt",
            LogicalColumn::Nullable {
                present: opt_present,
                value: Box::new(LogicalColumn::Primitive(ColumnData::I64(opt_vals))),
            },
        ),
        (
            "combo",
            LogicalColumn::Struct {
                fields: vec![
                    (
                        "lst".into(),
                        LogicalColumn::List {
                            offsets: lst_offsets,
                            values: Box::new(LogicalColumn::Primitive(ColumnData::I64(lst_values))),
                        },
                    ),
                    (
                        "fl".into(),
                        LogicalColumn::Nullable {
                            present: fl_present,
                            value: Box::new(LogicalColumn::Primitive(ColumnData::I64(fl_vals))),
                        },
                    ),
                ],
            },
        ),
    ]
}

fn write_nested(stripes: usize, per: usize) -> Vec<u8> {
    let mut buf = Vec::new();
    {
        let mut w = HeliumWriter::new(Cursor::new(&mut buf), nested_schema(), &registry()).unwrap();
        for s in 0..stripes {
            for (name, col) in nested_columns(per) {
                w.write_column(name, col).unwrap();
            }
            if s + 1 < stripes {
                w.finish_stripe().unwrap();
            }
        }
        w.finish().unwrap();
    }
    buf
}

#[test]
fn zero_copy_equivalent_to_decode_reencode_nested_types() {
    let (stripes, per) = (3, 48);
    let src = write_nested(stripes, per);
    // Subset: reordered, mixes Map / deep Struct / List / Nullable / primitive.
    let kept = ["attrs", "combo", "tags", "opt", "id"];

    let mut r_zc = HeliumReader::new(Cursor::new(src.clone()), &registry()).unwrap();
    let mut out_zc = Vec::new();
    r_zc.project_to(&kept, Cursor::new(&mut out_zc), &registry())
        .unwrap();

    let mut r_re = HeliumReader::new(Cursor::new(src.clone()), &registry()).unwrap();
    let mut out_re = Vec::new();
    project_decode_reencode(&mut r_re, &kept, Cursor::new(&mut out_re), &registry());

    let mut a = HeliumReader::new(Cursor::new(out_zc.clone()), &registry()).unwrap();
    let mut b = HeliumReader::new(Cursor::new(out_re.clone()), &registry()).unwrap();
    let mut s = HeliumReader::new(Cursor::new(src), &registry()).unwrap();
    assert_eq!(a.stripe_count(), stripes);
    assert_eq!(a.stripe_count(), b.stripe_count());
    for si in 0..a.stripe_count() {
        for &col in &kept {
            let zc = a.read_column_at_stripe(col, si).unwrap();
            let re = b.read_column_at_stripe(col, si).unwrap();
            let sr = s.read_column_at_stripe(col, si).unwrap();
            assert_eq!(
                zc, re,
                "stripe {si} col '{col}': zero-copy != decode-reencode"
            );
            assert_eq!(zc, sr, "stripe {si} col '{col}': zero-copy != source");
        }
    }
    assert_eq!(
        out_zc, out_re,
        "nested: zero-copy and decode-reencode should be byte-identical"
    );
}
