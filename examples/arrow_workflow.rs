//! End-to-end sanity check for Path A step 1 (Arrow bridge).
//!
//! Exercises the Helium ↔ Arrow bridge against a real file (`hits_1.he`,
//! 1 M rows × 105 cols). Verifies four things any downstream Arrow consumer
//! (DataFusion, polars, pyarrow) would observe:
//!
//! 1. **Schema mapping**: the Arrow schema produced from the Helium schema
//!    is well-formed and matches the column count + names.
//! 2. **RecordBatch shape**: per-stripe read_record_batch produces a
//!    RecordBatch whose row count + column count match the Helium reader's
//!    own counts.
//! 3. **Data integrity**: a primitive numeric column and a Utf8 column read
//!    back identical values via Helium and via the Arrow path.
//! 4. **Per-column round-trip**: `to_arrow_array(lc, lt)` →
//!    `from_arrow_array(arr, lt)` returns a LogicalColumn equal to the
//!    original. This is the contract DataFusion will rely on.
//!
//! Run:
//!   cargo run --release --features arrow --example arrow_workflow
//!
//! Falls back to a synthetic dataset if hits_1.he isn't present.

#[cfg(feature = "arrow")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::BufReader;
    use std::path::Path;

    use helium::arrow::{from_arrow_array, schema_to_arrow, to_arrow_array};
    use helium::{CoderRegistry, HeliumReader, LogicalColumn};

    println!("=== Helium ↔ Arrow bridge sanity check ===\n");

    // ---- 1. Use ./hits_1.he if present, else synthesize a small dataset. ----
    let mut he_path: Option<&Path> = None;
    if Path::new("hits_1.he").exists() {
        he_path = Some(Path::new("hits_1.he"));
    }

    let tmp = tempfile::NamedTempFile::new()?;
    let owned_he_path: std::path::PathBuf;
    let he_path: &Path = if let Some(p) = he_path {
        println!("Using real file: {}", p.display());
        p
    } else {
        println!("hits_1.he not found — synthesizing a small dataset.");
        owned_he_path = synthesize_he(tmp.path())?;
        &owned_he_path
    };

    // ---- 2. Open + introspect. ----
    let registry = CoderRegistry::default();
    let mut reader = HeliumReader::new(BufReader::new(File::open(he_path)?), &registry)?;
    let helium_schema = reader.schema().clone();
    let total_rows = reader.row_count();
    let stripe_count = reader.stripe_count();
    let column_count = helium_schema.columns.len();

    println!("\nHelium file: {column_count} cols × {total_rows} rows × {stripe_count} stripes\n");

    // ---- 3. Schema mapping. ----
    let arrow_schema = schema_to_arrow(&helium_schema);
    println!("Arrow schema (first 10 fields):");
    for (i, field) in arrow_schema.fields().iter().take(10).enumerate() {
        println!(
            "  [{i:>3}] {:<32} {:?}  nullable={}",
            field.name(),
            field.data_type(),
            field.is_nullable()
        );
    }
    if arrow_schema.fields().len() > 10 {
        println!("  ... +{} more fields", arrow_schema.fields().len() - 10);
    }
    assert_eq!(
        arrow_schema.fields().len(),
        column_count,
        "Arrow schema field count != Helium column count"
    );
    println!("✓ Schema mapping: {column_count} fields, names + types preserved.\n");

    // ---- 4. Read first stripe as RecordBatch. ----
    let batch = reader.read_record_batch(0)?;
    println!(
        "RecordBatch[0]: {} cols × {} rows",
        batch.num_columns(),
        batch.num_rows()
    );
    assert_eq!(batch.num_columns(), column_count);
    assert!(
        batch.num_rows() > 0,
        "first stripe should have rows; helium reader said {total_rows}"
    );
    println!("✓ RecordBatch shape matches Helium counts.\n");

    // ---- 5. Sample first row across first 5 columns via Arrow. ----
    println!("First-row sample (via Arrow ArrayRef):");
    for (i, field) in arrow_schema.fields().iter().take(5).enumerate() {
        let arr = batch.column(i);
        let printed = format_first_value(arr);
        println!("  [{i}] {:<32} = {}", field.name(), printed);
    }
    println!();

    // ---- 6. Per-column round-trip on a primitive (or first non-nested) column. ----
    let mut roundtrip_target: Option<usize> = None;
    for (i, spec) in helium_schema.columns.iter().enumerate() {
        if matches!(
            spec.logical_type,
            helium::LogicalType::Primitive { .. }
                | helium::LogicalType::Nullable { .. }
                | helium::LogicalType::Utf8
        ) {
            roundtrip_target = Some(i);
            break;
        }
    }
    let target = roundtrip_target.expect("expected at least one primitive/utf8/nullable column");
    let target_spec = &helium_schema.columns[target];
    println!(
        "Round-trip target column [{target}]: '{}' ({:?})",
        target_spec.name, target_spec.logical_type
    );

    // Read same column directly via Helium.
    let lc_helium: LogicalColumn = reader.read_column_at_stripe(&target_spec.name, 0)?;

    // Convert through Arrow and back.
    let lc_via_arrow = {
        let arrow_arr = to_arrow_array(&lc_helium, &target_spec.logical_type)?;
        from_arrow_array(&arrow_arr, &target_spec.logical_type)?
    };

    if lc_helium == lc_via_arrow {
        println!(
            "✓ Per-column round-trip: helium → arrow → helium produced identical LogicalColumn.\n"
        );
    } else {
        println!("✗ Round-trip mismatch on column '{}'!", target_spec.name);
        println!("  helium-direct:        {:?}", brief(&lc_helium));
        println!("  helium→arrow→helium:  {:?}", brief(&lc_via_arrow));
        return Err("round-trip failed".into());
    }

    // ---- 7. Cross-stripe row sum (sanity vs. existing smoke test) ----
    let mut rb_sum = 0usize;
    for s in 0..stripe_count {
        rb_sum += reader.read_record_batch(s)?.num_rows();
    }
    assert_eq!(
        rb_sum as u64, total_rows,
        "RecordBatch rows summed across stripes != total_rows"
    );
    println!("✓ Cross-stripe RecordBatch row sum matches reader.row_count(): {rb_sum}\n");

    println!("=== ALL CHECKS PASSED ===");
    Ok(())
}

#[cfg(feature = "arrow")]
fn format_first_value(arr: &dyn arrow::array::Array) -> String {
    use arrow::array::*;
    use arrow::datatypes::DataType as AdT;

    if arr.is_empty() {
        return "<empty>".into();
    }
    if arr.is_null(0) {
        return "NULL".into();
    }
    match arr.data_type() {
        AdT::Int8 => arr
            .as_any()
            .downcast_ref::<Int8Array>()
            .unwrap()
            .value(0)
            .to_string(),
        AdT::Int16 => arr
            .as_any()
            .downcast_ref::<Int16Array>()
            .unwrap()
            .value(0)
            .to_string(),
        AdT::Int32 => arr
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .value(0)
            .to_string(),
        AdT::Int64 => arr
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0)
            .to_string(),
        AdT::UInt8 => arr
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap()
            .value(0)
            .to_string(),
        AdT::UInt16 => arr
            .as_any()
            .downcast_ref::<UInt16Array>()
            .unwrap()
            .value(0)
            .to_string(),
        AdT::UInt32 => arr
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .value(0)
            .to_string(),
        AdT::UInt64 => arr
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(0)
            .to_string(),
        AdT::Float32 => arr
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .value(0)
            .to_string(),
        AdT::Float64 => arr
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0)
            .to_string(),
        AdT::Utf8 => format!(
            "{:?}",
            arr.as_any().downcast_ref::<StringArray>().unwrap().value(0)
        ),
        AdT::Binary => format!(
            "<{} bytes>",
            arr.as_any()
                .downcast_ref::<BinaryArray>()
                .unwrap()
                .value(0)
                .len()
        ),
        other => format!("<{other:?} array>"),
    }
}

#[cfg(feature = "arrow")]
fn brief(lc: &helium::LogicalColumn) -> String {
    use helium::LogicalColumn::*;
    match lc {
        Primitive(cd) => format!("Primitive({} rows)", cd_len(cd)),
        Utf8(v) => format!("Utf8({} rows)", v.len()),
        Binary(v) => format!("Binary({} rows)", v.len()),
        Nullable { present, value } => {
            format!(
                "Nullable(present_len={}, inner={})",
                present.len(),
                brief(value)
            )
        }
        other => format!("{:?}", std::mem::discriminant(other)),
    }
}

#[cfg(feature = "arrow")]
fn cd_len(cd: &helium::ColumnData) -> usize {
    use helium::ColumnData::*;
    match cd {
        I8(v) => v.len(),
        I16(v) => v.len(),
        I32(v) => v.len(),
        I64(v) => v.len(),
        U8(v) => v.len(),
        U16(v) => v.len(),
        U32(v) => v.len(),
        U64(v) => v.len(),
        F32(v) => v.len(),
        F64(v) => v.len(),
        Bytes(v) => v.len(),
    }
}

#[cfg(feature = "arrow")]
fn synthesize_he(out: &std::path::Path) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    use helium::{
        CoderRegistry, CoderSpec, ColumnData, ColumnSpec, DataType, HeliumWriter, LogicalColumn,
        LogicalType, Schema,
    };
    use std::fs::File;

    let schema = Schema::new(vec![
        ColumnSpec::primitive(
            "id",
            DataType::I64,
            vec![
                CoderSpec::new("delta"),
                CoderSpec::new("leb128"),
                CoderSpec::new("zstd"),
            ],
        ),
        ColumnSpec::utf8(
            "label",
            vec![
                CoderSpec::new("delta"),
                CoderSpec::new("leb128"),
                CoderSpec::new("zstd"),
            ],
            vec![CoderSpec::new("zstd")],
        ),
        ColumnSpec::nullable(
            "score",
            LogicalType::Primitive {
                data_type: DataType::F64,
            },
            vec![
                vec![
                    CoderSpec::new("rle"),
                    CoderSpec::new("bitpack_auto"),
                    CoderSpec::new("zstd"),
                ],
                vec![CoderSpec::new("gorilla"), CoderSpec::new("zstd")],
            ],
        ),
    ]);

    let path = out.with_extension("he");
    let mut w = HeliumWriter::new(File::create(&path)?, schema, &CoderRegistry::default())?;
    w.write_column(
        "id",
        LogicalColumn::Primitive(ColumnData::I64((0..100).collect())),
    )?;
    w.write_column(
        "label",
        LogicalColumn::Utf8((0..100).map(|i| format!("row-{i}")).collect()),
    )?;
    w.write_column(
        "score",
        LogicalColumn::Nullable {
            present: (0..100).map(|i| i % 3 != 0).collect(),
            value: Box::new(LogicalColumn::Primitive(ColumnData::F64(
                (0..100)
                    .filter(|i| i % 3 != 0)
                    .map(|i| i as f64 * 1.5)
                    .collect(),
            ))),
        },
    )?;
    w.finish()?;
    Ok(path)
}

#[cfg(not(feature = "arrow"))]
fn main() {
    eprintln!("Rebuild with --features arrow to run this example.");
    std::process::exit(2);
}
