//! Smoke test for the Arrow bridge: write a small multi-stripe `.he` in memory,
//! read it back as Arrow `RecordBatch`es, and verify the row counts match.
#[cfg(feature = "arrow")]
fn main() {
    use std::io::Cursor;

    use helium::{
        CoderRegistry, CoderSpec, ColumnData, ColumnSpec, DataType, HeliumReader, HeliumWriter,
        LogicalColumn, Schema,
    };

    let registry = CoderRegistry::default();
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "id",
        DataType::I64,
        vec![
            CoderSpec::new("delta"),
            CoderSpec::new("leb128"),
            CoderSpec::new("zstd"),
        ],
    )]);

    // Write two stripes (1000 + 500 rows) into an in-memory buffer.
    let mut buf = Vec::new();
    {
        let mut w = HeliumWriter::new(Cursor::new(&mut buf), schema, &registry).unwrap();
        w.write_column(
            "id",
            LogicalColumn::Primitive(ColumnData::I64((0..1000).collect())),
        )
        .unwrap();
        w.finish_stripe().unwrap();
        w.write_column(
            "id",
            LogicalColumn::Primitive(ColumnData::I64((1000..1500).collect())),
        )
        .unwrap();
        w.finish().unwrap();
    }

    let mut reader = HeliumReader::new(Cursor::new(buf), &registry).unwrap();
    let total_rows = reader.row_count();
    let stripe_count = reader.stripe_count();
    println!("in-memory .he: {total_rows} total rows across {stripe_count} stripes");

    let mut batch_row_sum = 0u64;
    for s in 0..stripe_count {
        let batch = reader.read_record_batch(s).unwrap();
        batch_row_sum += batch.num_rows() as u64;
    }

    println!("Sum of RecordBatch rows: {batch_row_sum}");
    assert_eq!(batch_row_sum, total_rows, "RecordBatch row count mismatch!");
    println!("Smoke test PASSED: row counts match.");
}

#[cfg(not(feature = "arrow"))]
fn main() {
    eprintln!("Rebuild with --features arrow to run this example.");
}
