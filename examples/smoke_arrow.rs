/// Smoke test: read hits_1.he as Arrow RecordBatches and verify row count matches.
#[cfg(feature = "arrow")]
fn main() {
    use helium::{CoderRegistry, HeliumReader};
    use std::fs::File;
    use std::io::BufReader;

    let path = "/Users/chizhao/Code/opensource/Helium/helium-core/hits_1.he";
    let file = File::open(path).expect("hits_1.he not found");
    let registry = CoderRegistry::default();
    let mut reader = HeliumReader::new(BufReader::new(file), &registry).unwrap();

    let total_rows = reader.row_count();
    let stripe_count = reader.stripe_count();
    println!("hits_1.he: {total_rows} total rows across {stripe_count} stripes");

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
