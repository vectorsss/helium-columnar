//! Smoke test: SELECT count(*) FROM hits_1 via DataFusion.
//!
//! Usage:
//!   cargo run --release --features datafusion --example datafusion_smoke

use std::sync::Arc;
use std::time::Instant;

use ::datafusion::prelude::*;
use helium::sql::HeliumTableProvider;

#[tokio::main]
async fn main() -> ::datafusion::error::Result<()> {
    let he_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "hits_1.he".to_string());

    let t0 = Instant::now();

    let provider = HeliumTableProvider::try_new(&he_path).unwrap_or_else(|e| {
        eprintln!("Cannot open {he_path}: {e}");
        std::process::exit(1);
    });

    println!(
        "Opened {he_path}: {} stripe(s), {} total rows, {} column(s)",
        provider.stripe_count(),
        provider.total_rows(),
        provider.helium_schema().columns.len(),
    );

    let ctx = SessionContext::new();
    ctx.register_table("hits_1", Arc::new(provider))?;

    let df = ctx.sql("SELECT count(*) FROM hits_1").await?;
    let batches = df.collect().await?;
    let elapsed = t0.elapsed();

    use arrow::array::Int64Array;
    let arr = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Int64Array");

    println!(
        "count(*) = {}  ({:.3}s total)",
        arr.value(0),
        elapsed.as_secs_f64()
    );

    Ok(())
}
