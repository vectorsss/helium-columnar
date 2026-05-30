//! Column-projection ("slice") benchmark: zero-copy byte copy vs the old
//! decode→re-encode path, on a wide multi-stripe file.
//!
//! Both produce the same output file; the difference is whether each kept
//! column is copied verbatim (new) or fully decoded and re-encoded (old).

use std::io::{Cursor, Seek, Write};

use criterion::{BatchSize, Criterion, Throughput, black_box, criterion_group, criterion_main};
use helium::{
    CoderRegistry, CoderSpec, ColumnData, ColumnSpec, DataType, HeliumReader, HeliumWriter,
    LogicalColumn, Schema,
};

const N_COLS: usize = 16;
const ROWS_PER_STRIPE: usize = 25_000;
const STRIPES: usize = 4;

fn pipe() -> Vec<CoderSpec> {
    // Real coder work: delta → leb128 → zstd.
    vec![
        CoderSpec::new("delta"),
        CoderSpec::new("leb128"),
        CoderSpec::new("zstd"),
    ]
}

fn col_name(i: usize) -> String {
    format!("c{i:02}")
}

fn wide_schema() -> Schema {
    Schema::new(
        (0..N_COLS)
            .map(|i| ColumnSpec::primitive(col_name(i), DataType::I64, pipe()))
            .collect(),
    )
}

/// Build a wide, multi-stripe `.he` file in memory.
///
/// Columns alternate between near-linear (timestamp-like, very compressible)
/// and high-entropy pseudo-random (measurement-like, where the re-encode's
/// zstd actually has to work) so the slice cost is representative of mixed
/// real data rather than a best/worst case.
fn build_source() -> Vec<u8> {
    let registry = CoderRegistry::default();
    let mut buf = Vec::new();
    {
        let mut w = HeliumWriter::new(Cursor::new(&mut buf), wide_schema(), &registry).unwrap();
        for s in 0..STRIPES {
            for i in 0..N_COLS {
                let mut state = (s as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (i as u64 + 1);
                let vals: Vec<i64> = (0..ROWS_PER_STRIPE)
                    .map(|r| {
                        if i % 2 == 0 {
                            // near-linear, highly compressible
                            (s * ROWS_PER_STRIPE + r) as i64 + i as i64 * 1_000_000
                        } else {
                            // LCG pseudo-random, low compressibility
                            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                            (state >> 16) as i64
                        }
                    })
                    .collect();
                w.write_column(
                    &col_name(i),
                    LogicalColumn::Primitive(ColumnData::I64(vals)),
                )
                .unwrap();
            }
            if s + 1 < STRIPES {
                w.finish_stripe().unwrap();
            }
        }
        w.finish().unwrap();
    }
    buf
}

/// The OLD projection path: decode each kept column and re-encode it through a
/// fresh writer (uses only public APIs — this is exactly what `project_to`
/// did before it became zero-copy).
fn project_decode_reencode<W: Write + Seek>(
    reader: &mut HeliumReader<Cursor<&[u8]>>,
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

fn bench_slice(c: &mut Criterion) {
    let registry = CoderRegistry::default();
    let src = build_source();
    // Project the first half (c00..c07): a mix of compressible (even) and
    // high-entropy (odd) columns.
    let kept: Vec<String> = (0..N_COLS / 2).map(col_name).collect();
    let kept_refs: Vec<&str> = kept.iter().map(String::as_str).collect();

    // Reader-open is excluded from the timed region (it is identical for both
    // paths) via `iter_batched`'s untimed setup, isolating the projection cost.
    let open = || HeliumReader::new(Cursor::new(src.as_slice()), &registry).unwrap();

    let total_rows = (ROWS_PER_STRIPE * STRIPES) as u64;
    let mut g = c.benchmark_group("slice/project_8_of_16_cols");
    g.throughput(Throughput::Elements(total_rows * (N_COLS / 2) as u64));

    g.bench_function("zero_copy", |b| {
        b.iter_batched(
            open,
            |mut reader| {
                let out = reader
                    .project_to(
                        black_box(&kept_refs),
                        Cursor::new(Vec::<u8>::new()),
                        &registry,
                    )
                    .unwrap();
                black_box(out.into_inner().len())
            },
            BatchSize::SmallInput,
        );
    });

    g.bench_function("decode_reencode", |b| {
        b.iter_batched(
            open,
            |mut reader| {
                let out = project_decode_reencode(
                    &mut reader,
                    black_box(&kept_refs),
                    Cursor::new(Vec::<u8>::new()),
                    &registry,
                );
                black_box(out.into_inner().len())
            },
            BatchSize::SmallInput,
        );
    });

    g.finish();
}

criterion_group!(benches, bench_slice);
criterion_main!(benches);
