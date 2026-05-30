//! Per-coder micro-benchmarks. Measures encode throughput for each coder on
//! a representative typed input.

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use helium::{
    BitpackAuto, BlockCoder, ColumnData, DataType, Delta, DeltaOfDelta, EliasFano, GorillaXor,
    Leb128, NonBlockCoder, Pcodec, Rle, Zstd,
};

mod datasets;

fn bench_integer_coders(c: &mut Criterion) {
    let n = 10_000;
    let xs = datasets::timestamps_jittered(n);
    let raw_bytes = n * std::mem::size_of::<i64>();

    let mut g = c.benchmark_group("integer_coders/i64_timestamps_jittered");
    g.throughput(Throughput::Bytes(raw_bytes as u64));

    g.bench_function("delta encode", |b| {
        let coder = Delta::new(DataType::I64).unwrap();
        b.iter(|| {
            coder
                .encode(black_box(&ColumnData::I64(xs.clone())))
                .unwrap()
        });
    });

    g.bench_function("delta_of_delta encode", |b| {
        let coder = DeltaOfDelta::new(DataType::I64).unwrap();
        b.iter(|| {
            coder
                .encode(black_box(&ColumnData::I64(xs.clone())))
                .unwrap()
        });
    });

    g.bench_function("leb128 encode", |b| {
        let coder = Leb128::new(DataType::I64).unwrap();
        b.iter(|| {
            coder
                .encode(black_box(&ColumnData::I64(xs.clone())))
                .unwrap()
        });
    });

    g.bench_function("pcodec encode", |b| {
        let coder = Pcodec::new(DataType::I64, None).unwrap();
        b.iter(|| {
            coder
                .encode_block(black_box(&ColumnData::I64(xs.clone())))
                .unwrap()
        });
    });

    g.finish();
}

fn bench_rle(c: &mut Criterion) {
    let n = 10_000;
    let xs = datasets::tags_low_cardinality_i64(n, 8);
    let raw_bytes = n * std::mem::size_of::<i64>();

    let mut g = c.benchmark_group("rle/i64_low_cardinality");
    g.throughput(Throughput::Bytes(raw_bytes as u64));

    g.bench_function("rle encode", |b| {
        let coder = Rle::new(DataType::I64).unwrap();
        b.iter(|| {
            coder
                .encode(black_box(&ColumnData::I64(xs.clone())))
                .unwrap()
        });
    });

    g.finish();
}

fn bench_bitpack(c: &mut Criterion) {
    let n = 10_000;
    let xs = datasets::unsigned_small_u32(n, 512);
    let raw_bytes = n * std::mem::size_of::<u32>();

    let mut g = c.benchmark_group("bitpack/u32_small");
    g.throughput(Throughput::Bytes(raw_bytes as u64));

    g.bench_function("bitpack_auto encode", |b| {
        let coder = BitpackAuto::new(DataType::U32).unwrap();
        b.iter(|| {
            coder
                .encode_block(black_box(&ColumnData::U32(xs.clone())))
                .unwrap()
        });
    });

    g.finish();
}

fn bench_elias_fano(c: &mut Criterion) {
    let n = 5_000;
    let xs = datasets::sorted_unique_u32(n, 10_000);
    let raw_bytes = n * std::mem::size_of::<u32>();

    let mut g = c.benchmark_group("elias_fano/u32_sorted");
    g.throughput(Throughput::Bytes(raw_bytes as u64));

    g.bench_function("elias_fano encode", |b| {
        let coder = EliasFano::new(DataType::U32).unwrap();
        b.iter(|| {
            coder
                .encode_block(black_box(&ColumnData::U32(xs.clone())))
                .unwrap()
        });
    });

    g.finish();
}

fn bench_gorilla(c: &mut Criterion) {
    let n = 10_000;
    let xs = datasets::measurements_gauge_f64(n);
    let raw_bytes = n * std::mem::size_of::<f64>();

    let mut g = c.benchmark_group("gorilla/f64_gauge");
    g.throughput(Throughput::Bytes(raw_bytes as u64));

    g.bench_function("gorilla encode", |b| {
        let coder = GorillaXor::new(DataType::F64).unwrap();
        b.iter(|| {
            coder
                .encode(black_box(&ColumnData::F64(xs.clone())))
                .unwrap()
        });
    });

    g.bench_function("pcodec f64 encode", |b| {
        let coder = Pcodec::new(DataType::F64, None).unwrap();
        b.iter(|| {
            coder
                .encode_block(black_box(&ColumnData::F64(xs.clone())))
                .unwrap()
        });
    });

    g.finish();
}

fn bench_block_compressors(c: &mut Criterion) {
    let n = 10_000;
    let xs = datasets::timestamps_jittered(n);
    // pre-run through delta+leb128 to get a realistic byte stream
    let prelude = {
        let d = Delta::new(DataType::I64).unwrap();
        let l = Leb128::new(DataType::I64).unwrap();
        let after_delta = d.encode(&ColumnData::I64(xs)).unwrap();
        let after_leb = l.encode(&after_delta).unwrap();
        let ColumnData::Bytes(b) = after_leb else {
            unreachable!()
        };
        b
    };

    let mut g = c.benchmark_group("block_compressors/timestamp_tail");
    g.throughput(Throughput::Bytes(prelude.len() as u64));

    g.bench_function("zstd encode", |b| {
        let coder = Zstd::default();
        b.iter(|| {
            coder
                .encode_block(black_box(&ColumnData::Bytes(prelude.clone())))
                .unwrap()
        });
    });

    g.bench_function("lz4 encode", |b| {
        let coder = helium::Lz4;
        b.iter(|| {
            coder
                .encode_block(black_box(&ColumnData::Bytes(prelude.clone())))
                .unwrap()
        });
    });

    g.finish();
}

criterion_group!(
    benches,
    bench_integer_coders,
    bench_rle,
    bench_bitpack,
    bench_elias_fano,
    bench_gorilla,
    bench_block_compressors,
);
criterion_main!(benches);
