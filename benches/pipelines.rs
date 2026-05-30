//! End-to-end pipeline benchmarks: encode + decode on realistic column shapes.
//! Also prints compression ratios for each pipeline to stdout on first run.

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use helium::{
    BitpackAuto, BlockCoder, ColumnData, DataType, Delta, DeltaOfDelta, GorillaXor, Leb128,
    NonBlockCoder, Pcodec, Pipeline, Rle, StageCoder, Zstd,
};

mod datasets;

fn nb<T: 'static + NonBlockCoder>(c: T) -> StageCoder {
    StageCoder::NonBlock(Box::new(c))
}
fn blk<T: 'static + BlockCoder>(c: T) -> StageCoder {
    StageCoder::Block(Box::new(c))
}

fn bench_timestamp_pipelines(c: &mut Criterion) {
    let n = 10_000;
    let xs = datasets::timestamps_jittered(n);
    let raw = n * std::mem::size_of::<i64>();

    let build_classic = || {
        Pipeline::new(
            DataType::I64,
            vec![
                nb(Delta::new(DataType::I64).unwrap()),
                nb(Leb128::new(DataType::I64).unwrap()),
                blk(Zstd::default()),
            ],
        )
        .unwrap()
    };
    let build_dod = || {
        Pipeline::new(
            DataType::I64,
            vec![
                nb(DeltaOfDelta::new(DataType::I64).unwrap()),
                nb(Leb128::new(DataType::I64).unwrap()),
                blk(Zstd::default()),
            ],
        )
        .unwrap()
    };
    let build_pco = || {
        Pipeline::new(
            DataType::I64,
            vec![blk(Pcodec::new(DataType::I64, None).unwrap())],
        )
        .unwrap()
    };

    // Print ratios once.
    for (name, p) in [
        ("delta+leb128+zstd", build_classic()),
        ("dod+leb128+zstd", build_dod()),
        ("pcodec", build_pco()),
    ] {
        let encoded = p.encode(ColumnData::I64(xs.clone())).unwrap();
        eprintln!(
            "timestamps_jittered[{n}]: {name} → {} bytes ({:.2}x)",
            encoded.len(),
            raw as f64 / encoded.len() as f64
        );
    }

    let mut g = c.benchmark_group("pipelines/timestamps_jittered");
    g.throughput(Throughput::Bytes(raw as u64));
    g.bench_function("delta+leb128+zstd encode", |b| {
        let p = build_classic();
        b.iter(|| p.encode(black_box(ColumnData::I64(xs.clone()))).unwrap());
    });
    g.bench_function("dod+leb128+zstd encode", |b| {
        let p = build_dod();
        b.iter(|| p.encode(black_box(ColumnData::I64(xs.clone()))).unwrap());
    });
    g.bench_function("pcodec encode", |b| {
        let p = build_pco();
        b.iter(|| p.encode(black_box(ColumnData::I64(xs.clone()))).unwrap());
    });
    g.finish();
}

fn bench_gauge_f64_pipelines(c: &mut Criterion) {
    let n = 10_000;
    let xs = datasets::measurements_gauge_f64(n);
    let raw = n * std::mem::size_of::<f64>();

    let build_gorilla = || {
        Pipeline::new(
            DataType::F64,
            vec![
                nb(GorillaXor::new(DataType::F64).unwrap()),
                blk(Zstd::default()),
            ],
        )
        .unwrap()
    };
    let build_pco = || {
        Pipeline::new(
            DataType::F64,
            vec![blk(Pcodec::new(DataType::F64, None).unwrap())],
        )
        .unwrap()
    };

    for (name, p) in [("gorilla+zstd", build_gorilla()), ("pcodec", build_pco())] {
        let encoded = p.encode(ColumnData::F64(xs.clone())).unwrap();
        eprintln!(
            "gauge_f64[{n}]: {name} → {} bytes ({:.2}x)",
            encoded.len(),
            raw as f64 / encoded.len() as f64
        );
    }

    let mut g = c.benchmark_group("pipelines/gauge_f64");
    g.throughput(Throughput::Bytes(raw as u64));
    g.bench_function("gorilla+zstd encode", |b| {
        let p = build_gorilla();
        b.iter(|| p.encode(black_box(ColumnData::F64(xs.clone()))).unwrap());
    });
    g.bench_function("pcodec encode", |b| {
        let p = build_pco();
        b.iter(|| p.encode(black_box(ColumnData::F64(xs.clone()))).unwrap());
    });
    g.finish();
}

fn bench_low_cardinality_tags(c: &mut Criterion) {
    let n = 10_000;
    let xs = datasets::tags_low_cardinality_i64(n, 8);
    let raw = n * std::mem::size_of::<i64>();

    let build_rle = || {
        Pipeline::new(
            DataType::I64,
            vec![
                nb(Rle::new(DataType::I64).unwrap()),
                nb(Leb128::new(DataType::I64).unwrap()),
                blk(Zstd::default()),
            ],
        )
        .unwrap()
    };
    let build_bitpack = || {
        Pipeline::new(
            DataType::I64,
            vec![
                blk(BitpackAuto::new(DataType::I64).unwrap()),
                blk(Zstd::default()),
            ],
        )
        .unwrap()
    };

    for (name, p) in [
        ("rle+leb128+zstd", build_rle()),
        ("bitpack_auto+zstd", build_bitpack()),
    ] {
        let encoded = p.encode(ColumnData::I64(xs.clone())).unwrap();
        eprintln!(
            "tags_8card[{n}]: {name} → {} bytes ({:.2}x)",
            encoded.len(),
            raw as f64 / encoded.len() as f64
        );
    }

    let mut g = c.benchmark_group("pipelines/tags_low_cardinality");
    g.throughput(Throughput::Bytes(raw as u64));
    g.bench_function("rle+leb128+zstd encode", |b| {
        let p = build_rle();
        b.iter(|| p.encode(black_box(ColumnData::I64(xs.clone()))).unwrap());
    });
    g.bench_function("bitpack_auto+zstd encode", |b| {
        let p = build_bitpack();
        b.iter(|| p.encode(black_box(ColumnData::I64(xs.clone()))).unwrap());
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_timestamp_pipelines,
    bench_gauge_f64_pipelines,
    bench_low_cardinality_tags,
);
criterion_main!(benches);
