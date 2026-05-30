use helium::{
    AccessPattern, BitpackAuto, BitpackFixed, ColumnData, DataType, Delta, DeltaMin, DeltaOfDelta,
    EliasFano, GorillaXor, HeliumError, Leb128, Lz4, Pcodec, Pipeline, Rle, StageCoder, Zstd,
};

fn nb<T: 'static + helium::NonBlockCoder>(c: T) -> StageCoder {
    StageCoder::NonBlock(Box::new(c))
}
fn blk<T: 'static + helium::BlockCoder>(c: T) -> StageCoder {
    StageCoder::Block(Box::new(c))
}

fn timestamp_pipeline() -> Pipeline {
    Pipeline::new(
        DataType::I64,
        vec![
            nb(Delta::new(DataType::I64).unwrap()),
            nb(Leb128::new(DataType::I64).unwrap()),
            blk(Zstd::default()),
        ],
    )
    .expect("canonical timestamp pipeline should be valid")
}

#[test]
fn full_pipeline_roundtrip_on_timestamps() {
    let original: Vec<i64> = (1_700_000_000..1_700_000_500).collect();
    let pipeline = timestamp_pipeline();

    let encoded = pipeline.encode(ColumnData::I64(original.clone())).unwrap();
    assert!(matches!(encoded, ColumnData::Bytes(_)));

    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::I64(original));
}

#[test]
fn delta_only_roundtrip() {
    let original: Vec<i64> = vec![10, 20, 15, 30, 30, -5, 0];
    let pipeline =
        Pipeline::new(DataType::I64, vec![nb(Delta::new(DataType::I64).unwrap())]).unwrap();

    let encoded = pipeline.encode(ColumnData::I64(original.clone())).unwrap();
    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::I64(original));
}

#[test]
fn delta_roundtrips_every_integer_width() {
    // Each width exercises its own wrapping and wide-range behavior.
    macro_rules! check {
        ($dt:ident, $variant:ident, $ty:ty, $values:expr) => {{
            let original: Vec<$ty> = $values;
            let pipeline =
                Pipeline::new(DataType::$dt, vec![nb(Delta::new(DataType::$dt).unwrap())]).unwrap();
            let encoded = pipeline
                .encode(ColumnData::$variant(original.clone()))
                .unwrap();
            let decoded = pipeline.decode(encoded).unwrap();
            assert_eq!(
                decoded,
                ColumnData::$variant(original),
                "{}",
                stringify!($dt)
            );
        }};
    }
    check!(I8, I8, i8, vec![-10, -5, 0, 5, 10, i8::MAX, i8::MIN]);
    check!(I16, I16, i16, vec![-1000, 0, 1000, i16::MAX, i16::MIN]);
    check!(I32, I32, i32, (0..100).collect());
    check!(I64, I64, i64, vec![i64::MIN, 0, i64::MAX]);
    check!(U8, U8, u8, (0u8..=255).collect());
    check!(U16, U16, u16, vec![0, 1, 1000, u16::MAX]);
    check!(U32, U32, u32, (0u32..100).collect());
    check!(U64, U64, u64, vec![0, u64::MAX, 100]);
}

#[test]
fn leb128_handles_negative_and_extremes() {
    let original: Vec<i64> = vec![0, -1, 1, i64::MIN, i64::MAX, -123_456_789, 987_654_321];
    let pipeline =
        Pipeline::new(DataType::I64, vec![nb(Leb128::new(DataType::I64).unwrap())]).unwrap();

    let encoded = pipeline.encode(ColumnData::I64(original.clone())).unwrap();
    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::I64(original));
}

#[test]
fn leb128_roundtrips_unsigned_widths() {
    let pipeline_u16 =
        Pipeline::new(DataType::U16, vec![nb(Leb128::new(DataType::U16).unwrap())]).unwrap();
    let original: Vec<u16> = vec![0, 127, 128, 16383, 16384, u16::MAX];
    let encoded = pipeline_u16
        .encode(ColumnData::U16(original.clone()))
        .unwrap();
    let decoded = pipeline_u16.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::U16(original));
}

#[test]
fn empty_column_roundtrip() {
    let original: Vec<i64> = vec![];
    let pipeline = timestamp_pipeline();

    let encoded = pipeline.encode(ColumnData::I64(original.clone())).unwrap();
    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::I64(original));
}

#[test]
fn single_element_roundtrip() {
    let original: Vec<i64> = vec![42];
    let pipeline = timestamp_pipeline();

    let encoded = pipeline.encode(ColumnData::I64(original.clone())).unwrap();
    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::I64(original));
}

#[test]
fn all_same_values_compress_heavily() {
    let original: Vec<i64> = vec![777; 10_000];
    let pipeline = timestamp_pipeline();

    let encoded = pipeline.encode(ColumnData::I64(original.clone())).unwrap();
    let ColumnData::Bytes(ref b) = encoded else {
        panic!("expected bytes output");
    };
    assert!(b.len() < 100, "compressed size {} too large", b.len());

    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::I64(original));
}

#[test]
fn pipeline_rejects_nonblock_after_block() {
    let result = Pipeline::new(
        DataType::I64,
        vec![
            nb(Leb128::new(DataType::I64).unwrap()),
            blk(Zstd::default()),
            nb(Delta::new(DataType::I64).unwrap()),
        ],
    );
    assert!(matches!(result, Err(HeliumError::PipelineOrder(_))));
}

#[test]
fn pipeline_rejects_type_mismatch() {
    let result = Pipeline::new(
        DataType::I64,
        vec![nb(Delta::new(DataType::I64).unwrap()), blk(Zstd::default())],
    );
    assert!(matches!(result, Err(HeliumError::TypeMismatch { .. })));
}

#[test]
fn pipeline_rejects_wrong_input_datatype() {
    let pipeline = timestamp_pipeline();
    let result = pipeline.encode(ColumnData::Bytes(vec![1, 2, 3]));
    assert!(matches!(result, Err(HeliumError::RuntimeType { .. })));
}

// ------------------------------------------------------------------
// RLE
// ------------------------------------------------------------------

#[test]
fn rle_roundtrip_low_cardinality() {
    let mut original = Vec::new();
    for (v, n) in [(1i64, 50), (2, 100), (3, 1), (1, 200), (7, 75)] {
        for _ in 0..n {
            original.push(v);
        }
    }
    let pipeline =
        Pipeline::new(DataType::I64, vec![nb(Rle::new(DataType::I64).unwrap())]).unwrap();

    let encoded = pipeline.encode(ColumnData::I64(original.clone())).unwrap();
    assert_eq!(encoded.len(), 10);
    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::I64(original));
}

#[test]
fn rle_then_leb128_then_zstd() {
    let original: Vec<i64> = std::iter::repeat_n(5, 1_000).collect();
    let pipeline = Pipeline::new(
        DataType::I64,
        vec![
            nb(Rle::new(DataType::I64).unwrap()),
            nb(Leb128::new(DataType::I64).unwrap()),
            blk(Zstd::default()),
        ],
    )
    .unwrap();

    let encoded = pipeline.encode(ColumnData::I64(original.clone())).unwrap();
    let ColumnData::Bytes(ref b) = encoded else {
        panic!("expected bytes");
    };
    assert!(b.len() < 40, "compressed size {} too large", b.len());

    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::I64(original));
}

#[test]
fn rle_splits_long_runs_for_small_count_types() {
    // Run of 1000 doesn't fit in one i8 count (max 127) — RLE must split it
    // into multiple pairs rather than erroring out.
    let pipeline = Pipeline::new(DataType::I8, vec![nb(Rle::new(DataType::I8).unwrap())]).unwrap();
    let original: Vec<i8> = vec![7; 1000];
    let encoded = pipeline.encode(ColumnData::I8(original.clone())).unwrap();
    let ColumnData::I8(ref pairs) = encoded else {
        panic!("expected i8 output");
    };
    // 1000 / 127 = 7 full chunks + 111 remainder = 8 pairs = 16 ints.
    assert_eq!(pairs.len() % 2, 0);
    assert!(pairs.len() >= 16 && pairs.len() <= 20);
    // All pairs are (7, positive_count); counts must sum to 1000.
    let mut total: i64 = 0;
    for chunk in pairs.chunks_exact(2) {
        assert_eq!(chunk[0], 7);
        assert!(chunk[1] > 0);
        total += chunk[1] as i64;
    }
    assert_eq!(total, 1000);

    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::I8(original));
}

// ------------------------------------------------------------------
// DeltaMin
// ------------------------------------------------------------------

#[test]
fn deltamin_roundtrip_concentrated_values() {
    let original: Vec<i64> = vec![-78, -82, -80, -79, -85, -77, -81, -83, -78, -80];
    let pipeline = Pipeline::new(
        DataType::I64,
        vec![blk(DeltaMin::new(DataType::I64).unwrap())],
    )
    .unwrap();

    let encoded = pipeline.encode(ColumnData::I64(original.clone())).unwrap();
    let ColumnData::I64(ref v) = encoded else {
        panic!("expected i64 output");
    };
    assert_eq!(v.len(), original.len() + 1);
    assert_eq!(v[0], -85);
    assert!(v[1..].iter().all(|&d| d >= 0));

    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::I64(original));
}

#[test]
fn deltamin_empty_column() {
    let pipeline = Pipeline::new(
        DataType::I64,
        vec![blk(DeltaMin::new(DataType::I64).unwrap())],
    )
    .unwrap();
    let encoded = pipeline.encode(ColumnData::I64(Vec::new())).unwrap();
    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::I64(Vec::new()));
}

// ------------------------------------------------------------------
// Pcodec
// ------------------------------------------------------------------

#[test]
fn pcodec_alone_roundtrip() {
    let original: Vec<i64> = (0..10_000).map(|i| i * 7 - 3).collect();
    let pipeline = Pipeline::new(
        DataType::I64,
        vec![blk(Pcodec::new(DataType::I64, None).unwrap())],
    )
    .unwrap();

    let encoded = pipeline.encode(ColumnData::I64(original.clone())).unwrap();
    let raw = original.len() * std::mem::size_of::<i64>();
    assert!(encoded.len() * 100 < raw);

    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::I64(original));
}

#[test]
fn pcodec_roundtrips_every_supported_type() {
    macro_rules! check {
        ($dt:ident, $variant:ident, $ty:ty, $vals:expr) => {{
            let original: Vec<$ty> = $vals;
            let pipeline = Pipeline::new(
                DataType::$dt,
                vec![blk(Pcodec::new(DataType::$dt, None).unwrap())],
            )
            .unwrap();
            let encoded = pipeline
                .encode(ColumnData::$variant(original.clone()))
                .unwrap();
            let decoded = pipeline.decode(encoded).unwrap();
            assert_eq!(
                decoded,
                ColumnData::$variant(original),
                "{}",
                stringify!($dt)
            );
        }};
    }
    check!(I8, I8, i8, (-100i8..100).collect());
    check!(I16, I16, i16, (-500i16..500).collect());
    check!(I32, I32, i32, (0..1000).collect());
    check!(I64, I64, i64, (0..1000).map(|i| i * 7).collect());
    check!(U8, U8, u8, (0u8..200).collect());
    check!(U16, U16, u16, (0u16..1000).collect());
    check!(U32, U32, u32, (0u32..1000).collect());
    check!(U64, U64, u64, (0u64..1000).collect());
    check!(F32, F32, f32, (0..1000).map(|i| i as f32 * 0.5).collect());
    check!(F64, F64, f64, (0..1000).map(|i| (i as f64).sin()).collect());
}

#[test]
fn pcodec_empty_roundtrip() {
    let pipeline = Pipeline::new(
        DataType::I64,
        vec![blk(Pcodec::new(DataType::I64, None).unwrap())],
    )
    .unwrap();
    let encoded = pipeline.encode(ColumnData::I64(Vec::new())).unwrap();
    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::I64(Vec::new()));
}

#[test]
fn delta_then_pcodec_roundtrip() {
    let original: Vec<i64> = (0..5_000).map(|i| 1_700_000_000 + i * 30).collect();
    let pipeline = Pipeline::new(
        DataType::I64,
        vec![
            nb(Delta::new(DataType::I64).unwrap()),
            blk(Pcodec::new(DataType::I64, None).unwrap()),
        ],
    )
    .unwrap();

    let encoded = pipeline.encode(ColumnData::I64(original.clone())).unwrap();
    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::I64(original));
}

// ------------------------------------------------------------------
// Lz4
// ------------------------------------------------------------------

#[test]
fn lz4_roundtrip_via_leb128() {
    let original: Vec<i64> = (0..5_000).map(|i| i % 23).collect();
    let pipeline = Pipeline::new(
        DataType::I64,
        vec![nb(Leb128::new(DataType::I64).unwrap()), blk(Lz4)],
    )
    .unwrap();

    let encoded = pipeline.encode(ColumnData::I64(original.clone())).unwrap();
    let raw = original.len() * std::mem::size_of::<i64>();
    assert!(encoded.len() < raw / 2);

    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::I64(original));
}

#[test]
fn lz4_handles_corrupted_input() {
    let pipeline = Pipeline::new(
        DataType::I64,
        vec![nb(Leb128::new(DataType::I64).unwrap()), blk(Lz4)],
    )
    .unwrap();
    let bogus = ColumnData::Bytes(vec![0xff; 64]);
    let result = pipeline.decode(bogus);
    assert!(matches!(result, Err(HeliumError::Corrupted { .. })));
}

// ------------------------------------------------------------------
// BitpackFixed
// ------------------------------------------------------------------

#[test]
fn bitpack_fixed_roundtrip_u32() {
    let original: Vec<u32> = (0..1000).map(|i| (i * 13) % 1024).collect();
    let pipeline = Pipeline::new(
        DataType::U32,
        vec![nb(BitpackFixed::new(DataType::U32, 10).unwrap())],
    )
    .unwrap();

    let encoded = pipeline.encode(ColumnData::U32(original.clone())).unwrap();
    let ColumnData::Bytes(ref b) = encoded else {
        panic!("expected bytes");
    };
    assert_eq!(b.len(), 8 + 1250);

    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::U32(original));
}

#[test]
fn bitpack_fixed_rejects_out_of_range() {
    let pipeline = Pipeline::new(
        DataType::U32,
        vec![nb(BitpackFixed::new(DataType::U32, 4).unwrap())],
    )
    .unwrap();
    let result = pipeline.encode(ColumnData::U32(vec![1, 2, 16, 3]));
    assert!(matches!(result, Err(HeliumError::CoderFailed { .. })));
}

#[test]
fn bitpack_fixed_rejects_negative_signed_input() {
    let pipeline = Pipeline::new(
        DataType::I32,
        vec![nb(BitpackFixed::new(DataType::I32, 8).unwrap())],
    )
    .unwrap();
    let result = pipeline.encode(ColumnData::I32(vec![1, -2, 3]));
    assert!(matches!(result, Err(HeliumError::CoderFailed { .. })));
}

#[test]
fn bitpack_fixed_rejects_width_out_of_range() {
    assert!(BitpackFixed::new(DataType::U32, 33).is_err());
    assert!(BitpackFixed::new(DataType::U32, 100).is_err());
    assert!(BitpackFixed::new(DataType::U64, 63).is_ok());
    assert!(BitpackFixed::new(DataType::U8, 0).is_ok());
}

#[test]
fn bitpack_fixed_empty() {
    let pipeline = Pipeline::new(
        DataType::U32,
        vec![nb(BitpackFixed::new(DataType::U32, 8).unwrap())],
    )
    .unwrap();
    let encoded = pipeline.encode(ColumnData::U32(Vec::new())).unwrap();
    assert_eq!(encoded.len(), 8);
    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::U32(Vec::new()));
}

// ------------------------------------------------------------------
// BitpackAuto
// ------------------------------------------------------------------

#[test]
fn bitpack_auto_roundtrip_u32() {
    let original: Vec<u32> = (0..200).map(|i| i * 2 + (i % 5)).collect();
    let max_val = *original.iter().max().unwrap();
    let expected_width = 64 - (max_val as u64).leading_zeros();

    let pipeline = Pipeline::new(
        DataType::U32,
        vec![blk(BitpackAuto::new(DataType::U32).unwrap())],
    )
    .unwrap();

    let encoded = pipeline.encode(ColumnData::U32(original.clone())).unwrap();
    let ColumnData::Bytes(ref b) = encoded else {
        panic!("expected bytes");
    };
    assert_eq!(b[8] as u32, expected_width);

    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::U32(original));
}

#[test]
fn bitpack_auto_all_zeros_uses_width_zero() {
    let original: Vec<u32> = vec![0; 100];
    let pipeline = Pipeline::new(
        DataType::U32,
        vec![blk(BitpackAuto::new(DataType::U32).unwrap())],
    )
    .unwrap();

    let encoded = pipeline.encode(ColumnData::U32(original.clone())).unwrap();
    let ColumnData::Bytes(ref b) = encoded else {
        panic!("expected bytes");
    };
    assert_eq!(b.len(), 9);
    assert_eq!(b[8], 0);

    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::U32(original));
}

#[test]
fn bitpack_auto_empty() {
    let pipeline = Pipeline::new(
        DataType::U32,
        vec![blk(BitpackAuto::new(DataType::U32).unwrap())],
    )
    .unwrap();
    let encoded = pipeline.encode(ColumnData::U32(Vec::new())).unwrap();
    assert_eq!(encoded.len(), 9);
    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::U32(Vec::new()));
}

// ------------------------------------------------------------------
// Gorilla XOR
// ------------------------------------------------------------------

fn gorilla_pipeline(dt: DataType) -> Pipeline {
    Pipeline::new(dt, vec![nb(GorillaXor::new(dt).unwrap())]).unwrap()
}

#[test]
fn gorilla_f64_roundtrip_slowly_drifting() {
    // Temperature-sensor shape: quantized to 0.1, slow sinusoidal drift, so
    // adjacent values are often identical and otherwise share most mantissa
    // bits. This is the real pattern Gorilla XOR was designed for.
    let original: Vec<f64> = (0..10_000)
        .map(|i| {
            let t = i as f64 * 0.01;
            ((20.0 + t.sin() * 2.0) * 10.0).round() / 10.0
        })
        .collect();
    let pipeline = gorilla_pipeline(DataType::F64);

    let encoded = pipeline.encode(ColumnData::F64(original.clone())).unwrap();
    let raw = original.len() * std::mem::size_of::<f64>();
    assert!(
        encoded.len() * 3 < raw,
        "expected >3x compression on quantized drifting floats, got {} vs {}",
        encoded.len(),
        raw
    );

    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::F64(original));
}

#[test]
fn gorilla_f64_roundtrip_constant() {
    let original: Vec<f64> = vec![42.5; 5_000];
    let pipeline = gorilla_pipeline(DataType::F64);

    let encoded = pipeline.encode(ColumnData::F64(original.clone())).unwrap();
    // 8B count + 8B first value + ~(n-1) single-bit zeros ≈ 16 + n/8.
    let raw = original.len() * std::mem::size_of::<f64>();
    assert!(encoded.len() * 30 < raw);

    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::F64(original));
}

#[test]
fn gorilla_f64_handles_special_values() {
    // NaN bit patterns intentionally vary to exercise the canonical bits path.
    let original: Vec<f64> = vec![
        0.0,
        -0.0,
        f64::MIN_POSITIVE,
        1.0,
        -1.0,
        1e-300,
        1e300,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let pipeline = gorilla_pipeline(DataType::F64);

    let encoded = pipeline.encode(ColumnData::F64(original.clone())).unwrap();
    let decoded = pipeline.decode(encoded).unwrap();
    // Bit-exact comparison via to_bits — NaN != NaN under ==, but bit patterns match.
    let ColumnData::F64(got) = decoded else {
        panic!("expected F64");
    };
    for (a, b) in original.iter().zip(got.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "mismatch: {a} vs {b}");
    }
}

#[test]
fn gorilla_f32_roundtrip() {
    let original: Vec<f32> = (0..2_000).map(|i| (i as f32 * 0.1).cos()).collect();
    let pipeline = gorilla_pipeline(DataType::F32);

    let encoded = pipeline.encode(ColumnData::F32(original.clone())).unwrap();
    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::F32(original));
}

#[test]
fn gorilla_empty_and_single() {
    for &dt in &[DataType::F32, DataType::F64] {
        let pipeline = gorilla_pipeline(dt);
        let empty_in = match dt {
            DataType::F32 => ColumnData::F32(Vec::new()),
            DataType::F64 => ColumnData::F64(Vec::new()),
            _ => unreachable!(),
        };
        let encoded = pipeline.encode(empty_in.clone()).unwrap();
        assert_eq!(pipeline.decode(encoded).unwrap(), empty_in);

        let single_in = match dt {
            DataType::F32 => ColumnData::F32(vec![42.5_f32]),
            DataType::F64 => ColumnData::F64(vec![42.5_f64]),
            _ => unreachable!(),
        };
        let encoded = pipeline.encode(single_in.clone()).unwrap();
        assert_eq!(pipeline.decode(encoded).unwrap(), single_in);
    }
}

#[test]
fn gorilla_rejects_non_float_types() {
    assert!(GorillaXor::new(DataType::I64).is_err());
    assert!(GorillaXor::new(DataType::U32).is_err());
    assert!(GorillaXor::new(DataType::Bytes).is_err());
}

#[test]
fn gorilla_then_zstd_composition() {
    let original: Vec<f64> = (0..5_000).map(|i| (i as f64).powf(1.1)).collect();
    let pipeline = Pipeline::new(
        DataType::F64,
        vec![
            nb(GorillaXor::new(DataType::F64).unwrap()),
            blk(Zstd::default()),
        ],
    )
    .unwrap();

    let encoded = pipeline.encode(ColumnData::F64(original.clone())).unwrap();
    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::F64(original));
}

// ------------------------------------------------------------------
// Elias-Fano
// ------------------------------------------------------------------

#[test]
fn elias_fano_roundtrip_inverted_index_shape() {
    // Inverted-index postings: strictly increasing u32 doc IDs drawn from a
    // large universe. Classic EF use case.
    let mut rng = 12345u32;
    let original: Vec<u32> = std::iter::from_fn(|| {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        Some(rng % 10_000)
    })
    .take(5000)
    .collect::<std::collections::BTreeSet<_>>()
    .into_iter()
    .collect();

    let pipeline = Pipeline::new(
        DataType::U32,
        vec![blk(EliasFano::new(DataType::U32).unwrap())],
    )
    .unwrap();

    let encoded = pipeline.encode(ColumnData::U32(original.clone())).unwrap();
    let raw = original.len() * std::mem::size_of::<u32>();
    // EF for ~5000 ids in [0, 10000] universe: ~2 bits + log2(2) = ~3 bits/value.
    // Raw is 32 bits/value. Expect at least 4x compression.
    assert!(
        encoded.len() * 4 < raw,
        "expected >4x compression, got {} vs {raw}",
        encoded.len()
    );

    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::U32(original));
}

#[test]
fn elias_fano_roundtrip_u64() {
    let original: Vec<u64> = (100..2000).step_by(7).collect();
    let pipeline = Pipeline::new(
        DataType::U64,
        vec![blk(EliasFano::new(DataType::U64).unwrap())],
    )
    .unwrap();

    let encoded = pipeline.encode(ColumnData::U64(original.clone())).unwrap();
    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::U64(original));
}

#[test]
fn elias_fano_rejects_non_increasing() {
    let pipeline = Pipeline::new(
        DataType::U32,
        vec![blk(EliasFano::new(DataType::U32).unwrap())],
    )
    .unwrap();
    // Equal values (not strictly increasing) rejected.
    let err = pipeline
        .encode(ColumnData::U32(vec![1, 2, 2, 3]))
        .unwrap_err();
    assert!(matches!(err, HeliumError::CoderFailed { .. }));
    // Decreasing also rejected.
    let err = pipeline.encode(ColumnData::U32(vec![5, 3, 7])).unwrap_err();
    assert!(matches!(err, HeliumError::CoderFailed { .. }));
}

#[test]
fn elias_fano_empty_and_single() {
    let pipeline = Pipeline::new(
        DataType::U32,
        vec![blk(EliasFano::new(DataType::U32).unwrap())],
    )
    .unwrap();
    let encoded = pipeline.encode(ColumnData::U32(Vec::new())).unwrap();
    assert_eq!(
        pipeline.decode(encoded).unwrap(),
        ColumnData::U32(Vec::new())
    );
    let encoded = pipeline.encode(ColumnData::U32(vec![42])).unwrap();
    assert_eq!(pipeline.decode(encoded).unwrap(), ColumnData::U32(vec![42]));
}

#[test]
fn elias_fano_rejects_non_unsigned_types() {
    assert!(EliasFano::new(DataType::I64).is_err());
    assert!(EliasFano::new(DataType::F64).is_err());
    assert!(EliasFano::new(DataType::U8).is_err());
}

// ------------------------------------------------------------------
// Delta-of-delta
// ------------------------------------------------------------------

#[test]
fn delta_of_delta_zeros_out_uniform_series() {
    let original: Vec<i64> = (0..1000).map(|i| 1_700_000_000 + i * 30).collect();
    let pipeline = Pipeline::new(
        DataType::I64,
        vec![nb(DeltaOfDelta::new(DataType::I64).unwrap())],
    )
    .unwrap();
    let encoded = pipeline.encode(ColumnData::I64(original.clone())).unwrap();
    let ColumnData::I64(ref d) = encoded else {
        panic!("expected i64");
    };
    // First two elements are warm-up (raw first value then first-delta minus
    // zero-prior). From index 2 on, uniform-step data produces all zeros.
    assert!(
        d[2..].iter().all(|&v| v == 0),
        "expected all-zero tail, got {:?}",
        &d[2..5]
    );
    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::I64(original));
}

#[test]
fn delta_of_delta_then_leb128_then_zstd_crushes_uniform_timestamps() {
    let original: Vec<i64> = (0..10_000).map(|i| 1_700_000_000 + i * 30).collect();
    let pipeline = Pipeline::new(
        DataType::I64,
        vec![
            nb(DeltaOfDelta::new(DataType::I64).unwrap()),
            nb(Leb128::new(DataType::I64).unwrap()),
            blk(Zstd::default()),
        ],
    )
    .unwrap();
    let encoded = pipeline.encode(ColumnData::I64(original.clone())).unwrap();
    let raw = original.len() * std::mem::size_of::<i64>();
    assert!(
        encoded.len() * 500 < raw,
        "expected >500x compression on uniform timestamps, got {} vs {raw}",
        encoded.len()
    );
    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::I64(original));
}

#[test]
fn delta_of_delta_handles_jitter() {
    let original: Vec<i64> = (0..500).map(|i| i * 30 + ((i % 3) - 1)).collect();
    let pipeline = Pipeline::new(
        DataType::I64,
        vec![nb(DeltaOfDelta::new(DataType::I64).unwrap())],
    )
    .unwrap();
    let encoded = pipeline.encode(ColumnData::I64(original.clone())).unwrap();
    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::I64(original));
}

#[test]
fn delta_of_delta_empty_and_single() {
    let pipeline = Pipeline::new(
        DataType::U32,
        vec![nb(DeltaOfDelta::new(DataType::U32).unwrap())],
    )
    .unwrap();
    for input in [Vec::<u32>::new(), vec![99], vec![10, 20]] {
        let encoded = pipeline.encode(ColumnData::U32(input.clone())).unwrap();
        let decoded = pipeline.decode(encoded).unwrap();
        assert_eq!(decoded, ColumnData::U32(input));
    }
}

#[test]
fn deltamin_then_bitpack_auto_then_zstd_on_signed() {
    // Signed I64 column with known non-negative deltamin output — composition
    // we deliberately enabled by letting bitpack accept any integer type.
    let original: Vec<i64> = (0..5_000).map(|i| 1_000_000 + (i % 50)).collect();
    let pipeline = Pipeline::new(
        DataType::I64,
        vec![
            blk(DeltaMin::new(DataType::I64).unwrap()),
            blk(BitpackAuto::new(DataType::I64).unwrap()),
            blk(Zstd::default()),
        ],
    )
    .unwrap();

    let encoded = pipeline.encode(ColumnData::I64(original.clone())).unwrap();
    let raw = original.len() * std::mem::size_of::<i64>();
    assert!(encoded.len() * 20 < raw);

    let decoded = pipeline.decode(encoded).unwrap();
    assert_eq!(decoded, ColumnData::I64(original));
}

// ============================================================================
// Type-matrix coverage: every applicable coder × every applicable integer
// type with edge-case inputs (empty, single, type-extremes). Closes the
// "generic code path probably works but wasn't individually tested" gap.
// ============================================================================

#[test]
fn leb128_roundtrips_all_integer_widths() {
    macro_rules! check {
        ($dt:ident, $variant:ident, $ty:ty, $vals:expr) => {{
            let original: Vec<$ty> = $vals;
            let pipeline =
                Pipeline::new(DataType::$dt, vec![nb(Leb128::new(DataType::$dt).unwrap())])
                    .unwrap();
            let encoded = pipeline
                .encode(ColumnData::$variant(original.clone()))
                .unwrap();
            let decoded = pipeline.decode(encoded).unwrap();
            assert_eq!(
                decoded,
                ColumnData::$variant(original),
                "leb128 mismatch for {:?}",
                DataType::$dt
            );
        }};
    }
    check!(I8, I8, i8, vec![i8::MIN, -1, 0, 1, i8::MAX]);
    check!(I16, I16, i16, vec![i16::MIN, -1000, 0, 1000, i16::MAX]);
    check!(I32, I32, i32, vec![i32::MIN, -1, 0, 1, 1_000_000, i32::MAX]);
    check!(I64, I64, i64, vec![i64::MIN, -1, 0, 1, i64::MAX]);
    check!(U8, U8, u8, (0u8..=255).collect());
    check!(U16, U16, u16, vec![0, 127, 128, 16383, 16384, u16::MAX]);
    check!(U32, U32, u32, vec![0, 128, 16384, 2_000_000, u32::MAX]);
    check!(U64, U64, u64, vec![0, 1, 128, 16384, u64::MAX]);
}

#[test]
fn rle_roundtrips_all_integer_widths() {
    macro_rules! check {
        ($dt:ident, $variant:ident, $ty:ty, $vals:expr) => {{
            let original: Vec<$ty> = $vals;
            let pipeline =
                Pipeline::new(DataType::$dt, vec![nb(Rle::new(DataType::$dt).unwrap())]).unwrap();
            let encoded = pipeline
                .encode(ColumnData::$variant(original.clone()))
                .unwrap();
            let decoded = pipeline.decode(encoded).unwrap();
            assert_eq!(
                decoded,
                ColumnData::$variant(original),
                "rle mismatch for {:?}",
                DataType::$dt
            );
        }};
    }
    // Stay under each type's max run-length (e.g. i8 max=127, u8 max=255).
    check!(I8, I8, i8, vec![7, 7, 7, -3, -3, 42]);
    check!(I16, I16, i16, vec![1000; 500]);
    check!(I32, I32, i32, {
        let mut v = Vec::new();
        v.extend(std::iter::repeat_n(-5, 200));
        v.extend(std::iter::repeat_n(7, 300));
        v
    });
    check!(I64, I64, i64, vec![42; 1000]);
    check!(U8, U8, u8, vec![0u8; 200]);
    check!(U16, U16, u16, vec![u16::MAX; 500]);
    check!(U32, U32, u32, vec![0xdead_beef; 2000]);
    check!(U64, U64, u64, vec![u64::MAX; 1000]);
}

#[test]
fn deltamin_roundtrips_all_integer_widths() {
    macro_rules! check {
        ($dt:ident, $variant:ident, $ty:ty, $vals:expr) => {{
            let original: Vec<$ty> = $vals;
            let pipeline = Pipeline::new(
                DataType::$dt,
                vec![blk(DeltaMin::new(DataType::$dt).unwrap())],
            )
            .unwrap();
            let encoded = pipeline
                .encode(ColumnData::$variant(original.clone()))
                .unwrap();
            let decoded = pipeline.decode(encoded).unwrap();
            assert_eq!(
                decoded,
                ColumnData::$variant(original),
                "deltamin mismatch for {:?}",
                DataType::$dt
            );
        }};
    }
    check!(I8, I8, i8, vec![-10, -5, 0, 5, 10]);
    check!(I16, I16, i16, vec![i16::MIN, 0, 100, i16::MAX]);
    check!(I32, I32, i32, vec![-1_000_000, 0, 1, 1_000_000]);
    check!(I64, I64, i64, vec![i64::MIN, -1, 0, 1, i64::MAX]);
    check!(U8, U8, u8, vec![10, 20, 30, 255]);
    check!(U16, U16, u16, vec![100, 200, u16::MAX]);
    check!(U32, U32, u32, vec![0, 1, u32::MAX]);
    check!(U64, U64, u64, vec![0, 1, u64::MAX]);
    // Empty and single-element on i32 as a spot check.
    let pipe = Pipeline::new(
        DataType::I32,
        vec![blk(DeltaMin::new(DataType::I32).unwrap())],
    )
    .unwrap();
    let enc = pipe.encode(ColumnData::I32(Vec::new())).unwrap();
    assert_eq!(pipe.decode(enc).unwrap(), ColumnData::I32(Vec::new()));
    let enc = pipe.encode(ColumnData::I32(vec![42])).unwrap();
    assert_eq!(pipe.decode(enc).unwrap(), ColumnData::I32(vec![42]));
}

#[test]
fn delta_of_delta_roundtrips_all_integer_widths() {
    macro_rules! check {
        ($dt:ident, $variant:ident, $ty:ty, $vals:expr) => {{
            let original: Vec<$ty> = $vals;
            let pipeline = Pipeline::new(
                DataType::$dt,
                vec![nb(DeltaOfDelta::new(DataType::$dt).unwrap())],
            )
            .unwrap();
            let encoded = pipeline
                .encode(ColumnData::$variant(original.clone()))
                .unwrap();
            let decoded = pipeline.decode(encoded).unwrap();
            assert_eq!(
                decoded,
                ColumnData::$variant(original),
                "dod mismatch for {:?}",
                DataType::$dt
            );
        }};
    }
    check!(I8, I8, i8, vec![0, 1, 2, 4, 7, -3, i8::MIN, i8::MAX]);
    check!(I16, I16, i16, (0..500i16).collect());
    check!(I32, I32, i32, vec![i32::MIN, -1, 0, 1, i32::MAX]);
    check!(I64, I64, i64, vec![i64::MIN, -1, 0, 1, i64::MAX]);
    check!(U8, U8, u8, vec![0, 1, 10, 255, 0, 1]);
    check!(U16, U16, u16, (0u16..1000).collect());
    check!(U32, U32, u32, vec![0, u32::MAX, 100, u32::MAX - 1]);
    check!(U64, U64, u64, vec![0, 1, u64::MAX, 2]);
}

#[test]
fn bitpack_fixed_roundtrips_all_integer_widths() {
    // Non-negative values that fit in a narrow width; check every type.
    macro_rules! check {
        ($dt:ident, $variant:ident, $ty:ty, $width:expr, $vals:expr) => {{
            let original: Vec<$ty> = $vals;
            let pipeline = Pipeline::new(
                DataType::$dt,
                vec![nb(BitpackFixed::new(DataType::$dt, $width).unwrap())],
            )
            .unwrap();
            let encoded = pipeline
                .encode(ColumnData::$variant(original.clone()))
                .unwrap();
            let decoded = pipeline.decode(encoded).unwrap();
            assert_eq!(
                decoded,
                ColumnData::$variant(original),
                "bitpack_fixed mismatch for {:?} width={}",
                DataType::$dt,
                $width
            );
        }};
    }
    check!(I8, I8, i8, 4, vec![0, 1, 2, 15]);
    check!(I16, I16, i16, 10, (0..1000i16).collect());
    check!(I32, I32, i32, 20, vec![0, 1, 1_000_000]);
    check!(I64, I64, i64, 40, vec![0, 1_000_000_000_000]);
    check!(U8, U8, u8, 8, (0u8..=255).collect());
    check!(U16, U16, u16, 16, vec![0, 1, u16::MAX]);
    check!(U32, U32, u32, 32, vec![0, u32::MAX / 2, u32::MAX]);
    check!(U64, U64, u64, 64, vec![0, 1, u64::MAX]);
}

#[test]
fn bitpack_auto_roundtrips_all_integer_widths() {
    macro_rules! check {
        ($dt:ident, $variant:ident, $ty:ty, $vals:expr) => {{
            let original: Vec<$ty> = $vals;
            let pipeline = Pipeline::new(
                DataType::$dt,
                vec![blk(BitpackAuto::new(DataType::$dt).unwrap())],
            )
            .unwrap();
            let encoded = pipeline
                .encode(ColumnData::$variant(original.clone()))
                .unwrap();
            let decoded = pipeline.decode(encoded).unwrap();
            assert_eq!(
                decoded,
                ColumnData::$variant(original),
                "bitpack_auto mismatch for {:?}",
                DataType::$dt
            );
        }};
    }
    // All input non-negative (bitpack requires this).
    check!(I8, I8, i8, vec![0, 1, 2, 7, 127]);
    check!(I16, I16, i16, vec![0, 100, 1000, i16::MAX]);
    check!(I32, I32, i32, vec![0, 1000, 1_000_000, i32::MAX]);
    check!(I64, I64, i64, vec![0, 1, 1_000_000_000, i64::MAX]);
    check!(U8, U8, u8, (0u8..=255).collect());
    check!(U16, U16, u16, vec![0, 1, 16383, u16::MAX]);
    check!(U32, U32, u32, vec![0, u32::MAX]);
    check!(U64, U64, u64, vec![0, 1, 2, u64::MAX]); // forces width=64
}

// ============================================================================
// AccessPattern propagation
// ============================================================================

#[test]
fn access_pattern_combine_rules() {
    use AccessPattern::*;
    assert_eq!(RandomAccess.combine(RandomAccess), RandomAccess);
    assert_eq!(RandomAccess.combine(SequentialOnly), SequentialOnly);
    assert_eq!(SequentialOnly.combine(RandomAccess), SequentialOnly);
    assert_eq!(SequentialOnly.combine(SequentialOnly), SequentialOnly);
}

#[test]
fn access_pattern_random_on_pure_ra_pipelines() {
    // bitpack_fixed alone — RA
    let p = Pipeline::new(
        DataType::U32,
        vec![nb(BitpackFixed::new(DataType::U32, 10).unwrap())],
    )
    .unwrap();
    assert_eq!(p.access_pattern(), AccessPattern::RandomAccess);

    // bitpack_auto alone — RA
    let p = Pipeline::new(
        DataType::U32,
        vec![blk(BitpackAuto::new(DataType::U32).unwrap())],
    )
    .unwrap();
    assert_eq!(p.access_pattern(), AccessPattern::RandomAccess);

    // elias_fano alone — RA
    let p = Pipeline::new(
        DataType::U32,
        vec![blk(EliasFano::new(DataType::U32).unwrap())],
    )
    .unwrap();
    assert_eq!(p.access_pattern(), AccessPattern::RandomAccess);
}

#[test]
fn access_pattern_sequential_if_any_stage_is_sequential() {
    // delta downgrades to SEQ even if followed by a RA coder
    let p = Pipeline::new(
        DataType::U32,
        vec![
            nb(Delta::new(DataType::U32).unwrap()),
            nb(BitpackFixed::new(DataType::U32, 10).unwrap()),
        ],
    )
    .unwrap();
    assert_eq!(p.access_pattern(), AccessPattern::SequentialOnly);

    // zstd tail downgrades a previously RA chain to SEQ
    let p = Pipeline::new(
        DataType::U32,
        vec![
            nb(BitpackFixed::new(DataType::U32, 10).unwrap()),
            blk(Zstd::default()),
        ],
    )
    .unwrap();
    assert_eq!(p.access_pattern(), AccessPattern::SequentialOnly);

    // Plain SEQ pipeline stays SEQ
    let p = Pipeline::new(
        DataType::I64,
        vec![
            nb(Delta::new(DataType::I64).unwrap()),
            nb(Leb128::new(DataType::I64).unwrap()),
            blk(Zstd::default()),
        ],
    )
    .unwrap();
    assert_eq!(p.access_pattern(), AccessPattern::SequentialOnly);
}
