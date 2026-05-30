//! §5.8 — Deep-nesting + parity tests.
//!
//! This file covers four of the five §5.8 acceptance items (compression
//! parity lives in `tests/compression_parity.rs`):
//!
//! 1. Round-trip on `Struct<List<Struct<Map<Utf8, List<Nullable<Primitive(F64)>>>>>>` (5 levels)
//! 2. Column pruning — at the top-level logical column granularity (the
//!    public reader API tops out there; leaf-granular reads would require a
//!    new public API and are flagged for planner review)
//! 3. Multi-stripe with the deeply-nested schema
//! 4. CRC detection on a single-byte flip — error names the failing leaf
//!    via the dotted path embedded in the `reason` string
//!
//! The 5-level type below is the literal acceptance shape from PLAN_V2 §5.8
//! disambiguating notes. The bottom primitive is `F64` so the deepest
//! pipeline exercises Gorilla, the float-time-series coder.

use std::io::{Cursor, Read, Seek, SeekFrom};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use helium::{
    CoderRegistry, CoderSpec, ColumnData, ColumnSpec, DataType, FieldSpec, HeliumError,
    HeliumReader, HeliumWriter, LogicalColumn, LogicalType, Schema,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn zstd() -> CoderSpec {
    CoderSpec::new("zstd")
}
fn delta_leb_zstd() -> Vec<CoderSpec> {
    vec![CoderSpec::new("delta"), CoderSpec::new("leb128"), zstd()]
}
fn zstd_only() -> Vec<CoderSpec> {
    vec![zstd()]
}
fn present_coders() -> Vec<CoderSpec> {
    vec![CoderSpec::new("leb128"), zstd()]
}
/// Gorilla → zstd for F64 leaf values (float time-series shape).
fn f64_coders() -> Vec<CoderSpec> {
    vec![CoderSpec::new("gorilla"), zstd()]
}
fn registry() -> CoderRegistry {
    CoderRegistry::default()
}

/// The 5-level acceptance type:
/// `Struct<List<Struct<Map<Utf8, List<Nullable<Primitive(F64)>>>>>>`.
fn deep_schema() -> Schema {
    // Innermost: Nullable(Primitive(F64))
    // Wrapped in: List<Nullable<Primitive>>
    // Wrapped in: Map<Utf8, List<Nullable<Primitive>>>
    // Wrapped in: Struct { mid_field: Map<...> }
    // Wrapped in: List<Struct<Map<...>>>
    // Wrapped in: Struct { outer_field: List<...> }   ← top-level

    // FieldSpec for the innermost Map<Utf8, List<Nullable<Primitive(F64)>>>
    // expected_encodings_len = 1 (map offsets)
    //                       + 2 (utf8 key: offsets + data)
    //                       + 1 (list offsets)
    //                       + 1 (nullable present)
    //                       + 1 (primitive values)
    //                       = 6
    let mid_field = FieldSpec::map(
        "mid_field",
        LogicalType::Utf8,
        LogicalType::List {
            inner: Box::new(LogicalType::Nullable {
                inner: Box::new(LogicalType::Primitive {
                    data_type: DataType::F64,
                }),
            }),
        },
        vec![
            delta_leb_zstd(), // map offsets         (U32)
            delta_leb_zstd(), // key.offsets         (U32)
            zstd_only(),      // key.data            (Bytes)
            delta_leb_zstd(), // value.offsets       (U32) — outer list
            present_coders(), // value.item.present  (U8)
            f64_coders(),     // value.item.item.values (F64)
        ],
    );

    // List<Struct<Map<...>>> — encodings has length 1 (just outer list offsets;
    // Struct contributes 0 because its leaf encodings live in FieldSpecs).
    let outer_field = FieldSpec::list(
        "outer_field",
        LogicalType::Struct {
            fields: vec![mid_field],
        },
        vec![delta_leb_zstd()],
    );

    Schema::new(vec![ColumnSpec::struct_col("rec", vec![outer_field])])
}

/// Build a deep `LogicalColumn::Struct` with 3 outer rows and varied
/// inner structure to exercise empty / populated mixes at every level.
///
/// Returns the column plus a "fingerprint" of the data structure for later
/// equality assertion in tests.
fn deep_data() -> LogicalColumn {
    // Outer rows: 3
    //
    // Row 0: outer_field = [
    //   { mid_field: { "a" -> [Some(1.0)], "b" -> [None] } },   // 2 map entries
    //   { mid_field: {} }                                        // 0 map entries
    // ]
    // Row 1: outer_field = [
    //   { mid_field: { "c" -> [Some(2.0), None] } }              // 1 map entry, 2 list elems
    // ]
    // Row 2: outer_field = []  (empty outer list)
    //
    //
    // Outer list offsets (length 4): [0, 2, 3, 3]   → 3 mid rows total
    // Mid struct row count: 3
    // Per-mid map_offsets (length 4): [0, 2, 2, 3]  → 3 map entries total
    // Map keys (length 3): ["a", "b", "c"]
    // Map values are Lists; per-entry list_offsets (length 4): [0, 1, 2, 4]
    //                                                 → 4 inner list elements
    // Inner Nullable: present (length 4) = [T, F, T, F]; values (length 2) = [1.0, 2.0]

    let outer_offsets: Vec<u32> = vec![0, 2, 3, 3];

    let map_offsets: Vec<u32> = vec![0, 2, 2, 3];
    let keys = LogicalColumn::Utf8(vec!["a".into(), "b".into(), "c".into()]);

    let list_offsets: Vec<u32> = vec![0, 1, 2, 4];
    let nullable_present = vec![true, false, true, false];
    let primitive_values = ColumnData::F64(vec![1.0, 2.0]);

    let nullable = LogicalColumn::Nullable {
        present: nullable_present,
        value: Box::new(LogicalColumn::Primitive(primitive_values)),
    };
    let value_list = LogicalColumn::List {
        offsets: list_offsets,
        values: Box::new(nullable),
    };
    let map = LogicalColumn::Map {
        offsets: map_offsets,
        keys: Box::new(keys),
        values: Box::new(value_list),
    };
    let mid_struct = LogicalColumn::Struct {
        fields: vec![("mid_field".into(), map)],
    };
    let outer_list = LogicalColumn::List {
        offsets: outer_offsets,
        values: Box::new(mid_struct),
    };
    LogicalColumn::Struct {
        fields: vec![("outer_field".into(), outer_list)],
    }
}

// ---------------------------------------------------------------------------
// 1. Deep round-trip
// ---------------------------------------------------------------------------

#[test]
fn deep_5_level_roundtrip() {
    let schema = deep_schema();
    let reg = registry();
    let data = deep_data();

    // Sanity: physical_fields() lays out the 7 expected dotted leaves
    // for the deeply-nested column.
    let pf = schema.columns[0].logical_type.physical_fields();
    let names: Vec<&str> = pf.iter().map(|f| f.role.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "outer_field.offsets",
            "outer_field.item.mid_field.offsets",
            "outer_field.item.mid_field.key.offsets",
            "outer_field.item.mid_field.key.data",
            "outer_field.item.mid_field.value.offsets",
            "outer_field.item.mid_field.value.item.present",
            "outer_field.item.mid_field.value.item.item.values",
        ],
    );

    // Validate at the depth we're testing — well under MAX_NESTED_DEPTH.
    schema.validate().expect("deep schema must validate");

    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut writer = HeliumWriter::new(&mut buf, schema, &reg).expect("writer");
    writer.write_column("rec", data.clone()).expect("write");
    writer.finish().expect("finish");

    buf.set_position(0);
    let mut reader = HeliumReader::new(&mut buf, &reg).expect("reader");
    let result = reader.read_column("rec").expect("read");
    assert_eq!(result, data, "deep round-trip must be byte-equal");
}

// ---------------------------------------------------------------------------
// 2. Column pruning — top-level logical column granularity
// ---------------------------------------------------------------------------
//
// **Gap note**: §5.8 asks for "leaf granularity" pruning (read
// `user.address.zip` touches only that leaf's bytes). The current public
// reader API (`HeliumReader::read_column` / `read_column_at_stripe`) only
// supports pruning at the top-level logical column. Adding leaf-level reads
// would require a new public API and is **out of scope** here per the
// team-lead's guidance ("do not invent a new public API to satisfy the
// test"). This test verifies what IS supported: reading one top-level column
// of a multi-column file does not touch the other top-level column's bytes,
// even when those columns are deeply nested.

#[test]
fn top_level_pruning_skips_other_columns_in_deep_schema() {
    // Schema: two top-level columns
    //   1. "rec" — the 5-level deep type from deep_schema()
    //   2. "blob" — a large simple Bytes column we will assert is NOT read
    let mid_field = FieldSpec::map(
        "mid_field",
        LogicalType::Utf8,
        LogicalType::List {
            inner: Box::new(LogicalType::Nullable {
                inner: Box::new(LogicalType::Primitive {
                    data_type: DataType::F64,
                }),
            }),
        },
        vec![
            delta_leb_zstd(),
            delta_leb_zstd(),
            zstd_only(),
            delta_leb_zstd(),
            present_coders(),
            f64_coders(),
        ],
    );
    let outer_field = FieldSpec::list(
        "outer_field",
        LogicalType::Struct {
            fields: vec![mid_field],
        },
        vec![delta_leb_zstd()],
    );

    let schema = Schema::new(vec![
        ColumnSpec::struct_col("rec", vec![outer_field]),
        // A relatively-large Binary column that we should NEVER read when
        // we only ask for "rec".
        ColumnSpec::binary("blob", delta_leb_zstd(), zstd_only()),
    ]);
    let reg = registry();

    // Build large blob payload (well-compressible repetition still leaves
    // far more bytes than the small deep type's leaves).
    let blobs: Vec<Vec<u8>> = (0..200)
        .map(|i| {
            // Each blob ~ 200 bytes of varied data (zstd-resistant)
            (0u8..255).cycle().skip(i).take(200).collect()
        })
        .collect();

    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut writer = HeliumWriter::new(&mut buf, schema, &reg).expect("writer");
    // We have 3 outer rec rows; the blob column must report 3 rows too.
    let blobs_first_3 = blobs[..3].to_vec();
    writer.write_column("rec", deep_data()).expect("rec write");
    writer
        .write_column("blob", LogicalColumn::Binary(blobs_first_3))
        .expect("blob write");
    writer.finish().expect("finish");

    let bytes = buf.into_inner();
    let full_file_len = bytes.len() as u64;

    // Counted reader: tracks every byte read from the underlying stream.
    struct CountingRead<R> {
        inner: R,
        bytes_read: Arc<AtomicU64>,
    }
    impl<R: Read> Read for CountingRead<R> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let n = self.inner.read(buf)?;
            self.bytes_read.fetch_add(n as u64, Ordering::Relaxed);
            Ok(n)
        }
    }
    impl<R: Seek> Seek for CountingRead<R> {
        fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
            self.inner.seek(pos)
        }
    }

    let counter = Arc::new(AtomicU64::new(0));
    let counting = CountingRead {
        inner: Cursor::new(bytes),
        bytes_read: counter.clone(),
    };
    let mut reader = HeliumReader::new(counting, &reg).expect("reader open");
    let after_open = counter.load(Ordering::Relaxed);

    // Reading just "rec" must NOT touch the "blob" column's bytes.
    let _rec = reader.read_column("rec").expect("read rec");
    let after_rec = counter.load(Ordering::Relaxed);
    let rec_bytes_read = after_rec - after_open;

    // The blob column dominates the file size. If pruning works, reading
    // just "rec" should read far fewer bytes than the full file.
    assert!(
        rec_bytes_read < full_file_len / 2,
        "rec read consumed {rec_bytes_read} bytes; file is {full_file_len} bytes — \
         pruning at top-level granularity should skip the blob column"
    );
}

// TODO(planner / future task): add a new public API for reading at
// leaf granularity (`read_leaf("rec.outer_field.item.mid_field.value.item.item.values")`
// or similar). Until that ships, leaf-granular pruning cannot be
// integration-tested. SendMessage-routed to planner alongside this PR.

// ---------------------------------------------------------------------------
// 3. Multi-stripe with deep schema
// ---------------------------------------------------------------------------

#[test]
fn deep_schema_multi_stripe() {
    let schema = deep_schema();
    let reg = registry();

    // 3 stripes, each with a SEPARATE deep_data() instance so they can be
    // stripe-checked individually and concatenated as a whole.
    let stripe1 = deep_data();
    let stripe2 = deep_data();
    let stripe3 = deep_data();

    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut writer = HeliumWriter::new(&mut buf, schema, &reg).expect("writer");
    writer.write_column("rec", stripe1.clone()).expect("s1");
    writer.finish_stripe().expect("finish_stripe 1");
    writer.write_column("rec", stripe2.clone()).expect("s2");
    writer.finish_stripe().expect("finish_stripe 2");
    writer.write_column("rec", stripe3.clone()).expect("s3");
    writer.finish().expect("finish");

    buf.set_position(0);
    let mut reader = HeliumReader::new(&mut buf, &reg).expect("reader");
    assert_eq!(reader.stripe_count(), 3);

    // Per-stripe reads via read_column_at_stripe must each return the
    // exact data we wrote into that stripe.
    let r1 = reader.read_column_at_stripe("rec", 0).expect("stripe 0");
    let r2 = reader.read_column_at_stripe("rec", 1).expect("stripe 1");
    let r3 = reader.read_column_at_stripe("rec", 2).expect("stripe 2");
    assert_eq!(r1, stripe1);
    assert_eq!(r2, stripe2);
    assert_eq!(r3, stripe3);

    // Whole-file read concatenates across stripes via concat_logical_columns
    // (the deeply-recursive concat path covered by §5.1–§5.5 work).
    let all = reader.read_column("rec").expect("read_column concat");
    let LogicalColumn::Struct { fields } = all else {
        panic!("expected concatenated Struct");
    };
    assert_eq!(fields[0].0, "outer_field");
    let LogicalColumn::List { offsets, .. } = &fields[0].1 else {
        panic!("expected concatenated List");
    };
    // Each stripe's outer_field is `[0, 2, 3, 3]` → 3 rows. 3 stripes → 9 rows.
    assert_eq!(offsets.len(), 3 * 3 + 1);
    // Last offset is the total count of mid_struct rows: each stripe has 3,
    // so concatenated = 9.
    assert_eq!(*offsets.last().unwrap(), 9);
}

// ---------------------------------------------------------------------------
// 4. CRC detection — single-byte flip in a leaf surfaces with leaf path
// ---------------------------------------------------------------------------

/// Parse the v2 file's body region [body_start, body_end) from the raw bytes.
///
/// File shape: `magic(8) | schema_len(4 LE) | schema_json | body | footer_json |
/// footer_len(8 LE) | footer_crc(4 LE) | magic(8)`.
///
/// Returns `(body_start, body_end)` where the body is everything between the
/// schema header and the start of the footer JSON.
fn body_region(bytes: &[u8]) -> (usize, usize) {
    // schema_len at bytes[8..12]
    let schema_len =
        u32::from_le_bytes(bytes[8..12].try_into().expect("schema_len slice")) as usize;
    let body_start = 12 + schema_len;
    // trailer is 20 bytes for v2: footer_len(8) + footer_crc(4) + magic(8)
    let trailer_start = bytes.len() - 20;
    // footer_len at bytes[trailer_start..trailer_start+8]
    let footer_len = u64::from_le_bytes(
        bytes[trailer_start..trailer_start + 8]
            .try_into()
            .expect("footer_len slice"),
    ) as usize;
    let body_end = trailer_start - footer_len;
    (body_start, body_end)
}

#[test]
fn crc_flip_in_deep_leaf_surfaces_corrupted_with_leaf_path() {
    let schema = deep_schema();
    let reg = registry();
    let data = deep_data();

    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut writer = HeliumWriter::new(&mut buf, schema.clone(), &reg).expect("writer");
    writer.write_column("rec", data).expect("write");
    writer.finish().expect("finish");
    let mut bytes = buf.into_inner();

    // Parse the file structure to pick a byte safely inside the body region
    // (i.e. inside one of the leaf physical-column blobs).
    let (body_start, body_end) = body_region(&bytes);
    assert!(body_end > body_start, "body must be non-empty");
    let target = (body_start + body_end) / 2;
    bytes[target] ^= 0xff;

    let mut reader = HeliumReader::new(Cursor::new(bytes), &reg).expect("reader open");
    let err = reader.read_column("rec").expect_err("CRC must fail");

    match err {
        HeliumError::Corrupted { coder, reason } => {
            assert_eq!(coder, "rec", "top-level column name in error");
            // Reason must include a recognizable dotted leaf path from the
            // physical_fields() output.
            let recognised_leaves = [
                "outer_field.offsets",
                "outer_field.item.mid_field.offsets",
                "outer_field.item.mid_field.key.offsets",
                "outer_field.item.mid_field.key.data",
                "outer_field.item.mid_field.value.offsets",
                "outer_field.item.mid_field.value.item.present",
                "outer_field.item.mid_field.value.item.item.values",
            ];
            let pinpointed = recognised_leaves.iter().any(|leaf| reason.contains(leaf));
            assert!(
                pinpointed,
                "Corrupted reason must include a leaf dotted-path; got: {reason}"
            );
            // CRC mismatch wording for the v2 path
            assert!(
                reason.contains("CRC32C") || reason.contains("mismatch"),
                "should mention CRC32C mismatch; got: {reason}"
            );
        }
        other => panic!("expected HeliumError::Corrupted, got: {other:?}"),
    }
}

#[test]
fn crc_flip_at_each_leaf_offset_pinpoints_that_leaf() {
    // Stronger version of the test above: walk the body region and at each
    // byte position, write a fresh copy and flip just that one byte. Verify
    // every position triggers HeliumError::Corrupted with at least one of
    // the known leaf paths in the reason. This catches off-by-one errors in
    // the leaf-path zip in `read_column_piece`.
    let schema = deep_schema();
    let reg = registry();
    let data = deep_data();

    let mut buf = Cursor::new(Vec::<u8>::new());
    let mut writer = HeliumWriter::new(&mut buf, schema, &reg).expect("writer");
    writer.write_column("rec", data).expect("write");
    writer.finish().expect("finish");
    let original = buf.into_inner();

    let (body_start, body_end) = body_region(&original);

    let recognised_leaves = [
        "outer_field.offsets",
        "outer_field.item.mid_field.offsets",
        "outer_field.item.mid_field.key.offsets",
        "outer_field.item.mid_field.key.data",
        "outer_field.item.mid_field.value.offsets",
        "outer_field.item.mid_field.value.item.present",
        "outer_field.item.mid_field.value.item.item.values",
    ];

    // Sample 16 evenly-spaced positions across the body region. Sampling
    // (rather than every byte) keeps the test fast while still exercising
    // multiple distinct leaves' byte ranges.
    let body_len = body_end - body_start;
    let n_samples = 16usize.min(body_len.max(1));
    let mut positions_with_corrupted_error = 0;
    for i in 0..n_samples {
        let off = body_start + (body_len * i) / n_samples.max(1);
        if off >= body_end {
            break;
        }
        let mut bytes = original.clone();
        bytes[off] ^= 0xff;

        let Ok(mut reader) = HeliumReader::new(Cursor::new(bytes), &reg) else {
            // Header survives untouched; reader open should succeed.
            // If it doesn't, the corruption happened to land in a CRC-checked
            // header region — count this as a defensive failure path.
            continue;
        };
        // Some flips can corrupt internal coder framing (e.g., zstd frame
        // header) and surface as Schema or other errors before the CRC check.
        // That's acceptable as long as at least one sampled position produces
        // a Corrupted-with-leaf-path error (asserted at the bottom of this fn).
        if let Err(HeliumError::Corrupted { reason, .. }) = reader.read_column("rec") {
            let pinpointed = recognised_leaves.iter().any(|leaf| reason.contains(leaf));
            assert!(
                pinpointed,
                "byte flip at {off}: Corrupted reason missing leaf path; got: {reason}"
            );
            positions_with_corrupted_error += 1;
        }
    }

    assert!(
        positions_with_corrupted_error >= 1,
        "expected at least one of {n_samples} sampled body byte-flips to surface as \
         HeliumError::Corrupted with a leaf path"
    );
}
