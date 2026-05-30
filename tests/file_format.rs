use std::io::{Cursor, Read, Seek, SeekFrom, Write};

use helium::{
    CoderRegistry, CoderSpec, ColumnData, ColumnSpec, DataType, HeliumError, HeliumReader,
    HeliumWriter, LogicalColumn, LogicalType, MAGIC, Schema,
};

fn timestamps_column() -> ColumnSpec {
    ColumnSpec::primitive(
        "ts",
        DataType::I64,
        vec![
            CoderSpec::new("delta"),
            CoderSpec::new("leb128"),
            CoderSpec::new("zstd").with_param("level", 5),
        ],
    )
}

fn rsrp_column() -> ColumnSpec {
    // Signed deltamin → bitpack_auto → zstd — composition enabled by
    // bitpack accepting non-negative signed input.
    ColumnSpec::primitive(
        "rsrp_abs",
        DataType::I64,
        vec![
            CoderSpec::new("deltamin"),
            CoderSpec::new("bitpack_auto"),
            CoderSpec::new("zstd"),
        ],
    )
}

fn value_column() -> ColumnSpec {
    ColumnSpec::primitive("value", DataType::I64, vec![CoderSpec::new("pcodec")])
}

fn tag_column() -> ColumnSpec {
    ColumnSpec::primitive(
        "tag",
        DataType::I64,
        vec![
            CoderSpec::new("rle"),
            CoderSpec::new("leb128"),
            CoderSpec::new("lz4"),
        ],
    )
}

fn name_column() -> ColumnSpec {
    ColumnSpec::utf8(
        "name",
        vec![
            CoderSpec::new("delta"),
            CoderSpec::new("leb128"),
            CoderSpec::new("zstd"),
        ],
        vec![CoderSpec::new("zstd")],
    )
}

fn nullable_weight_column() -> ColumnSpec {
    ColumnSpec::nullable_prim(
        "weight",
        DataType::F64,
        vec![
            CoderSpec::new("rle"),
            CoderSpec::new("bitpack_auto"),
            CoderSpec::new("zstd"),
        ],
        vec![CoderSpec::new("gorilla"), CoderSpec::new("zstd")],
    )
}

fn tags_array_column() -> ColumnSpec {
    ColumnSpec::array_of(
        "tags",
        DataType::I32,
        vec![
            CoderSpec::new("delta"),
            CoderSpec::new("leb128"),
            CoderSpec::new("zstd"),
        ],
        vec![CoderSpec::new("pcodec")],
    )
}

fn full_sample_schema() -> Schema {
    Schema::new(vec![
        timestamps_column(),
        rsrp_column(),
        value_column(),
        tag_column(),
        name_column(),
        nullable_weight_column(),
        tags_array_column(),
    ])
}

struct Sample {
    ts: Vec<i64>,
    rsrp: Vec<i64>,
    value: Vec<i64>,
    tag: Vec<i64>,
    name: Vec<String>,
    weight_present: Vec<bool>,
    weight_values: Vec<f64>,
    tags_offsets: Vec<u32>,
    tags_values: Vec<i32>,
}

fn make_sample(n: usize) -> Sample {
    let ts: Vec<i64> = (0..n).map(|i| 1_700_000_000 + i as i64 * 30).collect();
    let rsrp: Vec<i64> = (0..n).map(|i| 40 + (i as i64 % 11)).collect();
    let value: Vec<i64> = (0..n).map(|i| (i as i64 * 7) - (i as i64 / 3)).collect();
    let tag: Vec<i64> = (0..n).map(|i| (i as i64 / 100) % 4).collect();
    let name: Vec<String> = (0..n).map(|i| format!("user_{}", i % 97)).collect();

    let weight_present: Vec<bool> = (0..n).map(|i| i % 3 != 0).collect();
    let weight_values: Vec<f64> = weight_present
        .iter()
        .enumerate()
        .filter_map(|(i, &p)| {
            if p {
                Some(70.0 + (i as f64 * 0.01).sin())
            } else {
                None
            }
        })
        .collect();

    // Build Array<I32> offsets + values: each row has i%5 items.
    let mut tags_offsets = Vec::with_capacity(n + 1);
    tags_offsets.push(0u32);
    let mut tags_values = Vec::new();
    for i in 0..n {
        let k = i % 5;
        for j in 0..k {
            tags_values.push((i * 10 + j) as i32);
        }
        tags_offsets.push(tags_values.len() as u32);
    }

    Sample {
        ts,
        rsrp,
        value,
        tag,
        name,
        weight_present,
        weight_values,
        tags_offsets,
        tags_values,
    }
}

fn write_sample_file<W: Write + Seek>(sink: W, schema: Schema, s: &Sample) -> helium::Result<W> {
    let registry = CoderRegistry::default();
    let mut writer = HeliumWriter::new(sink, schema, &registry)?;
    writer.write_column(
        "ts",
        LogicalColumn::Primitive(ColumnData::I64(s.ts.clone())),
    )?;
    writer.write_column(
        "rsrp_abs",
        LogicalColumn::Primitive(ColumnData::I64(s.rsrp.clone())),
    )?;
    writer.write_column(
        "value",
        LogicalColumn::Primitive(ColumnData::I64(s.value.clone())),
    )?;
    writer.write_column(
        "tag",
        LogicalColumn::Primitive(ColumnData::I64(s.tag.clone())),
    )?;
    writer.write_column("name", LogicalColumn::Utf8(s.name.clone()))?;
    writer.write_column(
        "weight",
        LogicalColumn::NullablePrim {
            present: s.weight_present.clone(),
            values: ColumnData::F64(s.weight_values.clone()),
        },
    )?;
    writer.write_column(
        "tags",
        LogicalColumn::ArrayOf {
            offsets: s.tags_offsets.clone(),
            values: ColumnData::I32(s.tags_values.clone()),
        },
    )?;
    writer.finish()
}

// ---------------------------------------------------------------------------
// Full multi-column + multi-logical-type round-trip
// ---------------------------------------------------------------------------

#[test]
fn multi_column_roundtrip_in_memory() {
    let n = 1_000;
    let s = make_sample(n);

    let cursor = Cursor::new(Vec::new());
    let written = write_sample_file(cursor, full_sample_schema(), &s).unwrap();
    let bytes = written.into_inner();

    assert_eq!(&bytes[..8], MAGIC);
    assert_eq!(&bytes[bytes.len() - 8..], MAGIC);

    let registry = CoderRegistry::default();
    let mut reader = HeliumReader::new(Cursor::new(bytes), &registry).unwrap();
    assert_eq!(reader.row_count(), n as u64);

    let all = reader.read_all().unwrap();

    assert_eq!(
        all["ts"],
        LogicalColumn::Primitive(ColumnData::I64(s.ts.clone()))
    );
    assert_eq!(
        all["rsrp_abs"],
        LogicalColumn::Primitive(ColumnData::I64(s.rsrp.clone()))
    );
    assert_eq!(
        all["value"],
        LogicalColumn::Primitive(ColumnData::I64(s.value.clone()))
    );
    assert_eq!(
        all["tag"],
        LogicalColumn::Primitive(ColumnData::I64(s.tag.clone()))
    );
    assert_eq!(all["name"], LogicalColumn::Utf8(s.name.clone()));
    assert_eq!(
        all["weight"],
        LogicalColumn::NullablePrim {
            present: s.weight_present.clone(),
            values: ColumnData::F64(s.weight_values.clone()),
        }
    );
    assert_eq!(
        all["tags"],
        LogicalColumn::ArrayOf {
            offsets: s.tags_offsets.clone(),
            values: ColumnData::I32(s.tags_values.clone()),
        }
    );
}

#[test]
fn multi_column_roundtrip_on_disk() {
    let n = 500;
    let s = make_sample(n);

    let tmp = tempfile::NamedTempFile::new().unwrap();
    {
        let f = std::fs::OpenOptions::new()
            .write(true)
            .read(true)
            .truncate(true)
            .open(tmp.path())
            .unwrap();
        write_sample_file(f, full_sample_schema(), &s).unwrap();
    }

    let file = std::fs::File::open(tmp.path()).unwrap();
    let registry = CoderRegistry::default();
    let mut reader = HeliumReader::new(file, &registry).unwrap();
    assert_eq!(reader.row_count(), n as u64);
    let got = reader.read_column("name").unwrap();
    assert_eq!(got, LogicalColumn::Utf8(s.name));
}

#[test]
fn column_pruning_reads_only_requested_bytes() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    let n = 5_000;
    let s = make_sample(n);
    let cursor = Cursor::new(Vec::new());
    let bytes = write_sample_file(cursor, full_sample_schema(), &s)
        .unwrap()
        .into_inner();
    let full_file_len = bytes.len() as u64;

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
    let registry = CoderRegistry::default();
    let mut reader = HeliumReader::new(counting, &registry).unwrap();
    let after_open = counter.load(Ordering::Relaxed);
    assert!(after_open < full_file_len);

    let got = reader.read_column("tag").unwrap();
    assert_eq!(got, LogicalColumn::Primitive(ColumnData::I64(s.tag)));
    let after_column = counter.load(Ordering::Relaxed);
    let column_bytes = after_column - after_open;
    assert!(
        column_bytes < full_file_len / 2,
        "tag column read touched {column_bytes}B, file is {full_file_len}B"
    );
}

// ---------------------------------------------------------------------------
// Logical-type coverage
// ---------------------------------------------------------------------------

#[test]
fn utf8_roundtrip_including_non_ascii_and_empty() {
    let strings = vec![
        "".to_string(),
        "hello".to_string(),
        "你好，世界".to_string(),
        "🦀 Rust".to_string(),
        "".to_string(),
        "quite a long string ".repeat(10),
    ];
    let n = strings.len();

    let schema = Schema::new(vec![ColumnSpec::utf8(
        "s",
        vec![
            CoderSpec::new("delta"),
            CoderSpec::new("leb128"),
            CoderSpec::new("zstd"),
        ],
        vec![CoderSpec::new("zstd")],
    )]);

    let registry = CoderRegistry::default();
    let mut w = HeliumWriter::new(Cursor::new(Vec::new()), schema, &registry).unwrap();
    w.write_column("s", LogicalColumn::Utf8(strings.clone()))
        .unwrap();
    let bytes = w.finish().unwrap().into_inner();

    let mut r = HeliumReader::new(Cursor::new(bytes), &registry).unwrap();
    assert_eq!(r.row_count(), n as u64);
    assert_eq!(r.read_column("s").unwrap(), LogicalColumn::Utf8(strings));
}

#[test]
fn binary_roundtrip_with_invalid_utf8_bytes() {
    let blobs: Vec<Vec<u8>> = vec![
        vec![0xff, 0xfe, 0xfd],
        vec![],
        b"normal".to_vec(),
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    ];
    let schema = Schema::new(vec![ColumnSpec::binary(
        "b",
        vec![
            CoderSpec::new("delta"),
            CoderSpec::new("leb128"),
            CoderSpec::new("zstd"),
        ],
        vec![CoderSpec::new("zstd")],
    )]);

    let registry = CoderRegistry::default();
    let mut w = HeliumWriter::new(Cursor::new(Vec::new()), schema, &registry).unwrap();
    w.write_column("b", LogicalColumn::Binary(blobs.clone()))
        .unwrap();
    let bytes = w.finish().unwrap().into_inner();

    let mut r = HeliumReader::new(Cursor::new(bytes), &registry).unwrap();
    assert_eq!(r.read_column("b").unwrap(), LogicalColumn::Binary(blobs));
}

#[test]
fn nullable_prim_all_null_all_present_mixed() {
    let schema = Schema::new(vec![ColumnSpec::nullable_prim(
        "v",
        DataType::I32,
        vec![
            CoderSpec::new("rle"),
            CoderSpec::new("bitpack_auto"),
            CoderSpec::new("zstd"),
        ],
        vec![
            CoderSpec::new("delta"),
            CoderSpec::new("leb128"),
            CoderSpec::new("zstd"),
        ],
    )]);
    let registry = CoderRegistry::default();

    for (label, present, values) in [
        (
            "mixed",
            vec![true, false, true, true, false, false, true],
            vec![10i32, 20, 30, 40],
        ),
        ("all_present", vec![true; 10], (0..10i32).collect()),
        ("all_null", vec![false; 10], Vec::<i32>::new()),
    ] {
        let mut w = HeliumWriter::new(Cursor::new(Vec::new()), schema.clone(), &registry).unwrap();
        w.write_column(
            "v",
            LogicalColumn::NullablePrim {
                present: present.clone(),
                values: ColumnData::I32(values.clone()),
            },
        )
        .unwrap();
        let bytes = w.finish().unwrap().into_inner();

        let mut r = HeliumReader::new(Cursor::new(bytes), &registry).unwrap();
        let got = r.read_column("v").unwrap();
        assert_eq!(
            got,
            LogicalColumn::NullablePrim {
                present,
                values: ColumnData::I32(values),
            },
            "case: {label}"
        );
    }
}

#[test]
fn nullable_utf8_roundtrip() {
    let present = vec![true, false, true, false, true];
    let strings = vec!["a".to_string(), "bb".to_string(), "ccc".to_string()];

    let schema = Schema::new(vec![ColumnSpec::nullable_utf8(
        "s",
        vec![
            CoderSpec::new("rle"),
            CoderSpec::new("bitpack_auto"),
            CoderSpec::new("zstd"),
        ],
        vec![
            CoderSpec::new("delta"),
            CoderSpec::new("leb128"),
            CoderSpec::new("zstd"),
        ],
        vec![CoderSpec::new("zstd")],
    )]);
    let registry = CoderRegistry::default();
    let mut w = HeliumWriter::new(Cursor::new(Vec::new()), schema, &registry).unwrap();
    w.write_column(
        "s",
        LogicalColumn::NullableUtf8 {
            present: present.clone(),
            strings: strings.clone(),
        },
    )
    .unwrap();
    let bytes = w.finish().unwrap().into_inner();

    let mut r = HeliumReader::new(Cursor::new(bytes), &registry).unwrap();
    let got = r.read_column("s").unwrap();
    assert_eq!(got, LogicalColumn::NullableUtf8 { present, strings });
}

#[test]
fn array_of_utf8_roundtrip() {
    // Row 0: ["a", "bb"], Row 1: [], Row 2: ["ccc"]
    let offsets: Vec<u32> = vec![0, 2, 2, 3];
    let strings: Vec<String> = vec!["a".into(), "bb".into(), "ccc".into()];
    let schema = Schema::new(vec![ColumnSpec::array_of_utf8(
        "tags",
        vec![
            CoderSpec::new("delta"),
            CoderSpec::new("leb128"),
            CoderSpec::new("zstd"),
        ],
        vec![
            CoderSpec::new("delta"),
            CoderSpec::new("leb128"),
            CoderSpec::new("zstd"),
        ],
        vec![CoderSpec::new("zstd")],
    )]);
    let registry = CoderRegistry::default();
    let mut w = HeliumWriter::new(Cursor::new(Vec::new()), schema, &registry).unwrap();
    w.write_column(
        "tags",
        LogicalColumn::ArrayOfUtf8 {
            offsets: offsets.clone(),
            strings: strings.clone(),
        },
    )
    .unwrap();
    let bytes = w.finish().unwrap().into_inner();

    let mut r = HeliumReader::new(Cursor::new(bytes), &registry).unwrap();
    let got = r.read_column("tags").unwrap();
    assert_eq!(got, LogicalColumn::ArrayOfUtf8 { offsets, strings });
}

// ---------------------------------------------------------------------------
// Schema validation
// ---------------------------------------------------------------------------

#[test]
fn schema_json_roundtrip() {
    let schema = full_sample_schema();
    let json = schema.to_json().unwrap();
    let parsed = Schema::from_json(&json).unwrap();
    assert_eq!(schema, parsed);
}

#[test]
fn schema_rejects_duplicate_column_names() {
    let schema = Schema::new(vec![
        ColumnSpec::primitive("a", DataType::I64, vec![CoderSpec::new("delta")]),
        ColumnSpec::primitive("a", DataType::I64, vec![CoderSpec::new("leb128")]),
    ]);
    assert!(matches!(schema.validate(), Err(HeliumError::Schema { .. })));
}

#[test]
fn schema_rejects_unsupported_version() {
    let bytes = br#"{"version":999,"columns":[]}"#;
    let err = Schema::from_json(bytes).unwrap_err();
    assert!(matches!(err, HeliumError::Format(_)));
}

#[test]
fn schema_rejects_unknown_coder_id() {
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "ts",
        DataType::I64,
        vec![CoderSpec::new("made_up_coder")],
    )]);
    let err = schema.resolve_all(&CoderRegistry::default()).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("made_up_coder"),
        "error must name the coder: {msg}"
    );
}

#[test]
fn schema_rejects_missing_bitpack_width() {
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "v",
        DataType::U32,
        vec![CoderSpec::new("bitpack_fixed")],
    )]);
    let err = schema.resolve_all(&CoderRegistry::default()).unwrap_err();
    assert!(err.to_string().contains("width"));
}

#[test]
fn schema_rejects_encodings_count_mismatch() {
    // Utf8 needs 2 pipelines; giving 1 must fail.
    let schema = Schema::new(vec![ColumnSpec::new(
        "s",
        LogicalType::Utf8,
        vec![vec![CoderSpec::new("zstd")]],
    )]);
    assert!(matches!(schema.validate(), Err(HeliumError::Schema { .. })));
}

// ---------------------------------------------------------------------------
// Writer validation
// ---------------------------------------------------------------------------

#[test]
fn writer_rejects_non_bytes_terminal_pipeline() {
    // Deltamin alone ends in I64, not Bytes.
    let schema = Schema::new(vec![ColumnSpec::primitive(
        "v",
        DataType::I64,
        vec![CoderSpec::new("deltamin")],
    )]);
    let err =
        HeliumWriter::new(Cursor::new(Vec::new()), schema, &CoderRegistry::default()).unwrap_err();
    assert!(matches!(err, HeliumError::Schema { .. }));
}

#[test]
fn writer_rejects_unknown_column_name() {
    let schema = Schema::new(vec![timestamps_column()]);
    let registry = CoderRegistry::default();
    let mut w = HeliumWriter::new(Cursor::new(Vec::new()), schema, &registry).unwrap();
    let err = w
        .write_column(
            "not_there",
            LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 3])),
        )
        .unwrap_err();
    assert!(matches!(err, HeliumError::Schema { .. }));
}

#[test]
fn writer_rejects_duplicate_column_write() {
    let schema = Schema::new(vec![timestamps_column()]);
    let registry = CoderRegistry::default();
    let mut w = HeliumWriter::new(Cursor::new(Vec::new()), schema, &registry).unwrap();
    w.write_column(
        "ts",
        LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 3])),
    )
    .unwrap();
    let err = w
        .write_column(
            "ts",
            LogicalColumn::Primitive(ColumnData::I64(vec![4, 5, 6])),
        )
        .unwrap_err();
    assert!(matches!(err, HeliumError::Schema { .. }));
}

#[test]
fn writer_rejects_mismatched_row_counts() {
    let schema = Schema::new(vec![
        timestamps_column(),
        ColumnSpec::primitive(
            "b",
            DataType::I64,
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
        ),
    ]);
    let registry = CoderRegistry::default();
    let mut w = HeliumWriter::new(Cursor::new(Vec::new()), schema, &registry).unwrap();
    w.write_column(
        "ts",
        LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 3])),
    )
    .unwrap();
    let err = w
        .write_column("b", LogicalColumn::Primitive(ColumnData::I64(vec![1, 2])))
        .unwrap_err();
    assert!(matches!(err, HeliumError::Schema { .. }));
}

#[test]
fn writer_rejects_finish_with_missing_column() {
    let schema = Schema::new(vec![
        timestamps_column(),
        ColumnSpec::primitive(
            "b",
            DataType::I64,
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
        ),
    ]);
    let registry = CoderRegistry::default();
    let mut w = HeliumWriter::new(Cursor::new(Vec::new()), schema, &registry).unwrap();
    w.write_column(
        "ts",
        LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 3])),
    )
    .unwrap();
    let err = w.finish().unwrap_err();
    assert!(matches!(err, HeliumError::Schema { .. }));
}

// ---------------------------------------------------------------------------
// Reader corruption handling
// ---------------------------------------------------------------------------

fn good_file_bytes() -> Vec<u8> {
    let s = make_sample(100);
    write_sample_file(Cursor::new(Vec::new()), full_sample_schema(), &s)
        .unwrap()
        .into_inner()
}

#[test]
fn reader_rejects_bad_start_magic() {
    let mut bytes = good_file_bytes();
    bytes[0] ^= 0xff;
    let err = HeliumReader::new(Cursor::new(bytes), &CoderRegistry::default()).unwrap_err();
    assert!(matches!(err, HeliumError::Format(_)));
}

#[test]
fn reader_rejects_bad_end_magic() {
    let mut bytes = good_file_bytes();
    let len = bytes.len();
    bytes[len - 1] ^= 0xff;
    let err = HeliumReader::new(Cursor::new(bytes), &CoderRegistry::default()).unwrap_err();
    assert!(matches!(err, HeliumError::Format(_)));
}

#[test]
fn reader_rejects_truncated_file() {
    let bytes = good_file_bytes();
    let truncated = bytes[..bytes.len() / 2].to_vec();
    let err = HeliumReader::new(Cursor::new(truncated), &CoderRegistry::default()).unwrap_err();
    assert!(matches!(err, HeliumError::Format(_) | HeliumError::Io(_)));
}

// ---------------------------------------------------------------------------
// Dictionary encoding
// ---------------------------------------------------------------------------

#[test]
fn dict_utf8_roundtrip_low_cardinality() {
    // 10k rows drawn from 4 distinct strings. Perfect dict candidate.
    let raw: Vec<String> = (0..10_000)
        .map(|i| match i % 4 {
            0 => "pending".to_string(),
            1 => "running".to_string(),
            2 => "done".to_string(),
            _ => "failed".to_string(),
        })
        .collect();
    let n = raw.len();
    let encoded = LogicalColumn::dict_encode_utf8(raw.clone());
    assert!(
        matches!(&encoded, LogicalColumn::Dictionary { dictionary, .. }
            if matches!(dictionary.as_ref(), LogicalColumn::Utf8(d) if d.len() == 4))
    );

    let schema = Schema::new(vec![ColumnSpec::new(
        "status",
        LogicalType::Dictionary {
            inner: Box::new(LogicalType::Utf8),
        },
        vec![
            vec![
                CoderSpec::new("delta"),
                CoderSpec::new("leb128"),
                CoderSpec::new("zstd"),
            ],
            vec![CoderSpec::new("zstd")],
            vec![CoderSpec::new("bitpack_auto"), CoderSpec::new("zstd")],
        ],
    )]);
    let registry = CoderRegistry::default();
    let mut w = HeliumWriter::new(Cursor::new(Vec::new()), schema, &registry).unwrap();
    w.write_column("status", encoded.clone()).unwrap();
    let bytes = w.finish().unwrap().into_inner();

    // Should be tiny compared to the raw concatenation (~70 KB of string data).
    let raw_bytes: usize = raw.iter().map(|s| s.len()).sum();
    assert!(
        bytes.len() * 50 < raw_bytes,
        "dict didn't crush: {} vs {raw_bytes}",
        bytes.len()
    );

    let mut r = HeliumReader::new(Cursor::new(bytes), &registry).unwrap();
    assert_eq!(r.row_count(), n as u64);
    let got = r.read_column("status").unwrap();
    assert_eq!(got, encoded);

    // Materialize back to the original Vec<String>.
    let materialized = got.materialize_dict_utf8().unwrap();
    assert_eq!(materialized, raw);
}

#[test]
fn dict_prim_roundtrip_low_cardinality_ints() {
    let raw_ids: Vec<i64> = (0..10_000).map(|i| ((i * 31) % 17) as i64).collect();
    let encoded = LogicalColumn::dict_encode_primitive(ColumnData::I64(raw_ids.clone())).unwrap();
    let LogicalColumn::Dictionary { ref dictionary, .. } = encoded else {
        panic!("expected Dictionary from dict_encode_primitive");
    };
    assert_eq!(dictionary.row_count(), 17);

    let schema = Schema::new(vec![ColumnSpec::new(
        "bucket",
        LogicalType::Dictionary {
            inner: Box::new(LogicalType::Primitive {
                data_type: DataType::I64,
            }),
        },
        vec![
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
            vec![CoderSpec::new("bitpack_auto"), CoderSpec::new("zstd")],
        ],
    )]);
    let registry = CoderRegistry::default();
    let mut w = HeliumWriter::new(Cursor::new(Vec::new()), schema, &registry).unwrap();
    w.write_column("bucket", encoded.clone()).unwrap();
    let bytes = w.finish().unwrap().into_inner();

    let raw_size = raw_ids.len() * std::mem::size_of::<i64>();
    assert!(bytes.len() * 10 < raw_size);

    let mut r = HeliumReader::new(Cursor::new(bytes), &registry).unwrap();
    assert_eq!(r.read_column("bucket").unwrap(), encoded);
}

#[test]
fn dict_utf8_rejects_out_of_range_indices() {
    // Hand-crafted — index 5 when dict has only 2 entries.
    let bogus = LogicalColumn::Dictionary {
        dictionary: Box::new(LogicalColumn::Utf8(vec!["a".into(), "b".into()])),
        indices: vec![0, 1, 5, 0],
    };
    let schema = Schema::new(vec![ColumnSpec::new(
        "s",
        LogicalType::Dictionary {
            inner: Box::new(LogicalType::Utf8),
        },
        vec![
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
            vec![CoderSpec::new("zstd")],
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
        ],
    )]);
    let registry = CoderRegistry::default();
    let mut w = HeliumWriter::new(Cursor::new(Vec::new()), schema, &registry).unwrap();
    let err = w.write_column("s", bogus).unwrap_err();
    assert!(err.to_string().contains("out of range"));
}

#[test]
fn dict_prim_rejects_float_input() {
    let err = LogicalColumn::dict_encode_primitive(ColumnData::F64(vec![1.0, 2.0])).unwrap_err();
    assert!(err.to_string().contains("integer"));
}

// ---------------------------------------------------------------------------
// Format v2: multi-stripe + CRC
// ---------------------------------------------------------------------------

#[test]
fn magic_is_v5() {
    // Current writer emits v5 (self-contained); v6 is the catalog-mode variant.
    assert_eq!(MAGIC, helium::MAGIC_V5);
    assert_ne!(helium::MAGIC_V5, helium::MAGIC_V6);
}

fn multi_stripe_schema() -> Schema {
    Schema::new(vec![
        ColumnSpec::primitive(
            "ts",
            DataType::I64,
            vec![
                CoderSpec::new("delta"),
                CoderSpec::new("leb128"),
                CoderSpec::new("zstd"),
            ],
        ),
        ColumnSpec::utf8(
            "name",
            vec![
                CoderSpec::new("delta"),
                CoderSpec::new("leb128"),
                CoderSpec::new("zstd"),
            ],
            vec![CoderSpec::new("zstd")],
        ),
        ColumnSpec::nullable_prim(
            "weight",
            DataType::F64,
            vec![
                CoderSpec::new("rle"),
                CoderSpec::new("bitpack_auto"),
                CoderSpec::new("zstd"),
            ],
            vec![CoderSpec::new("gorilla"), CoderSpec::new("zstd")],
        ),
    ])
}

fn write_stripe<W: Write + Seek>(
    writer: &mut HeliumWriter<W>,
    start: i64,
    n: usize,
    null_rate_every: usize,
) {
    let ts: Vec<i64> = (0..n).map(|i| start + i as i64 * 30).collect();
    let names: Vec<String> = (0..n)
        .map(|i| format!("n_{}", (start as usize + i) % 7))
        .collect();
    let present: Vec<bool> = (0..n).map(|i| i % null_rate_every != 0).collect();
    let values: Vec<f64> = present
        .iter()
        .enumerate()
        .filter_map(|(i, &p)| {
            if p {
                Some(50.0 + ((i + start as usize) as f64 * 0.1).sin())
            } else {
                None
            }
        })
        .collect();
    writer
        .write_column("ts", LogicalColumn::Primitive(ColumnData::I64(ts)))
        .unwrap();
    writer
        .write_column("name", LogicalColumn::Utf8(names))
        .unwrap();
    writer
        .write_column(
            "weight",
            LogicalColumn::NullablePrim {
                present,
                values: ColumnData::F64(values),
            },
        )
        .unwrap();
}

#[test]
fn multi_stripe_roundtrip_with_concat() {
    let schema = multi_stripe_schema();
    let registry = CoderRegistry::default();
    let mut w = HeliumWriter::new(Cursor::new(Vec::new()), schema, &registry).unwrap();

    // 3 stripes of 100 rows each, different start offsets.
    write_stripe(&mut w, 1_700_000_000, 100, 3);
    w.finish_stripe().unwrap();
    write_stripe(&mut w, 1_800_000_000, 100, 5);
    w.finish_stripe().unwrap();
    write_stripe(&mut w, 1_900_000_000, 100, 2);
    let bytes = w.finish().unwrap().into_inner();

    // Verify v5 magic at both ends (current writer output).
    assert_eq!(&bytes[..8], helium::MAGIC_V5);
    assert_eq!(&bytes[bytes.len() - 8..], helium::MAGIC_V5);

    let mut r = HeliumReader::new(Cursor::new(bytes), &registry).unwrap();
    assert_eq!(r.stripe_count(), 3);
    assert_eq!(r.row_count(), 300);

    // Concat read across all stripes.
    let ts = r.read_column("ts").unwrap();
    let LogicalColumn::Primitive(ColumnData::I64(all_ts)) = ts else {
        panic!("expected i64");
    };
    assert_eq!(all_ts.len(), 300);
    assert_eq!(all_ts[0], 1_700_000_000);
    assert_eq!(all_ts[100], 1_800_000_000);
    assert_eq!(all_ts[200], 1_900_000_000);

    // Utf8 concat.
    let name = r.read_column("name").unwrap();
    let LogicalColumn::Utf8(all_names) = name else {
        panic!("expected utf8");
    };
    assert_eq!(all_names.len(), 300);

    // NullablePrim concat.
    let LogicalColumn::NullablePrim { present, values } = r.read_column("weight").unwrap() else {
        panic!("expected nullable prim");
    };
    assert_eq!(present.len(), 300);
    // present count across the 3 stripes = (100 - 100/3) + (100 - 100/5) + (100 - 100/2)
    //                                    = 66 + 80 + 50 = 196
    let present_count = present.iter().filter(|&&p| p).count();
    if let ColumnData::F64(v) = values {
        assert_eq!(v.len(), present_count);
    } else {
        panic!("expected f64 values");
    }
}

#[test]
fn multi_stripe_column_pruning_reads_only_target_bytes() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    let schema = multi_stripe_schema();
    let registry = CoderRegistry::default();
    let mut w = HeliumWriter::new(Cursor::new(Vec::new()), schema, &registry).unwrap();
    for s in 0..4 {
        write_stripe(&mut w, 1_700_000_000 + s * 10_000, 500, 4);
        if s < 3 {
            w.finish_stripe().unwrap();
        }
    }
    let bytes = w.finish().unwrap().into_inner();
    let full_len = bytes.len() as u64;

    struct Counting<R> {
        inner: R,
        read: Arc<AtomicU64>,
    }
    impl<R: Read> Read for Counting<R> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let n = self.inner.read(buf)?;
            self.read.fetch_add(n as u64, Ordering::Relaxed);
            Ok(n)
        }
    }
    impl<R: Seek> Seek for Counting<R> {
        fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
            self.inner.seek(pos)
        }
    }

    let counter = Arc::new(AtomicU64::new(0));
    let reader = Counting {
        inner: Cursor::new(bytes),
        read: counter.clone(),
    };
    let mut r = HeliumReader::new(reader, &registry).unwrap();
    let after_open = counter.load(Ordering::Relaxed);
    assert!(after_open < full_len);

    // Read just "name" from all stripes.
    let _ = r.read_column("name").unwrap();
    let after_col = counter.load(Ordering::Relaxed);
    let delta = after_col - after_open;
    // "name" is 2 physical columns; we skip ts (delta+leb128+zstd) and
    // weight (3+2 physical columns). Should be well under half the file.
    assert!(delta < full_len / 2, "pruning read {delta}B of {full_len}B");
}

#[test]
fn v2_crc_detects_single_byte_flip() {
    let schema = multi_stripe_schema();
    let registry = CoderRegistry::default();
    let mut w = HeliumWriter::new(Cursor::new(Vec::new()), schema, &registry).unwrap();
    write_stripe(&mut w, 1_700_000_000, 200, 3);
    let mut bytes = w.finish().unwrap().into_inner();

    // Use the reader to find where the body starts and ends, then corrupt a
    // byte that is guaranteed to be in the body (column data) region.
    // This avoids fragility when footer size grows (e.g. after adding new
    // per-column metadata fields).
    let r_tmp = HeliumReader::new(Cursor::new(bytes.clone()), &registry).unwrap();
    let (header_bytes, body_bytes, _) = r_tmp.region_sizes();
    // Corrupt a byte a quarter of the way into the body.
    let corrupt_offset = (header_bytes + body_bytes / 4) as usize;
    assert!(corrupt_offset < bytes.len(), "corrupt offset in range");
    bytes[corrupt_offset] ^= 0x55;

    let mut r = HeliumReader::new(Cursor::new(bytes), &registry).unwrap();
    // Some column read must flag a CRC error. The specific column depends on
    // where the corruption landed — we just verify *something* reports Corrupted.
    let mut saw_corrupted = false;
    for col in ["ts", "name", "weight"] {
        match r.read_column(col) {
            Err(HeliumError::Corrupted { .. }) => {
                saw_corrupted = true;
                break;
            }
            Err(HeliumError::Schema { .. }) => {
                // Decode may fail before CRC if the corruption cascades;
                // still a meaningful error.
                saw_corrupted = true;
                break;
            }
            Ok(_) => {}
            Err(e) => panic!("unexpected error on column {col}: {e}"),
        }
    }
    assert!(
        saw_corrupted,
        "CRC mismatch should have surfaced on at least one column read"
    );
}

#[test]
fn v2_footer_crc_detects_tampering() {
    let schema = multi_stripe_schema();
    let registry = CoderRegistry::default();
    let mut w = HeliumWriter::new(Cursor::new(Vec::new()), schema, &registry).unwrap();
    write_stripe(&mut w, 1_700_000_000, 50, 3);
    let mut bytes = w.finish().unwrap().into_inner();

    // Corrupt a byte near the end of the footer (past body, before trailer).
    // Trailer is 20 bytes; footer is just before that.
    let footer_target = bytes.len() - 25;
    bytes[footer_target] ^= 0x01;

    let err = HeliumReader::new(Cursor::new(bytes), &registry).unwrap_err();
    // Either footer CRC fails or footer JSON becomes invalid.
    assert!(matches!(
        err,
        HeliumError::Corrupted { .. } | HeliumError::Format(_) | HeliumError::Json(_)
    ));
}

#[test]
fn reader_rejects_dropped_v1_format() {
    // v1–v4 were removed before 1.0. A v1 magic must now be rejected as an
    // unsupported version rather than silently parsed.
    let mut file = Vec::new();
    file.extend_from_slice(b"HELIUM\x00\x01"); // dropped v1 magic
    file.extend_from_slice(&[0u8; 32]); // arbitrary trailing bytes
    let registry = CoderRegistry::default();
    let err = HeliumReader::new(Cursor::new(file), &registry).unwrap_err();
    assert!(
        matches!(err, HeliumError::Format(_)),
        "v1 magic must be rejected as unsupported format, got {err:?}"
    );
}

#[test]
fn finish_stripe_rejects_partial_stripe() {
    let schema = multi_stripe_schema();
    let registry = CoderRegistry::default();
    let mut w = HeliumWriter::new(Cursor::new(Vec::new()), schema, &registry).unwrap();
    w.write_column(
        "ts",
        LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 3])),
    )
    .unwrap();
    // Missing "name" and "weight" — finish_stripe must refuse.
    let err = w.finish_stripe().unwrap_err();
    assert!(matches!(err, HeliumError::Schema { .. }));
}

#[test]
fn dict_multi_stripe_refuses_concat() {
    let schema = Schema::new(vec![ColumnSpec::new(
        "s",
        LogicalType::Dictionary {
            inner: Box::new(LogicalType::Utf8),
        },
        vec![
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
            vec![CoderSpec::new("zstd")],
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
        ],
    )]);
    let registry = CoderRegistry::default();
    let mut w = HeliumWriter::new(Cursor::new(Vec::new()), schema, &registry).unwrap();

    let s1 = LogicalColumn::dict_encode_utf8(vec!["a".into(); 10]);
    w.write_column("s", s1).unwrap();
    w.finish_stripe().unwrap();
    let s2 = LogicalColumn::dict_encode_utf8(vec!["b".into(); 10]);
    w.write_column("s", s2).unwrap();
    let bytes = w.finish().unwrap().into_inner();

    let mut r = HeliumReader::new(Cursor::new(bytes), &registry).unwrap();
    // Concat read errors.
    let err = r.read_column("s").unwrap_err();
    assert!(
        err.to_string()
            .contains("dict columns cannot be concatenated")
    );
    // Per-stripe read works.
    let s1 = r.read_column_at_stripe("s", 0).unwrap();
    assert_eq!(
        s1.materialize_dict_utf8().unwrap(),
        vec!["a".to_string(); 10]
    );
    let s2 = r.read_column_at_stripe("s", 1).unwrap();
    assert_eq!(
        s2.materialize_dict_utf8().unwrap(),
        vec!["b".to_string(); 10]
    );
}

#[test]
fn dict_utf8_single_distinct_value() {
    let raw: Vec<String> = vec!["only".to_string(); 1000];
    let encoded = LogicalColumn::dict_encode_utf8(raw.clone());
    let LogicalColumn::Dictionary {
        ref dictionary,
        ref indices,
    } = encoded
    else {
        panic!("expected Dictionary from dict_encode_utf8");
    };
    assert_eq!(dictionary.row_count(), 1);
    assert!(indices.iter().all(|&i| i == 0));

    let schema = Schema::new(vec![ColumnSpec::new(
        "s",
        LogicalType::Dictionary {
            inner: Box::new(LogicalType::Utf8),
        },
        vec![
            vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")],
            vec![CoderSpec::new("zstd")],
            vec![
                CoderSpec::new("rle"),
                CoderSpec::new("leb128"),
                CoderSpec::new("zstd"),
            ],
        ],
    )]);
    let registry = CoderRegistry::default();
    let mut w = HeliumWriter::new(Cursor::new(Vec::new()), schema, &registry).unwrap();
    w.write_column("s", encoded.clone()).unwrap();
    let bytes = w.finish().unwrap().into_inner();

    let mut r = HeliumReader::new(Cursor::new(bytes), &registry).unwrap();
    let got = r.read_column("s").unwrap();
    assert_eq!(got.materialize_dict_utf8().unwrap(), raw);
}

// ============================================================================
// Type-matrix coverage for LogicalColumn through the full Writer+Reader path.
// Closes the "Primitive(T) / Array<T> / Nullable<T> probably works for T we
// didn't file-test" gap.
// ============================================================================

fn roundtrip_primitive(dt: DataType, data: ColumnData, coders: Vec<CoderSpec>) -> LogicalColumn {
    let schema = Schema::new(vec![ColumnSpec::primitive("c", dt, coders)]);
    let registry = CoderRegistry::default();
    let mut w = HeliumWriter::new(Cursor::new(Vec::new()), schema, &registry).unwrap();
    w.write_column("c", LogicalColumn::Primitive(data)).unwrap();
    let bytes = w.finish().unwrap().into_inner();
    let mut r = HeliumReader::new(Cursor::new(bytes), &registry).unwrap();
    r.read_column("c").unwrap()
}

#[test]
fn primitive_writer_reader_all_integer_types() {
    let int_pipe = || vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")];

    macro_rules! check {
        ($dt:ident, $variant:ident, $ty:ty, $vals:expr) => {{
            let vals: Vec<$ty> = $vals;
            let got = roundtrip_primitive(
                DataType::$dt,
                ColumnData::$variant(vals.clone()),
                int_pipe(),
            );
            assert_eq!(
                got,
                LogicalColumn::Primitive(ColumnData::$variant(vals)),
                "primitive writer/reader mismatch for {:?}",
                DataType::$dt
            );
        }};
    }
    check!(I8, I8, i8, vec![i8::MIN, -1, 0, 1, i8::MAX]);
    check!(I16, I16, i16, vec![i16::MIN, 0, i16::MAX]);
    check!(I32, I32, i32, (0i32..100).collect());
    check!(I64, I64, i64, vec![i64::MIN, 0, i64::MAX]);
    check!(U8, U8, u8, (0u8..=255).collect());
    check!(U16, U16, u16, vec![0, u16::MAX / 2, u16::MAX]);
    check!(U32, U32, u32, (0u32..100).collect());
    check!(U64, U64, u64, vec![0, 1, u64::MAX]);
}

#[test]
fn primitive_writer_reader_all_float_types() {
    macro_rules! check {
        ($dt:ident, $variant:ident, $ty:ty, $vals:expr) => {{
            let vals: Vec<$ty> = $vals;
            let got = roundtrip_primitive(
                DataType::$dt,
                ColumnData::$variant(vals.clone()),
                vec![CoderSpec::new("pcodec")],
            );
            assert_eq!(
                got,
                LogicalColumn::Primitive(ColumnData::$variant(vals)),
                "primitive writer/reader mismatch for {:?}",
                DataType::$dt
            );
        }};
    }
    check!(F32, F32, f32, (0..100).map(|i| i as f32 * 1.5).collect());
    check!(F64, F64, f64, (0..100).map(|i| (i as f64).sqrt()).collect());
}

#[test]
fn array_of_roundtrip_multiple_value_types() {
    macro_rules! check {
        ($dt:ident, $variant:ident, $ty:ty, $vals:expr, $pipe:expr) => {{
            let values: Vec<$ty> = $vals;
            let offsets: Vec<u32> = vec![0, 2, 2, values.len() as u32];
            let schema = Schema::new(vec![ColumnSpec::array_of(
                "a",
                DataType::$dt,
                vec![
                    CoderSpec::new("delta"),
                    CoderSpec::new("leb128"),
                    CoderSpec::new("zstd"),
                ],
                $pipe,
            )]);
            let registry = CoderRegistry::default();
            let mut w = HeliumWriter::new(Cursor::new(Vec::new()), schema, &registry).unwrap();
            w.write_column(
                "a",
                LogicalColumn::ArrayOf {
                    offsets: offsets.clone(),
                    values: ColumnData::$variant(values.clone()),
                },
            )
            .unwrap();
            let bytes = w.finish().unwrap().into_inner();
            let mut r = HeliumReader::new(Cursor::new(bytes), &registry).unwrap();
            let got = r.read_column("a").unwrap();
            assert_eq!(
                got,
                LogicalColumn::ArrayOf {
                    offsets,
                    values: ColumnData::$variant(values),
                },
                "Array<{:?}> mismatch",
                DataType::$dt
            );
        }};
    }
    check!(
        U64,
        U64,
        u64,
        vec![10, 20, 30, 40, 50],
        vec![CoderSpec::new("pcodec")]
    );
    check!(
        F64,
        F64,
        f64,
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![CoderSpec::new("pcodec")]
    );
    check!(
        I16,
        I16,
        i16,
        vec![-5, 0, 5, 10, 15],
        vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")]
    );
}

#[test]
fn nullable_prim_roundtrip_more_types() {
    // 5 rows, 3 present: rows 0, 2, 4.
    let present = vec![true, false, true, false, true];

    macro_rules! check {
        ($dt:ident, $variant:ident, $ty:ty, $vals:expr, $values_pipe:expr) => {{
            let values: Vec<$ty> = $vals;
            assert_eq!(values.len(), 3, "test expects 3 present values");
            let schema = Schema::new(vec![ColumnSpec::nullable_prim(
                "n",
                DataType::$dt,
                vec![
                    CoderSpec::new("rle"),
                    CoderSpec::new("bitpack_auto"),
                    CoderSpec::new("zstd"),
                ],
                $values_pipe,
            )]);
            let registry = CoderRegistry::default();
            let mut w = HeliumWriter::new(Cursor::new(Vec::new()), schema, &registry).unwrap();
            w.write_column(
                "n",
                LogicalColumn::NullablePrim {
                    present: present.clone(),
                    values: ColumnData::$variant(values.clone()),
                },
            )
            .unwrap();
            let bytes = w.finish().unwrap().into_inner();
            let mut r = HeliumReader::new(Cursor::new(bytes), &registry).unwrap();
            let got = r.read_column("n").unwrap();
            assert_eq!(
                got,
                LogicalColumn::NullablePrim {
                    present: present.clone(),
                    values: ColumnData::$variant(values),
                },
                "Nullable<{:?}> mismatch",
                DataType::$dt
            );
        }};
    }
    check!(
        I8,
        I8,
        i8,
        vec![-1, 0, 1],
        vec![CoderSpec::new("leb128"), CoderSpec::new("zstd")]
    );
    check!(
        U16,
        U16,
        u16,
        vec![0, 16_384, u16::MAX],
        vec![CoderSpec::new("bitpack_auto"), CoderSpec::new("zstd")]
    );
    check!(
        U32,
        U32,
        u32,
        vec![0, 1_000_000, u32::MAX],
        vec![CoderSpec::new("pcodec")]
    );
    check!(
        F32,
        F32,
        f32,
        vec![1.0_f32, 2.5, -0.75],
        vec![CoderSpec::new("gorilla"), CoderSpec::new("zstd")]
    );
}
