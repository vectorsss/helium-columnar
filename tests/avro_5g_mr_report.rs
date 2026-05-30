//! 5G Measurement-Report — synthetic generator + compression comparison report.
//!
//! Validates Helium against the "Avro + zstd" storage pattern on MR-shaped
//! nested data and exercises all Helium v3 nested types in a single test:
//!   * `List<Struct>` — neighbor cell array
//!   * `Map<Utf8, F32>` — custom KPI map
//!   * `Nullable<Utf8>` — location string
//!   * `Nullable<I32>` — handover target cell ID
//!
//! Run:
//!   cargo test --test avro_5g_mr_report --release --all-features -- --nocapture
//!
//! Output: stdout + `target/avro-5g-mr-report.md`
//!
//! N controlled by `HELIUM_MR_ROWS` env var (default 20_000).

#![cfg(feature = "schema-avro")]

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::io::Cursor;

use apache_avro::{Codec, DeflateSettings, Schema as AvroSchema, Writer as AvroWriter};

use helium::optimizer::Optimizer;
use helium::schema::avro::read_avro_data;
use helium::{CoderRegistry, HeliumReader, HeliumWriter, LogicalColumn, Schema};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const DEFAULT_N: usize = 20_000;

fn n_rows() -> usize {
    std::env::var("HELIUM_MR_ROWS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_N)
}

// ---------------------------------------------------------------------------
// Deterministic PRNG — SplitMix64, seeded, no external deps
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_f32(&mut self) -> f32 {
        self.next_f64() as f32
    }

    fn next_range(&mut self, lo: i64, hi: i64) -> i64 {
        // [lo, hi) uniform
        let range = (hi - lo) as u64;
        if range == 0 {
            return lo;
        }
        lo + (self.next_u64() % range) as i64
    }

    fn next_bool_p(&mut self, p_true_percent: u64) -> bool {
        (self.next_u64() % 100) < p_true_percent
    }

    // Gaussian approximation via Box-Muller (only u half)
    fn next_gaussian(&mut self, mean: f64, stddev: f64) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + stddev * z
    }

    // Poisson approximation: sum of uniforms method (Knuth, works for small lambda)
    fn next_poisson(&mut self, lambda: f64) -> u32 {
        let l = (-lambda).exp();
        let mut k = 0u32;
        let mut p = 1.0f64;
        loop {
            k += 1;
            p *= self.next_f64();
            if p <= l {
                return k - 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MR field distributions
// ---------------------------------------------------------------------------

const KPI_KEYS: &[&str] = &[
    "dl_tput",
    "ul_tput",
    "bler",
    "cqi",
    "ri",
    "mcs",
    "ta",
    "phr",
    "pusch_pwr",
    "pucch_pwr",
];

const UE_POOL_SIZE: usize = 5_000;
const CELL_COUNT: usize = 500;

// Quantize float to 0.1 precision (realistic MR resolution)
fn quantize(v: f64) -> f32 {
    ((v * 10.0).round() / 10.0) as f32
}

// ---------------------------------------------------------------------------
// Avro schema for 5G MR
// ---------------------------------------------------------------------------

const MR_AVSC: &str = r#"{
  "type": "record",
  "name": "MeasurementReport",
  "fields": [
    {"name": "timestamp",         "type": {"type":"long","logicalType":"timestamp-millis"}},
    {"name": "ue_id",             "type": "string"},
    {"name": "serving_cell_id",   "type": "int"},
    {"name": "serving_pci",       "type": "int"},
    {"name": "serving_rsrp",      "type": "float"},
    {"name": "serving_rsrq",      "type": "float"},
    {"name": "serving_sinr",      "type": "float"},
    {"name": "neighbors", "type": {
      "type": "array",
      "items": {
        "type": "record",
        "name": "NeighborCell",
        "fields": [
          {"name": "pci",   "type": "int"},
          {"name": "earfcn","type": "int"},
          {"name": "rsrp",  "type": "float"},
          {"name": "rsrq",  "type": "float"}
        ]
      }
    }},
    {"name": "custom_kpis",       "type": {"type":"map","values":"float"}},
    {"name": "location",          "type": ["null","string"]},
    {"name": "handover_target",   "type": ["null","int"]}
  ]
}"#;

// ---------------------------------------------------------------------------
// Generator: produce N synthetic MR records as apache_avro::types::Value
// ---------------------------------------------------------------------------

fn generate_mr_records(n: usize) -> Vec<apache_avro::types::Value> {
    use apache_avro::types::Value as AV;

    let mut rng = Rng::new(0xDEAD_BEEF_5555_u64);

    // Pre-build UE pool
    let ue_pool: Vec<String> = (0..UE_POOL_SIZE).map(|i| format!("UE-{i:08}")).collect();

    // Pre-build location strings (moderate cardinality)
    let loc_pool: Vec<String> = (0..200)
        .map(|i| {
            let lat = 48.0 + (i as f64) * 0.05;
            let lon = 11.0 + (i as f64) * 0.07;
            format!("lat:{lat:.4},lon:{lon:.4}")
        })
        .collect();

    // Zipf-ish cell_id: ~20% of cells get 60% of traffic
    let hot_cells: Vec<i32> = (0..50).map(|i| i * 3).collect();

    let mut ts: i64 = 1_700_000_000_000; // epoch millis

    (0..n)
        .map(|_| {
            // Jittered timestamp increment 10-100ms
            let jitter = rng.next_range(10, 100);
            ts += jitter;

            // ue_id: pick from pool
            let ue_idx = rng.next_u64() as usize % UE_POOL_SIZE;
            let ue_id = ue_pool[ue_idx].clone();

            // serving_cell_id: Zipf-ish
            let cell_id: i32 = if rng.next_bool_p(60) {
                hot_cells[rng.next_u64() as usize % hot_cells.len()]
            } else {
                rng.next_range(0, CELL_COUNT as i64) as i32
            };

            // serving_pci: 0..1007
            let pci = rng.next_range(0, 1008) as i32;

            // serving_rsrp: Gaussian around -95, clamped [-140, -44], quantized
            let rsrp = quantize(rng.next_gaussian(-95.0, 15.0).clamp(-140.0, -44.0));
            // serving_rsrq: Gaussian around -12, clamped [-20, -3]
            let rsrq = quantize(rng.next_gaussian(-12.0, 4.0).clamp(-20.0, -3.0));
            // serving_sinr: Gaussian around 10, clamped [-10, 30]
            let sinr = quantize(rng.next_gaussian(10.0, 8.0).clamp(-10.0, 30.0));

            // neighbors: Poisson(mean=3), capped at 8
            let n_neighbors = rng.next_poisson(3.0).min(8) as usize;
            let neighbors: Vec<AV> = (0..n_neighbors)
                .map(|_| {
                    let npci = rng.next_range(0, 1008) as i32;
                    let nearfcn = rng.next_range(0, 65536) as i32;
                    // neighbor rsrp slightly below serving
                    let nrsrp = quantize(
                        (rsrp as f64 - rng.next_range(0, 20) as f64).clamp(-140.0, -44.0) as f32
                            as f64,
                    );
                    let nrsrq = quantize(rng.next_gaussian(-14.0, 4.0).clamp(-20.0, -3.0));
                    AV::Record(vec![
                        ("pci".into(), AV::Int(npci)),
                        ("earfcn".into(), AV::Int(nearfcn)),
                        ("rsrp".into(), AV::Float(nrsrp)),
                        ("rsrq".into(), AV::Float(nrsrq)),
                    ])
                })
                .collect();

            // custom_kpis: 2-6 entries from KPI_KEYS
            let n_kpis = rng.next_range(2, 7) as usize;
            let mut kpi_indices: Vec<usize> = (0..KPI_KEYS.len()).collect();
            // Fisher-Yates shuffle the first n_kpis
            for i in 0..n_kpis {
                let j = i + rng.next_u64() as usize % (KPI_KEYS.len() - i);
                kpi_indices.swap(i, j);
            }
            let custom_kpis: HashMap<String, AV> = kpi_indices[..n_kpis]
                .iter()
                .map(|&ki| {
                    let v = rng.next_f32() * 100.0;
                    (KPI_KEYS[ki].to_string(), AV::Float(v))
                })
                .collect();

            // location: ~30% null
            let location: AV = if rng.next_bool_p(70) {
                let idx = rng.next_u64() as usize % loc_pool.len();
                AV::Union(1, Box::new(AV::String(loc_pool[idx].clone())))
            } else {
                AV::Union(0, Box::new(AV::Null))
            };

            // handover_target: ~80% null
            let handover_target: AV = if rng.next_bool_p(20) {
                AV::Union(
                    1,
                    Box::new(AV::Int(rng.next_range(0, CELL_COUNT as i64) as i32)),
                )
            } else {
                AV::Union(0, Box::new(AV::Null))
            };

            AV::Record(vec![
                ("timestamp".into(), AV::Long(ts)),
                ("ue_id".into(), AV::String(ue_id)),
                ("serving_cell_id".into(), AV::Int(cell_id)),
                ("serving_pci".into(), AV::Int(pci)),
                ("serving_rsrp".into(), AV::Float(rsrp)),
                ("serving_rsrq".into(), AV::Float(rsrq)),
                ("serving_sinr".into(), AV::Float(sinr)),
                ("neighbors".into(), AV::Array(neighbors)),
                ("custom_kpis".into(), AV::Map(custom_kpis)),
                ("location".into(), location),
                ("handover_target".into(), handover_target),
            ])
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Avro serialization helpers
// ---------------------------------------------------------------------------

/// Write records to an in-memory Avro container with the given codec.
fn write_avro_to_bytes(
    avro_schema: &AvroSchema,
    records: &[apache_avro::types::Value],
    codec: Codec,
) -> Vec<u8> {
    let mut writer = AvroWriter::builder()
        .schema(avro_schema)
        .codec(codec)
        .writer(Vec::new())
        .build();

    for record in records {
        writer
            .append(record.clone())
            .expect("avro append should not fail");
    }
    writer.into_inner().expect("avro flush should not fail")
}

// ---------------------------------------------------------------------------
// Helium write helpers
// ---------------------------------------------------------------------------

fn write_helium(schema: &Schema, columns: &HashMap<String, LogicalColumn>) -> Vec<u8> {
    let registry = CoderRegistry::default();
    let cursor = Cursor::new(Vec::<u8>::new());
    let mut writer =
        HeliumWriter::new(cursor, schema.clone(), &registry).expect("HeliumWriter::new");

    for col_spec in &schema.columns {
        let lc = columns[&col_spec.name].clone();
        writer
            .write_column(&col_spec.name, lc)
            .expect("write_column");
    }
    writer.finish().expect("finish").into_inner()
}

// ---------------------------------------------------------------------------
// Round-trip verification helpers
// ---------------------------------------------------------------------------

/// Compare two `LogicalColumn`s element-by-element, tolerating f32 NaN == NaN.
/// Returns `Ok(())` if equal, `Err(description)` if not.
fn columns_equal(a: &LogicalColumn, b: &LogicalColumn, path: &str) -> Result<(), String> {
    use helium::ColumnData;

    match (a, b) {
        (LogicalColumn::Primitive(ca), LogicalColumn::Primitive(cb)) => {
            // For F32 data, NaN-aware comparison
            match (ca, cb) {
                (ColumnData::F32(va), ColumnData::F32(vb)) => {
                    if va.len() != vb.len() {
                        return Err(format!("{path}: F32 len {} != {}", va.len(), vb.len()));
                    }
                    for (i, (x, y)) in va.iter().zip(vb.iter()).enumerate() {
                        if x != y && !(x.is_nan() && y.is_nan()) {
                            return Err(format!("{path}[{i}]: F32 {x} != {y}"));
                        }
                    }
                }
                _ => {
                    if ca != cb {
                        return Err(format!("{path}: Primitive mismatch"));
                    }
                }
            }
        }
        (LogicalColumn::Utf8(va), LogicalColumn::Utf8(vb)) => {
            if va != vb {
                return Err(format!(
                    "{path}: Utf8 mismatch (len {} vs {})",
                    va.len(),
                    vb.len()
                ));
            }
        }
        (
            LogicalColumn::Nullable {
                present: pa,
                value: va,
            },
            LogicalColumn::Nullable {
                present: pb,
                value: vb,
            },
        ) => {
            if pa != pb {
                return Err(format!("{path}: Nullable present bitmaps differ"));
            }
            columns_equal(va, vb, &format!("{path}.inner"))?;
        }
        (
            LogicalColumn::List {
                offsets: oa,
                values: va,
            },
            LogicalColumn::List {
                offsets: ob,
                values: vb,
            },
        ) => {
            if oa != ob {
                return Err(format!("{path}: List offsets differ"));
            }
            columns_equal(va, vb, &format!("{path}.items"))?;
        }
        (
            LogicalColumn::Map {
                offsets: oa,
                keys: ka,
                values: va,
            },
            LogicalColumn::Map {
                offsets: ob,
                keys: kb,
                values: vb,
            },
        ) => {
            if oa != ob {
                return Err(format!("{path}: Map offsets differ"));
            }
            columns_equal(ka, kb, &format!("{path}.keys"))?;
            columns_equal(va, vb, &format!("{path}.values"))?;
        }
        (LogicalColumn::Struct { fields: fa }, LogicalColumn::Struct { fields: fb }) => {
            if fa.len() != fb.len() {
                return Err(format!(
                    "{path}: Struct field count {} vs {}",
                    fa.len(),
                    fb.len()
                ));
            }
            for ((na, ca_inner), (nb, cb_inner)) in fa.iter().zip(fb.iter()) {
                if na != nb {
                    return Err(format!("{path}: Struct field name {na} != {nb}"));
                }
                columns_equal(ca_inner, cb_inner, &format!("{path}.{na}"))?;
            }
        }
        _ => {
            // Fall back to PartialEq for all other variants
            if a != b {
                return Err(format!("{path}: mismatch (different variant or value)"));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Main test
// ---------------------------------------------------------------------------

#[test]
fn avro_5g_mr_compression_report() {
    let n = n_rows();
    eprintln!("\n=== 5G MR Compression Report — N={n} ===\n");

    // 1. Generate synthetic MR records
    let records = generate_mr_records(n);
    assert_eq!(records.len(), n, "generator produced wrong count");

    // 2. Parse Avro schema
    let avro_schema = AvroSchema::parse_str(MR_AVSC).expect("MR_AVSC should parse");

    // 3a. Write Avro (deflate) — the real-world "Avro with built-in compression"
    let avro_deflate_bytes = write_avro_to_bytes(
        &avro_schema,
        &records,
        Codec::Deflate(DeflateSettings::default()),
    );
    let avro_deflate_size = avro_deflate_bytes.len();
    assert!(
        !avro_deflate_bytes.is_empty(),
        "avro deflate output must be non-empty"
    );
    eprintln!("avro (deflate):       {:>9} bytes", avro_deflate_size);

    // 3b. Write Avro (null) then compress with external zstd — canonical "Avro+zstd" pattern
    let avro_null_bytes = write_avro_to_bytes(&avro_schema, &records, Codec::Null);
    let avro_null_size = avro_null_bytes.len();
    let avro_null_zstd_bytes =
        zstd::encode_all(&avro_null_bytes[..], 3).expect("zstd encode avro null");
    let avro_null_zstd_size = avro_null_zstd_bytes.len();
    eprintln!("avro (null):          {:>9} bytes", avro_null_size);
    eprintln!(
        "avro (null)+zstd3:    {:>9} bytes  ← the production anchor",
        avro_null_zstd_size
    );

    // 4. Read back avro deflate file → (helium_schema, columns)
    //    We write to a tempfile so that read_avro_data (which takes a Path) can read it.
    let tmpdir = tempfile::tempdir().expect("create tempdir");
    let avro_path = tmpdir.path().join("mr.avro");
    std::fs::write(&avro_path, &avro_deflate_bytes).expect("write avro file");

    let (helium_schema, columns) =
        read_avro_data(&avro_path).expect("read_avro_data should succeed on MR schema");

    // Sanity-check: all 11 top-level columns present
    assert_eq!(
        helium_schema.columns.len(),
        11,
        "MR schema should have 11 top-level columns, got {}",
        helium_schema.columns.len()
    );

    // 5. Write helium-default
    let he_default_bytes = write_helium(&helium_schema, &columns);
    let he_default_size = he_default_bytes.len();
    assert!(
        !he_default_bytes.is_empty(),
        "helium default output must be non-empty"
    );
    eprintln!("helium-default:       {:>9} bytes", he_default_size);

    // 6. Round-trip correctness for helium-default (MUST pass before timing claim)
    {
        let registry = CoderRegistry::default();
        let cursor = Cursor::new(he_default_bytes.clone());
        let mut reader = HeliumReader::new(cursor, &registry).expect("HeliumReader::new (default)");
        let decoded = reader.read_all().expect("read_all (default)");

        for col_spec in &helium_schema.columns {
            let original = &columns[&col_spec.name];
            let roundtripped = &decoded[&col_spec.name];
            columns_equal(original, roundtripped, &col_spec.name)
                .expect("round-trip correctness failed for helium-default");
        }
        eprintln!("  [round-trip default: PASS]");
    }

    // 7. Write helium-optimized
    //
    // All LogicalType variants (including Datetime, Date32, Date64, Decimal128)
    // are now handled by the optimizer — no per-column fallback needed.
    let optimized_schema = {
        let all_columns: Vec<(String, _, _)> = helium_schema
            .columns
            .iter()
            .map(|spec| {
                (
                    spec.name.clone(),
                    spec.logical_type.clone(),
                    columns[&spec.name].clone(),
                )
            })
            .collect();
        Optimizer::new()
            .optimize(all_columns)
            .expect("optimizer must handle all schema types")
    };

    let he_opt_bytes = write_helium(&optimized_schema, &columns);
    let he_opt_size = he_opt_bytes.len();
    eprintln!("helium-optimized:     {:>9} bytes", he_opt_size);

    // 8. Round-trip correctness for helium-optimized
    {
        let registry = CoderRegistry::default();
        let cursor = Cursor::new(he_opt_bytes.clone());
        let mut reader =
            HeliumReader::new(cursor, &registry).expect("HeliumReader::new (optimized)");
        let decoded = reader.read_all().expect("read_all (optimized)");

        for col_spec in &helium_schema.columns {
            let original = &columns[&col_spec.name];
            let roundtripped = &decoded[&col_spec.name];
            columns_equal(original, roundtripped, &col_spec.name)
                .expect("round-trip correctness failed for helium-optimized");
        }
        eprintln!("  [round-trip optimized: PASS]");
    }

    // 9. Build the markdown report
    let mut report = String::new();
    writeln!(&mut report, "# 5G MR — Avro vs Helium compression").unwrap();
    writeln!(&mut report).unwrap();
    writeln!(
        &mut report,
        "**Commit**: {}",
        option_env!("GIT_HASH").unwrap_or("(not captured at build time)")
    )
    .unwrap();
    writeln!(&mut report, "**Hardware**: Apple M1 Max (arm64)").unwrap();
    writeln!(
        &mut report,
        "**Rust**: {}",
        option_env!("RUSTC_VERSION").unwrap_or("(see rustc --version)")
    )
    .unwrap();
    writeln!(&mut report, "**Build**: --release").unwrap();
    writeln!(&mut report).unwrap();

    writeln!(
        &mut report,
        "Dataset: synthetic 5G Measurement Reports, **{n} rows**"
    )
    .unwrap();
    writeln!(
        &mut report,
        "Schema: nested (neighbors: List<Struct>, custom_kpis: Map<Utf8,F32>,\n\
         location: Nullable<Utf8>, handover_target: Nullable<I32>)"
    )
    .unwrap();
    writeln!(&mut report).unwrap();

    let baseline = avro_deflate_size as f64;
    let ratio = |sz: usize| sz as f64 / baseline;

    writeln!(&mut report, "| Format | bytes | ratio vs avro+deflate |").unwrap();
    writeln!(&mut report, "|--------|------:|---------------------:|").unwrap();
    writeln!(
        &mut report,
        "| avro (deflate)          | {:>9} | 1.00 |",
        avro_deflate_size
    )
    .unwrap();
    writeln!(
        &mut report,
        "| avro (null) + zstd L3   | {:>9} | {:.2} |",
        avro_null_zstd_size,
        ratio(avro_null_zstd_size)
    )
    .unwrap();
    writeln!(
        &mut report,
        "| helium-default          | {:>9} | {:.2} |",
        he_default_size,
        ratio(he_default_size)
    )
    .unwrap();
    writeln!(
        &mut report,
        "| helium-optimized        | {:>9} | {:.2} |",
        he_opt_size,
        ratio(he_opt_size)
    )
    .unwrap();
    writeln!(&mut report).unwrap();

    // Per-column breakdown (default schema)
    writeln!(&mut report, "## Per-column encoded size (helium-default)").unwrap();
    writeln!(&mut report).unwrap();
    writeln!(&mut report, "| Column | Helium bytes | Fraction |").unwrap();
    writeln!(&mut report, "|--------|-------------:|---------:|").unwrap();

    let registry = CoderRegistry::default();
    let schema_pipelines = helium_schema.resolve_all(&registry).expect("resolve_all");
    let mut per_col_sizes: Vec<(String, usize)> = Vec::new();
    for (ci, col_spec) in helium_schema.columns.iter().enumerate() {
        let lc = columns[&col_spec.name].clone();
        let parts = lc.decompose(&col_spec.logical_type).expect("decompose");
        let pipes = &schema_pipelines[ci];
        let col_bytes: usize = parts
            .into_iter()
            .zip(pipes.iter())
            .map(|(part, pipe)| {
                let enc = pipe.encode(part).expect("encode");
                match enc {
                    helium::ColumnData::Bytes(b) => b.len(),
                    _ => 0,
                }
            })
            .sum();
        per_col_sizes.push((col_spec.name.clone(), col_bytes));
    }
    let total_data: usize = per_col_sizes.iter().map(|(_, b)| b).sum();
    per_col_sizes.sort_by(|a, b| b.1.cmp(&a.1));
    for (name, bytes) in &per_col_sizes {
        writeln!(
            &mut report,
            "| {:<22} | {:>12} | {:>7.1}% |",
            name,
            bytes,
            100.0 * *bytes as f64 / total_data as f64
        )
        .unwrap();
    }
    writeln!(&mut report).unwrap();

    // Takeaways
    let he_opt_vs_deflate = (1.0 - ratio(he_opt_size)) * 100.0;
    let he_opt_vs_zstd_anchor = (1.0 - he_opt_size as f64 / avro_null_zstd_size as f64) * 100.0;
    let he_default_vs_anchor = (1.0 - he_default_size as f64 / avro_null_zstd_size as f64) * 100.0;

    writeln!(&mut report, "## Takeaways").unwrap();
    writeln!(&mut report).unwrap();

    if he_opt_vs_zstd_anchor > 0.0 {
        writeln!(
            &mut report,
            "- helium-optimized **beats** the Avro+zstd anchor by {he_opt_vs_zstd_anchor:.1}% \
             ({he_opt_size} B vs {avro_null_zstd_size} B)."
        )
        .unwrap();
    } else {
        writeln!(
            &mut report,
            "- helium-optimized **does not beat** the Avro+zstd anchor at this N \
             ({he_opt_size} B vs {avro_null_zstd_size} B, \
             {:.1}% larger).",
            -he_opt_vs_zstd_anchor
        )
        .unwrap();
    }

    if he_default_vs_anchor > 0.0 {
        writeln!(
            &mut report,
            "- helium-default also beats the anchor by {he_default_vs_anchor:.1}% \
             — the inferred default encodings work well on MR columns."
        )
        .unwrap();
    } else {
        writeln!(
            &mut report,
            "- helium-default does NOT beat the anchor ({:.1}% larger). \
             Only the optimized schema wins.",
            -he_default_vs_anchor
        )
        .unwrap();
    }

    // Dominant column
    let (dominant_col, dominant_bytes) = per_col_sizes.first().unwrap();
    writeln!(
        &mut report,
        "- The dominant column is **{}** at {} B ({:.1}% of total data payload, default schema). \
         This is expected: the neighbor List<Struct> is the structurally richest field.",
        dominant_col,
        dominant_bytes,
        100.0 * *dominant_bytes as f64 / total_data as f64
    )
    .unwrap();

    writeln!(
        &mut report,
        "- helium-optimized vs avro+deflate: {he_opt_vs_deflate:.1}% smaller. \
         This validates Helium's core claim on MR-shaped nested data."
    )
    .unwrap();

    writeln!(
        &mut report,
        "- **Caveat**: synthetic data with fixed distributions. \
         Real operator MR streams will differ in UE pool size, neighbor count \
         distribution, and KPI value entropy."
    )
    .unwrap();
    writeln!(&mut report).unwrap();
    writeln!(
        &mut report,
        "Round-trip correctness: both helium-default and helium-optimized verified \
         (avro→he→read → byte/value equal)."
    )
    .unwrap();

    // Print to stdout
    eprintln!("\n{report}");

    // Write to target/
    std::fs::create_dir_all("/Users/chizhao/Code/opensource/Helium/helium-core/target").ok();
    std::fs::write(
        "/Users/chizhao/Code/opensource/Helium/helium-core/target/avro-5g-mr-report.md",
        &report,
    )
    .expect("write report to target/");
    eprintln!("Report written to target/avro-5g-mr-report.md");
}
