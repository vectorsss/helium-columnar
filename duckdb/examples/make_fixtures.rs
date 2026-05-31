//! Generate `.he` test fixtures that the CLI converters cannot easily produce:
//! a `Map`-typed column and a multi-stripe `Nullable` column whose non-null
//! values straddle stripe and chunk boundaries.
//!
//! Used by `smoke.sh` to exercise the DuckDB extension's `Map` writer and the
//! absolute-row indexing of the recursive `Nullable` read path.
//!
//! Usage: `make_fixtures <map_out.he> <nullable_out.he>`

use std::fs::File;

use helium::schema::encodings::default_encodings;
use helium::{
    ColumnData, ColumnSpec, CoderRegistry, DataType, HeliumWriter, LogicalColumn, LogicalType,
    Schema,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let map_out = args.next().ok_or("usage: make_fixtures <map.he> <nullable.he>")?;
    let nullable_out = args.next().ok_or("usage: make_fixtures <map.he> <nullable.he>")?;

    write_map_fixture(&map_out)?;
    write_nullable_multistripe_fixture(&nullable_out)?;
    Ok(())
}

/// A 4-row file with an `id` column and a `Map<Utf8, I64>` column.
fn write_map_fixture(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let key_lt = LogicalType::Utf8;
    let value_lt = LogicalType::Primitive { data_type: DataType::I64 };
    let map_lt = LogicalType::Map {
        key: Box::new(key_lt.clone()),
        value: Box::new(value_lt.clone()),
    };

    let schema = Schema::new(vec![
        ColumnSpec::primitive(
            "id",
            DataType::I64,
            vec![helium::CoderSpec::new("leb128"), helium::CoderSpec::new("zstd")],
        ),
        ColumnSpec::map("attrs", key_lt, value_lt, default_encodings(&map_lt)),
    ]);

    let registry = CoderRegistry::default();
    let file = File::create(path)?;
    let mut writer = HeliumWriter::new(file, schema, &registry)?;

    // 4 rows. attrs row i has i entries: {k0: 0, k1: 10, ...}.
    let id = LogicalColumn::Primitive(ColumnData::I64(vec![1, 2, 3, 4]));

    // Build the map: offsets [0,1,3,6,10], keys/values flattened.
    let mut offsets = vec![0u32];
    let mut keys: Vec<String> = Vec::new();
    let mut values: Vec<i64> = Vec::new();
    for row in 0..4 {
        let n = row + 1;
        for j in 0..n {
            keys.push(format!("k{j}"));
            values.push((j as i64) * 10);
        }
        // offsets always non-empty (seeded with 0).
        let last = *offsets.last().unwrap_or(&0);
        offsets.push(last + n as u32);
    }
    let attrs = LogicalColumn::Map {
        offsets,
        keys: Box::new(LogicalColumn::Utf8(keys)),
        values: Box::new(LogicalColumn::Primitive(ColumnData::I64(values))),
    };

    writer.write_column("id", id)?;
    writer.write_column("attrs", attrs)?;
    writer.finish()?;
    Ok(())
}

/// A multi-stripe file with a `Nullable<I64>` column whose nulls fall at
/// stripe and chunk boundaries — exercises absolute-row indexing of the
/// compacted inner values across boundaries.
fn write_nullable_multistripe_fixture(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let inner = LogicalType::Primitive { data_type: DataType::I64 };
    let nullable_lt = LogicalType::Nullable { inner: Box::new(inner.clone()) };

    let schema = Schema::new(vec![
        ColumnSpec::primitive(
            "id",
            DataType::I64,
            vec![helium::CoderSpec::new("leb128"), helium::CoderSpec::new("zstd")],
        ),
        ColumnSpec::nullable("v", inner, default_encodings(&nullable_lt)),
    ]);

    let registry = CoderRegistry::default();
    let file = File::create(path)?;
    let mut writer = HeliumWriter::new(file, schema, &registry)?;

    // 3 stripes of 5000 rows each. v is null when (row % 3 == 0), so nulls and
    // non-nulls straddle the 5000-row stripe boundaries and the 2048 chunk size.
    let stripe_rows = 5000usize;
    let mut global_row = 0i64;
    for _stripe in 0..3 {
        let ids: Vec<i64> = (0..stripe_rows as i64).map(|r| global_row + r).collect();
        let mut present = Vec::with_capacity(stripe_rows);
        let mut compact: Vec<i64> = Vec::new();
        for r in 0..stripe_rows {
            let abs = global_row + r as i64;
            if abs % 3 == 0 {
                present.push(false);
            } else {
                present.push(true);
                compact.push(abs * 2);
            }
        }
        writer.write_column("id", LogicalColumn::Primitive(ColumnData::I64(ids)))?;
        writer.write_column(
            "v",
            LogicalColumn::Nullable {
                present,
                value: Box::new(LogicalColumn::Primitive(ColumnData::I64(compact))),
            },
        )?;
        writer.finish_stripe()?;
        global_row += stripe_rows as i64;
    }
    writer.finish()?;
    Ok(())
}
