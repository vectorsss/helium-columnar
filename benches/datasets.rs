//! Shared synthetic dataset generators for benchmarks.
//!
//! All generators are deterministic — same inputs produce the same output
//! byte-for-byte across runs so benchmark comparisons stay honest.

#![allow(dead_code)]

pub fn timestamps_uniform(n: usize) -> Vec<i64> {
    (0..n).map(|i| 1_700_000_000 + i as i64 * 30).collect()
}

pub fn timestamps_jittered(n: usize) -> Vec<i64> {
    let mut v = 1_700_000_000i64;
    (0..n)
        .map(|i| {
            v += 30 + ((i as i64 * 2654435761) % 7);
            v
        })
        .collect()
}

pub fn measurements_gauge_f64(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 * 0.01;
            ((20.0 + t.sin() * 2.0) * 10.0).round() / 10.0
        })
        .collect()
}

pub fn measurements_scattered_i64(n: usize) -> Vec<i64> {
    (0..n).map(|i| -80 + (i as i64 * 17) % 25).collect()
}

pub fn tags_low_cardinality_i64(n: usize, card: usize) -> Vec<i64> {
    (0..n).map(|i| (i % card) as i64).collect()
}

pub fn strings_low_cardinality(n: usize, card: usize) -> Vec<String> {
    let dict: Vec<String> = (0..card).map(|i| format!("status_{i:03}")).collect();
    (0..n).map(|i| dict[i % card].clone()).collect()
}

pub fn sorted_unique_u32(n: usize, universe: u32) -> Vec<u32> {
    use std::collections::BTreeSet;
    let mut rng = 0x13374241u32;
    let mut set = BTreeSet::new();
    while set.len() < n {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        set.insert(rng % universe);
    }
    set.into_iter().collect()
}

pub fn nullable_f64(n: usize, null_rate: f64) -> (Vec<bool>, Vec<f64>) {
    let mut present = Vec::with_capacity(n);
    let mut values = Vec::new();
    for i in 0..n {
        let p = ((i as f64 * 0.618).fract()) >= null_rate;
        present.push(p);
        if p {
            values.push(70.0 + (i as f64 * 0.01).sin());
        }
    }
    (present, values)
}

pub fn unsigned_small_u32(n: usize, max: u32) -> Vec<u32> {
    (0..n).map(|i| (i as u32 * 13) % max).collect()
}
