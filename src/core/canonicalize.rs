//! Canonicalization for schema JSON, per PLAN_V2 §6.5 Surface B.
//!
//! The canonical form is what BLAKE3 hashes over — it must be **bit-stable**
//! across serde_json upgrades, struct-field reordering, and any user-side
//! whitespace differences. The contract:
//!
//! 1. **Lexicographic key ordering** at every object level (BTreeMap order).
//! 2. **No insignificant whitespace** between tokens.
//! 3. **UTF-8 NFC normalization** for object keys (string values are
//!    left as-is — they're user data, not schema metadata).
//! 4. **Integers as exact decimal**, no exponent (default `serde_json`
//!    behaviour).
//! 5. **Floats as shortest round-trip representation** (default Ryu via
//!    `serde_json`).
//!
//! This canonicalization spec is **wire-format-frozen** — the same stability
//! rule as coder IDs (PLAN_V2 §3 commitment 3). Changing rules invalidates
//! every existing catalog hash.
//!
//! The implementation deliberately avoids RFC 8785 / `json-canon` (extra dep
//! locked-in for full real-number generality we don't need) and does NOT just
//! call `serde_json::to_vec` on a `Schema` (couples the hash to serde_json's
//! struct-field internal ordering).

use serde_json::{Map, Value};
use unicode_normalization::UnicodeNormalization;

use super::error::{HeliumError, Result};
use super::schema::Schema;

/// Canonicalize a `Schema` and return its BLAKE3 hash. The same `Schema` value
/// always yields the same hash, regardless of `Schema::columns` field order
/// in memory or the `serde_json` version that produced the JSON.
///
/// Used by `HeliumWriter::with_catalog_ref` to assert that the caller-supplied
/// hash matches the schema's canonical hash, and by `helium-catalog` for
/// content-addressed catalog filenames (PLAN_V2 §6.5).
pub fn schema_hash(schema: &Schema) -> Result<blake3::Hash> {
    let raw = schema.to_json()?;
    let canonical = canonicalize_json(&raw).map_err(|e| {
        HeliumError::Format(format!(
            "failed to canonicalize schema JSON for hashing: {e}"
        ))
    })?;
    Ok(blake3::hash(&canonical))
}

/// Canonicalize a JSON byte slice. Returns the canonical bytes that should be
/// hashed (via BLAKE3) to obtain the catalog content-address.
///
/// The input must be valid JSON; non-JSON input returns the parse error
/// wrapped in a `serde_json::Error`.
pub fn canonicalize_json(input: &[u8]) -> serde_json::Result<Vec<u8>> {
    let value: Value = serde_json::from_slice(input)?;
    let canonical = canonicalize_value(value);
    // serde_json emits object keys in BTreeMap (alphabetical) order and no
    // insignificant whitespace by default — both required by the contract.
    serde_json::to_vec(&canonical)
}

/// Recursively canonicalize a parsed `Value` in-place: NFC-normalize object
/// keys and re-collect into a fresh `Map` so the keys are sorted.
fn canonicalize_value(v: Value) -> Value {
    match v {
        Value::Object(obj) => {
            // Re-collect into a BTreeMap-backed Map so keys end up sorted
            // when serialized. Apply NFC to each key.
            let mut sorted: Map<String, Value> = Map::new();
            for (k, child) in obj {
                let key_nfc: String = k.nfc().collect();
                sorted.insert(key_nfc, canonicalize_value(child));
            }
            Value::Object(sorted)
        }
        Value::Array(arr) => Value::Array(arr.into_iter().map(canonicalize_value).collect()),
        // Numbers: serde_json::Number serializes integers as exact decimal and
        // f64 via Ryu (shortest round-trip). Both align with the contract.
        // Strings: user data — never normalized (only object keys are).
        // Bool / Null: unchanged.
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two semantically-identical inputs with different key order must produce
    /// the same canonical bytes.
    #[test]
    fn key_order_independent() {
        let a = br#"{"b":2,"a":1}"#;
        let b = br#"{"a":1,"b":2}"#;
        assert_eq!(canonicalize_json(a).unwrap(), canonicalize_json(b).unwrap());
    }

    /// Whitespace must be stripped.
    #[test]
    fn whitespace_stripped() {
        let with_ws = br#"{
            "a" : 1 ,
            "b" : 2
        }"#;
        let without_ws = br#"{"a":1,"b":2}"#;
        assert_eq!(
            canonicalize_json(with_ws).unwrap(),
            canonicalize_json(without_ws).unwrap()
        );
    }

    /// Nested object/array combinations canonicalize recursively.
    #[test]
    fn recursive_canonicalization() {
        let weird = br#"{"z":[{"y":1,"x":2}],"a":{"c":3,"b":4}}"#;
        let canonical = canonicalize_json(weird).unwrap();
        let s = std::str::from_utf8(&canonical).unwrap();
        // a comes before z; b before c; x before y; arrays preserve order.
        assert_eq!(s, r#"{"a":{"b":4,"c":3},"z":[{"x":2,"y":1}]}"#);
    }

    /// Array element order is preserved (arrays are positional, not named).
    #[test]
    fn array_order_preserved() {
        let v = br#"[3, 1, 2]"#;
        let canonical = canonicalize_json(v).unwrap();
        assert_eq!(canonical, b"[3,1,2]");
    }

    /// Non-ASCII keys get NFC-normalized.
    /// `é` has two valid Unicode forms: precomposed (U+00E9) and decomposed
    /// (U+0065 U+0301). NFC produces the precomposed form. Both inputs must
    /// canonicalize to the same bytes.
    #[test]
    fn nfc_key_normalization() {
        // U+00E9 = 'é' (precomposed)
        let precomposed = "{\"caf\u{00E9}\":1}".as_bytes();
        // U+0065 + U+0301 = 'e' + combining acute accent (decomposed)
        let decomposed = "{\"caf\u{0065}\u{0301}\":1}".as_bytes();
        assert_eq!(
            canonicalize_json(precomposed).unwrap(),
            canonicalize_json(decomposed).unwrap()
        );
    }

    /// String *values* are not NFC-normalized — only object keys.
    /// (Schema string values are user payload, not metadata.)
    #[test]
    fn string_values_not_normalized() {
        let precomposed_value = "{\"k\":\"caf\u{00E9}\"}".as_bytes();
        let decomposed_value = "{\"k\":\"caf\u{0065}\u{0301}\"}".as_bytes();
        // These produce DIFFERENT canonical bytes — values are preserved verbatim.
        assert_ne!(
            canonicalize_json(precomposed_value).unwrap(),
            canonicalize_json(decomposed_value).unwrap()
        );
    }

    /// Integers serialize as exact decimal, no exponent.
    #[test]
    fn integers_no_exponent() {
        let v = br#"{"big":1000000000}"#;
        let canonical = canonicalize_json(v).unwrap();
        // Must NOT contain "e" or "E"
        assert!(!canonical.contains(&b'e'));
        assert!(!canonical.contains(&b'E'));
        assert_eq!(canonical, b"{\"big\":1000000000}");
    }

    /// Idempotence: canonicalize(canonicalize(x)) == canonicalize(x).
    #[test]
    fn idempotent() {
        let messy = br#"  { "z" : [3, 2, 1] , "a" : true }  "#;
        let once = canonicalize_json(messy).unwrap();
        let twice = canonicalize_json(&once).unwrap();
        assert_eq!(once, twice);
    }
}
