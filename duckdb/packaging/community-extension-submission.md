# Submitting helium_duckdb to DuckDB community-extensions

This documents the steps to publish the extension through DuckDB's
[community-extensions](https://github.com/duckdb/community-extensions)
repository. **Nothing here is executed automatically** — submission is a
deliberate, manual action against an external repo.

## Prerequisites before submitting

- [x] `cargo build --release` clean.
- [x] `cargo clippy --all-targets -- -D warnings` clean.
- [x] `bash duckdb/smoke.sh` ends with `=== All smoke tests passed ===`.
- [x] The build/load matrix (`duckdb/ci/extension-matrix.yml`) is green for all
      target platforms.
- [ ] A tagged release of `helium-columnar` the extension can pin to (community
      builds run from a fixed git ref, not a local path dependency — see below).

## One blocker to resolve first: the path dependency

`duckdb/Cargo.toml` currently depends on the core crate by **path**:

```toml
helium = { path = "..", package = "helium-columnar", default-features = false }
```

The community-extensions build system checks out *only* the extension repo at a
pinned ref and builds it in isolation, so a `path = ".."` dependency will not
resolve. Before submission, switch to a git or crates.io dependency pinned to a
released ref, e.g.:

```toml
helium = { git = "https://github.com/helium-rs/helium", tag = "v0.2.0", \
           package = "helium-columnar", default-features = false }
```

Keep the path dep for local development on a branch; flip to the pinned dep in
the submission commit.

## Submission steps (manual)

1. **Fork** `duckdb/community-extensions`.
2. **Add a descriptor** at `extensions/helium_duckdb/description.yml`:

   ```yaml
   extension:
     name: helium_duckdb
     description: Read Helium .he columnar files via a read_he() table function
     version: 0.1.0
     language: Rust
     build: cargo
     license: MIT
     maintainers:
       - <your-github-handle>

   repo:
     github: helium-rs/helium
     ref: <commit-sha-or-tag>          # the pinned ref from the step above
     # The extension crate lives in a subdirectory of the repo.
     subdirectory: duckdb
   ```

3. **Verify the C-API version** in the descriptor / build config matches
   `CAPI_VERSION` in `packaging/package.sh` (`v1.2.0` today).
4. **Open a PR** to `duckdb/community-extensions`. CI there rebuilds the
   extension across their platform matrix and signs the artifacts.
5. After merge, users install with:

   ```sql
   INSTALL helium_duckdb FROM community;
   LOAD helium_duckdb;
   SELECT * FROM read_he('file.he') LIMIT 5;
   ```

   (Community-signed extensions do **not** require the `-unsigned` flag that
   locally-built artifacts need.)

## Notes

- The `repo.ref` pins the exact source; bump it on every release so the
  community build is reproducible.
- Re-run `duckdb/smoke.sh` against the *community-installed* extension after the
  first publish to confirm the signed artifact behaves identically to the
  locally-packaged one.
