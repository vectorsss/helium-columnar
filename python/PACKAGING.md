# Packaging pyhelium

pyhelium ships as a compiled extension built with [maturin] + [PyO3]. The
extension is built **abi3** (stable ABI, `cp39-abi3`), so one wheel per platform
loads on every CPython >= 3.9 — no per-minor-version wheel needed.

## What is in place

- **abi3 build.** `Cargo.toml` enables `pyo3`'s `abi3-py39` feature; `maturin`
  then emits a `cp39-abi3` wheel. Verify locally:

  ```bash
  cd python
  maturin build --release      # → target/wheels/pyhelium-<v>-cp39-abi3-<platform>.whl
  ```

- **Wheel matrix.** `[tool.cibuildwheel]` in `pyproject.toml` configures the
  build (Linux x86_64 + aarch64, macOS x86_64 + arm64, Windows AMD64), installs
  a Rust toolchain inside the Linux containers, and runs the test suite against
  each wheel.

- **CI workflow.** `.github/workflows/pyhelium-wheels.yml` runs the matrix on
  demand (`workflow_dispatch`) and on a `pyhelium-v*` tag. It builds wheels +
  an sdist and uploads them as artifacts. A `publish` job is wired but gated
  (see below).

- **Per-PR smoke.** The existing `bindings` job in `ci.yml` builds the wheel
  and runs `pytest` on every PR (now with `pyarrow` + `pandas` installed for the
  Arrow tests). The full matrix is reserved for the wheels workflow to keep PR
  CI fast.

## Publishing to PyPI — manual steps (NOT yet enabled)

Publishing is intentionally **not** automated until the project is registered
on PyPI. To enable it:

1. **Register the project on PyPI** (or TestPyPI first) and choose one of:
   - **Trusted publishing (recommended).** Configure a PyPI "trusted publisher"
     for this GitHub repo + the `pyhelium-wheels.yml` workflow + the `pypi`
     environment. The workflow already requests `id-token: write`, so no secret
     is needed.
   - **API token.** Create a PyPI API token and add it as the `PYPI_API_TOKEN`
     repository secret.

2. **Tag a release.** The publish job runs only on a `pyhelium-v*` tag:

   ```bash
   git tag pyhelium-v0.1.0
   git push origin pyhelium-v0.1.0
   ```

3. **Verify on TestPyPI first** (optional but recommended): point the
   `gh-action-pypi-publish` step at TestPyPI (`repository-url:
   https://test.pypi.org/legacy/`) for a dry run before the real index.

Until step 1 is done the `publish` job will fail auth and is effectively a
no-op; the build-wheels / build-sdist jobs still produce downloadable artifacts.

## Local one-off publish (fallback)

If you need to publish from a workstation instead of CI:

```bash
cd python
maturin build --release
# or build the full set of wheels with cibuildwheel locally, then:
maturin upload target/wheels/*.whl     # prompts for credentials / token
```

[maturin]: https://www.maturin.rs/
[PyO3]: https://pyo3.rs/
