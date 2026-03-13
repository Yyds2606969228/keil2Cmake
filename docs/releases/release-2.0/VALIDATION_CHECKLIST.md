# Release 2.0.0 Validation Checklist

## A. Source and docs alignment

1. [ ] `README.md` links to latest release notes.
2. [ ] `README_EN.md` links to latest release notes.
3. [ ] `docs/releases/README.md` indexes `release/2.0.0`.
4. [ ] Release notes and upgrade guide are present and versioned.

## B. CLI behavior checks

1. [ ] `Keil2Cmake onnx --help` contains `--no-strict-validation`.
2. [ ] `--strict-validation` is rejected as unsupported.
3. [ ] strict mode is default when no switch is provided.

## C. TinyML validation policy

1. [ ] `status=failed` blocks artifact generation.
2. [ ] strict mode + `status=skipped` blocks artifact generation.
3. [ ] `--no-strict-validation` allows generation when status is `skipped`.

## D. Config behavior

1. [ ] `load_config()` does not write files.
2. [ ] `save_config()` is the only persistence path.
3. [ ] `edit_config()` persists through `save_config()`.

## E. Packaging consistency

1. [ ] `Keil2Cmake.spec` has no stale hiddenimports.
2. [ ] removed shim paths are not reintroduced.
3. [ ] `tests/test_packaging_spec.py` passes.

## F. Regression status

1. [ ] `python -m pytest -q` passes.
2. [ ] Warnings are reviewed and classified as non-blocking or blocking.
3. [ ] Release artifacts are generated in target environment (if applicable).
