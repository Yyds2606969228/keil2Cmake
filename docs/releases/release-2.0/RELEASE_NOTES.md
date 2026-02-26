# Keil2Cmake Release 2.0.0

- Release tag: `release/2.0`
- Release date: `2026-02-26`
- Scope: CLI, TinyML runtime/validation, config management, packaging consistency, release docs

## Highlights

1. TinyML consistency validation is strict by default.
2. CLI keeps only `--no-strict-validation` as the explicit relax switch.
3. Config module is read/write separated:
   - `load_config()` is read-only and does not write to disk.
   - `save_config()` is responsible for persistence.
4. ONNX/TinyML heavy dependencies are lazy-loaded to avoid help-path noise.
5. Packaging spec and source tree are kept consistent by automated tests.

## Key Changes

1. Validation policy
   - `strict_validation=True` by default.
   - `status=failed` always blocks generation.
   - `status=skipped` also blocks generation in strict mode.
2. CLI parameters
   - retained: `--model --weights --emit --output --no-strict-validation`
   - removed: `--strict-validation` explicit positive switch
3. Config behavior
   - no implicit write side effects in `load_config()`
   - explicit write path via `save_config()`
4. Packaging cleanup
   - removed stale `hiddenimports`
   - removed stale tinyml shim module path from packaging
   - added packaging consistency tests
5. Technical debt cleanup
   - replaced broad exception handling with typed exception boundaries in critical paths
   - replaced silent `OSError` swallowing with debug-level observability logs
   - removed old clean-up compatibility branches for legacy generator layouts

## Compatibility Notes

1. Existing scripts using `--strict-validation` must be updated (argument no longer accepted).
2. If you intentionally allow skipped consistency checks, pass `--no-strict-validation`.
3. Tools relying on `load_config()` side-effect writes must migrate to `save_config()`.
4. `clean_generated` now targets current generated layout only; old historical layout files are no longer cleaned automatically.

## Verification Summary

1. Unit/integration tests: `249 passed`.
2. Packaging consistency checks added and passing.
3. TinyML strict/non-strict validation paths covered by regression tests.

## Known Warnings (Non-blocking)

1. `onnxruntime` Windows warning in current local environment.
2. upstream deprecation warnings from protobuf/numpy-related dependencies.
