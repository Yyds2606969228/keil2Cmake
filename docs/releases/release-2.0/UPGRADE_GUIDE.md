# Upgrade Guide: 1.x -> 2.0.0

## 1. CLI migration

1. Remove `--strict-validation` from scripts.
2. Default behavior is already strict; no extra flag needed.
3. Use `--no-strict-validation` only when skipped consistency checks are acceptable.

Example:

```bash
# 2.0 default (strict)
Keil2Cmake onnx --model model.onnx

# 2.0 relaxed
Keil2Cmake onnx --model model.onnx --no-strict-validation
```

## 2. Python API migration (`config`)

1. `load_config()` now only reads and fills defaults in memory.
2. Persist config changes through `save_config(config)`.

Example:

```python
from keil2cmake.keil.config import load_config, save_config

cfg = load_config()
cfg["PATHS"]["ARMGCC_PATH"] = "D:/Toolchains/arm-gcc/bin"
save_config(cfg)
```

## 3. Packaging migration

1. Do not rely on removed tinyml shim module path.
2. Keep packaging config aligned with source modules only.
3. Use `tests/test_packaging_spec.py` as the contract for packaging consistency.

## 4. Cleanup behavior migration

1. `clean_generated` now cleans current generated layout only.
2. If your workspace still contains very old generator outputs, remove them manually once.

Suggested manual cleanup targets in old projects:

```text
cmake/internal/armcc/
cmake/internal/armclang/
cmake/internal/armgcc/
cmake/internal/common/
cmake/user/armcc/
cmake/user/armclang/
cmake/user/armgcc/
cmake/user/common/
```

## 5. Release validation commands

```bash
python -m pytest -q
```

Optional TinyML command smoke:

```bash
Keil2Cmake onnx --model model.onnx
Keil2Cmake onnx --model model.onnx --no-strict-validation
```
