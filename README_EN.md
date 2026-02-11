# Keil2Cmake

**English** | [中文](README.md)

Keil uVision -> CMake converter (ARM-GCC only, with clangd support).

## Features
- Parse `.uvprojx` and generate a CMake project
- ARM-GCC toolchain only
- Auto-generate `.clangd`
- Build and static-analysis presets

## Quick Start
### 1. Configure tool paths
Config file: `~/.config/keil2cmake/path.cfg`

Example:
```ini
[PATHS]
ARMGCC_PATH = D:/Toolchains/arm-gcc/bin
CMAKE_PATH = C:/Program Files/CMake/bin/cmake.exe
NINJA_PATH = D:/Tools/ninja/ninja.exe
CHECKCPP_PATH = D:/Tools/cppcheck/cppcheck.exe
OPENOCD_PATH = D:/Tools/openocd/bin/openocd.exe

[GENERAL]
LANGUAGE = en
```

### 2. Generate CMake project
```bash
Keil2Cmake project.uvprojx -o ./cmake_project
```

### 3. Build and static analysis
```bash
cmake --preset keil2cmake
cmake --build --preset build
cmake --build --preset check
```

### 4. Generate OpenOCD/Cortex-Debug templates (existing CMake project)
Run in your CMake project root:
```bash
Keil2Cmake openocd -mcu STM32F103C8 -debugger jlink
```

### 5. TinyML (ONNX -> C/static lib)
```bash
Keil2Cmake onnx --model model.onnx --backend c --quant int8 --weights flash --emit c
```
Generate ONNX Opset12 coverage matrix:
```bash
uv run --with onnx python scripts/generate_opset12_coverage.py
```
Output file: `docs/onnx_opset12_coverage_matrix.md`

Supported ops: Add/Sub/Mul/Div/Max/Min/Pow (common broadcast), Equal/Greater/Less/GreaterOrEqual/LessOrEqual, And/Or/Xor/Not, ArgMax/ArgMin, Abs/Neg/Exp/Erf/Sign/Sin/Cos/Log/Reciprocal/Sqrt/Floor/Ceil/Round, Relu/LeakyRelu/Elu/Selu/HardSigmoid/Sigmoid/Tanh/Softplus/Softsign/Clip, MatMul/Gemm, Softmax (static rank>=1), Reshape/Flatten/Squeeze/Unsqueeze/Identity/Cast/Gather/GatherND/GatherElements/ScatterElements/ScatterND/Expand/Where/Tile/Resize, Conv (2D/NCHW)/ConvTranspose (2D/NCHW, group=1, fp32), MaxPool/AveragePool, GlobalAveragePool/GlobalMaxPool, BatchNormalization/InstanceNormalization/LRN, Concat, Transpose, Pad (constant), Slice, ReduceMean/ReduceSum/ReduceMax/ReduceMin/ReduceProd/ReduceL1/ReduceL2/ReduceSumSquare (axes/keepdims supported), SpaceToDepth/DepthToSpace.
Supported backends: `backend=c` and `backend=cmsis-nn` (ESP-NN is deferred). CLI default is `quant=int8`; `quant` supports `fp32/int8/int16`. `int8/int16` requires Q/DQ (QuantizeLinear/DequantizeLinear) nodes; quantized compute covers `Cast/Gather/GatherND/GatherElements/ScatterElements/ScatterND/Expand/Where/Tile/Resize/SpaceToDepth/DepthToSpace/Add/Sub/Mul/Div/Relu/LeakyRelu/Elu/Selu/HardSigmoid/Sigmoid/Tanh/Softplus/Softsign/Exp/Erf/Sign/Sin/Cos/Log/Reciprocal/Sqrt/Floor/Ceil/Round/Pow/Identity/Abs/Neg/Clip/Max/Min/Conv/MatMul/Gemm/MaxPool/AveragePool/GlobalAveragePool/GlobalMaxPool/Squeeze/Unsqueeze/ReduceProd/ReduceL1/ReduceL2/ReduceSumSquare`. With `backend=cmsis-nn`, operators first try CMSIS-NN, then fallback to pure C; if pure C also lacks support, conversion fails.
Consistency checks run during conversion (based on ONNX ReferenceEvaluator; default tolerances `rtol=1e-3`, `atol=1e-4`); if unsupported or too large, validation is skipped.
To enable generated-C consistency regression on Windows, inject host GCC before running tests (session-only; do not write this into `path.cfg`):
```powershell
$env:CC = "C:/Users/qwer/Downloads/winlibs-x86_64-posix-seh-gcc-15.2.0-mingw-w64ucrt-13.0.0-r5/mingw64/bin/gcc.exe"
$env:PATH = "C:/Users/qwer/Downloads/winlibs-x86_64-posix-seh-gcc-15.2.0-mingw-w64ucrt-13.0.0-r5/mingw64/bin;$env:PATH"
uv run --with jinja2 --with onnx --with numpy --with onnxruntime python -m unittest tests.test_tinyml -v
```

## Generated Layout
```
project_root/
├── CMakeLists.txt
├── CMakePresets.json
├── .clangd
└── cmake/
    ├── internal/
    │   ├── toolchain.cmake
    │   ├── keil2cmake_default.ld
    │   └── keil2cmake_from_sct.ld
    └── user/
        └── keil2cmake_user.cmake

Generated after `cmake --preset keil2cmake`:
```
.vscode/launch.json
.vscode/tasks.json
cmake/user/openocd.cfg
```

For an existing CMake project, you can also run:
```
Keil2Cmake openocd -mcu <MCU> -debugger <daplink|jlink|stlink>
```

## Notes
- `cmake --build --preset check` runs static analysis using `CHECKCPP_PATH`.
- `.clangd` is tailored for ARM-GCC sysroot/internal includes.
- Enable newlib-nano with `K2C_USE_NEWLIB_NANO=ON`. For float support, set `K2C_NEWLIB_NANO_PRINTF_FLOAT=ON` / `K2C_NEWLIB_NANO_SCANF_FLOAT=ON`.
- Compatibility layer: detect ARMCC/ARMCLANG from Keil project, map MicroLIB to the default newlib-nano switch (override if needed).
- ASM compatibility: `.s/.asm` sources are included; ARMASM must be rewritten to GCC syntax. You can set `K2C_GCC_STARTUP` to point to a GCC startup file.
- Linker conversion: if a Keil `.sct` is provided, `cmake/internal/keil2cmake_from_sct.ld` is generated and used as the default GCC linker script (strict template conversion; please verify memory layout. If parsing fails, it falls back to the default script).
- checkcpp args: configure `K2C_CHECKCPP_ENABLE` or switch-style `K2C_CHECKCPP_ENABLE_*` (ON/OFF) to build `--enable`, plus `K2C_CHECKCPP_JOBS` / `K2C_CHECKCPP_EXCLUDES` / `K2C_CHECKCPP_INCONCLUSIVE` in `cmake/user/keil2cmake_user.cmake`.
- OpenOCD debug: generated after `cmake --preset keil2cmake` (`.vscode/launch.json` for cortex-debug, `.vscode/tasks.json` for download). Edit `cmake/user/openocd.cfg` to set probe/target. Manually select the probe in `CMakePresets.json` via `K2C_DEBUG_PROBE` (`daplink`/`jlink`/`stlink`); the device name is used to auto-fill a default `K2C_OPENOCD_TARGET` when possible; `OPENOCD_PATH` comes from `path.cfg`.
- Template-only: `Keil2Cmake openocd -mcu <MCU> -debugger <daplink|jlink|stlink>` generates `.vscode/launch.json`, `.vscode/tasks.json`, and `cmake/user/openocd.cfg` in the current directory (does not overwrite existing files).

---

**[GitHub](https://github.com/Yyds2606969228/keil2Cmake)**
**[Gitee](https://gitee.com/yyds6589/keil2cmake)**
