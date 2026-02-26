# Keil2Cmake

**中文** | [English](README_EN.md)

Keil uVision -> CMake 转换工具（ARM-GCC + CMake），并集成 OpenOCD 调试/下载配置与 TinyML（ONNX -> C）。

## 功能概览

1. `keil2cmake`：解析 `.uvprojx`，生成 ARM-GCC CMake 工程（含 `.clangd`、Presets、链接脚本转换）。
2. `openocd`：快速生成/更新 `openocd.cfg` 与 VSCode 调试任务，支持 preset 覆盖。
3. `tinyml`：ONNX 模型转 C 代码或静态库，面向 MCU 侧部署。

## 路径配置

配置文件：`~/.config/keil2cmake/path.cfg`

```ini
[PATHS]
ARMGCC_PATH = D:/Toolchains/arm-gcc/bin
CMAKE_PATH = C:/Program Files/CMake/bin/cmake.exe
NINJA_PATH = D:/Tools/ninja/ninja.exe
CHECKCPP_PATH = D:/Tools/cppcheck/cppcheck.exe
OPENOCD_PATH = D:/Tools/openocd/bin/openocd.exe

[GENERAL]
LANGUAGE = zh
```

## 快速开始

### 1. 从 Keil 工程生成 CMake 工程

```bash
Keil2Cmake project.uvprojx -o ./cmake_project
```

### 2. 配置/构建/静态分析

```bash
cmake --preset keil2cmake
cmake --build --preset build
cmake --build --preset check
```

### 3. 为已有 CMake 工程生成 OpenOCD 调试配置

```bash
Keil2Cmake openocd -mcu STM32F103C8 -debugger jlink
```

如需覆盖更新现有配置：

```bash
Keil2Cmake openocd -mcu <MCU> -debugger <daplink|jlink|stlink> --overwrite
```

### 4. TinyML（ONNX -> C/静态库）

```bash
Keil2Cmake onnx --model model.onnx --weights flash --emit c
```

严格一致性校验（默认开启，可显式开关）：

```bash
Keil2Cmake onnx --model model.onnx
Keil2Cmake onnx --model model.onnx --no-strict-validation
```

生成 Opset12 覆盖矩阵：

```bash
uv run --with onnx python scripts/generate_opset12_coverage.py
```

输出：`docs/onnx_opset12_coverage_matrix.md`

## 发布说明

- 发布说明目录：`docs/releases/README.md`
- 最新正式版本：`release/2.0.0`（2026-02-26）
- 最新发布说明：`docs/releases/release-2.0/RELEASE_NOTES.md`
- 升级指南：`docs/releases/release-2.0/UPGRADE_GUIDE.md`

## OpenOCD 说明

- 实际生效配置文件：`cmake/user/openocd.cfg`。
- `openocd.cfg` 采用 `INTERFACE + TARGET + TRANSPORT` 组合配置，不使用 `BOARD`。
- `cmake --preset keil2cmake` 支持覆盖：
  - `K2C_OPENOCD_PATH`
  - `K2C_OPENOCD_TARGET`
  - `K2C_OPENOCD_INTERFACE`
  - `K2C_OPENOCD_TRANSPORT`
- `K2C_OPENOCD_TARGET` 与 `K2C_OPENOCD_INTERFACE` 会根据 Keil 工程自动给出默认值（可覆盖）。
- `.vscode/launch.json` 和 `.vscode/tasks.json` 统一引用 `cmake/user/openocd.cfg`。

## TinyML 说明

- 后端生成链路：C 后端。
- 量化策略：模型导出阶段完成；转换阶段依据 Q/DQ（`QuantizeLinear`/`DequantizeLinear`）与张量 dtype 自动匹配 `int8/int16` 路径；无 Q/DQ 时默认 `fp32`。
- MCU 覆盖类型：`fp32/int8/int16`，内存布局按 4 字节对齐。
- ONNX Opset12 覆盖：`162/162`（含约束子集，详见覆盖矩阵）。
- `RNN/GRU/LSTM` 支持 `fp32/int8/int16`（限制项以覆盖矩阵为准）。
- 转换阶段默认执行一致性校验（ONNX ReferenceEvaluator，默认 `rtol=1e-3`、`atol=1e-4`）。
- 严格模式默认开启；若一致性校验被 `skipped` 也会直接失败（可通过 `--no-strict-validation` 放宽）。

Windows 下如需执行“生成 C 后再比对”的一致性回归：

```powershell
$env:CC = "C:/path/to/arm-gcc/bin/gcc.exe"
$env:PATH = "C:/path/to/arm-gcc/bin;$env:PATH"
uv run --with jinja2 --with onnx --with numpy --with onnxruntime python -m unittest tests.test_tinyml -v
```

## 生成文件结构

```text
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
        ├── keil2cmake_user.cmake
        ├── cppcheck.cmake
        └── openocd.cfg
```

并生成 VSCode 调试文件：

```text
.vscode/launch.json
.vscode/tasks.json
```

## 其他说明

- `cmake --build --preset check` 使用 `CHECKCPP_PATH` 对应工具进行静态分析。
- `.clangd` 已针对 ARM-GCC 做 sysroot/内部头文件处理。
- `K2C_USE_NEWLIB_NANO=ON` 可启用 newlib-nano；浮点支持可开启：
  - `K2C_NEWLIB_NANO_PRINTF_FLOAT=ON`
  - `K2C_NEWLIB_NANO_SCANF_FLOAT=ON`
- 兼容层会识别 Keil 的 ARMCC/ARMCLANG，并映射 MicroLIB 到 newlib-nano 默认开关（可覆盖）。
- 若 Keil 工程含 `.sct`，会生成 `cmake/internal/keil2cmake_from_sct.ld` 作为默认 GCC 链接脚本（解析失败回退默认脚本）。
- ARMASM 源文件需手动改写为 GCC 语法；可通过 `K2C_GCC_STARTUP` 指定 GCC 启动文件。

---

**[GitHub](https://github.com/Yyds2606969228/keil2Cmake)**
**[Gitee](https://gitee.com/yyds6589/keil2cmake)**
