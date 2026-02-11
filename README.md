# Keil2Cmake

**中文** | [English](README_EN.md)

Keil uVision -> CMake 转换工具（仅 ARM-GCC，含 clangd 支持）。

## 功能
- 自动解析 `.uvprojx` 并生成 CMake 工程
- 仅支持 ARM-GCC 工具链
- 自动生成 `.clangd` 配置
- 预置构建与静态分析命令

## 快速开始
### 1. 配置工具路径
配置文件：`~/.config/keil2cmake/path.cfg`

示例：
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

### 2. 生成 CMake 工程
```bash
Keil2Cmake project.uvprojx -o ./cmake_project
```

### 3. 构建与静态分析
```bash
cmake --preset keil2cmake
cmake --build --preset build
cmake --build --preset check
```

### 4. 仅生成 OpenOCD/Cortex-Debug 模板（已有 CMake 工程）
在 CMake 工程根目录执行：
```bash
Keil2Cmake openocd -mcu STM32F103C8 -debugger jlink
```

### 5. TinyML（ONNX -> C/静态库）
```bash
Keil2Cmake onnx --model model.onnx --backend c --quant int8 --weights flash --emit c
```
生成 ONNX Opset12 覆盖矩阵：
```bash
uv run --with onnx python scripts/generate_opset12_coverage.py
```
输出文件：`docs/onnx_opset12_coverage_matrix.md`

已支持算子：Add/Sub/Mul/Div/Max/Min/Pow（含常见广播）、Equal/Greater/Less/GreaterOrEqual/LessOrEqual、And/Or/Xor/Not、ArgMax/ArgMin、Abs/Neg/Exp/Erf/Sign/Sin/Cos/Log/Reciprocal/Sqrt/Floor/Ceil/Round、Relu/LeakyRelu/Elu/Selu/HardSigmoid/Sigmoid/Tanh/Softplus/Softsign/Clip、MatMul/Gemm、Softmax(静态 rank>=1)、Reshape/Flatten/Squeeze/Unsqueeze/Identity/Cast/Gather/GatherND/GatherElements/ScatterElements/ScatterND/Expand/Where/Tile/Resize、Conv(2D/NCHW)/ConvTranspose(2D/NCHW, group=1, fp32)、MaxPool/AveragePool、GlobalAveragePool/GlobalMaxPool、BatchNormalization/InstanceNormalization/LRN、Concat、Transpose、Pad(constant)、Slice、ReduceMean/ReduceSum/ReduceMax/ReduceMin/ReduceProd/ReduceL1/ReduceL2/ReduceSumSquare（支持 axes/keepdims）、SpaceToDepth/DepthToSpace。
当前支持 `backend=c` 与 `backend=cmsis-nn`（暂不集成 ESP-NN）；CLI 默认 `quant=int8`，`quant` 支持 `fp32/int8/int16`。`int8/int16` 需模型使用 Q/DQ（QuantizeLinear/DequantizeLinear）节点；目前量化计算覆盖 `Cast/Gather/GatherND/GatherElements/ScatterElements/ScatterND/Expand/Where/Tile/Resize/SpaceToDepth/DepthToSpace/Add/Sub/Mul/Div/Relu/LeakyRelu/Elu/Selu/HardSigmoid/Sigmoid/Tanh/Softplus/Softsign/Exp/Erf/Sign/Sin/Cos/Log/Reciprocal/Sqrt/Floor/Ceil/Round/Pow/Identity/Abs/Neg/Clip/Max/Min/Conv/MatMul/Gemm/MaxPool/AveragePool/GlobalAveragePool/GlobalMaxPool/Squeeze/Unsqueeze/ReduceProd/ReduceL1/ReduceL2/ReduceSumSquare`。当 `backend=cmsis-nn` 时，算子优先命中 CMSIS-NN，不支持则回退纯 C；若纯 C 也不支持则报错。
转换阶段默认执行一致性校验（基于 ONNX ReferenceEvaluator；默认容差 `rtol=1e-3`、`atol=1e-4`，环境不支持或模型超限将自动跳过）。
如需在 Windows 上启用“生成 C 代码后再运行比对”的一致性回归，请在执行测试前注入宿主机 GCC（仅作用于当前终端会话，不写入 `path.cfg`）：
```powershell
$env:CC = "C:/Users/qwer/Downloads/winlibs-x86_64-posix-seh-gcc-15.2.0-mingw-w64ucrt-13.0.0-r5/mingw64/bin/gcc.exe"
$env:PATH = "C:/Users/qwer/Downloads/winlibs-x86_64-posix-seh-gcc-15.2.0-mingw-w64ucrt-13.0.0-r5/mingw64/bin;$env:PATH"
uv run --with jinja2 --with onnx --with numpy --with onnxruntime python -m unittest tests.test_tinyml -v
```

## 生成文件结构
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

配置后生成（执行 `cmake --preset keil2cmake`）：
```
.vscode/launch.json
.vscode/tasks.json
cmake/user/openocd.cfg
```

已有 CMake 工程也可执行：
```
Keil2Cmake openocd -mcu <MCU> -debugger <daplink|jlink|stlink>
```

## 说明
- `cmake --build --preset check` 使用 `CHECKCPP_PATH` 指定的工具进行静态分析。
- `.clangd` 已针对 ARM-GCC 做了 sysroot/内部头文件处理。
- `K2C_USE_NEWLIB_NANO=ON` 可启用 newlib-nano；如需浮点支持，设置 `K2C_NEWLIB_NANO_PRINTF_FLOAT=ON` / `K2C_NEWLIB_NANO_SCANF_FLOAT=ON`。
- 兼容层：自动识别 Keil 使用的 ARMCC/ARMCLANG，并将 MicroLIB 设置映射为 newlib-nano 的默认开关（可手动覆盖）。
- 汇编兼容：检测到 `.s/.asm` 会直接加入构建（ARMASM 需手动改写为 GCC 语法）；可手动设置 `K2C_GCC_STARTUP` 指定 GCC 启动文件。
- 链接脚本转换：若 Keil 工程配置了 `.sct`，会生成 `cmake/internal/keil2cmake_from_sct.ld` 并作为默认 GCC 链接脚本（严格模板转换，需自行核对内存布局；解析失败会回退到默认脚本）。
- checkcpp 参数：可在 `cmake/user/keil2cmake_user.cmake` 配置 `K2C_CHECKCPP_ENABLE` 或一组开关 `K2C_CHECKCPP_ENABLE_*`（ON/OFF）来生成 `--enable`；同时支持 `K2C_CHECKCPP_JOBS` / `K2C_CHECKCPP_EXCLUDES` / `K2C_CHECKCPP_INCONCLUSIVE`。
- OpenOCD 调试：执行 `cmake --preset keil2cmake` 后生成 `.vscode/launch.json`（cortex-debug）和 `.vscode/tasks.json`（下载任务）；用户需在 `cmake/user/openocd.cfg` 中设置调试器与芯片配置。调试器在 `CMakePresets.json` 中通过 `K2C_DEBUG_PROBE` 手动选择（`daplink`/`jlink`/`stlink`）；芯片型号会尝试自动填充 `K2C_OPENOCD_TARGET` 默认值；`OPENOCD_PATH` 来自 `path.cfg`。
- 仅生成模板：`Keil2Cmake openocd -mcu <MCU> -debugger <daplink|jlink|stlink>` 会在当前目录生成 `.vscode/launch.json`、`.vscode/tasks.json` 和 `cmake/user/openocd.cfg`（若已存在则不覆盖）。

---

**[GitHub](https://github.com/Yyds2606969228/keil2Cmake)**
**[Gitee](https://gitee.com/yyds6589/keil2cmake)**
