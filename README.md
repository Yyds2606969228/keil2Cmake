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
```

## 说明
- `cmake --build --preset check` 使用 `CHECKCPP_PATH` 指定的工具进行静态分析。
- `.clangd` 已针对 ARM-GCC 做了 sysroot/内部头文件处理。
- `K2C_USE_NEWLIB_NANO=ON` 可启用 newlib-nano；如需浮点支持，设置 `K2C_NEWLIB_NANO_PRINTF_FLOAT=ON` / `K2C_NEWLIB_NANO_SCANF_FLOAT=ON`。
- 兼容层：自动识别 Keil 使用的 ARMCC/ARMCLANG，并将 MicroLIB 设置映射为 newlib-nano 的默认开关（可手动覆盖）。
- 汇编兼容：检测到 `.s/.asm` 会直接加入构建（ARMASM 需手动改写为 GCC 语法）；可手动设置 `K2C_GCC_STARTUP` 指定 GCC 启动文件。
- 链接脚本转换：若 Keil 工程配置了 `.sct`，会生成 `cmake/internal/keil2cmake_from_sct.ld` 并作为默认 GCC 链接脚本（严格模板转换，需自行核对内存布局；解析失败会回退到默认脚本）。
- checkcpp 参数：可在 `cmake/user/keil2cmake_user.cmake` 配置 `K2C_CHECKCPP_ENABLE` 或一组开关 `K2C_CHECKCPP_ENABLE_*`（ON/OFF）来生成 `--enable`；同时支持 `K2C_CHECKCPP_JOBS` / `K2C_CHECKCPP_EXCLUDES` / `K2C_CHECKCPP_INCONCLUSIVE`。

---

**[GitHub](https://github.com/Yyds2606969228/keil2Cmake)**
**[Gitee](https://gitee.com/yyds6589/keil2cmake)**
