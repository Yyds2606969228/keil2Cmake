# Keil2Cmake

**English** | [中文](README.md)

Keil uVision -> CMake converter (ARM-GCC only, with clangd support).

## Features
- Parse `.uvprojx` and generate a CMake project (ARM-GCC only)
- Sync CMake project settings back to Keil `.uvprojx`
- Generate OpenOCD/cortex-debug templates
- Auto-generate `.clangd` and build/static-analysis presets

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

### 5. Sync CMake project back to Keil
```bash
Keil2Cmake sync-keil --uvprojx ./project.uvprojx --cmake-root ./cmake_project
```

This updates source/include/define/flags from `cmake/user/keil2cmake_user.cmake` into the target `.uvprojx`.

## TinyML (Extracted)

TinyML has been extracted into the `src/k2c_tinyml/` subproject, and is no longer part of `Keil2Cmake.exe`.

Examples:
```bash
uv run --with jinja2 --with onnx --with numpy --with onnxruntime python src/k2c_tinyml/scripts/K2CTinyML.py onnx --model model.onnx --weights flash --emit c
uv run --with jinja2 --with onnx --with numpy --with onnxruntime python -m unittest discover -s src/k2c_tinyml/tests -v
```

## openocd_mcp Module

`openocd_mcp` has been split as an independent subproject:

- runtime code: `src/openocd_mcp/src/openocd_mcp`
- tests: `src/openocd_mcp/tests/openocd_mcp`
- entrypoint: `src/openocd_mcp/scripts/openocd_mcp.py`
- docs: `src/openocd_mcp/docs/debug_runtime/`

## Release Notes

- Release index: `docs/releases/README.md`
- Latest stable release: `release/2.0.0` (2026-02-26)
- Release notes: `docs/releases/release-2.0/RELEASE_NOTES.md`
- Upgrade guide: `docs/releases/release-2.0/UPGRADE_GUIDE.md`

TinyML operator coverage, quantization policy, and validation details are documented in `src/k2c_tinyml/`.

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
        ├── keil2cmake_user.cmake
        └── cppcheck.cmake

Generated directly after conversion:
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
- checkcpp args: configure `K2C_CHECKCPP_ENABLE` or switch-style `K2C_CHECKCPP_ENABLE_*` (ON/OFF) to build `--enable`, plus `K2C_CHECKCPP_JOBS` / `K2C_CHECKCPP_EXCLUDES` / `K2C_CHECKCPP_INCONCLUSIVE` in `cmake/user/cppcheck.cmake`.
- OpenOCD debug: `.vscode/launch.json` (cortex-debug) and `.vscode/tasks.json` (download) both use `cmake/user/openocd.cfg`; running `cmake --preset keil2cmake` can override and regenerate debug files via `K2C_OPENOCD_PATH` / `K2C_OPENOCD_TARGET` / `K2C_OPENOCD_INTERFACE`.
- Re-sync and overwrite OpenOCD files: `Keil2Cmake openocd -mcu <MCU> -debugger <daplink|jlink|stlink> --overwrite` updates `.vscode/launch.json`, `.vscode/tasks.json`, and `cmake/user/openocd.cfg`.
- CMake->Keil sync defaults to in-place update with `.bak` backup, and only manages the `K2C_Sync` group unless you choose another group.

---

**[GitHub](https://github.com/Yyds2606969228/keil2Cmake)**
**[Gitee](https://gitee.com/yyds6589/keil2cmake)**
