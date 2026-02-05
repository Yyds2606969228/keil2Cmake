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
```

## Notes
- `cmake --build --preset check` runs static analysis using `CHECKCPP_PATH`.
- `.clangd` is tailored for ARM-GCC sysroot/internal includes.
- Enable newlib-nano with `K2C_USE_NEWLIB_NANO=ON`. For float support, set `K2C_NEWLIB_NANO_PRINTF_FLOAT=ON` / `K2C_NEWLIB_NANO_SCANF_FLOAT=ON`.
- Compatibility layer: detect ARMCC/ARMCLANG from Keil project, map MicroLIB to the default newlib-nano switch (override if needed).
- ASM compatibility: `.s/.asm` sources are included; ARMASM must be rewritten to GCC syntax. You can set `K2C_GCC_STARTUP` to point to a GCC startup file.
- Linker conversion: if a Keil `.sct` is provided, `cmake/internal/keil2cmake_from_sct.ld` is generated and used as the default GCC linker script (strict template conversion; please verify memory layout. If parsing fails, it falls back to the default script).
- checkcpp args: configure `K2C_CHECKCPP_ENABLE` or switch-style `K2C_CHECKCPP_ENABLE_*` (ON/OFF) to build `--enable`, plus `K2C_CHECKCPP_JOBS` / `K2C_CHECKCPP_EXCLUDES` / `K2C_CHECKCPP_INCONCLUSIVE` in `cmake/user/keil2cmake_user.cmake`.

---

**[GitHub](https://github.com/Yyds2606969228/keil2Cmake)**
**[Gitee](https://gitee.com/yyds6589/keil2cmake)**
