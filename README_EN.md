# Keil2Cmake

[中文](README.md) | **English**

Keil uVision to CMake converter (v3.0), supporting three ARM toolchains, CMake Presets, and internationalization.

## ✨ Features

- 🔄 **Auto Conversion** Keil .uvprojx to CMake + CMakePresets.json
- 🛠️ **Three Toolchains** ARMCC (C5) / ARMCLANG (C6) / ARM-GCC
- 🌍 **i18n** Chinese/English bilingual (`--lang zh/en`)
- 🎯 **Smart Parsing** Auto-detect compiler type and optimization level
- 💡 **IDE Integration** Auto-generate `.clangd` config
- 📁 **Simplified Structure** Single toolchain + single user config file

## 🚀 Quick Start

### 1. Configure Compiler

```bash
Keil2Cmake -e ARMCC_PATH=D:/Keil_v5/ARM/ARMCC/bin/
Keil2Cmake -e ARMCC_INCLUDE=D:/Keil_v5/ARM/ARMCC/include/
Keil2Cmake --show-config  # View config
```

### 2. Convert Project

```bash
Keil2Cmake project.uvprojx           # Basic conversion
Keil2Cmake --lang en project.uvprojx # English output
```

### 3. Build

```bash
cmake --preset keil2cmake            # Use default compiler
cmake --build --preset keil2cmake

# Or switch compiler
cmake --preset keil2cmake-armclang
cmake --preset keil2cmake-armgcc
```

## 📋 Command Parameters

```bash
Keil2Cmake --help  # View full help
```

| Parameter | Description |
|-----------|-------------|
| `uvprojx` | Keil project file |
| `-o DIR` | Output directory (auto-derived by default) |
| `--compiler` | Override compiler: armcc/armclang/armgcc |
| `--optimize` | Override optimization: 0/1/2/3/s |
| `--lang` | Language: zh/en |
| `--clean` | Clean generated files |
| `-e KEY=VAL` | Edit config |
| `--show-config` | Show config |

**CMake Variables**:
- `K2C_COMPILER` - Compiler selection
- `K2C_OPTIMIZE_LEVEL` - Optimization level
- `K2C_LINKER_SCRIPT_SCT` / `K2C_LINKER_SCRIPT_LD` - Linker script override

View CMake options:
```bash
cmake --build --preset keil2cmake --target show-options
```

## 📁 Generated Files

```
project_root/
├── CMakeLists.txt           # Main build file
├── CMakePresets.json        # Preset config
├── .clangd                  # IDE code completion
└── cmake/
    ├── internal/            # ⚠️ Auto-generated, do not edit
    │   ├── toolchain.cmake
    │   ├── keil2cmake_default.sct
    │   └── keil2cmake_default.ld
    └── user/
        └── keil2cmake_user.cmake  # ✏️ User editable
```

**User Editable**: `cmake/user/keil2cmake_user.cmake`
- Source/header/macro lists
- Override optimization level and linker scripts

## ⚙️ Configuration

Config location: `~/.keil2cmake/config.json`

**Configurable Items**:
- `ARMCC_PATH` / `ARMCLANG_PATH` / `ARMGCC_PATH` - Compiler paths
- `ARMCC_INCLUDE` / `ARMCLANG_INCLUDE` - System headers
- `ARMGCC_SYSROOT` / `ARMGCC_INCLUDE` - GCC config
- `LANGUAGE` - Default language (zh/en)
- `MIN_VERSION` - Minimum CMake version

## 🔧 Optimization Levels

Keil `<Optim>` auto-mapping:

| Keil | ARMCC | ARMCLANG | GCC |
|------|-------|----------|-----|
| 0 | -O0 | -O0 | -O0 |
| 1 | -O1 | -O1 | -O1 |
| 2 | -O2 | -O2 | -O2 |
| 3 | -O3 | -O3 | -O3 |
| 4 | -O1 | -O1 | -O1 |
| 11 | -Ospace | -Oz | -Os |

## ❓ FAQ

**Compiler not found**
```bash
Keil2Cmake -e ARMCC_PATH=D:/Keil_v5/ARM/ARMCC/bin/
```

**Header files not found**
```bash
Keil2Cmake -e ARMCC_INCLUDE=D:/Keil_v5/ARM/ARMCC/include/
```

**Clangd not working**
- Check if `.clangd` file exists
- Reload VS Code (Ctrl+Shift+P → "Reload Window")

**View detailed output**
```bash
cmake --preset keil2cmake --debug-output
cmake --build build --verbose
```

## 📦 Development

```bash
# Clone and install
git clone https://github.com/Yyds2606969228/keil2Cmake.git
cd Keil2Cmake
pip install -r requirements.txt

# Run tests
python -m unittest discover -s tests -v

# Build executable (recommended using spec config)
pyinstaller Keil2Cmake.spec

# Or use command line
pyinstaller -F --name Keil2Cmake \
  --exclude-module tkinter \
  --hidden-import keil2cmake_cli \
  --hidden-import keil2cmake_common \
  --hidden-import i18n \
  --collect-submodules keil \
  --collect-submodules compiler \
  Keil2Cmake.py

# Generated: dist/Keil2Cmake.exe (Windows) or dist/Keil2Cmake (Linux/Mac)
```

## 📝 Changelog

### v3.0 (2026-01)
- ✨ CMake Presets + simplified structure
- ✨ Chinese/English i18n + smart compiler detection
- ✨ Optimization level mapping fix (ARMCC/ARMCLANG/GCC)
- ✨ Built-in help system (`--help` + `show-options`)

### v2.0
- ✅ Dynamic config + clangd support

### v1.0
- 🎉 Initial release

---

⭐ **[GitHub](https://github.com/Yyds2606969228/keil2Cmake)**
⭐ **[Gitee](https://gitee.com/yyds6589/keil2cmake)**
