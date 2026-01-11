# Keil2Cmake

**中文** | [English](README_EN.md)

Keil uVision 到 CMake 转换工具 (v3.0)，支持三大 ARM 工具链、CMake Presets、国际化输出。

## ✨ 功能特性

- 🔄 **自动转换** Keil .uvprojx 到 CMake + CMakePresets.json
- 🛠️ **三大工具链** ARMCC (C5) / ARMCLANG (C6) / ARM-GCC
- 🌍 **国际化** 中英文双语 (`--lang zh/en`)
- 🎯 **智能解析** 自动识别编译器类型和优化级别
- 💡 **IDE 集成** 自动生成 `.clangd` 配置
- 📁 **精简结构** 单一 toolchain + 单一用户配置文件

## 🚀 快速开始

### 1. 配置编译器

```bash
Keil2Cmake -e ARMCC_PATH=D:/Keil_v5/ARM/ARMCC/bin/
Keil2Cmake -e ARMCC_INCLUDE=D:/Keil_v5/ARM/ARMCC/include/
Keil2Cmake --show-config  # 查看配置
```

### 2. 转换项目

```bash
Keil2Cmake project.uvprojx           # 基本转换
Keil2Cmake --lang en project.uvprojx # 英文输出
```

### 3. 构建

```bash
cmake --preset keil2cmake            # 使用默认编译器
cmake --build --preset keil2cmake

# 或切换编译器
cmake --preset keil2cmake-armclang
cmake --preset keil2cmake-armgcc
```

## 📋 命令参数

```bash
Keil2Cmake --help  # 查看完整帮助
```

| 参数 | 说明 |
|------|------|
| `uvprojx` | Keil 项目文件 |
| `-o DIR` | 输出目录（默认自动推导）|
| `--compiler` | 覆盖编译器：armcc/armclang/armgcc |
| `--optimize` | 覆盖优化：0/1/2/3/s |
| `--lang` | 语言：zh/en |
| `--clean` | 清理生成文件 |
| `-e KEY=VAL` | 编辑配置 |
| `--show-config` | 显示配置 |

**CMake 变量**：
- `K2C_COMPILER` - 编译器选择
- `K2C_OPTIMIZE_LEVEL` - 优化级别
- `K2C_LINKER_SCRIPT_SCT` / `K2C_LINKER_SCRIPT_LD` - Linker 脚本覆盖

查看 CMake 选项：
```bash
cmake --build --preset keil2cmake --target show-options
```

## 📁 生成的文件

```
project_root/
├── CMakeLists.txt           # 主构建文件
├── CMakePresets.json        # 预设配置
├── .clangd                  # IDE 代码提示
└── cmake/
    ├── internal/            # ⚠️ 自动生成，勿编辑
    │   ├── toolchain.cmake
    │   ├── keil2cmake_default.sct
    │   └── keil2cmake_default.ld
    └── user/
        └── keil2cmake_user.cmake  # ✏️ 可编辑配置
```

**用户可编辑**：`cmake/user/keil2cmake_user.cmake`
- 源文件/头文件/宏定义列表
- 覆盖优化级别和 linker 脚本

## ⚙️ 配置文件

配置位置：`~/.keil2cmake/config.json`

**可配置项**：
- `ARMCC_PATH` / `ARMCLANG_PATH` / `ARMGCC_PATH` - 编译器路径
- `ARMCC_INCLUDE` / `ARMCLANG_INCLUDE` - 系统头文件
- `ARMGCC_SYSROOT` / `ARMGCC_INCLUDE` - GCC 配置
- `LANGUAGE` - 默认语言（zh/en）
- `MIN_VERSION` - 最低 CMake 版本

## 🔧 优化级别

Keil `<Optim>` 自动映射：

| Keil | ARMCC | ARMCLANG | GCC |
|------|-------|----------|-----|
| 0 | -O0 | -O0 | -O0 |
| 1 | -O1 | -O1 | -O1 |
| 2 | -O2 | -O2 | -O2 |
| 3 | -O3 | -O3 | -O3 |
| 4 | -O1 | -O1 | -O1 |
| 11 | -Ospace | -Oz | -Os |

## ❓ 常见问题

**找不到编译器**
```bash
Keil2Cmake -e ARMCC_PATH=D:/Keil_v5/ARM/ARMCC/bin/
```

**找不到头文件**
```bash
Keil2Cmake -e ARMCC_INCLUDE=D:/Keil_v5/ARM/ARMCC/include/
```

**Clangd 不工作**
- 检查 `.clangd` 文件是否存在
- 重启 VS Code（Ctrl+Shift+P → "Reload Window"）

**查看详细输出**
```bash
cmake --preset keil2cmake --debug-output
cmake --build build --verbose
```

## 📦 开发

```bash
# 克隆并安装
git clone https://gitee.com/yyds6589/keil2cmake.git
cd Keil2Cmake
pip install -r requirements.txt

# 运行测试
python -m unittest discover -s tests -v

# 构建可执行文件（推荐使用 spec 配置）
pyinstaller Keil2Cmake.spec

# 或使用命令行方式
pyinstaller -F --name Keil2Cmake \
  --exclude-module tkinter \
  --hidden-import keil2cmake_cli \
  --hidden-import keil2cmake_common \
  --hidden-import i18n \
  --collect-submodules keil \
  --collect-submodules compiler \
  Keil2Cmake.py

# 生成的可执行文件：dist/Keil2Cmake.exe (Windows) 或 dist/Keil2Cmake (Linux/Mac)
```

## 📝 更新日志

### v3.0 (2026-01)
- ✨ CMake Presets + 精简文件结构
- ✨ 中英文国际化 + 智能编译器识别
- ✨ 优化级别映射修复（ARMCC/ARMCLANG/GCC）
- ✨ 内置帮助系统（`--help` + `show-options`）

### v2.0
- ✅ 动态配置 + clangd 支持

### v1.0
- 🎉 初始版本

---

⭐ **[GitHub](https://github.com/Yyds2606969228/keil2Cmake)**
⭐ **[Gitee](https://gitee.com/yyds6589/keil2cmake)**
