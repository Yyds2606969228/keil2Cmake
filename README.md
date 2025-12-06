# Keil2Cmake

Keil uVision 到 CMake 转换工具 (v2.0)，支持动态配置编译器头文件路径，适配 clangd 代码提示。

本项目基于 [https://github.com/LoveApple14434/Keil2Cmake](https://github.com/LoveApple14434/Keil2Cmake) 进行了二次开发，主要改进包括：
- 添加了 clangd 智能提示支持
- 优化了编译器路径配置管理
- 优化了生成cmake的命令行参数

## 功能特性

- ✅ **自动转换** Keil uVision 项目 (.uvprojx) 到 CMake 构建系统
- ✅ **编译器支持** 同时支持 armcc 和 armclang 编译器
- ✅ **智能配置** 动态配置编译器和头文件路径
- ✅ **IDE集成** 完美支持 clangd 代码提示和智能补全
- ✅ **多平台** 支持 Windows、Linux 和 macOS
- ✅ **优化等级** 可配置编译优化等级 (-O0 到 -O3, -Os)
- ✅ **自动生成** 生成 .clangd 配置文件，提供 VS Code 代码支持

## 安装方式

### 方式一：使用可执行文件（推荐）

1. 从 [Releases](https://github.com/yourusername/Keil2Cmake/releases) 下载最新版本
2. 解压并将可执行文件加入系统路径

### 方式二：从源码运行

```bash
# 克隆仓库
git clone https://gitee.com/yyds6589/keil2cmake.git
cd Keil2Cmake

# 安装依赖
pip install -r requirements.txt

# 会在dist目录下生成exe可执行文件
pyinstaller -F --exclude-module tkinter Keil2Cmake.py
```

## 快速开始

### 配置编译器路径

首次使用需要配置编译器路径：

```bash
# 配置 ARMCC 编译器路径
Keil2Cmake.exe -e ARMCC_PATH=D:/Keil_v5/ARM/ARMCC/bin/

# 配置 ARMCLANG 编译器路径
Keil2Cmake.exe -e ARMCLANG_PATH=D:/Keil_v5/ARM/ARMCLANG/bin/

# 配置系统头文件路径
Keil2Cmake.exe -e ARMCC_INCLUDE=D:/Keil_v5/ARM/ARMCC/include/
Keil2Cmake.exe -e ARMCLANG_INCLUDE=D:/Keil_v5/ARM/ARMCLANG/include/

# 查看当前配置
Keil2Cmake.exe --show-config
```

### 构建项目

```bash
# 创建构建目录并配置
cmake -G Ninja -B build -DCOMPILE_NAME=armclang -DOPTIMIZE_LEVEL=2 .

# 构建项目
cmake --build build

```

## 命令行参数

| 参数 | 说明 |
|------|------|
| `uvprojx` | Keil 项目文件路径（必需） |
| `-o, --output` | 输出目录（默认：当前目录） |
| `--compiler` | 指定编译器类型：armcc 或 armclang |
| `-e, --edit` | 编辑配置：KEY=VALUE 格式 |
| `-sc, --show-config` | 显示当前配置 |

## 配置选项

### 工具链配置

- `ARMCC_PATH` - ARMCC 编译器路径
- `ARMCLANG_PATH` - ARMCLANG 编译器路径

### 头文件配置

- `ARMCC_INCLUDE` - ARMCC 系统头文件路径
- `ARMCLANG_INCLUDE` - ARMCLANG 系统头文件路径

### CMake 配置

- `MIN_VERSION` - 最低 CMake 版本要求（默认：3.20）

## 支持的设备

工具自动识别 STM32 系列设备并配置相应的 CPU 架构：

- **Cortex-M0**: STM32F0, STM32L0
- **Cortex-M0+**: STM32G0
- **Cortex-M3**: STM32F1, STM32L1, STM32F2
- **Cortex-M4**: STM32F3, STM32L4, STM32G4, STM32F4, STM32WB, STM32WL
- **Cortex-M7**: STM32F7, STM32H7
- **Cortex-M33**: STM32L5, STM32U5

## 生成的文件说明

转换成功后，将在输出目录生成以下文件：

### CMakeLists.txt
主构建文件，包含：
- 项目配置和源文件列表
- 编译选项和预处理器定义
- 自定义构建命令（生成 HEX/BIN 文件）
- 动态系统头文件路径配置

### toolchain.cmake
工具链配置文件，包含：
- ARM 交叉编译器配置
- CPU 架构和编译选项
- 链接器设置

### .clangd
Clangd 配置文件，提供：
- VS Code 代码智能提示
- 头文件路径解析
- 语法检查和代码补全

### Template.sct
链接器脚本模板，可根据需要修改。

## 故障排除

### 常见问题

1. **编译器路径错误**
   ```
   解决方案：使用 -e 参数重新配置编译器路径
   ```

2. **找不到系统头文件**
   ```
   解决方案：确保配置了正确的 INCLUDE 路径
   ```

3. **clangd 无法正常工作**
   ```
   解决方案：检查 .clangd 配置文件中的头文件路径是否正确
   ```

### 调试模式

使用 CMake 调试模式查看详细输出：

```bash
cmake -G Ninja -B build -DCOMPILE_NAME=armclang -DCMAKE_VERBOSE_MAKEFILE=ON .
```

## 示例项目

完整的使用示例：

```bash
# 1. 配置编译器
Keil2Cmake.exe -e ARMCLANG_PATH=D:/Keil_v5/ARM/ARMCLANG/bin/
Keil2Cmake.exe -e ARMCLANG_INCLUDE=D:/Keil_v5/ARM/ARMCLANG/include/

# 2. 转换项目
Keil2Cmake.exe project.uvprojx -o ./cmake_build my_stm32_project.uvprojx

# 3. 进入输出目录
cd cmake_build

# 4. 配置和构建
cmake -G Ninja -B build -DCOMPILE_NAME=armclang -DOPTIMIZE_LEVEL=2 .
cmake --build build

```

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 更新日志

### v1.0
- 🎉 初始版本发布
- ✅ 基本的 uVision 到 CMake 转换功能

---

⭐ 如果这个项目对你有帮助，请给个星标支持一下！