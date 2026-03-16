# Keil2Cmake

**中文** | [English](README_EN.md)

Keil uVision -> CMake 转换工具（ARM-GCC + CMake），并集成 OpenOCD 调试/下载配置。

## 功能概览

1. `keil2cmake`：解析 `.uvprojx`，生成 ARM-GCC CMake 工程（含 `.clangd`、Presets、链接脚本转换）。
2. `openocd`：快速生成/更新 `openocd.cfg` 与 VSCode 调试任务，支持 preset 覆盖。
4. `sync-keil`：将 `cmake/user/keil2cmake_user.cmake` 的源文件/路径/宏/flags 同步回 `.uvprojx`。

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

### 4. 启动 MCP 调试运行时

```bash
Keil2Cmake sync-keil --uvprojx ./project.uvprojx --cmake-root ./cmake_project
```

该子命令会把 CMake 用户配置同步回 Keil 工程，并更新指定 `.uvprojx`（会生成 `.bak` 备份）。参考：

- `src/openocd_mcp/docs/debug_runtime/README.md`
- `src/openocd_mcp/docs/debug_runtime/api_contract.md`
- `src/openocd_mcp/docs/debug_runtime/state_model.md`
- `src/openocd_mcp/docs/debug_runtime/error_codes.md`

## EXE 分发建议

Windows 下建议采用 **双入口分发**：

- `Keil2Cmake.exe`：主入口，负责工程生成、构建、OpenOCD 配置与全局方向编排
- `openocd-mcp.exe`：服务入口，位于独立子项目 `src/openocd_mcp/`

推荐关系：

- `Keil2Cmake.exe` 仍然是主入口
- `openocd-mcp.exe` 只承载调试、串口、取证、运行时自动化等 MCP 能力（与 `keil2cmake` 解耦）
- 全局编排仍留在 `keil2cmake.orchestrator`，不会因为双入口分发而失效

这样设计的原因是：

- `stdio` MCP 更适合使用独立的 console 可执行文件承载
- 普通工程转换/构建路径不需要被 MCP 运行时耦合
- 调试运行时按需启动，不会默认常驻，也不会与主 CLI 路径争用固定端口

TinyML 已抽离到 `src/k2c_tinyml/` 子项目（独立 CLI/EXE），不再包含在 `Keil2Cmake` 中。

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

## Debug Runtime（openocd_mcp 独立子项目）

`openocd-mcp` 已拆分为独立子项目，主要提供：

- OpenOCD TCL RPC 连接与目标控制
- 串口环形缓冲与关键字触发
- 外设、SVD、ELF 解析
- Python 沙箱任务运行时
- 面向 Agent 的 MCP 服务接口

推荐定位是：

- `keil2cmake`：工程生成、构建、配置、工件产出
- `openocd-mcp`：连接、取证、分析、自动化验证

当前仓库已提供面向 LLM / Agent 的全局方向编排层，用于管理：

- 工程生成与 configure / build 路由
- ELF / SVD / openocd.cfg 等工件登记
- 调试准备与技能分流
- triage / analysis / regression 的方向回流

该编排能力当前定位为：

- **由 Skill 驱动**：供 LLM / Agent 作为全局流程指导使用
- **由 orchestrator 模块承载**：以 `Goal -> Signal -> DirectionDecision -> AgentWorkItem` 为核心抽象，不以 CLI 子命令作为主入口
- **与双入口分发兼容**：`Keil2Cmake.exe` 聚焦工程转换，`openocd-mcp.exe` 由独立子项目提供

建议优先阅读：

- `src/openocd_mcp/docs/debug_runtime/orchestration.md`
- `src/openocd_mcp/skills/software-loop-orchestrator/SKILL.md`

## TinyML（已抽离）

TinyML 已抽离到 `src/k2c_tinyml/` 子项目（独立 CLI/EXE）。下方旧说明将逐步迁移到子项目文档。
更多说明见：`src/k2c_tinyml/README.md` 与 `src/k2c_tinyml/tinyML框架设计方案.md`。

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
