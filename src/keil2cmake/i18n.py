# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any


_SUPPORTED_LANGS = ("zh", "en")


_MESSAGES: dict[str, dict[str, str]] = {
    "zh": {
        # CLI
        "cli.error.uvprojx_required": "必须提供 .uvprojx 文件路径",
        "cli.error.clean_requires_target": "使用 --clean 时必须提供 .uvprojx 或通过 -o 指定要清理的输出目录",
        "cli.error.file_not_found": "错误: 文件不存在 - {path}",
        "cli.show_config.toolchains": "工具链配置:",
        "cli.show_config.includes": "头文件路径配置:",
        "cli.show_config.ninja": "Ninja 配置:",
        "cli.show_config.cmake": "CMake 配置:",
        "cli.done": "完成：已生成 CMake 工程配置",
        "cli.summary.project": "项目",
        "cli.summary.device": "设备",
        "cli.summary.compiler": "编译器",
        "cli.summary.keil_compiler": "Keil 编译器",
        "cli.summary.microlib": "MicroLIB",
        "cli.summary.optimize": "优化等级",
        "cli.help.description": "Keil uVision 转 CMake 工具（仅 ARM-GCC，含 clangd 支持）",
        "cli.help.uvprojx": "Keil .uvprojx 项目文件路径",
        "cli.help.output": "输出目录",
        "cli.help.clean": "清理生成的 CMake 文件",
        "cli.help.lang": "语言设置：zh（中文）或 en（英文）",
        "cli.help.compiler": "覆盖编译器：armgcc",
        "cli.help.optimize": "覆盖优化等级：0/1/2/3/s",
        "cli.help.edit": "编辑配置：KEY=VALUE",
        "cli.help.show_config": "显示当前配置",
        "cli.help.examples": """示例:
  %(prog)s project.uvprojx -o ./cmake_project
""",
        "cli.summary.output": "输出目录",
        "cli.build_cmds": "构建命令:",
        "cli.compat.note": "已启用 ARMCC/ARMCLANG 兼容模式，生成 ARM-GCC 工程（部分选项可能需要手工调整）",
        "cli.compat.armasm": "检测到汇编源文件，已加入构建（ARMASM 需重写为 GCC 语法）。",
        "cli.openocd.done": "完成：已生成 OpenOCD/Cortex-Debug 模板",
        "cli.openocd.summary.debugger": "调试器",
        "cli.openocd.summary.output": "输出目录",
        "cli.openocd.summary.openocd_path": "OpenOCD 路径",
        "cli.openocd.summary.gdb_path": "GDB 路径",
        "cli.openocd.summary.interface": "OpenOCD 接口",
        "cli.openocd.summary.target": "OpenOCD 目标",
        "cli.openocd.summary.executable": "ELF 模板",
        "cli.openocd.summary.launch": "launch.json",
        "cli.openocd.summary.tasks": "tasks.json",
        "cli.openocd.summary.config": "openocd.cfg",
        "cli.onnx.done": "完成：已生成 TinyML 产物",
        "cli.onnx.summary.model": "模型",
        "cli.onnx.summary.output": "输出目录",
        "cli.onnx.summary.backend": "后端",
        "cli.onnx.summary.quant": "量化",
        "cli.onnx.summary.weights": "权重位置",
        "cli.onnx.summary.emit": "产物类型",
        "cli.onnx.summary.header": "头文件",
        "cli.onnx.summary.source": "源文件",
        "cli.onnx.summary.manifest": "清单",
        "cli.onnx.summary.library": "静态库",
        "cli.onnx.summary.validation": "一致性校验",
        "cli.onnx.validation.passed": "通过",
        "cli.onnx.validation.skipped": "跳过",
        "cli.onnx.validation.failed": "失败",
        # Keil parsing
        "uvprojx.get_target": "读取目标信息...",
        "uvprojx.collect_sources": "收集源文件...",
        "uvprojx.set_includes": "读取头文件路径...",
        "uvprojx.load_defines": "读取宏定义...",
        "uvprojx.scatter": "读取 scatter/linker 脚本...",
        "uvprojx.device": "读取芯片信息...",
        "uvprojx.compiler": "读取编译器类型...",
        "uvprojx.flags": "读取编译/汇编/链接选项...",
        "uvprojx.optimize": "读取优化等级...",
        # Cleaning
        "clean.done": "清理完成：已移除 {count} 个 keil2cmake 生成文件",
        "clean.none": "清理完成：未发现可移除的 keil2cmake 生成文件",
        # Config
        "config.updated": "已更新配置 {configkey} = {value}",
        "config.error.format": "错误: 编辑格式应为 KEY=VALUE，实际为 {value}",
        "config.error.invalid_key": "错误: 无效的配置键 '{configkey}'。有效键: {valid}",

        # Generated CMake comments
        "gen.user.header.title": "# Keil2Cmake 生成的用户配置文件",
        "gen.user.header.safe": "# 可安全编辑：源文件/头文件/宏/flags。",
        "gen.user.header.no_overwrite": "# 重新运行生成器时，若文件已存在将不会覆盖。",
        "gen.user.defaults": "# Keil 工程默认设置",
        "gen.user.optimize": "# 覆盖优化等级: 0/1/2/3/s。留空 = 使用 Keil 默认值。",
        "gen.user.linker": "# 可选的 linker 脚本覆盖。留空时使用 cmake/internal 下的默认值。",
        "gen.toolchain.header.title": "# Keil2Cmake 自动生成的工具链文件",
        "gen.toolchain.select_compiler": "# 选择编译器（通常由 CMakePresets.json 提供）",
        "gen.toolchain.linker_scripts": "# 默认 linker 脚本位于 cmake/internal（可通过 K2C_LINKER_SCRIPT_LD 覆盖）",
        "gen.toolchain.bom_removed": "已从链接脚本中移除 BOM 字符",
        "gen.toolchain.sct_converted": "已从 scatter 生成 GCC 链接脚本",
        "gen.toolchain.sct_failed": "scatter 自动转换失败，已回退到默认链接脚本",
    },
    "en": {
        # CLI
        "cli.error.uvprojx_required": "A .uvprojx path is required",
        "cli.error.clean_requires_target": "With --clean you must provide a .uvprojx or specify -o for the output root to clean",
        "cli.error.file_not_found": "Error: file does not exist - {path}",
        "cli.show_config.toolchains": "Toolchain configuration:",
        "cli.show_config.includes": "Include paths configuration:",
        "cli.show_config.ninja": "Ninja configuration:",
        "cli.show_config.cmake": "CMake configuration:",
        "cli.done": "Done: CMake project generated",
        "cli.summary.project": "Project",
        "cli.summary.device": "Device",
        "cli.summary.compiler": "Compiler",
        "cli.summary.keil_compiler": "Keil Compiler",
        "cli.summary.microlib": "MicroLIB",
        "cli.summary.optimize": "Optimization",
        "cli.help.description": "Keil uVision to CMake converter (ARM-GCC only, with clangd support)",
        "cli.help.uvprojx": "Path to Keil .uvprojx project file",
        "cli.help.output": "Output directory",
        "cli.help.clean": "Clean generated CMake files",
        "cli.help.lang": "Language setting: zh (Chinese) or en (English)",
        "cli.help.compiler": "Override compiler: armgcc",
        "cli.help.optimize": "Override optimization level: 0/1/2/3/s",
        "cli.help.edit": "Edit config in format KEY=VALUE",
        "cli.help.show_config": "Show current configuration",
        "cli.help.examples": """Examples:
  %(prog)s project.uvprojx -o ./cmake_project
""",
        "cli.summary.output": "Output",
        "cli.build_cmds": "Build commands:",
        "cli.compat.note": "ARMCC/ARMCLANG compatibility enabled, generating ARM-GCC project (some options may need manual adjustment).",
        "cli.compat.armasm": "ASM sources detected and included in build (ARMASM must be rewritten to GCC syntax).",
        "cli.openocd.done": "Done: OpenOCD/Cortex-Debug templates generated",
        "cli.openocd.summary.debugger": "Debugger",
        "cli.openocd.summary.output": "Output",
        "cli.openocd.summary.openocd_path": "OpenOCD Path",
        "cli.openocd.summary.gdb_path": "GDB Path",
        "cli.openocd.summary.interface": "OpenOCD Interface",
        "cli.openocd.summary.target": "OpenOCD Target",
        "cli.openocd.summary.executable": "ELF Template",
        "cli.openocd.summary.launch": "launch.json",
        "cli.openocd.summary.tasks": "tasks.json",
        "cli.openocd.summary.config": "openocd.cfg",
        "cli.onnx.done": "Done: TinyML artifacts generated",
        "cli.onnx.summary.model": "Model",
        "cli.onnx.summary.output": "Output",
        "cli.onnx.summary.backend": "Backend",
        "cli.onnx.summary.quant": "Quant",
        "cli.onnx.summary.weights": "Weights",
        "cli.onnx.summary.emit": "Emit",
        "cli.onnx.summary.header": "Header",
        "cli.onnx.summary.source": "Source",
        "cli.onnx.summary.manifest": "Manifest",
        "cli.onnx.summary.library": "Library",
        "cli.onnx.summary.validation": "Consistency Check",
        "cli.onnx.validation.passed": "passed",
        "cli.onnx.validation.skipped": "skipped",
        "cli.onnx.validation.failed": "failed",
        # Keil parsing
        "uvprojx.get_target": "Reading target info...",
        "uvprojx.collect_sources": "Collecting source files...",
        "uvprojx.set_includes": "Reading include paths...",
        "uvprojx.load_defines": "Reading preprocessor defines...",
        "uvprojx.scatter": "Reading scatter/linker settings...",
        "uvprojx.device": "Reading device info...",
        "uvprojx.compiler": "Detecting compiler...",
        "uvprojx.flags": "Reading C/ASM/LD flags...",
        "uvprojx.optimize": "Reading optimization level...",
        # Cleaning
        "clean.done": "Clean complete: removed {count} keil2cmake generated files",
        "clean.none": "Clean complete: no keil2cmake generated files found",
        # Config
        "config.updated": "Updated: {configkey} = {value}",
        "config.error.format": "Error: edit format must be KEY=VALUE, got: {value}",
        "config.error.invalid_key": "Error: invalid key '{configkey}'. Valid keys: {valid}",

        # Generated CMake comments
        "gen.user.header.title": "# Generated user configuration for Keil2Cmake",
        "gen.user.header.safe": "# Safe to edit: sources/includes/defines/flags.",
        "gen.user.header.no_overwrite": "# If you re-run the generator, this file is NOT overwritten if it already exists.",
        "gen.user.defaults": "# Defaults from Keil project",
        "gen.user.optimize": "# Override optimize level: 0/1/2/3/s. Empty = use Keil default.",
        "gen.user.linker": "# Optional linker overrides. If empty, toolchain sets defaults under cmake/internal.",
        "gen.toolchain.header.title": "# Auto-generated toolchain by Keil2Cmake",
        "gen.toolchain.select_compiler": "# Select compiler (usually provided by CMakePresets.json)",
        "gen.toolchain.linker_scripts": "# Default linker scripts live in cmake/internal (can be overridden via K2C_LINKER_SCRIPT_LD)",
        "gen.toolchain.bom_removed": "Removed BOM from linker script",
        "gen.toolchain.sct_converted": "Converted scatter file to GCC linker script",
        "gen.toolchain.sct_failed": "Scatter conversion failed; falling back to default linker script",
    },
}


_current_lang = "zh"


def normalize_lang(lang: str | None) -> str:
    if not lang:
        return "zh"
    lang = str(lang).strip().lower()
    if lang in ("cn", "zh-cn", "zh_cn"):
        return "zh"
    if lang in ("en-us", "en_us"):
        return "en"
    return lang if lang in _SUPPORTED_LANGS else "zh"


def set_language(lang: str | None) -> str:
    global _current_lang
    _current_lang = normalize_lang(lang)
    return _current_lang


def get_language() -> str:
    return _current_lang


def t(key: str, **kwargs: Any) -> str:
    lang = _current_lang
    table = _MESSAGES.get(lang, _MESSAGES["zh"])
    template = table.get(key) or _MESSAGES["zh"].get(key) or key
    try:
        return template.format(**kwargs)
    except Exception:
        return template
