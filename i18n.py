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
        "cli.show_config.toolchains": "当前工具链配置:",
        "cli.show_config.includes": "当前头文件路径配置:",
        "cli.show_config.ninja": "Ninja 配置:",
        "cli.show_config.cmake": "CMake配置:",
        "cli.done": "✓ 成功生成 CMake 工程配置",
        "cli.summary.project": "项目",
        "cli.summary.device": "设备",
        "cli.summary.compiler": "编译器",
        "cli.summary.optimize": "优化等级",
        "cli.help.description": "Keil uVision 转 CMake 工具 (支持 ARM 嵌入式工具链和 clangd)",
        "cli.help.uvprojx": "Keil .uvprojx 项目文件路径",
        "cli.help.output": "输出目录。默认：自动从 .uvprojx 推导（MDK-ARM → 父目录）",
        "cli.help.clean": "清理生成的 CMake 文件",
        "cli.help.lang": "语言设置：zh（中文）或 en（英文）",
        "cli.help.compiler": "覆盖编译器：armcc / armclang / armgcc",
        "cli.help.optimize": "覆盖优化等级：0/1/2/3/s",
        "cli.help.edit": "编辑配置，格式：KEY=VALUE（如 ARMCC_PATH=D:/path）",
        "cli.help.show_config": "显示当前工具链和头文件路径配置",
        "cli.help.examples": """示例:
  %(prog)s project.uvprojx                    # 转换 Keil 项目为 CMake
  %(prog)s -e ARMCC_PATH=D:/Keil/ARMCC/bin/  # 修改编译器路径
  %(prog)s -e ARMCC_INCLUDE=D:/Keil/include/ # 修改头文件路径
  %(prog)s --show-config                      # 显示当前配置
  %(prog)s -o ./build project.uvprojx         # 指定输出目录
  %(prog)s --clean -o .                       # 清理生成的文件
  %(prog)s --lang en project.uvprojx          # 使用英文输出""",
        "cli.summary.output": "输出目录",
        "cli.build_cmds": "✓ 构建命令:",
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
        "clean.done": "✓ 清理完成: 已移除 {count} 个 keil2cmake 生成文件",
        "clean.none": "✓ 清理完成: 未发现可移除的 keil2cmake 生成文件",
        # Config
        "config.updated": "已更新配置: {configkey} = {value}",
        "config.error.format": "错误: 编辑格式应为 KEY=VALUE, 但得到的是: {value}",
        "config.error.invalid_key": "错误: 无效的配置键 '{configkey}'。有效的键: {valid}",

        # Generated CMake comments
        "gen.user.header.title": "# Keil2Cmake 生成的用户配置文件",
        "gen.user.header.safe": "# 可安全编辑: 源文件/头文件/宏/flags。",
        "gen.user.header.no_overwrite": "# 重新运行生成器时，若文件已存在将不会覆盖。",
        "gen.user.defaults": "# Keil 工程默认设置",
        "gen.user.optimize": "# 覆盖优化等级: 0/1/2/3/s。留空 = 使用 Keil 默认值",
        "gen.user.linker": "# 可选的链接器脚本覆盖。留空时，工具链使用 cmake/internal 下的默认值。",
        "gen.toolchain.header.title": "# Keil2Cmake 自动生成的工具链文件",
        "gen.toolchain.select_compiler": "# 选择编译器（通常由 CMakePresets.json 提供）",
        "gen.toolchain.linker_scripts": "# 默认链接器脚本位于 cmake/internal（可通过 K2C_LINKER_SCRIPT_* 覆盖）",
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
        "cli.done": "✓ CMake project generated successfully",
        "cli.summary.project": "Project",
        "cli.summary.device": "Device",
        "cli.summary.compiler": "Compiler",
        "cli.summary.optimize": "Optimization",
        "cli.help.description": "Keil uVision to CMake converter for ARM Embedded Toolchains (with clangd support)",
        "cli.help.uvprojx": "Path to Keil .uvprojx project file",
        "cli.help.output": "Output directory. Default: auto-derived from .uvprojx (MDK-ARM → parent)",
        "cli.help.clean": "Clean generated CMake files",
        "cli.help.lang": "Language setting: zh (Chinese) or en (English)",
        "cli.help.compiler": "Override compiler: armcc / armclang / armgcc",
        "cli.help.optimize": "Override optimization level: 0/1/2/3/s",
        "cli.help.edit": "Edit config in format KEY=VALUE (e.g., ARMCC_PATH=D:/path)",
        "cli.help.show_config": "Show current toolchain and include paths configuration",
        "cli.help.examples": """Examples:
  %(prog)s project.uvprojx                    # Convert Keil project to CMake
  %(prog)s -e ARMCC_PATH=D:/Keil/ARMCC/bin/  # Edit compiler path
  %(prog)s -e ARMCC_INCLUDE=D:/Keil/include/ # Edit include path
  %(prog)s --show-config                      # Show current configuration
  %(prog)s -o ./build project.uvprojx         # Specify output directory
  %(prog)s --clean -o .                       # Clean generated files
  %(prog)s --lang zh project.uvprojx          # Use Chinese output""",
        "cli.summary.output": "Output",
        "cli.build_cmds": "✓ Build commands:",
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
        "clean.done": "✓ Clean complete: removed {count} keil2cmake generated files",
        "clean.none": "✓ Clean complete: no keil2cmake generated files found",
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
        "gen.toolchain.linker_scripts": "# Default linker scripts live in cmake/internal (can be overridden via K2C_LINKER_SCRIPT_*)",
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
