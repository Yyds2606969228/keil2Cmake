#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keil uVision 转 CMake 工具（v2.0）
支持动态配置编译器头文件路径，完美适配 clangd 代码提示
"""

import xml.etree.ElementTree as ET
import os
import argparse
import configparser
from pathlib import Path

def get_config_path():
    """获取配置文件路径"""
    config_dir = os.path.join(Path.home(), ".keil2cmake")
    os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, "path.cfg")

def load_config():
    """加载配置文件"""
    config_path = get_config_path()
    config = configparser.ConfigParser()
    
    # 如果配置文件不存在，创建默认配置
    if not os.path.exists(config_path):
        config['TOOLCHAINS'] = {
            'ARMCC_PATH': 'D:/Program/Keil_v5/ARM/ARM_Compiler_5.06u7/bin/',
            'ARMCLANG_PATH': 'D:/Program/Keil_v5/ARM/ARMCLANG/bin/',
        }
        config['INCLUDES'] = {
            'ARMCC_INCLUDE': 'D:/Program/Keil_v5/ARM/ARM_Compiler_5.06u7/include/',
            'ARMCLANG_INCLUDE': 'D:/Program/Keil_v5/ARM/ARMCLANG/include/',
        }
        config['CMAKE'] = {
            'MIN_VERSION': '3.20'
        }
        save_config(config)
    
    config.read(config_path)
    return config

def save_config(config):
    """保存配置文件"""
    with open(get_config_path(), 'w') as configfile:
        config.write(configfile)

def edit_config(edit_string):
    """编辑配置文件"""
    config = load_config()
    
    # 解析编辑字符串 (格式: KEY=VALUE)
    if '=' not in edit_string:
        print(f"错误: 编辑格式应为 KEY=VALUE, 但得到的是: {edit_string}")
        return False
    
    key, value = edit_string.split('=', 1)
    key = key.strip().upper()
    
    # 验证键是否有效
    valid_toolchain_keys = ['ARMCC_PATH', 'ARMCLANG_PATH']
    valid_include_keys = ['ARMCC_INCLUDE', 'ARMCLANG_INCLUDE']
    valid_cmake_keys = ['MIN_VERSION']
    valid_keys = valid_toolchain_keys + valid_include_keys + valid_cmake_keys
    
    if key not in valid_keys:
        print(f"错误: 无效的配置键 '{key}'。有效的键: {', '.join(valid_keys)}")
        return False
    
    # 根据键类型更新相应配置节
    if key in valid_toolchain_keys:
        config['TOOLCHAINS'][key] = value
    elif key in valid_include_keys:
        config['INCLUDES'][key] = value
    elif key in valid_cmake_keys:
        config['CMAKE'][key] = value
    
    save_config(config)
    print(f"已更新配置: {key} = {value}")
    return True

def get_toolchain_path(toolchain_type):
    """获取指定工具链的路径"""
    config = load_config()
    key = f"{toolchain_type}_PATH"
    return config['TOOLCHAINS'].get(key, '')

def get_include_path(compiler_type):
    """获取指定编译器的系统头文件路径"""
    config = load_config()
    key = f"{compiler_type}_INCLUDE"
    return config['INCLUDES'].get(key, '')

def get_cmake_min_version():
    """获取最低CMake版本"""
    config = load_config()
    return config['CMAKE'].get('MIN_VERSION', '3.20')

def detect_cpu_architecture(device_name):
    """根据 STM32 系列名称推断 CPU 架构"""
    device_upper = device_name.upper()
    
    series_to_cpu = {
        'STM32F0': 'Cortex-M0',
        'STM32L0': 'Cortex-M0+',
        'STM32G0': 'Cortex-M0+',
        'STM32F1': 'Cortex-M3',
        'STM32L1': 'Cortex-M3',
        'STM32F2': 'Cortex-M3',
        'STM32F3': 'Cortex-M4',
        'STM32L4': 'Cortex-M4',
        'STM32G4': 'Cortex-M4',
        'STM32F4': 'Cortex-M4',
        'STM32L5': 'Cortex-M33',
        'STM32U5': 'Cortex-M33',
        'STM32F7': 'Cortex-M7',
        'STM32H7': 'Cortex-M7',
        'STM32WB': 'Cortex-M4',
        'STM32WL': 'Cortex-M4',
    }
    
    for series, cpu in series_to_cpu.items():
        if series in device_upper:
            return cpu
    
    return "Cortex-M4"

def get_arm_arch_for_clang(cpu_arch):
    """将 CPU 架构转换为 ARMClang 所需的架构版本"""
    cpu_to_arm_arch = {
        'Cortex-M0':   'armv6-m',
        'Cortex-M0+':  'armv6-m',
        'Cortex-M1':   'armv6-m',
        'Cortex-M3':   'armv7-m',
        'Cortex-M4':   'armv7e-m',
        'Cortex-M7':   'armv7e-m',
        'Cortex-M23':  'armv8-m.base',
        'Cortex-M33':  'armv8-m.main',
        'Cortex-M35P': 'armv8-m.main',
        'Cortex-M55':  'armv8.1-m.main',
        'Cortex-M85':  'armv8.1-m.main',
    }
    return cpu_to_arm_arch.get(cpu_arch, 'armv7e-m')

def parse_uvprojx(uvprojx_path):
    """解析uvprojx文件，提取项目配置"""
    tree = ET.parse(uvprojx_path)
    root = tree.getroot()
    
    print("Getting target info...")
    project_name = root.find('.//Targets/Target/TargetName').text
    output_dir = root.find('.//Targets/Target/TargetOption/TargetCommonOption/OutputDirectory').text or 'build/'
    
    print("Collecting source files...")
    source_files = []
    for group in root.findall('.//Groups/Group'):
        for file in group.findall('Files/File'):
            file_path = file.find('FilePath').text
            if file_path.endswith(('.c', '.C', '.cpp', '.s', '.S', '.asm')):
                source_files.append(file_path)
    
    print("Setting include paths...")
    include_paths = []
    includes = root.find('.//TargetOption/TargetArmAds/Cads/VariousControls/IncludePath')
    if includes is not None and includes.text:
        include_paths.extend(includes.text.split(';'))
    
    print("Loading preset defines...")
    defines = []
    defs = root.find('.//TargetOption/TargetArmAds/Cads/VariousControls/Define')
    if defs is not None and defs.text:
        defines.extend([d.strip() for d in defs.text.split(',')])
    
    print("Looking for scatter file...")
    linker_script = None
    scatter_file = root.find('.//TargetOption/TargetArmAds/LDads/ScatterFile')
    if scatter_file is not None and scatter_file.text:
        linker_script = scatter_file.text
    
    print("Getting device info...")
    device_name = None
    device_node = root.find('.//Targets/Target/TargetOption/TargetCommonOption/Device')
    if device_node is not None and device_node.text:
        device_name = device_node.text
    
    print("Validating compilor version...")
    use_armclang = False
    armclang_node = root.find('.//TargetOption/TargetArmAds/UseArmClang')
    if armclang_node is not None and armclang_node.text == '1':
        use_armclang = True
    
    print("Setting options for compilors and linkers...")
    c_flags = root.find('.//TargetOption/TargetArmAds/Cads/VariousControls/MiscControls') 
    asm_flags = root.find('.//TargetOption/TargetArmAds/Aads/VariousControls/MiscControls') 
    ld_flags = root.find('.//TargetOption/TargetArmAds/LDads/VariousControls/MiscControls') 
    c_flags = c_flags.text if c_flags is not None and c_flags.text else ''
    asm_flags = asm_flags.text if asm_flags is not None and asm_flags.text else ''
    ld_flags = ld_flags.text if ld_flags is not None else ''
    
    print("Defining optimize level...")
    opt_level = "0"
    opt_node = root.find('.//TargetOption/TargetArmAds/Cads/Optimization')
    if opt_node is not None and opt_node.text:
        opt_level = opt_node.text
    
    print()
    
    return {
        'project_name': project_name,
        'source_files': source_files,
        'include_paths': include_paths,
        'defines': defines,
        'linker_script': linker_script,
        'device': device_name.strip() if device_name else "Unknown",
        'c_flags': c_flags,
        'asm_flags': asm_flags,
        'ld_flags': ld_flags,
        'output_dir': output_dir,
        'use_armclang': use_armclang,
        'opt_level': opt_level
    }

def generate_cmakelists(project_data, output_dir):
    """生成CMakeLists.txt文件（含动态系统头文件路径）"""
    # 处理用户头文件路径
    include_paths_formatted = [f'"{p}"' if ' ' in p else p for p in project_data['include_paths']]
    source_files_formatted = [f'"{f}"' if ' ' in f else f for f in project_data['source_files']]
    
    src_files = "\n    ".join(source_files_formatted).replace('\\', '/')
    inc_paths = "\n    ".join(include_paths_formatted).replace('\\', '/')
    defines = "\n    ".join(project_data['defines'])
    
    # 设置系统头文件路径变量（供 .clangd 使用）
    system_includes_code = '''
# 系统头文件路径（根据编译器类型自动选择）
if(COMPILE_NAME STREQUAL "armclang")
    set(SYSTEM_INCLUDE_PATH "''' + get_include_path('ARMCLANG').replace('\\', '/') + '''")
elseif(COMPILE_NAME STREQUAL "armcc")
    set(SYSTEM_INCLUDE_PATH "''' + get_include_path('ARMCC').replace('\\', '/') + '''")
endif()

# 将系统路径添加到编译器搜索路径
include_directories(SYSTEM ${SYSTEM_INCLUDE_PATH})
'''
    
    cmake_min_version = get_cmake_min_version()

    content = f'''cmake_minimum_required(VERSION {cmake_min_version})

# 启用生成 compile_commands.json (供 clangd 使用)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 包含工具链配置
set(CMAKE_TOOLCHAIN_FILE ${{CMAKE_CURRENT_SOURCE_DIR}}/toolchain.cmake)

project({project_data["project_name"]} LANGUAGES C ASM)

# 编译器选择参数
set(COMPILE_NAME "${{compiler_type}}" CACHE STRING "编译器类型 (armcc, armclang)")
set_property(CACHE COMPILE_NAME PROPERTY STRINGS armcc armclang)

# 优化等级参数
set(OPTIMIZE_LEVEL "{project_data['opt_level']}" CACHE STRING "优化等级 (0-3, s)")
set_property(CACHE OPTIMIZE_LEVEL PROPERTY STRINGS 0 1 2 3 s)

# 添加可执行文件
add_executable(${{PROJECT_NAME}}
    {src_files}
)

# 生成 hex/bin 文件
add_custom_command(TARGET ${{PROJECT_NAME}} POST_BUILD
    COMMAND ${{CMAKE_FROMELF}} --i32combined --output="${{CMAKE_RUNTIME_OUTPUT_DIRECTORY}}${{PROJECT_NAME}}.hex" "$<TARGET_FILE:${{PROJECT_NAME}}>"
    COMMENT "Generating HEX file"
)

add_custom_command(TARGET ${{PROJECT_NAME}} POST_BUILD
    COMMAND ${{CMAKE_FROMELF}} --bin --output="${{CMAKE_RUNTIME_OUTPUT_DIRECTORY}}${{PROJECT_NAME}}.bin" "$<TARGET_FILE:${{PROJECT_NAME}}>"
    COMMENT "Generating BIN file"
)

{system_includes_code}

# 用户头文件目录
target_include_directories(${{PROJECT_NAME}} PRIVATE
    {inc_paths}
)

# 预处理器定义
target_compile_definitions(${{PROJECT_NAME}} PRIVATE
    {defines}
)

# 链接器脚本
set(LINKER_SCRIPT "${{CMAKE_SOURCE_DIR}}/Template.sct")

# 链接选项
target_link_options(${{PROJECT_NAME}} PRIVATE
    ${{LINKER_FLAGS}}
    "--scatter=${{LINKER_SCRIPT}}"
    {project_data['ld_flags']}
)

# 编译器检查
if(NOT COMPILE_NAME)
    set(COMPILE_NAME "{"armclang" if project_data["use_armclang"] else "armcc"}")
endif()

message(STATUS "使用编译器: ${{COMPILE_NAME}}")
message(STATUS "优化等级: -O${{OPTIMIZE_LEVEL}}")
message(STATUS "系统头文件路径: ${{SYSTEM_INCLUDE_PATH}}")
'''

    with open(os.path.join(output_dir, 'CMakeLists.txt'), 'w', encoding="UTF-8") as f:
        f.write(content)

def generate_toolchain(project_data, output_dir):
    """生成增强版 ARM 工具链配置文件"""
    cpu = detect_cpu_architecture(project_data['device'])
    arm_arch = get_arm_arch_for_clang(cpu)
    compiler_type = "armclang" if project_data['use_armclang'] else "armcc"
    armclang_path = get_toolchain_path('ARMCLANG')
    armcc_path = get_toolchain_path('ARMCC')

    content = f'''# ARM Compiler (Keil) 工具链配置 - 增强 clangd 兼容性
cmake_minimum_required(VERSION 3.20)

# 自动检测的架构信息
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR {cpu.lower()})
set(CMAKE_SYSTEM_ARCH {arm_arch})  # 关键：供 ARMClang/cmake 使用

# 允许从命令行传入编译器类型
if(NOT DEFINED COMPILE_NAME)
    set(COMPILE_NAME "{compiler_type}" CACHE STRING "编译器类型")
endif()

set(ARMCC_BIN "{armcc_path}")
set(ARMCLANG_BIN "{armclang_path}")

# 编译器选择逻辑
if(COMPILE_NAME STREQUAL "armclang")
    set(CMAKE_C_COMPILER "${{ARMCLANG_BIN}}armclang.exe")
    set(CMAKE_CXX_COMPILER "${{ARMCLANG_BIN}}armclang.exe")
    set(CMAKE_ASM_COMPILER "${{ARMCLANG_BIN}}armclang.exe")
    set(CMAKE_LINKER "${{ARMCLANG_BIN}}armlink.exe")
    set(CMAKE_AR "${{ARMCLANG_BIN}}armar.exe")
    set(CMAKE_FROMELF "${{ARMCLANG_BIN}}fromelf.exe")
    set(COMPILER_TYPE "armclang")
elseif(COMPILE_NAME STREQUAL "armcc")
    set(CMAKE_C_COMPILER "${{ARMCC_BIN}}armcc.exe")
    set(CMAKE_CXX_COMPILER "${{ARMCC_BIN}}armcc.exe")
    set(CMAKE_ASM_COMPILER "${{ARMCC_BIN}}armasm.exe")
    set(CMAKE_LINKER "${{ARMCC_BIN}}armlink.exe")
    set(CMAKE_AR "${{ARMCC_BIN}}armar.exe")
    set(CMAKE_FROMELF "${{ARMCC_BIN}}fromelf.exe")
    set(COMPILER_TYPE "armcc")
else()
    message(FATAL_ERROR "不支持的编译器: ${{COMPILE_NAME}}")
endif()

# 强制跳过编译器测试
set(CMAKE_C_COMPILER_WORKS 1 CACHE INTERNAL "")
set(CMAKE_CXX_COMPILER_WORKS 1 CACHE INTERNAL "")
set(CMAKE_ASM_COMPILER_WORKS 1 CACHE INTERNAL "")
set(CMAKE_C_COMPILER_FORCED TRUE CACHE INTERNAL "")
set(CMAKE_CXX_COMPILER_FORCED TRUE CACHE INTERNAL "")

# 根据编译器类型设置参数
if(COMPILER_TYPE STREQUAL "armclang")
    set(COMMON_FLAGS "--target=arm-arm-none-eabi -mcpu={cpu.lower()} -mthumb")
    set(CMAKE_C_FLAGS_INIT "${{COMMON_FLAGS}} -ffunction-sections -fdata-sections")
    set(CMAKE_CXX_FLAGS_INIT "${{COMMON_FLAGS}} -ffunction-sections -fdata-sections")
    set(CMAKE_ASM_FLAGS_INIT "${{COMMON_FLAGS}}")
else()
    set(COMMON_FLAGS "--cpu={cpu} --apcs=interwork")
    set(CMAKE_C_FLAGS_INIT "${{COMMON_FLAGS}} --c99 --split_sections")
    set(CMAKE_CXX_FLAGS_INIT "${{COMMON_FLAGS}} --cpp --split_sections")
    set(CMAKE_ASM_FLAGS_INIT "${{COMMON_FLAGS}}")
endif()

# 链接器选项
set(LINKER_FLAGS
    "--map"
    "--info=summarysizes,sizes,totals,unused,veneers"
    "--callgraph"
    "--xref"
    "--entry=Reset_Handler"
    "--strict"
    "--summary_stderr"
    "{project_data['ld_flags']}"
)

# 设置链接命令
set(CMAKE_C_LINK_EXECUTABLE 
    "${{CMAKE_LINKER}} <CMAKE_C_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>"
)

# 交叉编译设置
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# 构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "构建类型")
endif()

# 优化等级处理
if(NOT OPTIMIZE_LEVEL)
    set(OPTIMIZE_LEVEL "{project_data['opt_level']}" CACHE STRING "优化等级")
endif()

# 诊断信息
message(STATUS "=== 工具链配置 ===")
message(STATUS "STM32 设备: {project_data['device']}")
message(STATUS "CPU 架构: ${{CMAKE_SYSTEM_PROCESSOR}}")
message(STATUS "ARM 架构: ${{CMAKE_SYSTEM_ARCH}}")
message(STATUS "编译器: ${{COMPILER_TYPE}}")
message(STATUS "优化等级: -O${{OPTIMIZE_LEVEL}}")
'''

    with open(os.path.join(output_dir, 'toolchain.cmake'), 'w', encoding="UTF-8") as f:
        f.write(content)

def generate_clangd_config(output_dir, compiler_type, cpu_arch):
    """生成 .clangd 配置文件，根据编译器类型动态选择头文件路径"""
    
    # 根据编译器类型选择对应的系统头文件路径
    if compiler_type == "armclang":
        system_include = get_include_path('ARMCLANG').replace('\\', '/')
    else:
        system_include = get_include_path('ARMCC').replace('\\', '/')
    
    # 生成架构宏定义
    arch_define = {
        'Cortex-M0':  '__ARM_ARCH_6M__',
        'Cortex-M3':  '__ARM_ARCH_7M__',
        'Cortex-M4':  '__ARM_ARCH_7EM__',
        'Cortex-M7':  '__ARM_ARCH_7EM__',
        'Cortex-M33': '__ARM_ARCH_8M_MAIN__',
    }.get(cpu_arch, '__ARM_ARCH_7M__')

    clangd_content = f'''# 为 ARM 编译器生成的 clangd 配置
# 自动根据编译器类型选择系统头文件路径

CompileFlags:
  # 移除 ARM 编译器特有的不支持参数
  Remove: [
    -mcpu=*,
    -mthumb*,
    -mfloat-abi=*,
    -mfpu=*,
    -ffunction-sections,
    -fdata-sections,
    -g,
    -O*,
    -std=*,
    -W*,
    --cpu=*,
    --apcs=*,
    --split_sections,
    --debug,
    --cpp,
    --strict
  ]
  
  # 添加系统头文件路径（关键！）
  Add: [
    "--target=arm-arm-none-eabi",
    -isystem,
    "{system_include}",
    -D__CC_ARM,
    -D{arch_define},
    -D__MICROLIB,
    -DUSE_STDPERIPH_DRIVER,
    -DSTM32F10X_HD
  ]

Diagnostics:
  Suppress: [
    unknown-warning-option,
    invalid_token_after_toplevel_declarator,
    option_ignored,
    unused-command-line-argument,
    pp_file_not_found
  ]

Index:
  Background: Build
  StandardLibrary: None

WorkspaceSymbol:
  MaxNumberOfCandidates: 100
'''

    config_path = os.path.join(output_dir, '.clangd')
    with open(config_path, 'w', encoding='UTF-8') as f:
        f.write(clangd_content)

def generate_scatter_file(destination):
    """生成默认的 scatter 文件"""
    template_scatter = '''
; *************************************************************
; *** Scatter-Loading Description File generated by uVision ***
; *************************************************************

LR_IROM1 0x08000000 0x00100000  {    ; load region size_region
  ER_IROM1 0x08000000 0x00100000  {  ; load address = execution address
   *.o (RESET, +First)
   *(InRoot$$Sections)
   .ANY (+RO)
   .ANY (+XO)
  }
  RW_IRAM1 0x20000000 0x00020000  {  ; RW data
   .ANY (+RW +ZI)
  }
}
'''
    with open(os.path.join(destination,"Template.sct"),'w',encoding='UTF-8') as f:
        f.write(template_scatter.strip())

def main():
    parser = argparse.ArgumentParser(
        description='Keil uVision to CMake converter for ARM Embedded Toolchains (with clangd support)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  %(prog)s project.uvprojx                    # 转换项目
  %(prog)s -e ARMCC_PATH=D:/Keil/ARMCC/bin/  # 修改编译器路径
  %(prog)s -e ARMCC_INCLUDE=D:/Keil/include/ # 修改头文件路径
  %(prog)s --show-config                      # 显示当前配置
  %(prog)s -o ./cmake_output project.uvprojx  # 指定输出目录
        '''
    )
    
    parser.add_argument('uvprojx', nargs='?', help='Path to .uvprojx file')
    parser.add_argument('-o', '--output', default='.', help='Output directory (default: current directory)')
    parser.add_argument('--compiler', 
                        choices=['armcc', 'armclang'], 
                        default=None,
                        help='Override compiler: armcc or armclang')
    parser.add_argument('-e', '--edit', 
                        help='Edit config in format KEY=VALUE (e.g., ARMCC_PATH=D:/path)')
    parser.add_argument('-sc', '--show-config', action='store_true',
                        help='Show current toolchain and include paths')
    
    args = parser.parse_args()
    
    # 处理编辑配置请求
    if args.edit:
        success = edit_config(args.edit)
        return 0 if success else 1
    
    # 处理显示配置请求
    if args.show_config:
        config = load_config()
        print("当前工具链配置:")
        for key, value in config['TOOLCHAINS'].items():
            print(f"  {key} = {value}")
        print("\n当前头文件路径配置:")
        for key, value in config['INCLUDES'].items():
            print(f"  {key} = {value}")
        print("\nCMake配置:")
        for key, value in config['CMAKE'].items():
            print(f"  {key} = {value}")
        return 0
    
    # 检查必需参数
    if not args.uvprojx:
        parser.error("必须提供 .uvprojx 文件路径")
        return 1

    # 解析项目文件
    if not os.path.exists(args.uvprojx):
        print(f"错误: 文件不存在 - {args.uvprojx}")
        return 1
    
    try:
        project_data = parse_uvprojx(args.uvprojx)
    except Exception as e:
        print(f"解析失败: {e}")
        return 1
    
    # 应用命令行编译器覆盖
    if args.compiler:
        project_data['use_armclang'] = (args.compiler == 'armclang')
    
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    
    # 生成CMake文件
    generate_cmakelists(project_data, args.output)
    generate_toolchain(project_data, args.output)
    generate_scatter_file(args.output)
    generate_clangd_config(args.output, 
                          "armclang" if project_data['use_armclang'] else "armcc", 
                          detect_cpu_architecture(project_data['device']))
    
    print(f"\n✓ 成功生成 CMake 工程配置")
    print(f"  项目: {project_data['project_name']}")
    print(f"  设备: {project_data['device']}")
    print(f"  编译器: {'armclang' if project_data['use_armclang'] else 'armcc'}")
    print(f"  优化等级: -O{project_data['opt_level']}")
    print(f"  输出目录: {os.path.abspath(args.output)}")
    print(f"\n✓ 生成的文件:")
    print(f"  - CMakeLists.txt")
    print(f"  - toolchain.cmake")
    print(f"  - Template.sct")
    print(f"  - .clangd (支持 VS Code 代码提示)")
    print(f"\n✓ 构建命令:")
    print(f"  cmake -G Ninja -B build -DCOMPILE_NAME=armclang -DOPTIMIZE_LEVEL=2 .")
    print(f"  cmake --build build")
    print(f"\n✓ 配置命令:")
    print(f"  {parser.prog} --show-config")
    print(f"  {parser.prog} -e ARMCLANG_PATH=D:/path/to/armclang/bin/")
    print(f"  {parser.prog} -e ARMCLANG_INCLUDE=D:/path/to/armclang/include/")

if __name__ == '__main__':
    main()