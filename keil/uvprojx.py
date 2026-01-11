# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import os

from i18n import t


def parse_uvprojx(uvprojx_path: str) -> dict:
    """解析 uvprojx 文件，提取项目配置。"""
    tree = ET.parse(uvprojx_path)
    root = tree.getroot()

    print(t('uvprojx.get_target'))
    project_name = root.find('.//Targets/Target/TargetName').text
    out_node = root.find('.//Targets/Target/TargetOption/TargetCommonOption/OutputDirectory')
    output_dir = (out_node.text if out_node is not None and out_node.text else 'build/')

    print(t('uvprojx.collect_sources'))
    source_files = []
    for group in root.findall('.//Groups/Group'):
        for file in group.findall('Files/File'):
            file_path = file.find('FilePath').text
            if file_path.endswith(('.c', '.C', '.cpp', '.s', '.S', '.asm')):
                source_files.append(file_path)

    print(t('uvprojx.set_includes'))
    include_paths = []
    includes = root.find('.//TargetOption/TargetArmAds/Cads/VariousControls/IncludePath')
    if includes is not None and includes.text:
        include_paths.extend(includes.text.split(';'))

    print(t('uvprojx.load_defines'))
    defines = []
    defs = root.find('.//TargetOption/TargetArmAds/Cads/VariousControls/Define')
    if defs is not None and defs.text:
        defines.extend([d.strip() for d in defs.text.split(',')])

    print(t('uvprojx.scatter'))
    linker_script = None
    scatter_file = root.find('.//TargetOption/TargetArmAds/LDads/ScatterFile')
    if scatter_file is not None and scatter_file.text:
        linker_script = scatter_file.text

    print(t('uvprojx.device'))
    device_name = None
    device_node = root.find('.//Targets/Target/TargetOption/TargetCommonOption/Device')
    if device_node is not None and device_node.text:
        device_name = device_node.text

    print(t('uvprojx.compiler'))
    use_armclang = False
    # Check uAC6 node (0=ARMCC/Compiler5, 1=ARMCLANG/Compiler6)
    uac6_node = root.find('.//uAC6')
    if uac6_node is not None:
        print(f"  uAC6 = {uac6_node.text}")
        if uac6_node.text == '1':
            use_armclang = True
    else:
        print(f"  uAC6 node not found, defaulting to ARMCC")

    print(t('uvprojx.flags'))
    c_flags = root.find('.//TargetOption/TargetArmAds/Cads/VariousControls/MiscControls')
    asm_flags = root.find('.//TargetOption/TargetArmAds/Aads/VariousControls/MiscControls')
    ld_flags = root.find('.//TargetOption/TargetArmAds/LDads/VariousControls/MiscControls')
    c_flags = c_flags.text if c_flags is not None and c_flags.text else ''
    asm_flags = asm_flags.text if asm_flags is not None and asm_flags.text else ''
    ld_flags = ld_flags.text if ld_flags is not None and ld_flags.text else ''

    print(t('uvprojx.optimize'))
    keil_optim = '0'  # Store raw Keil Optim value for compiler-specific mapping later
    optim_node = root.find('.//TargetOption/TargetArmAds/Cads/Optim')
    if optim_node is not None and optim_node.text:
        keil_optim = optim_node.text
        print(f"  Keil Optim = {keil_optim}")
    else:
        print(f"  Optim not found, defaulting to 0")

    print()

    uvprojx_abs = os.path.abspath(uvprojx_path)
    uvprojx_dir = os.path.dirname(uvprojx_abs)

    return {
        'uvprojx_path': uvprojx_abs,
        'uvprojx_dir': uvprojx_dir,
        'project_name': project_name,
        'source_files': source_files,
        'include_paths': include_paths,
        'defines': defines,
        'linker_script': linker_script,
        'device': device_name.strip() if device_name else 'Unknown',
        'c_flags': c_flags,
        'asm_flags': asm_flags,
        'ld_flags': ld_flags,
        'output_dir': output_dir,
        'use_armclang': use_armclang,
        'keil_optim': keil_optim,  # Store raw Keil value for compiler-specific mapping
    }
