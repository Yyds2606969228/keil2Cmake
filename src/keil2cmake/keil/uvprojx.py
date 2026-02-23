# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import os

from ..i18n import t


def _parse_bool(text: str | None) -> bool | None:
    if text is None:
        return None
    value = str(text).strip().lower()
    if value in ('1', 'true', 'yes', 'on'):
        return True
    if value in ('0', 'false', 'no', 'off'):
        return False
    return None


def _find_bool(root: ET.Element, tag_names: list[str]) -> bool | None:
    for name in tag_names:
        node = root.find(f'.//{name}')
        if node is not None and node.text is not None:
            parsed = _parse_bool(node.text)
            if parsed is not None:
                return parsed
    return None


def _detect_keil_compiler(root: ET.Element) -> str:
    # uAC6=1 indicates Arm Compiler 6 (armclang) in many uvprojx files.
    # Fall back to armcc when not found.
    use_armclang = _find_bool(root, ['uAC6', 'UseArmClang'])
    return 'armclang' if use_armclang else 'armcc'


def _infer_debugger_from_text(text: str) -> str:
    if not text:
        return ''
    lowered = str(text).lower()
    normalized = lowered.replace('\\', '/').replace('_', '-')

    # Keep explicit probes first to reduce accidental matches.
    jlink_tokens = ('j-link', 'jlink', 'jl2cm3', 'jltagdi', 'segger')
    stlink_tokens = ('st-link', 'stlink', 'st-linkiii-keil-swo')
    daplink_tokens = (
        'daplink',
        'cmsis-dap',
        'cmsis-agdi',
        'cmsis_agdi',
        'ul2cm3',
        'ulp2cm3',
        'ulink',
    )

    if any(token in normalized for token in jlink_tokens):
        return 'jlink'
    if any(token in normalized for token in stlink_tokens):
        return 'stlink'
    if any(token in normalized for token in daplink_tokens):
        return 'daplink'
    return ''


def _detect_debugger(root: ET.Element, uvprojx_path: str) -> str:
    candidate_paths = [
        './/TargetDriverDllRegistry/SetRegEntry/Key',
        './/TargetDriverDllRegistry/SetRegEntry/Name',
        './/DebugOpt/pMon',
        './/DebugOpt/tDll',
        './/DebugOpt/tDlgDll',
        './/DebugOpt/tIfile',
        './/DebugOpt/sDll',
        './/DebugOpt/sDlgDll',
        './/DebugOpt/sIfile',
        './/Utilities/Flash2',
    ]

    for path in candidate_paths:
        for node in root.findall(path):
            text = node.text or ''
            inferred = _infer_debugger_from_text(text)
            if inferred:
                return inferred

    # Fallback: scan project XML text for known probe tokens.
    try:
        with open(uvprojx_path, 'r', encoding='utf-8', errors='ignore') as f:
            inferred = _infer_debugger_from_text(f.read())
            if inferred:
                return inferred
    except OSError:
        pass

    # Keil often stores user debug probe selection in sibling .uvoptx/.uvopt.
    stem, _ = os.path.splitext(uvprojx_path)
    for ext in ('.uvoptx', '.uvopt'):
        opt_path = stem + ext
        if not os.path.exists(opt_path):
            continue
        try:
            with open(opt_path, 'r', encoding='utf-8', errors='ignore') as f:
                inferred = _infer_debugger_from_text(f.read())
                if inferred:
                    return inferred
        except OSError:
            continue
    return ''


def parse_uvprojx(uvprojx_path: str) -> dict:
    """解析 uvprojx 文件，提取项目信息。"""
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
    keil_compiler = _detect_keil_compiler(root)
    print(f"  Keil compiler = {keil_compiler}")

    debugger = _detect_debugger(root, uvprojx_path)
    if debugger:
        print(f"  Debugger = {debugger}")

    use_microlib = _find_bool(root, ['UseMicroLIB', 'UseMicroLib', 'uMicrolib', 'uMicroLIB'])

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
        print("  Optim not found, defaulting to 0")

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
        'keil_optim': keil_optim,  # Store raw Keil value for compiler-specific mapping
        'keil_compiler': keil_compiler,
        'debugger': debugger,
        'use_microlib': use_microlib,
    }
