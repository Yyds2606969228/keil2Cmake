# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import shlex
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from ..common import norm_path


_SYNC_LIST_VARS = (
    'K2C_SOURCES',
    'K2C_INCLUDE_DIRS',
    'K2C_DEFINES',
)

_SYNC_SCALAR_VARS = (
    'K2C_KEIL_MISC_C_FLAGS',
    'K2C_KEIL_MISC_ASM_FLAGS',
    'K2C_KEIL_MISC_LD_FLAGS',
)


@dataclass
class SyncInput:
    sources: list[str]
    include_dirs: list[str]
    defines: list[str]
    misc_c_flags: str
    misc_asm_flags: str
    misc_ld_flags: str
    user_cmake_path: str


@dataclass
class SyncResult:
    uvprojx_path: str
    backup_path: str
    target_name: str
    group_name: str
    source_count: int
    added_count: int
    removed_count: int
    include_count: int
    define_count: int
    dry_run: bool


def _extract_set_body(text: str, var_name: str) -> str | None:
    pattern = re.compile(
        rf'set\s*\(\s*{re.escape(var_name)}\b(.*?)\)',
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1)


def _tokenize_set_body(body: str) -> list[str]:
    lexer = shlex.shlex(body, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = '#'
    tokens = [token.strip() for token in lexer if token.strip()]
    if 'CACHE' in tokens:
        tokens = tokens[:tokens.index('CACHE')]
    return tokens


def _extract_set_tokens(text: str, var_name: str) -> list[str]:
    body = _extract_set_body(text, var_name)
    if body is None:
        return []
    return _tokenize_set_body(body)


def _extract_set_scalar(text: str, var_name: str) -> str:
    tokens = _extract_set_tokens(text, var_name)
    if not tokens:
        return ''
    return ' '.join(tokens)


def _resolve_path(path_value: str, base_dir: str) -> str:
    raw = str(path_value).strip()
    if not raw:
        return ''
    if os.path.isabs(raw):
        return os.path.normpath(raw)
    return os.path.normpath(os.path.join(base_dir, raw))


def _to_uvprojx_path(abs_path: str, uvprojx_dir: str) -> str:
    try:
        rel = os.path.relpath(abs_path, uvprojx_dir)
        return norm_path(rel)
    except ValueError:
        return norm_path(abs_path)


def _normalize_for_compare(path_value: str, base_dir: str) -> str:
    abs_path = _resolve_path(path_value, base_dir)
    return os.path.normcase(os.path.normpath(abs_path))


def _file_type_for_path(path_value: str) -> str:
    ext = os.path.splitext(path_value)[1].lower()
    if ext == '.c':
        return '1'
    if ext in ('.cpp', '.cxx', '.cc'):
        return '8'
    if ext in ('.s', '.asm'):
        return '2'
    return '5'


def _ensure_child(parent: ET.Element, tag: str) -> ET.Element:
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    return child


def _ensure_path(parent: ET.Element, path_tags: list[str]) -> ET.Element:
    node = parent
    for tag in path_tags:
        node = _ensure_child(node, tag)
    return node


def _find_targets(root: ET.Element) -> list[ET.Element]:
    direct = root.findall('./Targets/Target')
    if direct:
        return direct
    return root.findall('.//Targets/Target')


def _select_target(root: ET.Element, target_name: str | None) -> ET.Element:
    targets = _find_targets(root)
    if not targets:
        raise ValueError('No <Target> found in uvprojx.')

    if target_name:
        for target in targets:
            if (target.findtext('TargetName') or '').strip() == target_name:
                return target
        raise ValueError(f"Target '{target_name}' not found in uvprojx.")

    return targets[0]


def _collect_group_paths(files_node: ET.Element, uvprojx_dir: str) -> dict[str, ET.Element]:
    out: dict[str, ET.Element] = {}
    for file_node in files_node.findall('File'):
        file_path = (file_node.findtext('FilePath') or '').strip()
        if not file_path:
            continue
        out[_normalize_for_compare(file_path, uvprojx_dir)] = file_node
    return out


def _append_file_entry(files_node: ET.Element, src_abs: str, uvprojx_dir: str) -> None:
    file_node = ET.SubElement(files_node, 'File')
    ET.SubElement(file_node, 'FileName').text = os.path.basename(src_abs)
    ET.SubElement(file_node, 'FileType').text = _file_type_for_path(src_abs)
    ET.SubElement(file_node, 'FilePath').text = _to_uvprojx_path(src_abs, uvprojx_dir)


def parse_keil2cmake_user(cmake_root: str) -> SyncInput:
    root = os.path.abspath(cmake_root)
    user_cmake_path = os.path.join(root, 'cmake', 'user', 'keil2cmake_user.cmake')
    if not os.path.isfile(user_cmake_path):
        raise FileNotFoundError(f'keil2cmake user config not found: {user_cmake_path}')

    text = ''
    with open(user_cmake_path, 'r', encoding='utf-8') as f:
        text = f.read()

    list_values: dict[str, list[str]] = {}
    for var in _SYNC_LIST_VARS:
        list_values[var] = _extract_set_tokens(text, var)

    scalar_values: dict[str, str] = {}
    for var in _SYNC_SCALAR_VARS:
        scalar_values[var] = _extract_set_scalar(text, var)

    sources = [
        _resolve_path(item, root)
        for item in list_values['K2C_SOURCES']
        if str(item).strip()
    ]
    includes = [
        _resolve_path(item, root)
        for item in list_values['K2C_INCLUDE_DIRS']
        if str(item).strip()
    ]
    defines = [str(item).strip() for item in list_values['K2C_DEFINES'] if str(item).strip()]

    return SyncInput(
        sources=sources,
        include_dirs=includes,
        defines=defines,
        misc_c_flags=scalar_values['K2C_KEIL_MISC_C_FLAGS'],
        misc_asm_flags=scalar_values['K2C_KEIL_MISC_ASM_FLAGS'],
        misc_ld_flags=scalar_values['K2C_KEIL_MISC_LD_FLAGS'],
        user_cmake_path=user_cmake_path,
    )


def sync_cmake_to_uvprojx(
    uvprojx_path: str,
    cmake_root: str,
    target_name: str | None = None,
    group_name: str = 'K2C_Sync',
    prune: bool = False,
    dry_run: bool = False,
) -> SyncResult:
    uvprojx_abs = os.path.abspath(uvprojx_path)
    if not os.path.isfile(uvprojx_abs):
        raise FileNotFoundError(f'uvprojx not found: {uvprojx_abs}')

    sync_data = parse_keil2cmake_user(cmake_root)
    uvprojx_dir = os.path.dirname(uvprojx_abs)

    tree = ET.parse(uvprojx_abs)
    root = tree.getroot()
    target = _select_target(root, target_name)
    selected_target_name = (target.findtext('TargetName') or '').strip() or '<unknown>'

    groups = _ensure_child(target, 'Groups')
    group = None
    for candidate in groups.findall('Group'):
        if (candidate.findtext('GroupName') or '').strip() == group_name:
            group = candidate
            break
    if group is None:
        group = ET.SubElement(groups, 'Group')
        ET.SubElement(group, 'GroupName').text = group_name
        ET.SubElement(group, 'Files')
    files_node = _ensure_child(group, 'Files')

    existing_map = _collect_group_paths(files_node, uvprojx_dir)
    new_keys = [_normalize_for_compare(src, uvprojx_dir) for src in sync_data.sources]
    new_key_set = set(new_keys)

    added_count = 0
    removed_count = 0

    if prune:
        current_nodes = list(files_node.findall('File'))
        for file_node in current_nodes:
            file_path = (file_node.findtext('FilePath') or '').strip()
            if not file_path:
                files_node.remove(file_node)
                removed_count += 1
                continue
            key = _normalize_for_compare(file_path, uvprojx_dir)
            if key not in new_key_set:
                files_node.remove(file_node)
                removed_count += 1
        existing_map = _collect_group_paths(files_node, uvprojx_dir)

    for src_abs in sync_data.sources:
        key = _normalize_for_compare(src_abs, uvprojx_dir)
        if key in existing_map:
            continue
        _append_file_entry(files_node, src_abs, uvprojx_dir)
        added_count += 1
        existing_map[key] = files_node.findall('File')[-1]

    include_node = _ensure_path(
        target,
        ['TargetOption', 'TargetArmAds', 'Cads', 'VariousControls', 'IncludePath'],
    )
    include_node.text = ';'.join([_to_uvprojx_path(p, uvprojx_dir) for p in sync_data.include_dirs])

    define_node = _ensure_path(
        target,
        ['TargetOption', 'TargetArmAds', 'Cads', 'VariousControls', 'Define'],
    )
    define_node.text = ','.join(sync_data.defines)

    misc_c_node = _ensure_path(
        target,
        ['TargetOption', 'TargetArmAds', 'Cads', 'VariousControls', 'MiscControls'],
    )
    misc_c_node.text = sync_data.misc_c_flags

    misc_asm_node = _ensure_path(
        target,
        ['TargetOption', 'TargetArmAds', 'Aads', 'VariousControls', 'MiscControls'],
    )
    misc_asm_node.text = sync_data.misc_asm_flags

    misc_ld_node = _ensure_path(
        target,
        ['TargetOption', 'TargetArmAds', 'LDads', 'VariousControls', 'MiscControls'],
    )
    misc_ld_node.text = sync_data.misc_ld_flags

    backup_path = uvprojx_abs + '.bak'
    if not dry_run:
        shutil.copy2(uvprojx_abs, backup_path)
        if hasattr(ET, 'indent'):
            ET.indent(tree, space='  ')
        tree.write(uvprojx_abs, encoding='utf-8', xml_declaration=True)

    return SyncResult(
        uvprojx_path=uvprojx_abs,
        backup_path=backup_path,
        target_name=selected_target_name,
        group_name=group_name,
        source_count=len(sync_data.sources),
        added_count=added_count,
        removed_count=removed_count,
        include_count=len(sync_data.include_dirs),
        define_count=len(sync_data.defines),
        dry_run=dry_run,
    )
