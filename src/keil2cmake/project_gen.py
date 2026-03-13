# -*- coding: utf-8 -*-

import os
import re
import logging

from .common import ensure_dir, norm_path
from .keil.device import detect_cpu_architecture
from .keil.config import get_cmake_min_version
from .i18n import t
from .template_engine import write_template

logger = logging.getLogger(__name__)


def _relativize_paths(paths: list[str], project_root: str, uvprojx_dir: str | None) -> list[str]:
    uv_base = uvprojx_dir or project_root
    out: list[str] = []
    for p in paths or []:
        raw = str(p).strip()
        if not raw:
            continue

        # Resolve relative paths as Keil does: relative to the .uvprojx directory.
        if os.path.isabs(raw):
            abs_path = os.path.normpath(raw)
        else:
            abs_path = os.path.normpath(os.path.join(uv_base, raw))

        # Prefer a nice relative path in generated CMake (more portable).
        try:
            rel = os.path.relpath(abs_path, project_root)
            out.append(norm_path(rel))
        except ValueError:
            # relpath can fail on Windows when paths are on different drives.
            out.append(norm_path(abs_path))
    return out


_GAS_PATTERN = re.compile(
    r'^\s*\.(syntax|section|global|thumb|thumb_func|word|align|type|size)\b',
    re.IGNORECASE,
)


def _is_gas_source(path: str) -> bool:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(200):
                line = f.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped or stripped.startswith((';', '//')):
                    continue
                if _GAS_PATTERN.search(line):
                    return True
    except OSError as exc:
        logger.debug("Failed to inspect ASM syntax in '%s': %s", path, exc)
        return False
    return False


def _device_token(device_name: str) -> str:
    if not device_name:
        return ''
    lower = device_name.lower()
    m = re.search(r'(stm32[a-z0-9]+)', lower)
    if m:
        return m.group(1)
    return ''


def _find_gcc_startup_candidate(uvprojx_dir: str, device_name: str) -> str:
    search_roots = [uvprojx_dir]
    parent = os.path.dirname(uvprojx_dir)
    if parent and parent not in search_roots:
        search_roots.append(parent)

    token = _device_token(device_name)
    best_path = ''
    best_score = -1
    skip_dirs = {'.git', 'build', 'out', 'dist', '.vscode', '.idea'}

    for root in search_roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for name in filenames:
                if not name.lower().startswith('startup_'):
                    continue
                if os.path.splitext(name)[1].lower() != '.s':
                    continue
                path = os.path.join(dirpath, name)
                if not _is_gas_source(path):
                    continue

                score = 1
                if token and token in name.lower():
                    score += 3
                if token and token in dirpath.lower():
                    score += 2
                if 'gcc' in dirpath.lower() or 'gnu' in dirpath.lower():
                    score += 1

                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(4096)
                        if '.isr_vector' in content:
                            score += 2
                        if 'Reset_Handler' in content:
                            score += 2
                except OSError as exc:
                    logger.debug("Failed to inspect startup candidate '%s': %s", path, exc)

                if score > best_score:
                    best_score = score
                    best_path = path

    return best_path


def _resolve_sources(project_data: dict, project_root: str) -> tuple[list[str], list[str], str]:
    uvprojx_dir = project_data.get('uvprojx_dir') or ''
    abs_sources: list[str] = []
    asm_sources: list[str] = []

    for raw in project_data.get('source_files', []):
        raw = str(raw).strip()
        if not raw:
            continue
        if os.path.isabs(raw):
            abs_path = os.path.normpath(raw)
        else:
            abs_path = os.path.normpath(os.path.join(uvprojx_dir, raw))
        abs_sources.append(abs_path)

    filtered_abs: list[str] = []
    for path in abs_sources:
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.s', '.S', '.asm'):
            asm_sources.append(path)
        filtered_abs.append(path)

    asm_detected = len(asm_sources) > 0
    gcc_startup_abs = ''
    if asm_detected:
        gcc_startup_abs = _find_gcc_startup_candidate(uvprojx_dir, project_data.get('device', ''))

    rel_sources = _relativize_paths(filtered_abs, project_root, uvprojx_dir)
    rel_asm_sources = _relativize_paths(asm_sources, project_root, uvprojx_dir)
    gcc_startup_rel = ''
    if gcc_startup_abs:
        gcc_startup_rel = _relativize_paths([gcc_startup_abs], project_root, uvprojx_dir)[0]

    return rel_sources, rel_asm_sources, gcc_startup_rel


def generate_cmake_structure(project_data: dict, project_root: str) -> None:
    """Generate layered CMake structure under project_root (ARM-GCC only)."""
    cmake_min_version = get_cmake_min_version()

    # Map Keil Optim value to GCC optimization level
    keil_optim = project_data.get('keil_optim', '0')
    keil_compiler = (project_data.get('keil_compiler') or 'armcc').lower()
    if keil_compiler == 'armclang':
        optim_map = {
            '0': '0',   # -O0
            '1': '1',   # -O1
            '2': '2',   # -O2
            '3': '3',   # -O3
            '4': '1',   # Keil default for AC6
            '11': 's',  # -Oz in ARMCLANG, map to -Os for GCC
        }
    else:
        optim_map = {
            '0': '0',   # -O0
            '1': '1',   # -O1
            '2': '2',   # -O2
            '3': '3',   # -O3
            '4': '1',   # Keil default for AC5
            '11': 's',  # -Ospace in ARMCC, map to -Os for GCC
        }

    opt_level = optim_map.get(keil_optim, '0')
    project_data['opt_level'] = opt_level

    user_dir = os.path.join(project_root, 'cmake', 'user')
    ensure_dir(user_dir)

    gen_sources, asm_sources, gcc_startup_rel = _resolve_sources(project_data, project_root)
    asm_detected = len(asm_sources) > 0
    project_data['asm_detected'] = asm_detected
    project_data['gcc_startup'] = gcc_startup_rel
    uvprojx_dir = project_data.get('uvprojx_dir')
    gen_includes = _relativize_paths(project_data.get('include_paths', []), project_root, uvprojx_dir)
    gen_defines = [d.strip() for d in project_data['defines'] if str(d).strip()]

    keil_misc_c_flags = project_data['c_flags'].replace('"', '\\"')
    keil_misc_asm_flags = project_data['asm_flags'].replace('"', '\\"')
    keil_misc_ld_flags = project_data['ld_flags'].replace('"', '\\"')

    user_cmake_path = os.path.join(user_dir, 'keil2cmake_user.cmake')
    if not os.path.exists(user_cmake_path):
        cpu_arch = detect_cpu_architecture(project_data['device'])
        use_newlib_nano_default = 'ON' if project_data.get('use_microlib') else 'OFF'
        write_template(
            'keil2cmake_user.cmake.j2',
            {
                'header_title': t('gen.user.header.title'),
                'header_safe': t('gen.user.header.safe'),
                'header_no_overwrite': t('gen.user.header.no_overwrite'),
                'defaults_header': t('gen.user.defaults'),
                'optimize_header': t('gen.user.optimize'),
                'linker_header': t('gen.user.linker'),
                'project_name': project_data['project_name'],
                'device': project_data['device'],
                'cpu_arch': cpu_arch,
                'default_opt_level': project_data['opt_level'],
                'use_newlib_nano_default': use_newlib_nano_default,
                'asm_detected': 'ON' if asm_detected else 'OFF',
                'asm_sources': asm_sources,
                'gcc_startup': gcc_startup_rel,
                'sources': gen_sources,
                'includes': gen_includes,
                'defines': gen_defines,
                'misc_c_flags': keil_misc_c_flags,
                'misc_asm_flags': keil_misc_asm_flags,
                'misc_ld_flags': keil_misc_ld_flags,
            },
            user_cmake_path,
            encoding='utf-8',
        )

    cppcheck_cmake_path = os.path.join(user_dir, 'cppcheck.cmake')
    if not os.path.exists(cppcheck_cmake_path):
        write_template(
            'cppcheck.cmake.j2',
            {
                'header_title': t('gen.user.header.title'),
                'header_safe': t('gen.user.header.safe'),
                'header_no_overwrite': t('gen.user.header.no_overwrite'),
            },
            cppcheck_cmake_path,
            encoding='utf-8',
        )

    write_template(
        'CMakeLists.txt.j2',
        {
            'cmake_min_version': cmake_min_version,
            'project_name': project_data['project_name'],
        },
        os.path.join(project_root, 'CMakeLists.txt'),
        encoding='utf-8',
    )


def clean_generated(project_root: str) -> int:
    """Remove keil2cmake generated artifacts under project_root (safe list)."""
    removed = 0

    root_files = ['CMakeLists.txt', 'CMakePresets.json', '.clangd']
    for name in root_files:
        path = os.path.join(project_root, name)
        if os.path.isfile(path):
            try:
                os.remove(path)
                removed += 1
            except OSError as exc:
                logger.debug("Failed to remove generated file '%s': %s", path, exc)

    vscode_dir = os.path.join(project_root, '.vscode')
    for name in ('launch.json', 'tasks.json'):
        path = os.path.join(vscode_dir, name)
        if os.path.isfile(path):
            try:
                os.remove(path)
                removed += 1
            except OSError as exc:
                logger.debug("Failed to remove generated file '%s': %s", path, exc)

    cmake_dir = os.path.join(project_root, 'cmake')
    internal_dir = os.path.join(cmake_dir, 'internal')
    user_dir = os.path.join(cmake_dir, 'user')

    generated_paths = [
        os.path.join(internal_dir, 'toolchain.cmake'),
        os.path.join(internal_dir, 'k2c_debug.cmake'),
        os.path.join(internal_dir, 'keil2cmake_default.sct'),
        os.path.join(internal_dir, 'keil2cmake_default.ld'),
        os.path.join(internal_dir, 'keil2cmake_from_sct.ld'),
        os.path.join(internal_dir, 'templates', 'openocd.cfg.in'),
        os.path.join(internal_dir, 'templates', 'launch.json.in'),
        os.path.join(internal_dir, 'templates', 'tasks.json.in'),

        os.path.join(user_dir, 'keil2cmake_user.cmake'),
        os.path.join(user_dir, 'cppcheck.cmake'),
        os.path.join(user_dir, 'openocd.cfg'),

        os.path.join(user_dir, 'common', 'keil2cmake_project.cmake'),
        os.path.join(user_dir, 'common', 'keil2cmake_user.cmake'),
    ]

    known = generated_paths
    for path in known:
        if os.path.isfile(path):
            try:
                os.remove(path)
                removed += 1
            except OSError as exc:
                logger.debug("Failed to remove generated file '%s': %s", path, exc)

    templates_dir = os.path.join(internal_dir, 'templates')
    try:
        if os.path.isdir(templates_dir) and not os.listdir(templates_dir):
            os.rmdir(templates_dir)
    except OSError as exc:
        logger.debug("Failed to remove empty templates dir '%s': %s", templates_dir, exc)

    if removed:
        print(t('clean.done', count=removed))
    else:
        print(t('clean.none'))
    return 0
