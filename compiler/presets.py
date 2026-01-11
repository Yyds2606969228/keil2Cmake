# -*- coding: utf-8 -*-

import os

from keil.config import get_cmake_min_version, get_ninja_enabled, get_ninja_path
from keil2cmake_common import norm_path, SUPPORTED_COMPILERS


def generate_cmake_presets(project_root: str, default_compiler: str):
    """Generate CMakePresets.json with one preset per compiler."""
    cmake_min_version = get_cmake_min_version()
    ninja_enabled = get_ninja_enabled()
    ninja_path = get_ninja_path()

    generator = 'Ninja' if ninja_enabled else 'Unix Makefiles'

    def _cache_block_for(toolchain_rel: str, compiler: str) -> str:
        lines = [
            f'      "CMAKE_TOOLCHAIN_FILE": "${{sourceDir}}/{toolchain_rel}"',
            f'      "K2C_COMPILER": "{compiler}"',
        ]
        if ninja_enabled and ninja_path and ninja_path.lower() != 'ninja':
            lines.append(f'      "CMAKE_MAKE_PROGRAM": "{norm_path(ninja_path)}"')
        return ",\n".join(lines)

    def _configure_preset(name: str, display: str, toolchain_rel: str, compiler: str, binary_dir: str) -> str:
        cache_block = _cache_block_for(toolchain_rel, compiler)
        return f'''        {{
            "name": "{name}",
            "displayName": "{display}",
            "generator": "{generator}",
            "binaryDir": "${{sourceDir}}/{binary_dir}",
            "cacheVariables": {{
{cache_block}
            }}
        }}'''

    default_compiler = (default_compiler or '').strip().lower()
    if default_compiler not in SUPPORTED_COMPILERS:
        default_compiler = 'armcc'

    toolchain_rel = 'cmake/internal/toolchain.cmake'
    configure_presets = [
        _configure_preset('keil2cmake-armcc', f'Keil2Cmake armcc ({generator})', toolchain_rel, 'armcc', 'build/armcc'),
        _configure_preset('keil2cmake-armclang', f'Keil2Cmake armclang ({generator})', toolchain_rel, 'armclang', 'build/armclang'),
        _configure_preset('keil2cmake-armgcc', f'Keil2Cmake armgcc ({generator})', toolchain_rel, 'armgcc', 'build/armgcc'),
        _configure_preset('keil2cmake', f'Keil2Cmake default ({default_compiler}) ({generator})', toolchain_rel, default_compiler, f'build/{default_compiler}'),
    ]

    build_presets = [
        '        {"name": "keil2cmake-armcc", "configurePreset": "keil2cmake-armcc"}',
        '        {"name": "keil2cmake-armclang", "configurePreset": "keil2cmake-armclang"}',
        '        {"name": "keil2cmake-armgcc", "configurePreset": "keil2cmake-armgcc"}',
        '        {"name": "keil2cmake", "configurePreset": "keil2cmake"}',
    ]

    join_with_newline = ',\n'
    configure_presets_block = join_with_newline.join(configure_presets)
    build_presets_block = join_with_newline.join(build_presets)

    presets = f'''{{
    "version": 3,
    "cmakeMinimumRequired": {{
        "major": {cmake_min_version.split('.')[0]},
        "minor": {cmake_min_version.split('.')[1] if '.' in cmake_min_version else 20},
        "patch": 0
    }},
    "configurePresets": [
{configure_presets_block}
    ],
    "buildPresets": [
{build_presets_block}
    ]
}}
'''

    with open(os.path.join(project_root, 'CMakePresets.json'), 'w', encoding='utf-8') as f:
        f.write(presets)
