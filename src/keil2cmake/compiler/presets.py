# -*- coding: utf-8 -*-

import os

from ..keil.config import get_cmake_min_version, get_ninja_path, get_cmake_path, get_checkcpp_path
from ..common import norm_path
from ..template_engine import write_template


def generate_cmake_presets(project_root: str):
    """Generate CMakePresets.json (ARM-GCC only)."""
    cmake_min_version = get_cmake_min_version()
    ninja_path = get_ninja_path()
    cmake_path = get_cmake_path()
    checkcpp_path = get_checkcpp_path()

    generator = 'Ninja' if str(ninja_path).strip() else 'Unix Makefiles'
    cmake_executable = norm_path(cmake_path) if str(cmake_path).strip() else ''
    cmake_make_program = ''
    if str(ninja_path).strip() and str(ninja_path).strip().lower() != 'ninja':
        cmake_make_program = norm_path(ninja_path)

    major = cmake_min_version.split('.')[0]
    minor = cmake_min_version.split('.')[1] if '.' in cmake_min_version else '20'

    write_template(
        'CMakePresets.json.j2',
        {
            'cmake_min_version_major': major,
            'cmake_min_version_minor': minor,
            'generator': generator,
            'cmake_executable': cmake_executable,
            'checkcpp_path': norm_path(checkcpp_path),
            'cmake_make_program': cmake_make_program,
        },
        os.path.join(project_root, 'CMakePresets.json'),
        encoding='utf-8',
    )
