# -*- coding: utf-8 -*-

import os

from ..keil.config import (
    get_cmake_min_version,
    get_ninja_path,
    get_cmake_path,
    get_checkcpp_path,
    get_openocd_path,
)
from .debug import infer_openocd_target
from ..common import norm_path
from ..template_engine import write_template


def generate_cmake_presets(project_root: str, project_data: dict | None = None):
    """Generate CMakePresets.json (ARM-GCC only)."""
    cmake_min_version = get_cmake_min_version()
    ninja_path = get_ninja_path()
    cmake_path = get_cmake_path()
    checkcpp_path = get_checkcpp_path()
    openocd_path = get_openocd_path()

    generator = 'Ninja' if str(ninja_path).strip() else 'Unix Makefiles'
    cmake_executable = norm_path(cmake_path) if str(cmake_path).strip() else ''
    cmake_make_program = ''
    if str(ninja_path).strip() and str(ninja_path).strip().lower() != 'ninja':
        cmake_make_program = norm_path(ninja_path)

    major = cmake_min_version.split('.')[0]
    minor = cmake_min_version.split('.')[1] if '.' in cmake_min_version else '20'

    device = ''
    if project_data:
        device = str(project_data.get('device', '') or '')

    write_template(
        'CMakePresets.json.j2',
        {
            'cmake_min_version_major': major,
            'cmake_min_version_minor': minor,
            'generator': generator,
            'cmake_executable': cmake_executable,
            'checkcpp_path': norm_path(checkcpp_path),
            'openocd_path': norm_path(openocd_path),
            'debug_probe': '',
            'openocd_target': infer_openocd_target(device),
            'cmake_make_program': cmake_make_program,
        },
        os.path.join(project_root, 'CMakePresets.json'),
        encoding='utf-8',
    )
