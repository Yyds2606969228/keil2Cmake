# -*- coding: utf-8 -*-

import os
from pathlib import Path

from .armgcc.layout import infer_gcc_internal_includes_from_armgcc_path
from ..keil.config import get_armgcc_path, get_sysroot_path
from ..keil.device import get_compiler_cpu_name
from ..common import norm_path
from ..template_engine import write_template


def _infer_gcc_toolchain_root(armgcc_path: str) -> str:
    if not armgcc_path:
        return ''

    p = Path(armgcc_path)
    if p.suffix.lower() == '.exe' or p.is_file():
        p = p.parent

    try:
        p = p.resolve()
    except (OSError, RuntimeError):
        p = Path(os.path.abspath(str(p)))

    candidates = [p, p.parent, p.parent.parent]
    for cand in candidates:
        try:
            if (cand / 'lib' / 'gcc' / 'arm-none-eabi').is_dir():
                return norm_path(str(cand))
        except (OSError, RuntimeError):
            continue

    return norm_path(str(p))


def generate_clangd_config(project_root: str, cpu_arch: str, use_microlib: bool | None = None):
    """生成 .clangd 配置文件（ARM-GCC 专用）。"""
    sysroot = norm_path(get_sysroot_path())
    armgcc_path = get_armgcc_path()
    armgcc_internal_includes = infer_gcc_internal_includes_from_armgcc_path(armgcc_path)
    gcc_toolchain_root = _infer_gcc_toolchain_root(armgcc_path)

    arch_define = {
        'Cortex-M0': '__ARM_ARCH_6M__',
        'Cortex-M3': '__ARM_ARCH_7M__',
        'Cortex-M4': '__ARM_ARCH_7EM__',
        'Cortex-M7': '__ARM_ARCH_7EM__',
        'Cortex-M33': '__ARM_ARCH_8M_MAIN__',
    }.get(cpu_arch, '__ARM_ARCH_7M__')

    cpu_name = get_compiler_cpu_name(cpu_arch)

    add_flags = [
        '--target=arm-none-eabi',
        f'-mcpu={cpu_name}',
        '-mthumb',
        '-D__ARM_ARCH_PROFILE=M',
    ]

    if gcc_toolchain_root:
        add_flags.append('--gcc-toolchain=' + gcc_toolchain_root)

    if sysroot:
        add_flags.append('--sysroot=' + sysroot)
        sysroot_include = norm_path(os.path.join(sysroot, 'include'))
        if sysroot_include:
            add_flags.extend(['-isystem', sysroot_include])

    for inc in armgcc_internal_includes:
        add_flags.extend(['-isystem', inc])

    add_flags.append(f'-D{arch_define}')
    if use_microlib:
        add_flags.append('-D__MICROLIB')
    add_flags.extend([
        '-DUSE_STDPERIPH_DRIVER',
        '-DSTM32F10X_HD',
        '-D__NO_EMBEDDED_ASM',
        '-fms-extensions',
    ])

    write_template(
        'clangd.j2',
        {'add_flags': add_flags},
        os.path.join(project_root, '.clangd'),
        encoding='utf-8',
    )
