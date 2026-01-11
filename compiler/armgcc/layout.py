# -*- coding: utf-8 -*-

import glob
import os
from pathlib import Path

from keil2cmake_common import expand_path, norm_path


def infer_sysroot_from_armgcc_path(armgcc_path: str) -> str:
    """Infer a reasonable --sysroot from ARMGCC_PATH.

    Common layouts:
      - <toolchain>/bin
      - <toolchain>/arm-none-eabi/bin
      - <toolchain>/bin/arm-none-eabi-gcc(.exe)

    Returns '' when inference fails.
    """
    armgcc_path = expand_path(armgcc_path)
    if not armgcc_path:
        return ''

    p = Path(armgcc_path)
    if p.suffix.lower() == '.exe' or p.is_file():
        p = p.parent

    try:
        p = p.resolve()
    except Exception:
        p = Path(os.path.abspath(str(p)))

    candidates = []
    if p.name.lower() == 'bin':
        if p.parent.name.lower() == 'arm-none-eabi':
            candidates.append(p.parent)
        candidates.append(p.parent / 'arm-none-eabi')
    else:
        candidates.append(p / 'arm-none-eabi')
    candidates.append(p.parent / 'arm-none-eabi')

    for cand in candidates:
        try:
            if (cand / 'include').is_dir():
                return str(cand)
        except Exception:
            continue
    return ''


def infer_gcc_internal_includes_from_armgcc_path(armgcc_path: str) -> list[str]:
    """Infer GCC internal include directories (include/include-fixed) from ARMGCC_PATH."""
    armgcc_path = expand_path(armgcc_path)
    if not armgcc_path:
        return []

    p = Path(armgcc_path)
    if p.suffix.lower() == '.exe' or p.is_file():
        p = p.parent

    try:
        p = p.resolve()
    except Exception:
        p = Path(os.path.abspath(str(p)))

    toolchain_root = p
    if p.name.lower() == 'bin':
        if p.parent.name.lower() == 'arm-none-eabi':
            toolchain_root = p.parent.parent
        else:
            toolchain_root = p.parent

    base = toolchain_root / 'lib' / 'gcc' / 'arm-none-eabi'
    patterns = [
        str(base / '*' / 'include'),
        str(base / '*' / 'include-fixed'),
    ]

    found: list[str] = []
    for pat in patterns:
        for match in glob.glob(pat):
            if os.path.isdir(match):
                found.append(norm_path(match))

    uniq: list[str] = []
    for x in found:
        if x and x not in uniq:
            uniq.append(x)
    return uniq
