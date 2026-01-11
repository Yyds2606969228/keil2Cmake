# -*- coding: utf-8 -*-

import os

SUPPORTED_COMPILERS = ("armcc", "armclang", "armgcc")


def expand_path(value: str) -> str:
    """Expand env vars and user home in config values."""
    if value is None:
        return ""
    value = str(value).strip()
    if not value:
        return ""
    value = os.path.expandvars(value)
    value = os.path.expanduser(value)
    return value


def norm_path(p: str) -> str:
    if not p:
        return ""
    return str(p).replace("\\", "/")


def cmake_quote(p: str) -> str:
    p = norm_path(p)
    if not p:
        return '""'
    if '"' in p:
        p = p.replace('"', '\\"')
    return f'"{p}"'


def format_cmake_list(items):
    return "\n    ".join(cmake_quote(i) for i in items if str(i).strip())


def ensure_dir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)
