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


def remove_bom_from_file(file_path: str) -> bool:
    """Remove BOM (Byte Order Mark) from a file if present.
    
    Returns True if BOM was found and removed, False otherwise.
    """
    if not os.path.exists(file_path):
        return False
    
    try:
        # Try to read with utf-8-sig to detect and remove BOM
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
        
        # Check if original file had BOM by reading as binary
        with open(file_path, 'rb') as f:
            raw = f.read(3)
            has_bom = raw.startswith(b'\xef\xbb\xbf')
        
        if has_bom:
            # Rewrite without BOM
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
    except Exception:
        pass
    
    return False
