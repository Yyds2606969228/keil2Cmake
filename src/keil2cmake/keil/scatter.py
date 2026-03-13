# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass
from typing import Callable

from ..template_engine import write_template


_DEFINE_RE = re.compile(r'^\s*#define\s+([A-Za-z_]\w*)\s+(.+?)\s*$')
_BLOCK_COMMENT_RE = re.compile(r'/\*.*?\*/', re.DOTALL)
_LINE_COMMENT_RE = re.compile(r'//.*$')
_SEMICOLON_COMMENT_RE = re.compile(r';.*$')

_ATTR_TOKENS = {
    'ABSOLUTE',
    'FIXED',
    'OVERLAY',
    'UNINIT',
    'EMPTY',
    'FIRST',
    'LAST',
    'ALIGN',
    'AT',
}


@dataclass(frozen=True)
class ScatterRegion:
    name: str
    base: int
    size: int
    base_expr: str
    size_expr: str


@dataclass(frozen=True)
class ScatterConversion:
    ok: bool
    reason: str
    rom_base: int | None = None
    rom_size: int | None = None
    ram_base: int | None = None
    ram_size: int | None = None
    stack_size: int | None = None
    heap_size: int | None = None
    rom_source: str | None = None
    ram_source: str | None = None


def _strip_comments(text: str) -> str:
    text = _BLOCK_COMMENT_RE.sub('', text)
    lines: list[str] = []
    for raw in text.splitlines():
        line = _LINE_COMMENT_RE.sub('', raw)
        line = _SEMICOLON_COMMENT_RE.sub('', line)
        lines.append(line)
    return '\n'.join(lines)


def _normalize_expr(expr: str) -> str:
    expr = expr.strip()
    if not expr:
        return expr
    expr = re.sub(r'\b(0x[0-9A-Fa-f]+|\d+)[uUlL]+\b', r'\1', expr)
    expr = re.sub(r'\b(\d+)\s*[kK]\b', r'(\1 * 1024)', expr)
    expr = re.sub(r'\b(\d+)\s*[mM]\b', r'(\1 * 1024 * 1024)', expr)
    return expr.strip()


def _safe_eval_expr(expr: str, macros: dict[str, str], _depth: int = 0, _stack: set[str] | None = None) -> int | None:
    if _depth > 20:
        return None
    if _stack is None:
        _stack = set()

    expr = _normalize_expr(expr)
    if not expr:
        return None

    if expr.startswith('+'):
        return None

    try:
        node = ast.parse(expr, mode='eval')
    except SyntaxError:
        return None

    def _eval(n: ast.AST) -> int:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return int(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            val = _eval(n.operand)
            return val if isinstance(n.op, ast.UAdd) else -val
        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, (ast.Div, ast.FloorDiv)):
                return left // right
            if isinstance(n.op, ast.Mod):
                return left % right
            if isinstance(n.op, ast.LShift):
                return left << right
            if isinstance(n.op, ast.RShift):
                return left >> right
            if isinstance(n.op, ast.BitOr):
                return left | right
            if isinstance(n.op, ast.BitAnd):
                return left & right
            if isinstance(n.op, ast.BitXor):
                return left ^ right
            raise ValueError("Unsupported operator")
        if isinstance(n, ast.Name):
            key = n.id
            if key in _stack:
                raise ValueError("Recursive macro")
            if key not in macros:
                raise ValueError("Unknown macro")
            _stack.add(key)
            val = _safe_eval_expr(macros[key], macros, _depth + 1, _stack)
            _stack.remove(key)
            if val is None:
                raise ValueError("Macro eval failed")
            return val
        raise ValueError("Unsupported expression")

    try:
        return _eval(node)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError, RecursionError):
        return None


def _parse_macros(text: str) -> dict[str, str]:
    macros: dict[str, str] = {}
    for line in text.splitlines():
        m = _DEFINE_RE.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2)
        macros[key] = value.strip()
    return macros


def _parse_regions(text: str, macros: dict[str, str]) -> list[ScatterRegion]:
    regions: list[ScatterRegion] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        head = ''
        if '{' in line:
            head = line.split('{', 1)[0].strip()
        else:
            candidate = line.strip()
            if not candidate:
                i += 1
                continue
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].lstrip().startswith('{'):
                head = candidate
                i = j  # skip the standalone "{" line
        if not head:
            i += 1
            continue

        tokens = head.split()
        if len(tokens) < 3:
            i += 1
            continue

        name = tokens[0]
        base_expr = tokens[1]
        size_expr = None
        for token in tokens[2:]:
            if token.upper() in _ATTR_TOKENS:
                continue
            size_expr = token
            break
        if size_expr is None:
            i += 1
            continue

        base_val = _safe_eval_expr(base_expr, macros)
        size_val = _safe_eval_expr(size_expr, macros)
        if base_val is None or size_val is None:
            i += 1
            continue

        regions.append(ScatterRegion(
            name=name,
            base=base_val,
            size=size_val,
            base_expr=base_expr,
            size_expr=size_expr,
        ))
        i += 1
    return regions


def _pick_from_macros(keys: list[str], macros: dict[str, str]) -> int | None:
    for key in keys:
        if key in macros:
            val = _safe_eval_expr(macros[key], macros)
            if val is not None:
                return val
    return None


def _pick_region(regions: list[ScatterRegion], predicate: Callable[[str], bool]) -> ScatterRegion | None:
    candidates = [r for r in regions if predicate(r.name)]
    if not candidates:
        return None
    return sorted(candidates, key=lambda r: r.base)[0]


def _is_rom(name: str) -> bool:
    upper = name.upper()
    return any(k in upper for k in ('IROM', 'ROM', 'FLASH', 'CODE'))


def _is_ram(name: str) -> bool:
    upper = name.upper()
    return any(k in upper for k in ('IRAM', 'RAM', 'SRAM', 'RW', 'ZI', 'DATA'))


def _format_hex(value: int | None) -> str:
    if value is None:
        return '0x0'
    digits = max(8, len(f"{value:X}"))
    return f'0x{value:0{digits}X}'


def convert_scatter_to_ld(sct_path: str, output_path: str) -> ScatterConversion:
    if not sct_path or not os.path.isfile(sct_path):
        return ScatterConversion(ok=False, reason='scatter file missing')

    try:
        with open(sct_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            raw = f.read()
    except OSError as exc:
        return ScatterConversion(ok=False, reason=str(exc))

    cleaned = _strip_comments(raw)
    macros = _parse_macros(cleaned)
    regions = _parse_regions(cleaned, macros)

    rom_base = _pick_from_macros(['__ROM_BASE', 'ROM_BASE', '__ROM_START', 'ROM_START'], macros)
    rom_size = _pick_from_macros(['__ROM_SIZE', 'ROM_SIZE', '__ROM_LENGTH', 'ROM_LENGTH'], macros)
    ram_base = _pick_from_macros(['__RAM_BASE', 'RAM_BASE', '__RAM_START', 'RAM_START'], macros)
    ram_size = _pick_from_macros(['__RAM_SIZE', 'RAM_SIZE', '__RAM_LENGTH', 'RAM_LENGTH'], macros)

    rom_source = None
    ram_source = None
    if rom_base is not None and rom_size is not None:
        rom_source = 'macro'
    if ram_base is not None and ram_size is not None:
        ram_source = 'macro'

    if rom_base is None or rom_size is None:
        rom_region = _pick_region(regions, _is_rom)
        if rom_region:
            rom_base, rom_size = rom_region.base, rom_region.size
            rom_source = rom_region.name

    if ram_base is None or ram_size is None:
        ram_region = _pick_region(regions, _is_ram)
        if ram_region:
            ram_base, ram_size = ram_region.base, ram_region.size
            ram_source = ram_region.name

    if rom_base is None or rom_size is None or ram_base is None or ram_size is None:
        return ScatterConversion(ok=False, reason='unable to resolve ROM/RAM from scatter file')

    stack_size = _pick_from_macros(
        ['__STACK_SIZE', 'STACK_SIZE', '__STACKSIZE', 'STACKSIZE', '_Min_Stack_Size'],
        macros,
    )
    heap_size = _pick_from_macros(
        ['__HEAP_SIZE', 'HEAP_SIZE', '__HEAPSIZE', 'HEAPSIZE', '_Min_Heap_Size'],
        macros,
    )

    write_template(
        'linker_from_sct.ld.j2',
        {
            'sct_name': os.path.basename(sct_path),
            'rom_base': _format_hex(rom_base),
            'rom_size': _format_hex(rom_size),
            'ram_base': _format_hex(ram_base),
            'ram_size': _format_hex(ram_size),
            'stack_size': _format_hex(stack_size) if stack_size is not None else None,
            'heap_size': _format_hex(heap_size) if heap_size is not None else None,
            'rom_source': rom_source or '',
            'ram_source': ram_source or '',
        },
        output_path,
        encoding='utf-8-sig',
    )

    return ScatterConversion(
        ok=True,
        reason='ok',
        rom_base=rom_base,
        rom_size=rom_size,
        ram_base=ram_base,
        ram_size=ram_size,
        stack_size=stack_size,
        heap_size=heap_size,
        rom_source=rom_source,
        ram_source=ram_source,
    )
