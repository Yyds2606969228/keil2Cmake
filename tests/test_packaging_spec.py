# -*- coding: utf-8 -*-

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
SPEC = ROOT / 'Keil2Cmake.spec'

ALLOWED_EXTERNAL_IMPORTS = {
    'fastmcp',
    'jinja2',
    'numpy',
    'onnx',
}

BLOCKED_HIDDENIMPORTS = {
    'keil2cmake.tinyml.operators',
    'keil2cmake.tinyml.onnx_loader',
}


def _read_hiddenimports() -> list[str]:
    text = SPEC.read_text(encoding='utf-8')
    match = re.search(r'hiddenimports=\[(.*?)\],', text, flags=re.S)
    if match is None:
        raise AssertionError('hiddenimports block not found in Keil2Cmake.spec')
    return re.findall(r"'([^']+)'", match.group(1))


def _module_exists(module_name: str) -> bool:
    path = SRC / Path(*module_name.split('.'))
    return path.with_suffix('.py').exists() or (path / '__init__.py').exists()


class TestPackagingSpecConsistency(unittest.TestCase):
    def test_hiddenimports_have_no_duplicates(self) -> None:
        hiddenimports = _read_hiddenimports()
        self.assertEqual(len(hiddenimports), len(set(hiddenimports)))

    def test_internal_hiddenimports_resolve_to_source_modules(self) -> None:
        hiddenimports = _read_hiddenimports()
        missing = [
            name for name in hiddenimports
            if (name.startswith('keil2cmake.') or name.startswith('openocd_mcp')) and not _module_exists(name)
        ]
        self.assertEqual(missing, [], f'missing internal hiddenimports: {missing}')

    def test_hiddenimports_external_scope_is_explicit(self) -> None:
        hiddenimports = _read_hiddenimports()
        unknown_external = [
            name for name in hiddenimports
            if not name.startswith('keil2cmake.') and not name.startswith('openocd_mcp') and name not in ALLOWED_EXTERNAL_IMPORTS
        ]
        self.assertEqual(
            unknown_external,
            [],
            f'unexpected external hiddenimports: {unknown_external}',
        )

    def test_blocked_hiddenimports_are_not_reintroduced(self) -> None:
        hiddenimports = set(_read_hiddenimports())
        self.assertTrue(
            hiddenimports.isdisjoint(BLOCKED_HIDDENIMPORTS),
            'blocked hiddenimports were reintroduced',
        )
