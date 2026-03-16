# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import unittest

raise unittest.SkipTest("tinyml 已抽离到 k2c_tinyml 子项目，主仓库不再运行该测试。")

if importlib.util.find_spec('onnx') is None:
    raise unittest.SkipTest('tinyml optional dependency onnx is missing')

import onnx


ROOT = Path(__file__).resolve().parents[1]
OPS_DIR = ROOT / "src" / "keil2cmake" / "tinyml" / "backends" / "c" / "ops"


def _registered_ops() -> set[str]:
    out: set[str] = set()
    for py in OPS_DIR.glob("*.py"):
        if py.name in ("__init__.py", "registry.py"):
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for dec in node.decorator_list:
                if (
                    isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Name)
                    and dec.func.id == "register_op"
                    and dec.args
                    and isinstance(dec.args[0], ast.Constant)
                    and isinstance(dec.args[0].value, str)
                ):
                    out.add(dec.args[0].value)
    return out


def _opset12_ops() -> set[str]:
    latest: dict[str, int] = {}
    for schema in onnx.defs.get_all_schemas_with_history():
        if schema.domain not in ("", "ai.onnx"):
            continue
        if schema.since_version > 12:
            continue
        prev = latest.get(schema.name)
        if prev is None or schema.since_version > prev:
            latest[schema.name] = schema.since_version
    return set(latest.keys())


class TestOpset12Coverage(unittest.TestCase):
    def test_all_opset12_ops_registered(self) -> None:
        missing = sorted(_opset12_ops() - _registered_ops())
        self.assertEqual(missing, [], msg=f"Missing opset12 handlers: {', '.join(missing)}")

