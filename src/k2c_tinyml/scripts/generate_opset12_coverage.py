#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

OUTPUT = ROOT / "docs" / "onnx_opset12_coverage_matrix.md"
CODEGEN = SRC / "k2c_tinyml" / "codegen.py"
C_OPS_DIR = SRC / "k2c_tinyml" / "backends" / "c" / "ops"
CMSIS_OPS_DIR = SRC / "k2c_tinyml" / "backends" / "cmsis_nn" / "ops"


@dataclass(frozen=True)
class Row:
    op: str
    c_native: bool
    cmsis_native: bool
    cmsis_backend: bool
    quantized: bool
    note: str


C_CONSTRAINTS: dict[str, str] = {
    "Softmax": "supports static rank>=1; axis can be any valid axis",
    "Gemm": "subset: transA=0, transB=0, alpha=1, beta=1",
    "Conv": "subset: 4D NCHW; N>=1; group>=1 with Cin=group*CperG and Cout%group==0",
    "ConvTranspose": "subset: 4D NCHW; currently fp32 and group=1",
    "MaxPool": "static rank>=3 (N,C,*)",
    "AveragePool": "static rank>=3 (N,C,*); supports count_include_pad",
    "GlobalAveragePool": "static rank>=3 (N,C,*), output [N,C,1,...,1]",
    "GlobalMaxPool": "static rank>=3 (N,C,*), output [N,C,1,...,1]",
    "GlobalLpPool": "static rank>=3 (N,C,*); p>0",
    "BatchNormalization": "static rank>=2 (N,C,*)",
    "MeanVarianceNormalization": "subset: fp32, 4D NCHW, axes=[0,2,3]",
    "InstanceNormalization": "rank>=3; computes on N,C,* (fp32)",
    "LRN": "rank>=3; channel window normalization (fp32)",
    "LpNormalization": "subset: fp32, static rank>=1",
    "LpPool": "static rank>=3 (N,C,*); p>0",
}

CMSIS_CONSTRAINTS: dict[str, str] = {}


def _extract_string_set(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Set):
        return {
            elt.value
            for elt in node.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        }
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "set"
        and len(node.args) == 1
        and isinstance(node.args[0], (ast.List, ast.Tuple, ast.Set))
    ):
        out: set[str] = set()
        for elt in node.args[0].elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                out.add(elt.value)
        return out
    return set()


def _load_quant_ops(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            if any(t.id == "quant_ops" for t in targets):
                out = _extract_string_set(node.value)
                if out:
                    return out
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "quant_ops":
            if node.value is None:
                continue
            out = _extract_string_set(node.value)
            if out:
                return out
    return set()


def _scan_registered_ops(ops_dir: Path) -> set[str]:
    if not ops_dir.exists():
        return set()
    out: set[str] = set()
    for py in sorted(ops_dir.glob("*.py")):
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


def _opset12_ops() -> list[str]:
    latest_by_name: dict[str, int] = {}
    for schema in onnx.defs.get_all_schemas_with_history():
        if schema.domain not in ("", "ai.onnx"):
            continue
        if schema.since_version > 12:
            continue
        prev = latest_by_name.get(schema.name)
        if prev is None or schema.since_version > prev:
            latest_by_name[schema.name] = schema.since_version
    return sorted(latest_by_name.keys())


def _mark(v: bool) -> str:
    return "Y" if v else "N"


def _build_rows(ops: list[str], quant_ops: set[str], c_ops: set[str], cmsis_ops: set[str]) -> list[Row]:
    rows: list[Row] = []
    for op in ops:
        c_native = op in c_ops
        cmsis_native = op in cmsis_ops
        cmsis_backend = c_native or cmsis_native
        quantized = op in quant_ops if quant_ops else False
        notes: list[str] = []
        if op in C_CONSTRAINTS:
            notes.append(f"C: {C_CONSTRAINTS[op]}")
        if op in CMSIS_CONSTRAINTS:
            notes.append(f"CMSIS: {CMSIS_CONSTRAINTS[op]}")
        if not notes and c_native:
            notes.append("basic support")
        if not notes and not c_native and not cmsis_native:
            notes.append("not implemented")
        rows.append(
            Row(
                op=op,
                c_native=c_native,
                cmsis_native=cmsis_native,
                cmsis_backend=cmsis_backend,
                quantized=quantized,
                note="; ".join(notes),
            )
        )
    return rows


def _render(rows: list[Row], quant_ops: set[str]) -> str:
    total = len(rows)
    c_supported = sum(1 for r in rows if r.c_native)
    cmsis_native = sum(1 for r in rows if r.cmsis_native)
    cmsis_backend = sum(1 for r in rows if r.cmsis_backend)
    quant_cov = sum(1 for r in rows if r.quantized and r.c_native)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    lines: list[str] = []
    lines.append("# ONNX Opset12 Coverage Matrix")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{now}`")
    lines.append("- Scope: `domain in ('', 'ai.onnx')` and `since_version <= 12`")
    lines.append("- Default CLI: `K2CTinyML onnx --model <model.onnx> --weights flash --emit c`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|---|---:|")
    lines.append(f"| Opset12 operators | {total} |")
    lines.append(f"| C native support | {c_supported} |")
    lines.append(f"| CMSIS-NN native support | {cmsis_native} |")
    lines.append(f"| `backend=cmsis-nn` available (with C fallback) | {cmsis_backend} |")
    lines.append(f"| Quantized coverage on C (`quant_ops` ∩ C support) | {quant_cov} |")
    lines.append("")
    lines.append("## Matrix")
    lines.append("")
    lines.append("| Operator | C | CMSIS-NN(native) | cmsis-nn backend | Quant(int8/int16) | Notes |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| {r.op} | {_mark(r.c_native)} | {_mark(r.cmsis_native)} | "
            f"{_mark(r.cmsis_backend)} | {_mark(r.quantized)} | {r.note} |"
        )
    lines.append("")
    if quant_ops:
        lines.append("## Quant Operator Set (`codegen.py::quant_ops`)")
        lines.append("")
        lines.append("```text")
        lines.append(", ".join(sorted(quant_ops)))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    quant_ops = _load_quant_ops(CODEGEN) if CODEGEN.exists() else set()
    c_ops = _scan_registered_ops(C_OPS_DIR)
    cmsis_ops = _scan_registered_ops(CMSIS_OPS_DIR)
    ops = _opset12_ops()
    rows = _build_rows(ops, quant_ops, c_ops, cmsis_ops)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(_render(rows, quant_ops), encoding="utf-8")
    print(f"[ok] wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

