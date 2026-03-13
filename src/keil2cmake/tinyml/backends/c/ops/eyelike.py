# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op


def _one_literal(dtype: str) -> str:
    if dtype == "float32":
        return "1.0f"
    if dtype in ("int8", "int16", "int32", "int64"):
        return "1"
    if dtype == "bool":
        return "1"
    raise ValueError("EyeLike output dtype is unsupported.")


def _zero_literal(dtype: str) -> str:
    if dtype == "float32":
        return "0.0f"
    if dtype in ("int8", "int16", "int32", "int64", "bool"):
        return "0"
    raise ValueError("EyeLike output dtype is unsupported.")


@register_op("EyeLike")
def emit_eyelike(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("EyeLike expects 1 input.")
    in_name = node.inputs[0]
    out_name = node.outputs[0]
    in_shape = [int(v) for v in ctx.shape(in_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    if len(in_shape) != 2 or len(out_shape) != 2:
        raise ValueError("EyeLike currently supports rank-2 tensors only.")
    if in_shape != out_shape:
        raise ValueError("EyeLike output shape must equal input shape.")

    out_dtype = ctx.dtype(out_name)
    _ = _one_literal(out_dtype)
    _ = _zero_literal(out_dtype)
    k = int(node.attrs.get("k", 0))

    rows = in_shape[0]
    cols = in_shape[1]
    out = ctx.map_ptr(out_name)
    one = _one_literal(out_dtype)
    zero = _zero_literal(out_dtype)

    ctx.lines.append(f"  for (size_t r = 0; r < {rows}; ++r) {{")
    ctx.lines.append(f"    for (size_t c = 0; c < {cols}; ++c) {{")
    ctx.lines.append(f"      int on_diag = ((int)c - (int)r) == {k};")
    ctx.lines.append(f"      {out}[r * (size_t){cols} + c] = on_diag ? ({one}) : ({zero});")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
