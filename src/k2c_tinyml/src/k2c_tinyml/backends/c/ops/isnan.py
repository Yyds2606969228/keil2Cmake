# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import tensor_size
from .registry import register_op


@register_op("IsNaN")
def emit_isnan(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("IsNaN expects 1 input.")
    in_name = node.inputs[0]
    out_name = node.outputs[0]
    if ctx.dtype(in_name) != "float32":
        raise ValueError("IsNaN currently supports float32 input only.")
    if ctx.dtype(out_name) != "bool":
        raise ValueError("IsNaN output dtype must be bool.")
    if ctx.shape(in_name) != ctx.shape(out_name):
        raise ValueError("IsNaN output shape mismatch.")

    inp = ctx.map_ptr(in_name)
    out = ctx.map_ptr(out_name)
    size = tensor_size(ctx.shape(out_name))
    ctx.lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
    ctx.lines.append(f"    {out}[i] = isnan({inp}[i]) ? 1u : 0u;")
    ctx.lines.append("  }")
