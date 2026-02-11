# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import tensor_size
from .registry import register_op


@register_op("Not")
def emit_not(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("Not expects 1 input.")
    in_name = node.inputs[0]
    out_name = node.outputs[0]
    if ctx.dtype(in_name) != "bool" or ctx.dtype(out_name) != "bool":
        raise ValueError("Not requires bool input/output.")
    inp = ctx.map_ptr(in_name)
    out = ctx.map_ptr(out_name)
    size = tensor_size(ctx.shape(out_name))
    ctx.lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
    ctx.lines.append(f"    {out}[i] = ({inp}[i] == 0) ? 1u : 0u;")
    ctx.lines.append("  }")
