# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import tensor_size
from .registry import register_op


@register_op("IsInf")
def emit_isinf(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("IsInf expects 1 input.")
    in_name = node.inputs[0]
    out_name = node.outputs[0]
    if ctx.dtype(in_name) != "float32":
        raise ValueError("IsInf currently supports float32 input only.")
    if ctx.dtype(out_name) != "bool":
        raise ValueError("IsInf output dtype must be bool.")
    if ctx.shape(in_name) != ctx.shape(out_name):
        raise ValueError("IsInf output shape mismatch.")

    detect_negative = int(node.attrs.get("detect_negative", 1))
    detect_positive = int(node.attrs.get("detect_positive", 1))
    if detect_negative not in (0, 1) or detect_positive not in (0, 1):
        raise ValueError("IsInf detect_negative/detect_positive must be 0 or 1.")

    inp = ctx.map_ptr(in_name)
    out = ctx.map_ptr(out_name)
    size = tensor_size(ctx.shape(out_name))
    ctx.lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
    ctx.lines.append(f"    float v = {inp}[i];")
    ctx.lines.append("    uint8_t m = isinf(v) ? 1u : 0u;")
    if detect_negative == 0:
        ctx.lines.append("    if (m && v < 0.0f) m = 0u;")
    if detect_positive == 0:
        ctx.lines.append("    if (m && v > 0.0f) m = 0u;")
    ctx.lines.append(f"    {out}[i] = m;")
    ctx.lines.append("  }")
