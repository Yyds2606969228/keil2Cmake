# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op
from ....operators.utils import emit_op_relu, tensor_size


@register_op("Relu")
def emit_relu(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("Relu expects 1 input.")
    out_tensor = node.outputs[0]
    out_dtype = ctx.dtype(out_tensor)
    out = ctx.map_ptr(out_tensor)
    a = ctx.map_ptr(node.inputs[0])
    size = tensor_size(ctx.shape(out_tensor))
    if out_dtype in ("int8", "int16"):
        if ctx.dtype(node.inputs[0]) != out_dtype:
            raise ValueError("Quantized Relu requires matching dtypes.")
        sa, za = ctx.qparams(node.inputs[0])
        so, zo = ctx.qparams(out_tensor)
        if sa != so or za != zo:
            raise ValueError("Quantized Relu requires same scale/zero.")
        ctx.lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
        ctx.lines.append(f"    int v = (int){a}[i];")
        ctx.lines.append(f"    if (v < {zo}) v = {zo};")
        ctx.lines.append(f"    {out}[i] = ({'int8_t' if out_dtype == 'int8' else 'int16_t'})v;")
        ctx.lines.append("  }")
        return
    if out_dtype != "float32":
        raise ValueError("Relu supports float32 or quantized int8/int16 only.")
    emit_op_relu(ctx.lines, out, a, size)

