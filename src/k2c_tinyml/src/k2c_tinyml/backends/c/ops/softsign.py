# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op
from ....operators.utils import emit_op_unary_quant, tensor_size


@register_op("Softsign")
def emit_softsign(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("Softsign expects 1 input.")
    out_tensor = node.outputs[0]
    out_dtype = ctx.dtype(out_tensor)
    out = ctx.map_ptr(out_tensor)
    inp = ctx.map_ptr(node.inputs[0])
    size = tensor_size(ctx.shape(out_tensor))
    expr = "(r / (1.0f + fabsf(r)))"
    if out_dtype in ("int8", "int16"):
        if ctx.dtype(node.inputs[0]) != out_dtype:
            raise ValueError("Quantized Softsign requires matching dtypes.")
        sa, za = ctx.qparams(node.inputs[0])
        so_zo = ctx.qparams_optional(out_tensor)
        if so_zo is None:
            so, zo = sa, za
        else:
            so, zo = so_zo
        emit_op_unary_quant(ctx.lines, out, inp, size, expr, out_dtype, sa, za, so, zo)
        return
    if out_dtype != "float32":
        raise ValueError("Softsign supports float32 or quantized int8/int16 only.")
    ctx.lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
    ctx.lines.append(f"    float v = {inp}[i];")
    ctx.lines.append(f"    {out}[i] = v / (1.0f + fabsf(v));")
    ctx.lines.append("  }")
