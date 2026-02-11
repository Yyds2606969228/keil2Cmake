# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import emit_op_unary_quant, tensor_size
from .registry import register_op


@register_op("Shrink")
def emit_shrink(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("Shrink expects 1 input.")
    in_name = node.inputs[0]
    out_name = node.outputs[0]
    in_dtype = ctx.dtype(in_name)
    out_dtype = ctx.dtype(out_name)
    if in_dtype != out_dtype:
        raise ValueError("Shrink output dtype must match input dtype.")

    lambd = float(node.attrs.get("lambd", 0.5))
    bias = float(node.attrs.get("bias", 0.0))
    inp = ctx.map_ptr(in_name)
    out = ctx.map_ptr(out_name)
    size = tensor_size(ctx.shape(out_name))

    if out_dtype == "float32":
        ctx.lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
        ctx.lines.append(f"    float v = {inp}[i];")
        ctx.lines.append(f"    if (v < -{lambd:.8f}f) {out}[i] = v + {bias:.8f}f;")
        ctx.lines.append(f"    else if (v > {lambd:.8f}f) {out}[i] = v - {bias:.8f}f;")
        ctx.lines.append(f"    else {out}[i] = 0.0f;")
        ctx.lines.append("  }")
        return

    if out_dtype in ("int8", "int16"):
        sa, za = ctx.qparams(in_name)
        so, zo = ctx.qparams(out_name)
        emit_op_unary_quant(
            ctx.lines,
            out,
            inp,
            size,
            f"(r < -{lambd:.8f}f) ? (r + {bias:.8f}f) : ((r > {lambd:.8f}f) ? (r - {bias:.8f}f) : 0.0f)",
            out_dtype,
            sa,
            za,
            so,
            zo,
        )
        return

    raise ValueError("Shrink currently supports float32/int8/int16 only.")
