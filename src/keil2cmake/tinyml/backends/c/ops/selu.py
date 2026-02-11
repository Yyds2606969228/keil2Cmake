# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import emit_op_unary_quant, tensor_size
from .registry import register_op


@register_op("Selu")
def emit_selu(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("Selu expects 1 input.")
    out_tensor = node.outputs[0]
    out_dtype = ctx.dtype(out_tensor)
    out = ctx.map_ptr(out_tensor)
    inp = ctx.map_ptr(node.inputs[0])
    size = tensor_size(ctx.shape(out_tensor))
    alpha = float(node.attrs.get("alpha", 1.6732631921768188))
    gamma = float(node.attrs.get("gamma", 1.0507010221481323))

    if out_dtype in ("int8", "int16"):
        if ctx.dtype(node.inputs[0]) != out_dtype:
            raise ValueError("Quantized Selu requires matching dtypes.")
        sa, za = ctx.qparams(node.inputs[0])
        so_zo = ctx.qparams_optional(out_tensor)
        if so_zo is None:
            so, zo = sa, za
        else:
            so, zo = so_zo
        expr = (
            f"(r > 0.0f ? ({gamma:.8f}f * r) "
            f": ({gamma:.8f}f * {alpha:.8f}f * (expf(r) - 1.0f)))"
        )
        emit_op_unary_quant(ctx.lines, out, inp, size, expr, out_dtype, sa, za, so, zo)
        return

    if out_dtype != "float32":
        raise ValueError("Selu supports float32 or quantized int8/int16 only.")
    ctx.lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
    ctx.lines.append(f"    float v = {inp}[i];")
    ctx.lines.append(
        f"    {out}[i] = v > 0.0f ? ({gamma:.8f}f * v) : ({gamma:.8f}f * {alpha:.8f}f * (expf(v) - 1.0f));"
    )
    ctx.lines.append("  }")
