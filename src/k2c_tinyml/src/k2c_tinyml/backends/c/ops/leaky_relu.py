# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op
from ....operators.utils import emit_op_leaky_relu, emit_op_unary_quant, tensor_size


@register_op("LeakyRelu")
def emit_leaky_relu(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("LeakyRelu expects 1 input.")
    out_tensor = node.outputs[0]
    out_dtype = ctx.dtype(out_tensor)
    out = ctx.map_ptr(out_tensor)
    a = ctx.map_ptr(node.inputs[0])
    size = tensor_size(ctx.shape(out_tensor))
    alpha = float(node.attrs.get("alpha", 0.01))
    if out_dtype in ("int8", "int16"):
        if ctx.dtype(node.inputs[0]) != out_dtype:
            raise ValueError("Quantized LeakyRelu requires matching dtypes.")
        sa, za = ctx.qparams(node.inputs[0])
        so_zo = ctx.qparams_optional(out_tensor)
        if so_zo is None:
            so, zo = sa, za
        else:
            so, zo = so_zo
        expr = f"(r >= 0.0f ? r : ({alpha:.8f}f * r))"
        emit_op_unary_quant(ctx.lines, out, a, size, expr, out_dtype, sa, za, so, zo)
        return
    if out_dtype != "float32":
        raise ValueError("LeakyRelu supports float32 or quantized int8/int16 only.")
    emit_op_leaky_relu(ctx.lines, out, a, size, alpha)

