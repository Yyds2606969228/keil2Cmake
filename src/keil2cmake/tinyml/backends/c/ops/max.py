# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op
from ....operators.utils import emit_op_binary_broadcast_func, tensor_size


@register_op("Max")
def emit_max(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 2:
        raise ValueError("Max expects 2 inputs.")
    out_tensor = node.outputs[0]
    out_dtype = ctx.dtype(out_tensor)
    out = ctx.map_ptr(out_tensor)
    a = ctx.map_ptr(node.inputs[0])
    b = ctx.map_ptr(node.inputs[1])
    out_shape = ctx.shape(out_tensor)
    a_shape = ctx.shape(node.inputs[0])
    b_shape = ctx.shape(node.inputs[1])
    if out_dtype in ("int8", "int16"):
        if out_shape != a_shape or out_shape != b_shape:
            raise ValueError("Quantized Max requires equal shapes.")
        if ctx.dtype(node.inputs[0]) != out_dtype or ctx.dtype(node.inputs[1]) != out_dtype:
            raise ValueError("Quantized Max requires matching dtypes.")
        sa, za = ctx.qparams(node.inputs[0])
        sb, zb = ctx.qparams(node.inputs[1])
        so_zo = ctx.qparams_optional(out_tensor)
        if so_zo is None:
            if sa != sb or za != zb:
                raise ValueError("Quantized Max requires output qparams when inputs differ.")
            so, zo = sa, za
        else:
            so, zo = so_zo
        qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
        size = tensor_size(out_shape)
        ctx.lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
        ctx.lines.append(f"    float ra = ((float){a}[i] - {za}) * {sa:.8f}f;")
        ctx.lines.append(f"    float rb = ((float){b}[i] - {zb}) * {sb:.8f}f;")
        ctx.lines.append("    float r = ra > rb ? ra : rb;")
        ctx.lines.append(f"    int q = (int)roundf(r / {so:.8f}f) + {zo};")
        ctx.lines.append(f"    if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"    if (q > {qmax}) q = {qmax};")
        ctx.lines.append(f"    {out}[i] = ({'int8_t' if out_dtype == 'int8' else 'int16_t'})q;")
        ctx.lines.append("  }")
        return
    if out_dtype != "float32":
        raise ValueError("Max supports float32 or quantized int8/int16 only.")
    emit_op_binary_broadcast_func(ctx.lines, out, a, b, out_shape, a_shape, b_shape, "fmaxf")

