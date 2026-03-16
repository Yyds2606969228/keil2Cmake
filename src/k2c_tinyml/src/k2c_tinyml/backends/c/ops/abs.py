# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op
from ....operators.utils import emit_op_unary_func, tensor_size


@register_op("Abs")
def emit_abs(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("Abs expects 1 input.")
    out_tensor = node.outputs[0]
    out_dtype = ctx.dtype(out_tensor)
    out = ctx.map_ptr(out_tensor)
    inp = ctx.map_ptr(node.inputs[0])
    size = tensor_size(ctx.shape(out_tensor))
    if out_dtype in ("int8", "int16"):
        if ctx.dtype(node.inputs[0]) != out_dtype:
            raise ValueError("Quantized Abs requires matching dtypes.")
        sa, za = ctx.qparams(node.inputs[0])
        so_zo = ctx.qparams_optional(out_tensor)
        if so_zo is None:
            so, zo = sa, za
        else:
            so, zo = so_zo
        qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
        ctx.lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
        ctx.lines.append(f"    float r = ((float){inp}[i] - {za}) * {sa:.8f}f;")
        ctx.lines.append("    if (r < 0.0f) r = -r;")
        ctx.lines.append(f"    int q = (int)roundf(r / {so:.8f}f) + {zo};")
        ctx.lines.append(f"    if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"    if (q > {qmax}) q = {qmax};")
        ctx.lines.append(f"    {out}[i] = ({'int8_t' if out_dtype == 'int8' else 'int16_t'})q;")
        ctx.lines.append("  }")
        return
    if out_dtype != "float32":
        raise ValueError("Abs supports float32 or quantized int8/int16 only.")
    emit_op_unary_func(ctx.lines, out, inp, size, "fabsf")

