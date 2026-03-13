# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import emit_op_copy, tensor_size
from .registry import register_op


@register_op("Dropout")
def emit_dropout(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 1:
        raise ValueError("Dropout expects at least 1 input.")
    if len(node.outputs) != 1:
        raise ValueError("Dropout currently supports 1 output only.")

    x_name = node.inputs[0]
    out_name = node.outputs[0]
    in_dtype = ctx.dtype(x_name)
    out_dtype = ctx.dtype(out_name)
    if in_dtype != out_dtype:
        raise ValueError("Dropout requires input/output dtype to match.")
    if ctx.shape(x_name) != ctx.shape(out_name):
        raise ValueError("Dropout requires input/output shape to match.")

    inp = ctx.map_ptr(x_name)
    out = ctx.map_ptr(out_name)
    size = tensor_size(ctx.shape(out_name))

    if out_dtype not in ("float32", "int8", "int16", "bool"):
        raise ValueError("Dropout supports float32/int8/int16/bool only.")

    if out_dtype in ("int8", "int16"):
        q_in = ctx.qparams_optional(x_name)
        q_out = ctx.qparams_optional(out_name)
        if q_in is not None and q_out is not None:
            si, zi = q_in
            so, zo = q_out
            if abs(si - so) <= 1e-12 and zi == zo:
                emit_op_copy(ctx.lines, out, inp, size)
                return
            qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
            ctype = "int8_t" if out_dtype == "int8" else "int16_t"
            ctx.lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
            ctx.lines.append(f"    float r = ((float){inp}[i] - {zi}) * {si:.8f}f;")
            ctx.lines.append(f"    int q = (int)roundf(r / {so:.8f}f) + {zo};")
            ctx.lines.append(f"    if (q < {qmin}) q = {qmin};")
            ctx.lines.append(f"    if (q > {qmax}) q = {qmax};")
            ctx.lines.append(f"    {out}[i] = ({ctype})q;")
            ctx.lines.append("  }")
            return
        if q_in is None and q_out is None:
            emit_op_copy(ctx.lines, out, inp, size)
            return
        raise ValueError("Dropout quantization params are inconsistent.")

    emit_op_copy(ctx.lines, out, inp, size)
