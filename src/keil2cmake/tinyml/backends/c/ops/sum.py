# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import tensor_size
from .registry import register_op


@register_op("Sum")
def emit_sum(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 1:
        raise ValueError("Sum expects at least 1 input.")
    out_name = node.outputs[0]
    out_dtype = ctx.dtype(out_name)
    out_shape = ctx.shape(out_name)
    out_size = tensor_size(out_shape)
    out_ptr = ctx.map_ptr(out_name)

    if out_dtype in ("int8", "int16"):
        for name in node.inputs:
            if ctx.dtype(name) != out_dtype:
                raise ValueError("Quantized Sum requires matching input/output dtypes.")
            if ctx.shape(name) != out_shape:
                raise ValueError("Quantized Sum requires equal input/output shapes.")
        so, zo = ctx.qparams(out_name)
        qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
        out_ctype = "int8_t" if out_dtype == "int8" else "int16_t"
        ctx.lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {{")
        ctx.lines.append("    float ro = 0.0f;")
        for idx, name in enumerate(node.inputs):
            inp = ctx.map_ptr(name)
            si, zi = ctx.qparams(name)
            ctx.lines.append(f"    float r{idx} = ((float){inp}[i] - {zi}) * {si:.8f}f;")
            ctx.lines.append(f"    ro += r{idx};")
        ctx.lines.append(f"    int q = (int)roundf(ro / {so:.8f}f) + {zo};")
        ctx.lines.append(f"    if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"    if (q > {qmax}) q = {qmax};")
        ctx.lines.append(f"    {out_ptr}[i] = ({out_ctype})q;")
        ctx.lines.append("  }")
        return

    if out_dtype != "float32":
        raise ValueError("Sum supports float32 or quantized int8/int16 only.")
    for name in node.inputs:
        if ctx.dtype(name) != "float32":
            raise ValueError("Float Sum requires float32 inputs.")
        if ctx.shape(name) != out_shape:
            raise ValueError("Float Sum currently requires equal input/output shapes.")
    ctx.lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {{")
    ctx.lines.append("    float acc = 0.0f;")
    for name in node.inputs:
        inp = ctx.map_ptr(name)
        ctx.lines.append(f"    acc += {inp}[i];")
    ctx.lines.append(f"    {out_ptr}[i] = acc;")
    ctx.lines.append("  }")
