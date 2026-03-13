# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op
from ....operators.utils import emit_op_clip, tensor_size


@register_op("Clip")
def emit_clip(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("Clip expects 1 input.")
    if "min" not in node.attrs or "max" not in node.attrs:
        raise ValueError("Clip requires min/max attributes.")
    out_tensor = node.outputs[0]
    out_dtype = ctx.dtype(out_tensor)
    out = ctx.map_ptr(out_tensor)
    a = ctx.map_ptr(node.inputs[0])
    size = tensor_size(ctx.shape(out_tensor))
    min_v = float(node.attrs["min"])
    max_v = float(node.attrs["max"])
    if out_dtype in ("int8", "int16"):
        if ctx.dtype(node.inputs[0]) != out_dtype:
            raise ValueError("Quantized Clip requires matching dtypes.")
        sa, za = ctx.qparams(node.inputs[0])
        so_zo = ctx.qparams_optional(out_tensor)
        if so_zo is None:
            so, zo = sa, za
        else:
            so, zo = so_zo
        qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
        ctx.lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
        ctx.lines.append(f"    float r = ((float){a}[i] - {za}) * {sa:.8f}f;")
        ctx.lines.append(f"    if (r < {min_v:.8f}f) r = {min_v:.8f}f;")
        ctx.lines.append(f"    if (r > {max_v:.8f}f) r = {max_v:.8f}f;")
        ctx.lines.append(f"    int q = (int)roundf(r / {so:.8f}f) + {zo};")
        ctx.lines.append(f"    if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"    if (q > {qmax}) q = {qmax};")
        ctx.lines.append(f"    {out}[i] = ({'int8_t' if out_dtype == 'int8' else 'int16_t'})q;")
        ctx.lines.append("  }")
        return
    if out_dtype != "float32":
        raise ValueError("Clip supports float32 or quantized int8/int16 only.")
    emit_op_clip(ctx.lines, out, a, size, min_v, max_v)

