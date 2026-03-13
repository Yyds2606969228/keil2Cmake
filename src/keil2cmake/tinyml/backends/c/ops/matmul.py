# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op
from ....operators.utils import emit_op_matmul


@register_op("MatMul")
def emit_matmul(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 2:
        raise ValueError("MatMul expects 2 inputs.")
    out_tensor = node.outputs[0]
    out_dtype = ctx.dtype(out_tensor)
    out = ctx.map_ptr(out_tensor)
    a_name, b_name = node.inputs
    a = ctx.map_ptr(a_name)
    b = ctx.map_ptr(b_name)
    a_shape = ctx.shape(a_name)
    b_shape = ctx.shape(b_name)
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise ValueError("MatMul supports 2D tensors only.")
    m, k1 = a_shape
    k2, n = b_shape
    if k1 != k2:
        raise ValueError("MatMul dimension mismatch.")
    if out_dtype in ("int8", "int16"):
        if ctx.dtype(a_name) != out_dtype or ctx.dtype(b_name) != out_dtype:
            raise ValueError("Quantized MatMul requires matching dtypes.")
        sa, za = ctx.qparams(a_name)
        sb, zb = ctx.qparams(b_name)
        so, zo = ctx.qparams(out_tensor)
        qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
        ctx.lines.append(f"  for (size_t i = 0; i < {m}; ++i) {{")
        ctx.lines.append(f"    for (size_t j = 0; j < {n}; ++j) {{")
        ctx.lines.append("      float sum = 0.0f;")
        ctx.lines.append(f"      for (size_t t = 0; t < {k1}; ++t) {{")
        ctx.lines.append(
            f"        float ra = ((float){a}[i * {k1} + t] - {za}) * {sa:.8f}f;"
        )
        ctx.lines.append(
            f"        float rb = ((float){b}[t * {n} + j] - {zb}) * {sb:.8f}f;"
        )
        ctx.lines.append("        sum += ra * rb;")
        ctx.lines.append("      }")
        ctx.lines.append(f"      int q = (int)roundf(sum / {so:.8f}f) + {zo};")
        ctx.lines.append(f"      if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"      if (q > {qmax}) q = {qmax};")
        ctx.lines.append(
            f"      {out}[i * {n} + j] = ({'int8_t' if out_dtype == 'int8' else 'int16_t'})q;"
        )
        ctx.lines.append("    }")
        ctx.lines.append("  }")
        return
    if out_dtype != "float32":
        raise ValueError("MatMul supports float32 or quantized int8/int16 only.")
    emit_op_matmul(ctx.lines, out, a, b, m, k1, n)

