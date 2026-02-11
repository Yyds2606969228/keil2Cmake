# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import get_const_ints, get_const_scalar
from .registry import register_op


def _scalar_int(ctx: EmitContext, name: str) -> int:
    vals = get_const_ints(ctx.model, name)
    if len(vals) != 1:
        raise ValueError("QLinearMatMul integer parameter must be scalar constant.")
    return int(vals[0])


def _scalar_float(ctx: EmitContext, name: str) -> float:
    return float(get_const_scalar(ctx.model, name))


@register_op("QLinearMatMul")
def emit_qlinear_matmul(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 8:
        raise ValueError(
            "QLinearMatMul expects 8 inputs: a,a_scale,a_zero,b,b_scale,b_zero,y_scale,y_zero."
        )
    a_name = node.inputs[0]
    b_name = node.inputs[3]
    out_name = node.outputs[0]
    a_dtype = ctx.dtype(a_name)
    b_dtype = ctx.dtype(b_name)
    out_dtype = ctx.dtype(out_name)
    if a_dtype not in ("int8", "int16") or b_dtype not in ("int8", "int16"):
        raise ValueError("QLinearMatMul currently supports int8/int16 inputs only.")
    if out_dtype not in ("int8", "int16"):
        raise ValueError("QLinearMatMul currently supports int8/int16 output only.")

    a_shape = ctx.shape(a_name)
    b_shape = ctx.shape(b_name)
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise ValueError("QLinearMatMul currently supports 2D tensors only.")
    m, k1 = a_shape
    k2, n = b_shape
    if k1 != k2:
        raise ValueError("QLinearMatMul dimension mismatch.")

    a_scale = _scalar_float(ctx, node.inputs[1])
    a_zero = _scalar_int(ctx, node.inputs[2])
    b_scale = _scalar_float(ctx, node.inputs[4])
    b_zero = _scalar_int(ctx, node.inputs[5])
    y_scale = _scalar_float(ctx, node.inputs[6])
    y_zero = _scalar_int(ctx, node.inputs[7])
    if y_scale == 0.0:
        raise ValueError("QLinearMatMul y_scale must be non-zero.")

    if out_dtype == "int8":
        qmin, qmax, ctype = -128, 127, "int8_t"
    else:
        qmin, qmax, ctype = -32768, 32767, "int16_t"

    a = ctx.map_ptr(a_name)
    b = ctx.map_ptr(b_name)
    out = ctx.map_ptr(out_name)
    mul_scale = a_scale * b_scale
    ctx.lines.append(f"  for (size_t i = 0; i < {m}; ++i) {{")
    ctx.lines.append(f"    for (size_t j = 0; j < {n}; ++j) {{")
    ctx.lines.append("      int64_t acc = 0;")
    ctx.lines.append(f"      for (size_t t = 0; t < {k1}; ++t) {{")
    ctx.lines.append(f"        int64_t av = (int64_t){a}[i * {k1} + t] - {a_zero};")
    ctx.lines.append(f"        int64_t bv = (int64_t){b}[t * {n} + j] - {b_zero};")
    ctx.lines.append("        acc += av * bv;")
    ctx.lines.append("      }")
    ctx.lines.append(f"      float real_v = (float)acc * {mul_scale:.12g}f;")
    ctx.lines.append(f"      int q = (int)roundf(real_v / {y_scale:.12g}f) + {y_zero};")
    ctx.lines.append(f"      if (q < {qmin}) q = {qmin};")
    ctx.lines.append(f"      if (q > {qmax}) q = {qmax};")
    ctx.lines.append(f"      {out}[i * {n} + j] = ({ctype})q;")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
