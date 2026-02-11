# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import get_const_ints
from .registry import register_op


def _zp_scalar(ctx: EmitContext, name: str | None) -> int:
    if not name:
        return 0
    vals = get_const_ints(ctx.model, name)
    if len(vals) != 1:
        raise ValueError("MatMulInteger zero_point must be scalar constant.")
    return int(vals[0])


@register_op("MatMulInteger")
def emit_matmul_integer(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 2:
        raise ValueError("MatMulInteger expects at least 2 inputs.")
    a_name = node.inputs[0]
    b_name = node.inputs[1]
    a_dtype = ctx.dtype(a_name)
    b_dtype = ctx.dtype(b_name)
    out_name = node.outputs[0]
    out_dtype = ctx.dtype(out_name)
    if a_dtype not in ("int8", "int16") or b_dtype not in ("int8", "int16"):
        raise ValueError("MatMulInteger currently supports int8/int16 inputs only.")
    if out_dtype not in ("int32", "int64"):
        raise ValueError("MatMulInteger output dtype must be int32/int64.")

    a_shape = ctx.shape(a_name)
    b_shape = ctx.shape(b_name)
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise ValueError("MatMulInteger currently supports 2D tensors only.")
    m, k1 = a_shape
    k2, n = b_shape
    if k1 != k2:
        raise ValueError("MatMulInteger dimension mismatch.")

    a_zp = _zp_scalar(ctx, node.inputs[2] if len(node.inputs) >= 3 and node.inputs[2] else None)
    b_zp = _zp_scalar(ctx, node.inputs[3] if len(node.inputs) >= 4 and node.inputs[3] else None)

    a = ctx.map_ptr(a_name)
    b = ctx.map_ptr(b_name)
    out = ctx.map_ptr(out_name)
    ctx.lines.append(f"  for (size_t i = 0; i < {m}; ++i) {{")
    ctx.lines.append(f"    for (size_t j = 0; j < {n}; ++j) {{")
    ctx.lines.append("      int64_t acc = 0;")
    ctx.lines.append(f"      for (size_t t = 0; t < {k1}; ++t) {{")
    ctx.lines.append(f"        int64_t av = (int64_t){a}[i * {k1} + t] - {a_zp};")
    ctx.lines.append(f"        int64_t bv = (int64_t){b}[t * {n} + j] - {b_zp};")
    ctx.lines.append("        acc += av * bv;")
    ctx.lines.append("      }")
    if out_dtype == "int32":
        ctx.lines.append("      if (acc < -2147483648LL) acc = -2147483648LL;")
        ctx.lines.append("      if (acc > 2147483647LL) acc = 2147483647LL;")
        ctx.lines.append(f"      {out}[i * {n} + j] = (int32_t)acc;")
    else:
        ctx.lines.append(f"      {out}[i * {n} + j] = (int64_t)acc;")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
