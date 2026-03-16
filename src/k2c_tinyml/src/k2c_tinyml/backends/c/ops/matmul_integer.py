# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import tensor_size
from .registry import register_op
from .matmul_common import build_matmul_batch_plan


def _zp_scalar_expr(ctx: EmitContext, name: str | None) -> str:
    if not name:
        return "0"
    shape = [int(v) for v in ctx.shape(name)]
    size = 1 if len(shape) == 0 else tensor_size(shape)
    if size != 1:
        raise ValueError("MatMulInteger zero_point must be scalar.")
    zp = ctx.map_ptr(name)
    return f"((int64_t){zp}[0])"


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

    a_shape = [int(v) for v in ctx.shape(a_name)]
    b_shape = [int(v) for v in ctx.shape(b_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    plan = build_matmul_batch_plan("MatMulInteger", a_shape, b_shape)
    expect_out_shape = [*plan.batch_shape, plan.m, plan.n]
    if out_shape != expect_out_shape:
        raise ValueError("MatMulInteger output shape mismatch.")

    a_zp_expr = _zp_scalar_expr(ctx, node.inputs[2] if len(node.inputs) >= 3 and node.inputs[2] else None)
    b_zp_expr = _zp_scalar_expr(ctx, node.inputs[3] if len(node.inputs) >= 4 and node.inputs[3] else None)

    a = ctx.map_ptr(a_name)
    b = ctx.map_ptr(b_name)
    out = ctx.map_ptr(out_name)
    a_zp_sym = ctx.next_symbol("k2c_mmi_azp")
    b_zp_sym = ctx.next_symbol("k2c_mmi_bzp")
    ctx.lines.append(f"  int64_t {a_zp_sym} = {a_zp_expr};")
    ctx.lines.append(f"  int64_t {b_zp_sym} = {b_zp_expr};")

    batch_rank = len(plan.batch_shape)
    out_batch_dims = ctx.next_symbol("k2c_mmi_batch_dims")
    a_batch_strides = ctx.next_symbol("k2c_mmi_a_batch_strides")
    b_batch_strides = ctx.next_symbol("k2c_mmi_b_batch_strides")
    dims_vals = ", ".join(str(int(v)) for v in plan.batch_shape) if batch_rank > 0 else "1"
    a_stride_vals = ", ".join(str(int(v)) for v in plan.a_batch_strides) if batch_rank > 0 else "0"
    b_stride_vals = ", ".join(str(int(v)) for v in plan.b_batch_strides) if batch_rank > 0 else "0"
    ctx.lines.append(f"  static const int32_t {out_batch_dims}[{max(1, batch_rank)}] = {{ {dims_vals} }};")
    ctx.lines.append(f"  static const int32_t {a_batch_strides}[{max(1, batch_rank)}] = {{ {a_stride_vals} }};")
    ctx.lines.append(f"  static const int32_t {b_batch_strides}[{max(1, batch_rank)}] = {{ {b_stride_vals} }};")
    ctx.lines.append(f"  for (size_t batch_i = 0; batch_i < {plan.batch_size}; ++batch_i) {{")
    ctx.lines.append("    size_t a_batch_off = 0;")
    ctx.lines.append("    size_t b_batch_off = 0;")
    if batch_rank > 0:
        ctx.lines.append("    size_t tmp = batch_i;")
        ctx.lines.append(f"    for (int axis = {batch_rank - 1}; axis >= 0; --axis) {{")
        ctx.lines.append(f"      size_t coord = tmp % (size_t){out_batch_dims}[axis];")
        ctx.lines.append(f"      tmp /= (size_t){out_batch_dims}[axis];")
        ctx.lines.append(f"      a_batch_off += coord * (size_t){a_batch_strides}[axis];")
        ctx.lines.append(f"      b_batch_off += coord * (size_t){b_batch_strides}[axis];")
        ctx.lines.append("    }")
    ctx.lines.append(f"    size_t out_batch_off = batch_i * (size_t){plan.m * plan.n};")
    ctx.lines.append(f"    for (size_t i = 0; i < {plan.m}; ++i) {{")
    ctx.lines.append(f"      for (size_t j = 0; j < {plan.n}; ++j) {{")
    ctx.lines.append("      int64_t acc = 0;")
    ctx.lines.append(f"      for (size_t t = 0; t < {plan.k}; ++t) {{")
    ctx.lines.append(f"        size_t ai = a_batch_off + i * (size_t){plan.k} + t;")
    ctx.lines.append(f"        size_t bi = b_batch_off + t * (size_t){plan.n} + j;")
    ctx.lines.append(f"        int64_t av = (int64_t){a}[ai] - {a_zp_sym};")
    ctx.lines.append(f"        int64_t bv = (int64_t){b}[bi] - {b_zp_sym};")
    ctx.lines.append("        acc += av * bv;")
    ctx.lines.append("      }")
    if out_dtype == "int32":
        ctx.lines.append("      if (acc < -2147483648LL) acc = -2147483648LL;")
        ctx.lines.append("      if (acc > 2147483647LL) acc = 2147483647LL;")
        ctx.lines.append(f"      {out}[out_batch_off + i * (size_t){plan.n} + j] = (int32_t)acc;")
    else:
        ctx.lines.append(f"      {out}[out_batch_off + i * (size_t){plan.n} + j] = (int64_t)acc;")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
