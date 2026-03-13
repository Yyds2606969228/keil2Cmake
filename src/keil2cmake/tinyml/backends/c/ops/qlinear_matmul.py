# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import tensor_size
from .registry import register_op
from .matmul_common import build_matmul_batch_plan


def _scalar_int_expr(ctx: EmitContext, name: str) -> str:
    shape = [int(v) for v in ctx.shape(name)]
    size = 1 if len(shape) == 0 else tensor_size(shape)
    if size != 1:
        raise ValueError("QLinearMatMul integer parameter must be scalar.")
    ptr = ctx.map_ptr(name)
    return f"((int64_t){ptr}[0])"


def _scalar_float_expr(ctx: EmitContext, name: str) -> str:
    shape = [int(v) for v in ctx.shape(name)]
    size = 1 if len(shape) == 0 else tensor_size(shape)
    if size != 1:
        raise ValueError("QLinearMatMul scale parameter must be scalar.")
    ptr = ctx.map_ptr(name)
    return f"((float){ptr}[0])"


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

    a_shape = [int(v) for v in ctx.shape(a_name)]
    b_shape = [int(v) for v in ctx.shape(b_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    plan = build_matmul_batch_plan("QLinearMatMul", a_shape, b_shape)
    expect_out_shape = [*plan.batch_shape, plan.m, plan.n]
    if out_shape != expect_out_shape:
        raise ValueError("QLinearMatMul output shape mismatch.")

    a_scale_expr = _scalar_float_expr(ctx, node.inputs[1])
    a_zero_expr = _scalar_int_expr(ctx, node.inputs[2])
    b_scale_expr = _scalar_float_expr(ctx, node.inputs[4])
    b_zero_expr = _scalar_int_expr(ctx, node.inputs[5])
    y_scale_expr = _scalar_float_expr(ctx, node.inputs[6])
    y_zero_expr = _scalar_int_expr(ctx, node.inputs[7])

    if out_dtype == "int8":
        qmin, qmax, ctype = -128, 127, "int8_t"
    else:
        qmin, qmax, ctype = -32768, 32767, "int16_t"

    a = ctx.map_ptr(a_name)
    b = ctx.map_ptr(b_name)
    out = ctx.map_ptr(out_name)
    a_scale_sym = ctx.next_symbol("k2c_qmm_a_scale")
    b_scale_sym = ctx.next_symbol("k2c_qmm_b_scale")
    y_scale_sym = ctx.next_symbol("k2c_qmm_y_scale")
    a_zero_sym = ctx.next_symbol("k2c_qmm_a_zero")
    b_zero_sym = ctx.next_symbol("k2c_qmm_b_zero")
    y_zero_sym = ctx.next_symbol("k2c_qmm_y_zero")
    mul_scale_sym = ctx.next_symbol("k2c_qmm_mul_scale")
    ctx.lines.append(f"  float {a_scale_sym} = {a_scale_expr};")
    ctx.lines.append(f"  float {b_scale_sym} = {b_scale_expr};")
    ctx.lines.append(f"  float {y_scale_sym} = {y_scale_expr};")
    ctx.lines.append(f"  int64_t {a_zero_sym} = {a_zero_expr};")
    ctx.lines.append(f"  int64_t {b_zero_sym} = {b_zero_expr};")
    ctx.lines.append(f"  int64_t {y_zero_sym} = {y_zero_expr};")
    ctx.lines.append(f"  if ({y_scale_sym} == 0.0f) {{")
    ctx.lines.append(f"    {y_scale_sym} = 1.0f;")
    ctx.lines.append("  }")
    ctx.lines.append(f"  float {mul_scale_sym} = {a_scale_sym} * {b_scale_sym};")

    batch_rank = len(plan.batch_shape)
    out_batch_dims = ctx.next_symbol("k2c_qmm_batch_dims")
    a_batch_strides = ctx.next_symbol("k2c_qmm_a_batch_strides")
    b_batch_strides = ctx.next_symbol("k2c_qmm_b_batch_strides")
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
    ctx.lines.append("        int64_t acc = 0;")
    ctx.lines.append(f"        for (size_t t = 0; t < {plan.k}; ++t) {{")
    ctx.lines.append(f"          size_t ai = a_batch_off + i * (size_t){plan.k} + t;")
    ctx.lines.append(f"          size_t bi = b_batch_off + t * (size_t){plan.n} + j;")
    ctx.lines.append(f"          int64_t av = (int64_t){a}[ai] - {a_zero_sym};")
    ctx.lines.append(f"          int64_t bv = (int64_t){b}[bi] - {b_zero_sym};")
    ctx.lines.append("          acc += av * bv;")
    ctx.lines.append("        }")
    ctx.lines.append(f"        float real_v = (float)acc * {mul_scale_sym};")
    ctx.lines.append(f"        int q = (int)roundf(real_v / {y_scale_sym}) + (int){y_zero_sym};")
    ctx.lines.append(f"        if (q < {qmin}) q = {qmin};")
    ctx.lines.append(f"        if (q > {qmax}) q = {qmax};")
    ctx.lines.append(f"        {out}[out_batch_off + i * (size_t){plan.n} + j] = ({ctype})q;")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
