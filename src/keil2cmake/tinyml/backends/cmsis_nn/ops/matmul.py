# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import quantize_multiplier
from ...c.ops.matmul import emit_matmul as emit_matmul_c
from .registry import register_op


@register_op("MatMul")
def emit_matmul(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 2:
        raise ValueError("MatMul expects 2 inputs.")
    out_tensor = node.outputs[0]
    out_dtype = ctx.dtype(out_tensor)
    a_name, b_name = node.inputs

    if out_dtype != "int8":
        ctx.backend_used = "c"
        emit_matmul_c(ctx, node)
        return
    if ctx.dtype(a_name) != "int8" or ctx.dtype(b_name) != "int8":
        ctx.backend_used = "c"
        emit_matmul_c(ctx, node)
        return

    a_shape = ctx.shape(a_name)
    b_shape = ctx.shape(b_name)
    if len(a_shape) != 2 or len(b_shape) != 2:
        ctx.backend_used = "c"
        emit_matmul_c(ctx, node)
        return
    m, k1 = a_shape
    k2, n = b_shape
    if k1 != k2:
        ctx.backend_used = "c"
        emit_matmul_c(ctx, node)
        return

    bias = ctx.cmsis_bias(node)
    if not bias:
        ctx.backend_used = "c"
        emit_matmul_c(ctx, node)
        return

    a = ctx.map_ptr(a_name)
    b = ctx.map_ptr(b_name)
    out = ctx.map_ptr(out_tensor)
    sa, za = ctx.qparams(a_name)
    sb, zb = ctx.qparams(b_name)
    so, zo = ctx.qparams(out_tensor)
    real_multiplier = (sa * sb) / so if so != 0.0 else 0.0
    multiplier, shift = quantize_multiplier(real_multiplier)
    t_weight = ctx.cmsis_weight_t(b_name)
    ksum = ctx.cmsis_kernel_sum(b_name)
    use_runtime = False
    if not t_weight or not ksum:
        t_weight = ctx.next_symbol("k2c_cmsis_wt")
        ksum = ctx.next_symbol("k2c_cmsis_ks")
        use_runtime = True
    status_var = ctx.next_symbol("k2c_cmsis_status")
    done_label = ctx.next_symbol("k2c_cmsis_done")

    ctx.lines.append("#ifdef K2C_USE_CMSIS_NN")
    if use_runtime:
        ctx.lines.append(f"  static int8_t {t_weight}[{k1} * {n}];")
        ctx.lines.append(f"  static int32_t {ksum}[{n}];")
        ctx.lines.append(f"  for (size_t j = 0; j < {n}; ++j) {{")
        ctx.lines.append("    int32_t acc = 0;")
        ctx.lines.append(f"    for (size_t t = 0; t < {k1}; ++t) {{")
        ctx.lines.append(f"      int8_t v = {b}[t * {n} + j];")
        ctx.lines.append(f"      {t_weight}[j * {k1} + t] = v;")
        ctx.lines.append("      acc += v;")
        ctx.lines.append("    }")
        ctx.lines.append(f"    {ksum}[j] = acc;")
        ctx.lines.append("  }")
    ctx.lines.append(f"  arm_cmsis_nn_status {status_var} = ARM_CMSIS_NN_SUCCESS;")
    ctx.lines.append(f"  for (size_t i = 0; i < {m}; ++i) {{")
    ctx.lines.append(f"    const int8_t* lhs = {a} + i * {k1};")
    ctx.lines.append(f"    int8_t* dst = {out} + i * {n};")
    ctx.lines.append(
        f"    {status_var} = arm_nn_vec_mat_mult_t_s8("
        f"lhs, {t_weight}, {ksum}, {bias}, dst, {-za}, {zo}, {multiplier}, {shift}, "
        f"{k1}, {n}, -128, 127, 1, {-zb});"
    )
    ctx.lines.append(f"    if ({status_var} != ARM_CMSIS_NN_SUCCESS) break;")
    ctx.lines.append("  }")
    ctx.lines.append(f"  if ({status_var} == ARM_CMSIS_NN_SUCCESS) goto {done_label};")
    ctx.lines.append("#endif")

    emit_matmul_c(ctx, node)

    ctx.lines.append("#ifdef K2C_USE_CMSIS_NN")
    ctx.lines.append(f"{done_label}:")
    ctx.lines.append("#endif")
