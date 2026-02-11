# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import quantize_multiplier, tensor_size
from ...c.ops.mul import emit_mul as emit_mul_c
from .registry import register_op


@register_op("Mul")
def emit_mul(ctx: EmitContext, node: NodeInfo) -> None:
    def _fallback(reason: str) -> None:
        ctx.backend_used = "c"
        ctx.fallback_reason = reason
        emit_mul_c(ctx, node)

    if len(node.inputs) != 2:
        raise ValueError("Mul expects 2 inputs.")
    out_name = node.outputs[0]
    a_name, b_name = node.inputs
    out_dtype = ctx.dtype(out_name)
    if out_dtype not in ("int8", "int16"):
        _fallback("cmsis-nn mul supports int8/int16 only")
        return
    if ctx.dtype(a_name) != out_dtype or ctx.dtype(b_name) != out_dtype:
        _fallback("cmsis-nn mul requires matching input/output dtypes")
        return

    out_shape = ctx.shape(out_name)
    a_shape = ctx.shape(a_name)
    b_shape = ctx.shape(b_name)
    if out_shape != a_shape or out_shape != b_shape:
        _fallback("cmsis-nn mul requires equal shapes")
        return

    try:
        sa, za = ctx.qparams(a_name)
        sb, zb = ctx.qparams(b_name)
        so, zo = ctx.qparams(out_name)
    except ValueError:
        _fallback("cmsis-nn mul requires quantization parameters")
        return

    real_multiplier = (sa * sb) / so if so != 0.0 else 0.0
    out_mult, out_shift = quantize_multiplier(real_multiplier)
    size = tensor_size(out_shape)
    a = ctx.map_ptr(a_name)
    b = ctx.map_ptr(b_name)
    out = ctx.map_ptr(out_name)
    status_var = ctx.next_symbol("k2c_cmsis_status")
    done_label = ctx.next_symbol("k2c_cmsis_done")
    if out_dtype == "int8":
        func = "arm_elementwise_mul_s8"
        act_min, act_max = -128, 127
    else:
        func = "arm_elementwise_mul_s16"
        act_min, act_max = -32768, 32767

    ctx.lines.append("#ifdef K2C_USE_CMSIS_NN")
    ctx.lines.append(
        f"  arm_cmsis_nn_status {status_var} = {func}("
        f"{a}, {b}, {-za}, {-zb}, {out}, {zo}, {out_mult}, {out_shift}, {act_min}, {act_max}, {size});"
    )
    ctx.lines.append(f"  if ({status_var} == ARM_CMSIS_NN_SUCCESS) goto {done_label};")
    ctx.lines.append("#endif")

    emit_mul_c(ctx, node)

    ctx.lines.append("#ifdef K2C_USE_CMSIS_NN")
    ctx.lines.append(f"{done_label}:")
    ctx.lines.append("#endif")
