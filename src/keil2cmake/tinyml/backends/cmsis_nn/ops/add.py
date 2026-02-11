# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import tensor_size
from ...c.ops.add import emit_add as emit_add_c
from .registry import register_op


@register_op("Add")
def emit_add(ctx: EmitContext, node: NodeInfo) -> None:
    def _fallback(reason: str) -> None:
        ctx.backend_used = "c"
        ctx.fallback_reason = reason
        emit_add_c(ctx, node)

    if len(node.inputs) != 2:
        raise ValueError("Add expects 2 inputs.")
    out_tensor = node.outputs[0]
    out_dtype = ctx.dtype(out_tensor)
    a_name = node.inputs[0]
    b_name = node.inputs[1]
    if out_dtype not in ("int8", "int16"):
        _fallback("cmsis-nn add supports int8/int16 only")
        return
    if ctx.dtype(a_name) != out_dtype or ctx.dtype(b_name) != out_dtype:
        _fallback("cmsis-nn add requires matching input/output dtypes")
        return

    out_shape = ctx.shape(out_tensor)
    a_shape = ctx.shape(a_name)
    b_shape = ctx.shape(b_name)
    if out_shape != a_shape or out_shape != b_shape:
        _fallback("cmsis-nn add requires equal shapes")
        return

    try:
        scale_a, zp_a = ctx.qparams(a_name)
        scale_b, zp_b = ctx.qparams(b_name)
        scale_o, zp_o = ctx.qparams(out_tensor)
    except ValueError:
        _fallback("cmsis-nn add requires quantization parameters")
        return

    # Keep CMSIS-NN path strict and safe: only enable when three tensors share the same scale.
    if abs(scale_a - scale_b) > 1e-12 or abs(scale_a - scale_o) > 1e-12:
        _fallback("cmsis-nn add requires same quant scale for input/output")
        return

    out = ctx.map_ptr(out_tensor)
    a = ctx.map_ptr(a_name)
    b = ctx.map_ptr(b_name)
    status_var = ctx.next_symbol("k2c_cmsis_status")
    done_label = ctx.next_symbol("k2c_cmsis_done")
    if out_dtype == "int8":
        func = "arm_elementwise_add_s8"
        act_min = "-128"
        act_max = "127"
    else:
        func = "arm_elementwise_add_s16"
        act_min = "-32768"
        act_max = "32767"

    size = tensor_size(out_shape)
    ctx.lines.append("#ifdef K2C_USE_CMSIS_NN")
    ctx.lines.append(
        f"  arm_cmsis_nn_status {status_var} = {func}("
        f"{a}, {b}, {-zp_a}, 1, 0, {-zp_b}, 1, 0, 0, {out}, {zp_o}, 1, 0, {act_min}, {act_max}, {size});"
    )
    ctx.lines.append(f"  if ({status_var} == ARM_CMSIS_NN_SUCCESS) goto {done_label};")
    ctx.lines.append("#endif")

    emit_add_c(ctx, node)

    ctx.lines.append("#ifdef K2C_USE_CMSIS_NN")
    ctx.lines.append(f"{done_label}:")
    ctx.lines.append("#endif")
