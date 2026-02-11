# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import tensor_size
from ...c.ops.reshape import emit_reshape as emit_reshape_c
from .registry import register_op


@register_op("Reshape")
def emit_reshape(ctx: EmitContext, node: NodeInfo) -> None:
    def _fallback(reason: str) -> None:
        ctx.backend_used = "c"
        ctx.fallback_reason = reason
        emit_reshape_c(ctx, node)

    if len(node.inputs) < 1:
        raise ValueError("Reshape expects at least 1 input.")
    x_name = node.inputs[0]
    out_name = node.outputs[0]
    out_dtype = ctx.dtype(out_name)
    if out_dtype not in ("int8", "int16"):
        _fallback("cmsis-nn reshape supports int8/int16 only")
        return
    if ctx.dtype(x_name) != out_dtype:
        _fallback("cmsis-nn reshape requires matching input/output dtypes")
        return
    if tensor_size(ctx.shape(x_name)) != tensor_size(ctx.shape(out_name)):
        _fallback("cmsis-nn reshape requires same element count")
        return

    inp = ctx.map_ptr(x_name)
    out = ctx.map_ptr(out_name)
    size = tensor_size(ctx.shape(out_name))
    done_label = ctx.next_symbol("k2c_cmsis_done")
    func = "arm_copy_q7" if out_dtype == "int8" else "arm_copy_q15"
    ctx.lines.append("#ifdef K2C_USE_CMSIS_NN")
    ctx.lines.append(f"  {func}({inp}, {out}, {size});")
    ctx.lines.append(f"  goto {done_label};")
    ctx.lines.append("#endif")

    emit_reshape_c(ctx, node)

    ctx.lines.append("#ifdef K2C_USE_CMSIS_NN")
    ctx.lines.append(f"{done_label}:")
    ctx.lines.append("#endif")
