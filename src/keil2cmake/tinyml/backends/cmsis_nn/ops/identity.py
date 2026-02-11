# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import tensor_size
from ...c.ops.identity import emit_identity as emit_identity_c
from .registry import register_op


@register_op("Identity")
def emit_identity(ctx: EmitContext, node: NodeInfo) -> None:
    def _fallback(reason: str) -> None:
        ctx.backend_used = "c"
        ctx.fallback_reason = reason
        emit_identity_c(ctx, node)

    if len(node.inputs) != 1:
        raise ValueError("Identity expects 1 input.")
    x_name = node.inputs[0]
    out_name = node.outputs[0]
    out_dtype = ctx.dtype(out_name)
    if out_dtype not in ("int8", "int16"):
        _fallback("cmsis-nn identity supports int8/int16 only")
        return
    if ctx.dtype(x_name) != out_dtype:
        _fallback("cmsis-nn identity requires matching input/output dtypes")
        return
    if ctx.shape(x_name) != ctx.shape(out_name):
        _fallback("cmsis-nn identity requires equal shapes")
        return

    size = tensor_size(ctx.shape(out_name))
    inp = ctx.map_ptr(x_name)
    out = ctx.map_ptr(out_name)
    done_label = ctx.next_symbol("k2c_cmsis_done")
    func = "arm_copy_q7" if out_dtype == "int8" else "arm_copy_q15"
    ctx.lines.append("#ifdef K2C_USE_CMSIS_NN")
    ctx.lines.append(f"  {func}({inp}, {out}, {size});")
    ctx.lines.append(f"  goto {done_label};")
    ctx.lines.append("#endif")

    emit_identity_c(ctx, node)

    ctx.lines.append("#ifdef K2C_USE_CMSIS_NN")
    ctx.lines.append(f"{done_label}:")
    ctx.lines.append("#endif")
