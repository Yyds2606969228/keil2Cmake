# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import tensor_size
from ...c.ops.relu import emit_relu as emit_relu_c
from .registry import register_op


@register_op("Relu")
def emit_relu(ctx: EmitContext, node: NodeInfo) -> None:
    def _fallback(reason: str) -> None:
        ctx.backend_used = "c"
        ctx.fallback_reason = reason
        emit_relu_c(ctx, node)

    if len(node.inputs) != 1:
        raise ValueError("Relu expects 1 input.")
    x_name = node.inputs[0]
    out_name = node.outputs[0]
    out_dtype = ctx.dtype(out_name)
    if out_dtype not in ("int8", "int16"):
        _fallback("cmsis-nn relu supports int8/int16 only")
        return
    if ctx.dtype(x_name) != out_dtype:
        _fallback("cmsis-nn relu requires matching input/output dtypes")
        return
    if ctx.shape(x_name) != ctx.shape(out_name):
        _fallback("cmsis-nn relu requires equal shapes")
        return

    try:
        sx, zx = ctx.qparams(x_name)
        so, zo = ctx.qparams(out_name)
    except ValueError:
        _fallback("cmsis-nn relu requires quantization parameters")
        return
    if abs(sx - so) > 1e-12 or zx != zo:
        _fallback("cmsis-nn relu requires same quant scale/zero for input/output")
        return

    size = tensor_size(ctx.shape(out_name))
    x = ctx.map_ptr(x_name)
    out = ctx.map_ptr(out_name)
    done_label = ctx.next_symbol("k2c_cmsis_done")
    func = "arm_relu_q7" if out_dtype == "int8" else "arm_relu_q15"

    ctx.lines.append("#ifdef K2C_USE_CMSIS_NN")
    ctx.lines.append(f"  if ({out} != {x}) {{")
    ctx.lines.append(f"    for (size_t i = 0; i < {size}; ++i) {out}[i] = {x}[i];")
    ctx.lines.append("  }")
    ctx.lines.append(f"  {func}({out}, {size});")
    ctx.lines.append(f"  goto {done_label};")
    ctx.lines.append("#endif")

    emit_relu_c(ctx, node)

    ctx.lines.append("#ifdef K2C_USE_CMSIS_NN")
    ctx.lines.append(f"{done_label}:")
    ctx.lines.append("#endif")
