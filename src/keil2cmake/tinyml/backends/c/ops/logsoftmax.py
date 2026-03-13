# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import normalize_axis, product
from .registry import register_op


@register_op("LogSoftmax")
def emit_logsoftmax(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("LogSoftmax expects 1 input.")
    in_name = node.inputs[0]
    out_name = node.outputs[0]
    in_dtype = ctx.dtype(in_name)
    out_dtype = ctx.dtype(out_name)
    quant_mode = in_dtype in ("int8", "int16") or out_dtype in ("int8", "int16")
    if quant_mode:
        if in_dtype not in ("int8", "int16") or out_dtype not in ("int8", "int16"):
            raise ValueError("LogSoftmax quantized path requires int8/int16 input/output.")
        sx, zx = ctx.qparams(in_name)
        so, zo = ctx.qparams(out_name)
        qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
        qctype = "int8_t" if out_dtype == "int8" else "int16_t"
    elif in_dtype != "float32" or out_dtype != "float32":
        raise ValueError("LogSoftmax supports float32 or quantized int8/int16.")

    in_shape = ctx.shape(in_name)
    out_shape = ctx.shape(out_name)
    if in_shape != out_shape:
        raise ValueError("LogSoftmax output shape mismatch.")
    rank = len(in_shape)
    if rank <= 0:
        raise ValueError("LogSoftmax expects rank >= 1.")

    axis = normalize_axis(int(node.attrs.get("axis", 1)), rank)
    axis_dim = int(in_shape[axis])
    if axis_dim <= 0:
        raise ValueError("LogSoftmax axis dimension must be positive.")
    outer = product(in_shape[:axis]) if axis > 0 else 1
    inner = product(in_shape[axis + 1 :]) if axis + 1 < rank else 1

    inp = ctx.map_ptr(in_name)
    out = ctx.map_ptr(out_name)
    ctx.lines.append(f"  for (size_t outer_i = 0; outer_i < {outer}; ++outer_i) {{")
    ctx.lines.append(f"    for (size_t inner_i = 0; inner_i < {inner}; ++inner_i) {{")
    if quant_mode:
        ctx.lines.append(
            f"      float max_v = ((float){inp}[(outer_i * {axis_dim}) * {inner} + inner_i] - {zx}) * {sx:.8f}f;"
        )
    else:
        ctx.lines.append(
            f"      float max_v = {inp}[(outer_i * {axis_dim}) * {inner} + inner_i];"
        )
    ctx.lines.append(f"      for (size_t axis_i = 1; axis_i < {axis_dim}; ++axis_i) {{")
    if quant_mode:
        ctx.lines.append(
            f"        float v = ((float){inp}[(outer_i * {axis_dim} + axis_i) * {inner} + inner_i] - {zx}) * {sx:.8f}f;"
        )
    else:
        ctx.lines.append(
            f"        float v = {inp}[(outer_i * {axis_dim} + axis_i) * {inner} + inner_i];"
        )
    ctx.lines.append("        if (v > max_v) max_v = v;")
    ctx.lines.append("      }")
    ctx.lines.append("      float sum_exp = 0.0f;")
    ctx.lines.append(f"      for (size_t axis_i = 0; axis_i < {axis_dim}; ++axis_i) {{")
    if quant_mode:
        ctx.lines.append(
            f"        float v = ((float){inp}[(outer_i * {axis_dim} + axis_i) * {inner} + inner_i] - {zx}) * {sx:.8f}f;"
        )
    else:
        ctx.lines.append(
            f"        float v = {inp}[(outer_i * {axis_dim} + axis_i) * {inner} + inner_i];"
        )
    ctx.lines.append("        sum_exp += expf(v - max_v);")
    ctx.lines.append("      }")
    ctx.lines.append("      float log_sum = logf(sum_exp);")
    ctx.lines.append(f"      for (size_t axis_i = 0; axis_i < {axis_dim}; ++axis_i) {{")
    ctx.lines.append(
        f"        size_t idx = (outer_i * {axis_dim} + axis_i) * {inner} + inner_i;"
    )
    if quant_mode:
        ctx.lines.append(f"        float v = ((float){inp}[idx] - {zx}) * {sx:.8f}f;")
        ctx.lines.append("        float r = v - max_v - log_sum;")
        ctx.lines.append(f"        int q = (int)roundf(r / {so:.8f}f) + {zo};")
        ctx.lines.append(f"        if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"        if (q > {qmax}) q = {qmax};")
        ctx.lines.append(f"        {out}[idx] = ({qctype})q;")
    else:
        ctx.lines.append(f"        {out}[idx] = {inp}[idx] - max_v - log_sum;")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
