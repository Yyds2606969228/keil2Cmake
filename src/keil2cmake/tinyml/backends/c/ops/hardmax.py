# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import normalize_axis, product
from .registry import register_op


def _cast_type(dtype: str) -> str:
    if dtype == "float32":
        return "float"
    if dtype == "bool":
        return "uint8_t"
    if dtype == "int8":
        return "int8_t"
    if dtype == "int16":
        return "int16_t"
    if dtype == "int32":
        return "int32_t"
    if dtype == "int64":
        return "int64_t"
    raise ValueError("Hardmax output dtype is unsupported.")


@register_op("Hardmax")
def emit_hardmax(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("Hardmax expects 1 input.")
    in_name = node.inputs[0]
    out_name = node.outputs[0]
    in_dtype = ctx.dtype(in_name)
    out_dtype = ctx.dtype(out_name)
    if in_dtype != out_dtype:
        raise ValueError("Hardmax output dtype must match input dtype.")
    if in_dtype not in ("float32", "int8", "int16", "int32", "int64", "bool"):
        raise ValueError("Hardmax input dtype is unsupported.")

    in_shape = ctx.shape(in_name)
    out_shape = ctx.shape(out_name)
    if in_shape != out_shape:
        raise ValueError("Hardmax output shape mismatch.")
    rank = len(in_shape)
    if rank <= 0:
        raise ValueError("Hardmax expects rank >= 1.")

    axis = normalize_axis(int(node.attrs.get("axis", 1)), rank)
    axis_dim = int(in_shape[axis])
    if axis_dim <= 0:
        raise ValueError("Hardmax axis dimension must be positive.")
    outer = product(in_shape[:axis]) if axis > 0 else 1
    inner = product(in_shape[axis + 1 :]) if axis + 1 < rank else 1

    inp = ctx.map_ptr(in_name)
    out = ctx.map_ptr(out_name)
    cast_t = _cast_type(out_dtype)
    if out_dtype in ("int8", "int16"):
        so, zo = ctx.qparams(out_name)
        qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
        one_q = int(round(1.0 / so) + zo)
        if one_q < qmin:
            one_q = qmin
        if one_q > qmax:
            one_q = qmax
        zero_q = zo
        if zero_q < qmin:
            zero_q = qmin
        if zero_q > qmax:
            zero_q = qmax
        one = str(one_q)
        zero = str(zero_q)
    else:
        one = "1.0f" if out_dtype == "float32" else "1"
        zero = "0.0f" if out_dtype == "float32" else "0"

    ctx.lines.append(f"  for (size_t outer_i = 0; outer_i < {outer}; ++outer_i) {{")
    ctx.lines.append(f"    for (size_t inner_i = 0; inner_i < {inner}; ++inner_i) {{")
    ctx.lines.append(
        f"      float best_v = (float){inp}[(outer_i * {axis_dim}) * {inner} + inner_i];"
    )
    ctx.lines.append("      size_t best_idx = 0;")
    ctx.lines.append(f"      for (size_t axis_i = 1; axis_i < {axis_dim}; ++axis_i) {{")
    ctx.lines.append(
        f"        float v = (float){inp}[(outer_i * {axis_dim} + axis_i) * {inner} + inner_i];"
    )
    ctx.lines.append("        if (v > best_v) {")
    ctx.lines.append("          best_v = v;")
    ctx.lines.append("          best_idx = axis_i;")
    ctx.lines.append("        }")
    ctx.lines.append("      }")
    ctx.lines.append(f"      for (size_t axis_i = 0; axis_i < {axis_dim}; ++axis_i) {{")
    ctx.lines.append(f"        size_t idx = (outer_i * {axis_dim} + axis_i) * {inner} + inner_i;")
    ctx.lines.append(
        f"        {out}[idx] = ({cast_t})((axis_i == best_idx) ? {one} : {zero});"
    )
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
