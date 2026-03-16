# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import tensor_size
from .registry import register_op


def _ctype(dtype: str) -> str:
    if dtype == "int8":
        return "int8_t"
    if dtype == "int16":
        return "int16_t"
    if dtype == "int32":
        return "int32_t"
    if dtype == "int64":
        return "int64_t"
    raise ValueError("Range output dtype is unsupported.")


@register_op("Range")
def emit_range(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 3:
        raise ValueError("Range expects 3 inputs: start, limit, delta.")

    start_name, limit_name, delta_name = node.inputs
    out_name = node.outputs[0]
    out_dtype = ctx.dtype(out_name)
    if out_dtype not in ("float32", "int8", "int16", "int32", "int64"):
        raise ValueError("Range output dtype is unsupported.")
    if ctx.dtype(start_name) != out_dtype or ctx.dtype(limit_name) != out_dtype or ctx.dtype(delta_name) != out_dtype:
        raise ValueError("Range input/output dtypes must match.")

    out_shape = ctx.shape(out_name)
    if len(out_shape) != 1:
        raise ValueError("Range output shape must be 1D.")
    size = tensor_size(out_shape)
    if size <= 0:
        raise ValueError("Range output size must be positive.")

    start = ctx.map_ptr(start_name)
    limit = ctx.map_ptr(limit_name)
    delta = ctx.map_ptr(delta_name)
    out = ctx.map_ptr(out_name)

    if out_dtype == "float32":
        ctx.lines.append(f"  float k2c_range_start = {start}[0];")
        ctx.lines.append(f"  float k2c_range_limit = {limit}[0];")
        ctx.lines.append(f"  float k2c_range_delta = {delta}[0];")
        ctx.lines.append("  (void)k2c_range_limit;")
        ctx.lines.append("  if (k2c_range_delta == 0.0f) {")
        ctx.lines.append(f"    for (size_t i = 0; i < {size}; ++i) {{")
        ctx.lines.append(f"      {out}[i] = 0.0f;")
        ctx.lines.append("    }")
        ctx.lines.append("  } else {")
        ctx.lines.append(f"    for (size_t i = 0; i < {size}; ++i) {{")
        ctx.lines.append(f"      {out}[i] = k2c_range_start + (float)i * k2c_range_delta;")
        ctx.lines.append("    }")
        ctx.lines.append("  }")
        return

    ctype = _ctype(out_dtype)
    ctx.lines.append(f"  int64_t k2c_range_start = (int64_t){start}[0];")
    ctx.lines.append(f"  int64_t k2c_range_limit = (int64_t){limit}[0];")
    ctx.lines.append(f"  int64_t k2c_range_delta = (int64_t){delta}[0];")
    ctx.lines.append("  (void)k2c_range_limit;")
    ctx.lines.append("  if (k2c_range_delta == 0) {")
    ctx.lines.append(f"    for (size_t i = 0; i < {size}; ++i) {{")
    ctx.lines.append(f"      {out}[i] = ({ctype})0;")
    ctx.lines.append("    }")
    ctx.lines.append("  } else {")
    ctx.lines.append(f"    for (size_t i = 0; i < {size}; ++i) {{")
    ctx.lines.append("      int64_t v = k2c_range_start + (int64_t)i * k2c_range_delta;")
    ctx.lines.append(f"      {out}[i] = ({ctype})v;")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
