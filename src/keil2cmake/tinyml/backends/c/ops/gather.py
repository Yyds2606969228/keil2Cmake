# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import normalize_axis, product, tensor_size
from .registry import register_op


@register_op("Gather")
def emit_gather(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 2:
        raise ValueError("Gather expects 2 inputs.")
    data_name = node.inputs[0]
    idx_name = node.inputs[1]
    out_name = node.outputs[0]
    if ctx.dtype(data_name) != ctx.dtype(out_name):
        raise ValueError("Gather output dtype must match data dtype.")
    if ctx.dtype(idx_name) not in ("int8", "int16", "int32", "int64"):
        raise ValueError("Gather indices dtype must be integer.")

    data_shape = [int(v) for v in ctx.shape(data_name)]
    idx_shape = [int(v) for v in ctx.shape(idx_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    if len(data_shape) == 0:
        raise ValueError("Gather does not support scalar input.")

    axis = normalize_axis(int(node.attrs.get("axis", 0)), len(data_shape))
    axis_dim = int(data_shape[axis])
    expected_shape = list(data_shape[:axis]) + list(idx_shape) + list(data_shape[axis + 1 :])
    if expected_shape != out_shape:
        raise ValueError("Gather output shape mismatch.")

    outer = product(data_shape[:axis]) if axis > 0 else 1
    inner = product(data_shape[axis + 1 :]) if axis + 1 < len(data_shape) else 1
    idx_size = tensor_size(idx_shape) if idx_shape else 1
    expected_out = outer * idx_size * inner
    if tensor_size(out_shape) != expected_out:
        raise ValueError("Gather output size mismatch.")

    data = ctx.map_ptr(data_name)
    idx = ctx.map_ptr(idx_name)
    out = ctx.map_ptr(out_name)
    ctx.lines.append(f"  for (size_t outer_i = 0; outer_i < {outer}; ++outer_i) {{")
    ctx.lines.append(f"    for (size_t k_i = 0; k_i < {idx_size}; ++k_i) {{")
    ctx.lines.append(f"      int64_t idx_v = (int64_t){idx}[k_i];")
    ctx.lines.append(f"      if (idx_v < 0) idx_v += (int64_t){axis_dim};")
    ctx.lines.append(f"      if (idx_v < 0 || idx_v >= (int64_t){axis_dim}) idx_v = 0;")
    ctx.lines.append(f"      size_t src_base = (outer_i * {axis_dim} + (size_t)idx_v) * {inner};")
    ctx.lines.append(f"      size_t dst_base = (outer_i * {idx_size} + k_i) * {inner};")
    ctx.lines.append(f"      for (size_t inner_i = 0; inner_i < {inner}; ++inner_i) {{")
    ctx.lines.append(f"        {out}[dst_base + inner_i] = {data}[src_base + inner_i];")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
