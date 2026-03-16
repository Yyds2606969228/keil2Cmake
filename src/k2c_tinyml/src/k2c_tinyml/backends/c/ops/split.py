# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import normalize_axis, product
from .registry import register_op


@register_op("Split")
def emit_split(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 1:
        raise ValueError("Split expects at least 1 input.")
    if len(node.outputs) < 1:
        raise ValueError("Split expects at least 1 output.")

    data_name = node.inputs[0]
    in_shape = [int(v) for v in ctx.shape(data_name)]
    in_dtype = ctx.dtype(data_name)
    rank = len(in_shape)
    axis = normalize_axis(int(node.attrs.get("axis", 0)), rank)
    in_axis_dim = in_shape[axis]

    out_shapes = [list(ctx.shape(name)) for name in node.outputs]
    out_dims: list[int] = []
    for out_name, out_shape in zip(node.outputs, out_shapes):
        if ctx.dtype(out_name) != in_dtype:
            raise ValueError("Split requires matching input/output dtypes.")
        if len(out_shape) != rank:
            raise ValueError("Split output rank mismatch.")
        for i in range(rank):
            if i == axis:
                continue
            if int(out_shape[i]) != int(in_shape[i]):
                raise ValueError("Split output shape mismatch.")
        out_dim = int(out_shape[axis])
        if out_dim <= 0:
            raise ValueError("Split output axis dim must be positive.")
        out_dims.append(out_dim)
    if sum(out_dims) != in_axis_dim:
        raise ValueError("Split outputs do not cover full axis dimension.")

    outer = product(in_shape[:axis]) if axis > 0 else 1
    inner = product(in_shape[axis + 1 :]) if axis + 1 < rank else 1
    inp = ctx.map_ptr(data_name)
    out_ptrs = [ctx.map_ptr(name) for name in node.outputs]

    ctx.lines.append(f"  for (size_t outer_i = 0; outer_i < {outer}; ++outer_i) {{")
    ctx.lines.append(f"    size_t in_base = outer_i * {in_axis_dim} * {inner};")
    ctx.lines.append("    size_t in_off = in_base;")
    for idx, (out_ptr, out_dim) in enumerate(zip(out_ptrs, out_dims)):
        block = out_dim * inner
        ctx.lines.append(f"    size_t out_base_{idx} = outer_i * {out_dim} * {inner};")
        ctx.lines.append(f"    for (size_t i = 0; i < {block}; ++i) {{")
        ctx.lines.append(f"      {out_ptr}[out_base_{idx} + i] = {inp}[in_off + i];")
        ctx.lines.append("    }")
        ctx.lines.append(f"    in_off += {block};")
    ctx.lines.append("  }")
