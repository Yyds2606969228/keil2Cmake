# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import normalize_axis, tensor_size
from .registry import register_op


def _strides(shape: list[int]) -> list[int]:
    out = [1] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        out[i] = acc
        acc *= int(shape[i])
    return out


@register_op("GatherElements")
def emit_gather_elements(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 2:
        raise ValueError("GatherElements expects 2 inputs.")
    data_name = node.inputs[0]
    idx_name = node.inputs[1]
    out_name = node.outputs[0]
    data_dtype = ctx.dtype(data_name)
    out_dtype = ctx.dtype(out_name)
    if data_dtype != out_dtype:
        raise ValueError("GatherElements output dtype must match data dtype.")
    if ctx.dtype(idx_name) not in ("int8", "int16", "int32", "int64"):
        raise ValueError("GatherElements indices dtype must be integer.")

    data_shape = [int(v) for v in ctx.shape(data_name)]
    idx_shape = [int(v) for v in ctx.shape(idx_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    if idx_shape != out_shape:
        raise ValueError("GatherElements output shape must equal indices shape.")
    rank = len(data_shape)
    if rank <= 0 or rank != len(idx_shape):
        raise ValueError("GatherElements requires data/indices with same rank >= 1.")

    axis = normalize_axis(int(node.attrs.get("axis", 0)), rank)
    for dim_i in range(rank):
        if dim_i == axis:
            continue
        if idx_shape[dim_i] > data_shape[dim_i]:
            raise ValueError("GatherElements indices shape exceeds data shape on non-axis dim.")
    axis_dim = int(data_shape[axis])
    if axis_dim <= 0:
        raise ValueError("GatherElements axis dimension must be positive.")

    data = ctx.map_ptr(data_name)
    idx = ctx.map_ptr(idx_name)
    out = ctx.map_ptr(out_name)
    out_size = tensor_size(out_shape)

    in_shape_sym = ctx.next_symbol("k2c_ge_in_shape")
    in_stride_sym = ctx.next_symbol("k2c_ge_in_stride")
    out_stride_sym = ctx.next_symbol("k2c_ge_out_stride")
    in_shape_vals = ", ".join(str(v) for v in data_shape)
    in_stride_vals = ", ".join(str(v) for v in _strides(data_shape))
    out_stride_vals = ", ".join(str(v) for v in _strides(out_shape))

    ctx.lines.append(f"  static const int32_t {in_shape_sym}[{rank}] = {{ {in_shape_vals} }};")
    ctx.lines.append(f"  static const int32_t {in_stride_sym}[{rank}] = {{ {in_stride_vals} }};")
    ctx.lines.append(f"  static const int32_t {out_stride_sym}[{rank}] = {{ {out_stride_vals} }};")
    ctx.lines.append(f"  for (size_t linear_i = 0; linear_i < {out_size}; ++linear_i) {{")
    ctx.lines.append("    size_t rem = linear_i;")
    ctx.lines.append("    int32_t src_index = 0;")
    ctx.lines.append(f"    for (size_t dim_i = 0; dim_i < {rank}; ++dim_i) {{")
    ctx.lines.append(f"      int32_t coord = (int32_t)(rem / (size_t){out_stride_sym}[dim_i]);")
    ctx.lines.append(f"      rem = rem % (size_t){out_stride_sym}[dim_i];")
    ctx.lines.append("      int32_t data_coord = coord;")
    ctx.lines.append(f"      if ((int32_t)dim_i == {axis}) {{")
    ctx.lines.append(f"        data_coord = (int32_t){idx}[linear_i];")
    ctx.lines.append(f"        if (data_coord < 0) data_coord += {in_shape_sym}[dim_i];")
    ctx.lines.append(f"        if (data_coord < 0 || data_coord >= {in_shape_sym}[dim_i]) data_coord = 0;")
    ctx.lines.append("      }")
    ctx.lines.append(f"      src_index += data_coord * {in_stride_sym}[dim_i];")
    ctx.lines.append("    }")
    ctx.lines.append(f"    {out}[linear_i] = {data}[src_index];")
    ctx.lines.append("  }")
