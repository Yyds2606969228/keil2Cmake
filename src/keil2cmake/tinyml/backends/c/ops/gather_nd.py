# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import tensor_size
from .registry import register_op


def _strides(shape: list[int]) -> list[int]:
    out = [1] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        out[i] = acc
        acc *= int(shape[i])
    return out


@register_op("GatherND")
def emit_gather_nd(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 2:
        raise ValueError("GatherND expects 2 inputs.")
    data_name, idx_name = node.inputs
    out_name = node.outputs[0]

    if ctx.dtype(data_name) != ctx.dtype(out_name):
        raise ValueError("GatherND output dtype must match data dtype.")
    if ctx.dtype(idx_name) not in ("int8", "int16", "int32", "int64"):
        raise ValueError("GatherND indices dtype must be integer.")

    data_shape = [int(v) for v in ctx.shape(data_name)]
    idx_shape = [int(v) for v in ctx.shape(idx_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    if len(data_shape) <= 0 or len(idx_shape) <= 0:
        raise ValueError("GatherND requires non-scalar data/indices.")

    batch_dims = int(node.attrs.get("batch_dims", 0))
    if batch_dims < 0:
        batch_dims += min(len(data_shape), len(idx_shape) - 1)
    if batch_dims < 0 or batch_dims >= len(data_shape) or batch_dims >= len(idx_shape):
        raise ValueError("GatherND batch_dims out of range.")

    k = int(idx_shape[-1])
    if k < 0 or k > (len(data_shape) - batch_dims):
        raise ValueError("GatherND indices last dim out of range.")

    expected = list(data_shape[:batch_dims]) + list(idx_shape[batch_dims:-1]) + list(data_shape[batch_dims + k :])
    if expected != out_shape:
        raise ValueError("GatherND output shape mismatch.")

    out_size = tensor_size(out_shape)
    data_rank = len(data_shape)
    idx_rank = len(idx_shape)
    out_rank = len(out_shape)
    idx_suffix_rank = idx_rank - batch_dims - 1
    tail_rank = data_rank - batch_dims - k

    data = ctx.map_ptr(data_name)
    idx = ctx.map_ptr(idx_name)
    out = ctx.map_ptr(out_name)

    data_shape_sym = ctx.next_symbol("k2c_gnd_data_shape")
    data_stride_sym = ctx.next_symbol("k2c_gnd_data_stride")
    idx_stride_sym = ctx.next_symbol("k2c_gnd_idx_stride")
    out_dims_sym = ctx.next_symbol("k2c_gnd_out_dims")
    data_shape_vals = ", ".join(str(v) for v in data_shape)
    data_stride_vals = ", ".join(str(v) for v in _strides(data_shape))
    idx_stride_vals = ", ".join(str(v) for v in _strides(idx_shape))
    out_dims_vals = ", ".join(str(v) for v in out_shape) if out_shape else "1"

    ctx.lines.append(f"  static const int32_t {data_shape_sym}[{data_rank}] = {{ {data_shape_vals} }};")
    ctx.lines.append(f"  static const int32_t {data_stride_sym}[{data_rank}] = {{ {data_stride_vals} }};")
    ctx.lines.append(f"  static const int32_t {idx_stride_sym}[{idx_rank}] = {{ {idx_stride_vals} }};")
    ctx.lines.append(f"  static const int32_t {out_dims_sym}[{max(1, out_rank)}] = {{ {out_dims_vals} }};")
    ctx.lines.append(f"  for (size_t out_i = 0; out_i < {out_size}; ++out_i) {{")
    if out_rank > 0:
        ctx.lines.append(f"    int64_t out_coord[{out_rank}];")
        ctx.lines.append("    size_t tmp = out_i;")
        ctx.lines.append(f"    for (int axis = {out_rank - 1}; axis >= 0; --axis) {{")
        ctx.lines.append(f"      out_coord[axis] = (int64_t)(tmp % (size_t){out_dims_sym}[axis]);")
        ctx.lines.append(f"      tmp /= (size_t){out_dims_sym}[axis];")
        ctx.lines.append("    }")
    ctx.lines.append("    int64_t data_idx = 0;")
    for axis in range(batch_dims):
        ctx.lines.append(f"    data_idx += out_coord[{axis}] * (int64_t){data_stride_sym}[{axis}];")
    ctx.lines.append("    int valid = 1;")
    ctx.lines.append(f"    for (int j = 0; j < {k}; ++j) {{")
    ctx.lines.append("      int64_t idx_pos = 0;")
    for axis in range(batch_dims):
        ctx.lines.append(f"      idx_pos += out_coord[{axis}] * (int64_t){idx_stride_sym}[{axis}];")
    for axis in range(idx_suffix_rank):
        out_axis = batch_dims + axis
        idx_axis = batch_dims + axis
        ctx.lines.append(
            f"      idx_pos += out_coord[{out_axis}] * (int64_t){idx_stride_sym}[{idx_axis}];"
        )
    ctx.lines.append(f"      idx_pos += (int64_t)j * (int64_t){idx_stride_sym}[{idx_rank - 1}];")
    ctx.lines.append(f"      int64_t v = (int64_t){idx}[idx_pos];")
    ctx.lines.append(f"      int64_t dim = (int64_t){data_shape_sym}[{batch_dims} + j];")
    ctx.lines.append("      if (v < 0) v += dim;")
    ctx.lines.append("      if (v < 0 || v >= dim) { valid = 0; break; }")
    ctx.lines.append(f"      data_idx += v * (int64_t){data_stride_sym}[{batch_dims} + j];")
    ctx.lines.append("    }")
    if tail_rank > 0:
        for axis in range(tail_rank):
            out_axis = batch_dims + idx_suffix_rank + axis
            data_axis = batch_dims + k + axis
            ctx.lines.append(
                f"    data_idx += out_coord[{out_axis}] * (int64_t){data_stride_sym}[{data_axis}];"
            )
    ctx.lines.append("    if (!valid) data_idx = 0;")
    ctx.lines.append(f"    {out}[out_i] = {data}[data_idx];")
    ctx.lines.append("  }")
