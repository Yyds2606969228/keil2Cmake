# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import tensor_size
from .registry import register_op


def _broadcast_strides(in_shape: list[int], out_shape: list[int]) -> list[int]:
    out_rank = len(out_shape)
    in_rank = len(in_shape)
    if in_rank > out_rank:
        raise ValueError("Expand input rank exceeds output rank.")
    aligned = [1] * (out_rank - in_rank) + list(in_shape)
    raw_strides = [0] * out_rank
    stride = 1
    for axis in range(out_rank - 1, -1, -1):
        dim = int(aligned[axis])
        if dim <= 0:
            raise ValueError("Expand requires known positive dimensions.")
        raw_strides[axis] = stride
        stride *= dim
    strides: list[int] = []
    for axis, in_dim in enumerate(aligned):
        out_dim = int(out_shape[axis])
        if in_dim == out_dim:
            strides.append(raw_strides[axis])
        elif in_dim == 1:
            strides.append(0)
        else:
            raise ValueError("Expand shape is not broadcast-compatible.")
    return strides


@register_op("Expand")
def emit_expand(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 2:
        raise ValueError("Expand expects 2 inputs.")
    data_name = node.inputs[0]
    shape_name = node.inputs[1]
    out_name = node.outputs[0]
    out_dtype = ctx.dtype(out_name)
    if ctx.dtype(data_name) != out_dtype:
        raise ValueError("Expand output dtype must match input dtype.")
    if out_dtype in ("int8", "int16"):
        si, zi = ctx.qparams(data_name)
        qo = ctx.qparams_optional(out_name)
        if qo is not None:
            so, zo = qo
            if abs(si - so) > 1e-12 or zi != zo:
                raise ValueError("Quantized Expand requires same input/output qparams.")
    out_shape = ctx.shape(out_name)
    shape_dtype = ctx.dtype(shape_name)
    if shape_dtype not in ("int8", "int16", "int32", "int64"):
        raise ValueError("Expand shape dtype must be integer.")
    shape_shape = [int(v) for v in ctx.shape(shape_name)]
    if len(shape_shape) != 1 or int(shape_shape[0]) != len(out_shape):
        raise ValueError("Expand shape input must be 1D and match output rank.")

    in_shape = ctx.shape(data_name)
    in_strides = _broadcast_strides(in_shape, out_shape)
    out_dims_name = ctx.next_symbol("k2c_expand_out_dims")
    in_strides_name = ctx.next_symbol("k2c_expand_in_strides")
    out_size = tensor_size(out_shape)
    rank = len(out_shape)
    data = ctx.map_ptr(data_name)
    out = ctx.map_ptr(out_name)

    out_dims_vals = ", ".join(str(int(v)) for v in out_shape)
    in_stride_vals = ", ".join(str(int(v)) for v in in_strides)
    ctx.lines.append(f"  static const int32_t {out_dims_name}[{rank}] = {{ {out_dims_vals} }};")
    ctx.lines.append(f"  static const int32_t {in_strides_name}[{rank}] = {{ {in_stride_vals} }};")
    ctx.lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {{")
    ctx.lines.append("    size_t tmp = i;")
    ctx.lines.append("    size_t src_idx = 0;")
    ctx.lines.append(f"    for (int axis = {rank - 1}; axis >= 0; --axis) {{")
    ctx.lines.append(f"      size_t coord = tmp % (size_t){out_dims_name}[axis];")
    ctx.lines.append(f"      tmp /= (size_t){out_dims_name}[axis];")
    ctx.lines.append(f"      src_idx += coord * (size_t){in_strides_name}[axis];")
    ctx.lines.append("    }")
    ctx.lines.append(f"    {out}[i] = {data}[src_idx];")
    ctx.lines.append("  }")
