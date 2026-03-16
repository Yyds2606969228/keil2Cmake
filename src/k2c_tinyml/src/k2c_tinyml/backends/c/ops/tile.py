# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import get_const_ints, tensor_size
from .registry import register_op


def _strides(shape: list[int]) -> list[int]:
    out = [0] * len(shape)
    stride = 1
    for axis in range(len(shape) - 1, -1, -1):
        out[axis] = stride
        stride *= int(shape[axis])
    return out


@register_op("Tile")
def emit_tile(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 2:
        raise ValueError("Tile expects 2 inputs.")
    data_name = node.inputs[0]
    reps_name = node.inputs[1]
    out_name = node.outputs[0]
    out_dtype = ctx.dtype(out_name)
    if ctx.dtype(data_name) != out_dtype:
        raise ValueError("Tile output dtype must match input dtype.")
    if out_dtype in ("int8", "int16"):
        si, zi = ctx.qparams(data_name)
        qo = ctx.qparams_optional(out_name)
        if qo is not None:
            so, zo = qo
            if abs(si - so) > 1e-12 or zi != zo:
                raise ValueError("Quantized Tile requires same input/output qparams.")

    in_shape = ctx.shape(data_name)
    out_shape = ctx.shape(out_name)
    reps = [int(v) for v in get_const_ints(ctx.model, reps_name)]
    if len(reps) != len(in_shape):
        raise ValueError("Tile currently requires repeats rank equals input rank.")
    expected = [int(in_shape[i]) * int(reps[i]) for i in range(len(in_shape))]
    if expected != list(out_shape):
        raise ValueError("Tile output shape mismatch.")

    rank = len(out_shape)
    in_strides = _strides(in_shape)
    out_dims_name = ctx.next_symbol("k2c_tile_out_dims")
    in_dims_name = ctx.next_symbol("k2c_tile_in_dims")
    in_strides_name = ctx.next_symbol("k2c_tile_in_strides")
    out_size = tensor_size(out_shape)
    data = ctx.map_ptr(data_name)
    out = ctx.map_ptr(out_name)

    out_dims_vals = ", ".join(str(int(v)) for v in out_shape)
    in_dims_vals = ", ".join(str(int(v)) for v in in_shape)
    in_strides_vals = ", ".join(str(int(v)) for v in in_strides)
    ctx.lines.append(f"  static const int32_t {out_dims_name}[{rank}] = {{ {out_dims_vals} }};")
    ctx.lines.append(f"  static const int32_t {in_dims_name}[{rank}] = {{ {in_dims_vals} }};")
    ctx.lines.append(f"  static const int32_t {in_strides_name}[{rank}] = {{ {in_strides_vals} }};")
    ctx.lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {{")
    ctx.lines.append("    size_t tmp = i;")
    ctx.lines.append("    size_t src_idx = 0;")
    ctx.lines.append(f"    for (int axis = {rank - 1}; axis >= 0; --axis) {{")
    ctx.lines.append(f"      size_t coord = tmp % (size_t){out_dims_name}[axis];")
    ctx.lines.append(f"      tmp /= (size_t){out_dims_name}[axis];")
    ctx.lines.append(f"      size_t in_coord = coord % (size_t){in_dims_name}[axis];")
    ctx.lines.append(f"      src_idx += in_coord * (size_t){in_strides_name}[axis];")
    ctx.lines.append("    }")
    ctx.lines.append(f"    {out}[i] = {data}[src_idx];")
    ctx.lines.append("  }")
