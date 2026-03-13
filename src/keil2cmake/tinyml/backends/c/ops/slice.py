# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op
from ....operators.utils import get_const_ints, normalize_axis, tensor_size


def _row_major_strides(shape: list[int]) -> list[int]:
    out = [1] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        out[i] = acc
        acc *= int(shape[i])
    return out


@register_op("Slice")
def emit_slice(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 3:
        raise ValueError("Slice expects at least 3 inputs.")
    out_name = node.outputs[0]
    out = ctx.map_ptr(out_name)
    data_name = node.inputs[0]
    inp = ctx.map_ptr(data_name)
    in_shape = [int(v) for v in ctx.shape(data_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    rank = len(in_shape)
    if rank <= 0 or len(out_shape) != rank:
        raise ValueError("Slice rank mismatch.")

    starts = node.attrs.get("starts")
    ends = node.attrs.get("ends")
    axes = node.attrs.get("axes")
    steps = node.attrs.get("steps")
    if starts is None:
        starts = get_const_ints(ctx.model, node.inputs[1])
    if ends is None:
        ends = get_const_ints(ctx.model, node.inputs[2])
    if axes is None and len(node.inputs) >= 4:
        axes = get_const_ints(ctx.model, node.inputs[3])
    if steps is None and len(node.inputs) >= 5:
        steps = get_const_ints(ctx.model, node.inputs[4])

    axes = list(range(rank)) if axes is None else [int(v) for v in axes]
    starts = [int(v) for v in starts]
    ends = [int(v) for v in ends]
    steps = [1] * len(axes) if steps is None else [int(v) for v in steps]

    if len(axes) != len(starts) or len(axes) != len(ends) or len(axes) != len(steps):
        raise ValueError("Slice axes/starts/ends/steps length mismatch.")

    begin = [0] * rank
    stride = [1] * rank
    for idx, axis_v in enumerate(axes):
        axis = normalize_axis(int(axis_v), rank)
        dim = int(in_shape[axis])
        s = int(starts[idx])
        e = int(ends[idx])
        st = int(steps[idx])
        if st == 0:
            raise ValueError("Slice step must be non-zero.")

        if s < 0:
            s += dim
        if e < 0:
            e += dim

        if st > 0:
            if s < 0:
                s = 0
            if s > dim:
                s = dim
            if e < 0:
                e = 0
            if e > dim:
                e = dim
            span = e - s
            out_dim = 0 if span <= 0 else (span + st - 1) // st
        else:
            if s < -1:
                s = -1
            if s >= dim:
                s = dim - 1
            if e < -1:
                e = -1
            if e >= dim:
                e = dim - 1
            span = s - e
            abs_step = -st
            out_dim = 0 if span <= 0 else (span + abs_step - 1) // abs_step

        if out_dim <= 0 or out_shape[axis] != out_dim:
            raise ValueError("Slice output shape mismatch.")
        begin[axis] = s
        stride[axis] = st

    for axis in range(rank):
        if axis not in [normalize_axis(int(v), rank) for v in axes]:
            if out_shape[axis] != in_shape[axis]:
                raise ValueError("Slice output shape mismatch.")

    in_stride = _row_major_strides(in_shape)
    out_size = tensor_size(out_shape)
    out_dims_sym = ctx.next_symbol("k2c_slice_out_dims")
    in_stride_sym = ctx.next_symbol("k2c_slice_in_stride")
    begin_sym = ctx.next_symbol("k2c_slice_begin")
    step_sym = ctx.next_symbol("k2c_slice_step")
    out_dims_vals = ", ".join(str(v) for v in out_shape)
    in_stride_vals = ", ".join(str(v) for v in in_stride)
    begin_vals = ", ".join(str(v) for v in begin)
    step_vals = ", ".join(str(v) for v in stride)

    ctx.lines.append(f"  static const int32_t {out_dims_sym}[{rank}] = {{ {out_dims_vals} }};")
    ctx.lines.append(f"  static const int32_t {in_stride_sym}[{rank}] = {{ {in_stride_vals} }};")
    ctx.lines.append(f"  static const int32_t {begin_sym}[{rank}] = {{ {begin_vals} }};")
    ctx.lines.append(f"  static const int32_t {step_sym}[{rank}] = {{ {step_vals} }};")
    ctx.lines.append(f"  for (size_t out_i = 0; out_i < {out_size}; ++out_i) {{")
    ctx.lines.append("    size_t tmp = out_i;")
    ctx.lines.append("    int64_t in_idx = 0;")
    ctx.lines.append(f"    for (int axis = {rank - 1}; axis >= 0; --axis) {{")
    ctx.lines.append(f"      int64_t coord = (int64_t)(tmp % (size_t){out_dims_sym}[axis]);")
    ctx.lines.append(f"      tmp /= (size_t){out_dims_sym}[axis];")
    ctx.lines.append(
        f"      int64_t ic = (int64_t){begin_sym}[axis] + coord * (int64_t){step_sym}[axis];"
    )
    ctx.lines.append(f"      in_idx += ic * (int64_t){in_stride_sym}[axis];")
    ctx.lines.append("    }")
    ctx.lines.append(f"    {out}[out_i] = {inp}[in_idx];")
    ctx.lines.append("  }")
