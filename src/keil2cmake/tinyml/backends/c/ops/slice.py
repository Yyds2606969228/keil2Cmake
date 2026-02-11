# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op
from ....operators.utils import emit_op_slice, get_const_ints, normalize_axis


@register_op("Slice")
def emit_slice(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 3:
        raise ValueError("Slice expects at least 3 inputs.")
    out_tensor = node.outputs[0]
    out = ctx.map_ptr(out_tensor)
    data_name = node.inputs[0]
    inp = ctx.map_ptr(data_name)
    in_shape = ctx.shape(data_name)
    out_shape = ctx.shape(out_tensor)

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

    axes = list(range(len(in_shape))) if axes is None else [int(v) for v in axes]
    starts = [int(v) for v in starts]
    ends = [int(v) for v in ends]
    if steps is not None:
        steps = [int(v) for v in steps]
        if any(v != 1 for v in steps):
            raise ValueError("Slice steps != 1 is not supported.")

    if len(axes) != len(starts) or len(axes) != len(ends):
        raise ValueError("Slice axes/starts/ends length mismatch.")

    norm_starts = []
    for idx, axis in enumerate(axes):
        axis = normalize_axis(int(axis), len(in_shape))
        dim = in_shape[axis]
        s = starts[idx]
        e = ends[idx]
        if s < 0:
            s += dim
        if e < 0:
            e += dim
        if s < 0:
            s = 0
        if e > dim:
            e = dim
        if e < s:
            e = s
        norm_starts.append(s)
        if out_shape[axis] != (e - s):
            raise ValueError("Slice output shape mismatch.")

    emit_op_slice(ctx.lines, out, inp, in_shape, out_shape, norm_starts, axes)

