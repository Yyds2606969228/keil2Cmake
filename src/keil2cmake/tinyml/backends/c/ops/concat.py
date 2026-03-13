# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op
from ....operators.utils import emit_op_concat, normalize_axis


@register_op("Concat")
def emit_concat(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 2:
        raise ValueError("Concat expects at least 2 inputs.")
    out_tensor = node.outputs[0]
    out = ctx.map_ptr(out_tensor)
    out_shape = ctx.shape(out_tensor)
    rank = len(out_shape)
    axis = normalize_axis(int(node.attrs.get("axis", 0)), rank)
    inputs = []
    shapes = []
    for name in node.inputs:
        inputs.append(ctx.map_ptr(name))
        shapes.append(ctx.shape(name))
    emit_op_concat(ctx.lines, out, inputs, shapes, axis, out_shape)

