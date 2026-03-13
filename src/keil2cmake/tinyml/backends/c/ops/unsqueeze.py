# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op
from ....operators.utils import emit_op_copy, tensor_size


@register_op("Unsqueeze")
def emit_unsqueeze(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 1:
        raise ValueError("Unsqueeze expects at least 1 input.")
    out_tensor = node.outputs[0]
    out = ctx.map_ptr(out_tensor)
    a = ctx.map_ptr(node.inputs[0])
    size = tensor_size(ctx.shape(out_tensor))
    emit_op_copy(ctx.lines, out, a, size)

