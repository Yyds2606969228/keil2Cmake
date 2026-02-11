# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .pool_nd_common import emit_pool_nd
from .registry import register_op


@register_op("MaxPool")
def emit_maxpool(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("MaxPool expects 1 input.")
    emit_pool_nd(
        ctx,
        attrs=node.attrs,
        x_name=node.inputs[0],
        out_name=node.outputs[0],
        mode="max",
    )

