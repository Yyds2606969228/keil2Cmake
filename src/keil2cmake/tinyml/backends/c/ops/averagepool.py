# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .pool_nd_common import emit_pool_nd
from .registry import register_op


@register_op("AveragePool")
def emit_avgpool(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("AveragePool expects 1 input.")
    count_include_pad = int(node.attrs.get("count_include_pad", 0))
    emit_pool_nd(
        ctx,
        attrs=node.attrs,
        x_name=node.inputs[0],
        out_name=node.outputs[0],
        mode="avg",
        count_include_pad=count_include_pad,
    )

