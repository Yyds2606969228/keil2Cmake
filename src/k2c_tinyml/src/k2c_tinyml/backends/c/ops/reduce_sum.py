# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op
from .reduce_common import emit_reduce


@register_op("ReduceSum")
def emit_reduce_sum(ctx: EmitContext, node: NodeInfo) -> None:
    emit_reduce(ctx, node, "ReduceSum", "sum")

