# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op
from .reduce_common import emit_reduce


@register_op("ReduceMin")
def emit_reduce_min(ctx: EmitContext, node: NodeInfo) -> None:
    emit_reduce(ctx, node, "ReduceMin", "min")

