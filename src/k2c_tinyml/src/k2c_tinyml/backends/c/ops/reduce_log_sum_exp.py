# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op
from .reduce_common import emit_reduce


@register_op("ReduceLogSumExp")
def emit_reduce_log_sum_exp(ctx: EmitContext, node: NodeInfo) -> None:
    emit_reduce(ctx, node, "ReduceLogSumExp", "log_sum_exp")
