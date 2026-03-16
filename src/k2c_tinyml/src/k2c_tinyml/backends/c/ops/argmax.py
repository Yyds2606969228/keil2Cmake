# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .arg_reduce_common import emit_arg_reduce
from .registry import register_op


@register_op("ArgMax")
def emit_argmax(ctx: EmitContext, node: NodeInfo) -> None:
    emit_arg_reduce(ctx, node, "ArgMax", "max")
