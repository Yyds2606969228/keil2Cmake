# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .logical_common import emit_logical_binary
from .registry import register_op


@register_op("Or")
def emit_or(ctx: EmitContext, node: NodeInfo) -> None:
    emit_logical_binary(ctx, node, "Or", "av || bv")
