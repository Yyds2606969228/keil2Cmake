# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .logical_common import emit_logical_binary
from .registry import register_op


@register_op("Xor")
def emit_xor(ctx: EmitContext, node: NodeInfo) -> None:
    emit_logical_binary(ctx, node, "Xor", "(av != bv)")
