# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .compare_common import emit_compare
from .registry import register_op


@register_op("Less")
def emit_less(ctx: EmitContext, node: NodeInfo) -> None:
    emit_compare(ctx, node, "Less", "<")
