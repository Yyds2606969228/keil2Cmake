# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op


@register_op("SplitToSequence")
def emit_split_to_sequence(ctx: EmitContext, node: NodeInfo) -> None:
    raise ValueError(
        "SplitToSequence is not directly representable in the current tensor-only IR. "
        "Please lower sequence ops before codegen."
    )
