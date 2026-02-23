# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op


@register_op("SequenceLength")
def emit_sequence_length(ctx: EmitContext, node: NodeInfo) -> None:
    raise ValueError(
        "SequenceLength is not directly representable in the current tensor-only IR. "
        "Please lower sequence ops before codegen."
    )
