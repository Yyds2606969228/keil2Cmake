# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op


@register_op("Loop")
def emit_loop(ctx: EmitContext, node: NodeInfo) -> None:
    raise ValueError(
        "Loop requires subgraph execution, which is not supported in the current C backend. "
        "Please unroll/fold Loop before codegen."
    )
