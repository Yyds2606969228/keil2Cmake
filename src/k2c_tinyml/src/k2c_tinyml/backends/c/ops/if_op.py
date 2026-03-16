# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op


@register_op("If")
def emit_if(ctx: EmitContext, node: NodeInfo) -> None:
    raise ValueError(
        "If requires subgraph execution, which is not supported in the current C backend. "
        "Please fold/inline control flow before codegen."
    )
