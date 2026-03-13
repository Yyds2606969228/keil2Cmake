# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op


@register_op("Scan")
def emit_scan(ctx: EmitContext, node: NodeInfo) -> None:
    raise ValueError(
        "Scan requires subgraph execution, which is not supported in the current C backend. "
        "Please lower Scan into primitive ops before codegen."
    )
