# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op
from .scatter_elements import emit_scatter_elements


@register_op("Scatter")
def emit_scatter(ctx: EmitContext, node: NodeInfo) -> None:
    # Compatibility subset: old Scatter op maps to ScatterElements semantics,
    # but keeps historical behavior without reduction.
    reduction = node.attrs.get("reduction", "none")
    if isinstance(reduction, bytes):
        reduction = reduction.decode("utf-8", errors="ignore")
    if str(reduction).strip().lower() != "none":
        raise ValueError("Scatter does not support reduction; use ScatterElements/ScatterND.")
    emit_scatter_elements(ctx, node)
