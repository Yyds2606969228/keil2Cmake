# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op
from ....operators.utils import emit_op_copy, tensor_size


@register_op("Identity")
def emit_identity(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("Identity expects 1 input.")
    out_tensor = node.outputs[0]
    out_dtype = ctx.dtype(out_tensor)
    out = ctx.map_ptr(out_tensor)
    a = ctx.map_ptr(node.inputs[0])
    size = tensor_size(ctx.shape(out_tensor))
    if out_dtype not in ("float32", "int8", "int16"):
        raise ValueError("Identity supports float32/int8/int16 only.")
    if ctx.dtype(node.inputs[0]) != out_dtype:
        raise ValueError("Identity requires matching dtypes.")
    emit_op_copy(ctx.lines, out, a, size)

