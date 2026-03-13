# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import tensor_size
from .registry import register_op


@register_op("Size")
def emit_size(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("Size expects 1 input.")
    out_name = node.outputs[0]
    out_shape = ctx.shape(out_name)
    out_dtype = ctx.dtype(out_name)
    if out_dtype not in ("int64", "int32", "float32"):
        raise ValueError("Size output dtype must be int64/int32/float32.")
    if out_shape != []:
        raise ValueError("Size output shape must be scalar [].")
    total = tensor_size(ctx.shape(node.inputs[0]))
    out = ctx.map_ptr(out_name)
    if out_dtype == "int64":
        cast_t = "int64_t"
    elif out_dtype == "int32":
        cast_t = "int32_t"
    else:
        cast_t = "float"
    ctx.lines.append(f"  {out}[0] = ({cast_t}){total};")
