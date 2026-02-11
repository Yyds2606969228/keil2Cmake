# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op


@register_op("Shape")
def emit_shape(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("Shape expects 1 input.")
    out_name = node.outputs[0]
    in_shape = ctx.shape(node.inputs[0])
    out_shape = ctx.shape(out_name)
    out_dtype = ctx.dtype(out_name)
    if out_dtype not in ("int64", "int32", "float32"):
        raise ValueError("Shape output dtype must be int64/int32/float32.")
    if out_shape != [len(in_shape)]:
        raise ValueError("Shape output shape mismatch.")
    out = ctx.map_ptr(out_name)
    if out_dtype == "int64":
        cast_t = "int64_t"
    elif out_dtype == "int32":
        cast_t = "int32_t"
    else:
        cast_t = "float"
    for idx, dim in enumerate(in_shape):
        if int(dim) <= 0:
            raise ValueError("Shape requires known positive dimensions.")
        ctx.lines.append(f"  {out}[{idx}] = ({cast_t}){int(dim)};")
