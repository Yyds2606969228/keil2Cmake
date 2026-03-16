# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import tensor_size
from .registry import register_op


@register_op("StringNormalizer")
def emit_string_normalizer(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise ValueError("StringNormalizer expects 1 input and 1 output.")
    in_name = node.inputs[0]
    out_name = node.outputs[0]
    in_dtype = ctx.dtype(in_name)
    out_dtype = ctx.dtype(out_name)
    if in_dtype != out_dtype:
        raise ValueError("StringNormalizer requires matching input/output dtype.")
    if in_dtype not in ("int8", "int16", "int32", "int64", "uint8", "float32", "bool"):
        raise ValueError(
            "StringNormalizer string tensors are not supported by current IR; "
            "only pre-tokenized numeric tensors are supported."
        )
    in_shape = [int(v) for v in ctx.shape(in_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    if in_shape != out_shape:
        raise ValueError("StringNormalizer output shape must match input shape.")
    size = tensor_size(out_shape)
    x = ctx.map_ptr(in_name)
    y = ctx.map_ptr(out_name)
    ctx.lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{ {y}[i] = {x}[i]; }}")
