# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op
from ....operators.utils import tensor_size


@register_op("DequantizeLinear")
def emit_dequantize_linear(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 2:
        raise ValueError("DequantizeLinear expects at least 2 inputs.")
    if len(node.outputs) != 1:
        raise ValueError("DequantizeLinear expects exactly 1 output.")
    out_tensor = node.outputs[0]
    out = ctx.map_ptr(out_tensor)
    inp = ctx.map_ptr(node.inputs[0])
    in_shape = ctx.shape(node.inputs[0])
    size = tensor_size(in_shape)

    scale_name = node.inputs[1]
    if tensor_size(ctx.shape(scale_name)) != 1:
        raise ValueError("DequantizeLinear scale must be scalar.")
    scale_dtype = ctx.dtype(scale_name)
    if scale_dtype not in ("float32", "int8", "int16", "int32", "int64", "uint8"):
        raise ValueError("DequantizeLinear scale must be numeric scalar.")
    scale_ptr = ctx.map_ptr(scale_name)

    dtype = ctx.dtype(node.inputs[0])
    if dtype not in ("uint8", "int8", "int16"):
        raise ValueError("DequantizeLinear input must be uint8/int8/int16.")

    zero_sym = ctx.next_symbol("k2c_dq_zero")
    scale_sym = ctx.next_symbol("k2c_dq_scale")
    ctx.lines.append(f"  float {scale_sym} = (float)({scale_ptr}[0]);")
    ctx.lines.append(f"  int {zero_sym} = 0;")
    if len(node.inputs) >= 3:
        zero_name = node.inputs[2]
        if tensor_size(ctx.shape(zero_name)) != 1:
            raise ValueError("DequantizeLinear supports scalar zero_point only.")
        zero_dtype = ctx.dtype(zero_name)
        if zero_dtype not in ("float32", "int8", "int16", "int32", "int64", "uint8"):
            raise ValueError("DequantizeLinear zero_point must be numeric scalar.")
        zero_ptr = ctx.map_ptr(zero_name)
        ctx.lines.append(f"  {zero_sym} = (int)({zero_ptr}[0]);")

    ctx.lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
    ctx.lines.append(f"    {out}[i] = ((float){inp}[i] - {zero_sym}) * {scale_sym};")
    ctx.lines.append("  }")

