# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import tensor_size
from .registry import register_op


@register_op("DynamicQuantizeLinear")
def emit_dynamic_quantize_linear(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("DynamicQuantizeLinear expects 1 input.")
    if len(node.outputs) != 3:
        raise ValueError("DynamicQuantizeLinear expects 3 outputs.")

    x_name = node.inputs[0]
    y_name, y_scale_name, y_zero_name = node.outputs

    if ctx.dtype(x_name) != "float32":
        raise ValueError("DynamicQuantizeLinear input must be float32.")
    if ctx.dtype(y_name) != "uint8":
        raise ValueError("DynamicQuantizeLinear output Y must be uint8.")
    if ctx.dtype(y_scale_name) != "float32":
        raise ValueError("DynamicQuantizeLinear output Y_Scale must be float32.")
    if ctx.dtype(y_zero_name) != "uint8":
        raise ValueError("DynamicQuantizeLinear output Y_ZeroPoint must be uint8.")

    if tensor_size(ctx.shape(y_scale_name)) != 1:
        raise ValueError("DynamicQuantizeLinear output Y_Scale must be scalar.")
    if tensor_size(ctx.shape(y_zero_name)) != 1:
        raise ValueError("DynamicQuantizeLinear output Y_ZeroPoint must be scalar.")

    x_shape = ctx.shape(x_name)
    y_shape = ctx.shape(y_name)
    if x_shape != y_shape:
        raise ValueError("DynamicQuantizeLinear Y shape must match input shape.")
    size = tensor_size(x_shape)
    if size <= 0:
        raise ValueError("DynamicQuantizeLinear input size must be positive.")

    x = ctx.map_ptr(x_name)
    y = ctx.map_ptr(y_name)
    y_scale = ctx.map_ptr(y_scale_name)
    y_zero = ctx.map_ptr(y_zero_name)
    min_var = ctx.next_symbol("k2c_dq_min")
    max_var = ctx.next_symbol("k2c_dq_max")
    scale_var = ctx.next_symbol("k2c_dq_scale")
    zero_var = ctx.next_symbol("k2c_dq_zero")

    ctx.lines.append(f"  float {min_var} = {x}[0];")
    ctx.lines.append(f"  float {max_var} = {x}[0];")
    ctx.lines.append(f"  for (size_t i = 1; i < {size}; ++i) {{")
    ctx.lines.append(f"    float v = {x}[i];")
    ctx.lines.append(f"    if (v < {min_var}) {min_var} = v;")
    ctx.lines.append(f"    if (v > {max_var}) {max_var} = v;")
    ctx.lines.append("  }")
    ctx.lines.append(f"  float {scale_var} = ({max_var} - {min_var}) / 255.0f;")
    ctx.lines.append(f"  if ({scale_var} <= 0.0f) {scale_var} = 1.0f;")
    ctx.lines.append(f"  int {zero_var} = (int)roundf(-{min_var} / {scale_var});")
    ctx.lines.append(f"  if ({zero_var} < 0) {zero_var} = 0;")
    ctx.lines.append(f"  if ({zero_var} > 255) {zero_var} = 255;")
    ctx.lines.append(f"  {y_scale}[0] = {scale_var};")
    ctx.lines.append(f"  {y_zero}[0] = (uint8_t){zero_var};")
    ctx.lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
    ctx.lines.append(f"    int q = (int)roundf({x}[i] / {scale_var}) + {zero_var};")
    ctx.lines.append("    if (q < 0) q = 0;")
    ctx.lines.append("    if (q > 255) q = 255;")
    ctx.lines.append(f"    {y}[i] = (uint8_t)q;")
    ctx.lines.append("  }")
