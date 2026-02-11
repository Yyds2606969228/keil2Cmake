# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import tensor_size
from .compare_common import _broadcast_strides


def emit_logical_binary(ctx: EmitContext, node: NodeInfo, op_name: str, expr: str) -> None:
    if len(node.inputs) != 2:
        raise ValueError(f"{op_name} expects 2 inputs.")
    a_name, b_name = node.inputs
    out_name = node.outputs[0]
    if ctx.dtype(a_name) != "bool" or ctx.dtype(b_name) != "bool" or ctx.dtype(out_name) != "bool":
        raise ValueError(f"{op_name} requires bool input/output.")

    out_shape = ctx.shape(out_name)
    a_shape = ctx.shape(a_name)
    b_shape = ctx.shape(b_name)
    a_strides = _broadcast_strides(a_shape, out_shape, op_name)
    b_strides = _broadcast_strides(b_shape, out_shape, op_name)
    rank = len(out_shape)
    out_size = tensor_size(out_shape)

    out = ctx.map_ptr(out_name)
    a = ctx.map_ptr(a_name)
    b = ctx.map_ptr(b_name)
    out_dims_name = ctx.next_symbol("k2c_logic_out_dims")
    a_strides_name = ctx.next_symbol("k2c_logic_a_strides")
    b_strides_name = ctx.next_symbol("k2c_logic_b_strides")
    out_dims_vals = ", ".join(str(int(v)) for v in out_shape)
    a_strides_vals = ", ".join(str(int(v)) for v in a_strides)
    b_strides_vals = ", ".join(str(int(v)) for v in b_strides)

    ctx.lines.append(f"  static const int32_t {out_dims_name}[{rank}] = {{ {out_dims_vals} }};")
    ctx.lines.append(f"  static const int32_t {a_strides_name}[{rank}] = {{ {a_strides_vals} }};")
    ctx.lines.append(f"  static const int32_t {b_strides_name}[{rank}] = {{ {b_strides_vals} }};")
    ctx.lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {{")
    ctx.lines.append("    size_t tmp = i;")
    ctx.lines.append("    size_t ai = 0;")
    ctx.lines.append("    size_t bi = 0;")
    ctx.lines.append(f"    for (int axis = {rank - 1}; axis >= 0; --axis) {{")
    ctx.lines.append(f"      size_t coord = tmp % (size_t){out_dims_name}[axis];")
    ctx.lines.append(f"      tmp /= (size_t){out_dims_name}[axis];")
    ctx.lines.append(f"      ai += coord * (size_t){a_strides_name}[axis];")
    ctx.lines.append(f"      bi += coord * (size_t){b_strides_name}[axis];")
    ctx.lines.append("    }")
    ctx.lines.append(f"    uint8_t av = ({a}[ai] != 0) ? 1u : 0u;")
    ctx.lines.append(f"    uint8_t bv = ({b}[bi] != 0) ? 1u : 0u;")
    ctx.lines.append(f"    {out}[i] = ({expr}) ? 1u : 0u;")
    ctx.lines.append("  }")
