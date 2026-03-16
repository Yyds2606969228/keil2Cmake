# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op
from ....operators.utils import emit_op_copy, tensor_size


@register_op("Transpose")
def emit_transpose(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("Transpose expects 1 input.")
    out_tensor = node.outputs[0]
    out = ctx.map_ptr(out_tensor)
    in_name = node.inputs[0]
    inp = ctx.map_ptr(in_name)
    in_shape = ctx.shape(in_name)
    out_shape = ctx.shape(out_tensor)
    rank = len(in_shape)
    if rank != len(out_shape):
        raise ValueError("Transpose input/output rank mismatch.")
    perm = node.attrs.get("perm")
    if perm is None:
        perm = list(reversed(range(rank)))
    perm = [int(v) for v in perm]
    if len(perm) != rank:
        raise ValueError("Transpose perm length mismatch.")
    if sorted(perm) != list(range(rank)):
        raise ValueError("Transpose perm is invalid.")

    size = tensor_size(out_shape)
    if size <= 1:
        emit_op_copy(ctx.lines, out, inp, size)
        return

    in_strides: list[int] = []
    stride = 1
    for dim in reversed(in_shape):
        in_strides.append(stride)
        stride *= int(dim)
    in_strides = list(reversed(in_strides))

    out_dims_sym = ctx.next_symbol("k2c_transpose_out_dims")
    in_strides_sym = ctx.next_symbol("k2c_transpose_in_strides")
    perm_sym = ctx.next_symbol("k2c_transpose_perm")
    out_dims_vals = ", ".join(str(int(v)) for v in out_shape)
    in_strides_vals = ", ".join(str(int(v)) for v in in_strides)
    perm_vals = ", ".join(str(v) for v in perm)
    lines = ctx.lines
    lines.append(f"  static const int32_t {out_dims_sym}[{rank}] = {{ {out_dims_vals} }};")
    lines.append(f"  static const int32_t {in_strides_sym}[{rank}] = {{ {in_strides_vals} }};")
    lines.append(f"  static const int32_t {perm_sym}[{rank}] = {{ {perm_vals} }};")
    lines.append(f"  for (size_t out_i = 0; out_i < {size}; ++out_i) {{")
    lines.append("    size_t tmp = out_i;")
    lines.append("    size_t in_i = 0;")
    lines.append(f"    for (int axis = {rank - 1}; axis >= 0; --axis) {{")
    lines.append(f"      size_t coord = tmp % (size_t){out_dims_sym}[axis];")
    lines.append(f"      tmp /= (size_t){out_dims_sym}[axis];")
    lines.append(f"      in_i += coord * (size_t){in_strides_sym}[{perm_sym}[axis]];")
    lines.append("    }")
    lines.append(f"    {out}[out_i] = {inp}[in_i];")
    lines.append("  }")

