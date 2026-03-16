# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import tensor_size


def _broadcast_strides(in_shape: list[int], out_shape: list[int], op_name: str) -> list[int]:
    out_rank = len(out_shape)
    in_rank = len(in_shape)
    if in_rank > out_rank:
        raise ValueError(f"{op_name} broadcast rank mismatch.")
    aligned = [1] * (out_rank - in_rank) + list(in_shape)
    raw_strides = [0] * out_rank
    stride = 1
    for axis in range(out_rank - 1, -1, -1):
        dim = int(aligned[axis])
        if dim <= 0:
            raise ValueError(f"{op_name} requires known positive dimensions.")
        raw_strides[axis] = stride
        stride *= dim
    out: list[int] = []
    for axis, in_dim in enumerate(aligned):
        out_dim = int(out_shape[axis])
        if in_dim == out_dim:
            out.append(raw_strides[axis])
        elif in_dim == 1:
            out.append(0)
        else:
            raise ValueError(f"{op_name} input is not broadcast-compatible with output.")
    return out


def emit_compare(ctx: EmitContext, node: NodeInfo, op_name: str, op_symbol: str) -> None:
    if len(node.inputs) != 2:
        raise ValueError(f"{op_name} expects 2 inputs.")
    a_name, b_name = node.inputs
    out_name = node.outputs[0]

    out_dtype = ctx.dtype(out_name)
    if out_dtype != "bool":
        raise ValueError(f"{op_name} output dtype must be bool.")
    a_dtype = ctx.dtype(a_name)
    b_dtype = ctx.dtype(b_name)
    if a_dtype != b_dtype:
        raise ValueError(f"{op_name} requires both inputs to have same dtype.")
    if a_dtype not in ("float32", "int8", "int16", "int32", "int64", "bool"):
        raise ValueError(f"{op_name} input dtype is unsupported.")

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
    out_dims_name = ctx.next_symbol("k2c_cmp_out_dims")
    a_strides_name = ctx.next_symbol("k2c_cmp_a_strides")
    b_strides_name = ctx.next_symbol("k2c_cmp_b_strides")

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
    ctx.lines.append(f"    {out}[i] = ({a}[ai] {op_symbol} {b}[bi]) ? 1u : 0u;")
    ctx.lines.append("  }")
