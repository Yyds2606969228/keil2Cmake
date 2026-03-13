# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import tensor_size
from .registry import register_op


def _broadcast_strides(in_shape: list[int], out_shape: list[int]) -> list[int]:
    out_rank = len(out_shape)
    in_rank = len(in_shape)
    if in_rank > out_rank:
        raise ValueError("BitShift broadcast rank mismatch.")
    aligned = [1] * (out_rank - in_rank) + list(in_shape)
    raw_strides = [0] * out_rank
    stride = 1
    for axis in range(out_rank - 1, -1, -1):
        dim = int(aligned[axis])
        if dim <= 0:
            raise ValueError("BitShift requires known positive dimensions.")
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
            raise ValueError("BitShift input is not broadcast-compatible with output.")
    return out


def _dtype_bits(dtype: str) -> int:
    if dtype == "int8":
        return 8
    if dtype == "int16":
        return 16
    if dtype == "int32":
        return 32
    if dtype == "int64":
        return 64
    raise ValueError("BitShift supports int8/int16/int32/int64 only.")


def _signed_ctype(dtype: str) -> str:
    if dtype == "int8":
        return "int8_t"
    if dtype == "int16":
        return "int16_t"
    if dtype == "int32":
        return "int32_t"
    if dtype == "int64":
        return "int64_t"
    raise ValueError("BitShift dtype is unsupported.")


def _unsigned_ctype(dtype: str) -> str:
    if dtype == "int8":
        return "uint8_t"
    if dtype == "int16":
        return "uint16_t"
    if dtype == "int32":
        return "uint32_t"
    if dtype == "int64":
        return "uint64_t"
    raise ValueError("BitShift dtype is unsupported.")


@register_op("BitShift")
def emit_bitshift(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 2:
        raise ValueError("BitShift expects 2 inputs.")
    a_name, b_name = node.inputs
    out_name = node.outputs[0]
    out_dtype = ctx.dtype(out_name)
    if out_dtype not in ("int8", "int16", "int32", "int64"):
        raise ValueError("BitShift supports int8/int16/int32/int64 only.")
    if ctx.dtype(a_name) != out_dtype or ctx.dtype(b_name) != out_dtype:
        raise ValueError("BitShift input/output dtypes must match.")

    out_shape = ctx.shape(out_name)
    a_shape = ctx.shape(a_name)
    b_shape = ctx.shape(b_name)
    a_strides = _broadcast_strides(a_shape, out_shape)
    b_strides = _broadcast_strides(b_shape, out_shape)
    rank = len(out_shape)
    out_size = tensor_size(out_shape)
    direction_attr = node.attrs.get("direction", "RIGHT")
    if isinstance(direction_attr, bytes):
        direction = direction_attr.decode("utf-8", errors="ignore").upper()
    else:
        direction = str(direction_attr).upper()
    if direction not in ("LEFT", "RIGHT"):
        raise ValueError("BitShift direction must be LEFT or RIGHT.")

    bits = _dtype_bits(out_dtype)
    sctype = _signed_ctype(out_dtype)
    uctype = _unsigned_ctype(out_dtype)
    out_dims_name = ctx.next_symbol("k2c_bshift_out_dims")
    a_strides_name = ctx.next_symbol("k2c_bshift_a_strides")
    b_strides_name = ctx.next_symbol("k2c_bshift_b_strides")
    out_dims_vals = ", ".join(str(int(v)) for v in out_shape)
    a_stride_vals = ", ".join(str(int(v)) for v in a_strides)
    b_stride_vals = ", ".join(str(int(v)) for v in b_strides)

    a = ctx.map_ptr(a_name)
    b = ctx.map_ptr(b_name)
    out = ctx.map_ptr(out_name)
    ctx.lines.append(f"  static const int32_t {out_dims_name}[{rank}] = {{ {out_dims_vals} }};")
    ctx.lines.append(f"  static const int32_t {a_strides_name}[{rank}] = {{ {a_stride_vals} }};")
    ctx.lines.append(f"  static const int32_t {b_strides_name}[{rank}] = {{ {b_stride_vals} }};")
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
    ctx.lines.append(f"    int64_t sh = (int64_t){b}[bi];")
    ctx.lines.append("    if (sh < 0) sh = 0;")
    ctx.lines.append(f"    if (sh >= {bits}) sh = {bits - 1};")
    ctx.lines.append(f"    {uctype} uv = ({uctype}){a}[ai];")
    if direction == "LEFT":
        ctx.lines.append(f"    {uctype} rv = ({uctype})(uv << (uint32_t)sh);")
    else:
        ctx.lines.append(f"    {uctype} rv = ({uctype})(uv >> (uint32_t)sh);")
    ctx.lines.append(f"    {out}[i] = ({sctype})rv;")
    ctx.lines.append("  }")
