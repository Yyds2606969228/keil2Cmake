# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op


def _strides(shape: list[int]) -> list[int]:
    out = [1] * len(shape)
    acc = 1
    for idx in range(len(shape) - 1, -1, -1):
        out[idx] = acc
        acc *= int(shape[idx])
    return out


def _is_nonzero_value(dtype: str, value: float) -> bool:
    if dtype == "bool":
        return int(value) != 0
    return float(value) != 0.0


@register_op("NonZero")
def emit_nonzero(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("NonZero expects 1 input.")
    in_name = node.inputs[0]
    out_name = node.outputs[0]
    out_dtype = ctx.dtype(out_name)
    if out_dtype not in ("int64", "int32"):
        raise ValueError("NonZero output dtype must be int64/int32.")

    in_tensor = ctx.model.tensors.get(in_name)
    if in_tensor is None or in_tensor.data is None:
        raise ValueError("NonZero currently supports constant input only.")
    in_shape = [int(v) for v in in_tensor.shape]
    if any(v <= 0 for v in in_shape):
        raise ValueError("NonZero requires known positive input shape.")
    rank = len(in_shape)
    if rank <= 0:
        raise ValueError("NonZero input rank must be >= 1.")

    out_shape = [int(v) for v in ctx.shape(out_name)]
    if len(out_shape) != 2 or out_shape[0] != rank:
        raise ValueError("NonZero output shape must be [rank, count].")
    count = out_shape[1]
    if count < 0:
        raise ValueError("NonZero output count must be non-negative.")

    strides = _strides(in_shape)
    coords: list[list[int]] = []
    for linear_i, raw in enumerate(in_tensor.data):
        if not _is_nonzero_value(in_tensor.dtype, raw):
            continue
        rem = linear_i
        one: list[int] = []
        for stride in strides:
            one.append(int(rem // stride))
            rem = int(rem % stride)
        coords.append(one)
    if len(coords) != count:
        raise ValueError("NonZero output shape count does not match constant input non-zero count.")

    out = ctx.map_ptr(out_name)
    ctype = "int64_t" if out_dtype == "int64" else "int32_t"
    values: list[int] = []
    for axis in range(rank):
        for idx in range(count):
            values.append(int(coords[idx][axis]))
    if not values:
        return
    sym = ctx.next_symbol("k2c_nonzero_idx")
    vals = ", ".join(str(v) for v in values)
    total = rank * count
    ctx.lines.append(f"  static const {ctype} {sym}[{total}] = {{ {vals} }};")
    ctx.lines.append(f"  for (size_t i = 0; i < {total}; ++i) {{")
    ctx.lines.append(f"    {out}[i] = {sym}[i];")
    ctx.lines.append("  }")
