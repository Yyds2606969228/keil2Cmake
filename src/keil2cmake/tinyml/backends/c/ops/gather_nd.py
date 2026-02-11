# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import product, tensor_size
from .registry import register_op


def _strides(shape: list[int]) -> list[int]:
    out = [1] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        out[i] = acc
        acc *= int(shape[i])
    return out


@register_op("GatherND")
def emit_gather_nd(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 2:
        raise ValueError("GatherND expects 2 inputs.")
    data_name, idx_name = node.inputs
    out_name = node.outputs[0]

    if ctx.dtype(data_name) != ctx.dtype(out_name):
        raise ValueError("GatherND output dtype must match data dtype.")
    if ctx.dtype(idx_name) not in ("int8", "int16", "int32", "int64"):
        raise ValueError("GatherND indices dtype must be integer.")

    batch_dims = int(node.attrs.get("batch_dims", 0))
    if batch_dims != 0:
        raise ValueError("GatherND currently supports batch_dims=0 only.")

    data_shape = [int(v) for v in ctx.shape(data_name)]
    idx_shape = [int(v) for v in ctx.shape(idx_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    if len(data_shape) <= 0 or len(idx_shape) <= 0:
        raise ValueError("GatherND requires non-scalar data/indices.")
    k = int(idx_shape[-1])
    if k < 0 or k > len(data_shape):
        raise ValueError("GatherND indices last dim out of range.")

    expected = list(idx_shape[:-1]) + list(data_shape[k:])
    if expected != out_shape:
        raise ValueError("GatherND output shape mismatch.")

    tuple_count = int(product(idx_shape[:-1])) if len(idx_shape) > 1 else 1
    tail_size = int(product(data_shape[k:])) if k < len(data_shape) else 1
    out_size = tensor_size(out_shape)
    if out_size != tuple_count * tail_size:
        raise ValueError("GatherND output size mismatch.")

    data = ctx.map_ptr(data_name)
    idx = ctx.map_ptr(idx_name)
    out = ctx.map_ptr(out_name)

    shape_sym = ctx.next_symbol("k2c_gnd_shape")
    stride_sym = ctx.next_symbol("k2c_gnd_stride")
    shape_vals = ", ".join(str(v) for v in data_shape)
    stride_vals = ", ".join(str(v) for v in _strides(data_shape))

    ctx.lines.append(f"  static const int32_t {shape_sym}[{len(data_shape)}] = {{ {shape_vals} }};")
    ctx.lines.append(f"  static const int32_t {stride_sym}[{len(data_shape)}] = {{ {stride_vals} }};")
    ctx.lines.append(f"  for (size_t tuple_i = 0; tuple_i < {tuple_count}; ++tuple_i) {{")
    ctx.lines.append("    int32_t base = 0;")
    ctx.lines.append(f"    for (size_t j = 0; j < {k}; ++j) {{")
    ctx.lines.append(f"      int32_t v = (int32_t){idx}[tuple_i * {k} + j];")
    ctx.lines.append(f"      if (v < 0) v += {shape_sym}[j];")
    ctx.lines.append(f"      if (v < 0 || v >= {shape_sym}[j]) v = 0;")
    ctx.lines.append(f"      base += v * {stride_sym}[j];")
    ctx.lines.append("    }")
    ctx.lines.append(f"    for (size_t t = 0; t < {tail_size}; ++t) {{")
    ctx.lines.append(f"      {out}[tuple_i * {tail_size} + t] = {data}[base + (int32_t)t];")
    ctx.lines.append("    }")
    ctx.lines.append("  }")

