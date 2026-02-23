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


def _normalize_reduction(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    return str(value).strip().lower()


@register_op("ScatterND")
def emit_scatter_nd(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 3:
        raise ValueError("ScatterND expects 3 inputs.")
    data_name, idx_name, upd_name = node.inputs
    out_name = node.outputs[0]

    data_dtype = ctx.dtype(data_name)
    if ctx.dtype(out_name) != data_dtype or ctx.dtype(upd_name) != data_dtype:
        raise ValueError("ScatterND requires matching data/update/output dtype.")
    if ctx.dtype(idx_name) not in ("int8", "int16", "int32", "int64"):
        raise ValueError("ScatterND indices dtype must be integer.")

    data_shape = [int(v) for v in ctx.shape(data_name)]
    idx_shape = [int(v) for v in ctx.shape(idx_name)]
    upd_shape = [int(v) for v in ctx.shape(upd_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    if out_shape != data_shape:
        raise ValueError("ScatterND output shape must equal data shape.")
    if len(data_shape) <= 0 or len(idx_shape) <= 0:
        raise ValueError("ScatterND requires non-scalar data/indices.")
    reduction = _normalize_reduction(node.attrs.get("reduction", "none"))
    if reduction not in ("none", "add", "mul", "max", "min"):
        raise ValueError("ScatterND reduction must be none/add/mul/max/min.")
    if data_dtype == "bool" and reduction != "none":
        raise ValueError("ScatterND bool dtype supports reduction=none only.")

    k = int(idx_shape[-1])
    if k < 0 or k > len(data_shape):
        raise ValueError("ScatterND indices last dim out of range.")
    if k == 0:
        raise ValueError("ScatterND currently does not support indices last dim = 0.")

    batch_shape = idx_shape[:-1]
    tail_shape = data_shape[k:]
    expected_upd_shape = list(batch_shape) + list(tail_shape)
    if upd_shape != expected_upd_shape:
        raise ValueError("ScatterND updates shape mismatch.")

    tuple_count = int(product(batch_shape)) if batch_shape else 1
    tail_size = int(product(tail_shape)) if tail_shape else 1
    out_size = tensor_size(out_shape)
    if tensor_size(upd_shape) != tuple_count * tail_size:
        raise ValueError("ScatterND updates size mismatch.")

    data = ctx.map_ptr(data_name)
    idx = ctx.map_ptr(idx_name)
    upd = ctx.map_ptr(upd_name)
    out = ctx.map_ptr(out_name)

    data_shape_sym = ctx.next_symbol("k2c_scnd_shape")
    data_stride_sym = ctx.next_symbol("k2c_scnd_stride")
    data_shape_vals = ", ".join(str(v) for v in data_shape)
    data_stride_vals = ", ".join(str(v) for v in _strides(data_shape))

    ctx.lines.append(f"  static const int32_t {data_shape_sym}[{len(data_shape)}] = {{ {data_shape_vals} }};")
    ctx.lines.append(f"  static const int32_t {data_stride_sym}[{len(data_shape)}] = {{ {data_stride_vals} }};")
    ctx.lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {{ {out}[i] = {data}[i]; }}")
    ctx.lines.append(f"  for (size_t tuple_i = 0; tuple_i < {tuple_count}; ++tuple_i) {{")
    ctx.lines.append("    int64_t base = 0;")
    ctx.lines.append("    int valid = 1;")
    ctx.lines.append(f"    for (size_t j = 0; j < {k}; ++j) {{")
    ctx.lines.append(f"      int64_t v = (int64_t){idx}[tuple_i * {k} + j];")
    ctx.lines.append(f"      if (v < 0) v += (int64_t){data_shape_sym}[j];")
    ctx.lines.append(f"      if (v < 0 || v >= (int64_t){data_shape_sym}[j]) {{ valid = 0; break; }}")
    ctx.lines.append(f"      base += v * (int64_t){data_stride_sym}[j];")
    ctx.lines.append("    }")
    ctx.lines.append("    if (!valid) continue;")
    ctx.lines.append(f"    for (size_t t = 0; t < {tail_size}; ++t) {{")
    ctx.lines.append("      size_t dst_i = (size_t)base + t;")
    if reduction == "none":
        ctx.lines.append(f"      {out}[dst_i] = {upd}[tuple_i * {tail_size} + t];")
    elif reduction == "add":
        ctx.lines.append(f"      {out}[dst_i] += {upd}[tuple_i * {tail_size} + t];")
    elif reduction == "mul":
        ctx.lines.append(f"      {out}[dst_i] *= {upd}[tuple_i * {tail_size} + t];")
    elif reduction == "max":
        ctx.lines.append(
            f"      {out}[dst_i] = ({out}[dst_i] > {upd}[tuple_i * {tail_size} + t]) ? "
            f"{out}[dst_i] : {upd}[tuple_i * {tail_size} + t];"
        )
    else:
        ctx.lines.append(
            f"      {out}[dst_i] = ({out}[dst_i] < {upd}[tuple_i * {tail_size} + t]) ? "
            f"{out}[dst_i] : {upd}[tuple_i * {tail_size} + t];"
        )
    ctx.lines.append("    }")
    ctx.lines.append("  }")
