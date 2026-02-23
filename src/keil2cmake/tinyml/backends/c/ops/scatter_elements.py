# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import normalize_axis, tensor_size
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


@register_op("ScatterElements")
def emit_scatter_elements(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 3:
        raise ValueError("ScatterElements expects 3 inputs.")
    data_name, idx_name, upd_name = node.inputs
    out_name = node.outputs[0]

    data_dtype = ctx.dtype(data_name)
    if ctx.dtype(out_name) != data_dtype or ctx.dtype(upd_name) != data_dtype:
        raise ValueError("ScatterElements requires matching data/update/output dtype.")
    if ctx.dtype(idx_name) not in ("int8", "int16", "int32", "int64"):
        raise ValueError("ScatterElements indices dtype must be integer.")

    data_shape = [int(v) for v in ctx.shape(data_name)]
    idx_shape = [int(v) for v in ctx.shape(idx_name)]
    upd_shape = [int(v) for v in ctx.shape(upd_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]

    rank = len(data_shape)
    if rank <= 0 or len(idx_shape) != rank or len(upd_shape) != rank:
        raise ValueError("ScatterElements requires same rank >= 1.")
    if idx_shape != upd_shape:
        raise ValueError("ScatterElements requires indices/updates same shape.")
    if out_shape != data_shape:
        raise ValueError("ScatterElements output shape must equal data shape.")
    reduction = _normalize_reduction(node.attrs.get("reduction", "none"))
    if reduction not in ("none", "add", "mul", "max", "min"):
        raise ValueError("ScatterElements reduction must be none/add/mul/max/min.")
    if data_dtype == "bool" and reduction != "none":
        raise ValueError("ScatterElements bool dtype supports reduction=none only.")

    axis = normalize_axis(int(node.attrs.get("axis", 0)), rank)
    for dim_i in range(rank):
        if idx_shape[dim_i] > data_shape[dim_i]:
            raise ValueError("ScatterElements indices/updates shape exceeds data shape.")

    data = ctx.map_ptr(data_name)
    idx = ctx.map_ptr(idx_name)
    upd = ctx.map_ptr(upd_name)
    out = ctx.map_ptr(out_name)
    out_size = tensor_size(out_shape)
    upd_size = tensor_size(upd_shape)

    data_shape_sym = ctx.next_symbol("k2c_sc_shape")
    data_stride_sym = ctx.next_symbol("k2c_sc_dstride")
    idx_stride_sym = ctx.next_symbol("k2c_sc_istride")
    data_shape_vals = ", ".join(str(v) for v in data_shape)
    data_stride_vals = ", ".join(str(v) for v in _strides(data_shape))
    idx_stride_vals = ", ".join(str(v) for v in _strides(idx_shape))

    ctx.lines.append(f"  static const int32_t {data_shape_sym}[{rank}] = {{ {data_shape_vals} }};")
    ctx.lines.append(f"  static const int32_t {data_stride_sym}[{rank}] = {{ {data_stride_vals} }};")
    ctx.lines.append(f"  static const int32_t {idx_stride_sym}[{rank}] = {{ {idx_stride_vals} }};")
    ctx.lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {{ {out}[i] = {data}[i]; }}")
    ctx.lines.append(f"  for (size_t linear_i = 0; linear_i < {upd_size}; ++linear_i) {{")
    ctx.lines.append(f"    int64_t idx_v = (int64_t){idx}[linear_i];")
    ctx.lines.append(f"    if (idx_v < 0) idx_v += (int64_t){data_shape_sym}[{axis}];")
    ctx.lines.append(f"    if (idx_v < 0 || idx_v >= (int64_t){data_shape_sym}[{axis}]) continue;")
    ctx.lines.append("    size_t rem = linear_i;")
    ctx.lines.append("    int64_t dst = 0;")
    ctx.lines.append(f"    for (size_t dim_i = 0; dim_i < {rank}; ++dim_i) {{")
    ctx.lines.append(f"      int64_t coord = (int64_t)(rem / (size_t){idx_stride_sym}[dim_i]);")
    ctx.lines.append(f"      rem = rem % (size_t){idx_stride_sym}[dim_i];")
    ctx.lines.append(f"      if ((int32_t)dim_i == {axis}) coord = idx_v;")
    ctx.lines.append(f"      dst += coord * (int64_t){data_stride_sym}[dim_i];")
    ctx.lines.append("    }")
    if reduction == "none":
        ctx.lines.append(f"    {out}[(size_t)dst] = {upd}[linear_i];")
    elif reduction == "add":
        ctx.lines.append(f"    {out}[(size_t)dst] += {upd}[linear_i];")
    elif reduction == "mul":
        ctx.lines.append(f"    {out}[(size_t)dst] *= {upd}[linear_i];")
    elif reduction == "max":
        ctx.lines.append(
            f"    {out}[(size_t)dst] = ({out}[(size_t)dst] > {upd}[linear_i]) ? "
            f"{out}[(size_t)dst] : {upd}[linear_i];"
        )
    else:
        ctx.lines.append(
            f"    {out}[(size_t)dst] = ({out}[(size_t)dst] < {upd}[linear_i]) ? "
            f"{out}[(size_t)dst] : {upd}[linear_i];"
        )
    ctx.lines.append("  }")
