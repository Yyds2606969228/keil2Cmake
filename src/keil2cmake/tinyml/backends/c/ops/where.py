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
        raise ValueError("Where broadcast rank mismatch.")
    aligned = [1] * (out_rank - in_rank) + list(in_shape)
    raw_strides = [0] * out_rank
    stride = 1
    for axis in range(out_rank - 1, -1, -1):
        dim = int(aligned[axis])
        if dim <= 0:
            raise ValueError("Where requires known positive dimensions.")
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
            raise ValueError("Where input is not broadcast-compatible with output.")
    return out


@register_op("Where")
def emit_where(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 3:
        raise ValueError("Where expects 3 inputs.")
    cond_name, x_name, y_name = node.inputs
    out_name = node.outputs[0]
    out_dtype = ctx.dtype(out_name)
    if ctx.dtype(x_name) != out_dtype or ctx.dtype(y_name) != out_dtype:
        raise ValueError("Where expects X/Y dtype equal to output dtype.")
    cond_dtype = ctx.dtype(cond_name)
    if cond_dtype not in ("bool", "float32", "int8", "int16", "int32", "int64"):
        raise ValueError("Where condition dtype is unsupported.")
    if out_dtype in ("int8", "int16"):
        sx, zx = ctx.qparams(x_name)
        sy, zy = ctx.qparams(y_name)
        qo = ctx.qparams_optional(out_name)
        if abs(sx - sy) > 1e-12 or zx != zy:
            raise ValueError("Quantized Where requires same qparams for X/Y.")
        if qo is not None:
            so, zo = qo
            if abs(sx - so) > 1e-12 or zx != zo:
                raise ValueError("Quantized Where requires output qparams equal to X/Y.")

    out_shape = ctx.shape(out_name)
    cond_shape = ctx.shape(cond_name)
    x_shape = ctx.shape(x_name)
    y_shape = ctx.shape(y_name)
    cond_strides = _broadcast_strides(cond_shape, out_shape)
    x_strides = _broadcast_strides(x_shape, out_shape)
    y_strides = _broadcast_strides(y_shape, out_shape)
    rank = len(out_shape)
    out_size = tensor_size(out_shape)
    out_dims_name = ctx.next_symbol("k2c_where_out_dims")
    cond_strides_name = ctx.next_symbol("k2c_where_cond_strides")
    x_strides_name = ctx.next_symbol("k2c_where_x_strides")
    y_strides_name = ctx.next_symbol("k2c_where_y_strides")

    cond = ctx.map_ptr(cond_name)
    x = ctx.map_ptr(x_name)
    y = ctx.map_ptr(y_name)
    out = ctx.map_ptr(out_name)
    cond_true_expr = f"{cond}[ci] != 0"
    if cond_dtype == "float32":
        cond_true_expr = f"{cond}[ci] != 0.0f"

    out_dims_vals = ", ".join(str(int(v)) for v in out_shape)
    cond_stride_vals = ", ".join(str(int(v)) for v in cond_strides)
    x_stride_vals = ", ".join(str(int(v)) for v in x_strides)
    y_stride_vals = ", ".join(str(int(v)) for v in y_strides)
    ctx.lines.append(f"  static const int32_t {out_dims_name}[{rank}] = {{ {out_dims_vals} }};")
    ctx.lines.append(
        f"  static const int32_t {cond_strides_name}[{rank}] = {{ {cond_stride_vals} }};"
    )
    ctx.lines.append(f"  static const int32_t {x_strides_name}[{rank}] = {{ {x_stride_vals} }};")
    ctx.lines.append(f"  static const int32_t {y_strides_name}[{rank}] = {{ {y_stride_vals} }};")
    ctx.lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {{")
    ctx.lines.append("    size_t tmp = i;")
    ctx.lines.append("    size_t ci = 0;")
    ctx.lines.append("    size_t xi = 0;")
    ctx.lines.append("    size_t yi = 0;")
    ctx.lines.append(f"    for (int axis = {rank - 1}; axis >= 0; --axis) {{")
    ctx.lines.append(f"      size_t coord = tmp % (size_t){out_dims_name}[axis];")
    ctx.lines.append(f"      tmp /= (size_t){out_dims_name}[axis];")
    ctx.lines.append(f"      ci += coord * (size_t){cond_strides_name}[axis];")
    ctx.lines.append(f"      xi += coord * (size_t){x_strides_name}[axis];")
    ctx.lines.append(f"      yi += coord * (size_t){y_strides_name}[axis];")
    ctx.lines.append("    }")
    ctx.lines.append(f"    {out}[i] = ({cond_true_expr}) ? {x}[xi] : {y}[yi];")
    ctx.lines.append("  }")
