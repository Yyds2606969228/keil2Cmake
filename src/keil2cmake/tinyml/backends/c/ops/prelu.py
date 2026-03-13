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
        raise ValueError("PRelu broadcast rank mismatch.")
    aligned = [1] * (out_rank - in_rank) + list(in_shape)
    raw_strides = [0] * out_rank
    stride = 1
    for axis in range(out_rank - 1, -1, -1):
        dim = int(aligned[axis])
        if dim <= 0:
            raise ValueError("PRelu requires known positive dimensions.")
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
            raise ValueError("PRelu input is not broadcast-compatible with output.")
    return out


@register_op("PRelu")
def emit_prelu(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 2:
        raise ValueError("PRelu expects 2 inputs.")
    x_name, slope_name = node.inputs
    out_name = node.outputs[0]
    out_dtype = ctx.dtype(out_name)
    x_dtype = ctx.dtype(x_name)
    slope_dtype = ctx.dtype(slope_name)
    quant_mode = out_dtype in ("int8", "int16")
    if quant_mode:
        if x_dtype != out_dtype:
            raise ValueError("PRelu quantized path requires input/output dtypes to match.")
        if slope_dtype not in ("float32", out_dtype):
            raise ValueError("PRelu quantized path requires float32 or matching quantized slope.")
        sx, zx = ctx.qparams(x_name)
        so, zo = ctx.qparams(out_name)
        ss = 0.0
        zs = 0
        if slope_dtype == out_dtype:
            ss, zs = ctx.qparams(slope_name)
        qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
        qctype = "int8_t" if out_dtype == "int8" else "int16_t"
    else:
        if out_dtype != "float32" or x_dtype != "float32" or slope_dtype != "float32":
            raise ValueError("PRelu supports float32 or quantized int8/int16.")

    out_shape = ctx.shape(out_name)
    x_shape = ctx.shape(x_name)
    slope_shape = ctx.shape(slope_name)
    x_strides = _broadcast_strides(x_shape, out_shape)
    slope_strides = _broadcast_strides(slope_shape, out_shape)
    rank = len(out_shape)
    out_size = tensor_size(out_shape)

    out_dims_name = ctx.next_symbol("k2c_prelu_out_dims")
    x_strides_name = ctx.next_symbol("k2c_prelu_x_strides")
    slope_strides_name = ctx.next_symbol("k2c_prelu_slope_strides")
    out_dims_vals = ", ".join(str(int(v)) for v in out_shape)
    x_stride_vals = ", ".join(str(int(v)) for v in x_strides)
    slope_stride_vals = ", ".join(str(int(v)) for v in slope_strides)

    x = ctx.map_ptr(x_name)
    slope = ctx.map_ptr(slope_name)
    out = ctx.map_ptr(out_name)
    ctx.lines.append(f"  static const int32_t {out_dims_name}[{rank}] = {{ {out_dims_vals} }};")
    ctx.lines.append(f"  static const int32_t {x_strides_name}[{rank}] = {{ {x_stride_vals} }};")
    ctx.lines.append(
        f"  static const int32_t {slope_strides_name}[{rank}] = {{ {slope_stride_vals} }};"
    )
    ctx.lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {{")
    ctx.lines.append("    size_t tmp = i;")
    ctx.lines.append("    size_t xi = 0;")
    ctx.lines.append("    size_t si = 0;")
    ctx.lines.append(f"    for (int axis = {rank - 1}; axis >= 0; --axis) {{")
    ctx.lines.append(f"      size_t coord = tmp % (size_t){out_dims_name}[axis];")
    ctx.lines.append(f"      tmp /= (size_t){out_dims_name}[axis];")
    ctx.lines.append(f"      xi += coord * (size_t){x_strides_name}[axis];")
    ctx.lines.append(f"      si += coord * (size_t){slope_strides_name}[axis];")
    ctx.lines.append("    }")
    if quant_mode:
        ctx.lines.append(f"    float v = ((float){x}[xi] - {zx}) * {sx:.8f}f;")
        if slope_dtype == "float32":
            ctx.lines.append(f"    float s = {slope}[si];")
        else:
            ctx.lines.append(f"    float s = ((float){slope}[si] - {zs}) * {ss:.8f}f;")
        ctx.lines.append("    float r = v >= 0.0f ? v : v * s;")
        ctx.lines.append(f"    int q = (int)roundf(r / {so:.8f}f) + {zo};")
        ctx.lines.append(f"    if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"    if (q > {qmax}) q = {qmax};")
        ctx.lines.append(f"    {out}[i] = ({qctype})q;")
    else:
        ctx.lines.append(f"    float v = {x}[xi];")
        ctx.lines.append(f"    {out}[i] = v >= 0.0f ? v : v * {slope}[si];")
    ctx.lines.append("  }")
