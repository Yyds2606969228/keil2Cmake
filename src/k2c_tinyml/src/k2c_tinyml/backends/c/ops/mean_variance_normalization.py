# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import normalize_axis, product
from .registry import register_op


def _normalize_axes(attrs: dict, rank: int) -> list[int]:
    raw_axes = attrs.get("axes", [0, 2, 3])
    if isinstance(raw_axes, (list, tuple)):
        axes = [int(v) for v in raw_axes]
    else:
        axes = [int(raw_axes)]
    out: list[int] = []
    seen: set[int] = set()
    for axis in axes:
        norm = normalize_axis(axis, rank)
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


@register_op("MeanVarianceNormalization")
def emit_mean_variance_normalization(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("MeanVarianceNormalization expects 1 input.")
    x_name = node.inputs[0]
    out_name = node.outputs[0]

    x_dtype = ctx.dtype(x_name)
    out_dtype = ctx.dtype(out_name)
    quant_mode = x_dtype in ("int8", "int16") or out_dtype in ("int8", "int16")
    if quant_mode:
        if x_dtype != out_dtype or x_dtype not in ("int8", "int16"):
            raise ValueError("MeanVarianceNormalization quantized path requires matching int8/int16 input/output.")
        sx, zx = ctx.qparams(x_name)
        so, zo = ctx.qparams(out_name)
        qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
        qctype = "int8_t" if out_dtype == "int8" else "int16_t"
    elif x_dtype != "float32" or out_dtype != "float32":
        raise ValueError("MeanVarianceNormalization supports float32 or quantized int8/int16.")

    x_shape = [int(v) for v in ctx.shape(x_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    if x_shape != out_shape:
        raise ValueError("MeanVarianceNormalization output shape mismatch.")
    rank = len(x_shape)
    if rank <= 0:
        raise ValueError("MeanVarianceNormalization expects rank >= 1.")

    axes = _normalize_axes(node.attrs, rank)
    axis_set = set(axes)
    keep_axes = [axis for axis in range(rank) if axis not in axis_set]

    keep_count = product([x_shape[axis] for axis in keep_axes]) if keep_axes else 1
    reduce_count = product([x_shape[axis] for axis in axes]) if axes else 1
    total = product(x_shape)

    keep_strides = [0] * rank
    stride = 1
    for axis in range(len(keep_axes) - 1, -1, -1):
        dim_axis = keep_axes[axis]
        keep_strides[dim_axis] = stride
        stride *= int(x_shape[dim_axis])

    dims_vals = ", ".join(str(v) for v in x_shape)
    keep_stride_vals = ", ".join(str(v) for v in keep_strides)
    dims_sym = ctx.next_symbol("k2c_mvn_dims")
    keep_sym = ctx.next_symbol("k2c_mvn_keep_strides")
    mean_sym = ctx.next_symbol("k2c_mvn_mean")
    var_sym = ctx.next_symbol("k2c_mvn_var")

    x = ctx.map_ptr(x_name)
    out = ctx.map_ptr(out_name)
    eps = float(node.attrs.get("epsilon", 1e-12))

    ctx.lines.append(f"  static const int32_t {dims_sym}[{rank}] = {{ {dims_vals} }};")
    ctx.lines.append(f"  static const int32_t {keep_sym}[{rank}] = {{ {keep_stride_vals} }};")
    ctx.lines.append(f"  static float {mean_sym}[{keep_count}];")
    ctx.lines.append(f"  static float {var_sym}[{keep_count}];")
    ctx.lines.append(f"  for (size_t gi = 0; gi < {keep_count}; ++gi) {{")
    ctx.lines.append(f"    {mean_sym}[gi] = 0.0f;")
    ctx.lines.append(f"    {var_sym}[gi] = 0.0f;")
    ctx.lines.append("  }")

    ctx.lines.append(f"  for (size_t idx = 0; idx < {total}; ++idx) {{")
    ctx.lines.append("    size_t tmp = idx;")
    ctx.lines.append("    size_t group = 0;")
    for axis in range(rank - 1, -1, -1):
        ctx.lines.append(f"    size_t c{axis} = tmp % (size_t){dims_sym}[{axis}];")
        ctx.lines.append(f"    tmp /= (size_t){dims_sym}[{axis}];")
        ctx.lines.append(f"    if ({keep_sym}[{axis}] != 0) group += c{axis} * (size_t){keep_sym}[{axis}];")
    if quant_mode:
        ctx.lines.append(f"    float xv = ((float){x}[idx] - {zx}) * {sx:.8f}f;")
    else:
        ctx.lines.append(f"    float xv = {x}[idx];")
    ctx.lines.append(f"    {mean_sym}[group] += xv;")
    ctx.lines.append("  }")

    ctx.lines.append(f"  for (size_t gi = 0; gi < {keep_count}; ++gi) {{")
    ctx.lines.append(f"    {mean_sym}[gi] /= (float){reduce_count};")
    ctx.lines.append("  }")

    ctx.lines.append(f"  for (size_t idx = 0; idx < {total}; ++idx) {{")
    ctx.lines.append("    size_t tmp = idx;")
    ctx.lines.append("    size_t group = 0;")
    for axis in range(rank - 1, -1, -1):
        ctx.lines.append(f"    size_t c{axis} = tmp % (size_t){dims_sym}[{axis}];")
        ctx.lines.append(f"    tmp /= (size_t){dims_sym}[{axis}];")
        ctx.lines.append(f"    if ({keep_sym}[{axis}] != 0) group += c{axis} * (size_t){keep_sym}[{axis}];")
    if quant_mode:
        ctx.lines.append(f"    float xv = ((float){x}[idx] - {zx}) * {sx:.8f}f;")
    else:
        ctx.lines.append(f"    float xv = {x}[idx];")
    ctx.lines.append(f"    float dv = xv - {mean_sym}[group];")
    ctx.lines.append(f"    {var_sym}[group] += dv * dv;")
    ctx.lines.append("  }")

    ctx.lines.append(f"  for (size_t gi = 0; gi < {keep_count}; ++gi) {{")
    ctx.lines.append(f"    {var_sym}[gi] /= (float){reduce_count};")
    ctx.lines.append("  }")

    ctx.lines.append(f"  for (size_t idx = 0; idx < {total}; ++idx) {{")
    ctx.lines.append("    size_t tmp = idx;")
    ctx.lines.append("    size_t group = 0;")
    for axis in range(rank - 1, -1, -1):
        ctx.lines.append(f"    size_t c{axis} = tmp % (size_t){dims_sym}[{axis}];")
        ctx.lines.append(f"    tmp /= (size_t){dims_sym}[{axis}];")
        ctx.lines.append(f"    if ({keep_sym}[{axis}] != 0) group += c{axis} * (size_t){keep_sym}[{axis}];")
    if quant_mode:
        ctx.lines.append(f"    float xv = ((float){x}[idx] - {zx}) * {sx:.8f}f;")
        ctx.lines.append(f"    float rv = (xv - {mean_sym}[group]) / sqrtf({var_sym}[group] + {eps:.8f}f);")
        ctx.lines.append(f"    int q = (int)roundf(rv / {so:.8f}f) + {zo};")
        ctx.lines.append(f"    if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"    if (q > {qmax}) q = {qmax};")
        ctx.lines.append(f"    {out}[idx] = ({qctype})q;")
    else:
        ctx.lines.append(f"    {out}[idx] = ({x}[idx] - {mean_sym}[group]) / sqrtf({var_sym}[group] + {eps:.8f}f);")
    ctx.lines.append("  }")
