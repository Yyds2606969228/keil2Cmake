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
        raise ValueError("Mod broadcast rank mismatch.")
    aligned = [1] * (out_rank - in_rank) + list(in_shape)
    raw_strides = [0] * out_rank
    stride = 1
    for axis in range(out_rank - 1, -1, -1):
        dim = int(aligned[axis])
        if dim <= 0:
            raise ValueError("Mod requires known positive dimensions.")
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
            raise ValueError("Mod input is not broadcast-compatible with output.")
    return out


@register_op("Mod")
def emit_mod(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 2:
        raise ValueError("Mod expects 2 inputs.")
    a_name, b_name = node.inputs
    out_name = node.outputs[0]
    out_dtype = ctx.dtype(out_name)
    a_dtype = ctx.dtype(a_name)
    b_dtype = ctx.dtype(b_name)
    quant_mode = out_dtype in ("int8", "int16")
    if quant_mode:
        if a_dtype != out_dtype or b_dtype != out_dtype:
            raise ValueError("Mod quantized path requires matching int8/int16 dtypes.")
        sa, za = ctx.qparams(a_name)
        sb, zb = ctx.qparams(b_name)
        so, zo = ctx.qparams(out_name)
        qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
        qctype = "int8_t" if out_dtype == "int8" else "int16_t"
    elif out_dtype != "float32" or a_dtype != "float32" or b_dtype != "float32":
        raise ValueError("Mod supports float32 or quantized int8/int16.")

    out_shape = ctx.shape(out_name)
    a_shape = ctx.shape(a_name)
    b_shape = ctx.shape(b_name)
    a_strides = _broadcast_strides(a_shape, out_shape)
    b_strides = _broadcast_strides(b_shape, out_shape)
    rank = len(out_shape)
    out_size = tensor_size(out_shape)
    fmod_mode = int(node.attrs.get("fmod", 0))

    out_dims_name = ctx.next_symbol("k2c_mod_out_dims")
    a_strides_name = ctx.next_symbol("k2c_mod_a_strides")
    b_strides_name = ctx.next_symbol("k2c_mod_b_strides")
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
    if quant_mode:
        ctx.lines.append(f"    float av = ((float){a}[ai] - {za}) * {sa:.8f}f;")
        ctx.lines.append(f"    float bv = ((float){b}[bi] - {zb}) * {sb:.8f}f;")
        if fmod_mode == 1:
            ctx.lines.append("    float rv = fmodf(av, bv);")
        else:
            ctx.lines.append("    float rv = av - floorf(av / bv) * bv;")
        ctx.lines.append(f"    int q = (int)roundf(rv / {so:.8f}f) + {zo};")
        ctx.lines.append(f"    if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"    if (q > {qmax}) q = {qmax};")
        ctx.lines.append(f"    {out}[i] = ({qctype})q;")
    elif fmod_mode == 1:
        ctx.lines.append(f"    {out}[i] = fmodf({a}[ai], {b}[bi]);")
    else:
        ctx.lines.append(f"    {out}[i] = {a}[ai] - floorf({a}[ai] / {b}[bi]) * {b}[bi];")
    ctx.lines.append("  }")
