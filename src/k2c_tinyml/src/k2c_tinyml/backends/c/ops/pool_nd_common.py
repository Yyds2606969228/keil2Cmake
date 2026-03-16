# -*- coding: utf-8 -*-

from __future__ import annotations

from ....operators.context import EmitContext
from ....operators.utils import product, tensor_size


def _row_major_strides(shape: list[int]) -> list[int]:
    if not shape:
        return []
    out = [1] * len(shape)
    acc = 1
    for axis in range(len(shape) - 1, -1, -1):
        out[axis] = acc
        acc *= int(shape[axis])
    return out


def emit_pool_nd(
    ctx: EmitContext,
    *,
    attrs: dict,
    x_name: str,
    out_name: str,
    mode: str,
    p: int = 2,
    count_include_pad: int = 0,
) -> None:
    if mode not in ("max", "avg", "lp"):
        raise ValueError("Pool mode is invalid.")

    x_shape = [int(v) for v in ctx.shape(x_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    if len(x_shape) < 3 or len(out_shape) != len(x_shape):
        raise ValueError("Pool expects rank >= 3 and matching ranks.")
    if x_shape[0] != out_shape[0]:
        raise ValueError("Pool batch dimension mismatch.")
    if x_shape[1] != out_shape[1]:
        raise ValueError("Pool channel mismatch.")

    rank = len(x_shape)
    spatial = rank - 2
    spatial_in = x_shape[2:]
    spatial_out = out_shape[2:]

    kernel = [int(v) for v in attrs.get("kernel_shape", [])]
    if len(kernel) != spatial:
        raise ValueError("Pool kernel_shape rank mismatch.")
    strides = [int(v) for v in attrs.get("strides", [1] * spatial)]
    if len(strides) != spatial:
        raise ValueError("Pool strides rank mismatch.")
    dilations = [int(v) for v in attrs.get("dilations", [1] * spatial)]
    if len(dilations) != spatial:
        raise ValueError("Pool dilations rank mismatch.")
    pads = [int(v) for v in attrs.get("pads", [0] * (spatial * 2))]
    if len(pads) == spatial:
        pads = pads + pads
    if len(pads) != spatial * 2:
        raise ValueError("Pool pads rank mismatch.")
    pad_head = pads[:spatial]

    if mode == "avg" and count_include_pad not in (0, 1):
        raise ValueError("AveragePool count_include_pad must be 0 or 1.")
    if mode == "lp" and p <= 0:
        raise ValueError("LpPool p must be positive.")

    x = ctx.map_ptr(x_name)
    out = ctx.map_ptr(out_name)
    out_dtype = ctx.dtype(out_name)
    x_dtype = ctx.dtype(x_name)

    quant_mode = out_dtype in ("int8", "int16")
    if quant_mode:
        if x_dtype != out_dtype:
            raise ValueError("Quantized pool requires matching input/output dtypes.")
        sa, za = ctx.qparams(x_name)
        so, zo = ctx.qparams(out_name)
        qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
        qctype = "int8_t" if out_dtype == "int8" else "int16_t"
    else:
        if x_dtype != "float32" or out_dtype != "float32":
            raise ValueError("Pool float path requires float32 tensors.")

    out_size = int(tensor_size(out_shape))
    kernel_size = int(product(kernel))
    in_strides = _row_major_strides(x_shape)
    out_dims_vals = ", ".join(str(v) for v in out_shape)
    in_dims_vals = ", ".join(str(v) for v in x_shape)
    in_strides_vals = ", ".join(str(v) for v in in_strides)
    kernel_dims_vals = ", ".join(str(v) for v in kernel)
    stride_vals = ", ".join(str(v) for v in strides)
    dilation_vals = ", ".join(str(v) for v in dilations)
    pad_vals = ", ".join(str(v) for v in pad_head)
    spatial_in_vals = ", ".join(str(v) for v in spatial_in)

    out_dims_sym = ctx.next_symbol("k2c_pool_out_dims")
    in_dims_sym = ctx.next_symbol("k2c_pool_in_dims")
    in_strides_sym = ctx.next_symbol("k2c_pool_in_strides")
    kernel_dims_sym = ctx.next_symbol("k2c_pool_k_dims")
    strides_sym = ctx.next_symbol("k2c_pool_strides")
    dilations_sym = ctx.next_symbol("k2c_pool_dils")
    pads_sym = ctx.next_symbol("k2c_pool_pads")
    out_coord_sym = ctx.next_symbol("k2c_pool_out_coord")
    spatial_in_sym = ctx.next_symbol("k2c_pool_spatial_in")

    ctx.lines.append(f"  static const int32_t {out_dims_sym}[{rank}] = {{ {out_dims_vals} }};")
    ctx.lines.append(f"  static const int32_t {in_dims_sym}[{rank}] = {{ {in_dims_vals} }};")
    ctx.lines.append(f"  static const int32_t {in_strides_sym}[{rank}] = {{ {in_strides_vals} }};")
    ctx.lines.append(f"  static const int32_t {kernel_dims_sym}[{spatial}] = {{ {kernel_dims_vals} }};")
    ctx.lines.append(f"  static const int32_t {spatial_in_sym}[{spatial}] = {{ {spatial_in_vals} }};")
    ctx.lines.append(f"  static const int32_t {strides_sym}[{spatial}] = {{ {stride_vals} }};")
    ctx.lines.append(f"  static const int32_t {dilations_sym}[{spatial}] = {{ {dilation_vals} }};")
    ctx.lines.append(f"  static const int32_t {pads_sym}[{spatial}] = {{ {pad_vals} }};")
    ctx.lines.append(f"  for (size_t out_i = 0; out_i < {out_size}; ++out_i) {{")
    ctx.lines.append("    size_t tmp = out_i;")
    ctx.lines.append("    size_t in_base = 0;")
    ctx.lines.append(f"    int32_t {out_coord_sym}[{spatial}];")
    ctx.lines.append(f"    for (int axis = {rank - 1}; axis >= 0; --axis) {{")
    ctx.lines.append(f"      int32_t dim = {out_dims_sym}[axis];")
    ctx.lines.append("      int32_t coord = (int32_t)(tmp % (size_t)dim);")
    ctx.lines.append("      tmp /= (size_t)dim;")
    ctx.lines.append("      if (axis >= 2) {")
    ctx.lines.append(f"        {out_coord_sym}[axis - 2] = coord;")
    ctx.lines.append("      } else {")
    ctx.lines.append(f"        in_base += (size_t)coord * (size_t){in_strides_sym}[axis];")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    if mode == "max":
        ctx.lines.append("    float acc = -3.402823466e+38F;")
    else:
        ctx.lines.append("    float acc = 0.0f;")
    ctx.lines.append("    size_t valid_count = 0;")
    ctx.lines.append(f"    for (size_t k_i = 0; k_i < {kernel_size}; ++k_i) {{")
    ctx.lines.append("      size_t ktmp = k_i;")
    ctx.lines.append("      size_t in_idx = in_base;")
    ctx.lines.append("      int valid = 1;")
    ctx.lines.append(f"      for (int s = {spatial - 1}; s >= 0; --s) {{")
    ctx.lines.append(f"        int32_t kdim = {kernel_dims_sym}[s];")
    ctx.lines.append("        int32_t kcoord = (int32_t)(ktmp % (size_t)kdim);")
    ctx.lines.append("        ktmp /= (size_t)kdim;")
    ctx.lines.append(
        f"        int32_t in_coord = {out_coord_sym}[s] * {strides_sym}[s] + "
        f"kcoord * {dilations_sym}[s] - {pads_sym}[s];"
    )
    ctx.lines.append(f"        if (in_coord < 0 || in_coord >= {spatial_in_sym}[s]) {{")
    ctx.lines.append("          valid = 0;")
    ctx.lines.append("          break;")
    ctx.lines.append("        }")
    ctx.lines.append(f"        in_idx += (size_t)in_coord * (size_t){in_strides_sym}[s + 2];")
    ctx.lines.append("      }")
    ctx.lines.append("      if (!valid) {")
    ctx.lines.append("        continue;")
    ctx.lines.append("      }")
    if quant_mode:
        ctx.lines.append(f"      float v = ((float){x}[in_idx] - {za}) * {sa:.8f}f;")
    else:
        ctx.lines.append(f"      float v = {x}[in_idx];")
    if mode == "max":
        ctx.lines.append("      if (v > acc) acc = v;")
    elif mode == "avg":
        ctx.lines.append("      acc += v;")
        ctx.lines.append("      valid_count += 1;")
    else:
        if p == 1:
            ctx.lines.append("      acc += fabsf(v);")
        elif p == 2:
            ctx.lines.append("      float av = fabsf(v);")
            ctx.lines.append("      acc += av * av;")
        else:
            ctx.lines.append(f"      acc += powf(fabsf(v), {float(p):.8f}f);")
        ctx.lines.append("      valid_count += 1;")
    ctx.lines.append("    }")

    if mode == "avg":
        if count_include_pad == 1:
            ctx.lines.append(f"    float denom = (float){kernel_size};")
        else:
            ctx.lines.append("    float denom = (float)valid_count;")
        ctx.lines.append("    if (denom > 0.0f) acc = acc / denom;")
        ctx.lines.append("    else acc = 0.0f;")
    elif mode == "lp":
        if p == 1:
            ctx.lines.append("    acc = acc;")
        elif p == 2:
            ctx.lines.append("    acc = sqrtf(acc);")
        else:
            ctx.lines.append(f"    acc = powf(acc, {1.0 / float(p):.8f}f);")

    if quant_mode:
        ctx.lines.append(f"    int q = (int)roundf(acc / {so:.8f}f) + {zo};")
        ctx.lines.append(f"    if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"    if (q > {qmax}) q = {qmax};")
        ctx.lines.append(f"    {out}[out_i] = ({qctype})q;")
    else:
        ctx.lines.append(f"    {out}[out_i] = acc;")
    ctx.lines.append("  }")
