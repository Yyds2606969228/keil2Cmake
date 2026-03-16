# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import product
from .registry import register_op


@register_op("CumSum")
def emit_cumsum(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 2:
        raise ValueError("CumSum expects 2 inputs: data, axis.")
    x_name = node.inputs[0]
    axis_name = node.inputs[1]
    out_name = node.outputs[0]

    x_dtype = ctx.dtype(x_name)
    out_dtype = ctx.dtype(out_name)
    if x_dtype != out_dtype:
        raise ValueError("CumSum input/output dtype must match.")
    if out_dtype not in ("float32", "int8", "int16", "int32", "int64"):
        raise ValueError("CumSum supports float32/int8/int16/int32/int64 only.")
    axis_dtype = ctx.dtype(axis_name)
    if axis_dtype not in ("int8", "int16", "int32", "int64"):
        raise ValueError("CumSum axis input must be integer scalar.")

    in_shape = [int(v) for v in ctx.shape(x_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    if in_shape != out_shape:
        raise ValueError("CumSum output shape must equal input shape.")
    if len(in_shape) <= 0:
        raise ValueError("CumSum expects rank >= 1.")

    axis_shape = [int(v) for v in ctx.shape(axis_name)]
    axis_size = product(axis_shape) if axis_shape else 1
    if axis_size != 1:
        raise ValueError("CumSum axis input must be scalar.")

    exclusive = int(node.attrs.get("exclusive", 0))
    reverse = int(node.attrs.get("reverse", 0))
    if exclusive not in (0, 1) or reverse not in (0, 1):
        raise ValueError("CumSum exclusive/reverse must be 0 or 1.")

    rank = len(in_shape)
    in_strides = [1] * rank
    acc_stride = 1
    for i in range(rank - 1, -1, -1):
        in_strides[i] = acc_stride
        acc_stride *= in_shape[i]
    total = product(in_shape)

    inp = ctx.map_ptr(x_name)
    axis_ptr = ctx.map_ptr(axis_name)
    out = ctx.map_ptr(out_name)
    in_dims_sym = ctx.next_symbol("k2c_cumsum_dims")
    in_strides_sym = ctx.next_symbol("k2c_cumsum_strides")
    in_dims_vals = ", ".join(str(int(v)) for v in in_shape)
    in_strides_vals = ", ".join(str(int(v)) for v in in_strides)
    ctx.lines.append(f"  static const int32_t {in_dims_sym}[{rank}] = {{ {in_dims_vals} }};")
    ctx.lines.append(f"  static const int32_t {in_strides_sym}[{rank}] = {{ {in_strides_vals} }};")
    ctx.lines.append(f"  int64_t axis_raw = (int64_t){axis_ptr}[0];")
    ctx.lines.append(f"  if (axis_raw < 0) axis_raw += (int64_t){rank};")
    ctx.lines.append(f"  if (axis_raw < 0 || axis_raw >= (int64_t){rank}) axis_raw = 0;")
    ctx.lines.append("  size_t axis_u = (size_t)axis_raw;")
    ctx.lines.append(f"  size_t axis_dim = (size_t){in_dims_sym}[axis_u];")
    ctx.lines.append(f"  size_t axis_stride = (size_t){in_strides_sym}[axis_u];")
    ctx.lines.append("  size_t block = axis_dim * axis_stride;")

    ctx.lines.append(f"  for (size_t base = 0; base < {total}; base += block) {{")
    ctx.lines.append("    for (size_t off = 0; off < axis_stride; ++off) {")
    if out_dtype == "float32":
        ctx.lines.append("      float acc = 0.0f;")
    else:
        ctx.lines.append("      int64_t acc = 0;")
    ctx.lines.append("      for (size_t axis_i = 0; axis_i < axis_dim; ++axis_i) {")
    if reverse == 1:
        ctx.lines.append("        size_t src_axis = axis_dim - 1 - axis_i;")
        ctx.lines.append("        size_t dst_axis = src_axis;")
    else:
        ctx.lines.append("        size_t src_axis = axis_i;")
        ctx.lines.append("        size_t dst_axis = axis_i;")
    ctx.lines.append("        size_t src_idx = base + src_axis * axis_stride + off;")
    ctx.lines.append("        size_t dst_idx = base + dst_axis * axis_stride + off;")
    if exclusive == 1:
        if out_dtype == "float32":
            ctx.lines.append(f"        {out}[dst_idx] = acc;")
        elif out_dtype == "int8":
            ctx.lines.append("        int64_t q = acc;")
            ctx.lines.append("        if (q < -128) q = -128;")
            ctx.lines.append("        if (q > 127) q = 127;")
            ctx.lines.append(f"        {out}[dst_idx] = (int8_t)q;")
        elif out_dtype == "int16":
            ctx.lines.append("        int64_t q = acc;")
            ctx.lines.append("        if (q < -32768) q = -32768;")
            ctx.lines.append("        if (q > 32767) q = 32767;")
            ctx.lines.append(f"        {out}[dst_idx] = (int16_t)q;")
        elif out_dtype == "int32":
            ctx.lines.append("        int64_t q = acc;")
            ctx.lines.append("        if (q < -2147483648LL) q = -2147483648LL;")
            ctx.lines.append("        if (q > 2147483647LL) q = 2147483647LL;")
            ctx.lines.append(f"        {out}[dst_idx] = (int32_t)q;")
        else:
            ctx.lines.append(f"        {out}[dst_idx] = (int64_t)acc;")
        if out_dtype == "float32":
            ctx.lines.append(f"        acc += {inp}[src_idx];")
        else:
            ctx.lines.append(f"        acc += (int64_t){inp}[src_idx];")
    else:
        if out_dtype == "float32":
            ctx.lines.append(f"        acc += {inp}[src_idx];")
            ctx.lines.append(f"        {out}[dst_idx] = acc;")
        else:
            ctx.lines.append(f"        acc += (int64_t){inp}[src_idx];")
            if out_dtype == "int8":
                ctx.lines.append("        if (acc < -128) acc = -128;")
                ctx.lines.append("        if (acc > 127) acc = 127;")
                ctx.lines.append(f"        {out}[dst_idx] = (int8_t)acc;")
            elif out_dtype == "int16":
                ctx.lines.append("        if (acc < -32768) acc = -32768;")
                ctx.lines.append("        if (acc > 32767) acc = 32767;")
                ctx.lines.append(f"        {out}[dst_idx] = (int16_t)acc;")
            elif out_dtype == "int32":
                ctx.lines.append("        if (acc < -2147483648LL) acc = -2147483648LL;")
                ctx.lines.append("        if (acc > 2147483647LL) acc = 2147483647LL;")
                ctx.lines.append(f"        {out}[dst_idx] = (int32_t)acc;")
            else:
                ctx.lines.append(f"        {out}[dst_idx] = (int64_t)acc;")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
