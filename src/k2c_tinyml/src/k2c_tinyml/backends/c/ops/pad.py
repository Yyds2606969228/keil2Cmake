# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import get_const_ints, get_const_scalar, tensor_size
from .registry import register_op


def _row_major_strides(shape: list[int]) -> list[int]:
    rank = len(shape)
    out = [1] * rank
    acc = 1
    for i in range(rank - 1, -1, -1):
        out[i] = acc
        acc *= int(shape[i])
    return out


@register_op("Pad")
def emit_pad(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 1:
        raise ValueError("Pad expects at least 1 input.")
    mode = node.attrs.get("mode", "constant")
    if isinstance(mode, bytes):
        mode = mode.decode("utf-8", errors="ignore")
    mode = str(mode).lower()
    if mode not in ("constant", "reflect", "edge"):
        raise ValueError("Pad mode must be constant/reflect/edge.")

    out_name = node.outputs[0]
    in_name = node.inputs[0]
    out = ctx.map_ptr(out_name)
    inp = ctx.map_ptr(in_name)
    in_shape = [int(v) for v in ctx.shape(in_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    rank = len(in_shape)
    if rank <= 0 or len(out_shape) != rank:
        raise ValueError("Pad rank mismatch.")

    pads = node.attrs.get("pads")
    if pads is None and len(node.inputs) >= 2:
        pads = get_const_ints(ctx.model, node.inputs[1])
    if pads is None:
        raise ValueError("Pad pads not provided.")
    if len(pads) != rank * 2:
        raise ValueError("Pad pads length mismatch.")
    pad_begin = [int(v) for v in pads[:rank]]
    pad_end = [int(v) for v in pads[rank:]]
    for axis in range(rank):
        if pad_begin[axis] < 0 or pad_end[axis] < 0:
            raise ValueError("Pad does not support negative pads.")
        if out_shape[axis] != in_shape[axis] + pad_begin[axis] + pad_end[axis]:
            raise ValueError("Pad output shape mismatch.")
        if mode == "reflect" and in_shape[axis] <= 1 and (pad_begin[axis] > 0 or pad_end[axis] > 0):
            raise ValueError("Pad reflect mode requires input dim > 1 when padding is non-zero.")

    value = node.attrs.get("value", None)
    if value is None and len(node.inputs) >= 3:
        value = get_const_scalar(ctx.model, node.inputs[2])
    value = float(value) if value is not None else 0.0

    out_dtype = ctx.dtype(out_name)
    fill_expr: str
    if out_dtype == "float32":
        fill_expr = f"{value:.8f}f"
    elif out_dtype == "bool":
        fill_expr = "1" if int(round(value)) != 0 else "0"
    elif out_dtype in ("int8", "int16"):
        so, zo = ctx.qparams(out_name)
        qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
        q = int(round(value / so) + zo)
        if q < qmin:
            q = qmin
        if q > qmax:
            q = qmax
        fill_expr = str(q)
    elif out_dtype in ("int32", "int64"):
        fill_expr = str(int(round(value)))
    else:
        raise ValueError("Pad output dtype is unsupported.")

    in_stride = _row_major_strides(in_shape)
    out_size = tensor_size(out_shape)
    in_dims_sym = ctx.next_symbol("k2c_pad_in_dims")
    out_dims_sym = ctx.next_symbol("k2c_pad_out_dims")
    in_stride_sym = ctx.next_symbol("k2c_pad_in_stride")
    pad_begin_sym = ctx.next_symbol("k2c_pad_begin")

    in_dims_vals = ", ".join(str(v) for v in in_shape)
    out_dims_vals = ", ".join(str(v) for v in out_shape)
    in_stride_vals = ", ".join(str(v) for v in in_stride)
    pad_begin_vals = ", ".join(str(v) for v in pad_begin)

    ctx.lines.append(f"  static const int32_t {in_dims_sym}[{rank}] = {{ {in_dims_vals} }};")
    ctx.lines.append(f"  static const int32_t {out_dims_sym}[{rank}] = {{ {out_dims_vals} }};")
    ctx.lines.append(f"  static const int32_t {in_stride_sym}[{rank}] = {{ {in_stride_vals} }};")
    ctx.lines.append(f"  static const int32_t {pad_begin_sym}[{rank}] = {{ {pad_begin_vals} }};")

    ctx.lines.append(f"  for (size_t out_i = 0; out_i < {out_size}; ++out_i) {{")
    ctx.lines.append("    size_t tmp = out_i;")
    ctx.lines.append("    int64_t in_idx = 0;")
    ctx.lines.append("    int valid = 1;")
    ctx.lines.append(f"    for (int axis = {rank - 1}; axis >= 0; --axis) {{")
    ctx.lines.append(f"      int64_t coord = (int64_t)(tmp % (size_t){out_dims_sym}[axis]);")
    ctx.lines.append(f"      tmp /= (size_t){out_dims_sym}[axis];")
    ctx.lines.append(f"      int64_t ic = coord - (int64_t){pad_begin_sym}[axis];")
    if mode == "constant":
        ctx.lines.append(f"      if (ic < 0 || ic >= (int64_t){in_dims_sym}[axis]) {{ valid = 0; }}")
    elif mode == "reflect":
        ctx.lines.append(f"      int64_t dim = (int64_t){in_dims_sym}[axis];")
        ctx.lines.append("      while (ic < 0 || ic >= dim) {")
        ctx.lines.append("        if (ic < 0) ic = -ic;")
        ctx.lines.append("        if (ic >= dim) ic = (dim << 1) - ic - 2;")
        ctx.lines.append("      }")
    else:
        ctx.lines.append("      if (ic < 0) ic = 0;")
        ctx.lines.append(f"      if (ic >= (int64_t){in_dims_sym}[axis]) ic = (int64_t){in_dims_sym}[axis] - 1;")
    ctx.lines.append("      if (valid) in_idx += ic * (int64_t)" + in_stride_sym + "[axis];")
    ctx.lines.append("    }")
    if mode == "constant":
        ctx.lines.append(f"    {out}[out_i] = valid ? {inp}[in_idx] : ({fill_expr});")
    else:
        ctx.lines.append(f"    {out}[out_i] = {inp}[in_idx];")
    ctx.lines.append("  }")
