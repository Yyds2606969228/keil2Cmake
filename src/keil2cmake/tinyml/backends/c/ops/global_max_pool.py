# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import product
from .registry import register_op


@register_op("GlobalMaxPool")
def emit_global_max_pool(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("GlobalMaxPool expects 1 input.")
    x_name = node.inputs[0]
    out_name = node.outputs[0]
    x_shape = ctx.shape(x_name)
    out_shape = ctx.shape(out_name)
    if len(x_shape) < 3 or len(out_shape) != len(x_shape):
        raise ValueError("GlobalMaxPool expects rank >= 3 and matching ranks.")
    n, c = int(x_shape[0]), int(x_shape[1])
    n_out, c_out = int(out_shape[0]), int(out_shape[1])
    expected_out = [n, c] + [1] * (len(x_shape) - 2)
    if n != n_out:
        raise ValueError("GlobalMaxPool batch dimension mismatch.")
    if c != c_out or [int(v) for v in out_shape] != expected_out:
        raise ValueError("GlobalMaxPool output shape mismatch.")
    spatial = int(product(x_shape[2:]))
    out_inner = int(product(out_shape[2:]))

    out_dtype = ctx.dtype(out_name)
    x_dtype = ctx.dtype(x_name)
    if out_dtype not in ("float32", "int8", "int16"):
        raise ValueError("GlobalMaxPool supports float32/int8/int16 only.")
    if x_dtype != out_dtype:
        raise ValueError("GlobalMaxPool requires matching input/output dtypes.")

    x = ctx.map_ptr(x_name)
    out = ctx.map_ptr(out_name)
    if out_dtype == "float32":
        ctx.lines.append(f"  for (size_t ni = 0; ni < {n}; ++ni) {{")
        ctx.lines.append(f"    for (size_t ch = 0; ch < {c}; ++ch) {{")
        ctx.lines.append("      float acc = -3.402823466e+38F;")
        ctx.lines.append(f"      for (size_t i = 0; i < {spatial}; ++i) {{")
        ctx.lines.append(f"        size_t in_idx = ((ni * {c} + ch) * {spatial}) + i;")
        ctx.lines.append(f"        float v = {x}[in_idx];")
        ctx.lines.append("        if (v > acc) acc = v;")
        ctx.lines.append("      }")
        ctx.lines.append(f"      size_t out_idx = ((ni * {c_out} + ch) * {out_inner});")
        ctx.lines.append(f"      {out}[out_idx] = acc;")
        ctx.lines.append("    }")
        ctx.lines.append("  }")
        return

    sa, za = ctx.qparams(x_name)
    so, zo = ctx.qparams(out_name)
    qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
    ctype = "int8_t" if out_dtype == "int8" else "int16_t"
    ctx.lines.append(f"  for (size_t ni = 0; ni < {n}; ++ni) {{")
    ctx.lines.append(f"    for (size_t ch = 0; ch < {c}; ++ch) {{")
    ctx.lines.append("      float acc = -3.402823466e+38F;")
    ctx.lines.append(f"      for (size_t i = 0; i < {spatial}; ++i) {{")
    ctx.lines.append(f"        size_t in_idx = ((ni * {c} + ch) * {spatial}) + i;")
    ctx.lines.append(
        f"        float v = ((float){x}[in_idx] - {za}) * {sa:.8f}f;"
    )
    ctx.lines.append("        if (v > acc) acc = v;")
    ctx.lines.append("      }")
    ctx.lines.append(f"      int q = (int)roundf(acc / {so:.8f}f) + {zo};")
    ctx.lines.append(f"      if (q < {qmin}) q = {qmin};")
    ctx.lines.append(f"      if (q > {qmax}) q = {qmax};")
    ctx.lines.append(f"      size_t out_idx = ((ni * {c_out} + ch) * {out_inner});")
    ctx.lines.append(f"      {out}[out_idx] = ({ctype})q;")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
