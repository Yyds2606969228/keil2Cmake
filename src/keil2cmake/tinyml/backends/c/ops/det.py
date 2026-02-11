# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import tensor_size
from .registry import register_op


@register_op("Det")
def emit_det(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("Det expects 1 input.")
    x_name = node.inputs[0]
    out_name = node.outputs[0]
    x_dtype = ctx.dtype(x_name)
    out_dtype = ctx.dtype(out_name)
    quant_in = x_dtype in ("int8", "int16")
    quant_out = out_dtype in ("int8", "int16")
    if x_dtype not in ("float32", "int8", "int16"):
        raise ValueError("Det input dtype is unsupported.")
    if out_dtype not in ("float32", "int8", "int16"):
        raise ValueError("Det output dtype is unsupported.")

    x_shape = [int(v) for v in ctx.shape(x_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    if len(x_shape) != 2:
        raise ValueError("Det currently supports 2D input only.")
    n0, n1 = x_shape
    if n0 != n1:
        raise ValueError("Det requires square matrix.")
    if tensor_size(out_shape) != 1:
        raise ValueError("Det output must be scalar.")

    n = int(n0)
    x = ctx.map_ptr(x_name)
    out = ctx.map_ptr(out_name)
    mat_sym = ctx.next_symbol("k2c_det_mat")
    done_label = ctx.next_symbol("k2c_det_done")

    ctx.lines.append(f"  static float {mat_sym}[{n} * {n}];")
    if quant_in:
        sx, zx = ctx.qparams(x_name)
        ctx.lines.append(
            f"  for (size_t i = 0; i < {n} * {n}; ++i) {{ {mat_sym}[i] = ((float){x}[i] - {zx}) * {sx:.8f}f; }}"
        )
    else:
        ctx.lines.append(f"  for (size_t i = 0; i < {n} * {n}; ++i) {{ {mat_sym}[i] = {x}[i]; }}")
    ctx.lines.append("  float det_v = 1.0f;")
    ctx.lines.append("  int det_sign = 1;")
    ctx.lines.append(f"  for (size_t i = 0; i < {n}; ++i) {{")
    ctx.lines.append("    size_t pivot = i;")
    ctx.lines.append(f"    float max_abs = fabsf({mat_sym}[i * {n} + i]);")
    ctx.lines.append(f"    for (size_t r = i + 1; r < {n}; ++r) {{")
    ctx.lines.append(f"      float av = fabsf({mat_sym}[r * {n} + i]);")
    ctx.lines.append("      if (av > max_abs) { max_abs = av; pivot = r; }")
    ctx.lines.append("    }")
    ctx.lines.append("    if (max_abs <= 1e-12f) { det_v = 0.0f; goto " + done_label + "; }")
    ctx.lines.append("    if (pivot != i) {")
    ctx.lines.append(f"      for (size_t c = i; c < {n}; ++c) {{")
    ctx.lines.append(f"        float tmp = {mat_sym}[i * {n} + c];")
    ctx.lines.append(f"        {mat_sym}[i * {n} + c] = {mat_sym}[pivot * {n} + c];")
    ctx.lines.append(f"        {mat_sym}[pivot * {n} + c] = tmp;")
    ctx.lines.append("      }")
    ctx.lines.append("      det_sign = -det_sign;")
    ctx.lines.append("    }")
    ctx.lines.append(f"    float pivot_v = {mat_sym}[i * {n} + i];")
    ctx.lines.append(f"    for (size_t r = i + 1; r < {n}; ++r) {{")
    ctx.lines.append(f"      float factor = {mat_sym}[r * {n} + i] / pivot_v;")
    ctx.lines.append(f"      for (size_t c = i + 1; c < {n}; ++c) {{")
    ctx.lines.append(f"        {mat_sym}[r * {n} + c] -= factor * {mat_sym}[i * {n} + c];")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("    det_v *= pivot_v;")
    ctx.lines.append("  }")
    ctx.lines.append(f"{done_label}:")
    ctx.lines.append("  if (det_sign < 0) det_v = -det_v;")
    if quant_out:
        so, zo = ctx.qparams(out_name)
        qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
        qctype = "int8_t" if out_dtype == "int8" else "int16_t"
        ctx.lines.append(f"  int q = (int)roundf(det_v / {so:.8f}f) + {zo};")
        ctx.lines.append(f"  if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"  if (q > {qmax}) q = {qmax};")
        ctx.lines.append(f"  {out}[0] = ({qctype})q;")
    else:
        ctx.lines.append(f"  {out}[0] = det_v;")
