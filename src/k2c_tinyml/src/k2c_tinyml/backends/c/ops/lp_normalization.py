# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import normalize_axis, product
from .registry import register_op


@register_op("LpNormalization")
def emit_lp_normalization(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("LpNormalization expects 1 input.")
    x_name = node.inputs[0]
    out_name = node.outputs[0]
    x_dtype = ctx.dtype(x_name)
    out_dtype = ctx.dtype(out_name)
    quant_mode = x_dtype in ("int8", "int16") or out_dtype in ("int8", "int16")
    if quant_mode:
        if x_dtype != out_dtype or x_dtype not in ("int8", "int16"):
            raise ValueError("LpNormalization quantized path requires matching int8/int16 input/output.")
        sx, zx = ctx.qparams(x_name)
        so, zo = ctx.qparams(out_name)
        qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
        qctype = "int8_t" if out_dtype == "int8" else "int16_t"
    elif x_dtype != "float32" or out_dtype != "float32":
        raise ValueError("LpNormalization supports float32 or quantized int8/int16.")

    x_shape = [int(v) for v in ctx.shape(x_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    if x_shape != out_shape:
        raise ValueError("LpNormalization output shape mismatch.")
    rank = len(x_shape)
    if rank <= 0:
        raise ValueError("LpNormalization input rank must be >= 1.")

    axis = normalize_axis(int(node.attrs.get("axis", -1)), rank)
    p = int(node.attrs.get("p", 2))
    if p <= 0:
        raise ValueError("LpNormalization p must be positive.")

    axis_size = int(x_shape[axis])
    if axis_size <= 0:
        raise ValueError("LpNormalization axis dimension must be positive.")
    outer = product(x_shape[:axis]) if axis > 0 else 1
    inner = product(x_shape[axis + 1 :]) if axis + 1 < rank else 1

    x = ctx.map_ptr(x_name)
    out = ctx.map_ptr(out_name)

    ctx.lines.append(f"  for (size_t outer_i = 0; outer_i < {outer}; ++outer_i) {{")
    ctx.lines.append(f"    for (size_t inner_i = 0; inner_i < {inner}; ++inner_i) {{")
    ctx.lines.append(f"      size_t base = outer_i * {axis_size} * {inner} + inner_i;")
    ctx.lines.append("      float acc = 0.0f;")
    ctx.lines.append(f"      for (size_t axis_i = 0; axis_i < {axis_size}; ++axis_i) {{")
    ctx.lines.append(f"        size_t idx = base + axis_i * {inner};")
    if quant_mode:
        ctx.lines.append(f"        float xv = ((float){x}[idx] - {zx}) * {sx:.8f}f;")
        ctx.lines.append("        float av = fabsf(xv);")
    else:
        ctx.lines.append(f"        float av = fabsf({x}[idx]);")
    if p == 1:
        ctx.lines.append("        acc += av;")
    elif p == 2:
        ctx.lines.append("        acc += av * av;")
    else:
        ctx.lines.append(f"        acc += powf(av, {float(p):.8f}f);")
    ctx.lines.append("      }")
    if p == 1:
        ctx.lines.append("      float norm = acc;")
    elif p == 2:
        ctx.lines.append("      float norm = sqrtf(acc);")
    else:
        ctx.lines.append(f"      float norm = powf(acc, {1.0 / float(p):.8f}f);")
    ctx.lines.append(f"      for (size_t axis_i = 0; axis_i < {axis_size}; ++axis_i) {{")
    ctx.lines.append(f"        size_t idx = base + axis_i * {inner};")
    if quant_mode:
        ctx.lines.append(f"        float xv = ((float){x}[idx] - {zx}) * {sx:.8f}f;")
        ctx.lines.append("        float rv = (norm <= 0.0f) ? 0.0f : (xv / norm);")
        ctx.lines.append(f"        int q = (int)roundf(rv / {so:.8f}f) + {zo};")
        ctx.lines.append(f"        if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"        if (q > {qmax}) q = {qmax};")
        ctx.lines.append(f"        {out}[idx] = ({qctype})q;")
    else:
        ctx.lines.append(f"        {out}[idx] = (norm <= 0.0f) ? 0.0f : ({x}[idx] / norm);")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
