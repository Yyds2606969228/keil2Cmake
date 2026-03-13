# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import product
from .registry import register_op


@register_op("LRN")
def emit_lrn(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("LRN expects 1 input.")

    x_name = node.inputs[0]
    out_name = node.outputs[0]
    x_shape = ctx.shape(x_name)
    out_shape = ctx.shape(out_name)
    if x_shape != out_shape:
        raise ValueError("LRN input/output shape mismatch.")
    if len(x_shape) < 3:
        raise ValueError("LRN expects rank >= 3.")

    x_dtype = ctx.dtype(x_name)
    out_dtype = ctx.dtype(out_name)
    quant_mode = x_dtype in ("int8", "int16") or out_dtype in ("int8", "int16")
    if quant_mode:
        if x_dtype != out_dtype or x_dtype not in ("int8", "int16"):
            raise ValueError("LRN quantized path requires matching int8/int16 input/output.")
        sx, zx = ctx.qparams(x_name)
        so, zo = ctx.qparams(out_name)
        qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
        qctype = "int8_t" if out_dtype == "int8" else "int16_t"
    elif x_dtype != "float32" or out_dtype != "float32":
        raise ValueError("LRN supports float32 or quantized int8/int16.")

    n = int(x_shape[0])
    c = int(x_shape[1])
    if n <= 0 or c <= 0:
        raise ValueError("LRN requires known positive N/C.")
    inner = int(product(x_shape[2:]))
    if inner <= 0:
        raise ValueError("LRN requires known positive spatial size.")

    size = int(node.attrs.get("size", 0))
    if size <= 0:
        raise ValueError("LRN requires positive 'size'.")
    alpha = float(node.attrs.get("alpha", 1e-4))
    beta = float(node.attrs.get("beta", 0.75))
    bias = float(node.attrs.get("bias", 1.0))

    radius = size // 2
    x = ctx.map_ptr(x_name)
    out = ctx.map_ptr(out_name)

    ctx.lines.append(f"  for (size_t ni = 0; ni < {n}; ++ni) {{")
    ctx.lines.append(f"    for (size_t ch = 0; ch < {c}; ++ch) {{")
    ctx.lines.append(f"      int c_start = (int)ch - {radius};")
    ctx.lines.append(f"      int c_end = (int)ch + {size - radius - 1};")
    ctx.lines.append("      if (c_start < 0) c_start = 0;")
    ctx.lines.append(f"      if (c_end >= {c}) c_end = {c} - 1;")
    ctx.lines.append(f"      for (size_t i = 0; i < {inner}; ++i) {{")
    ctx.lines.append("        float sq_sum = 0.0f;")
    ctx.lines.append("        for (int cc = c_start; cc <= c_end; ++cc) {")
    ctx.lines.append(f"          size_t idx = ((ni * {c} + (size_t)cc) * {inner}) + i;")
    if quant_mode:
        ctx.lines.append(f"          float v = ((float){x}[idx] - {zx}) * {sx:.8f}f;")
    else:
        ctx.lines.append(f"          float v = {x}[idx];")
    ctx.lines.append("          sq_sum += v * v;")
    ctx.lines.append("        }")
    ctx.lines.append(f"        size_t out_idx = ((ni * {c} + ch) * {inner}) + i;")
    ctx.lines.append(
        f"        float norm = powf({bias:.8f}f + ({alpha:.8f}f / (float){size}) * sq_sum, {beta:.8f}f);"
    )
    if quant_mode:
        ctx.lines.append(f"        float xv = ((float){x}[out_idx] - {zx}) * {sx:.8f}f;")
        ctx.lines.append("        float rv = xv / norm;")
        ctx.lines.append(f"        int q = (int)roundf(rv / {so:.8f}f) + {zo};")
        ctx.lines.append(f"        if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"        if (q > {qmax}) q = {qmax};")
        ctx.lines.append(f"        {out}[out_idx] = ({qctype})q;")
    else:
        ctx.lines.append(f"        {out}[out_idx] = {x}[out_idx] / norm;")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
