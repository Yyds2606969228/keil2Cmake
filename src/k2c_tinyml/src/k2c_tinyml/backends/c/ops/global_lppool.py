# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import product
from .registry import register_op


@register_op("GlobalLpPool")
def emit_global_lppool(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("GlobalLpPool expects 1 input.")
    x_name = node.inputs[0]
    out_name = node.outputs[0]
    x_dtype = ctx.dtype(x_name)
    out_dtype = ctx.dtype(out_name)
    quant_mode = x_dtype in ("int8", "int16") or out_dtype in ("int8", "int16")
    if quant_mode:
        if x_dtype != out_dtype or x_dtype not in ("int8", "int16"):
            raise ValueError("GlobalLpPool quantized path requires matching int8/int16 input/output.")
        sx, zx = ctx.qparams(x_name)
        so, zo = ctx.qparams(out_name)
        qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
        qctype = "int8_t" if out_dtype == "int8" else "int16_t"
    elif x_dtype != "float32" or out_dtype != "float32":
        raise ValueError("GlobalLpPool supports float32 or quantized int8/int16.")

    x_shape = [int(v) for v in ctx.shape(x_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    if len(x_shape) < 3 or len(out_shape) != len(x_shape):
        raise ValueError("GlobalLpPool expects rank >= 3 and matching ranks.")

    n, c = int(x_shape[0]), int(x_shape[1])
    n_out, c_out = int(out_shape[0]), int(out_shape[1])
    expected_out = [n, c] + [1] * (len(x_shape) - 2)
    if n != n_out:
        raise ValueError("GlobalLpPool batch dimension mismatch.")
    if c != c_out or [int(v) for v in out_shape] != expected_out:
        raise ValueError("GlobalLpPool output shape mismatch.")
    spatial = int(product(x_shape[2:]))
    out_inner = int(product(out_shape[2:]))

    p = int(node.attrs.get("p", 2))
    if p <= 0:
        raise ValueError("GlobalLpPool p must be positive.")

    x = ctx.map_ptr(x_name)
    out = ctx.map_ptr(out_name)
    ctx.lines.append(f"  for (size_t ni = 0; ni < {n}; ++ni) {{")
    ctx.lines.append(f"    for (size_t ch = 0; ch < {c}; ++ch) {{")
    ctx.lines.append("      float acc = 0.0f;")
    ctx.lines.append(f"      for (size_t i = 0; i < {spatial}; ++i) {{")
    ctx.lines.append(f"        size_t in_idx = ((ni * {c} + ch) * {spatial}) + i;")
    if quant_mode:
        ctx.lines.append(f"        float xv = ((float){x}[in_idx] - {zx}) * {sx:.8f}f;")
    if p == 1:
        if quant_mode:
            ctx.lines.append("        acc += fabsf(xv);")
        else:
            ctx.lines.append(f"        acc += fabsf({x}[in_idx]);")
    elif p == 2:
        if quant_mode:
            ctx.lines.append("        float av = fabsf(xv);")
        else:
            ctx.lines.append(f"        float av = fabsf({x}[in_idx]);")
        ctx.lines.append("        acc += av * av;")
    else:
        if quant_mode:
            ctx.lines.append(f"        acc += powf(fabsf(xv), {float(p):.8f}f);")
        else:
            ctx.lines.append(f"        acc += powf(fabsf({x}[in_idx]), {float(p):.8f}f);")
    ctx.lines.append("      }")
    if p == 1:
        ctx.lines.append("      float v = acc;")
    elif p == 2:
        ctx.lines.append("      float v = sqrtf(acc);")
    else:
        ctx.lines.append(f"      float v = powf(acc, {1.0 / float(p):.8f}f);")
    ctx.lines.append(f"      size_t out_idx = ((ni * {c_out} + ch) * {out_inner});")
    if quant_mode:
        ctx.lines.append(f"      int q = (int)roundf(v / {so:.8f}f) + {zo};")
        ctx.lines.append(f"      if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"      if (q > {qmax}) q = {qmax};")
        ctx.lines.append(f"      {out}[out_idx] = ({qctype})q;")
    else:
        ctx.lines.append(f"      {out}[out_idx] = v;")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
