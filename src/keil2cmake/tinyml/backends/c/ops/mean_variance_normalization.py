# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import normalize_axis
from .registry import register_op


def _axes_023(attrs: dict, rank: int) -> bool:
    axes = attrs.get("axes", [0, 2, 3])
    if not isinstance(axes, (list, tuple)):
        return False
    if len(axes) != 3:
        return False
    norm = {normalize_axis(int(v), rank) for v in axes}
    return norm == {0, 2, 3}


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
    if len(x_shape) != 4:
        raise ValueError("MeanVarianceNormalization currently supports 4D NCHW only.")
    if not _axes_023(node.attrs, len(x_shape)):
        raise ValueError("MeanVarianceNormalization currently supports axes=[0,2,3] only.")

    n, c, h, w_in = x_shape
    nhw = int(n) * int(h) * int(w_in)
    x = ctx.map_ptr(x_name)
    out = ctx.map_ptr(out_name)

    ctx.lines.append(f"  for (size_t ch = 0; ch < {c}; ++ch) {{")
    ctx.lines.append("    float mean = 0.0f;")
    ctx.lines.append(f"    for (size_t ni = 0; ni < {n}; ++ni) {{")
    ctx.lines.append(f"      for (size_t i = 0; i < {h} * {w_in}; ++i) {{")
    ctx.lines.append(f"        size_t idx = ((ni * {c} + ch) * {h} * {w_in}) + i;")
    if quant_mode:
        ctx.lines.append(f"        mean += ((float){x}[idx] - {zx}) * {sx:.8f}f;")
    else:
        ctx.lines.append(f"        mean += {x}[idx];")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append(f"    mean /= (float){nhw};")
    ctx.lines.append("    float var = 0.0f;")
    ctx.lines.append(f"    for (size_t ni = 0; ni < {n}; ++ni) {{")
    ctx.lines.append(f"      for (size_t i = 0; i < {h} * {w_in}; ++i) {{")
    ctx.lines.append(f"        size_t idx = ((ni * {c} + ch) * {h} * {w_in}) + i;")
    if quant_mode:
        ctx.lines.append(f"        float xv = ((float){x}[idx] - {zx}) * {sx:.8f}f;")
        ctx.lines.append("        float dv = xv - mean;")
    else:
        ctx.lines.append(f"        float dv = {x}[idx] - mean;")
    ctx.lines.append("        var += dv * dv;")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append(f"    var /= (float){nhw};")
    ctx.lines.append("    float inv_std = 1.0f / sqrtf(var + 1e-12f);")
    ctx.lines.append(f"    for (size_t ni = 0; ni < {n}; ++ni) {{")
    ctx.lines.append(f"      for (size_t i = 0; i < {h} * {w_in}; ++i) {{")
    ctx.lines.append(f"        size_t idx = ((ni * {c} + ch) * {h} * {w_in}) + i;")
    if quant_mode:
        ctx.lines.append(f"        float xv = ((float){x}[idx] - {zx}) * {sx:.8f}f;")
        ctx.lines.append("        float rv = (xv - mean) * inv_std;")
        ctx.lines.append(f"        int q = (int)roundf(rv / {so:.8f}f) + {zo};")
        ctx.lines.append(f"        if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"        if (q > {qmax}) q = {qmax};")
        ctx.lines.append(f"        {out}[idx] = ({qctype})q;")
    else:
        ctx.lines.append(f"        {out}[idx] = ({x}[idx] - mean) * inv_std;")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
