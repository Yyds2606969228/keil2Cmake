# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import product
from .registry import register_op


@register_op("InstanceNormalization")
def emit_instance_normalization(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 3:
        raise ValueError("InstanceNormalization expects 3 inputs.")

    x_name, scale_name, bias_name = node.inputs
    out_name = node.outputs[0]
    x_shape = ctx.shape(x_name)
    out_shape = ctx.shape(out_name)
    if x_shape != out_shape:
        raise ValueError("InstanceNormalization input/output shape mismatch.")
    if len(x_shape) < 3:
        raise ValueError("InstanceNormalization expects rank >= 3.")

    x_dtype = ctx.dtype(x_name)
    out_dtype = ctx.dtype(out_name)
    scale_dtype = ctx.dtype(scale_name)
    bias_dtype = ctx.dtype(bias_name)
    quant_mode = x_dtype in ("int8", "int16") or out_dtype in ("int8", "int16")
    if quant_mode:
        if x_dtype != out_dtype or x_dtype not in ("int8", "int16"):
            raise ValueError("InstanceNormalization quantized path requires matching int8/int16 input/output.")
        if scale_dtype != "float32" or bias_dtype != "float32":
            raise ValueError("InstanceNormalization quantized path requires float32 scale/bias.")
        sx, zx = ctx.qparams(x_name)
        so, zo = ctx.qparams(out_name)
        qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
        qctype = "int8_t" if out_dtype == "int8" else "int16_t"
    else:
        if out_dtype != "float32" or x_dtype != "float32" or scale_dtype != "float32" or bias_dtype != "float32":
            raise ValueError("InstanceNormalization supports float32 or quantized int8/int16.")

    n = int(x_shape[0])
    c = int(x_shape[1])
    if n <= 0 or c <= 0:
        raise ValueError("InstanceNormalization requires known positive N/C.")
    inner = int(product(x_shape[2:]))
    if inner <= 0:
        raise ValueError("InstanceNormalization requires known positive spatial size.")

    scale_shape = ctx.shape(scale_name)
    bias_shape = ctx.shape(bias_name)
    if len(scale_shape) != 1 or len(bias_shape) != 1:
        raise ValueError("InstanceNormalization expects 1D scale and bias.")
    if int(scale_shape[0]) != c or int(bias_shape[0]) != c:
        raise ValueError("InstanceNormalization scale/bias channel mismatch.")

    x = ctx.map_ptr(x_name)
    scale = ctx.map_ptr(scale_name)
    bias = ctx.map_ptr(bias_name)
    out = ctx.map_ptr(out_name)
    eps = float(node.attrs.get("epsilon", 1e-5))

    ctx.lines.append(f"  for (size_t ni = 0; ni < {n}; ++ni) {{")
    ctx.lines.append(f"    for (size_t ch = 0; ch < {c}; ++ch) {{")
    ctx.lines.append("      float mean_v = 0.0f;")
    ctx.lines.append(f"      for (size_t i = 0; i < {inner}; ++i) {{")
    ctx.lines.append(f"        size_t idx = ((ni * {c} + ch) * {inner}) + i;")
    if quant_mode:
        ctx.lines.append(f"        mean_v += ((float){x}[idx] - {zx}) * {sx:.8f}f;")
    else:
        ctx.lines.append(f"        mean_v += {x}[idx];")
    ctx.lines.append("      }")
    ctx.lines.append(f"      mean_v /= (float){inner};")
    ctx.lines.append("      float var_v = 0.0f;")
    ctx.lines.append(f"      for (size_t i = 0; i < {inner}; ++i) {{")
    ctx.lines.append(f"        size_t idx = ((ni * {c} + ch) * {inner}) + i;")
    if quant_mode:
        ctx.lines.append(f"        float xv = ((float){x}[idx] - {zx}) * {sx:.8f}f;")
        ctx.lines.append("        float dv = xv - mean_v;")
    else:
        ctx.lines.append(f"        float dv = {x}[idx] - mean_v;")
    ctx.lines.append("        var_v += dv * dv;")
    ctx.lines.append("      }")
    ctx.lines.append(f"      var_v /= (float){inner};")
    ctx.lines.append(f"      float inv_std = 1.0f / sqrtf(var_v + {eps:.8f}f);")
    ctx.lines.append(f"      float s = {scale}[ch];")
    ctx.lines.append(f"      float b = {bias}[ch];")
    ctx.lines.append(f"      for (size_t i = 0; i < {inner}; ++i) {{")
    ctx.lines.append(f"        size_t idx = ((ni * {c} + ch) * {inner}) + i;")
    if quant_mode:
        ctx.lines.append(f"        float xv = ((float){x}[idx] - {zx}) * {sx:.8f}f;")
        ctx.lines.append("        float rv = (xv - mean_v) * inv_std * s + b;")
        ctx.lines.append(f"        int q = (int)roundf(rv / {so:.8f}f) + {zo};")
        ctx.lines.append(f"        if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"        if (q > {qmax}) q = {qmax};")
        ctx.lines.append(f"        {out}[idx] = ({qctype})q;")
    else:
        ctx.lines.append(f"        {out}[idx] = ({x}[idx] - mean_v) * inv_std * s + b;")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
