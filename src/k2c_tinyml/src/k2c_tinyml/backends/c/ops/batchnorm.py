# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op
from ....operators.utils import emit_op_batch_norm, product


@register_op("BatchNormalization")
def emit_batchnorm(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 5:
        raise ValueError("BatchNormalization expects 5 inputs.")
    out_tensor = node.outputs[0]
    out = ctx.map_ptr(out_tensor)
    x_name, scale_name, bias_name, mean_name, var_name = node.inputs
    x_dtype = ctx.dtype(x_name)
    out_dtype = ctx.dtype(out_tensor)
    quant_mode = x_dtype in ("int8", "int16") or out_dtype in ("int8", "int16")
    if quant_mode:
        if x_dtype != out_dtype or x_dtype not in ("int8", "int16"):
            raise ValueError("BatchNormalization quantized path requires matching int8/int16 input/output.")
        if (
            ctx.dtype(scale_name) != "float32"
            or ctx.dtype(bias_name) != "float32"
            or ctx.dtype(mean_name) != "float32"
            or ctx.dtype(var_name) != "float32"
        ):
            raise ValueError("BatchNormalization quantized path requires float32 scale/bias/mean/var.")
        x_shape = [int(v) for v in ctx.shape(x_name)]
        if len(x_shape) < 2:
            raise ValueError("BatchNormalization expects rank >= 2 input.")
        n = int(x_shape[0])
        c = int(x_shape[1])
        inner = int(product(x_shape[2:])) if len(x_shape) > 2 else 1
        sx, zx = ctx.qparams(x_name)
        so, zo = ctx.qparams(out_tensor)
        qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
        qctype = "int8_t" if out_dtype == "int8" else "int16_t"
        x = ctx.map_ptr(x_name)
        scale = ctx.map_ptr(scale_name)
        bias = ctx.map_ptr(bias_name)
        mean = ctx.map_ptr(mean_name)
        var = ctx.map_ptr(var_name)
        epsilon = float(node.attrs.get("epsilon", 1e-5))

        ctx.lines.append(f"  for (size_t ch = 0; ch < {c}; ++ch) {{")
        ctx.lines.append(f"    float scale_v = {scale}[ch];")
        ctx.lines.append(f"    float bias_v = {bias}[ch];")
        ctx.lines.append(f"    float mean_v = {mean}[ch];")
        ctx.lines.append(f"    float var_v = {var}[ch];")
        ctx.lines.append(f"    float inv_std = 1.0f / sqrtf(var_v + {epsilon:.8f}f);")
        ctx.lines.append(f"    for (size_t ni = 0; ni < {n}; ++ni) {{")
        ctx.lines.append(f"      for (size_t i = 0; i < {inner}; ++i) {{")
        ctx.lines.append(f"        size_t idx = ((ni * {c} + ch) * {inner}) + i;")
        ctx.lines.append(f"        float xv = ((float){x}[idx] - {zx}) * {sx:.8f}f;")
        ctx.lines.append("        float rv = ((xv - mean_v) * inv_std) * scale_v + bias_v;")
        ctx.lines.append(f"        int q = (int)roundf(rv / {so:.8f}f) + {zo};")
        ctx.lines.append(f"        if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"        if (q > {qmax}) q = {qmax};")
        ctx.lines.append(f"        {out}[idx] = ({qctype})q;")
        ctx.lines.append("      }")
        ctx.lines.append("    }")
        ctx.lines.append("  }")
        return

    if (
        x_dtype != "float32"
        or out_dtype != "float32"
        or ctx.dtype(scale_name) != "float32"
        or ctx.dtype(bias_name) != "float32"
        or ctx.dtype(mean_name) != "float32"
        or ctx.dtype(var_name) != "float32"
    ):
        raise ValueError("BatchNormalization float32 path requires float32 tensors.")

    x = ctx.map_ptr(x_name)
    scale = ctx.map_ptr(scale_name)
    bias = ctx.map_ptr(bias_name)
    mean = ctx.map_ptr(mean_name)
    var = ctx.map_ptr(var_name)
    x_shape = ctx.shape(x_name)
    epsilon = float(node.attrs.get("epsilon", 1e-5))
    mode = "test"
    momentum_val = None
    # Converter currently targets inference graph generation.
    # Keep BatchNormalization in inference semantics for all opsets.
    emit_op_batch_norm(
        ctx.lines,
        out,
        x,
        scale,
        bias,
        mean,
        var,
        x_shape,
        epsilon,
        mode,
        momentum_val,
    )

