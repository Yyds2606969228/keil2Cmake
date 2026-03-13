# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import emit_op_conv2d
from .registry import register_op


@register_op("Conv")
def emit_conv(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 2:
        raise ValueError("Conv expects at least 2 inputs.")
    out_tensor = node.outputs[0]
    out_dtype = ctx.dtype(out_tensor)
    out = ctx.map_ptr(out_tensor)
    x_name = node.inputs[0]
    w_name = node.inputs[1]
    b_name = node.inputs[2] if len(node.inputs) > 2 else None
    x = ctx.map_ptr(x_name)
    w = ctx.map_ptr(w_name)
    b = ctx.map_ptr(b_name) if b_name else None
    x_shape = ctx.shape(x_name)
    w_shape = ctx.shape(w_name)
    out_shape = ctx.shape(out_tensor)
    strides = list(node.attrs.get("strides", [1, 1]))
    pads = list(node.attrs.get("pads", [0, 0, 0, 0]))
    dilations = list(node.attrs.get("dilations", [1, 1]))
    groups = int(node.attrs.get("group", 1))

    if len(x_shape) != 4 or len(w_shape) != 4 or len(out_shape) != 4:
        raise ValueError("Conv expects 4D tensors (NCHW).")
    n, c_in, h, w_in = x_shape
    m, c_per_g, k_h, k_w = w_shape
    n_out, c_out, out_h, out_w = out_shape
    if n != n_out:
        raise ValueError("Conv batch dimension mismatch.")
    if groups <= 0:
        raise ValueError("Conv group must be positive.")
    if c_out != m or c_in != c_per_g * groups:
        raise ValueError("Conv channel mismatch.")
    if m % groups != 0:
        raise ValueError("Conv output channels must be divisible by groups.")

    if len(strides) != 2:
        raise ValueError("Conv strides length mismatch.")
    if len(dilations) != 2:
        raise ValueError("Conv dilations length mismatch.")
    if len(pads) == 2:
        pads = [pads[0], pads[1], pads[0], pads[1]]
    if len(pads) != 4:
        raise ValueError("Conv pads length mismatch.")
    stride_h, stride_w = int(strides[0]), int(strides[1])
    pad_h0, pad_w0, pad_h1, pad_w1 = [int(v) for v in pads]
    dil_h, dil_w = int(dilations[0]), int(dilations[1])
    _ = pad_h1, pad_w1
    oc_per_group = m // groups

    if out_dtype in ("int8", "int16"):
        if ctx.dtype(x_name) != out_dtype or ctx.dtype(w_name) != out_dtype:
            raise ValueError("Quantized Conv requires matching dtypes for input and weight.")
        b_dtype = ctx.dtype(b_name) if b_name else None
        if b_dtype not in (None, "float32", "int32", "int64", "int8", "int16"):
            raise ValueError("Quantized Conv bias dtype is not supported.")

        sx, zx = ctx.qparams(x_name)
        sw, zw = ctx.qparams(w_name)
        so, zo = ctx.qparams(out_tensor)
        qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
        qctype = "int8_t" if out_dtype == "int8" else "int16_t"
        bias_scale = sx * sw

        ctx.lines.append(f"  for (size_t ni = 0; ni < {n}; ++ni) {{")
        ctx.lines.append(f"    for (size_t oc = 0; oc < {m}; ++oc) {{")
        ctx.lines.append(f"      for (size_t oh = 0; oh < {out_h}; ++oh) {{")
        ctx.lines.append(f"        for (size_t ow = 0; ow < {out_w}; ++ow) {{")
        if b is not None:
            if b_dtype == "float32":
                ctx.lines.append(f"          float sum = {b}[oc];")
            elif b_dtype in ("int32", "int64"):
                ctx.lines.append(f"          float sum = ((float){b}[oc]) * {bias_scale:.8f}f;")
            else:
                sb, zb = ctx.qparams(b_name)
                ctx.lines.append(f"          float sum = ((float){b}[oc] - {zb}) * {sb:.8f}f;")
        else:
            ctx.lines.append("          float sum = 0.0f;")
        ctx.lines.append(f"          size_t g = oc / {oc_per_group};")
        ctx.lines.append(f"          size_t ic_begin = g * {c_per_g};")
        ctx.lines.append(f"          for (size_t ic_local = 0; ic_local < {c_per_g}; ++ic_local) {{")
        ctx.lines.append("            size_t ic = ic_begin + ic_local;")
        ctx.lines.append(f"            for (size_t kh = 0; kh < {k_h}; ++kh) {{")
        ctx.lines.append(f"              for (size_t kw = 0; kw < {k_w}; ++kw) {{")
        ctx.lines.append(
            f"                int in_h = (int)(oh * {stride_h} + kh * {dil_h}) - {pad_h0};"
        )
        ctx.lines.append(
            f"                int in_w = (int)(ow * {stride_w} + kw * {dil_w}) - {pad_w0};"
        )
        ctx.lines.append(
            "                if (in_h >= 0 && in_h < (int)"
            + f"{h}"
            + " && in_w >= 0 && in_w < (int)"
            + f"{w_in}"
            + ") {"
        )
        ctx.lines.append(
            f"                  size_t in_idx = ((ni * {c_in} + ic) * {h} + (size_t)in_h) * {w_in} + (size_t)in_w;"
        )
        ctx.lines.append(
            f"                  size_t w_idx = ((oc * {c_per_g} + ic_local) * {k_h} + kh) * {k_w} + kw;"
        )
        ctx.lines.append(
            f"                  float rx = ((float){x}[in_idx] - {zx}) * {sx:.8f}f;"
        )
        ctx.lines.append(
            f"                  float rw = ((float){w}[w_idx] - {zw}) * {sw:.8f}f;"
        )
        ctx.lines.append("                  sum += rx * rw;")
        ctx.lines.append("                }")
        ctx.lines.append("              }")
        ctx.lines.append("            }")
        ctx.lines.append("          }")
        ctx.lines.append(f"          int q = (int)roundf(sum / {so:.8f}f) + {zo};")
        ctx.lines.append(f"          if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"          if (q > {qmax}) q = {qmax};")
        ctx.lines.append(
            f"          {out}[((ni * {c_out} + oc) * {out_h} + oh) * {out_w} + ow] = ({qctype})q;"
        )
        ctx.lines.append("        }")
        ctx.lines.append("      }")
        ctx.lines.append("    }")
        ctx.lines.append("  }")
        return

    if out_dtype != "float32":
        raise ValueError("Conv supports float32 or quantized int8/int16 only.")
    emit_op_conv2d(
        ctx.lines,
        out,
        x,
        w,
        b,
        x_shape,
        w_shape,
        out_shape,
        [stride_h, stride_w],
        [pad_h0, pad_w0, pad_h1, pad_w1],
        [dil_h, dil_w],
        groups,
    )
