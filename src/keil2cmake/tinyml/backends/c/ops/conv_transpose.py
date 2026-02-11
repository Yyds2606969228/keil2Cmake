# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op


@register_op("ConvTranspose")
def emit_conv_transpose(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 2:
        raise ValueError("ConvTranspose expects at least 2 inputs.")
    out_name = node.outputs[0]
    out_dtype = ctx.dtype(out_name)
    x_name = node.inputs[0]
    w_name = node.inputs[1]
    b_name = node.inputs[2] if len(node.inputs) > 2 else None
    x_dtype = ctx.dtype(x_name)
    w_dtype = ctx.dtype(w_name)
    quant_mode = x_dtype in ("int8", "int16") or out_dtype in ("int8", "int16")
    if quant_mode:
        if x_dtype != out_dtype or w_dtype != out_dtype or out_dtype not in ("int8", "int16"):
            raise ValueError("ConvTranspose quantized path requires matching int8/int16 input/weight/output.")
        b_dtype = ctx.dtype(b_name) if b_name is not None else None
        if b_dtype not in (None, "float32", "int32", "int64", "int8", "int16"):
            raise ValueError("ConvTranspose quantized path does not support this bias dtype.")
        sx, zx = ctx.qparams(x_name)
        sw, zw = ctx.qparams(w_name)
        so, zo = ctx.qparams(out_name)
        qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
        qctype = "int8_t" if out_dtype == "int8" else "int16_t"
        bias_scale = sx * sw
        sb = 0.0
        zb = 0
        if b_name is not None and b_dtype in ("int8", "int16"):
            sb, zb = ctx.qparams(b_name)
    else:
        if out_dtype != "float32":
            raise ValueError("ConvTranspose currently supports float32 only.")
        if x_dtype != "float32" or w_dtype != "float32":
            raise ValueError("ConvTranspose expects float32 input/weight.")
        if b_name is not None and ctx.dtype(b_name) != "float32":
            raise ValueError("ConvTranspose bias currently supports float32 only.")

    x_shape = [int(v) for v in ctx.shape(x_name)]
    w_shape = [int(v) for v in ctx.shape(w_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    if len(x_shape) != 4 or len(w_shape) != 4 or len(out_shape) != 4:
        raise ValueError("ConvTranspose expects 4D NCHW tensors.")

    n, c_in, h, w_in = x_shape
    wc_in, c_out_per_group, k_h, k_w = w_shape
    n_out, c_out, out_h, out_w = out_shape
    if n != n_out:
        raise ValueError("ConvTranspose batch mismatch.")
    group = int(node.attrs.get("group", 1))
    if group != 1:
        raise ValueError("ConvTranspose currently supports group=1 only.")
    if wc_in != c_in:
        raise ValueError("ConvTranspose weight/input channel mismatch.")
    if c_out != c_out_per_group:
        raise ValueError("ConvTranspose output channel mismatch.")

    strides = [int(v) for v in node.attrs.get("strides", [1, 1])]
    dilations = [int(v) for v in node.attrs.get("dilations", [1, 1])]
    pads = [int(v) for v in node.attrs.get("pads", [0, 0, 0, 0])]
    if len(strides) != 2:
        raise ValueError("ConvTranspose strides length mismatch.")
    if len(dilations) != 2:
        raise ValueError("ConvTranspose dilations length mismatch.")
    if len(pads) == 2:
        pads = [pads[0], pads[1], pads[0], pads[1]]
    if len(pads) != 4:
        raise ValueError("ConvTranspose pads length mismatch.")
    output_padding = [int(v) for v in node.attrs.get("output_padding", [0, 0])]
    if len(output_padding) != 2:
        raise ValueError("ConvTranspose output_padding length mismatch.")

    stride_h, stride_w = strides
    dil_h, dil_w = dilations
    pad_h0, pad_w0, pad_h1, pad_w1 = pads
    out_pad_h, out_pad_w = output_padding

    expected_h = (h - 1) * stride_h - pad_h0 - pad_h1 + dil_h * (k_h - 1) + out_pad_h + 1
    expected_w = (w_in - 1) * stride_w - pad_w0 - pad_w1 + dil_w * (k_w - 1) + out_pad_w + 1
    if out_h != expected_h or out_w != expected_w:
        raise ValueError("ConvTranspose output shape mismatch.")

    x = ctx.map_ptr(x_name)
    w = ctx.map_ptr(w_name)
    out = ctx.map_ptr(out_name)
    b = ctx.map_ptr(b_name) if b_name is not None else None

    if quant_mode:
        total = n * c_out * out_h * out_w
        acc = ctx.next_symbol("k2c_deconv_acc")
        ctx.lines.append(f"  static float {acc}[{total}];")
        ctx.lines.append(f"  for (size_t ni = 0; ni < {n}; ++ni) {{")
        ctx.lines.append(f"    for (size_t oc = 0; oc < {c_out}; ++oc) {{")
        ctx.lines.append(f"      float b_init = 0.0f;")
        if b is not None:
            b_dtype = ctx.dtype(b_name)
            if b_dtype == "float32":
                ctx.lines.append(f"      b_init = {b}[oc];")
            elif b_dtype in ("int32", "int64"):
                ctx.lines.append(f"      b_init = ((float){b}[oc]) * {bias_scale:.8f}f;")
            else:
                ctx.lines.append(f"      b_init = ((float){b}[oc] - {zb}) * {sb:.8f}f;")
        ctx.lines.append(f"      for (size_t oh = 0; oh < {out_h}; ++oh) {{")
        ctx.lines.append(f"        for (size_t ow = 0; ow < {out_w}; ++ow) {{")
        ctx.lines.append(f"          {acc}[((ni * {c_out} + oc) * {out_h} + oh) * {out_w} + ow] = b_init;")
        ctx.lines.append("        }")
        ctx.lines.append("      }")
        ctx.lines.append("    }")
        ctx.lines.append("  }")

        ctx.lines.append(f"  for (size_t ni = 0; ni < {n}; ++ni) {{")
        ctx.lines.append(f"    for (size_t ic = 0; ic < {c_in}; ++ic) {{")
        ctx.lines.append(f"      for (size_t ih = 0; ih < {h}; ++ih) {{")
        ctx.lines.append(f"        for (size_t iw = 0; iw < {w_in}; ++iw) {{")
        ctx.lines.append(
            f"          float xv = ((float){x}[((ni * {c_in} + ic) * {h} + ih) * {w_in} + iw] - {zx}) * {sx:.8f}f;"
        )
        ctx.lines.append(f"          for (size_t kh = 0; kh < {k_h}; ++kh) {{")
        ctx.lines.append(f"            for (size_t kw = 0; kw < {k_w}; ++kw) {{")
        ctx.lines.append(f"              int oh = (int)(ih * {stride_h} + kh * {dil_h}) - {pad_h0};")
        ctx.lines.append(f"              int ow = (int)(iw * {stride_w} + kw * {dil_w}) - {pad_w0};")
        ctx.lines.append(f"              if (oh < 0 || ow < 0 || oh >= (int){out_h} || ow >= (int){out_w}) continue;")
        ctx.lines.append(f"              for (size_t oc = 0; oc < {c_out}; ++oc) {{")
        ctx.lines.append(
            f"                size_t w_idx = ((ic * {c_out} + oc) * {k_h} + kh) * {k_w} + kw;"
        )
        ctx.lines.append(
            f"                float wv = ((float){w}[w_idx] - {zw}) * {sw:.8f}f;"
        )
        ctx.lines.append(
            f"                size_t out_idx = ((ni * {c_out} + oc) * {out_h} + (size_t)oh) * {out_w} + (size_t)ow;"
        )
        ctx.lines.append(f"                {acc}[out_idx] += xv * wv;")
        ctx.lines.append("              }")
        ctx.lines.append("            }")
        ctx.lines.append("          }")
        ctx.lines.append("        }")
        ctx.lines.append("      }")
        ctx.lines.append("    }")
        ctx.lines.append("  }")

        ctx.lines.append(f"  for (size_t i = 0; i < {total}; ++i) {{")
        ctx.lines.append(f"    int q = (int)roundf({acc}[i] / {so:.8f}f) + {zo};")
        ctx.lines.append(f"    if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"    if (q > {qmax}) q = {qmax};")
        ctx.lines.append(f"    {out}[i] = ({qctype})q;")
        ctx.lines.append("  }")
        return

    ctx.lines.append(f"  for (size_t ni = 0; ni < {n}; ++ni) {{")
    ctx.lines.append(f"    for (size_t oc = 0; oc < {c_out}; ++oc) {{")
    ctx.lines.append(f"      for (size_t oh = 0; oh < {out_h}; ++oh) {{")
    ctx.lines.append(f"        for (size_t ow = 0; ow < {out_w}; ++ow) {{")
    if b is None:
        ctx.lines.append(f"          {out}[((ni * {c_out} + oc) * {out_h} + oh) * {out_w} + ow] = 0.0f;")
    else:
        ctx.lines.append(f"          {out}[((ni * {c_out} + oc) * {out_h} + oh) * {out_w} + ow] = {b}[oc];")
    ctx.lines.append("        }")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")

    ctx.lines.append(f"  for (size_t ni = 0; ni < {n}; ++ni) {{")
    ctx.lines.append(f"    for (size_t ic = 0; ic < {c_in}; ++ic) {{")
    ctx.lines.append(f"      for (size_t ih = 0; ih < {h}; ++ih) {{")
    ctx.lines.append(f"        for (size_t iw = 0; iw < {w_in}; ++iw) {{")
    ctx.lines.append(f"          float v = {x}[((ni * {c_in} + ic) * {h} + ih) * {w_in} + iw];")
    ctx.lines.append(f"          for (size_t kh = 0; kh < {k_h}; ++kh) {{")
    ctx.lines.append(f"            for (size_t kw = 0; kw < {k_w}; ++kw) {{")
    ctx.lines.append(
        f"              int oh = (int)(ih * {stride_h} + kh * {dil_h}) - {pad_h0};"
    )
    ctx.lines.append(
        f"              int ow = (int)(iw * {stride_w} + kw * {dil_w}) - {pad_w0};"
    )
    ctx.lines.append(f"              if (oh < 0 || ow < 0 || oh >= (int){out_h} || ow >= (int){out_w}) continue;")
    ctx.lines.append(f"              for (size_t oc = 0; oc < {c_out}; ++oc) {{")
    ctx.lines.append(
        f"                size_t w_idx = ((ic * {c_out} + oc) * {k_h} + kh) * {k_w} + kw;"
    )
    ctx.lines.append(
        f"                size_t out_idx = ((ni * {c_out} + oc) * {out_h} + (size_t)oh) * {out_w} + (size_t)ow;"
    )
    ctx.lines.append(f"                {out}[out_idx] += v * {w}[w_idx];")
    ctx.lines.append("              }")
    ctx.lines.append("            }")
    ctx.lines.append("          }")
    ctx.lines.append("        }")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
