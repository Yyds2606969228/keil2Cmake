# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import get_const_ints, get_const_scalar
from .registry import register_op


def _const_float_values(ctx: EmitContext, name: str) -> list[float]:
    tensor = ctx.model.tensors.get(name)
    if tensor is None or tensor.data is None or len(tensor.data) == 0:
        raise ValueError(f"QLinearConv constant '{name}' is missing.")
    return [float(v) for v in tensor.data]


def _scalar_float(ctx: EmitContext, name: str) -> float:
    return float(get_const_scalar(ctx.model, name))


def _scalar_int(ctx: EmitContext, name: str) -> int:
    vals = get_const_ints(ctx.model, name)
    if len(vals) != 1:
        raise ValueError("QLinearConv integer parameter must be scalar constant.")
    return int(vals[0])


@register_op("QLinearConv")
def emit_qlinear_conv(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 8:
        raise ValueError(
            "QLinearConv expects at least 8 inputs: x,x_scale,x_zero,w,w_scale,w_zero,y_scale,y_zero."
        )
    if len(node.inputs) > 9:
        raise ValueError("QLinearConv supports at most one optional bias input.")
    x_name = node.inputs[0]
    w_name = node.inputs[3]
    out_name = node.outputs[0]
    b_name = node.inputs[8] if len(node.inputs) >= 9 and node.inputs[8] else None

    x_dtype = ctx.dtype(x_name)
    w_dtype = ctx.dtype(w_name)
    out_dtype = ctx.dtype(out_name)
    if x_dtype not in ("int8", "int16") or w_dtype not in ("int8", "int16"):
        raise ValueError("QLinearConv currently supports int8/int16 input and weight.")
    if out_dtype not in ("int8", "int16"):
        raise ValueError("QLinearConv output must be int8/int16.")

    x_scale = _scalar_float(ctx, node.inputs[1])
    x_zero = _scalar_int(ctx, node.inputs[2])
    y_scale = _scalar_float(ctx, node.inputs[6])
    y_zero = _scalar_int(ctx, node.inputs[7])
    if y_scale == 0.0:
        raise ValueError("QLinearConv y_scale must be non-zero.")

    x_shape = ctx.shape(x_name)
    w_shape = ctx.shape(w_name)
    out_shape = ctx.shape(out_name)
    if len(x_shape) != 4 or len(w_shape) != 4 or len(out_shape) != 4:
        raise ValueError("QLinearConv expects 4D tensors (NCHW).")
    n, c_in, h, w_in = [int(v) for v in x_shape]
    m, c_per_g, k_h, k_w = [int(v) for v in w_shape]
    n_out, c_out, out_h, out_w = [int(v) for v in out_shape]
    if n != n_out:
        raise ValueError("QLinearConv batch dimension mismatch.")
    if c_out != m:
        raise ValueError("QLinearConv output channel mismatch.")

    groups = int(node.attrs.get("group", 1))
    if groups <= 0:
        raise ValueError("QLinearConv group must be positive.")
    if c_in != c_per_g * groups:
        raise ValueError("QLinearConv input channel mismatch.")
    if m % groups != 0:
        raise ValueError("QLinearConv output channels must be divisible by group.")
    oc_per_group = m // groups

    strides = list(node.attrs.get("strides", [1, 1]))
    if len(strides) != 2:
        raise ValueError("QLinearConv strides length mismatch.")
    dilations = list(node.attrs.get("dilations", [1, 1]))
    if len(dilations) != 2:
        raise ValueError("QLinearConv dilations length mismatch.")
    pads = list(node.attrs.get("pads", [0, 0, 0, 0]))
    if len(pads) == 2:
        pads = [pads[0], pads[1], pads[0], pads[1]]
    if len(pads) != 4:
        raise ValueError("QLinearConv pads length mismatch.")
    stride_h, stride_w = int(strides[0]), int(strides[1])
    dil_h, dil_w = int(dilations[0]), int(dilations[1])
    pad_h0, pad_w0 = int(pads[0]), int(pads[1])

    w_scales = _const_float_values(ctx, node.inputs[4])
    w_zeros = get_const_ints(ctx.model, node.inputs[5])
    if len(w_scales) not in (1, m):
        raise ValueError("QLinearConv w_scale must be scalar or per-output-channel.")
    if len(w_zeros) not in (1, m):
        raise ValueError("QLinearConv w_zero_point must be scalar or per-output-channel.")

    w_scale_sym = None
    if len(w_scales) == m:
        w_scale_sym = ctx.next_symbol("k2c_qconv_w_scale")
        vals = ", ".join(f"{v:.12g}f" for v in w_scales)
        ctx.lines.append(f"  static const float {w_scale_sym}[{m}] = {{ {vals} }};")
    w_zero_sym = None
    if len(w_zeros) == m:
        w_zero_sym = ctx.next_symbol("k2c_qconv_w_zero")
        vals = ", ".join(str(int(v)) for v in w_zeros)
        ctx.lines.append(f"  static const int32_t {w_zero_sym}[{m}] = {{ {vals} }};")

    b_dtype = ctx.dtype(b_name) if b_name else None
    if b_dtype not in (None, "float32", "int32", "int64"):
        raise ValueError("QLinearConv bias supports float32/int32/int64 only.")

    x = ctx.map_ptr(x_name)
    w = ctx.map_ptr(w_name)
    b = ctx.map_ptr(b_name) if b_name else None
    out = ctx.map_ptr(out_name)

    if out_dtype == "int8":
        qmin, qmax, out_ctype = -128, 127, "int8_t"
    else:
        qmin, qmax, out_ctype = -32768, 32767, "int16_t"

    if len(w_scales) == 1:
        w_scale_expr = f"{float(w_scales[0]):.12g}f"
    else:
        w_scale_expr = f"{w_scale_sym}[oc]"
    if len(w_zeros) == 1:
        w_zero_expr = str(int(w_zeros[0]))
    else:
        w_zero_expr = f"{w_zero_sym}[oc]"

    ctx.lines.append(f"  for (size_t ni = 0; ni < {n}; ++ni) {{")
    ctx.lines.append(f"    for (size_t oc = 0; oc < {m}; ++oc) {{")
    ctx.lines.append(f"      for (size_t oh = 0; oh < {out_h}; ++oh) {{")
    ctx.lines.append(f"        for (size_t ow = 0; ow < {out_w}; ++ow) {{")
    if b is not None:
        if b_dtype == "float32":
            ctx.lines.append(f"          float sum = {b}[oc];")
        else:
            ctx.lines.append(f"          float sum = ((float){b}[oc]) * {x_scale:.12g}f * {w_scale_expr};")
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
        f"                  float rx = ((float){x}[in_idx] - {x_zero}) * {x_scale:.12g}f;"
    )
    ctx.lines.append(
        f"                  float rw = ((float){w}[w_idx] - (float){w_zero_expr}) * {w_scale_expr};"
    )
    ctx.lines.append("                  sum += rx * rw;")
    ctx.lines.append("                }")
    ctx.lines.append("              }")
    ctx.lines.append("            }")
    ctx.lines.append("          }")
    ctx.lines.append(f"          int q = (int)roundf(sum / {y_scale:.12g}f) + {y_zero};")
    ctx.lines.append(f"          if (q < {qmin}) q = {qmin};")
    ctx.lines.append(f"          if (q > {qmax}) q = {qmax};")
    ctx.lines.append(
        f"          {out}[((ni * {c_out} + oc) * {out_h} + oh) * {out_w} + ow] = ({out_ctype})q;"
    )
    ctx.lines.append("        }")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
