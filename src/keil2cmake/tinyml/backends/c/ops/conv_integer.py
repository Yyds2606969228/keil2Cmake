# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import get_const_ints
from .registry import register_op


def _zero_point_scalar(ctx: EmitContext, name: str | None) -> int:
    if not name:
        return 0
    vals = get_const_ints(ctx.model, name)
    if len(vals) != 1:
        raise ValueError("ConvInteger zero_point must be scalar constant.")
    return int(vals[0])


@register_op("ConvInteger")
def emit_conv_integer(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 2:
        raise ValueError("ConvInteger expects at least 2 inputs.")
    if len(node.inputs) > 4:
        raise ValueError("ConvInteger supports at most 4 inputs.")
    x_name = node.inputs[0]
    w_name = node.inputs[1]
    out_name = node.outputs[0]

    x_dtype = ctx.dtype(x_name)
    w_dtype = ctx.dtype(w_name)
    out_dtype = ctx.dtype(out_name)
    if x_dtype not in ("int8", "int16") or w_dtype not in ("int8", "int16"):
        raise ValueError("ConvInteger currently supports int8/int16 inputs only.")
    if out_dtype not in ("int32", "int64"):
        raise ValueError("ConvInteger output must be int32/int64.")

    x_shape = ctx.shape(x_name)
    w_shape = ctx.shape(w_name)
    out_shape = ctx.shape(out_name)
    if len(x_shape) != 4 or len(w_shape) != 4 or len(out_shape) != 4:
        raise ValueError("ConvInteger expects 4D tensors (NCHW).")

    n, c_in, h, w_in = [int(v) for v in x_shape]
    m, c_per_g, k_h, k_w = [int(v) for v in w_shape]
    n_out, c_out, out_h, out_w = [int(v) for v in out_shape]
    if n != n_out:
        raise ValueError("ConvInteger batch dimension mismatch.")
    if c_out != m:
        raise ValueError("ConvInteger output channel mismatch.")

    groups = int(node.attrs.get("group", 1))
    if groups <= 0:
        raise ValueError("ConvInteger group must be positive.")
    if c_in != c_per_g * groups:
        raise ValueError("ConvInteger input channel mismatch.")
    if m % groups != 0:
        raise ValueError("ConvInteger output channels must be divisible by group.")
    oc_per_group = m // groups

    strides = list(node.attrs.get("strides", [1, 1]))
    if len(strides) != 2:
        raise ValueError("ConvInteger strides length mismatch.")
    dilations = list(node.attrs.get("dilations", [1, 1]))
    if len(dilations) != 2:
        raise ValueError("ConvInteger dilations length mismatch.")
    pads = list(node.attrs.get("pads", [0, 0, 0, 0]))
    if len(pads) == 2:
        pads = [pads[0], pads[1], pads[0], pads[1]]
    if len(pads) != 4:
        raise ValueError("ConvInteger pads length mismatch.")

    stride_h, stride_w = int(strides[0]), int(strides[1])
    dil_h, dil_w = int(dilations[0]), int(dilations[1])
    pad_h0, pad_w0 = int(pads[0]), int(pads[1])
    x_zero = _zero_point_scalar(ctx, node.inputs[2] if len(node.inputs) >= 3 and node.inputs[2] else None)
    w_zero = _zero_point_scalar(ctx, node.inputs[3] if len(node.inputs) >= 4 and node.inputs[3] else None)

    x = ctx.map_ptr(x_name)
    w = ctx.map_ptr(w_name)
    out = ctx.map_ptr(out_name)

    ctx.lines.append(f"  for (size_t ni = 0; ni < {n}; ++ni) {{")
    ctx.lines.append(f"    for (size_t oc = 0; oc < {m}; ++oc) {{")
    ctx.lines.append(f"      for (size_t oh = 0; oh < {out_h}; ++oh) {{")
    ctx.lines.append(f"        for (size_t ow = 0; ow < {out_w}; ++ow) {{")
    ctx.lines.append("          int64_t acc = 0;")
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
    ctx.lines.append(f"                  int64_t xv = (int64_t){x}[in_idx] - {x_zero};")
    ctx.lines.append(f"                  int64_t wv = (int64_t){w}[w_idx] - {w_zero};")
    ctx.lines.append("                  acc += xv * wv;")
    ctx.lines.append("                }")
    ctx.lines.append("              }")
    ctx.lines.append("            }")
    ctx.lines.append("          }")
    if out_dtype == "int32":
        ctx.lines.append("          if (acc < -2147483648LL) acc = -2147483648LL;")
        ctx.lines.append("          if (acc > 2147483647LL) acc = 2147483647LL;")
        ctx.lines.append(
            f"          {out}[((ni * {c_out} + oc) * {out_h} + oh) * {out_w} + ow] = (int32_t)acc;"
        )
    else:
        ctx.lines.append(
            f"          {out}[((ni * {c_out} + oc) * {out_h} + oh) * {out_w} + ow] = (int64_t)acc;"
        )
    ctx.lines.append("        }")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
