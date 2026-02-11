# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op


@register_op("SpaceToDepth")
def emit_space_to_depth(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("SpaceToDepth expects 1 input.")
    x_name = node.inputs[0]
    out_name = node.outputs[0]
    out_dtype = ctx.dtype(out_name)
    if ctx.dtype(x_name) != out_dtype:
        raise ValueError("SpaceToDepth requires matching input/output dtypes.")
    if out_dtype in ("int8", "int16"):
        si, zi = ctx.qparams(x_name)
        qo = ctx.qparams_optional(out_name)
        if qo is not None:
            so, zo = qo
            if abs(si - so) > 1e-12 or zi != zo:
                raise ValueError("Quantized SpaceToDepth requires same input/output qparams.")

    x_shape = ctx.shape(x_name)
    out_shape = ctx.shape(out_name)
    if len(x_shape) != 4 or len(out_shape) != 4:
        raise ValueError("SpaceToDepth expects 4D NCHW tensors.")
    n, c, h, w_in = x_shape
    n_out, c_out, out_h, out_w = out_shape
    block = int(node.attrs.get("blocksize", 0))
    if block <= 0:
        raise ValueError("SpaceToDepth requires positive blocksize.")
    if h % block != 0 or w_in % block != 0:
        raise ValueError("SpaceToDepth requires H/W divisible by blocksize.")
    if n_out != n or c_out != c * block * block or out_h != h // block or out_w != w_in // block:
        raise ValueError("SpaceToDepth output shape mismatch.")

    x = ctx.map_ptr(x_name)
    out = ctx.map_ptr(out_name)
    ctx.lines.append(f"  for (size_t n_i = 0; n_i < {n}; ++n_i) {{")
    ctx.lines.append(f"    for (size_t c_i = 0; c_i < {c}; ++c_i) {{")
    ctx.lines.append(f"      for (size_t oh = 0; oh < {out_h}; ++oh) {{")
    ctx.lines.append(f"        for (size_t ow = 0; ow < {out_w}; ++ow) {{")
    ctx.lines.append(f"          for (size_t bh = 0; bh < {block}; ++bh) {{")
    ctx.lines.append(f"            for (size_t bw = 0; bw < {block}; ++bw) {{")
    ctx.lines.append(f"              size_t oc = c_i * {block * block} + bh * {block} + bw;")
    ctx.lines.append(f"              size_t ih = oh * {block} + bh;")
    ctx.lines.append(f"              size_t iw = ow * {block} + bw;")
    ctx.lines.append(
        f"              size_t out_idx = ((n_i * {c_out} + oc) * {out_h} + oh) * {out_w} + ow;"
    )
    ctx.lines.append(
        f"              size_t in_idx = ((n_i * {c} + c_i) * {h} + ih) * {w_in} + iw;"
    )
    ctx.lines.append(f"              {out}[out_idx] = {x}[in_idx];")
    ctx.lines.append("            }")
    ctx.lines.append("          }")
    ctx.lines.append("        }")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
