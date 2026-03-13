# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op


@register_op("DepthToSpace")
def emit_depth_to_space(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("DepthToSpace expects 1 input.")
    x_name = node.inputs[0]
    out_name = node.outputs[0]
    out_dtype = ctx.dtype(out_name)
    if ctx.dtype(x_name) != out_dtype:
        raise ValueError("DepthToSpace requires matching input/output dtypes.")
    if out_dtype in ("int8", "int16"):
        si, zi = ctx.qparams(x_name)
        qo = ctx.qparams_optional(out_name)
        if qo is not None:
            so, zo = qo
            if abs(si - so) > 1e-12 or zi != zo:
                raise ValueError("Quantized DepthToSpace requires same input/output qparams.")

    x_shape = ctx.shape(x_name)
    out_shape = ctx.shape(out_name)
    if len(x_shape) != 4 or len(out_shape) != 4:
        raise ValueError("DepthToSpace expects 4D NCHW tensors.")
    n, c, h, w_in = x_shape
    n_out, out_c, out_h, out_w = out_shape
    block = int(node.attrs.get("blocksize", 0))
    if block <= 0:
        raise ValueError("DepthToSpace requires positive blocksize.")
    if c % (block * block) != 0:
        raise ValueError("DepthToSpace requires C divisible by blocksize^2.")
    expected_out_c = c // (block * block)
    if n_out != n or out_c != expected_out_c or out_h != h * block or out_w != w_in * block:
        raise ValueError("DepthToSpace output shape mismatch.")

    mode = node.attrs.get("mode", "DCR")
    if isinstance(mode, bytes):
        mode = mode.decode("utf-8", errors="ignore")
    mode = str(mode).upper()
    if mode not in ("DCR", "CRD"):
        raise ValueError("DepthToSpace mode must be DCR or CRD.")

    x = ctx.map_ptr(x_name)
    out = ctx.map_ptr(out_name)
    ctx.lines.append(f"  for (size_t n_i = 0; n_i < {n}; ++n_i) {{")
    ctx.lines.append(f"    for (size_t oc = 0; oc < {out_c}; ++oc) {{")
    ctx.lines.append(f"      for (size_t oh = 0; oh < {out_h}; ++oh) {{")
    ctx.lines.append(f"        for (size_t ow = 0; ow < {out_w}; ++ow) {{")
    ctx.lines.append(f"          size_t ih = oh / {block};")
    ctx.lines.append(f"          size_t iw = ow / {block};")
    ctx.lines.append(f"          size_t bh = oh % {block};")
    ctx.lines.append(f"          size_t bw = ow % {block};")
    if mode == "DCR":
        ctx.lines.append(f"          size_t ic = oc * {block * block} + bh * {block} + bw;")
    else:
        ctx.lines.append(f"          size_t ic = (bh * {block} + bw) * {out_c} + oc;")
    ctx.lines.append(
        f"          size_t in_idx = ((n_i * {c} + ic) * {h} + ih) * {w_in} + iw;"
    )
    ctx.lines.append(
        f"          size_t out_idx = ((n_i * {out_c} + oc) * {out_h} + oh) * {out_w} + ow;"
    )
    ctx.lines.append(f"          {out}[out_idx] = {x}[in_idx];")
    ctx.lines.append("        }")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
