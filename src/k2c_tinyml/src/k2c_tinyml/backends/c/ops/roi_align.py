# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op


def _decode_mode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").lower()
    return str(value).lower()


@register_op("RoiAlign")
def emit_roi_align(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 3:
        raise ValueError("RoiAlign expects 3 inputs: X, rois, batch_indices.")
    if len(node.outputs) != 1:
        raise ValueError("RoiAlign expects exactly 1 output.")

    x_name, rois_name, batch_name = node.inputs
    out_name = node.outputs[0]

    if ctx.dtype(x_name) != "float32" or ctx.dtype(rois_name) != "float32":
        raise ValueError("RoiAlign currently supports float32 X/rois only.")
    if ctx.dtype(batch_name) not in ("int32", "int64"):
        raise ValueError("RoiAlign batch_indices must be int32/int64.")
    if ctx.dtype(out_name) != "float32":
        raise ValueError("RoiAlign output must be float32.")

    x_shape = [int(v) for v in ctx.shape(x_name)]
    rois_shape = [int(v) for v in ctx.shape(rois_name)]
    batch_shape = [int(v) for v in ctx.shape(batch_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]

    if len(x_shape) != 4:
        raise ValueError("RoiAlign X must be 4D NCHW.")
    if len(rois_shape) != 2 or int(rois_shape[1]) != 4:
        raise ValueError("RoiAlign rois must be [num_rois,4].")
    if len(batch_shape) != 1:
        raise ValueError("RoiAlign batch_indices must be 1D.")
    if len(out_shape) != 4:
        raise ValueError("RoiAlign output must be 4D.")

    n, c, h, w_in = x_shape
    num_rois = int(rois_shape[0])
    if int(batch_shape[0]) != num_rois:
        raise ValueError("RoiAlign rois and batch_indices size mismatch.")

    out_n, out_c, out_h, out_w = out_shape
    if out_n != num_rois or out_c != c:
        raise ValueError("RoiAlign output shape mismatch with rois/channels.")

    out_h_attr = int(node.attrs.get("output_height", 1))
    out_w_attr = int(node.attrs.get("output_width", 1))
    if out_h_attr <= 0 or out_w_attr <= 0:
        raise ValueError("RoiAlign output_height/output_width must be positive.")
    if out_h != out_h_attr or out_w != out_w_attr:
        raise ValueError("RoiAlign output shape mismatch with attrs.")

    sampling_ratio = int(node.attrs.get("sampling_ratio", 0))
    if sampling_ratio < 0:
        raise ValueError("RoiAlign sampling_ratio must be >= 0.")
    spatial_scale = float(node.attrs.get("spatial_scale", 1.0))
    mode = _decode_mode(node.attrs.get("mode", "avg"))
    if mode not in ("avg", "max"):
        raise ValueError("RoiAlign mode must be avg/max.")

    x = ctx.map_ptr(x_name)
    rois = ctx.map_ptr(rois_name)
    batch_idx = ctx.map_ptr(batch_name)
    out = ctx.map_ptr(out_name)

    ctx.lines.append(f"  for (size_t r = 0; r < {num_rois}; ++r) {{")
    ctx.lines.append(f"    int b = (int){batch_idx}[r];")
    ctx.lines.append(f"    if (b < 0) b = 0;")
    ctx.lines.append(f"    if (b >= {n}) b = {n - 1};")
    ctx.lines.append(f"    float roi_x1 = {rois}[r * 4 + 0] * {spatial_scale:.12g}f;")
    ctx.lines.append(f"    float roi_y1 = {rois}[r * 4 + 1] * {spatial_scale:.12g}f;")
    ctx.lines.append(f"    float roi_x2 = {rois}[r * 4 + 2] * {spatial_scale:.12g}f;")
    ctx.lines.append(f"    float roi_y2 = {rois}[r * 4 + 3] * {spatial_scale:.12g}f;")
    ctx.lines.append("    float roi_w = roi_x2 - roi_x1;")
    ctx.lines.append("    float roi_h = roi_y2 - roi_y1;")
    ctx.lines.append("    if (roi_w < 1.0f) roi_w = 1.0f;")
    ctx.lines.append("    if (roi_h < 1.0f) roi_h = 1.0f;")
    ctx.lines.append(f"    float bin_h = roi_h / (float){out_h};")
    ctx.lines.append(f"    float bin_w = roi_w / (float){out_w};")
    if sampling_ratio > 0:
        ctx.lines.append(f"    int samp_h = {sampling_ratio};")
        ctx.lines.append(f"    int samp_w = {sampling_ratio};")
    else:
        ctx.lines.append(f"    int samp_h = (int)ceilf(roi_h / (float){out_h});")
        ctx.lines.append(f"    int samp_w = (int)ceilf(roi_w / (float){out_w});")
        ctx.lines.append("    if (samp_h < 1) samp_h = 1;")
        ctx.lines.append("    if (samp_w < 1) samp_w = 1;")

    ctx.lines.append(f"    for (size_t ch = 0; ch < {c}; ++ch) {{")
    ctx.lines.append(f"      for (size_t ph = 0; ph < {out_h}; ++ph) {{")
    ctx.lines.append(f"        for (size_t pw = 0; pw < {out_w}; ++pw) {{")
    if mode == "max":
        ctx.lines.append("          float acc = -3.402823466e38f;")
    else:
        ctx.lines.append("          float acc = 0.0f;")
    ctx.lines.append("          for (int iy = 0; iy < samp_h; ++iy) {")
    ctx.lines.append(
        "            float yy = roi_y1 + ((float)ph + ((float)iy + 0.5f) / (float)samp_h) * bin_h;"
    )
    ctx.lines.append("            for (int ix = 0; ix < samp_w; ++ix) {")
    ctx.lines.append(
        "              float xx = roi_x1 + ((float)pw + ((float)ix + 0.5f) / (float)samp_w) * bin_w;"
    )
    ctx.lines.append("              float v = 0.0f;")
    ctx.lines.append(
        f"              if (!(yy < -1.0f || yy > (float){h} || xx < -1.0f || xx > (float){w_in})) {{"
    )
    ctx.lines.append("                if (yy < 0.0f) yy = 0.0f;")
    ctx.lines.append("                if (xx < 0.0f) xx = 0.0f;")
    ctx.lines.append(f"                if (yy > (float)({h} - 1)) yy = (float)({h} - 1);")
    ctx.lines.append(f"                if (xx > (float)({w_in} - 1)) xx = (float)({w_in} - 1);")
    ctx.lines.append("                int y0 = (int)floorf(yy);")
    ctx.lines.append("                int x0 = (int)floorf(xx);")
    ctx.lines.append(f"                int y1 = (y0 + 1 < {h}) ? (y0 + 1) : y0;")
    ctx.lines.append(f"                int x1 = (x0 + 1 < {w_in}) ? (x0 + 1) : x0;")
    ctx.lines.append("                float ly = yy - (float)y0;")
    ctx.lines.append("                float lx = xx - (float)x0;")
    ctx.lines.append("                float hy = 1.0f - ly;")
    ctx.lines.append("                float hx = 1.0f - lx;")
    ctx.lines.append(
        f"                size_t idx00 = ((size_t)b * {c} + ch) * {h} * {w_in} + (size_t)y0 * {w_in} + (size_t)x0;"
    )
    ctx.lines.append(
        f"                size_t idx01 = ((size_t)b * {c} + ch) * {h} * {w_in} + (size_t)y0 * {w_in} + (size_t)x1;"
    )
    ctx.lines.append(
        f"                size_t idx10 = ((size_t)b * {c} + ch) * {h} * {w_in} + (size_t)y1 * {w_in} + (size_t)x0;"
    )
    ctx.lines.append(
        f"                size_t idx11 = ((size_t)b * {c} + ch) * {h} * {w_in} + (size_t)y1 * {w_in} + (size_t)x1;"
    )
    ctx.lines.append(
        f"                v = {x}[idx00] * hy * hx + {x}[idx01] * hy * lx + {x}[idx10] * ly * hx + {x}[idx11] * ly * lx;"
    )
    ctx.lines.append("              }")
    if mode == "max":
        ctx.lines.append("              if (v > acc) acc = v;")
    else:
        ctx.lines.append("              acc += v;")
    ctx.lines.append("            }")
    ctx.lines.append("          }")
    if mode == "avg":
        ctx.lines.append("          acc = acc / (float)(samp_h * samp_w);")
    ctx.lines.append(
        f"          {out}[((r * {c} + ch) * {out_h} + ph) * {out_w} + pw] = acc;"
    )
    ctx.lines.append("        }")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
