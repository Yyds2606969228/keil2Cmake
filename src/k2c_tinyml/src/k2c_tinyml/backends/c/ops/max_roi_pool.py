# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op


@register_op("MaxRoiPool")
def emit_max_roi_pool(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 2:
        raise ValueError("MaxRoiPool expects 2 inputs: X, rois.")
    if len(node.outputs) != 1:
        raise ValueError("MaxRoiPool expects 1 output.")

    x_name, rois_name = node.inputs
    out_name = node.outputs[0]
    x_dtype = ctx.dtype(x_name)
    out_dtype = ctx.dtype(out_name)
    if ctx.dtype(rois_name) != "float32":
        raise ValueError("MaxRoiPool rois dtype must be float32.")
    if x_dtype != out_dtype:
        raise ValueError("MaxRoiPool input/output dtype must match.")
    if x_dtype not in ("float32", "int8", "int16"):
        raise ValueError("MaxRoiPool supports float32/int8/int16 X/output only.")
    if out_dtype == "int8":
        qmin, qmax, qctype = -128, 127, "int8_t"
        init_max = "-128.0f"
    elif out_dtype == "int16":
        qmin, qmax, qctype = -32768, 32767, "int16_t"
        init_max = "-32768.0f"
    else:
        init_max = "-3.402823466e38f"

    x_shape = [int(v) for v in ctx.shape(x_name)]
    rois_shape = [int(v) for v in ctx.shape(rois_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    if len(x_shape) != 4:
        raise ValueError("MaxRoiPool X must be 4D NCHW.")
    if len(rois_shape) != 2 or rois_shape[1] != 5:
        raise ValueError("MaxRoiPool rois must be [num_rois,5].")
    if len(out_shape) != 4:
        raise ValueError("MaxRoiPool output must be 4D.")

    n_size, c_size, in_h, in_w = x_shape
    num_rois = rois_shape[0]
    out_n, out_c, ph, pw = out_shape
    if out_n != num_rois or out_c != c_size:
        raise ValueError("MaxRoiPool output shape mismatch.")

    pooled_shape = node.attrs.get("pooled_shape", [1, 1])
    if len(pooled_shape) != 2:
        raise ValueError("MaxRoiPool pooled_shape must have 2 values.")
    if int(pooled_shape[0]) != ph or int(pooled_shape[1]) != pw:
        raise ValueError("MaxRoiPool output pooled shape mismatch.")
    spatial_scale = float(node.attrs.get("spatial_scale", 1.0))

    x = ctx.map_ptr(x_name)
    rois = ctx.map_ptr(rois_name)
    out = ctx.map_ptr(out_name)

    ctx.lines.append(f"  for (size_t r = 0; r < {num_rois}; ++r) {{")
    ctx.lines.append(f"    int b = (int){rois}[r * 5 + 0];")
    ctx.lines.append("    if (b < 0) b = 0;")
    ctx.lines.append(f"    if (b >= {n_size}) b = {n_size - 1};")
    ctx.lines.append(f"    float x1 = {rois}[r * 5 + 1] * {spatial_scale:.9g}f;")
    ctx.lines.append(f"    float y1 = {rois}[r * 5 + 2] * {spatial_scale:.9g}f;")
    ctx.lines.append(f"    float x2 = {rois}[r * 5 + 3] * {spatial_scale:.9g}f;")
    ctx.lines.append(f"    float y2 = {rois}[r * 5 + 4] * {spatial_scale:.9g}f;")
    ctx.lines.append("    float roi_w = x2 - x1;")
    ctx.lines.append("    float roi_h = y2 - y1;")
    ctx.lines.append("    if (roi_w < 1.0f) roi_w = 1.0f;")
    ctx.lines.append("    if (roi_h < 1.0f) roi_h = 1.0f;")
    ctx.lines.append(f"    float bin_w = roi_w / (float){pw};")
    ctx.lines.append(f"    float bin_h = roi_h / (float){ph};")
    ctx.lines.append(f"    for (size_t c = 0; c < {c_size}; ++c) {{")
    ctx.lines.append(f"      for (size_t ph_i = 0; ph_i < {ph}; ++ph_i) {{")
    ctx.lines.append(f"        for (size_t pw_i = 0; pw_i < {pw}; ++pw_i) {{")
    ctx.lines.append("          int hs = (int)floorf(y1 + (float)ph_i * bin_h);")
    ctx.lines.append("          int ws = (int)floorf(x1 + (float)pw_i * bin_w);")
    ctx.lines.append("          int he = (int)ceilf(y1 + (float)(ph_i + 1) * bin_h);")
    ctx.lines.append("          int we = (int)ceilf(x1 + (float)(pw_i + 1) * bin_w);")
    ctx.lines.append("          if (hs < 0) hs = 0;")
    ctx.lines.append("          if (ws < 0) ws = 0;")
    ctx.lines.append(f"          if (he > {in_h}) he = {in_h};")
    ctx.lines.append(f"          if (we > {in_w}) we = {in_w};")
    ctx.lines.append(f"          float max_v = {init_max};")
    ctx.lines.append("          if (hs >= he || ws >= we) {")
    ctx.lines.append("            max_v = 0.0f;")
    ctx.lines.append("          } else {")
    ctx.lines.append("            for (int ih = hs; ih < he; ++ih) {")
    ctx.lines.append("              for (int iw = ws; iw < we; ++iw) {")
    ctx.lines.append(f"                float v = (float){x}[((size_t)b * {c_size} + c) * {in_h} * {in_w} + (size_t)ih * {in_w} + (size_t)iw];")
    ctx.lines.append("                if (v > max_v) max_v = v;")
    ctx.lines.append("              }")
    ctx.lines.append("            }")
    ctx.lines.append("          }")
    if out_dtype == "float32":
        ctx.lines.append(f"          {out}[((r * {c_size} + c) * {ph} + ph_i) * {pw} + pw_i] = max_v;")
    else:
        ctx.lines.append("          int32_t q = (int32_t)roundf(max_v);")
        ctx.lines.append(f"          if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"          if (q > {qmax}) q = {qmax};")
        ctx.lines.append(f"          {out}[((r * {c_size} + c) * {ph} + ph_i) * {pw} + pw_i] = ({qctype})q;")
    ctx.lines.append("        }")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
