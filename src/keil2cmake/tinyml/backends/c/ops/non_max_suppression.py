# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import tensor_size
from .registry import register_op


def _ensure_scalar(ctx: EmitContext, name: str, what: str) -> None:
    if tensor_size(ctx.shape(name)) != 1:
        raise ValueError(f"NonMaxSuppression {what} must be scalar.")


def _emit_runtime_scalar_int(ctx: EmitContext, name: str | None, default: int, var_name: str) -> str:
    ctx.lines.append(f"  int32_t {var_name} = {int(default)};")
    if not name:
        return var_name
    _ensure_scalar(ctx, name, "max_output_boxes_per_class")
    dtype = ctx.dtype(name)
    if dtype not in ("uint8", "int8", "int16", "int32", "int64", "float32"):
        raise ValueError(
            "NonMaxSuppression max_output_boxes_per_class must be uint8/int8/int16/int32/int64/float32 scalar."
        )
    ptr = ctx.map_ptr(name)
    ctx.lines.append(f"  {var_name} = (int32_t)({ptr}[0]);")
    return var_name


def _emit_runtime_scalar_float(ctx: EmitContext, name: str | None, default: float, var_name: str, what: str) -> str:
    ctx.lines.append(f"  float {var_name} = {default:.12g}f;")
    if not name:
        return var_name
    _ensure_scalar(ctx, name, what)
    dtype = ctx.dtype(name)
    if dtype not in ("uint8", "int8", "int16", "int32", "int64", "float32"):
        raise ValueError(
            f"NonMaxSuppression {what} must be uint8/int8/int16/int32/int64/float32 scalar."
        )
    ptr = ctx.map_ptr(name)
    ctx.lines.append(f"  {var_name} = (float)({ptr}[0]);")
    return var_name


@register_op("NonMaxSuppression")
def emit_non_max_suppression(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 2:
        raise ValueError("NonMaxSuppression expects at least 2 inputs.")
    if len(node.inputs) > 5:
        raise ValueError("NonMaxSuppression currently supports up to 5 inputs.")
    if len(node.outputs) != 1:
        raise ValueError("NonMaxSuppression expects exactly 1 output.")

    boxes_name = node.inputs[0]
    scores_name = node.inputs[1]
    out_name = node.outputs[0]

    if ctx.dtype(boxes_name) != "float32" or ctx.dtype(scores_name) != "float32":
        raise ValueError("NonMaxSuppression currently supports float32 boxes/scores only.")
    out_dtype = ctx.dtype(out_name)
    if out_dtype not in ("int64", "int32"):
        raise ValueError("NonMaxSuppression output dtype must be int64/int32.")

    boxes_shape = [int(v) for v in ctx.shape(boxes_name)]
    scores_shape = [int(v) for v in ctx.shape(scores_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    if len(boxes_shape) != 3 or len(scores_shape) != 3:
        raise ValueError("NonMaxSuppression expects boxes/scores rank=3.")
    if len(out_shape) != 2 or int(out_shape[1]) != 3:
        raise ValueError("NonMaxSuppression output shape must be [N,3].")
    if int(boxes_shape[2]) != 4:
        raise ValueError("NonMaxSuppression boxes last dimension must be 4.")
    if int(boxes_shape[0]) != int(scores_shape[0]):
        raise ValueError("NonMaxSuppression batch dimension mismatch.")
    if int(boxes_shape[1]) != int(scores_shape[2]):
        raise ValueError("NonMaxSuppression spatial dimension mismatch.")
    out_cap = int(out_shape[0])
    if out_cap <= 0:
        raise ValueError("NonMaxSuppression output first dimension must be positive.")

    max_boxes_name = node.inputs[2] if len(node.inputs) >= 3 and node.inputs[2] else None
    iou_name = node.inputs[3] if len(node.inputs) >= 4 and node.inputs[3] else None
    score_name = node.inputs[4] if len(node.inputs) >= 5 and node.inputs[4] else None
    center_point_box = int(node.attrs.get("center_point_box", 0))
    if center_point_box not in (0, 1):
        raise ValueError("NonMaxSuppression center_point_box must be 0 or 1.")

    num_batches = int(boxes_shape[0])
    spatial_dim = int(boxes_shape[1])
    num_classes = int(scores_shape[1])
    if spatial_dim <= 0:
        raise ValueError("NonMaxSuppression spatial dimension must be positive.")

    boxes = ctx.map_ptr(boxes_name)
    scores = ctx.map_ptr(scores_name)
    out = ctx.map_ptr(out_name)
    idx_ctype = "int64_t" if out_dtype == "int64" else "int32_t"
    out_pos = ctx.next_symbol("k2c_nms_out_pos")
    used = ctx.next_symbol("k2c_nms_used")
    selected = ctx.next_symbol("k2c_nms_selected")
    max_output_var = ctx.next_symbol("k2c_nms_max_output")
    iou_var = ctx.next_symbol("k2c_nms_iou_threshold")
    score_var = ctx.next_symbol("k2c_nms_score_threshold")

    ctx.lines.append(f"  size_t {out_pos} = 0;")
    ctx.lines.append(f"  static int32_t {used}[{spatial_dim}];")
    ctx.lines.append(f"  static int32_t {selected}[{spatial_dim}];")
    _emit_runtime_scalar_int(ctx, max_boxes_name, 0, max_output_var)
    _emit_runtime_scalar_float(ctx, iou_name, 0.0, iou_var, "iou_threshold")
    _emit_runtime_scalar_float(ctx, score_name, -3.402823466e38, score_var, "score_threshold")
    ctx.lines.append(f"  if ({max_output_var} > {spatial_dim}) {max_output_var} = {spatial_dim};")
    ctx.lines.append(f"  for (size_t i = 0; i < {out_cap}; ++i) {{")
    ctx.lines.append(f"    {out}[i * 3 + 0] = ({idx_ctype})-1;")
    ctx.lines.append(f"    {out}[i * 3 + 1] = ({idx_ctype})-1;")
    ctx.lines.append(f"    {out}[i * 3 + 2] = ({idx_ctype})-1;")
    ctx.lines.append("  }")
    ctx.lines.append(f"  if ({max_output_var} > 0) {{")
    ctx.lines.append(f"    for (size_t b = 0; b < {num_batches}; ++b) {{")
    ctx.lines.append(f"      for (size_t c = 0; c < {num_classes}; ++c) {{")
    ctx.lines.append(f"      for (size_t i = 0; i < {spatial_dim}; ++i) {used}[i] = 0;")
    ctx.lines.append("      size_t selected_count = 0;")
    ctx.lines.append(f"      for (size_t pick = 0; pick < {spatial_dim}; ++pick) {{")
    ctx.lines.append("        int has_best = 0;")
    ctx.lines.append("        size_t best_idx = 0;")
    ctx.lines.append("        float best_score = 0.0f;")
    ctx.lines.append(f"        for (size_t bi = 0; bi < {spatial_dim}; ++bi) {{")
    ctx.lines.append(f"          if ({used}[bi]) continue;")
    ctx.lines.append(
        f"          float score = {scores}[(b * {num_classes} + c) * {spatial_dim} + bi];"
    )
    ctx.lines.append(f"          if (score < {score_var}) continue;")
    ctx.lines.append(
        "          if (!has_best || score > best_score || (score == best_score && bi < best_idx)) {"
    )
    ctx.lines.append("            has_best = 1;")
    ctx.lines.append("            best_score = score;")
    ctx.lines.append("            best_idx = bi;")
    ctx.lines.append("          }")
    ctx.lines.append("        }")
    ctx.lines.append("        if (!has_best) break;")
    ctx.lines.append(f"        {used}[best_idx] = 1;")
    ctx.lines.append("        int keep = 1;")
    ctx.lines.append(
        f"        float b1a = {boxes}[(b * {spatial_dim} + best_idx) * 4 + 0];"
    )
    ctx.lines.append(
        f"        float b1b = {boxes}[(b * {spatial_dim} + best_idx) * 4 + 1];"
    )
    ctx.lines.append(
        f"        float b1c = {boxes}[(b * {spatial_dim} + best_idx) * 4 + 2];"
    )
    ctx.lines.append(
        f"        float b1d = {boxes}[(b * {spatial_dim} + best_idx) * 4 + 3];"
    )
    if center_point_box == 0:
        ctx.lines.append("        float b1_y1 = b1a;")
        ctx.lines.append("        float b1_x1 = b1b;")
        ctx.lines.append("        float b1_y2 = b1c;")
        ctx.lines.append("        float b1_x2 = b1d;")
    else:
        ctx.lines.append("        float b1_x1 = b1a - b1c * 0.5f;")
        ctx.lines.append("        float b1_y1 = b1b - b1d * 0.5f;")
        ctx.lines.append("        float b1_x2 = b1a + b1c * 0.5f;")
        ctx.lines.append("        float b1_y2 = b1b + b1d * 0.5f;")
    ctx.lines.append("        if (b1_x1 > b1_x2) { float t = b1_x1; b1_x1 = b1_x2; b1_x2 = t; }")
    ctx.lines.append("        if (b1_y1 > b1_y2) { float t = b1_y1; b1_y1 = b1_y2; b1_y2 = t; }")
    ctx.lines.append("        float area1_w = b1_x2 - b1_x1;")
    ctx.lines.append("        float area1_h = b1_y2 - b1_y1;")
    ctx.lines.append("        if (area1_w < 0.0f) area1_w = 0.0f;")
    ctx.lines.append("        if (area1_h < 0.0f) area1_h = 0.0f;")
    ctx.lines.append("        float area1 = area1_w * area1_h;")
    ctx.lines.append("        for (size_t si = 0; si < selected_count; ++si) {")
    ctx.lines.append(f"          size_t prev_idx = (size_t){selected}[si];")
    ctx.lines.append(
        f"          float b2a = {boxes}[(b * {spatial_dim} + prev_idx) * 4 + 0];"
    )
    ctx.lines.append(
        f"          float b2b = {boxes}[(b * {spatial_dim} + prev_idx) * 4 + 1];"
    )
    ctx.lines.append(
        f"          float b2c = {boxes}[(b * {spatial_dim} + prev_idx) * 4 + 2];"
    )
    ctx.lines.append(
        f"          float b2d = {boxes}[(b * {spatial_dim} + prev_idx) * 4 + 3];"
    )
    if center_point_box == 0:
        ctx.lines.append("          float b2_y1 = b2a;")
        ctx.lines.append("          float b2_x1 = b2b;")
        ctx.lines.append("          float b2_y2 = b2c;")
        ctx.lines.append("          float b2_x2 = b2d;")
    else:
        ctx.lines.append("          float b2_x1 = b2a - b2c * 0.5f;")
        ctx.lines.append("          float b2_y1 = b2b - b2d * 0.5f;")
        ctx.lines.append("          float b2_x2 = b2a + b2c * 0.5f;")
        ctx.lines.append("          float b2_y2 = b2b + b2d * 0.5f;")
    ctx.lines.append("          if (b2_x1 > b2_x2) { float t = b2_x1; b2_x1 = b2_x2; b2_x2 = t; }")
    ctx.lines.append("          if (b2_y1 > b2_y2) { float t = b2_y1; b2_y1 = b2_y2; b2_y2 = t; }")
    ctx.lines.append("          float area2_w = b2_x2 - b2_x1;")
    ctx.lines.append("          float area2_h = b2_y2 - b2_y1;")
    ctx.lines.append("          if (area2_w < 0.0f) area2_w = 0.0f;")
    ctx.lines.append("          if (area2_h < 0.0f) area2_h = 0.0f;")
    ctx.lines.append("          float area2 = area2_w * area2_h;")
    ctx.lines.append("          float inter_x1 = (b1_x1 > b2_x1) ? b1_x1 : b2_x1;")
    ctx.lines.append("          float inter_y1 = (b1_y1 > b2_y1) ? b1_y1 : b2_y1;")
    ctx.lines.append("          float inter_x2 = (b1_x2 < b2_x2) ? b1_x2 : b2_x2;")
    ctx.lines.append("          float inter_y2 = (b1_y2 < b2_y2) ? b1_y2 : b2_y2;")
    ctx.lines.append("          float inter_w = inter_x2 - inter_x1;")
    ctx.lines.append("          float inter_h = inter_y2 - inter_y1;")
    ctx.lines.append("          if (inter_w < 0.0f) inter_w = 0.0f;")
    ctx.lines.append("          if (inter_h < 0.0f) inter_h = 0.0f;")
    ctx.lines.append("          float inter_area = inter_w * inter_h;")
    ctx.lines.append("          float denom = area1 + area2 - inter_area;")
    ctx.lines.append("          float iou = (denom > 0.0f) ? (inter_area / denom) : 0.0f;")
    ctx.lines.append(f"          if (iou > {iou_var}) {{ keep = 0; break; }}")
    ctx.lines.append("        }")
    ctx.lines.append("        if (!keep) continue;")
    ctx.lines.append(f"        {selected}[selected_count] = (int32_t)best_idx;")
    ctx.lines.append("        selected_count += 1;")
    ctx.lines.append(f"        if ({out_pos} < {out_cap}) {{")
    ctx.lines.append(f"          {out}[{out_pos} * 3 + 0] = ({idx_ctype})b;")
    ctx.lines.append(f"          {out}[{out_pos} * 3 + 1] = ({idx_ctype})c;")
    ctx.lines.append(f"          {out}[{out_pos} * 3 + 2] = ({idx_ctype})best_idx;")
    ctx.lines.append(f"          {out_pos} += 1;")
    ctx.lines.append("        }")
    ctx.lines.append(f"        if (selected_count >= (size_t){max_output_var}) break;")
    ctx.lines.append("      }")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
