# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import get_const_ints
from .registry import register_op


def _decode_attr_str(value: object, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").lower()
    return str(value).lower()


def _const_floats(ctx: EmitContext, name: str) -> list[float]:
    tensor = ctx.model.tensors.get(name)
    if tensor is None or tensor.data is None:
        raise ValueError(f"Resize constant input '{name}' is missing.")
    return [float(v) for v in tensor.data]


@register_op("Resize")
def emit_resize(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 1:
        raise ValueError("Resize expects at least 1 input.")
    x_name = node.inputs[0]
    out_name = node.outputs[0]

    in_shape = ctx.shape(x_name)
    out_shape = ctx.shape(out_name)
    if len(in_shape) != 4 or len(out_shape) != 4:
        raise ValueError("Resize currently supports 4D NCHW only.")
    n, c, in_h, in_w = [int(v) for v in in_shape]
    n_out, c_out, out_h, out_w = [int(v) for v in out_shape]
    if n != n_out or c != c_out:
        raise ValueError("Resize N/C dimensions must be unchanged.")
    if in_h <= 0 or in_w <= 0 or out_h <= 0 or out_w <= 0:
        raise ValueError("Resize requires known positive dimensions.")

    mode = _decode_attr_str(node.attrs.get("mode"), "nearest")
    if mode != "nearest":
        raise ValueError("Resize currently supports mode='nearest' only.")
    coord_mode = _decode_attr_str(node.attrs.get("coordinate_transformation_mode"), "half_pixel")
    if coord_mode not in ("asymmetric", "half_pixel", "align_corners", "pytorch_half_pixel"):
        raise ValueError("Resize coordinate_transformation_mode is unsupported.")
    nearest_mode = _decode_attr_str(node.attrs.get("nearest_mode"), "round_prefer_floor")
    if nearest_mode not in ("round_prefer_floor", "round_prefer_ceil", "floor", "ceil"):
        raise ValueError("Resize nearest_mode is unsupported.")

    scales_name = node.inputs[2] if len(node.inputs) >= 3 and node.inputs[2] else None
    sizes_name = node.inputs[3] if len(node.inputs) >= 4 and node.inputs[3] else None
    if sizes_name is not None:
        sizes = [int(v) for v in get_const_ints(ctx.model, sizes_name)]
        if sizes != out_shape:
            raise ValueError("Resize sizes must match inferred output shape.")
    elif scales_name is not None:
        scales = _const_floats(ctx, scales_name)
        if len(scales) != 4:
            raise ValueError("Resize scales rank mismatch.")
        expected = [
            int(in_shape[0] * scales[0]),
            int(in_shape[1] * scales[1]),
            int(in_shape[2] * scales[2]),
            int(in_shape[3] * scales[3]),
        ]
        if expected != out_shape:
            raise ValueError("Resize scales do not match inferred output shape.")

    in_dtype = ctx.dtype(x_name)
    out_dtype = ctx.dtype(out_name)
    if in_dtype != out_dtype:
        raise ValueError("Resize requires input/output dtype to match.")
    if out_dtype not in ("float32", "int8", "int16"):
        raise ValueError("Resize supports float32/int8/int16 only.")

    inp = ctx.map_ptr(x_name)
    out = ctx.map_ptr(out_name)
    ctype = "int8_t" if out_dtype == "int8" else "int16_t"
    use_requant = False
    if out_dtype in ("int8", "int16"):
        si, zi = ctx.qparams(x_name)
        so_zo = ctx.qparams_optional(out_name)
        if so_zo is None:
            so, zo = si, zi
        else:
            so, zo = so_zo
        use_requant = abs(si - so) > 1e-12 or zi != zo
        if out_dtype == "int8":
            qmin, qmax = -128, 127
        else:
            qmin, qmax = -32768, 32767

    ctx.lines.append(f"  for (size_t n_i = 0; n_i < {n}; ++n_i) {{")
    ctx.lines.append(f"    for (size_t c_i = 0; c_i < {c}; ++c_i) {{")
    ctx.lines.append(f"      for (size_t oh_i = 0; oh_i < {out_h}; ++oh_i) {{")
    if coord_mode == "asymmetric":
        ctx.lines.append(f"        float src_hf = ((float)oh_i * (float){in_h}) / (float){out_h};")
    elif coord_mode == "half_pixel":
        ctx.lines.append(
            f"        float src_hf = (((float)oh_i + 0.5f) * (float){in_h}) / (float){out_h} - 0.5f;"
        )
    elif coord_mode == "align_corners":
        if out_h == 1:
            ctx.lines.append("        float src_hf = 0.0f;")
        else:
            ctx.lines.append(
                f"        float src_hf = (float)oh_i * (float)({in_h} - 1) / (float)({out_h} - 1);"
            )
    else:
        if out_h == 1:
            ctx.lines.append("        float src_hf = 0.0f;")
        else:
            ctx.lines.append(
                f"        float src_hf = (((float)oh_i + 0.5f) * (float){in_h}) / (float){out_h} - 0.5f;"
            )
    if nearest_mode == "floor":
        ctx.lines.append("        int in_h_idx = (int)floorf(src_hf);")
    elif nearest_mode == "ceil":
        ctx.lines.append("        int in_h_idx = (int)ceilf(src_hf);")
    elif nearest_mode == "round_prefer_ceil":
        ctx.lines.append("        int in_h_idx = (int)ceilf(src_hf - 0.5f);")
    else:
        ctx.lines.append("        int in_h_idx = (int)floorf(src_hf + 0.5f);")
    ctx.lines.append("        if (in_h_idx < 0) in_h_idx = 0;")
    ctx.lines.append(f"        if (in_h_idx > {in_h - 1}) in_h_idx = {in_h - 1};")
    ctx.lines.append(f"        for (size_t ow_i = 0; ow_i < {out_w}; ++ow_i) {{")
    if coord_mode == "asymmetric":
        ctx.lines.append(f"          float src_wf = ((float)ow_i * (float){in_w}) / (float){out_w};")
    elif coord_mode == "half_pixel":
        ctx.lines.append(
            f"          float src_wf = (((float)ow_i + 0.5f) * (float){in_w}) / (float){out_w} - 0.5f;"
        )
    elif coord_mode == "align_corners":
        if out_w == 1:
            ctx.lines.append("          float src_wf = 0.0f;")
        else:
            ctx.lines.append(
                f"          float src_wf = (float)ow_i * (float)({in_w} - 1) / (float)({out_w} - 1);"
            )
    else:
        if out_w == 1:
            ctx.lines.append("          float src_wf = 0.0f;")
        else:
            ctx.lines.append(
                f"          float src_wf = (((float)ow_i + 0.5f) * (float){in_w}) / (float){out_w} - 0.5f;"
            )
    if nearest_mode == "floor":
        ctx.lines.append("          int in_w_idx = (int)floorf(src_wf);")
    elif nearest_mode == "ceil":
        ctx.lines.append("          int in_w_idx = (int)ceilf(src_wf);")
    elif nearest_mode == "round_prefer_ceil":
        ctx.lines.append("          int in_w_idx = (int)ceilf(src_wf - 0.5f);")
    else:
        ctx.lines.append("          int in_w_idx = (int)floorf(src_wf + 0.5f);")
    ctx.lines.append("          if (in_w_idx < 0) in_w_idx = 0;")
    ctx.lines.append(f"          if (in_w_idx > {in_w - 1}) in_w_idx = {in_w - 1};")
    ctx.lines.append(
        f"          size_t in_idx = ((n_i * {c} + c_i) * {in_h} + (size_t)in_h_idx) * {in_w} + (size_t)in_w_idx;"
    )
    ctx.lines.append(
        f"          size_t out_idx = ((n_i * {c} + c_i) * {out_h} + oh_i) * {out_w} + ow_i;"
    )
    if out_dtype in ("int8", "int16") and use_requant:
        ctx.lines.append(f"          float r = ((float){inp}[in_idx] - {zi}) * {si:.8f}f;")
        ctx.lines.append(f"          int q = (int)roundf(r / {so:.8f}f) + {zo};")
        ctx.lines.append(f"          if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"          if (q > {qmax}) q = {qmax};")
        ctx.lines.append(f"          {out}[out_idx] = ({ctype})q;")
    else:
        ctx.lines.append(f"          {out}[out_idx] = {inp}[in_idx];")
    ctx.lines.append("        }")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
