# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op
from ....operators.utils import normalize_axis, product


@register_op("Softmax")
def emit_softmax(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("Softmax expects 1 input.")
    in_name = node.inputs[0]
    out_tensor = node.outputs[0]
    in_dtype = ctx.dtype(in_name)
    out_dtype = ctx.dtype(out_tensor)
    quant_mode = in_dtype in ("int8", "int16") or out_dtype in ("int8", "int16")
    if quant_mode:
        if in_dtype not in ("int8", "int16") or out_dtype not in ("int8", "int16"):
            raise ValueError("Softmax quantized path requires int8/int16 input/output.")
        sx, zx = ctx.qparams(in_name)
        so, zo = ctx.qparams(out_tensor)
        qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
        qctype = "int8_t" if out_dtype == "int8" else "int16_t"
    elif in_dtype != "float32" or out_dtype != "float32":
        raise ValueError("Softmax supports float32 or quantized int8/int16.")
    out = ctx.map_ptr(out_tensor)
    a = ctx.map_ptr(in_name)
    shape = ctx.shape(out_tensor)
    if len(shape) == 0:
        raise ValueError("Softmax does not support scalar tensor.")
    axis = int(node.attrs.get("axis", -1))
    rank = len(shape)
    axis = normalize_axis(axis, rank)
    axis_size = int(shape[axis])
    if axis_size <= 0:
        raise ValueError("Softmax axis dimension must be positive.")
    outer = product(shape[:axis]) if axis > 0 else 1
    inner = product(shape[axis + 1 :]) if axis + 1 < rank else 1
    tmp_name = ctx.next_symbol("k2c_softmax_tmp") if quant_mode else ""

    lines = ctx.lines
    lines.append(f"  for (size_t outer_i = 0; outer_i < {outer}; ++outer_i) {{")
    lines.append(f"    for (size_t inner_i = 0; inner_i < {inner}; ++inner_i) {{")
    if quant_mode:
        lines.append(f"      float {tmp_name}[{axis_size}];")
    lines.append(f"      size_t base = outer_i * {axis_size} * {inner} + inner_i;")
    if quant_mode:
        lines.append(f"      float max_v = ((float){a}[base] - {zx}) * {sx:.8f}f;")
    else:
        lines.append(f"      float max_v = {a}[base];")
    lines.append(f"      for (size_t axis_i = 1; axis_i < {axis_size}; ++axis_i) {{")
    lines.append(f"        size_t idx = base + axis_i * {inner};")
    if quant_mode:
        lines.append(f"        float v = ((float){a}[idx] - {zx}) * {sx:.8f}f;")
    else:
        lines.append(f"        float v = {a}[idx];")
    lines.append("        if (v > max_v) max_v = v;")
    lines.append("      }")
    lines.append("      float sum = 0.0f;")
    lines.append(f"      for (size_t axis_i = 0; axis_i < {axis_size}; ++axis_i) {{")
    lines.append(f"        size_t idx = base + axis_i * {inner};")
    if quant_mode:
        lines.append(f"        float v = ((float){a}[idx] - {zx}) * {sx:.8f}f;")
        lines.append("        float e = expf(v - max_v);")
    else:
        lines.append(f"        float e = expf({a}[idx] - max_v);")
    if quant_mode:
        lines.append(f"        {tmp_name}[axis_i] = e;")
    else:
        lines.append(f"        {out}[idx] = e;")
    lines.append("        sum += e;")
    lines.append("      }")
    lines.append(f"      for (size_t axis_i = 0; axis_i < {axis_size}; ++axis_i) {{")
    lines.append(f"        size_t idx = base + axis_i * {inner};")
    if quant_mode:
        lines.append(f"        float p = {tmp_name}[axis_i] / sum;")
        lines.append(f"        int q = (int)roundf(p / {so:.8f}f) + {zo};")
        lines.append(f"        if (q < {qmin}) q = {qmin};")
        lines.append(f"        if (q > {qmax}) q = {qmax};")
        lines.append(f"        {out}[idx] = ({qctype})q;")
    else:
        lines.append(f"        {out}[idx] = {out}[idx] / sum;")
    lines.append("      }")
    lines.append("    }")
    lines.append("  }")

