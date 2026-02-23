# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op


def _seed_u32(node: NodeInfo) -> int:
    seed_attr = node.attrs.get("seed", None)
    if seed_attr is None:
        return 1
    seed = int(abs(float(seed_attr)) * 1000003.0) & 0xFFFFFFFF
    if seed == 0:
        seed = 1
    return seed


@register_op("Multinomial")
def emit_multinomial(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("Multinomial expects 1 input.")
    if len(node.outputs) != 1:
        raise ValueError("Multinomial expects 1 output.")

    x_name = node.inputs[0]
    out_name = node.outputs[0]
    x_dtype = ctx.dtype(x_name)
    if x_dtype not in ("float32", "int8", "int16"):
        raise ValueError("Multinomial supports float32/int8/int16 input only.")
    out_dtype = ctx.dtype(out_name)
    if out_dtype not in ("int64", "int32"):
        raise ValueError("Multinomial output dtype must be int64/int32.")

    x_shape = [int(v) for v in ctx.shape(x_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    if len(x_shape) != 2:
        raise ValueError("Multinomial currently supports 2D input [batch, classes].")
    batch, classes = x_shape
    if classes <= 0:
        raise ValueError("Multinomial classes dimension must be positive.")

    sample_size = int(node.attrs.get("sample_size", 1))
    if sample_size <= 0:
        raise ValueError("Multinomial sample_size must be positive.")
    if out_shape != [batch, sample_size]:
        raise ValueError("Multinomial output shape mismatch.")

    x = ctx.map_ptr(x_name)
    out = ctx.map_ptr(out_name)
    out_ctype = "int64_t" if out_dtype == "int64" else "int32_t"
    seed = _seed_u32(node)
    state = ctx.next_symbol("k2c_mult_state")

    ctx.lines.append(f"  uint32_t {state} = (uint32_t){seed}u;")
    ctx.lines.append(f"  for (size_t b = 0; b < {batch}; ++b) {{")
    ctx.lines.append("    float sum_prob = 0.0f;")
    ctx.lines.append(f"    for (size_t c = 0; c < {classes}; ++c) {{")
    ctx.lines.append(f"      float p = (float){x}[b * {classes} + c];")
    ctx.lines.append("      if (p > 0.0f) sum_prob += p;")
    ctx.lines.append("    }")
    ctx.lines.append(f"    for (size_t s = 0; s < {sample_size}; ++s) {{")
    ctx.lines.append("      int32_t picked = 0;")
    ctx.lines.append("      if (sum_prob > 0.0f) {")
    ctx.lines.append(f"        {state} = {state} * 1664525u + 1013904223u;")
    ctx.lines.append(f"        float u = (float)({state} >> 8) * (1.0f / 16777216.0f) * sum_prob;")
    ctx.lines.append("        float acc = 0.0f;")
    ctx.lines.append(f"        for (size_t c = 0; c < {classes}; ++c) {{")
    ctx.lines.append(f"          float p = (float){x}[b * {classes} + c];")
    ctx.lines.append("          if (p <= 0.0f) continue;")
    ctx.lines.append("          acc += p;")
    ctx.lines.append("          if (u <= acc) { picked = (int32_t)c; break; }")
    ctx.lines.append("          picked = (int32_t)c;")
    ctx.lines.append("        }")
    ctx.lines.append("      }")
    ctx.lines.append(f"      {out}[b * {sample_size} + s] = ({out_ctype})picked;")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
