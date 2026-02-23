# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import tensor_size
from .registry import register_op


def _seed_u32(node: NodeInfo) -> int:
    seed_attr = node.attrs.get("seed", None)
    if seed_attr is None:
        return 1
    seed = int(abs(float(seed_attr)) * 1000003.0) & 0xFFFFFFFF
    if seed == 0:
        seed = 1
    return seed


@register_op("RandomUniform")
def emit_random_uniform(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 0:
        raise ValueError("RandomUniform expects 0 inputs.")
    if len(node.outputs) != 1:
        raise ValueError("RandomUniform expects 1 output.")

    out_name = node.outputs[0]
    out_dtype = ctx.dtype(out_name)
    if out_dtype not in ("float32", "int8", "int16"):
        raise ValueError("RandomUniform supports float32/int8/int16 output only.")

    out_shape = [int(v) for v in ctx.shape(out_name)]
    if any(v <= 0 for v in out_shape):
        raise ValueError("RandomUniform requires known positive output shape.")
    out_size = tensor_size(out_shape)
    low = float(node.attrs.get("low", 0.0))
    high = float(node.attrs.get("high", 1.0))
    if not high > low:
        raise ValueError("RandomUniform requires high > low.")
    seed = _seed_u32(node)
    quant_mode = out_dtype in ("int8", "int16")
    if out_dtype == "int8":
        qmin, qmax, qctype = -128, 127, "int8_t"
    elif out_dtype == "int16":
        qmin, qmax, qctype = -32768, 32767, "int16_t"

    out = ctx.map_ptr(out_name)
    state = ctx.next_symbol("k2c_ru_state")
    ctx.lines.append(f"  uint32_t {state} = (uint32_t){seed}u;")
    ctx.lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {{")
    ctx.lines.append(f"    {state} = {state} * 1664525u + 1013904223u;")
    ctx.lines.append(f"    float u = (float)({state} >> 8) * (1.0f / 16777216.0f);")
    ctx.lines.append(f"    float v = {low:.9g}f + ({high - low:.9g}f) * u;")
    if quant_mode:
        ctx.lines.append("    int32_t q = (int32_t)roundf(v);")
        ctx.lines.append(f"    if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"    if (q > {qmax}) q = {qmax};")
        ctx.lines.append(f"    {out}[i] = ({qctype})q;")
    else:
        ctx.lines.append(f"    {out}[i] = v;")
    ctx.lines.append("  }")
