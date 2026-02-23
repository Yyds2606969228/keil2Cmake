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


@register_op("RandomNormalLike")
def emit_random_normal_like(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("RandomNormalLike expects 1 input.")
    if len(node.outputs) != 1:
        raise ValueError("RandomNormalLike expects 1 output.")

    in_name = node.inputs[0]
    out_name = node.outputs[0]
    in_shape = [int(v) for v in ctx.shape(in_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    if in_shape != out_shape:
        raise ValueError("RandomNormalLike output shape must match input shape.")
    if any(v <= 0 for v in out_shape):
        raise ValueError("RandomNormalLike requires known positive shape.")

    out_dtype = ctx.dtype(out_name)
    if out_dtype not in ("float32", "int8", "int16"):
        raise ValueError("RandomNormalLike supports float32/int8/int16 output only.")

    out_size = tensor_size(out_shape)
    mean = float(node.attrs.get("mean", 0.0))
    scale = float(node.attrs.get("scale", 1.0))
    if not scale > 0.0:
        raise ValueError("RandomNormalLike requires scale > 0.")
    seed = _seed_u32(node)
    quant_mode = out_dtype in ("int8", "int16")
    if out_dtype == "int8":
        qmin, qmax, qctype = -128, 127, "int8_t"
    elif out_dtype == "int16":
        qmin, qmax, qctype = -32768, 32767, "int16_t"
    out = ctx.map_ptr(out_name)

    state = ctx.next_symbol("k2c_rnl_state")
    ctx.lines.append(f"  uint32_t {state} = (uint32_t){seed}u;")
    ctx.lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {{")
    ctx.lines.append(f"    {state} = {state} * 1664525u + 1013904223u;")
    ctx.lines.append(f"    float u1 = (float)({state} >> 8) * (1.0f / 16777216.0f);")
    ctx.lines.append("    if (u1 < 1e-7f) u1 = 1e-7f;")
    ctx.lines.append(f"    {state} = {state} * 1664525u + 1013904223u;")
    ctx.lines.append(f"    float u2 = (float)({state} >> 8) * (1.0f / 16777216.0f);")
    ctx.lines.append("    float mag = sqrtf(-2.0f * logf(u1));")
    ctx.lines.append("    float z = mag * cosf(6.283185307179586f * u2);")
    ctx.lines.append(f"    float v = {mean:.9g}f + {scale:.9g}f * z;")
    if quant_mode:
        ctx.lines.append("    int32_t q = (int32_t)roundf(v);")
        ctx.lines.append(f"    if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"    if (q > {qmax}) q = {qmax};")
        ctx.lines.append(f"    {out}[i] = ({qctype})q;")
    else:
        ctx.lines.append(f"    {out}[i] = v;")
    ctx.lines.append("  }")
