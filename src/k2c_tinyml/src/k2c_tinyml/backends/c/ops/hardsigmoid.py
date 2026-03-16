# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op
from ....operators.utils import emit_op_unary_quant, tensor_size


@register_op("HardSigmoid")
def emit_hardsigmoid(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("HardSigmoid expects 1 input.")
    out_tensor = node.outputs[0]
    out_dtype = ctx.dtype(out_tensor)
    out = ctx.map_ptr(out_tensor)
    inp = ctx.map_ptr(node.inputs[0])
    size = tensor_size(ctx.shape(out_tensor))
    alpha = float(node.attrs.get("alpha", 0.2))
    beta = float(node.attrs.get("beta", 0.5))
    expr = (
        f"((({alpha:.8f}f * r) + {beta:.8f}f) < 0.0f ? 0.0f : "
        f"((({alpha:.8f}f * r) + {beta:.8f}f) > 1.0f ? 1.0f : (({alpha:.8f}f * r) + {beta:.8f}f)))"
    )
    if out_dtype in ("int8", "int16"):
        if ctx.dtype(node.inputs[0]) != out_dtype:
            raise ValueError("Quantized HardSigmoid requires matching dtypes.")
        sa, za = ctx.qparams(node.inputs[0])
        so_zo = ctx.qparams_optional(out_tensor)
        if so_zo is None:
            so, zo = sa, za
        else:
            so, zo = so_zo
        emit_op_unary_quant(ctx.lines, out, inp, size, expr, out_dtype, sa, za, so, zo)
        return
    if out_dtype != "float32":
        raise ValueError("HardSigmoid supports float32 or quantized int8/int16 only.")
    ctx.lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
    ctx.lines.append(f"    float v = ({alpha:.8f}f * {inp}[i]) + {beta:.8f}f;")
    ctx.lines.append("    if (v < 0.0f) v = 0.0f;")
    ctx.lines.append("    if (v > 1.0f) v = 1.0f;")
    ctx.lines.append(f"    {out}[i] = v;")
    ctx.lines.append("  }")
