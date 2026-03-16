# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op
from ....operators.utils import emit_op_global_avg_pool, product


@register_op("GlobalAveragePool")
def emit_global_avg_pool(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("GlobalAveragePool expects 1 input.")
    out_tensor = node.outputs[0]
    out_dtype = ctx.dtype(out_tensor)
    out = ctx.map_ptr(out_tensor)
    x_name = node.inputs[0]
    x = ctx.map_ptr(x_name)
    x_shape = ctx.shape(x_name)
    out_shape = ctx.shape(out_tensor)
    if out_dtype in ("int8", "int16"):
        if ctx.dtype(x_name) != out_dtype:
            raise ValueError("Quantized GlobalAveragePool requires matching dtypes.")
        sa, za = ctx.qparams(x_name)
        so, zo = ctx.qparams(out_tensor)
        qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
        if len(x_shape) < 3 or len(out_shape) != len(x_shape):
            raise ValueError("GlobalAveragePool expects rank >= 3 and matching ranks.")
        n, c = int(x_shape[0]), int(x_shape[1])
        n_out, c_out = int(out_shape[0]), int(out_shape[1])
        expected_out = [n, c] + [1] * (len(x_shape) - 2)
        if n != n_out:
            raise ValueError("GlobalAveragePool batch dimension mismatch.")
        if c != c_out or [int(v) for v in out_shape] != expected_out:
            raise ValueError("GlobalAveragePool output shape mismatch.")
        spatial = int(product(x_shape[2:]))
        out_inner = int(product(out_shape[2:]))
        ctx.lines.append(f"  for (size_t ni = 0; ni < {n}; ++ni) {{")
        ctx.lines.append(f"    for (size_t ch = 0; ch < {c}; ++ch) {{")
        ctx.lines.append("      float acc = 0.0f;")
        ctx.lines.append(f"      for (size_t i = 0; i < {spatial}; ++i) {{")
        ctx.lines.append(f"        size_t in_idx = ((ni * {c} + ch) * {spatial}) + i;")
        ctx.lines.append(
            f"        acc += ((float){x}[in_idx] - {za}) * {sa:.8f}f;"
        )
        ctx.lines.append("      }")
        ctx.lines.append(f"      acc = acc / (float)({spatial});")
        ctx.lines.append(f"      int q = (int)roundf(acc / {so:.8f}f) + {zo};")
        ctx.lines.append(f"      if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"      if (q > {qmax}) q = {qmax};")
        ctx.lines.append(f"      size_t out_idx = ((ni * {c_out} + ch) * {out_inner});")
        ctx.lines.append(
            f"      {out}[out_idx] = ({'int8_t' if out_dtype == 'int8' else 'int16_t'})q;"
        )
        ctx.lines.append("    }")
        ctx.lines.append("  }")
        return
    emit_op_global_avg_pool(ctx.lines, out, x, x_shape, out_shape)

