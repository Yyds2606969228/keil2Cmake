# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op
from ....operators.utils import emit_op_gemm, tensor_size


@register_op("Gemm")
def emit_gemm(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 2:
        raise ValueError("Gemm expects at least 2 inputs.")
    out_tensor = node.outputs[0]
    out_dtype = ctx.dtype(out_tensor)
    out = ctx.map_ptr(out_tensor)
    a_name = node.inputs[0]
    b_name = node.inputs[1]
    c_name = node.inputs[2] if len(node.inputs) >= 3 else None
    a = ctx.map_ptr(a_name)
    b = ctx.map_ptr(b_name)
    c = ctx.map_ptr(c_name) if c_name else None
    a_shape = ctx.shape(a_name)
    b_shape = ctx.shape(b_name)
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise ValueError("Gemm supports 2D tensors only.")
    m, k1 = a_shape
    k2, n = b_shape
    if k1 != k2:
        raise ValueError("Gemm dimension mismatch.")
    if node.attrs.get("transA", 0) != 0 or node.attrs.get("transB", 0) != 0:
        raise ValueError("Gemm transA/transB is not supported.")
    if node.attrs.get("alpha", 1.0) != 1.0 or node.attrs.get("beta", 1.0) != 1.0:
        raise ValueError("Gemm alpha/beta is not supported.")
    c_is_matrix = False
    if c_name:
        c_shape = ctx.shape(c_name)
        c_size = tensor_size(c_shape)
        if c_size == n:
            c_is_matrix = False
        elif c_size == m * n:
            c_is_matrix = True
        else:
            raise ValueError("Gemm bias shape not supported.")
    if out_dtype in ("int8", "int16"):
        if ctx.dtype(a_name) != out_dtype or ctx.dtype(b_name) != out_dtype:
            raise ValueError("Quantized Gemm requires matching dtypes.")
        c_dtype = ctx.dtype(c_name) if c_name else None
        if c_dtype not in (None, "float32", "int32", "int64", "int8", "int16"):
            raise ValueError("Quantized Gemm bias dtype is not supported.")
        sa, za = ctx.qparams(a_name)
        sb, zb = ctx.qparams(b_name)
        so, zo = ctx.qparams(out_tensor)
        qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
        bias_scale = sa * sb
        ctx.lines.append(f"  for (size_t i = 0; i < {m}; ++i) {{")
        ctx.lines.append(f"    for (size_t j = 0; j < {n}; ++j) {{")
        ctx.lines.append("      float sum = 0.0f;")
        if c_name:
            if c_is_matrix:
                idx_expr = f"i * {n} + j"
            else:
                idx_expr = "j"
            if c_dtype == "float32":
                ctx.lines.append(f"      sum += {c}[{idx_expr}];")
            elif c_dtype in ("int32", "int64"):
                ctx.lines.append(f"      sum += ((float){c}[{idx_expr}]) * {bias_scale:.8f}f;")
            else:
                sc, zc = ctx.qparams(c_name)
                ctx.lines.append(f"      sum += ((float){c}[{idx_expr}] - {zc}) * {sc:.8f}f;")
        ctx.lines.append(f"      for (size_t t = 0; t < {k1}; ++t) {{")
        ctx.lines.append(
            f"        float ra = ((float){a}[i * {k1} + t] - {za}) * {sa:.8f}f;"
        )
        ctx.lines.append(
            f"        float rb = ((float){b}[t * {n} + j] - {zb}) * {sb:.8f}f;"
        )
        ctx.lines.append("        sum += ra * rb;")
        ctx.lines.append("      }")
        ctx.lines.append(f"      int q = (int)roundf(sum / {so:.8f}f) + {zo};")
        ctx.lines.append(f"      if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"      if (q > {qmax}) q = {qmax};")
        ctx.lines.append(
            f"      {out}[i * {n} + j] = ({'int8_t' if out_dtype == 'int8' else 'int16_t'})q;"
        )
        ctx.lines.append("    }")
        ctx.lines.append("  }")
        return
    if out_dtype != "float32":
        raise ValueError("Gemm supports float32 or quantized int8/int16 only.")
    emit_op_gemm(ctx.lines, out, a, b, c, m, k1, n, c_is_matrix)

