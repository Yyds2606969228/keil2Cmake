# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import tensor_size
from .registry import register_op


def _classify_c(c_shape: list[int], m: int, n: int) -> tuple[str, str]:
    c_rank = len(c_shape)
    c_size = tensor_size(c_shape)
    if c_size == 1:
        return "scalar", "0"
    if c_rank == 1:
        if c_shape[0] == n:
            return "vec_n", "j"
        if c_shape[0] == m:
            return "vec_m", "i"
        if c_shape[0] == m * n:
            return "matrix", f"i * {n} + j"
        raise ValueError("Gemm bias shape is not broadcastable to [M,N].")
    if c_rank == 2:
        r, c = int(c_shape[0]), int(c_shape[1])
        if r == m and c == n:
            return "matrix", f"i * {n} + j"
        if r == 1 and c == n:
            return "vec_n", "j"
        if r == m and c == 1:
            return "vec_m", "i"
        if r == 1 and c == 1:
            return "scalar", "0"
        raise ValueError("Gemm bias shape is not broadcastable to [M,N].")
    raise ValueError("Gemm bias rank > 2 is not supported.")


@register_op("Gemm")
def emit_gemm(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 2:
        raise ValueError("Gemm expects at least 2 inputs.")

    out_name = node.outputs[0]
    out_dtype = ctx.dtype(out_name)
    out = ctx.map_ptr(out_name)

    a_name = node.inputs[0]
    b_name = node.inputs[1]
    c_name = node.inputs[2] if len(node.inputs) >= 3 and node.inputs[2] else None
    a = ctx.map_ptr(a_name)
    b = ctx.map_ptr(b_name)
    c = ctx.map_ptr(c_name) if c_name else None

    a_shape = [int(v) for v in ctx.shape(a_name)]
    b_shape = [int(v) for v in ctx.shape(b_name)]
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise ValueError("Gemm supports 2D tensors only.")

    a_rows, a_cols = int(a_shape[0]), int(a_shape[1])
    b_rows, b_cols = int(b_shape[0]), int(b_shape[1])
    trans_a = int(node.attrs.get("transA", 0))
    trans_b = int(node.attrs.get("transB", 0))
    if trans_a not in (0, 1) or trans_b not in (0, 1):
        raise ValueError("Gemm transA/transB must be 0 or 1.")

    m = a_cols if trans_a == 1 else a_rows
    k1 = a_rows if trans_a == 1 else a_cols
    k2 = b_cols if trans_b == 1 else b_rows
    n = b_rows if trans_b == 1 else b_cols
    if k1 != k2:
        raise ValueError("Gemm dimension mismatch.")

    out_shape = [int(v) for v in ctx.shape(out_name)]
    if out_shape != [m, n]:
        raise ValueError("Gemm output shape mismatch.")

    alpha = float(node.attrs.get("alpha", 1.0))
    beta = float(node.attrs.get("beta", 1.0))

    c_mode = "none"
    c_index = "0"
    if c_name is not None:
        c_shape = [int(v) for v in ctx.shape(c_name)]
        c_mode, c_index = _classify_c(c_shape, m, n)

    if out_dtype in ("int8", "int16"):
        if ctx.dtype(a_name) != out_dtype or ctx.dtype(b_name) != out_dtype:
            raise ValueError("Quantized Gemm requires matching dtypes for A/B/output.")
        c_dtype = ctx.dtype(c_name) if c_name else None
        if c_dtype not in (None, "float32", "int32", "int64", "int8", "int16"):
            raise ValueError("Quantized Gemm bias dtype is not supported.")

        sa, za = ctx.qparams(a_name)
        sb, zb = ctx.qparams(b_name)
        so, zo = ctx.qparams(out_name)
        qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
        qctype = "int8_t" if out_dtype == "int8" else "int16_t"
        bias_scale = sa * sb
        sc = 0.0
        zc = 0
        if c_name is not None and c_dtype in ("int8", "int16"):
            sc, zc = ctx.qparams(c_name)

        ctx.lines.append(f"  for (size_t i = 0; i < {m}; ++i) {{")
        ctx.lines.append(f"    for (size_t j = 0; j < {n}; ++j) {{")
        ctx.lines.append("      float sum = 0.0f;")
        ctx.lines.append(f"      for (size_t t = 0; t < {k1}; ++t) {{")
        if trans_a == 0:
            ctx.lines.append(f"        size_t a_idx = i * {a_cols} + t;")
        else:
            ctx.lines.append(f"        size_t a_idx = t * {a_cols} + i;")
        if trans_b == 0:
            ctx.lines.append(f"        size_t b_idx = t * {b_cols} + j;")
        else:
            ctx.lines.append(f"        size_t b_idx = j * {b_cols} + t;")
        ctx.lines.append(f"        float ra = ((float){a}[a_idx] - {za}) * {sa:.8f}f;")
        ctx.lines.append(f"        float rb = ((float){b}[b_idx] - {zb}) * {sb:.8f}f;")
        ctx.lines.append("        sum += ra * rb;")
        ctx.lines.append("      }")
        if alpha != 1.0:
            ctx.lines.append(f"      sum *= {alpha:.8f}f;")
        if c_name is not None:
            if c_mode == "matrix":
                idx_expr = c_index
            elif c_mode == "vec_n":
                idx_expr = "j"
            elif c_mode == "vec_m":
                idx_expr = "i"
            else:
                idx_expr = "0"
            if c_dtype == "float32":
                c_expr = f"{c}[{idx_expr}]"
            elif c_dtype in ("int32", "int64"):
                c_expr = f"((float){c}[{idx_expr}] * {bias_scale:.8f}f)"
            else:
                c_expr = f"(((float){c}[{idx_expr}] - {zc}) * {sc:.8f}f)"
            if beta == 1.0:
                ctx.lines.append(f"      sum += {c_expr};")
            else:
                ctx.lines.append(f"      sum += {beta:.8f}f * ({c_expr});")
        ctx.lines.append(f"      int q = (int)roundf(sum / {so:.8f}f) + {zo};")
        ctx.lines.append(f"      if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"      if (q > {qmax}) q = {qmax};")
        ctx.lines.append(f"      {out}[i * {n} + j] = ({qctype})q;")
        ctx.lines.append("    }")
        ctx.lines.append("  }")
        return

    if out_dtype != "float32" or ctx.dtype(a_name) != "float32" or ctx.dtype(b_name) != "float32":
        raise ValueError("Gemm float path requires float32 A/B/output.")
    c_dtype = ctx.dtype(c_name) if c_name else None
    if c_dtype not in (None, "float32", "int8", "int16", "int32", "int64"):
        raise ValueError("Gemm bias dtype is unsupported.")

    ctx.lines.append(f"  for (size_t i = 0; i < {m}; ++i) {{")
    ctx.lines.append(f"    for (size_t j = 0; j < {n}; ++j) {{")
    ctx.lines.append("      float sum = 0.0f;")
    ctx.lines.append(f"      for (size_t t = 0; t < {k1}; ++t) {{")
    if trans_a == 0:
        ctx.lines.append(f"        size_t a_idx = i * {a_cols} + t;")
    else:
        ctx.lines.append(f"        size_t a_idx = t * {a_cols} + i;")
    if trans_b == 0:
        ctx.lines.append(f"        size_t b_idx = t * {b_cols} + j;")
    else:
        ctx.lines.append(f"        size_t b_idx = j * {b_cols} + t;")
    ctx.lines.append(f"        sum += (float){a}[a_idx] * (float){b}[b_idx];")
    ctx.lines.append("      }")
    if alpha != 1.0:
        ctx.lines.append(f"      sum *= {alpha:.8f}f;")
    if c_name is not None:
        if c_mode == "matrix":
            idx_expr = c_index
        elif c_mode == "vec_n":
            idx_expr = "j"
        elif c_mode == "vec_m":
            idx_expr = "i"
        else:
            idx_expr = "0"
        c_expr = f"(float){c}[{idx_expr}]"
        if beta == 1.0:
            ctx.lines.append(f"      sum += {c_expr};")
        else:
            ctx.lines.append(f"      sum += {beta:.8f}f * ({c_expr});")
    ctx.lines.append(f"      {out}[i * {n} + j] = sum;")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
