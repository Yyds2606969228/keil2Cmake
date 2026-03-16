# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op


def _equation_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").replace(" ", "")
    return str(value).replace(" ", "")


@register_op("Einsum")
def emit_einsum(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 2:
        raise ValueError("Einsum expects 2 inputs in this implementation.")
    if len(node.outputs) != 1:
        raise ValueError("Einsum expects 1 output.")

    eq = _equation_text(node.attrs.get("equation"))
    if eq not in ("ij,jk->ik", "bij,bjk->bik", "bij,jk->bik", "ij,bjk->bik"):
        raise ValueError(
            "Einsum supports equations: ij,jk->ik / bij,bjk->bik / bij,jk->bik / ij,bjk->bik."
        )

    a_name, b_name = node.inputs
    out_name = node.outputs[0]
    a_shape = [int(v) for v in ctx.shape(a_name)]
    b_shape = [int(v) for v in ctx.shape(b_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    out_dtype = ctx.dtype(out_name)
    if out_dtype not in ("float32", "int8", "int16"):
        raise ValueError("Einsum supports float32/int8/int16 only.")
    is_quant = out_dtype in ("int8", "int16")
    if is_quant:
        if ctx.dtype(a_name) != out_dtype or ctx.dtype(b_name) != out_dtype:
            raise ValueError("Quantized Einsum requires matching dtypes.")
        sa, za = ctx.qparams(a_name)
        sb, zb = ctx.qparams(b_name)
        so, zo = ctx.qparams(out_name)
        qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
        qctype = "int8_t" if out_dtype == "int8" else "int16_t"
    else:
        if ctx.dtype(a_name) != "float32" or ctx.dtype(b_name) != "float32":
            raise ValueError("Float Einsum requires float32 inputs.")

    a = ctx.map_ptr(a_name)
    b = ctx.map_ptr(b_name)
    out = ctx.map_ptr(out_name)

    if eq == "ij,jk->ik":
        if len(a_shape) != 2 or len(b_shape) != 2 or len(out_shape) != 2:
            raise ValueError("Einsum ij,jk->ik expects 2D/2D->2D.")
        m, k = a_shape
        kb, n = b_shape
        mo, no = out_shape
        if k != kb or m != mo or n != no:
            raise ValueError("Einsum ij,jk->ik shape mismatch.")
        ctx.lines.append(f"  for (size_t i = 0; i < {m}; ++i) {{")
        ctx.lines.append(f"    for (size_t j = 0; j < {n}; ++j) {{")
        ctx.lines.append("      float sum = 0.0f;")
        ctx.lines.append(f"      for (size_t t = 0; t < {k}; ++t) {{")
        if is_quant:
            ctx.lines.append(
                f"        float ra = ((float){a}[i * {k} + t] - {za}) * {sa:.8f}f;"
            )
            ctx.lines.append(
                f"        float rb = ((float){b}[t * {n} + j] - {zb}) * {sb:.8f}f;"
            )
            ctx.lines.append("        sum += ra * rb;")
        else:
            ctx.lines.append(f"        sum += {a}[i * {k} + t] * {b}[t * {n} + j];")
        ctx.lines.append("      }")
        if is_quant:
            ctx.lines.append(f"      int q = (int)roundf(sum / {so:.8f}f) + {zo};")
            ctx.lines.append(f"      if (q < {qmin}) q = {qmin};")
            ctx.lines.append(f"      if (q > {qmax}) q = {qmax};")
            ctx.lines.append(f"      {out}[i * {n} + j] = ({qctype})q;")
        else:
            ctx.lines.append(f"      {out}[i * {n} + j] = sum;")
        ctx.lines.append("    }")
        ctx.lines.append("  }")
        return

    # 3D output family: bij,bjk->bik / bij,jk->bik / ij,bjk->bik
    if len(out_shape) != 3:
        raise ValueError("Einsum 3D variants require 3D output.")
    bdim, m, n = out_shape
    if eq == "bij,bjk->bik":
        if len(a_shape) != 3 or len(b_shape) != 3:
            raise ValueError("Einsum bij,bjk->bik expects 3D and 3D inputs.")
        ba, ma, k = a_shape
        bb, kb, nb = b_shape
        if ba != bdim or bb != bdim or ma != m or nb != n or k != kb:
            raise ValueError("Einsum bij,bjk->bik shape mismatch.")
        a_base = "b_i * " + str(m) + " * " + str(k)
        b_base = "b_i * " + str(k) + " * " + str(n)
    elif eq == "bij,jk->bik":
        if len(a_shape) != 3 or len(b_shape) != 2:
            raise ValueError("Einsum bij,jk->bik expects 3D and 2D inputs.")
        ba, ma, k = a_shape
        kb, nb = b_shape
        if ba != bdim or ma != m or nb != n or k != kb:
            raise ValueError("Einsum bij,jk->bik shape mismatch.")
        a_base = "b_i * " + str(m) + " * " + str(k)
        b_base = "0"
    else:  # ij,bjk->bik
        if len(a_shape) != 2 or len(b_shape) != 3:
            raise ValueError("Einsum ij,bjk->bik expects 2D and 3D inputs.")
        ma, k = a_shape
        bb, kb, nb = b_shape
        if bb != bdim or ma != m or nb != n or k != kb:
            raise ValueError("Einsum ij,bjk->bik shape mismatch.")
        a_base = "0"
        b_base = "b_i * " + str(k) + " * " + str(n)

    ctx.lines.append(f"  for (size_t b_i = 0; b_i < {bdim}; ++b_i) {{")
    ctx.lines.append(f"    for (size_t i = 0; i < {m}; ++i) {{")
    ctx.lines.append(f"      for (size_t j = 0; j < {n}; ++j) {{")
    ctx.lines.append("        float sum = 0.0f;")
    ctx.lines.append(f"        for (size_t t = 0; t < {k}; ++t) {{")
    if is_quant:
        ctx.lines.append(
            f"          float ra = ((float){a}[{a_base} + i * {k} + t] - {za}) * {sa:.8f}f;"
        )
        ctx.lines.append(
            f"          float rb = ((float){b}[{b_base} + t * {n} + j] - {zb}) * {sb:.8f}f;"
        )
        ctx.lines.append("          sum += ra * rb;")
    else:
        ctx.lines.append(
            f"          sum += {a}[{a_base} + i * {k} + t] * {b}[{b_base} + t * {n} + j];"
        )
    ctx.lines.append("        }")
    if is_quant:
        ctx.lines.append(f"        int q = (int)roundf(sum / {so:.8f}f) + {zo};")
        ctx.lines.append(f"        if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"        if (q > {qmax}) q = {qmax};")
        ctx.lines.append(f"        {out}[(b_i * {m} + i) * {n} + j] = ({qctype})q;")
    else:
        ctx.lines.append(f"        {out}[(b_i * {m} + i) * {n} + j] = sum;")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
