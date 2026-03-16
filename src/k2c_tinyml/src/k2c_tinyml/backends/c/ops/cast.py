# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import emit_op_copy, tensor_size
from .registry import register_op


@register_op("Cast")
def emit_cast(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("Cast expects 1 input.")
    in_name = node.inputs[0]
    out_name = node.outputs[0]
    in_dtype = ctx.dtype(in_name)
    out_dtype = ctx.dtype(out_name)
    if in_dtype not in ("float32", "int8", "int16", "bool", "int32", "int64"):
        raise ValueError("Cast input dtype is unsupported.")
    if out_dtype not in ("float32", "int8", "int16", "bool", "int32", "int64"):
        raise ValueError("Cast output dtype is unsupported.")

    out = ctx.map_ptr(out_name)
    inp = ctx.map_ptr(in_name)
    size = tensor_size(ctx.shape(out_name))
    if in_dtype == out_dtype:
        emit_op_copy(ctx.lines, out, inp, size)
        return

    if out_dtype == "bool":
        ctx.lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
        if in_dtype == "float32":
            ctx.lines.append(f"    {out}[i] = ({inp}[i] != 0.0f) ? 1u : 0u;")
        else:
            ctx.lines.append(f"    {out}[i] = ({inp}[i] != 0) ? 1u : 0u;")
        ctx.lines.append("  }")
        return

    if out_dtype == "float32":
        ctx.lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
        ctx.lines.append(f"    {out}[i] = (float){inp}[i];")
        ctx.lines.append("  }")
        return

    if out_dtype == "int8":
        qmin, qmax = -128, 127
        ctype = "int8_t"
    elif out_dtype == "int16":
        qmin, qmax = -32768, 32767
        ctype = "int16_t"
    elif out_dtype == "int32":
        qmin, qmax = -2147483648, 2147483647
        ctype = "int32_t"
    else:
        # int64 output does not need saturation in this path.
        ctx.lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
        if in_dtype == "float32":
            ctx.lines.append(f"    {out}[i] = (int64_t){inp}[i];")
        else:
            ctx.lines.append(f"    {out}[i] = (int64_t){inp}[i];")
        ctx.lines.append("  }")
        return

    ctx.lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
    if in_dtype == "float32":
        ctx.lines.append(f"    float v = {inp}[i];")
        ctx.lines.append("    int32_t q = (int32_t)v;")
    else:
        ctx.lines.append(f"    int32_t q = (int32_t){inp}[i];")
    ctx.lines.append(f"    if (q < {qmin}) q = {qmin};")
    ctx.lines.append(f"    if (q > {qmax}) q = {qmax};")
    ctx.lines.append(f"    {out}[i] = ({ctype})q;")
    ctx.lines.append("  }")
