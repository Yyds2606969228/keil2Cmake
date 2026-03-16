# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import tensor_size
from .registry import register_op


@register_op("Sign")
def emit_sign(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("Sign expects 1 input.")
    out_tensor = node.outputs[0]
    out_dtype = ctx.dtype(out_tensor)
    in_name = node.inputs[0]
    in_dtype = ctx.dtype(in_name)
    out = ctx.map_ptr(out_tensor)
    inp = ctx.map_ptr(in_name)
    size = tensor_size(ctx.shape(out_tensor))

    if in_dtype != out_dtype:
        raise ValueError("Sign requires same input/output dtype.")

    if out_dtype in ("int8", "int16"):
        ctx.lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
        ctx.lines.append(f"    int v = (int){inp}[i];")
        ctx.lines.append(f"    {out}[i] = v > 0 ? 1 : (v < 0 ? -1 : 0);")
        ctx.lines.append("  }")
        return

    if out_dtype == "float32":
        ctx.lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
        ctx.lines.append(f"    float v = {inp}[i];")
        ctx.lines.append(f"    {out}[i] = v > 0.0f ? 1.0f : (v < 0.0f ? -1.0f : 0.0f);")
        ctx.lines.append("  }")
        return

    if out_dtype == "int32":
        ctx.lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
        ctx.lines.append(f"    int32_t v = {inp}[i];")
        ctx.lines.append(f"    {out}[i] = v > 0 ? 1 : (v < 0 ? -1 : 0);")
        ctx.lines.append("  }")
        return

    if out_dtype == "int64":
        ctx.lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
        ctx.lines.append(f"    int64_t v = {inp}[i];")
        ctx.lines.append(f"    {out}[i] = v > 0 ? 1 : (v < 0 ? -1 : 0);")
        ctx.lines.append("  }")
        return

    raise ValueError("Sign supports float32/int8/int16/int32/int64 only.")
