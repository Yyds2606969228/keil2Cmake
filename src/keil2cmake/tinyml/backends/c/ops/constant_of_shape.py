# -*- coding: utf-8 -*-

from __future__ import annotations

from onnx import numpy_helper

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import tensor_size
from .registry import register_op


def _parse_value(node: NodeInfo) -> float:
    if "value" not in node.attrs:
        return 0.0
    attr = node.attrs["value"]
    if isinstance(attr, (int, float, bool)):
        return float(attr)
    if isinstance(attr, (list, tuple)):
        if not attr:
            return 0.0
        return float(attr[0])
    arr = numpy_helper.to_array(attr)
    if arr.size == 0:
        return 0.0
    return float(arr.reshape(-1)[0])


@register_op("ConstantOfShape")
def emit_constant_of_shape(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("ConstantOfShape expects 1 input.")
    out_name = node.outputs[0]
    out_dtype = ctx.dtype(out_name)
    out_shape = ctx.shape(out_name)
    size = tensor_size(out_shape)
    out = ctx.map_ptr(out_name)
    value = _parse_value(node)

    if out_dtype == "float32":
        expr = f"{value:.8f}f"
    elif out_dtype == "bool":
        expr = "1" if value != 0.0 else "0"
    elif out_dtype in ("int8", "int16", "int32", "int64"):
        expr = str(int(round(value)))
    else:
        raise ValueError("ConstantOfShape output dtype is unsupported.")

    ctx.lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
    ctx.lines.append(f"    {out}[i] = {expr};")
    ctx.lines.append("  }")
