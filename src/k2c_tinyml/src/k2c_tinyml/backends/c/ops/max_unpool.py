# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import get_const_ints, tensor_size
from .registry import register_op


@register_op("MaxUnpool")
def emit_max_unpool(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 2 or len(node.inputs) > 3:
        raise ValueError("MaxUnpool expects 2 or 3 inputs.")
    if len(node.outputs) != 1:
        raise ValueError("MaxUnpool expects 1 output.")

    x_name = node.inputs[0]
    idx_name = node.inputs[1]
    shape_name = node.inputs[2] if len(node.inputs) == 3 and node.inputs[2] else None
    out_name = node.outputs[0]

    x_dtype = ctx.dtype(x_name)
    out_dtype = ctx.dtype(out_name)
    if x_dtype != out_dtype:
        raise ValueError("MaxUnpool input/output dtype must match.")
    if x_dtype not in ("float32", "int8", "int16", "int32", "int64"):
        raise ValueError("MaxUnpool dtype is unsupported.")
    if ctx.dtype(idx_name) not in ("int64", "int32"):
        raise ValueError("MaxUnpool indices dtype must be int64/int32.")

    x_shape = [int(v) for v in ctx.shape(x_name)]
    idx_shape = [int(v) for v in ctx.shape(idx_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    if len(x_shape) != 4 or len(idx_shape) != 4 or len(out_shape) != 4:
        raise ValueError("MaxUnpool currently supports 4D NCHW tensors only.")
    if x_shape != idx_shape:
        raise ValueError("MaxUnpool indices shape must match X shape.")
    if out_shape[0] != x_shape[0] or out_shape[1] != x_shape[1]:
        raise ValueError("MaxUnpool output N/C mismatch.")

    if shape_name is not None:
        shape_vals = [int(v) for v in get_const_ints(ctx.model, shape_name)]
        if shape_vals != out_shape:
            raise ValueError("MaxUnpool output_shape input must match inferred output shape.")

    n_size, c_size, in_h, in_w = x_shape
    out_h, out_w = out_shape[2], out_shape[3]
    x = ctx.map_ptr(x_name)
    idx = ctx.map_ptr(idx_name)
    out = ctx.map_ptr(out_name)
    out_size = tensor_size(out_shape)
    in_size = tensor_size(x_shape)

    ctx.lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {{ {out}[i] = ({out_dtype == 'float32' and '0.0f' or '0'}); }}")
    ctx.lines.append(f"  for (size_t n = 0; n < {n_size}; ++n) {{")
    ctx.lines.append(f"    for (size_t c = 0; c < {c_size}; ++c) {{")
    ctx.lines.append(f"      for (size_t ih = 0; ih < {in_h}; ++ih) {{")
    ctx.lines.append(f"        for (size_t iw = 0; iw < {in_w}; ++iw) {{")
    ctx.lines.append(f"          size_t in_i = ((n * {c_size} + c) * {in_h} + ih) * {in_w} + iw;")
    ctx.lines.append(f"          int64_t flat = (int64_t){idx}[in_i];")
    ctx.lines.append(f"          if (flat < 0 || flat >= (int64_t)({out_h} * {out_w})) continue;")
    ctx.lines.append(f"          size_t out_i = ((n * {c_size} + c) * {out_h} * {out_w}) + (size_t)flat;")
    ctx.lines.append(f"          {out}[out_i] = {x}[in_i];")
    ctx.lines.append("        }")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")

    # Suppress unused warning in case static analyzers infer constant-zero loops.
    ctx.lines.append(f"  (void){in_size};")
