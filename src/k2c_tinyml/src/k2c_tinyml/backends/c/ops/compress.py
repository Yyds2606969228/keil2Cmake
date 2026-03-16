# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import normalize_axis, product, tensor_size
from .registry import register_op


def _to_bool_list(values: list[float], dtype: str) -> list[bool]:
    if dtype == "bool":
        return [int(v) != 0 for v in values]
    if dtype in ("int8", "int16", "int32", "int64"):
        return [int(v) != 0 for v in values]
    if dtype == "float32":
        return [float(v) != 0.0 for v in values]
    raise ValueError("Compress condition dtype is unsupported.")


@register_op("Compress")
def emit_compress(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 2:
        raise ValueError("Compress expects 2 inputs: data, condition.")
    data_name, cond_name = node.inputs
    out_name = node.outputs[0]
    if ctx.dtype(data_name) != ctx.dtype(out_name):
        raise ValueError("Compress output dtype must match data dtype.")

    data_shape = [int(v) for v in ctx.shape(data_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    if len(data_shape) <= 0:
        raise ValueError("Compress expects rank >= 1.")

    cond_tensor = ctx.model.tensors.get(cond_name)
    if cond_tensor is None or cond_tensor.data is None:
        raise ValueError("Compress currently requires constant condition tensor.")
    if len(cond_tensor.shape) != 1:
        raise ValueError("Compress condition must be 1D.")
    cond_vals = _to_bool_list(cond_tensor.data, cond_tensor.dtype)
    if len(cond_vals) <= 0:
        raise ValueError("Compress condition must be non-empty.")

    inp = ctx.map_ptr(data_name)
    out = ctx.map_ptr(out_name)
    axis_attr = node.attrs.get("axis", None)

    if axis_attr is None:
        in_size = tensor_size(data_shape)
        limit = min(in_size, len(cond_vals))
        selected = [i for i in range(limit) if cond_vals[i]]
        if len(selected) <= 0:
            raise ValueError("Compress selected count is zero; zero-dim output is unsupported.")
        if out_shape != [len(selected)]:
            raise ValueError("Compress output shape mismatch.")
        idx_sym = ctx.next_symbol("k2c_comp_idx")
        idx_vals = ", ".join(str(int(v)) for v in selected)
        ctx.lines.append(f"  static const int32_t {idx_sym}[{len(selected)}] = {{ {idx_vals} }};")
        ctx.lines.append(f"  for (size_t i = 0; i < {len(selected)}; ++i) {{")
        ctx.lines.append(f"    {out}[i] = {inp}[{idx_sym}[i]];")
        ctx.lines.append("  }")
        return

    axis = normalize_axis(int(axis_attr), len(data_shape))
    axis_dim = int(data_shape[axis])
    limit = min(axis_dim, len(cond_vals))
    selected_axis = [i for i in range(limit) if cond_vals[i]]
    if len(selected_axis) <= 0:
        raise ValueError("Compress selected count is zero; zero-dim output is unsupported.")
    expected_out = list(data_shape)
    expected_out[axis] = len(selected_axis)
    if out_shape != expected_out:
        raise ValueError("Compress output shape mismatch.")

    outer = product(data_shape[:axis]) if axis > 0 else 1
    inner = product(data_shape[axis + 1 :]) if axis + 1 < len(data_shape) else 1
    out_axis_dim = len(selected_axis)
    idx_sym = ctx.next_symbol("k2c_comp_axis_idx")
    idx_vals = ", ".join(str(int(v)) for v in selected_axis)
    ctx.lines.append(f"  static const int32_t {idx_sym}[{out_axis_dim}] = {{ {idx_vals} }};")
    ctx.lines.append(f"  for (size_t outer_i = 0; outer_i < {outer}; ++outer_i) {{")
    ctx.lines.append(f"    for (size_t out_axis_i = 0; out_axis_i < {out_axis_dim}; ++out_axis_i) {{")
    ctx.lines.append(f"      size_t src_axis_i = (size_t){idx_sym}[out_axis_i];")
    ctx.lines.append(f"      for (size_t inner_i = 0; inner_i < {inner}; ++inner_i) {{")
    ctx.lines.append(
        f"        size_t in_idx = ((outer_i * (size_t){axis_dim} + src_axis_i) * (size_t){inner}) + inner_i;"
    )
    ctx.lines.append(
        f"        size_t out_idx = ((outer_i * (size_t){out_axis_dim} + out_axis_i) * (size_t){inner}) + inner_i;"
    )
    ctx.lines.append(f"        {out}[out_idx] = {inp}[in_idx];")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
