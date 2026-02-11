# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import normalize_axis, product, tensor_size
from .registry import register_op


def _const_indices(ctx: EmitContext, name: str, axis_dim: int) -> list[int]:
    tensor = ctx.model.tensors.get(name)
    if tensor is None or tensor.data is None:
        raise ValueError("Gather currently supports constant indices only.")
    out: list[int] = []
    for v in tensor.data:
        idx = int(v)
        if idx < 0:
            idx += axis_dim
        if idx < 0 or idx >= axis_dim:
            raise ValueError("Gather index out of range.")
        out.append(idx)
    return out


@register_op("Gather")
def emit_gather(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 2:
        raise ValueError("Gather expects 2 inputs.")
    data_name = node.inputs[0]
    idx_name = node.inputs[1]
    out_name = node.outputs[0]
    if ctx.dtype(data_name) != ctx.dtype(out_name):
        raise ValueError("Gather output dtype must match data dtype.")

    data_shape = ctx.shape(data_name)
    idx_shape = ctx.shape(idx_name)
    out_shape = ctx.shape(out_name)
    if len(data_shape) == 0:
        raise ValueError("Gather does not support scalar input.")
    if len(idx_shape) != 1:
        raise ValueError("Gather currently supports 1D indices only.")

    axis = normalize_axis(int(node.attrs.get("axis", 0)), len(data_shape))
    axis_dim = int(data_shape[axis])
    indices = _const_indices(ctx, idx_name, axis_dim)
    k = int(idx_shape[0])
    if k != len(indices):
        raise ValueError("Gather indices shape mismatch.")

    outer = product(data_shape[:axis]) if axis > 0 else 1
    inner = product(data_shape[axis + 1 :]) if axis + 1 < len(data_shape) else 1
    expected_out = outer * k * inner
    if tensor_size(out_shape) != expected_out:
        raise ValueError("Gather output shape mismatch.")

    idx_sym = ctx.next_symbol("k2c_gather_idx")
    data = ctx.map_ptr(data_name)
    out = ctx.map_ptr(out_name)
    idx_vals = ", ".join(str(v) for v in indices)
    ctx.lines.append(f"  static const int32_t {idx_sym}[{k}] = {{ {idx_vals} }};")
    ctx.lines.append(f"  for (size_t outer_i = 0; outer_i < {outer}; ++outer_i) {{")
    ctx.lines.append(f"    for (size_t k_i = 0; k_i < {k}; ++k_i) {{")
    ctx.lines.append(f"      size_t src_base = (outer_i * {axis_dim} + (size_t){idx_sym}[k_i]) * {inner};")
    ctx.lines.append(f"      size_t dst_base = (outer_i * {k} + k_i) * {inner};")
    ctx.lines.append(f"      for (size_t inner_i = 0; inner_i < {inner}; ++inner_i) {{")
    ctx.lines.append(f"        {out}[dst_base + inner_i] = {data}[src_base + inner_i];")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
