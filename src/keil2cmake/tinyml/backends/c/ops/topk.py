# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import normalize_axis, product
from .registry import register_op


def _index_ctype(dtype: str) -> str:
    if dtype == "int64":
        return "int64_t"
    if dtype == "int32":
        return "int32_t"
    raise ValueError("TopK indices dtype must be int64/int32.")


@register_op("TopK")
def emit_topk(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 2:
        raise ValueError("TopK expects 2 inputs.")
    if len(node.outputs) != 2:
        raise ValueError("TopK currently requires 2 outputs (values, indices).")
    x_name, k_name = node.inputs[0], node.inputs[1]
    v_name, i_name = node.outputs[0], node.outputs[1]

    x_dtype = ctx.dtype(x_name)
    v_dtype = ctx.dtype(v_name)
    if x_dtype != v_dtype:
        raise ValueError("TopK values dtype must match input dtype.")
    if x_dtype not in ("float32", "int8", "int16", "int32", "int64"):
        raise ValueError("TopK input dtype is unsupported.")
    idx_ctype = _index_ctype(ctx.dtype(i_name))

    k_tensor = ctx.model.tensors.get(k_name)
    if k_tensor is None or k_tensor.data is None or len(k_tensor.data) == 0:
        raise ValueError("TopK K input must be constant.")
    k = int(k_tensor.data[0])
    if k <= 0:
        raise ValueError("TopK K must be positive.")

    x_shape = [int(v) for v in ctx.shape(x_name)]
    v_shape = [int(v) for v in ctx.shape(v_name)]
    i_shape = [int(v) for v in ctx.shape(i_name)]
    if v_shape != i_shape:
        raise ValueError("TopK values/indices output shapes must match.")
    rank = len(x_shape)
    if rank <= 0:
        raise ValueError("TopK input rank must be >= 1.")
    axis = normalize_axis(int(node.attrs.get("axis", -1)), rank)
    if k > x_shape[axis]:
        raise ValueError("TopK K exceeds axis dimension.")

    expected = list(x_shape)
    expected[axis] = k
    if v_shape != expected:
        raise ValueError("TopK output shape mismatch.")

    largest = int(node.attrs.get("largest", 1))
    sorted_flag = int(node.attrs.get("sorted", 1))
    if largest not in (0, 1):
        raise ValueError("TopK largest must be 0 or 1.")
    if sorted_flag not in (0, 1):
        raise ValueError("TopK sorted must be 0 or 1.")

    outer = product(x_shape[:axis]) if axis > 0 else 1
    axis_dim = int(x_shape[axis])
    inner = product(x_shape[axis + 1 :]) if axis + 1 < rank else 1
    x = ctx.map_ptr(x_name)
    v = ctx.map_ptr(v_name)
    iout = ctx.map_ptr(i_name)

    ctx.lines.append(f"  for (size_t outer_i = 0; outer_i < {outer}; ++outer_i) {{")
    ctx.lines.append(f"    for (size_t inner_i = 0; inner_i < {inner}; ++inner_i) {{")
    ctx.lines.append(f"      for (size_t top_i = 0; top_i < {k}; ++top_i) {{")
    ctx.lines.append("        int has_best = 0;")
    ctx.lines.append("        float best_val = 0.0f;")
    ctx.lines.append("        size_t best_idx = 0;")
    ctx.lines.append(f"        for (size_t axis_i = 0; axis_i < {axis_dim}; ++axis_i) {{")
    ctx.lines.append("          int used = 0;")
    ctx.lines.append("          for (size_t prev_i = 0; prev_i < top_i; ++prev_i) {")
    ctx.lines.append(
        f"            size_t prev_out = (outer_i * {k} + prev_i) * {inner} + inner_i;"
    )
    ctx.lines.append(f"            if ((size_t){iout}[prev_out] == axis_i) {{ used = 1; break; }}")
    ctx.lines.append("          }")
    ctx.lines.append("          if (used) continue;")
    ctx.lines.append(
        f"          size_t src_idx = (outer_i * {axis_dim} + axis_i) * {inner} + inner_i;"
    )
    ctx.lines.append(f"          float cur = (float){x}[src_idx];")
    ctx.lines.append("          if (!has_best) {")
    ctx.lines.append("            has_best = 1;")
    ctx.lines.append("            best_val = cur;")
    ctx.lines.append("            best_idx = axis_i;")
    ctx.lines.append("          } else {")
    if largest == 1:
        ctx.lines.append(
            "            if (cur > best_val || (cur == best_val && axis_i < best_idx)) {"
        )
    else:
        ctx.lines.append(
            "            if (cur < best_val || (cur == best_val && axis_i < best_idx)) {"
        )
    ctx.lines.append("              best_val = cur;")
    ctx.lines.append("              best_idx = axis_i;")
    ctx.lines.append("            }")
    ctx.lines.append("          }")
    ctx.lines.append("        }")
    ctx.lines.append(
        f"        size_t out_idx = (outer_i * {k} + top_i) * {inner} + inner_i;"
    )
    ctx.lines.append(f"        {iout}[out_idx] = ({idx_ctype})best_idx;")
    ctx.lines.append(
        f"        size_t src_best = (outer_i * {axis_dim} + best_idx) * {inner} + inner_i;"
    )
    ctx.lines.append(f"        {v}[out_idx] = {x}[src_best];")
    ctx.lines.append("      }")
    if sorted_flag == 0:
        ctx.lines.append("      (void)0;")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
