# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import tensor_size
from .registry import register_op


def _value_ctype(dtype: str) -> str:
    if dtype == "float32":
        return "float"
    if dtype == "bool":
        return "uint8_t"
    if dtype == "int8":
        return "int8_t"
    if dtype == "int16":
        return "int16_t"
    if dtype == "int32":
        return "int32_t"
    if dtype == "int64":
        return "int64_t"
    raise ValueError("Unique input dtype is unsupported.")


def _index_ctype(dtype: str) -> str:
    if dtype == "int64":
        return "int64_t"
    if dtype == "int32":
        return "int32_t"
    raise ValueError("Unique index output dtype must be int64/int32.")


@register_op("Unique")
def emit_unique(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1:
        raise ValueError("Unique expects 1 input.")
    if len(node.outputs) < 1 or len(node.outputs) > 4:
        raise ValueError("Unique expects 1..4 outputs.")
    if node.attrs.get("axis", None) is not None:
        raise ValueError("Unique currently supports axis=None only.")

    x_name = node.inputs[0]
    y_name = node.outputs[0]
    idx_name = node.outputs[1] if len(node.outputs) >= 2 and node.outputs[1] else None
    inv_name = node.outputs[2] if len(node.outputs) >= 3 and node.outputs[2] else None
    cnt_name = node.outputs[3] if len(node.outputs) >= 4 and node.outputs[3] else None

    x_dtype = ctx.dtype(x_name)
    if x_dtype not in ("float32", "bool", "int8", "int16", "int32", "int64"):
        raise ValueError("Unique input dtype is unsupported.")
    if ctx.dtype(y_name) != x_dtype:
        raise ValueError("Unique Y dtype must match input dtype.")
    y_shape = [int(v) for v in ctx.shape(y_name)]
    if len(y_shape) != 1:
        raise ValueError("Unique currently supports 1D Y output.")

    x_shape = [int(v) for v in ctx.shape(x_name)]
    x_size = tensor_size(x_shape)
    if idx_name is not None:
        idx_ctype = _index_ctype(ctx.dtype(idx_name))
        idx_shape = [int(v) for v in ctx.shape(idx_name)]
        if len(idx_shape) != 1:
            raise ValueError("Unique indices output must be 1D.")
    else:
        idx_ctype = "int64_t"
        idx_shape = []
    if inv_name is not None:
        inv_ctype = _index_ctype(ctx.dtype(inv_name))
        inv_shape = [int(v) for v in ctx.shape(inv_name)]
        if len(inv_shape) != 1:
            raise ValueError("Unique inverse_indices output must be 1D.")
    else:
        inv_ctype = "int64_t"
        inv_shape = []
    if cnt_name is not None:
        cnt_ctype = _index_ctype(ctx.dtype(cnt_name))
        cnt_shape = [int(v) for v in ctx.shape(cnt_name)]
        if len(cnt_shape) != 1:
            raise ValueError("Unique counts output must be 1D.")
    else:
        cnt_ctype = "int64_t"
        cnt_shape = []

    y_cap = int(y_shape[0])
    if y_cap <= 0:
        raise ValueError("Unique Y capacity must be positive.")
    if idx_name is not None and int(idx_shape[0]) <= 0:
        raise ValueError("Unique indices capacity must be positive.")
    if inv_name is not None and int(inv_shape[0]) <= 0:
        raise ValueError("Unique inverse_indices capacity must be positive.")
    if cnt_name is not None and int(cnt_shape[0]) <= 0:
        raise ValueError("Unique counts capacity must be positive.")

    sorted_attr = int(node.attrs.get("sorted", 1))
    if sorted_attr not in (0, 1):
        raise ValueError("Unique sorted must be 0 or 1.")

    x = ctx.map_ptr(x_name)
    y = ctx.map_ptr(y_name)
    idx = ctx.map_ptr(idx_name) if idx_name is not None else None
    inv = ctx.map_ptr(inv_name) if inv_name is not None else None
    cnt = ctx.map_ptr(cnt_name) if cnt_name is not None else None

    val_ctype = _value_ctype(x_dtype)
    order_sym = ctx.next_symbol("k2c_unique_order")
    uvals_sym = ctx.next_symbol("k2c_unique_vals")
    first_sym = ctx.next_symbol("k2c_unique_first")
    counts_sym = ctx.next_symbol("k2c_unique_counts")
    invbuf_sym = ctx.next_symbol("k2c_unique_inv")
    remap_sym = ctx.next_symbol("k2c_unique_remap")

    ctx.lines.append(f"  {val_ctype} {uvals_sym}[{x_size}];")
    ctx.lines.append(f"  int32_t {first_sym}[{x_size}];")
    ctx.lines.append(f"  int32_t {counts_sym}[{x_size}];")
    ctx.lines.append(f"  int32_t {invbuf_sym}[{x_size}];")
    ctx.lines.append(f"  int32_t {order_sym}[{x_size}];")
    ctx.lines.append(f"  int32_t {remap_sym}[{x_size}];")
    ctx.lines.append("  int32_t unique_count = 0;")
    ctx.lines.append(f"  for (size_t i = 0; i < {x_size}; ++i) {{")
    ctx.lines.append(f"    {val_ctype} v = {x}[i];")
    ctx.lines.append("    int32_t found = -1;")
    ctx.lines.append("    for (int32_t u = 0; u < unique_count; ++u) {")
    if x_dtype == "float32":
        ctx.lines.append(f"      if ({uvals_sym}[u] == v) {{ found = u; break; }}")
    else:
        ctx.lines.append(f"      if ({uvals_sym}[u] == v) {{ found = u; break; }}")
    ctx.lines.append("    }")
    ctx.lines.append("    if (found < 0) {")
    ctx.lines.append(f"      {uvals_sym}[unique_count] = v;")
    ctx.lines.append(f"      {first_sym}[unique_count] = (int32_t)i;")
    ctx.lines.append(f"      {counts_sym}[unique_count] = 1;")
    ctx.lines.append(f"      {invbuf_sym}[i] = unique_count;")
    ctx.lines.append("      unique_count += 1;")
    ctx.lines.append("    } else {")
    ctx.lines.append(f"      {counts_sym}[found] += 1;")
    ctx.lines.append(f"      {invbuf_sym}[i] = found;")
    ctx.lines.append("    }")
    ctx.lines.append("  }")

    ctx.lines.append("  for (int32_t u = 0; u < unique_count; ++u) {")
    ctx.lines.append(f"    {order_sym}[u] = u;")
    ctx.lines.append("  }")
    if sorted_attr == 1:
        ctx.lines.append("  for (int32_t i = 0; i < unique_count; ++i) {")
        ctx.lines.append("    for (int32_t j = i + 1; j < unique_count; ++j) {")
        ctx.lines.append(f"      int32_t oi = {order_sym}[i];")
        ctx.lines.append(f"      int32_t oj = {order_sym}[j];")
        ctx.lines.append(f"      if ({uvals_sym}[oj] < {uvals_sym}[oi]) {{")
        ctx.lines.append(f"        int32_t t = {order_sym}[i];")
        ctx.lines.append(f"        {order_sym}[i] = {order_sym}[j];")
        ctx.lines.append(f"        {order_sym}[j] = t;")
        ctx.lines.append("      }")
        ctx.lines.append("    }")
        ctx.lines.append("  }")
    ctx.lines.append("  for (int32_t i = 0; i < unique_count; ++i) {")
    ctx.lines.append(f"    {remap_sym}[{order_sym}[i]] = i;")
    ctx.lines.append("  }")

    ctx.lines.append(f"  for (size_t i = 0; i < {y_cap}; ++i) {{")
    ctx.lines.append("    if ((int32_t)i < unique_count) {")
    ctx.lines.append(f"      int32_t u = {order_sym}[i];")
    ctx.lines.append(f"      {y}[i] = {uvals_sym}[u];")
    ctx.lines.append("    } else {")
    if x_dtype == "float32":
        ctx.lines.append(f"      {y}[i] = 0.0f;")
    else:
        ctx.lines.append(f"      {y}[i] = ({val_ctype})0;")
    ctx.lines.append("    }")
    ctx.lines.append("  }")

    if idx is not None:
        idx_cap = int(idx_shape[0])
        ctx.lines.append(f"  for (size_t i = 0; i < {idx_cap}; ++i) {{")
        ctx.lines.append("    if ((int32_t)i < unique_count) {")
        ctx.lines.append(f"      int32_t u = {order_sym}[i];")
        ctx.lines.append(f"      {idx}[i] = ({idx_ctype}){first_sym}[u];")
        ctx.lines.append("    } else {")
        ctx.lines.append(f"      {idx}[i] = ({idx_ctype})-1;")
        ctx.lines.append("    }")
        ctx.lines.append("  }")

    if inv is not None:
        inv_cap = int(inv_shape[0])
        ctx.lines.append(f"  for (size_t i = 0; i < {inv_cap}; ++i) {{")
        ctx.lines.append(f"    if (i < {x_size}) {{")
        ctx.lines.append(f"      int32_t u = {invbuf_sym}[i];")
        ctx.lines.append(f"      {inv}[i] = ({inv_ctype}){remap_sym}[u];")
        ctx.lines.append("    } else {")
        ctx.lines.append(f"      {inv}[i] = ({inv_ctype})-1;")
        ctx.lines.append("    }")
        ctx.lines.append("  }")

    if cnt is not None:
        cnt_cap = int(cnt_shape[0])
        ctx.lines.append(f"  for (size_t i = 0; i < {cnt_cap}; ++i) {{")
        ctx.lines.append("    if ((int32_t)i < unique_count) {")
        ctx.lines.append(f"      int32_t u = {order_sym}[i];")
        ctx.lines.append(f"      {cnt}[i] = ({cnt_ctype}){counts_sym}[u];")
        ctx.lines.append("    } else {")
        ctx.lines.append(f"      {cnt}[i] = ({cnt_ctype})0;")
        ctx.lines.append("    }")
        ctx.lines.append("  }")
