# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import normalize_axis, tensor_size


def _strides(shape: list[int]) -> list[int]:
    if not shape:
        return []
    out = [0] * len(shape)
    stride = 1
    for axis in range(len(shape) - 1, -1, -1):
        out[axis] = stride
        stride *= int(shape[axis])
    return out


def _ctype_for_dtype(dtype: str) -> str:
    if dtype == "float32":
        return "float"
    if dtype == "int8":
        return "int8_t"
    if dtype == "int16":
        return "int16_t"
    if dtype == "int32":
        return "int32_t"
    if dtype == "int64":
        return "int64_t"
    if dtype == "bool":
        return "uint8_t"
    raise ValueError("Arg op input dtype is unsupported.")


def emit_arg_reduce(ctx: EmitContext, node: NodeInfo, op_name: str, mode: str) -> None:
    if len(node.inputs) != 1:
        raise ValueError(f"{op_name} expects 1 input.")
    data_name = node.inputs[0]
    out_name = node.outputs[0]
    in_shape = [int(v) for v in ctx.shape(data_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    rank = len(in_shape)
    if rank <= 0:
        raise ValueError(f"{op_name} requires rank >= 1.")
    axis = normalize_axis(int(node.attrs.get("axis", 0)), rank)
    keepdims = int(node.attrs.get("keepdims", 1))
    if keepdims not in (0, 1):
        raise ValueError(f"{op_name} keepdims must be 0 or 1.")
    select_last = int(node.attrs.get("select_last_index", 0))
    if select_last not in (0, 1):
        raise ValueError(f"{op_name} select_last_index must be 0 or 1.")

    expected = list(in_shape)
    if keepdims == 1:
        expected[axis] = 1
    else:
        expected = [in_shape[i] for i in range(rank) if i != axis]
    if expected != out_shape:
        raise ValueError(f"{op_name} output shape mismatch with axis/keepdims.")

    out_dtype = ctx.dtype(out_name)
    if out_dtype not in ("int64", "int32"):
        raise ValueError(f"{op_name} output dtype must be int64/int32.")
    out_ctype = "int64_t" if out_dtype == "int64" else "int32_t"
    in_ctype = _ctype_for_dtype(ctx.dtype(data_name))

    out_rank = len(out_shape)
    out_size = tensor_size(out_shape)
    axis_dim = int(in_shape[axis])
    in_dims = ", ".join(str(v) for v in in_shape)
    in_strides = _strides(in_shape)
    in_strides_vals = ", ".join(str(v) for v in in_strides)
    out_dims_vals = ", ".join(str(v) for v in out_shape) if out_rank > 0 else "1"
    out_strides = _strides(out_shape)
    out_strides_vals = ", ".join(str(v) for v in out_strides) if out_rank > 0 else "1"
    if keepdims == 1:
        out_to_in = list(range(rank))
    else:
        out_to_in = [i for i in range(rank) if i != axis]
    out_to_in_vals = ", ".join(str(v) for v in out_to_in) if out_rank > 0 else "0"

    in_dims_name = ctx.next_symbol("k2c_arg_in_dims")
    in_strides_name = ctx.next_symbol("k2c_arg_in_strides")
    out_dims_name = ctx.next_symbol("k2c_arg_out_dims")
    out_strides_name = ctx.next_symbol("k2c_arg_out_strides")
    out_to_in_name = ctx.next_symbol("k2c_arg_out_to_in")
    data = ctx.map_ptr(data_name)
    out = ctx.map_ptr(out_name)

    ctx.lines.append(f"  static const int32_t {in_dims_name}[{rank}] = {{ {in_dims} }};")
    ctx.lines.append(f"  static const int32_t {in_strides_name}[{rank}] = {{ {in_strides_vals} }};")
    ctx.lines.append(f"  static const int32_t {out_dims_name}[{max(1, out_rank)}] = {{ {out_dims_vals} }};")
    ctx.lines.append(
        f"  static const int32_t {out_strides_name}[{max(1, out_rank)}] = {{ {out_strides_vals} }};"
    )
    ctx.lines.append(f"  static const int32_t {out_to_in_name}[{max(1, out_rank)}] = {{ {out_to_in_vals} }};")

    ctx.lines.append(f"  for (size_t out_i = 0; out_i < {out_size}; ++out_i) {{")
    ctx.lines.append("    size_t tmp = out_i;")
    ctx.lines.append("    size_t base_idx = 0;")
    if out_rank > 0:
        ctx.lines.append(f"    for (int out_axis = {out_rank - 1}; out_axis >= 0; --out_axis) {{")
        ctx.lines.append(f"      size_t coord = tmp % (size_t){out_dims_name}[out_axis];")
        ctx.lines.append(f"      tmp /= (size_t){out_dims_name}[out_axis];")
        ctx.lines.append(f"      int in_axis = {out_to_in_name}[out_axis];")
        ctx.lines.append(f"      base_idx += coord * (size_t){in_strides_name}[in_axis];")
        ctx.lines.append("    }")
    ctx.lines.append("    int32_t best_k = 0;")
    ctx.lines.append(f"    {in_ctype} best_v = {data}[base_idx];")
    ctx.lines.append(f"    for (int32_t k = 1; k < {axis_dim}; ++k) {{")
    ctx.lines.append(f"      size_t in_idx = base_idx + (size_t)k * (size_t){in_strides_name}[{axis}];")
    ctx.lines.append(f"      {in_ctype} v = {data}[in_idx];")
    if mode == "max":
        if select_last == 1:
            ctx.lines.append("      if (v >= best_v) { best_v = v; best_k = k; }")
        else:
            ctx.lines.append("      if (v > best_v) { best_v = v; best_k = k; }")
    else:
        if select_last == 1:
            ctx.lines.append("      if (v <= best_v) { best_v = v; best_k = k; }")
        else:
            ctx.lines.append("      if (v < best_v) { best_v = v; best_k = k; }")
    ctx.lines.append("    }")
    ctx.lines.append(f"    {out}[out_i] = ({out_ctype})best_k;")
    ctx.lines.append("  }")
