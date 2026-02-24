# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import product, tensor_size
from .registry import register_op


def _strides(shape: list[int]) -> list[int]:
    out = [1] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        out[i] = acc
        acc *= int(shape[i])
    return out


def _scalar_literal(value: float, dtype: str) -> str:
    if dtype == "float32":
        return f"{float(value):.8f}f"
    if dtype == "bool":
        return "1" if int(round(float(value))) != 0 else "0"
    if dtype in ("int8", "int16", "int32", "int64"):
        return str(int(round(float(value))))
    raise ValueError("OneHot output dtype is unsupported.")


@register_op("OneHot")
def emit_onehot(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 3:
        raise ValueError("OneHot expects 3 inputs: indices, depth, values.")
    idx_name, depth_name, values_name = node.inputs
    out_name = node.outputs[0]

    idx_dtype = ctx.dtype(idx_name)
    if idx_dtype not in ("int8", "int16", "int32", "int64"):
        raise ValueError("OneHot indices dtype must be integer.")
    depth_dtype = ctx.dtype(depth_name)
    if depth_dtype not in ("int8", "int16", "int32", "int64"):
        raise ValueError("OneHot depth dtype must be integer.")

    out_dtype = ctx.dtype(out_name)
    if out_dtype not in ("float32", "int8", "int16", "int32", "int64", "bool"):
        raise ValueError("OneHot output dtype is unsupported.")

    if ctx.dtype(values_name) != out_dtype:
        raise ValueError("OneHot values dtype must match output dtype.")
    values_shape = [int(v) for v in ctx.shape(values_name)]
    if product(values_shape) != 2:
        raise ValueError("OneHot values must contain exactly [off, on].")
    depth_shape = [int(v) for v in ctx.shape(depth_name)]
    if product(depth_shape) != 1:
        raise ValueError("OneHot depth must be scalar.")

    idx_shape = [int(v) for v in ctx.shape(idx_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    out_rank = len(idx_shape) + 1
    axis = int(node.attrs.get("axis", -1))
    if axis < 0:
        axis += out_rank
    if axis < 0 or axis >= out_rank:
        raise ValueError("OneHot axis out of range.")

    if len(out_shape) != out_rank:
        raise ValueError("OneHot output rank mismatch.")

    idx_size = tensor_size(idx_shape) if idx_shape else 1
    out_size = tensor_size(out_shape)
    idx_strides = _strides(idx_shape) if idx_shape else []
    out_strides = _strides(out_shape)
    expected_depth = int(out_shape[axis])
    if expected_depth <= 0:
        raise ValueError("OneHot output depth dimension must be positive.")

    idx = ctx.map_ptr(idx_name)
    depth = ctx.map_ptr(depth_name)
    values = ctx.map_ptr(values_name)
    out = ctx.map_ptr(out_name)
    idx_strides_sym = ctx.next_symbol("k2c_oh_idx_stride")
    out_strides_sym = ctx.next_symbol("k2c_oh_out_stride")

    idx_stride_vals = ", ".join(str(v) for v in idx_strides) if idx_strides else "1"
    out_stride_vals = ", ".join(str(v) for v in out_strides)
    ctx.lines.append(f"  static const int32_t {idx_strides_sym}[{max(1, len(idx_shape))}] = {{ {idx_stride_vals} }};")
    ctx.lines.append(f"  static const int32_t {out_strides_sym}[{len(out_shape)}] = {{ {out_stride_vals} }};")
    ctx.lines.append(f"  int64_t depth_v = (int64_t){depth}[0];")
    ctx.lines.append("  if (depth_v <= 0) depth_v = 1;")
    ctx.lines.append(f"  if (depth_v != {expected_depth}) depth_v = {expected_depth};")
    ctx.lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {{ {out}[i] = {values}[0]; }}")
    ctx.lines.append(f"  for (size_t linear_i = 0; linear_i < {idx_size}; ++linear_i) {{")
    ctx.lines.append(f"    int64_t class_v = (int64_t){idx}[{ '0' if len(idx_shape) == 0 else 'linear_i' }];")
    ctx.lines.append("    class_v = class_v % depth_v;")
    ctx.lines.append("    if (class_v < 0) class_v += depth_v;")
    ctx.lines.append("    size_t rem = linear_i;")
    ctx.lines.append("    size_t out_linear = 0;")
    ctx.lines.append("    size_t idx_axis_i = 0;")
    ctx.lines.append(f"    for (size_t out_axis_i = 0; out_axis_i < {out_rank}; ++out_axis_i) {{")
    ctx.lines.append("      int64_t coord = 0;")
    ctx.lines.append(f"      if ((int32_t)out_axis_i == {axis}) {{")
    ctx.lines.append("        coord = class_v;")
    ctx.lines.append("      } else {")
    ctx.lines.append("        coord = (int64_t)(rem / (size_t)" + idx_strides_sym + "[idx_axis_i]);")
    ctx.lines.append("        rem = rem % (size_t)" + idx_strides_sym + "[idx_axis_i];")
    ctx.lines.append("        idx_axis_i += 1;")
    ctx.lines.append("      }")
    ctx.lines.append("      out_linear += (size_t)coord * (size_t)" + out_strides_sym + "[out_axis_i];")
    ctx.lines.append("    }")
    ctx.lines.append(f"    {out}[out_linear] = {values}[1];")
    ctx.lines.append("  }")
