# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import normalize_axis, product
from .registry import register_op


def _strides(shape: list[int]) -> list[int]:
    out = [1] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        out[i] = acc
        acc *= int(shape[i])
    return out


@register_op("ReverseSequence")
def emit_reverse_sequence(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 2:
        raise ValueError("ReverseSequence expects 2 inputs.")
    x_name, seq_name = node.inputs
    out_name = node.outputs[0]

    x_dtype = ctx.dtype(x_name)
    out_dtype = ctx.dtype(out_name)
    if x_dtype != out_dtype:
        raise ValueError("ReverseSequence requires matching input/output dtype.")
    if x_dtype not in ("float32", "bool", "int8", "int16", "int32", "int64"):
        raise ValueError("ReverseSequence dtype is unsupported.")
    if ctx.dtype(seq_name) not in ("int8", "int16", "int32", "int64"):
        raise ValueError("ReverseSequence sequence_lens must be integer.")

    x_shape = [int(v) for v in ctx.shape(x_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    if x_shape != out_shape:
        raise ValueError("ReverseSequence output shape mismatch.")
    rank = len(x_shape)
    if rank < 2:
        raise ValueError("ReverseSequence requires rank >= 2.")

    batch_axis = normalize_axis(int(node.attrs.get("batch_axis", 1)), rank)
    time_axis = normalize_axis(int(node.attrs.get("time_axis", 0)), rank)
    if batch_axis == time_axis:
        raise ValueError("ReverseSequence batch_axis and time_axis must differ.")

    seq_tensor = ctx.model.tensors.get(seq_name)
    if seq_tensor is None or seq_tensor.data is None:
        raise ValueError("ReverseSequence currently requires constant sequence_lens.")
    batch_dim = int(x_shape[batch_axis])
    time_dim = int(x_shape[time_axis])
    seq_raw = [int(v) for v in seq_tensor.data]
    if len(seq_raw) != batch_dim:
        raise ValueError("ReverseSequence sequence_lens size mismatch.")
    seq_vals = [min(max(v, 0), time_dim) for v in seq_raw]

    other_dims = [x_shape[i] for i in range(rank) if i not in (batch_axis, time_axis)]
    other_size = product(other_dims) if other_dims else 1
    strides = _strides(x_shape)
    x = ctx.map_ptr(x_name)
    out = ctx.map_ptr(out_name)
    seq_sym = ctx.next_symbol("k2c_revseq_lens")
    seq_init = ", ".join(str(v) for v in seq_vals)
    ctx.lines.append(f"  static const int32_t {seq_sym}[{batch_dim}] = {{ {seq_init} }};")
    ctx.lines.append(f"  for (size_t b = 0; b < {batch_dim}; ++b) {{")
    ctx.lines.append(f"    int32_t seq = {seq_sym}[b];")
    ctx.lines.append(f"    for (size_t t = 0; t < {time_dim}; ++t) {{")
    ctx.lines.append("      size_t src_t = (t < (size_t)seq) ? ((size_t)seq - 1 - t) : t;")
    ctx.lines.append(f"      for (size_t other_i = 0; other_i < {other_size}; ++other_i) {{")
    ctx.lines.append("        size_t rem = other_i;")
    ctx.lines.append(
        f"        size_t dst = b * (size_t){strides[batch_axis]} + t * (size_t){strides[time_axis]};"
    )
    ctx.lines.append(
        f"        size_t src = b * (size_t){strides[batch_axis]} + src_t * (size_t){strides[time_axis]};"
    )
    for dim_i in range(rank - 1, -1, -1):
        if dim_i in (batch_axis, time_axis):
            continue
        dim_v = int(x_shape[dim_i])
        stride_v = int(strides[dim_i])
        ctx.lines.append(f"        size_t coord_{dim_i} = rem % (size_t){dim_v};")
        ctx.lines.append(f"        rem /= (size_t){dim_v};")
        ctx.lines.append(f"        dst += coord_{dim_i} * (size_t){stride_v};")
        ctx.lines.append(f"        src += coord_{dim_i} * (size_t){stride_v};")
    ctx.lines.append(f"        {out}[dst] = {x}[src];")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")
