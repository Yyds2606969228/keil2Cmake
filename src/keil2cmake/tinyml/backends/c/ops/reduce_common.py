# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import get_const_ints, normalize_axis, tensor_size


def _strides(shape: list[int]) -> list[int]:
    if not shape:
        return []
    out = [0] * len(shape)
    stride = 1
    for axis in range(len(shape) - 1, -1, -1):
        out[axis] = stride
        stride *= int(shape[axis])
    return out


def _parse_axes(ctx: EmitContext, node: NodeInfo, rank: int) -> list[int]:
    axes = node.attrs.get("axes")
    if axes is None and len(node.inputs) >= 2 and node.inputs[1]:
        axes = get_const_ints(ctx.model, node.inputs[1])
    if axes is None:
        return list(range(rank))
    seen: set[int] = set()
    out: list[int] = []
    for v in axes:
        axis = normalize_axis(int(v), rank)
        if axis in seen:
            continue
        seen.add(axis)
        out.append(axis)
    return out


def _expected_reduce_shape(in_shape: list[int], axes: list[int], keepdims: int) -> list[int]:
    axis_set = set(axes)
    if keepdims == 1:
        return [1 if i in axis_set else int(in_shape[i]) for i in range(len(in_shape))]
    return [int(in_shape[i]) for i in range(len(in_shape)) if i not in axis_set]


def emit_reduce(ctx: EmitContext, node: NodeInfo, op_name: str, mode: str) -> None:
    if len(node.inputs) < 1:
        raise ValueError(f"{op_name} expects at least 1 input.")
    data_name = node.inputs[0]
    out_name = node.outputs[0]
    in_shape = [int(v) for v in ctx.shape(data_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    in_dtype = ctx.dtype(data_name)
    out_dtype = ctx.dtype(out_name)
    rank = len(in_shape)
    if rank <= 0:
        raise ValueError(f"{op_name} requires rank >= 1.")

    keepdims = int(node.attrs.get("keepdims", 1))
    if keepdims not in (0, 1):
        raise ValueError(f"{op_name} keepdims must be 0 or 1.")
    axes = _parse_axes(ctx, node, rank)
    expected = _expected_reduce_shape(in_shape, axes, keepdims)
    if expected != out_shape:
        raise ValueError(f"{op_name} output shape mismatch with axes/keepdims.")

    in_size = tensor_size(in_shape)
    out_size = tensor_size(out_shape)
    out_rank = len(out_shape)
    axis_set = set(axes)
    reduce_count = 1
    for axis in axes:
        reduce_count *= int(in_shape[axis])

    in_dims_name = ctx.next_symbol("k2c_reduce_in_dims")
    reduce_mask_name = ctx.next_symbol("k2c_reduce_mask")
    out_strides_name = ctx.next_symbol("k2c_reduce_out_strides")
    in_dims_vals = ", ".join(str(int(v)) for v in in_shape)
    reduce_mask_vals = ", ".join("1" if i in axis_set else "0" for i in range(rank))
    out_strides = _strides(out_shape)
    if out_rank > 0:
        out_strides_vals = ", ".join(str(int(v)) for v in out_strides)
        ctx.lines.append(f"  static const int32_t {out_strides_name}[{out_rank}] = {{ {out_strides_vals} }};")
    else:
        ctx.lines.append(f"  static const int32_t {out_strides_name}[1] = {{ 1 }};")
    ctx.lines.append(f"  static const int32_t {in_dims_name}[{rank}] = {{ {in_dims_vals} }};")
    ctx.lines.append(f"  static const int32_t {reduce_mask_name}[{rank}] = {{ {reduce_mask_vals} }};")

    inp = ctx.map_ptr(data_name)
    out = ctx.map_ptr(out_name)

    if in_dtype in ("int8", "int16") or out_dtype in ("int8", "int16"):
        if in_dtype != out_dtype or in_dtype not in ("int8", "int16"):
            raise ValueError(f"{op_name} quantized path requires matching int8/int16 input/output.")
        si, zi = ctx.qparams(data_name)
        so, zo = ctx.qparams(out_name)
        qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
        out_ctype = "int8_t" if out_dtype == "int8" else "int16_t"
        acc_name = ctx.next_symbol("k2c_reduce_acc")

        if mode in ("sum", "mean", "l1", "l2", "sum_square", "log_sum", "log_sum_exp"):
            ctx.lines.append(f"  float {acc_name}[{out_size}] = {{0}};")
        elif mode == "prod":
            ctx.lines.append(f"  float {acc_name}[{out_size}];")
            ctx.lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {acc_name}[i] = 1.0f;")
        elif mode == "max":
            ctx.lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {acc_name}[i] = -3.402823466e+38F;")
        elif mode == "min":
            ctx.lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {acc_name}[i] = 3.402823466e+38F;")
        else:
            raise ValueError(f"{op_name} mode is invalid.")

        ctx.lines.append(f"  for (size_t i = 0; i < {in_size}; ++i) {{")
        ctx.lines.append("    size_t tmp = i;")
        ctx.lines.append("    size_t out_idx = 0;")
        if keepdims == 0 and out_rank > 0:
            ctx.lines.append(f"    int out_axis = {out_rank - 1};")
        ctx.lines.append(f"    for (int axis = {rank - 1}; axis >= 0; --axis) {{")
        ctx.lines.append(f"      size_t coord = tmp % (size_t){in_dims_name}[axis];")
        ctx.lines.append(f"      tmp /= (size_t){in_dims_name}[axis];")
        ctx.lines.append(f"      if ({reduce_mask_name}[axis] == 0) {{")
        if keepdims == 1:
            ctx.lines.append(f"        out_idx += coord * (size_t){out_strides_name}[axis];")
        elif out_rank > 0:
            ctx.lines.append(f"        out_idx += coord * (size_t){out_strides_name}[out_axis];")
            ctx.lines.append("        out_axis -= 1;")
        ctx.lines.append("      }")
        ctx.lines.append("    }")
        ctx.lines.append(f"    float xv = ((float){inp}[i] - {zi}) * {si:.8f}f;")
        if mode in ("sum", "mean", "log_sum"):
            ctx.lines.append(f"    {acc_name}[out_idx] += xv;")
        elif mode == "log_sum_exp":
            ctx.lines.append(f"    {acc_name}[out_idx] += expf(xv);")
        elif mode == "max":
            ctx.lines.append(f"    if (xv > {acc_name}[out_idx]) {acc_name}[out_idx] = xv;")
        elif mode == "min":
            ctx.lines.append(f"    if (xv < {acc_name}[out_idx]) {acc_name}[out_idx] = xv;")
        elif mode == "prod":
            ctx.lines.append(f"    {acc_name}[out_idx] *= xv;")
        elif mode == "l1":
            ctx.lines.append(f"    {acc_name}[out_idx] += fabsf(xv);")
        elif mode in ("l2", "sum_square"):
            ctx.lines.append(f"    {acc_name}[out_idx] += xv * xv;")
        ctx.lines.append("  }")

        if mode == "mean" and reduce_count > 1:
            ctx.lines.append(
                f"  for (size_t i = 0; i < {out_size}; ++i) {acc_name}[i] = {acc_name}[i] / (float){reduce_count};"
            )
        if mode == "l2":
            ctx.lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {acc_name}[i] = sqrtf({acc_name}[i]);")
        if mode in ("log_sum", "log_sum_exp"):
            ctx.lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {acc_name}[i] = logf({acc_name}[i]);")

        ctx.lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {{")
        ctx.lines.append(f"    int q = (int)roundf({acc_name}[i] / {so:.8f}f) + {zo};")
        ctx.lines.append(f"    if (q < {qmin}) q = {qmin};")
        ctx.lines.append(f"    if (q > {qmax}) q = {qmax};")
        ctx.lines.append(f"    {out}[i] = ({out_ctype})q;")
        ctx.lines.append("  }")
        return

    if in_dtype != "float32" or out_dtype != "float32":
        raise ValueError(f"{op_name} currently supports float32 or quantized int8/int16.")

    if mode in ("sum", "mean", "l1", "l2", "sum_square", "log_sum", "log_sum_exp"):
        ctx.lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {out}[i] = 0.0f;")
    elif mode == "max":
        ctx.lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {out}[i] = -3.402823466e+38F;")
    elif mode == "min":
        ctx.lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {out}[i] = 3.402823466e+38F;")
    elif mode == "prod":
        ctx.lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {out}[i] = 1.0f;")
    else:
        raise ValueError(f"{op_name} mode is invalid.")

    ctx.lines.append(f"  for (size_t i = 0; i < {in_size}; ++i) {{")
    ctx.lines.append("    size_t tmp = i;")
    ctx.lines.append("    size_t out_idx = 0;")
    if keepdims == 0 and out_rank > 0:
        ctx.lines.append(f"    int out_axis = {out_rank - 1};")
    ctx.lines.append(f"    for (int axis = {rank - 1}; axis >= 0; --axis) {{")
    ctx.lines.append(f"      size_t coord = tmp % (size_t){in_dims_name}[axis];")
    ctx.lines.append(f"      tmp /= (size_t){in_dims_name}[axis];")
    ctx.lines.append(f"      if ({reduce_mask_name}[axis] == 0) {{")
    if keepdims == 1:
        ctx.lines.append(f"        out_idx += coord * (size_t){out_strides_name}[axis];")
    elif out_rank > 0:
        ctx.lines.append(f"        out_idx += coord * (size_t){out_strides_name}[out_axis];")
        ctx.lines.append("        out_axis -= 1;")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    if mode in ("sum", "mean", "log_sum"):
        ctx.lines.append(f"    {out}[out_idx] += {inp}[i];")
    elif mode == "log_sum_exp":
        ctx.lines.append(f"    {out}[out_idx] += expf({inp}[i]);")
    elif mode == "max":
        ctx.lines.append(f"    if ({inp}[i] > {out}[out_idx]) {out}[out_idx] = {inp}[i];")
    elif mode == "min":
        ctx.lines.append(f"    if ({inp}[i] < {out}[out_idx]) {out}[out_idx] = {inp}[i];")
    elif mode == "prod":
        ctx.lines.append(f"    {out}[out_idx] *= {inp}[i];")
    elif mode == "l1":
        ctx.lines.append(f"    {out}[out_idx] += fabsf({inp}[i]);")
    elif mode in ("l2", "sum_square"):
        ctx.lines.append(f"    {out}[out_idx] += {inp}[i] * {inp}[i];")
    else:
        raise ValueError(f"{op_name} mode is invalid.")
    ctx.lines.append("  }")

    if mode == "mean" and reduce_count > 1:
        ctx.lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {out}[i] = {out}[i] / (float){reduce_count};")
    if mode == "l2":
        ctx.lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {out}[i] = sqrtf({out}[i]);")
    if mode in ("log_sum", "log_sum_exp"):
        ctx.lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {out}[i] = logf({out}[i]);")
