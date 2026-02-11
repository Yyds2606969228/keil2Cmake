# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Iterable

from ..ir import ModelIR


def tensor_size(shape: Iterable[int]) -> int:
    size = 1
    for dim in shape:
        if dim <= 0:
            raise ValueError("Tensor shape contains unknown or invalid dimension.")
        size *= int(dim)
    return int(size)


def product(shape: Iterable[int]) -> int:
    size = 1
    for dim in shape:
        size *= int(dim)
    return int(size)


def quantize_multiplier(real_multiplier: float) -> tuple[int, int]:
    if real_multiplier == 0.0:
        return 0, 0
    if real_multiplier < 0.0:
        raise ValueError("Multiplier must be non-negative.")
    significand, exponent = math.frexp(real_multiplier)
    q = int(round(significand * (1 << 31)))
    if q == (1 << 31):
        q //= 2
        exponent += 1
    shift = 31 - exponent
    return int(q), int(shift)


def normalize_axis(axis: int, rank: int) -> int:
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError("Axis out of range.")
    return axis


def get_shape(model: ModelIR, name: str) -> list[int]:
    if name not in model.tensors:
        raise ValueError(f"Missing tensor shape for '{name}'.")
    return model.tensors[name].shape


def get_const_ints(model: ModelIR, name: str) -> list[int]:
    if name not in model.tensors:
        raise ValueError(f"Missing const tensor '{name}'.")
    tensor = model.tensors[name]
    if tensor.data is None:
        raise ValueError(f"Const tensor '{name}' has no data.")
    if tensor.dtype == "float32":
        return [int(v) for v in tensor.data]
    if tensor.dtype in ("int64", "int32", "int16", "int8"):
        return [int(v) for v in tensor.data]
    raise ValueError("Only float/int const tensors are supported.")


def get_const_scalar(model: ModelIR, name: str) -> float:
    if name not in model.tensors:
        raise ValueError(f"Missing const tensor '{name}'.")
    tensor = model.tensors[name]
    if tensor.data is None or len(tensor.data) == 0:
        raise ValueError(f"Const tensor '{name}' has no data.")
    return float(tensor.data[0])


def emit_op_relu(lines: list[str], out: str, a: str, size: int) -> None:
    lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
    lines.append(f"    float v = {a}[i];")
    lines.append(f"    {out}[i] = v > 0.0f ? v : 0.0f;")
    lines.append("  }")


def emit_op_leaky_relu(lines: list[str], out: str, a: str, size: int, alpha: float) -> None:
    lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
    lines.append(f"    float v = {a}[i];")
    lines.append(f"    {out}[i] = v >= 0.0f ? v : ({alpha:.8f}f * v);")
    lines.append("  }")


def emit_op_sigmoid(lines: list[str], out: str, a: str, size: int) -> None:
    lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
    lines.append(f"    {out}[i] = 1.0f / (1.0f + expf(-{a}[i]));")
    lines.append("  }")


def emit_op_tanh(lines: list[str], out: str, a: str, size: int) -> None:
    lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
    lines.append(f"    {out}[i] = tanhf({a}[i]);")
    lines.append("  }")


def emit_op_clip(lines: list[str], out: str, a: str, size: int, min_v: float, max_v: float) -> None:
    lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
    lines.append(f"    float v = {a}[i];")
    lines.append(f"    if (v < {min_v:.8f}f) v = {min_v:.8f}f;")
    lines.append(f"    if (v > {max_v:.8f}f) v = {max_v:.8f}f;")
    lines.append(f"    {out}[i] = v;")
    lines.append("  }")


def emit_op_unary_func(lines: list[str], out: str, a: str, size: int, func: str) -> None:
    lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
    lines.append(f"    {out}[i] = {func}({a}[i]);")
    lines.append("  }")


def emit_op_unary_quant(
    lines: list[str],
    out: str,
    inp: str,
    size: int,
    expr: str,
    dtype: str,
    sa: float,
    za: int,
    so: float,
    zo: int,
) -> None:
    if dtype == "int8":
        qmin, qmax = -128, 127
        ctype = "int8_t"
    elif dtype == "int16":
        qmin, qmax = -32768, 32767
        ctype = "int16_t"
    else:
        raise ValueError("Unsupported quantized dtype.")
    lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
    lines.append(f"    float r = ((float){inp}[i] - {za}) * {sa:.8f}f;")
    lines.append(f"    float v = {expr};")
    lines.append(f"    int q = (int)roundf(v / {so:.8f}f) + {zo};")
    lines.append(f"    if (q < {qmin}) q = {qmin};")
    lines.append(f"    if (q > {qmax}) q = {qmax};")
    lines.append(f"    {out}[i] = ({ctype})q;")
    lines.append("  }")


def emit_op_matmul(lines: list[str], out: str, a: str, b: str, m: int, k: int, n: int) -> None:
    lines.append(f"  for (size_t i = 0; i < {m}; ++i) {{")
    lines.append(f"    for (size_t j = 0; j < {n}; ++j) {{")
    lines.append("      float sum = 0.0f;")
    lines.append(f"      for (size_t t = 0; t < {k}; ++t) {{")
    lines.append(f"        sum += {a}[i * {k} + t] * {b}[t * {n} + j];")
    lines.append("      }")
    lines.append(f"      {out}[i * {n} + j] = sum;")
    lines.append("    }")
    lines.append("  }")


def emit_op_gemm(
    lines: list[str],
    out: str,
    a: str,
    b: str,
    c: str | None,
    m: int,
    k: int,
    n: int,
    c_is_matrix: bool,
) -> None:
    emit_op_matmul(lines, out, a, b, m, k, n)
    if c:
        lines.append(f"  for (size_t i = 0; i < {m}; ++i) {{")
        lines.append(f"    for (size_t j = 0; j < {n}; ++j) {{")
        if c_is_matrix:
            lines.append(f"      {out}[i * {n} + j] += {c}[i * {n} + j];")
        else:
            lines.append(f"      {out}[i * {n} + j] += {c}[j];")
        lines.append("    }")
        lines.append("  }")


def emit_op_softmax(lines: list[str], out: str, a: str, size: int) -> None:
    lines.append("  float max_v = " + f"{a}[0];")
    lines.append(f"  for (size_t i = 1; i < {size}; ++i) {{")
    lines.append(f"    if ({a}[i] > max_v) max_v = {a}[i];")
    lines.append("  }")
    lines.append("  float sum = 0.0f;")
    lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
    lines.append(f"    float e = expf({a}[i] - max_v);")
    lines.append(f"    {out}[i] = e;")
    lines.append("    sum += e;")
    lines.append("  }")
    lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
    lines.append(f"    {out}[i] = {out}[i] / sum;")
    lines.append("  }")


def emit_op_copy(lines: list[str], out: str, a: str, size: int) -> None:
    lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
    lines.append(f"    {out}[i] = {a}[i];")
    lines.append("  }")


def emit_op_binary_simple(
    lines: list[str],
    out: str,
    a: str,
    b: str,
    size: int,
    op: str,
    scalar_left: bool = False,
    scalar_right: bool = False,
) -> None:
    lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
    if scalar_left:
        lines.append(f"    {out}[i] = {a} {op} {b}[i];")
    elif scalar_right:
        lines.append(f"    {out}[i] = {a}[i] {op} {b};")
    else:
        lines.append(f"    {out}[i] = {a}[i] {op} {b}[i];")
    lines.append("  }")


def _broadcast_strides(
    in_shape: list[int],
    out_shape: list[int],
) -> list[int]:
    out_rank = len(out_shape)
    in_rank = len(in_shape)
    if in_rank > out_rank:
        raise ValueError("Broadcast rank mismatch.")
    aligned = [1] * (out_rank - in_rank) + list(in_shape)
    raw_strides = [0] * out_rank
    stride = 1
    for axis in range(out_rank - 1, -1, -1):
        dim = int(aligned[axis])
        if dim <= 0:
            raise ValueError("Broadcast requires known positive dimensions.")
        raw_strides[axis] = stride
        stride *= dim
    out: list[int] = []
    for axis, in_dim in enumerate(aligned):
        out_dim = int(out_shape[axis])
        if out_dim <= 0:
            raise ValueError("Broadcast requires known positive dimensions.")
        if in_dim == out_dim:
            out.append(raw_strides[axis])
        elif in_dim == 1:
            out.append(0)
        else:
            raise ValueError("Unsupported broadcast pattern.")
    return out


def emit_op_binary_broadcast(
    lines: list[str],
    out: str,
    a: str,
    b: str,
    out_shape: list[int],
    a_shape: list[int],
    b_shape: list[int],
    op: str,
) -> None:
    out_size = tensor_size(out_shape)
    a_size = tensor_size(a_shape)
    b_size = tensor_size(b_shape)

    if a_size == b_size == out_size:
        emit_op_binary_simple(lines, out, a, b, out_size, op)
        return
    if a_size == 1 and b_size == out_size:
        emit_op_binary_simple(lines, out, f"{a}[0]", b, out_size, op, scalar_left=True)
        return
    if b_size == 1 and a_size == out_size:
        emit_op_binary_simple(lines, out, a, f"{b}[0]", out_size, op, scalar_right=True)
        return

    if len(out_shape) == 0:
        lines.append(f"  {out}[0] = {a}[0] {op} {b}[0];")
        return

    rank = len(out_shape)
    a_strides = _broadcast_strides(a_shape, out_shape)
    b_strides = _broadcast_strides(b_shape, out_shape)
    lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {{")
    lines.append("    size_t tmp = i;")
    lines.append("    size_t ai = 0;")
    lines.append("    size_t bi = 0;")
    for axis in range(rank - 1, -1, -1):
        dim = int(out_shape[axis])
        a_stride = int(a_strides[axis])
        b_stride = int(b_strides[axis])
        lines.append(f"    size_t coord_{axis} = tmp % (size_t){dim};")
        lines.append(f"    tmp /= (size_t){dim};")
        if a_stride != 0:
            lines.append(f"    ai += coord_{axis} * (size_t){a_stride};")
        if b_stride != 0:
            lines.append(f"    bi += coord_{axis} * (size_t){b_stride};")
    lines.append(f"    {out}[i] = {a}[ai] {op} {b}[bi];")
    lines.append("  }")


def emit_op_binary_broadcast_func(
    lines: list[str],
    out: str,
    a: str,
    b: str,
    out_shape: list[int],
    a_shape: list[int],
    b_shape: list[int],
    func: str,
) -> None:
    out_size = tensor_size(out_shape)
    a_size = tensor_size(a_shape)
    b_size = tensor_size(b_shape)

    if a_size == b_size == out_size:
        lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {{")
        lines.append(f"    {out}[i] = {func}({a}[i], {b}[i]);")
        lines.append("  }")
        return
    if a_size == 1 and b_size == out_size:
        lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {{")
        lines.append(f"    {out}[i] = {func}({a}[0], {b}[i]);")
        lines.append("  }")
        return
    if b_size == 1 and a_size == out_size:
        lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {{")
        lines.append(f"    {out}[i] = {func}({a}[i], {b}[0]);")
        lines.append("  }")
        return

    if len(out_shape) == 0:
        lines.append(f"  {out}[0] = {func}({a}[0], {b}[0]);")
        return

    rank = len(out_shape)
    a_strides = _broadcast_strides(a_shape, out_shape)
    b_strides = _broadcast_strides(b_shape, out_shape)
    lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {{")
    lines.append("    size_t tmp = i;")
    lines.append("    size_t ai = 0;")
    lines.append("    size_t bi = 0;")
    for axis in range(rank - 1, -1, -1):
        dim = int(out_shape[axis])
        a_stride = int(a_strides[axis])
        b_stride = int(b_strides[axis])
        lines.append(f"    size_t coord_{axis} = tmp % (size_t){dim};")
        lines.append(f"    tmp /= (size_t){dim};")
        if a_stride != 0:
            lines.append(f"    ai += coord_{axis} * (size_t){a_stride};")
        if b_stride != 0:
            lines.append(f"    bi += coord_{axis} * (size_t){b_stride};")
    lines.append(f"    {out}[i] = {func}({a}[ai], {b}[bi]);")
    lines.append("  }")


def emit_op_conv2d(
    lines: list[str],
    out: str,
    x: str,
    w: str,
    b: str | None,
    x_shape: list[int],
    w_shape: list[int],
    out_shape: list[int],
    strides: list[int],
    pads: list[int],
    dilations: list[int],
    groups: int,
) -> None:
    if len(x_shape) != 4 or len(w_shape) != 4 or len(out_shape) != 4:
        raise ValueError("Conv expects 4D tensors (NCHW).")
    n, c_in, h, w_in = x_shape
    m, c_per_g, k_h, k_w = w_shape
    n_out, c_out, out_h, out_w = out_shape
    if n != n_out:
        raise ValueError("Conv batch dimension mismatch.")
    if groups <= 0:
        raise ValueError("Conv group must be positive.")
    if c_out != m or c_in != c_per_g * groups:
        raise ValueError("Conv channel mismatch.")
    if m % groups != 0:
        raise ValueError("Conv output channels must be divisible by groups.")
    oc_per_group = m // groups
    stride_h, stride_w = strides
    pad_h0, pad_w0, pad_h1, pad_w1 = pads
    _ = pad_h1, pad_w1
    dil_h, dil_w = dilations

    lines.append(f"  for (size_t ni = 0; ni < {n}; ++ni) {{")
    lines.append(f"    for (size_t oc = 0; oc < {m}; ++oc) {{")
    lines.append(f"      for (size_t oh = 0; oh < {out_h}; ++oh) {{")
    lines.append(f"        for (size_t ow = 0; ow < {out_w}; ++ow) {{")
    if b:
        lines.append(f"          float sum = {b}[oc];")
    else:
        lines.append("          float sum = 0.0f;")
    lines.append(f"          size_t g = oc / {oc_per_group};")
    lines.append(f"          size_t ic_begin = g * {c_per_g};")
    lines.append(f"          for (size_t ic_local = 0; ic_local < {c_per_g}; ++ic_local) {{")
    lines.append("            size_t ic = ic_begin + ic_local;")
    lines.append(f"            for (size_t kh = 0; kh < {k_h}; ++kh) {{")
    lines.append(f"              for (size_t kw = 0; kw < {k_w}; ++kw) {{")
    lines.append(
        f"                int in_h = (int)(oh * {stride_h} + kh * {dil_h}) - {pad_h0};"
    )
    lines.append(
        f"                int in_w = (int)(ow * {stride_w} + kw * {dil_w}) - {pad_w0};"
    )
    lines.append(
        "                if (in_h >= 0 && in_h < (int)"
        + f"{h}"
        + " && in_w >= 0 && in_w < (int)"
        + f"{w_in}"
        + ") {"
    )
    lines.append(
        f"                  size_t in_idx = ((ni * {c_in} + ic) * {h} + (size_t)in_h) * {w_in} + (size_t)in_w;"
    )
    lines.append(
        f"                  size_t w_idx = ((oc * {c_per_g} + ic_local) * {k_h} + kh) * {k_w} + kw;"
    )
    lines.append(f"                  sum += {x}[in_idx] * {w}[w_idx];")
    lines.append("                }")
    lines.append("              }")
    lines.append("            }")
    lines.append("          }")
    lines.append(
        f"          {out}[((ni * {c_out} + oc) * {out_h} + oh) * {out_w} + ow] = sum;"
    )
    lines.append("        }")
    lines.append("      }")
    lines.append("    }")
    lines.append("  }")


def emit_op_pool2d(
    lines: list[str],
    out: str,
    x: str,
    x_shape: list[int],
    out_shape: list[int],
    kernel: list[int],
    strides: list[int],
    pads: list[int],
    mode: str,
) -> None:
    if len(x_shape) != 4 or len(out_shape) != 4:
        raise ValueError("Pool expects 4D tensors (NCHW).")
    n, c, h, w_in = x_shape
    n_out, c_out, out_h, out_w = out_shape
    if n != n_out:
        raise ValueError("Pool batch dimension mismatch.")
    if c != c_out:
        raise ValueError("Pool channel mismatch.")
    k_h, k_w = kernel
    stride_h, stride_w = strides
    pad_h0, pad_w0, _, _ = pads

    lines.append(f"  for (size_t ni = 0; ni < {n}; ++ni) {{")
    lines.append(f"    for (size_t ch = 0; ch < {c}; ++ch) {{")
    lines.append(f"      for (size_t oh = 0; oh < {out_h}; ++oh) {{")
    lines.append(f"        for (size_t ow = 0; ow < {out_w}; ++ow) {{")
    if mode == "max":
        lines.append("          float acc = -3.402823466e+38F;")
    else:
        lines.append("          float acc = 0.0f;")
        lines.append("          size_t count = 0;")
    lines.append(f"          for (size_t kh = 0; kh < {k_h}; ++kh) {{")
    lines.append(f"            for (size_t kw = 0; kw < {k_w}; ++kw) {{")
    lines.append(
        f"              int in_h = (int)(oh * {stride_h} + kh) - {pad_h0};"
    )
    lines.append(
        f"              int in_w = (int)(ow * {stride_w} + kw) - {pad_w0};"
    )
    lines.append(
        "              if (in_h >= 0 && in_h < (int)"
        + f"{h}"
        + " && in_w >= 0 && in_w < (int)"
        + f"{w_in}"
        + ") {"
    )
    lines.append(
        f"                size_t in_idx = ((ni * {c} + ch) * {h} + (size_t)in_h) * {w_in} + (size_t)in_w;"
    )
    if mode == "max":
        lines.append(f"                float v = {x}[in_idx];")
        lines.append("                if (v > acc) acc = v;")
    else:
        lines.append(f"                acc += {x}[in_idx];")
        lines.append("                count += 1;")
    lines.append("              }")
    lines.append("            }")
    lines.append("          }")
    if mode == "avg":
        lines.append("          if (count > 0) acc = acc / (float)count;")
    lines.append(
        f"          {out}[((ni * {c_out} + ch) * {out_h} + oh) * {out_w} + ow] = acc;"
    )
    lines.append("        }")
    lines.append("      }")
    lines.append("    }")
    lines.append("  }")


def emit_op_global_avg_pool(
    lines: list[str],
    out: str,
    x: str,
    x_shape: list[int],
    out_shape: list[int],
) -> None:
    if len(x_shape) < 3 or len(out_shape) != len(x_shape):
        raise ValueError("GlobalAveragePool expects rank >= 3 and matching ranks.")
    n = int(x_shape[0])
    c = int(x_shape[1])
    n_out = int(out_shape[0])
    c_out = int(out_shape[1])
    if n != n_out:
        raise ValueError("GlobalAveragePool batch dimension mismatch.")
    expected_out = [n, c] + [1] * (len(x_shape) - 2)
    if c != c_out or [int(v) for v in out_shape] != expected_out:
        raise ValueError("GlobalAveragePool output shape mismatch.")
    spatial = product(x_shape[2:])
    out_inner = product(out_shape[2:])
    lines.append(f"  for (size_t ni = 0; ni < {n}; ++ni) {{")
    lines.append(f"    for (size_t ch = 0; ch < {c}; ++ch) {{")
    lines.append("      float acc = 0.0f;")
    lines.append(f"      for (size_t i = 0; i < {spatial}; ++i) {{")
    lines.append(f"        size_t in_idx = ((ni * {c} + ch) * {spatial}) + i;")
    lines.append(f"        acc += {x}[in_idx];")
    lines.append("      }")
    lines.append(f"      size_t out_idx = ((ni * {c_out} + ch) * {out_inner});")
    lines.append(f"      {out}[out_idx] = acc / (float)({spatial});")
    lines.append("    }")
    lines.append("  }")


def emit_op_batch_norm(
    lines: list[str],
    out: str,
    x: str,
    scale: str,
    bias: str,
    mean: str,
    var: str,
    x_shape: list[int],
    epsilon: float,
    mode: str,
    momentum: float | None = None,
) -> None:
    if len(x_shape) < 2:
        raise ValueError("BatchNormalization expects rank >= 2 input.")
    n = int(x_shape[0])
    c = int(x_shape[1])
    inner = product(x_shape[2:]) if len(x_shape) > 2 else 1
    if mode not in ("test", "training", "momentum"):
        raise ValueError("BatchNormalization mode is invalid.")
    if mode == "momentum" and momentum is None:
        raise ValueError("BatchNormalization momentum mode requires momentum.")
    nhw = n * inner
    lines.append(f"  for (size_t ch = 0; ch < {c}; ++ch) {{")
    lines.append(f"    float scale_v = {scale}[ch];")
    lines.append(f"    float bias_v = {bias}[ch];")
    if mode in ("training", "momentum"):
        lines.append("    float saved_mean = 0.0f;")
        lines.append(f"    for (size_t ni = 0; ni < {n}; ++ni) {{")
        lines.append(f"      for (size_t i = 0; i < {inner}; ++i) {{")
        lines.append(f"        size_t idx = ((ni * {c} + ch) * {inner}) + i;")
        lines.append(f"        saved_mean += {x}[idx];")
        lines.append("      }")
        lines.append("    }")
        lines.append(f"    saved_mean /= (float)({nhw});")
        lines.append("    float saved_var = 0.0f;")
        lines.append(f"    for (size_t ni = 0; ni < {n}; ++ni) {{")
        lines.append(f"      for (size_t i = 0; i < {inner}; ++i) {{")
        lines.append(f"        size_t idx = ((ni * {c} + ch) * {inner}) + i;")
        lines.append(f"        float dv = {x}[idx] - saved_mean;")
        lines.append("        saved_var += dv * dv;")
        lines.append("      }")
        lines.append("    }")
        lines.append(f"    saved_var /= (float)({nhw});")
    if mode == "test":
        lines.append(f"    float mean_v = {mean}[ch];")
        lines.append(f"    float var_v = {var}[ch];")
    elif mode == "training":
        lines.append("    float mean_v = saved_mean;")
        lines.append("    float var_v = saved_var;")
    else:
        lines.append(
            f"    float mean_v = {mean}[ch] * {momentum:.8f}f + saved_mean * {1.0 - float(momentum):.8f}f;"
        )
        lines.append(
            f"    float var_v = {var}[ch] * {momentum:.8f}f + saved_var * {1.0 - float(momentum):.8f}f;"
        )
    lines.append(f"    float inv_std = 1.0f / sqrtf(var_v + {epsilon:.8f}f);")
    lines.append(f"    for (size_t ni = 0; ni < {n}; ++ni) {{")
    lines.append(f"      for (size_t i = 0; i < {inner}; ++i) {{")
    lines.append(f"        size_t idx = ((ni * {c} + ch) * {inner}) + i;")
    lines.append(f"        float v = ({x}[idx] - mean_v) * inv_std;")
    lines.append(f"        {out}[idx] = v * scale_v + bias_v;")
    lines.append("      }")
    lines.append("    }")
    lines.append("  }")


def emit_op_concat(
    lines: list[str],
    out: str,
    inputs: list[str],
    input_shapes: list[list[int]],
    axis: int,
    out_shape: list[int],
) -> None:
    rank = len(out_shape)
    axis = normalize_axis(axis, rank)
    outer = product(out_shape[:axis]) if axis > 0 else 1
    inner = product(out_shape[axis + 1 :]) if axis + 1 < rank else 1
    axis_dim_out = out_shape[axis]

    lines.append(f"  for (size_t outer = 0; outer < {outer}; ++outer) {{")
    lines.append(f"    size_t out_base = outer * {axis_dim_out} * {inner};")
    lines.append("    size_t offset = 0;")
    for idx, inp in enumerate(inputs):
        axis_dim = input_shapes[idx][axis]
        lines.append(f"    size_t in_base_{idx} = outer * {axis_dim} * {inner};")
        lines.append(f"    for (size_t j = 0; j < {axis_dim} * {inner}; ++j) {{")
        lines.append(f"      {out}[out_base + offset + j] = {inp}[in_base_{idx} + j];")
        lines.append("    }")
        lines.append(f"    offset += {axis_dim} * {inner};")
    lines.append("  }")


def emit_op_transpose(
    lines: list[str],
    out: str,
    inp: str,
    in_shape: list[int],
    out_shape: list[int],
    perm: list[int],
) -> None:
    rank = len(in_shape)
    if rank != len(perm):
        raise ValueError("Transpose perm length mismatch.")
    if rank == 1:
        size = tensor_size(in_shape)
        emit_op_copy(lines, out, inp, size)
        return
    if rank == 2:
        if perm != [1, 0]:
            raise ValueError("Transpose supports 2D perm [1,0] only.")
        rows, cols = in_shape
        lines.append(f"  for (size_t i = 0; i < {rows}; ++i) {{")
        lines.append(f"    for (size_t j = 0; j < {cols}; ++j) {{")
        lines.append(f"      {out}[j * {rows} + i] = {inp}[i * {cols} + j];")
        lines.append("    }")
        lines.append("  }")
        return
    if rank == 3:
        o0, o1, o2 = out_shape
        in0, in1, in2 = in_shape
        lines.append(f"  for (size_t o0 = 0; o0 < {o0}; ++o0) {{")
        lines.append(f"    for (size_t o1 = 0; o1 < {o1}; ++o1) {{")
        lines.append(f"      for (size_t o2 = 0; o2 < {o2}; ++o2) {{")
        lines.append("        size_t in0_idx = 0;")
        lines.append("        size_t in1_idx = 0;")
        lines.append("        size_t in2_idx = 0;")
        lines.append(f"        in{perm[0]}_idx = o0;")
        lines.append(f"        in{perm[1]}_idx = o1;")
        lines.append(f"        in{perm[2]}_idx = o2;")
        lines.append(f"        size_t in_idx = (in0_idx * {in1} + in1_idx) * {in2} + in2_idx;")
        lines.append(f"        size_t out_idx = (o0 * {o1} + o1) * {o2} + o2;")
        lines.append(f"        {out}[out_idx] = {inp}[in_idx];")
        lines.append("      }")
        lines.append("    }")
        lines.append("  }")
        return
    if rank == 4:
        o0, o1, o2, o3 = out_shape
        in0, in1, in2, in3 = in_shape
        lines.append(f"  for (size_t o0 = 0; o0 < {o0}; ++o0) {{")
        lines.append(f"    for (size_t o1 = 0; o1 < {o1}; ++o1) {{")
        lines.append(f"      for (size_t o2 = 0; o2 < {o2}; ++o2) {{")
        lines.append(f"        for (size_t o3 = 0; o3 < {o3}; ++o3) {{")
        lines.append("          size_t in0_idx = 0;")
        lines.append("          size_t in1_idx = 0;")
        lines.append("          size_t in2_idx = 0;")
        lines.append("          size_t in3_idx = 0;")
        lines.append(f"          in{perm[0]}_idx = o0;")
        lines.append(f"          in{perm[1]}_idx = o1;")
        lines.append(f"          in{perm[2]}_idx = o2;")
        lines.append(f"          in{perm[3]}_idx = o3;")
        lines.append(
            f"          size_t in_idx = ((in0_idx * {in1} + in1_idx) * {in2} + in2_idx) * {in3} + in3_idx;"
        )
        lines.append(
            f"          size_t out_idx = ((o0 * {o1} + o1) * {o2} + o2) * {o3} + o3;"
        )
        lines.append(f"          {out}[out_idx] = {inp}[in_idx];")
        lines.append("        }")
        lines.append("      }")
        lines.append("    }")
        lines.append("  }")
        return
    raise ValueError("Transpose supports rank up to 4.")


def _row_major_strides(shape: list[int]) -> list[int]:
    rank = len(shape)
    if rank <= 0:
        return []
    strides = [1] * rank
    for axis in range(rank - 2, -1, -1):
        strides[axis] = strides[axis + 1] * int(shape[axis + 1])
    return strides


def emit_op_pad(
    lines: list[str],
    out: str,
    inp: str,
    in_shape: list[int],
    out_shape: list[int],
    pads: list[int],
    value: float | str,
) -> None:
    rank = len(in_shape)
    if rank <= 0:
        raise ValueError("Pad expects rank >= 1.")
    if len(out_shape) != rank:
        raise ValueError("Pad output rank mismatch.")
    if len(pads) != rank * 2:
        raise ValueError("Pad pads length mismatch.")
    pad_begin = [int(v) for v in pads[:rank]]
    pad_end = [int(v) for v in pads[rank:]]
    for axis in range(rank):
        if int(out_shape[axis]) != int(in_shape[axis]) + pad_begin[axis] + pad_end[axis]:
            raise ValueError("Pad output shape mismatch.")

    in_strides = _row_major_strides(in_shape)
    out_size = tensor_size(out_shape)
    fill_expr = value if isinstance(value, str) else f"{float(value):.8f}f"

    lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {{")
    lines.append("    size_t tmp = i;")
    lines.append("    size_t in_idx = 0;")
    lines.append("    int valid = 1;")
    for axis in range(rank - 1, -1, -1):
        out_dim = int(out_shape[axis])
        in_dim = int(in_shape[axis])
        in_stride = int(in_strides[axis])
        pad0 = int(pad_begin[axis])
        lines.append(f"    size_t coord_{axis} = tmp % (size_t){out_dim};")
        lines.append(f"    tmp /= (size_t){out_dim};")
        lines.append(f"    int in_coord_{axis} = (int)coord_{axis} - {pad0};")
        lines.append(
            f"    if (in_coord_{axis} < 0 || in_coord_{axis} >= {in_dim}) {{ valid = 0; }}"
        )
        lines.append(f"    else {{ in_idx += (size_t)in_coord_{axis} * (size_t){in_stride}; }}")
    lines.append("    if (valid) {")
    lines.append(f"      {out}[i] = {inp}[in_idx];")
    lines.append("    } else {")
    lines.append(f"      {out}[i] = {fill_expr};")
    lines.append("    }")
    lines.append("  }")


def emit_op_slice(
    lines: list[str],
    out: str,
    inp: str,
    in_shape: list[int],
    out_shape: list[int],
    starts: list[int],
    axes: list[int],
) -> None:
    rank = len(in_shape)
    if rank <= 0:
        raise ValueError("Slice expects rank >= 1.")
    if len(out_shape) != rank:
        raise ValueError("Slice output rank mismatch.")
    starts_full = [0] * rank
    for idx, axis in enumerate(axes):
        axis = normalize_axis(axis, rank)
        start = int(starts[idx])
        if start < 0:
            start += int(in_shape[axis])
        if start < 0 or start > int(in_shape[axis]):
            raise ValueError("Slice start out of range.")
        if start + int(out_shape[axis]) > int(in_shape[axis]):
            raise ValueError("Slice output shape mismatch.")
        starts_full[axis] = start

    in_strides = _row_major_strides(in_shape)
    out_size = tensor_size(out_shape)
    lines.append(f"  for (size_t i = 0; i < {out_size}; ++i) {{")
    lines.append("    size_t tmp = i;")
    lines.append("    size_t in_idx = 0;")
    for axis in range(rank - 1, -1, -1):
        out_dim = int(out_shape[axis])
        in_stride = int(in_strides[axis])
        start = int(starts_full[axis])
        lines.append(f"    size_t coord_{axis} = tmp % (size_t){out_dim};")
        lines.append(f"    tmp /= (size_t){out_dim};")
        lines.append(f"    in_idx += (coord_{axis} + (size_t){start}) * (size_t){in_stride};")
    lines.append(f"    {out}[i] = {inp}[in_idx];")
    lines.append("  }")


def emit_op_reduce_all(
    lines: list[str],
    out: str,
    inp: str,
    in_shape: list[int],
    mode: str,
) -> None:
    size = tensor_size(in_shape)
    if mode in ("mean", "sum"):
        lines.append("  float acc = 0.0f;")
        lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
        lines.append(f"    acc += {inp}[i];")
        lines.append("  }")
        if mode == "mean":
            lines.append(f"  {out}[0] = acc / (float){size};")
        else:
            lines.append(f"  {out}[0] = acc;")
        return
    if mode == "max":
        lines.append("  float acc = -3.402823466e+38F;")
        lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
        lines.append(f"    float v = {inp}[i];")
        lines.append("    if (v > acc) acc = v;")
        lines.append("  }")
        lines.append(f"  {out}[0] = acc;")
        return
    if mode == "min":
        lines.append("  float acc = 3.402823466e+38F;")
        lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
        lines.append(f"    float v = {inp}[i];")
        lines.append("    if (v < acc) acc = v;")
        lines.append("  }")
        lines.append(f"  {out}[0] = acc;")
        return
    raise ValueError("Unsupported reduce mode.")
