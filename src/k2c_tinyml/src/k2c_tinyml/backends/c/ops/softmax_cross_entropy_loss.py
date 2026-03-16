# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import product
from .registry import register_op


def _decode_attr_str(value: object, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").lower()
    return str(value).lower()


@register_op("SoftmaxCrossEntropyLoss")
def emit_softmax_cross_entropy_loss(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 2 or len(node.inputs) > 3:
        raise ValueError("SoftmaxCrossEntropyLoss expects 2 or 3 inputs.")
    if len(node.outputs) < 1 or len(node.outputs) > 2:
        raise ValueError("SoftmaxCrossEntropyLoss expects 1 or 2 outputs.")

    x_name = node.inputs[0]
    t_name = node.inputs[1]
    w_name = node.inputs[2] if len(node.inputs) == 3 and node.inputs[2] else None
    loss_name = node.outputs[0]
    logp_name = node.outputs[1] if len(node.outputs) == 2 and node.outputs[1] else None

    x_dtype = ctx.dtype(x_name)
    if x_dtype not in ("float32", "int8", "int16"):
        raise ValueError("SoftmaxCrossEntropyLoss supports float32/int8/int16 logits only.")
    if ctx.dtype(loss_name) != "float32":
        raise ValueError("SoftmaxCrossEntropyLoss loss output dtype must be float32.")
    t_dtype = ctx.dtype(t_name)
    if t_dtype not in ("int64", "int32"):
        raise ValueError("SoftmaxCrossEntropyLoss target dtype must be int64/int32.")
    if w_name is not None and ctx.dtype(w_name) not in ("float32", "int8", "int16", "int32", "int64"):
        raise ValueError("SoftmaxCrossEntropyLoss weight dtype is unsupported.")
    if logp_name is not None and ctx.dtype(logp_name) != "float32":
        raise ValueError("SoftmaxCrossEntropyLoss log_prob dtype must be float32.")

    x_shape = [int(v) for v in ctx.shape(x_name)]
    t_shape = [int(v) for v in ctx.shape(t_name)]
    loss_shape = [int(v) for v in ctx.shape(loss_name)]
    if len(x_shape) < 2:
        raise ValueError("SoftmaxCrossEntropyLoss expects logits rank >= 2 [N,C,...].")
    n_size = int(x_shape[0])
    c_size = int(x_shape[1])
    spatial_shape = x_shape[2:]
    inner = int(product(spatial_shape)) if spatial_shape else 1
    sample_count = n_size * inner
    expected_target = [n_size] + spatial_shape
    if t_shape != expected_target:
        raise ValueError("SoftmaxCrossEntropyLoss target shape mismatch.")
    if w_name is not None:
        w_shape = [int(v) for v in ctx.shape(w_name)]
        if w_shape != [c_size]:
            raise ValueError("SoftmaxCrossEntropyLoss weight shape mismatch.")
    if logp_name is not None:
        logp_shape = [int(v) for v in ctx.shape(logp_name)]
        if logp_shape != x_shape:
            raise ValueError("SoftmaxCrossEntropyLoss log_prob shape mismatch.")

    reduction = _decode_attr_str(node.attrs.get("reduction", "mean"), "mean")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("SoftmaxCrossEntropyLoss reduction must be none/mean/sum.")
    if reduction == "none":
        if loss_shape != t_shape:
            raise ValueError("SoftmaxCrossEntropyLoss loss shape must match target for reduction=none.")
    else:
        if loss_shape != []:
            raise ValueError("SoftmaxCrossEntropyLoss loss shape must be scalar for reduction!=none.")

    ignore_index = int(node.attrs.get("ignore_index", -100))
    x = ctx.map_ptr(x_name)
    t = ctx.map_ptr(t_name)
    loss_out = ctx.map_ptr(loss_name)
    logp_out = ctx.map_ptr(logp_name) if logp_name is not None else None
    w = ctx.map_ptr(w_name) if w_name is not None else None

    x_q = ctx.qparams_optional(x_name)
    x_is_quant = x_dtype in ("int8", "int16") and x_q is not None
    if x_is_quant:
        sx, zx = x_q

    w_dtype = ctx.dtype(w_name) if w_name is not None else None
    w_q = ctx.qparams_optional(w_name) if w_name is not None else None

    ctx.lines.append("  float total_loss = 0.0f;")
    ctx.lines.append("  float total_weight = 0.0f;")
    ctx.lines.append(f"  for (size_t i = 0; i < {sample_count}; ++i) {{")
    ctx.lines.append(f"    size_t n_i = i / {inner};")
    ctx.lines.append(f"    size_t in_pos = i % {inner};")
    ctx.lines.append(f"    size_t base = (n_i * {c_size} * {inner}) + in_pos;")
    if x_is_quant:
        ctx.lines.append(f"    float max_v = ((float){x}[base] - {zx}) * {sx:.8f}f;")
    else:
        ctx.lines.append(f"    float max_v = (float){x}[base];")
    ctx.lines.append(f"    for (size_t c = 1; c < {c_size}; ++c) {{")
    ctx.lines.append(f"      size_t x_idx = base + c * {inner};")
    if x_is_quant:
        ctx.lines.append(f"      float v = ((float){x}[x_idx] - {zx}) * {sx:.8f}f;")
    else:
        ctx.lines.append(f"      float v = (float){x}[x_idx];")
    ctx.lines.append("      if (v > max_v) max_v = v;")
    ctx.lines.append("    }")
    ctx.lines.append("    float exp_sum = 0.0f;")
    ctx.lines.append(f"    for (size_t c = 0; c < {c_size}; ++c) {{")
    ctx.lines.append(f"      size_t x_idx = base + c * {inner};")
    if x_is_quant:
        ctx.lines.append(f"      float xv = ((float){x}[x_idx] - {zx}) * {sx:.8f}f;")
    else:
        ctx.lines.append(f"      float xv = (float){x}[x_idx];")
    ctx.lines.append("      exp_sum += expf(xv - max_v);")
    ctx.lines.append("    }")
    ctx.lines.append("    float log_sum = logf(exp_sum);")
    if logp_out is not None:
        ctx.lines.append(f"    for (size_t c = 0; c < {c_size}; ++c) {{")
        ctx.lines.append(f"      size_t x_idx = base + c * {inner};")
        if x_is_quant:
            ctx.lines.append(f"      float xv = ((float){x}[x_idx] - {zx}) * {sx:.8f}f;")
        else:
            ctx.lines.append(f"      float xv = (float){x}[x_idx];")
        ctx.lines.append(f"      {logp_out}[x_idx] = xv - max_v - log_sum;")
        ctx.lines.append("    }")

    ctx.lines.append(f"    int64_t cls = (int64_t){t}[i];")
    ctx.lines.append("    float sample_loss = 0.0f;")
    ctx.lines.append(f"    if (cls != {ignore_index} && cls >= 0 && cls < {c_size}) {{")
    ctx.lines.append(f"      size_t cls_idx = base + (size_t)cls * {inner};")
    if x_is_quant:
        ctx.lines.append(f"      float cls_v = ((float){x}[cls_idx] - {zx}) * {sx:.8f}f;")
    else:
        ctx.lines.append(f"      float cls_v = (float){x}[cls_idx];")
    ctx.lines.append("      float logp = cls_v - max_v - log_sum;")
    if w is not None:
        if w_dtype in ("int8", "int16") and w_q is not None:
            sw, zw = w_q
            ctx.lines.append(f"      float wv = ((float){w}[(size_t)cls] - {zw}) * {sw:.8f}f;")
        else:
            ctx.lines.append(f"      float wv = (float){w}[(size_t)cls];")
        ctx.lines.append("      sample_loss = -logp * wv;")
        ctx.lines.append("      total_weight += wv;")
    else:
        ctx.lines.append("      sample_loss = -logp;")
        ctx.lines.append("      total_weight += 1.0f;")
    ctx.lines.append("      total_loss += sample_loss;")
    ctx.lines.append("    }")
    if reduction == "none":
        ctx.lines.append(f"    {loss_out}[i] = sample_loss;")
    ctx.lines.append("  }")
    if reduction == "sum":
        ctx.lines.append(f"  {loss_out}[0] = total_loss;")
    elif reduction == "mean":
        ctx.lines.append(f"  {loss_out}[0] = (total_weight > 0.0f) ? (total_loss / total_weight) : 0.0f;")
