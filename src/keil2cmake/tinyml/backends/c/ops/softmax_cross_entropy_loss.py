# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
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
    if w_name is not None and ctx.dtype(w_name) not in ("float32", "int8", "int16"):
        raise ValueError("SoftmaxCrossEntropyLoss weight dtype must be float32/int8/int16.")
    if logp_name is not None and ctx.dtype(logp_name) != "float32":
        raise ValueError("SoftmaxCrossEntropyLoss log_prob dtype must be float32.")

    x_shape = [int(v) for v in ctx.shape(x_name)]
    t_shape = [int(v) for v in ctx.shape(t_name)]
    loss_shape = [int(v) for v in ctx.shape(loss_name)]
    if len(x_shape) != 2:
        raise ValueError("SoftmaxCrossEntropyLoss currently supports logits rank=2 [N,C].")
    if len(t_shape) != 1:
        raise ValueError("SoftmaxCrossEntropyLoss currently supports target rank=1 [N].")
    n_size, c_size = x_shape
    if t_shape[0] != n_size:
        raise ValueError("SoftmaxCrossEntropyLoss target shape mismatch.")
    if w_name is not None:
        w_shape = [int(v) for v in ctx.shape(w_name)]
        if w_shape != [c_size]:
            raise ValueError("SoftmaxCrossEntropyLoss weight shape mismatch.")
    if logp_name is not None:
        logp_shape = [int(v) for v in ctx.shape(logp_name)]
        if logp_shape != [n_size, c_size]:
            raise ValueError("SoftmaxCrossEntropyLoss log_prob shape mismatch.")

    reduction = _decode_attr_str(node.attrs.get("reduction", "mean"), "mean")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("SoftmaxCrossEntropyLoss reduction must be none/mean/sum.")
    if reduction == "none":
        if loss_shape != [n_size]:
            raise ValueError("SoftmaxCrossEntropyLoss loss shape must be [N] for reduction=none.")
    else:
        if loss_shape != []:
            raise ValueError("SoftmaxCrossEntropyLoss loss shape must be scalar for reduction!=none.")

    ignore_index = int(node.attrs.get("ignore_index", -100))
    x = ctx.map_ptr(x_name)
    t = ctx.map_ptr(t_name)
    loss_out = ctx.map_ptr(loss_name)
    logp_out = ctx.map_ptr(logp_name) if logp_name is not None else None
    w = ctx.map_ptr(w_name) if w_name is not None else None

    ctx.lines.append("  float total_loss = 0.0f;")
    ctx.lines.append("  float total_weight = 0.0f;")
    ctx.lines.append(f"  for (size_t n = 0; n < {n_size}; ++n) {{")
    ctx.lines.append(f"    float max_v = (float){x}[n * {c_size} + 0];")
    ctx.lines.append(f"    for (size_t c = 1; c < {c_size}; ++c) {{")
    ctx.lines.append(f"      float v = (float){x}[n * {c_size} + c];")
    ctx.lines.append("      if (v > max_v) max_v = v;")
    ctx.lines.append("    }")
    ctx.lines.append("    float exp_sum = 0.0f;")
    ctx.lines.append(f"    for (size_t c = 0; c < {c_size}; ++c) {{")
    ctx.lines.append(f"      exp_sum += expf((float){x}[n * {c_size} + c] - max_v);")
    ctx.lines.append("    }")
    ctx.lines.append("    float log_sum = logf(exp_sum);")
    if logp_out is not None:
        ctx.lines.append(f"    for (size_t c = 0; c < {c_size}; ++c) {{")
        ctx.lines.append(f"      {logp_out}[n * {c_size} + c] = (float){x}[n * {c_size} + c] - max_v - log_sum;")
        ctx.lines.append("    }")
    ctx.lines.append(f"    int64_t cls = (int64_t){t}[n];")
    ctx.lines.append("    float sample_loss = 0.0f;")
    ctx.lines.append(f"    if (cls != {ignore_index} && cls >= 0 && cls < {c_size}) {{")
    ctx.lines.append(f"      float logp = (float){x}[n * {c_size} + (size_t)cls] - max_v - log_sum;")
    if w is not None:
        ctx.lines.append(f"      float wv = (float){w}[(size_t)cls];")
        ctx.lines.append("      sample_loss = -logp * wv;")
        ctx.lines.append("      total_weight += wv;")
    else:
        ctx.lines.append("      sample_loss = -logp;")
        ctx.lines.append("      total_weight += 1.0f;")
    ctx.lines.append("      total_loss += sample_loss;")
    ctx.lines.append("    }")
    if reduction == "none":
        ctx.lines.append(f"    {loss_out}[n] = sample_loss;")
    ctx.lines.append("  }")
    if reduction == "sum":
        ctx.lines.append(f"  {loss_out}[0] = total_loss;")
    elif reduction == "mean":
        ctx.lines.append(f"  {loss_out}[0] = (total_weight > 0.0f) ? (total_loss / total_weight) : 0.0f;")
