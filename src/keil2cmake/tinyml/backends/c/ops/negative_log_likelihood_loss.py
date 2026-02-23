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


@register_op("NegativeLogLikelihoodLoss")
def emit_negative_log_likelihood_loss(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 2 or len(node.inputs) > 3:
        raise ValueError("NegativeLogLikelihoodLoss expects 2 or 3 inputs.")
    if len(node.outputs) != 1:
        raise ValueError("NegativeLogLikelihoodLoss expects 1 output.")

    x_name = node.inputs[0]
    t_name = node.inputs[1]
    w_name = node.inputs[2] if len(node.inputs) == 3 and node.inputs[2] else None
    out_name = node.outputs[0]

    x_dtype = ctx.dtype(x_name)
    if x_dtype not in ("float32", "int8", "int16"):
        raise ValueError("NegativeLogLikelihoodLoss supports float32/int8/int16 input only.")
    if ctx.dtype(out_name) != "float32":
        raise ValueError("NegativeLogLikelihoodLoss output dtype must be float32.")
    t_dtype = ctx.dtype(t_name)
    if t_dtype not in ("int64", "int32"):
        raise ValueError("NegativeLogLikelihoodLoss target dtype must be int64/int32.")
    if w_name is not None and ctx.dtype(w_name) not in ("float32", "int8", "int16"):
        raise ValueError("NegativeLogLikelihoodLoss weight dtype must be float32/int8/int16.")

    x_shape = [int(v) for v in ctx.shape(x_name)]
    t_shape = [int(v) for v in ctx.shape(t_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    if len(x_shape) != 2:
        raise ValueError("NegativeLogLikelihoodLoss currently supports input rank=2 [N,C].")
    if len(t_shape) != 1:
        raise ValueError("NegativeLogLikelihoodLoss currently supports target rank=1 [N].")
    n_size, c_size = x_shape
    if t_shape[0] != n_size:
        raise ValueError("NegativeLogLikelihoodLoss target shape mismatch.")
    if w_name is not None:
        w_shape = [int(v) for v in ctx.shape(w_name)]
        if w_shape != [c_size]:
            raise ValueError("NegativeLogLikelihoodLoss weight shape mismatch.")

    reduction = _decode_attr_str(node.attrs.get("reduction", "mean"), "mean")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("NegativeLogLikelihoodLoss reduction must be none/mean/sum.")
    if reduction == "none":
        if out_shape != [n_size]:
            raise ValueError("NegativeLogLikelihoodLoss output shape must be [N] when reduction=none.")
    else:
        if out_shape != []:
            raise ValueError("NegativeLogLikelihoodLoss output shape must be scalar when reduction!=none.")

    ignore_index = int(node.attrs.get("ignore_index", -100))
    x = ctx.map_ptr(x_name)
    t = ctx.map_ptr(t_name)
    out = ctx.map_ptr(out_name)
    w = ctx.map_ptr(w_name) if w_name is not None else None

    if reduction == "none":
        ctx.lines.append(f"  for (size_t n = 0; n < {n_size}; ++n) {{")
        ctx.lines.append(f"    int64_t cls = (int64_t){t}[n];")
        ctx.lines.append("    float loss = 0.0f;")
        ctx.lines.append(f"    if (cls != {ignore_index} && cls >= 0 && cls < {c_size}) {{")
        ctx.lines.append(f"      float xv = (float){x}[n * {c_size} + (size_t)cls];")
        if w is not None:
            ctx.lines.append(f"      float wv = (float){w}[(size_t)cls];")
            ctx.lines.append("      loss = -xv * wv;")
        else:
            ctx.lines.append("      loss = -xv;")
        ctx.lines.append("    }")
        ctx.lines.append(f"    {out}[n] = loss;")
        ctx.lines.append("  }")
        return

    ctx.lines.append("  float total_loss = 0.0f;")
    ctx.lines.append("  float total_weight = 0.0f;")
    ctx.lines.append(f"  for (size_t n = 0; n < {n_size}; ++n) {{")
    ctx.lines.append(f"    int64_t cls = (int64_t){t}[n];")
    ctx.lines.append(f"    if (cls == {ignore_index} || cls < 0 || cls >= {c_size}) continue;")
    ctx.lines.append(f"    float xv = (float){x}[n * {c_size} + (size_t)cls];")
    if w is not None:
        ctx.lines.append(f"    float wv = (float){w}[(size_t)cls];")
        ctx.lines.append("    total_loss += -xv * wv;")
        ctx.lines.append("    total_weight += wv;")
    else:
        ctx.lines.append("    total_loss += -xv;")
        ctx.lines.append("    total_weight += 1.0f;")
    ctx.lines.append("  }")
    if reduction == "sum":
        ctx.lines.append(f"  {out}[0] = total_loss;")
    else:
        ctx.lines.append(f"  {out}[0] = (total_weight > 0.0f) ? (total_loss / total_weight) : 0.0f;")
