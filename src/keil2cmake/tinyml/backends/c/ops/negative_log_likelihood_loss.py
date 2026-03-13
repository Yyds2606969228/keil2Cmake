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
    if w_name is not None and ctx.dtype(w_name) not in ("float32", "int8", "int16", "int32", "int64"):
        raise ValueError("NegativeLogLikelihoodLoss weight dtype is unsupported.")

    x_shape = [int(v) for v in ctx.shape(x_name)]
    t_shape = [int(v) for v in ctx.shape(t_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    if len(x_shape) < 2:
        raise ValueError("NegativeLogLikelihoodLoss expects input rank >= 2 [N,C,...].")
    n_size = int(x_shape[0])
    c_size = int(x_shape[1])
    spatial_shape = x_shape[2:]
    inner = int(product(spatial_shape)) if spatial_shape else 1
    expected_target = [n_size] + spatial_shape
    if t_shape != expected_target:
        raise ValueError("NegativeLogLikelihoodLoss target shape mismatch.")
    if w_name is not None:
        w_shape = [int(v) for v in ctx.shape(w_name)]
        if w_shape != [c_size]:
            raise ValueError("NegativeLogLikelihoodLoss weight shape mismatch.")

    reduction = _decode_attr_str(node.attrs.get("reduction", "mean"), "mean")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("NegativeLogLikelihoodLoss reduction must be none/mean/sum.")
    if reduction == "none":
        if out_shape != t_shape:
            raise ValueError("NegativeLogLikelihoodLoss output shape must match target when reduction=none.")
    else:
        if out_shape != []:
            raise ValueError("NegativeLogLikelihoodLoss output shape must be scalar when reduction!=none.")

    x = ctx.map_ptr(x_name)
    t = ctx.map_ptr(t_name)
    out = ctx.map_ptr(out_name)
    w = ctx.map_ptr(w_name) if w_name is not None else None
    ignore_index = int(node.attrs.get("ignore_index", -100))
    sample_count = n_size * inner

    x_q = ctx.qparams_optional(x_name)
    x_is_quant = x_dtype in ("int8", "int16") and x_q is not None
    if x_is_quant:
        sx, zx = x_q

    w_dtype = ctx.dtype(w_name) if w_name is not None else None
    w_q = ctx.qparams_optional(w_name) if w_name is not None else None

    if reduction == "none":
        ctx.lines.append(f"  for (size_t i = 0; i < {sample_count}; ++i) {{")
        ctx.lines.append(f"    int64_t cls = (int64_t){t}[i];")
        ctx.lines.append("    float loss = 0.0f;")
        ctx.lines.append(f"    if (cls != {ignore_index} && cls >= 0 && cls < {c_size}) {{")
        ctx.lines.append(f"      size_t n_i = i / {inner};")
        ctx.lines.append(f"      size_t in_pos = i % {inner};")
        ctx.lines.append(f"      size_t x_idx = ((n_i * {c_size} + (size_t)cls) * {inner}) + in_pos;")
        if x_is_quant:
            ctx.lines.append(f"      float xv = ((float){x}[x_idx] - {zx}) * {sx:.8f}f;")
        else:
            ctx.lines.append(f"      float xv = (float){x}[x_idx];")
        if w is not None:
            if w_dtype in ("int8", "int16") and w_q is not None:
                sw, zw = w_q
                ctx.lines.append(f"      float wv = ((float){w}[(size_t)cls] - {zw}) * {sw:.8f}f;")
            else:
                ctx.lines.append(f"      float wv = (float){w}[(size_t)cls];")
            ctx.lines.append("      loss = -xv * wv;")
        else:
            ctx.lines.append("      loss = -xv;")
        ctx.lines.append("    }")
        ctx.lines.append(f"    {out}[i] = loss;")
        ctx.lines.append("  }")
        return

    ctx.lines.append("  float total_loss = 0.0f;")
    ctx.lines.append("  float total_weight = 0.0f;")
    ctx.lines.append(f"  for (size_t i = 0; i < {sample_count}; ++i) {{")
    ctx.lines.append(f"    int64_t cls = (int64_t){t}[i];")
    ctx.lines.append(f"    if (cls == {ignore_index} || cls < 0 || cls >= {c_size}) continue;")
    ctx.lines.append(f"    size_t n_i = i / {inner};")
    ctx.lines.append(f"    size_t in_pos = i % {inner};")
    ctx.lines.append(f"    size_t x_idx = ((n_i * {c_size} + (size_t)cls) * {inner}) + in_pos;")
    if x_is_quant:
        ctx.lines.append(f"    float xv = ((float){x}[x_idx] - {zx}) * {sx:.8f}f;")
    else:
        ctx.lines.append(f"    float xv = (float){x}[x_idx];")
    if w is not None:
        if w_dtype in ("int8", "int16") and w_q is not None:
            sw, zw = w_q
            ctx.lines.append(f"    float wv = ((float){w}[(size_t)cls] - {zw}) * {sw:.8f}f;")
        else:
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
