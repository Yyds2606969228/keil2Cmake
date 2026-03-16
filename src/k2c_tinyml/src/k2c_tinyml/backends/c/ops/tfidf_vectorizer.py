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


@register_op("TfIdfVectorizer")
def emit_tfidf_vectorizer(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise ValueError("TfIdfVectorizer expects 1 input and 1 output.")
    x_name = node.inputs[0]
    out_name = node.outputs[0]

    x_dtype = ctx.dtype(x_name)
    if x_dtype not in ("int64", "int32"):
        raise ValueError("TfIdfVectorizer currently supports int32/int64 token IDs only.")
    if ctx.dtype(out_name) != "float32":
        raise ValueError("TfIdfVectorizer output dtype must be float32.")

    x_shape = [int(v) for v in ctx.shape(x_name)]
    out_shape = [int(v) for v in ctx.shape(out_name)]
    if len(x_shape) != 2:
        raise ValueError("TfIdfVectorizer currently supports rank-2 input [N,T].")
    if len(out_shape) != 2 or out_shape[0] != x_shape[0]:
        raise ValueError("TfIdfVectorizer output shape mismatch.")

    min_gram = int(node.attrs.get("min_gram_length", 1))
    max_gram = int(node.attrs.get("max_gram_length", 1))
    if min_gram != 1 or max_gram != 1:
        raise ValueError("TfIdfVectorizer currently supports unigram only.")

    mode = _decode_attr_str(node.attrs.get("mode", "tf"), "tf")
    if mode not in ("tf", "tfidf"):
        raise ValueError("TfIdfVectorizer currently supports mode=tf/tfidf only.")

    pool = node.attrs.get("pool_int64s", None)
    if pool is None or len(pool) != out_shape[1]:
        raise ValueError("TfIdfVectorizer requires pool_int64s with length = output_dim.")
    pool_vals = [int(v) for v in pool]
    pool_sym = ctx.next_symbol("k2c_tfidf_pool")

    weights = node.attrs.get("weights", None)
    w_sym = None
    if mode == "tfidf":
        if weights is None or len(weights) != out_shape[1]:
            raise ValueError("TfIdfVectorizer mode=tfidf requires weights with output_dim length.")
        w_sym = ctx.next_symbol("k2c_tfidf_w")
        w_vals = ", ".join(f"{float(v):.9g}f" for v in weights)
        ctx.lines.append(f"  static const float {w_sym}[{out_shape[1]}] = {{ {w_vals} }};")

    pool_lit = ", ".join(str(v) for v in pool_vals)
    ctx.lines.append(f"  static const int64_t {pool_sym}[{out_shape[1]}] = {{ {pool_lit} }};")
    x = ctx.map_ptr(x_name)
    out = ctx.map_ptr(out_name)
    n_size, t_size = x_shape
    dim = out_shape[1]

    ctx.lines.append(f"  for (size_t n = 0; n < {n_size}; ++n) {{")
    ctx.lines.append(f"    for (size_t d = 0; d < {dim}; ++d) {out}[n * {dim} + d] = 0.0f;")
    ctx.lines.append(f"    for (size_t t = 0; t < {t_size}; ++t) {{")
    ctx.lines.append(f"      int64_t tok = (int64_t){x}[n * {t_size} + t];")
    ctx.lines.append(f"      for (size_t d = 0; d < {dim}; ++d) {{")
    ctx.lines.append(f"        if (tok == {pool_sym}[d]) {out}[n * {dim} + d] += 1.0f;")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    if w_sym is not None:
        ctx.lines.append(f"    for (size_t d = 0; d < {dim}; ++d) {out}[n * {dim} + d] *= {w_sym}[d];")
    ctx.lines.append("  }")
