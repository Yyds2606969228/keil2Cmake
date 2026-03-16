# -*- coding: utf-8 -*-

from __future__ import annotations

from ....ir import NodeInfo
from ....operators.context import EmitContext
from .registry import register_op
from .recurrent_common import (
    assert_rec_dtype,
    build_activation_specs,
    direction_and_count,
    emit_activation_assign,
    emit_clip,
    emit_fill_real_zero,
    emit_store_real,
    read_real_expr,
)


@register_op("RNN")
def emit_rnn(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 3:
        raise ValueError("RNN expects at least 3 inputs: X, W, R.")
    if len(node.outputs) < 1 or len(node.outputs) > 2:
        raise ValueError("RNN expects 1 or 2 outputs: Y, [Y_h].")

    x_name = node.inputs[0]
    w_name = node.inputs[1]
    r_name = node.inputs[2]
    b_name = node.inputs[3] if len(node.inputs) >= 4 and node.inputs[3] else None
    seq_lens_name = node.inputs[4] if len(node.inputs) >= 5 and node.inputs[4] else None
    h0_name = node.inputs[5] if len(node.inputs) >= 6 and node.inputs[5] else None
    y_name = node.outputs[0]
    yh_name = node.outputs[1] if len(node.outputs) >= 2 and node.outputs[1] else None

    assert_rec_dtype(ctx, "RNN", x_name, "X")
    assert_rec_dtype(ctx, "RNN", w_name, "W")
    assert_rec_dtype(ctx, "RNN", r_name, "R")
    assert_rec_dtype(ctx, "RNN", y_name, "Y")
    if b_name is not None:
        assert_rec_dtype(ctx, "RNN", b_name, "B")
    if h0_name is not None:
        assert_rec_dtype(ctx, "RNN", h0_name, "initial_h")
    if yh_name is not None:
        assert_rec_dtype(ctx, "RNN", yh_name, "Y_h")
    if seq_lens_name is not None and ctx.dtype(seq_lens_name) not in ("int64", "int32"):
        raise ValueError("RNN sequence_lens dtype must be int64/int32.")

    direction, expect_dirs = direction_and_count("RNN", node.attrs.get("direction", "forward"))

    x_shape = [int(v) for v in ctx.shape(x_name)]
    w_shape = [int(v) for v in ctx.shape(w_name)]
    r_shape = [int(v) for v in ctx.shape(r_name)]
    y_shape = [int(v) for v in ctx.shape(y_name)]
    if len(x_shape) != 3 or len(w_shape) != 3 or len(r_shape) != 3:
        raise ValueError("RNN expects X/W/R rank=3.")
    t_size, b_size, i_size = x_shape
    num_dir, h_size, i_w = w_shape
    num_dir_r, h_r0, h_r1 = r_shape
    if num_dir != expect_dirs or num_dir_r != expect_dirs:
        raise ValueError("RNN num_directions does not match direction attribute.")
    if i_w != i_size or h_r0 != h_size or h_r1 != h_size:
        raise ValueError("RNN W/R shape mismatch.")
    if y_shape != [t_size, num_dir, b_size, h_size]:
        raise ValueError("RNN Y output shape mismatch.")

    act_specs = build_activation_specs(
        "RNN",
        raw_acts=node.attrs.get("activations", None),
        raw_alpha=node.attrs.get("activation_alpha", None),
        raw_beta=node.attrs.get("activation_beta", None),
        expected_count=num_dir,
        default_cycle=["tanh"],
    )
    clip_val = node.attrs.get("clip", None)
    clip_sym = None
    if clip_val is not None:
        clip_f = float(clip_val)
        if clip_f < 0.0:
            raise ValueError("RNN clip must be non-negative.")
        clip_sym = ctx.next_symbol("k2c_rnn_clip")
        ctx.lines.append(f"  const float {clip_sym} = {clip_f:.9g}f;")

    if b_name is not None:
        b_shape = [int(v) for v in ctx.shape(b_name)]
        if b_shape != [num_dir, 2 * h_size]:
            raise ValueError("RNN bias shape must be [num_directions, 2*hidden_size].")
    if h0_name is not None:
        h0_shape = [int(v) for v in ctx.shape(h0_name)]
        if h0_shape != [num_dir, b_size, h_size]:
            raise ValueError("RNN initial_h shape mismatch.")
    if yh_name is not None:
        yh_shape = [int(v) for v in ctx.shape(yh_name)]
        if yh_shape != [num_dir, b_size, h_size]:
            raise ValueError("RNN Y_h shape mismatch.")
    if seq_lens_name is not None:
        seq_shape = [int(v) for v in ctx.shape(seq_lens_name)]
        if seq_shape != [b_size]:
            raise ValueError("RNN sequence_lens shape must be [batch_size].")

    seq = ctx.map_ptr(seq_lens_name) if seq_lens_name is not None else None

    h_state = ctx.next_symbol("k2c_rnn_h")
    h_next = ctx.next_symbol("k2c_rnn_hn")

    ctx.lines.append(f"  float {h_state}[{num_dir} * {b_size} * {h_size}];")
    ctx.lines.append(f"  float {h_next}[{h_size}];")
    if h0_name is not None:
        ctx.lines.append(f"  for (size_t d_i = 0; d_i < {num_dir}; ++d_i) {{")
        ctx.lines.append(f"    for (size_t b_i = 0; b_i < {b_size}; ++b_i) {{")
        ctx.lines.append(f"      for (size_t h_i = 0; h_i < {h_size}; ++h_i) {{")
        ctx.lines.append(
            f"        {h_state}[(d_i * {b_size} + b_i) * {h_size} + h_i] = "
            f"{read_real_expr(ctx, h0_name, f'(d_i * {b_size} + b_i) * {h_size} + h_i')};"
        )
        ctx.lines.append("      }")
        ctx.lines.append("    }")
        ctx.lines.append("  }")
    else:
        ctx.lines.append(f"  for (size_t i = 0; i < {num_dir * b_size * h_size}; ++i) {h_state}[i] = 0.0f;")

    emit_fill_real_zero(
        ctx,
        indent="  ",
        tensor_name=y_name,
        count_expr=str(t_size * num_dir * b_size * h_size),
    )
    ctx.lines.append(f"  for (size_t d_i = 0; d_i < {num_dir}; ++d_i) {{")
    if direction == "bidirectional":
        ctx.lines.append("    int dir_rev = (d_i == 1) ? 1 : 0;")
    elif direction == "reverse":
        ctx.lines.append("    int dir_rev = 1;")
    else:
        ctx.lines.append("    int dir_rev = 0;")
    ctx.lines.append(f"    for (size_t b_i = 0; b_i < {b_size}; ++b_i) {{")
    if seq is not None:
        ctx.lines.append(f"      int64_t seq_len = (int64_t){seq}[b_i];")
        ctx.lines.append("      if (seq_len < 0) seq_len = 0;")
        ctx.lines.append(f"      if (seq_len > {t_size}) seq_len = {t_size};")
    else:
        ctx.lines.append(f"      int64_t seq_len = {t_size};")
    ctx.lines.append("      for (int64_t step_i = 0; step_i < seq_len; ++step_i) {")
    ctx.lines.append("        size_t t_src = dir_rev ? (size_t)(seq_len - 1 - step_i) : (size_t)step_i;")
    ctx.lines.append(f"        for (size_t h_i = 0; h_i < {h_size}; ++h_i) {{")
    ctx.lines.append("          float sum = 0.0f;")
    ctx.lines.append(f"          for (size_t i_i = 0; i_i < {i_size}; ++i_i) {{")
    ctx.lines.append(
        f"            float xv = {read_real_expr(ctx, x_name, f'(t_src * {b_size} + b_i) * {i_size} + i_i')};"
    )
    ctx.lines.append(
        f"            float ww = {read_real_expr(ctx, w_name, f'((d_i * {h_size} + h_i) * {i_size}) + i_i')};"
    )
    ctx.lines.append("            sum += xv * ww;")
    ctx.lines.append("          }")
    ctx.lines.append(f"          for (size_t hh = 0; hh < {h_size}; ++hh) {{")
    ctx.lines.append(f"            float hv = {h_state}[(d_i * {b_size} + b_i) * {h_size} + hh];")
    ctx.lines.append(
        f"            float rr = {read_real_expr(ctx, r_name, f'((d_i * {h_size} + h_i) * {h_size}) + hh')};"
    )
    ctx.lines.append("            sum += hv * rr;")
    ctx.lines.append("          }")
    if b_name is not None:
        ctx.lines.append(
            f"          sum += {read_real_expr(ctx, b_name, f'd_i * {2 * h_size} + h_i')};"
        )
        ctx.lines.append(
            f"          sum += {read_real_expr(ctx, b_name, f'd_i * {2 * h_size} + {h_size} + h_i')};"
        )
    ctx.lines.append("          float pre = sum;")
    emit_clip(ctx, indent="          ", value_var="pre", clip_var=clip_sym)
    ctx.lines.append("          float h_val = 0.0f;")
    emit_activation_assign(
        ctx,
        indent="          ",
        out_var="h_val",
        pre_var="pre",
        specs_by_dir=act_specs,
        dir_var="d_i",
    )
    ctx.lines.append(f"          {h_next}[h_i] = h_val;")
    ctx.lines.append("        }")
    ctx.lines.append(f"        for (size_t h_i = 0; h_i < {h_size}; ++h_i) {{")
    ctx.lines.append(
        f"          {h_state}[(d_i * {b_size} + b_i) * {h_size} + h_i] = {h_next}[h_i];"
    )
    emit_store_real(
        ctx,
        indent="          ",
        tensor_name=y_name,
        idx_expr=f"((t_src * {num_dir} + d_i) * {b_size} + b_i) * {h_size} + h_i",
        value_expr=f"{h_next}[h_i]",
    )
    ctx.lines.append("        }")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append("  }")

    if yh_name is not None:
        ctx.lines.append(f"  for (size_t d_i = 0; d_i < {num_dir}; ++d_i) {{")
        ctx.lines.append(f"    for (size_t b_i = 0; b_i < {b_size}; ++b_i) {{")
        ctx.lines.append(f"      for (size_t h_i = 0; h_i < {h_size}; ++h_i) {{")
        emit_store_real(
            ctx,
            indent="        ",
            tensor_name=yh_name,
            idx_expr=f"(d_i * {b_size} + b_i) * {h_size} + h_i",
            value_expr=f"{h_state}[(d_i * {b_size} + b_i) * {h_size} + h_i]",
        )
        ctx.lines.append("      }")
        ctx.lines.append("    }")
        ctx.lines.append("  }")
