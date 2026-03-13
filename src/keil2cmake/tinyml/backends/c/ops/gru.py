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


@register_op("GRU")
def emit_gru(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 3:
        raise ValueError("GRU expects at least 3 inputs: X, W, R.")
    if len(node.outputs) < 1 or len(node.outputs) > 2:
        raise ValueError("GRU expects 1 or 2 outputs: Y, [Y_h].")

    x_name = node.inputs[0]
    w_name = node.inputs[1]
    r_name = node.inputs[2]
    b_name = node.inputs[3] if len(node.inputs) >= 4 and node.inputs[3] else None
    seq_lens_name = node.inputs[4] if len(node.inputs) >= 5 and node.inputs[4] else None
    h0_name = node.inputs[5] if len(node.inputs) >= 6 and node.inputs[5] else None
    y_name = node.outputs[0]
    yh_name = node.outputs[1] if len(node.outputs) >= 2 and node.outputs[1] else None

    assert_rec_dtype(ctx, "GRU", x_name, "X")
    assert_rec_dtype(ctx, "GRU", w_name, "W")
    assert_rec_dtype(ctx, "GRU", r_name, "R")
    assert_rec_dtype(ctx, "GRU", y_name, "Y")
    if b_name is not None:
        assert_rec_dtype(ctx, "GRU", b_name, "B")
    if h0_name is not None:
        assert_rec_dtype(ctx, "GRU", h0_name, "initial_h")
    if yh_name is not None:
        assert_rec_dtype(ctx, "GRU", yh_name, "Y_h")
    if seq_lens_name is not None and ctx.dtype(seq_lens_name) not in ("int64", "int32"):
        raise ValueError("GRU sequence_lens dtype must be int64/int32.")

    direction, expect_dirs = direction_and_count("GRU", node.attrs.get("direction", "forward"))
    linear_before_reset = int(node.attrs.get("linear_before_reset", 0))
    if linear_before_reset not in (0, 1):
        raise ValueError("GRU linear_before_reset must be 0 or 1.")

    x_shape = [int(v) for v in ctx.shape(x_name)]
    w_shape = [int(v) for v in ctx.shape(w_name)]
    r_shape = [int(v) for v in ctx.shape(r_name)]
    y_shape = [int(v) for v in ctx.shape(y_name)]
    if len(x_shape) != 3 or len(w_shape) != 3 or len(r_shape) != 3:
        raise ValueError("GRU expects X/W/R rank=3.")
    t_size, b_size, i_size = x_shape
    num_dir, wh, i_w = w_shape
    num_dir_r, rh0, rh1 = r_shape
    if num_dir != expect_dirs or num_dir_r != expect_dirs:
        raise ValueError("GRU num_directions does not match direction attribute.")
    if wh % 3 != 0 or rh0 % 3 != 0:
        raise ValueError("GRU W/R gate dimension must be 3*hidden_size.")
    h_size = wh // 3
    if i_w != i_size or rh0 != 3 * h_size or rh1 != h_size:
        raise ValueError("GRU W/R shape mismatch.")
    if y_shape != [t_size, num_dir, b_size, h_size]:
        raise ValueError("GRU Y output shape mismatch.")

    act_specs = build_activation_specs(
        "GRU",
        raw_acts=node.attrs.get("activations", None),
        raw_alpha=node.attrs.get("activation_alpha", None),
        raw_beta=node.attrs.get("activation_beta", None),
        expected_count=2 * num_dir,
        default_cycle=["sigmoid", "tanh"],
    )
    f_specs = [act_specs[2 * d] for d in range(num_dir)]
    g_specs = [act_specs[2 * d + 1] for d in range(num_dir)]
    clip_val = node.attrs.get("clip", None)
    clip_sym = None
    if clip_val is not None:
        clip_f = float(clip_val)
        if clip_f < 0.0:
            raise ValueError("GRU clip must be non-negative.")
        clip_sym = ctx.next_symbol("k2c_gru_clip")
        ctx.lines.append(f"  const float {clip_sym} = {clip_f:.9g}f;")

    if b_name is not None:
        b_shape = [int(v) for v in ctx.shape(b_name)]
        if b_shape != [num_dir, 6 * h_size]:
            raise ValueError("GRU bias shape must be [num_directions, 6*hidden_size].")
    if h0_name is not None:
        h0_shape = [int(v) for v in ctx.shape(h0_name)]
        if h0_shape != [num_dir, b_size, h_size]:
            raise ValueError("GRU initial_h shape mismatch.")
    if yh_name is not None:
        yh_shape = [int(v) for v in ctx.shape(yh_name)]
        if yh_shape != [num_dir, b_size, h_size]:
            raise ValueError("GRU Y_h shape mismatch.")
    if seq_lens_name is not None:
        seq_shape = [int(v) for v in ctx.shape(seq_lens_name)]
        if seq_shape != [b_size]:
            raise ValueError("GRU sequence_lens shape must be [batch_size].")

    seq = ctx.map_ptr(seq_lens_name) if seq_lens_name is not None else None

    h_state = ctx.next_symbol("k2c_gru_h")
    h_next = ctx.next_symbol("k2c_gru_hn")
    z_buf = ctx.next_symbol("k2c_gru_z")
    r_buf = ctx.next_symbol("k2c_gru_r")

    ctx.lines.append(f"  float {h_state}[{num_dir} * {b_size} * {h_size}];")
    ctx.lines.append(f"  float {h_next}[{h_size}];")
    ctx.lines.append(f"  float {z_buf}[{h_size}];")
    ctx.lines.append(f"  float {r_buf}[{h_size}];")
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
    ctx.lines.append("          float z_sum = 0.0f;")
    ctx.lines.append("          float r_sum = 0.0f;")
    ctx.lines.append(f"          for (size_t i_i = 0; i_i < {i_size}; ++i_i) {{")
    ctx.lines.append(
        f"            float xv = {read_real_expr(ctx, x_name, f'(t_src * {b_size} + b_i) * {i_size} + i_i')};"
    )
    ctx.lines.append(
        f"            z_sum += xv * {read_real_expr(ctx, w_name, f'((d_i * {3 * h_size} + h_i) * {i_size}) + i_i')};"
    )
    ctx.lines.append(
        f"            r_sum += xv * {read_real_expr(ctx, w_name, f'((d_i * {3 * h_size} + {h_size} + h_i) * {i_size}) + i_i')};"
    )
    ctx.lines.append("          }")
    ctx.lines.append(f"          for (size_t hh = 0; hh < {h_size}; ++hh) {{")
    ctx.lines.append(f"            float hv = {h_state}[(d_i * {b_size} + b_i) * {h_size} + hh];")
    ctx.lines.append(
        f"            z_sum += hv * {read_real_expr(ctx, r_name, f'((d_i * {3 * h_size} + h_i) * {h_size}) + hh')};"
    )
    ctx.lines.append(
        f"            r_sum += hv * {read_real_expr(ctx, r_name, f'((d_i * {3 * h_size} + {h_size} + h_i) * {h_size}) + hh')};"
    )
    ctx.lines.append("          }")
    if b_name is not None:
        ctx.lines.append(
            f"          z_sum += {read_real_expr(ctx, b_name, f'd_i * {6 * h_size} + h_i')} + "
            f"{read_real_expr(ctx, b_name, f'd_i * {6 * h_size} + {3 * h_size} + h_i')};"
        )
        ctx.lines.append(
            f"          r_sum += {read_real_expr(ctx, b_name, f'd_i * {6 * h_size} + {h_size} + h_i')} + "
            f"{read_real_expr(ctx, b_name, f'd_i * {6 * h_size} + {4 * h_size} + h_i')};"
        )
    emit_clip(ctx, indent="          ", value_var="z_sum", clip_var=clip_sym)
    emit_clip(ctx, indent="          ", value_var="r_sum", clip_var=clip_sym)
    ctx.lines.append("          float z_val = 0.0f;")
    emit_activation_assign(
        ctx,
        indent="          ",
        out_var="z_val",
        pre_var="z_sum",
        specs_by_dir=f_specs,
        dir_var="d_i",
    )
    ctx.lines.append("          float r_val = 0.0f;")
    emit_activation_assign(
        ctx,
        indent="          ",
        out_var="r_val",
        pre_var="r_sum",
        specs_by_dir=f_specs,
        dir_var="d_i",
    )
    ctx.lines.append(f"          {z_buf}[h_i] = z_val;")
    ctx.lines.append(f"          {r_buf}[h_i] = r_val;")
    ctx.lines.append("        }")

    ctx.lines.append(f"        for (size_t h_i = 0; h_i < {h_size}; ++h_i) {{")
    ctx.lines.append("          float h_sum = 0.0f;")
    ctx.lines.append(f"          for (size_t i_i = 0; i_i < {i_size}; ++i_i) {{")
    ctx.lines.append(
        f"            float xv = {read_real_expr(ctx, x_name, f'(t_src * {b_size} + b_i) * {i_size} + i_i')};"
    )
    ctx.lines.append(
        f"            h_sum += xv * {read_real_expr(ctx, w_name, f'((d_i * {3 * h_size} + {2 * h_size} + h_i) * {i_size}) + i_i')};"
    )
    ctx.lines.append("          }")
    if linear_before_reset == 0:
        ctx.lines.append(f"          for (size_t hh = 0; hh < {h_size}; ++hh) {{")
        ctx.lines.append(
            f"            float hv = {h_state}[(d_i * {b_size} + b_i) * {h_size} + hh] * {r_buf}[hh];"
        )
        ctx.lines.append(
            f"            h_sum += hv * {read_real_expr(ctx, r_name, f'((d_i * {3 * h_size} + {2 * h_size} + h_i) * {h_size}) + hh')};"
        )
        ctx.lines.append("          }")
        if b_name is not None:
            ctx.lines.append(
                f"          h_sum += {read_real_expr(ctx, b_name, f'd_i * {6 * h_size} + {2 * h_size} + h_i')} + "
                f"{read_real_expr(ctx, b_name, f'd_i * {6 * h_size} + {5 * h_size} + h_i')};"
            )
    else:
        ctx.lines.append("          float rec_sum = 0.0f;")
        ctx.lines.append(f"          for (size_t hh = 0; hh < {h_size}; ++hh) {{")
        ctx.lines.append(f"            float hv = {h_state}[(d_i * {b_size} + b_i) * {h_size} + hh];")
        ctx.lines.append(
            f"            rec_sum += hv * {read_real_expr(ctx, r_name, f'((d_i * {3 * h_size} + {2 * h_size} + h_i) * {h_size}) + hh')};"
        )
        ctx.lines.append("          }")
        if b_name is not None:
            ctx.lines.append(
                f"          rec_sum += {read_real_expr(ctx, b_name, f'd_i * {6 * h_size} + {5 * h_size} + h_i')};"
            )
            ctx.lines.append(
                f"          h_sum += {read_real_expr(ctx, b_name, f'd_i * {6 * h_size} + {2 * h_size} + h_i')};"
            )
        ctx.lines.append(f"          h_sum += {r_buf}[h_i] * rec_sum;")
    emit_clip(ctx, indent="          ", value_var="h_sum", clip_var=clip_sym)
    ctx.lines.append("          float h_tilde = 0.0f;")
    emit_activation_assign(
        ctx,
        indent="          ",
        out_var="h_tilde",
        pre_var="h_sum",
        specs_by_dir=g_specs,
        dir_var="d_i",
    )
    ctx.lines.append(f"          float h_prev = {h_state}[(d_i * {b_size} + b_i) * {h_size} + h_i];")
    ctx.lines.append(f"          float z = {z_buf}[h_i];")
    ctx.lines.append(f"          {h_next}[h_i] = (1.0f - z) * h_tilde + z * h_prev;")
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
