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


@register_op("LSTM")
def emit_lstm(ctx: EmitContext, node: NodeInfo) -> None:
    if len(node.inputs) < 3:
        raise ValueError("LSTM expects at least 3 inputs: X, W, R.")
    if len(node.outputs) < 1 or len(node.outputs) > 3:
        raise ValueError("LSTM expects 1..3 outputs: Y, [Y_h], [Y_c].")

    x_name = node.inputs[0]
    w_name = node.inputs[1]
    r_name = node.inputs[2]
    b_name = node.inputs[3] if len(node.inputs) >= 4 and node.inputs[3] else None
    seq_lens_name = node.inputs[4] if len(node.inputs) >= 5 and node.inputs[4] else None
    h0_name = node.inputs[5] if len(node.inputs) >= 6 and node.inputs[5] else None
    c0_name = node.inputs[6] if len(node.inputs) >= 7 and node.inputs[6] else None
    p_name = node.inputs[7] if len(node.inputs) >= 8 and node.inputs[7] else None
    y_name = node.outputs[0]
    yh_name = node.outputs[1] if len(node.outputs) >= 2 and node.outputs[1] else None
    yc_name = node.outputs[2] if len(node.outputs) >= 3 and node.outputs[2] else None

    assert_rec_dtype(ctx, "LSTM", x_name, "X")
    assert_rec_dtype(ctx, "LSTM", w_name, "W")
    assert_rec_dtype(ctx, "LSTM", r_name, "R")
    assert_rec_dtype(ctx, "LSTM", y_name, "Y")
    if b_name is not None:
        assert_rec_dtype(ctx, "LSTM", b_name, "B")
    if h0_name is not None:
        assert_rec_dtype(ctx, "LSTM", h0_name, "initial_h")
    if c0_name is not None:
        assert_rec_dtype(ctx, "LSTM", c0_name, "initial_c")
    if yh_name is not None:
        assert_rec_dtype(ctx, "LSTM", yh_name, "Y_h")
    if yc_name is not None:
        assert_rec_dtype(ctx, "LSTM", yc_name, "Y_c")
    if p_name is not None:
        assert_rec_dtype(ctx, "LSTM", p_name, "P")
    if seq_lens_name is not None and ctx.dtype(seq_lens_name) not in ("int64", "int32"):
        raise ValueError("LSTM sequence_lens dtype must be int64/int32.")

    direction, expect_dirs = direction_and_count("LSTM", node.attrs.get("direction", "forward"))
    input_forget = int(node.attrs.get("input_forget", 0))
    if input_forget not in (0, 1):
        raise ValueError("LSTM input_forget must be 0 or 1.")

    x_shape = [int(v) for v in ctx.shape(x_name)]
    w_shape = [int(v) for v in ctx.shape(w_name)]
    r_shape = [int(v) for v in ctx.shape(r_name)]
    y_shape = [int(v) for v in ctx.shape(y_name)]
    if len(x_shape) != 3 or len(w_shape) != 3 or len(r_shape) != 3:
        raise ValueError("LSTM expects X/W/R rank=3.")
    t_size, b_size, i_size = x_shape
    num_dir, wh, i_w = w_shape
    num_dir_r, rh0, rh1 = r_shape
    if num_dir != expect_dirs or num_dir_r != expect_dirs:
        raise ValueError("LSTM num_directions does not match direction attribute.")
    if wh % 4 != 0 or rh0 % 4 != 0:
        raise ValueError("LSTM W/R gate dimension must be 4*hidden_size.")
    h_size = wh // 4
    if i_w != i_size or rh0 != 4 * h_size or rh1 != h_size:
        raise ValueError("LSTM W/R shape mismatch.")
    if y_shape != [t_size, num_dir, b_size, h_size]:
        raise ValueError("LSTM Y output shape mismatch.")

    act_specs = build_activation_specs(
        "LSTM",
        raw_acts=node.attrs.get("activations", None),
        raw_alpha=node.attrs.get("activation_alpha", None),
        raw_beta=node.attrs.get("activation_beta", None),
        expected_count=3 * num_dir,
        default_cycle=["sigmoid", "tanh", "tanh"],
    )
    f_specs = [act_specs[3 * d] for d in range(num_dir)]
    g_specs = [act_specs[3 * d + 1] for d in range(num_dir)]
    h_specs = [act_specs[3 * d + 2] for d in range(num_dir)]
    clip_val = node.attrs.get("clip", None)
    clip_sym = None
    if clip_val is not None:
        clip_f = float(clip_val)
        if clip_f < 0.0:
            raise ValueError("LSTM clip must be non-negative.")
        clip_sym = ctx.next_symbol("k2c_lstm_clip")
        ctx.lines.append(f"  const float {clip_sym} = {clip_f:.9g}f;")

    if b_name is not None:
        b_shape = [int(v) for v in ctx.shape(b_name)]
        if b_shape != [num_dir, 8 * h_size]:
            raise ValueError("LSTM bias shape must be [num_directions, 8*hidden_size].")
    if h0_name is not None:
        h0_shape = [int(v) for v in ctx.shape(h0_name)]
        if h0_shape != [num_dir, b_size, h_size]:
            raise ValueError("LSTM initial_h shape mismatch.")
    if c0_name is not None:
        c0_shape = [int(v) for v in ctx.shape(c0_name)]
        if c0_shape != [num_dir, b_size, h_size]:
            raise ValueError("LSTM initial_c shape mismatch.")
    if yh_name is not None:
        yh_shape = [int(v) for v in ctx.shape(yh_name)]
        if yh_shape != [num_dir, b_size, h_size]:
            raise ValueError("LSTM Y_h shape mismatch.")
    if yc_name is not None:
        yc_shape = [int(v) for v in ctx.shape(yc_name)]
        if yc_shape != [num_dir, b_size, h_size]:
            raise ValueError("LSTM Y_c shape mismatch.")
    if p_name is not None:
        p_shape = [int(v) for v in ctx.shape(p_name)]
        if p_shape != [num_dir, 3 * h_size]:
            raise ValueError("LSTM peephole shape must be [num_directions, 3*hidden_size].")
    if seq_lens_name is not None:
        seq_shape = [int(v) for v in ctx.shape(seq_lens_name)]
        if seq_shape != [b_size]:
            raise ValueError("LSTM sequence_lens shape must be [batch_size].")

    seq = ctx.map_ptr(seq_lens_name) if seq_lens_name is not None else None

    h_state = ctx.next_symbol("k2c_lstm_h")
    c_state = ctx.next_symbol("k2c_lstm_c")
    h_next = ctx.next_symbol("k2c_lstm_hn")
    c_next = ctx.next_symbol("k2c_lstm_cn")

    ctx.lines.append(f"  float {h_state}[{num_dir} * {b_size} * {h_size}];")
    ctx.lines.append(f"  float {c_state}[{num_dir} * {b_size} * {h_size}];")
    ctx.lines.append(f"  float {h_next}[{h_size}];")
    ctx.lines.append(f"  float {c_next}[{h_size}];")
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
    if c0_name is not None:
        ctx.lines.append(f"  for (size_t d_i = 0; d_i < {num_dir}; ++d_i) {{")
        ctx.lines.append(f"    for (size_t b_i = 0; b_i < {b_size}; ++b_i) {{")
        ctx.lines.append(f"      for (size_t h_i = 0; h_i < {h_size}; ++h_i) {{")
        ctx.lines.append(
            f"        {c_state}[(d_i * {b_size} + b_i) * {h_size} + h_i] = "
            f"{read_real_expr(ctx, c0_name, f'(d_i * {b_size} + b_i) * {h_size} + h_i')};"
        )
        ctx.lines.append("      }")
        ctx.lines.append("    }")
        ctx.lines.append("  }")
    else:
        ctx.lines.append(f"  for (size_t i = 0; i < {num_dir * b_size * h_size}; ++i) {c_state}[i] = 0.0f;")

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
    ctx.lines.append("          float i_sum = 0.0f;")
    ctx.lines.append("          float o_sum = 0.0f;")
    ctx.lines.append("          float f_sum = 0.0f;")
    ctx.lines.append("          float g_sum = 0.0f;")
    ctx.lines.append(f"          for (size_t i_i = 0; i_i < {i_size}; ++i_i) {{")
    ctx.lines.append(
        f"            float xv = {read_real_expr(ctx, x_name, f'(t_src * {b_size} + b_i) * {i_size} + i_i')};"
    )
    ctx.lines.append(
        f"            i_sum += xv * {read_real_expr(ctx, w_name, f'((d_i * {4 * h_size} + h_i) * {i_size}) + i_i')};"
    )
    ctx.lines.append(
        f"            o_sum += xv * {read_real_expr(ctx, w_name, f'((d_i * {4 * h_size} + {h_size} + h_i) * {i_size}) + i_i')};"
    )
    ctx.lines.append(
        f"            f_sum += xv * {read_real_expr(ctx, w_name, f'((d_i * {4 * h_size} + {2 * h_size} + h_i) * {i_size}) + i_i')};"
    )
    ctx.lines.append(
        f"            g_sum += xv * {read_real_expr(ctx, w_name, f'((d_i * {4 * h_size} + {3 * h_size} + h_i) * {i_size}) + i_i')};"
    )
    ctx.lines.append("          }")
    ctx.lines.append(f"          for (size_t hh = 0; hh < {h_size}; ++hh) {{")
    ctx.lines.append(f"            float hv = {h_state}[(d_i * {b_size} + b_i) * {h_size} + hh];")
    ctx.lines.append(
        f"            i_sum += hv * {read_real_expr(ctx, r_name, f'((d_i * {4 * h_size} + h_i) * {h_size}) + hh')};"
    )
    ctx.lines.append(
        f"            o_sum += hv * {read_real_expr(ctx, r_name, f'((d_i * {4 * h_size} + {h_size} + h_i) * {h_size}) + hh')};"
    )
    ctx.lines.append(
        f"            f_sum += hv * {read_real_expr(ctx, r_name, f'((d_i * {4 * h_size} + {2 * h_size} + h_i) * {h_size}) + hh')};"
    )
    ctx.lines.append(
        f"            g_sum += hv * {read_real_expr(ctx, r_name, f'((d_i * {4 * h_size} + {3 * h_size} + h_i) * {h_size}) + hh')};"
    )
    ctx.lines.append("          }")
    if b_name is not None:
        ctx.lines.append(
            f"          i_sum += {read_real_expr(ctx, b_name, f'd_i * {8 * h_size} + h_i')} + "
            f"{read_real_expr(ctx, b_name, f'd_i * {8 * h_size} + {4 * h_size} + h_i')};"
        )
        ctx.lines.append(
            f"          o_sum += {read_real_expr(ctx, b_name, f'd_i * {8 * h_size} + {h_size} + h_i')} + "
            f"{read_real_expr(ctx, b_name, f'd_i * {8 * h_size} + {5 * h_size} + h_i')};"
        )
        ctx.lines.append(
            f"          f_sum += {read_real_expr(ctx, b_name, f'd_i * {8 * h_size} + {2 * h_size} + h_i')} + "
            f"{read_real_expr(ctx, b_name, f'd_i * {8 * h_size} + {6 * h_size} + h_i')};"
        )
        ctx.lines.append(
            f"          g_sum += {read_real_expr(ctx, b_name, f'd_i * {8 * h_size} + {3 * h_size} + h_i')} + "
            f"{read_real_expr(ctx, b_name, f'd_i * {8 * h_size} + {7 * h_size} + h_i')};"
        )
    ctx.lines.append(f"          float c_prev = {c_state}[(d_i * {b_size} + b_i) * {h_size} + h_i];")
    if p_name is not None:
        ctx.lines.append(
            f"          i_sum += {read_real_expr(ctx, p_name, f'd_i * {3 * h_size} + h_i')} * c_prev;"
        )
        ctx.lines.append(
            f"          f_sum += {read_real_expr(ctx, p_name, f'd_i * {3 * h_size} + {2 * h_size} + h_i')} * c_prev;"
        )
    emit_clip(ctx, indent="          ", value_var="i_sum", clip_var=clip_sym)
    emit_clip(ctx, indent="          ", value_var="f_sum", clip_var=clip_sym)
    emit_clip(ctx, indent="          ", value_var="g_sum", clip_var=clip_sym)
    ctx.lines.append("          float i_gate = 0.0f;")
    emit_activation_assign(
        ctx,
        indent="          ",
        out_var="i_gate",
        pre_var="i_sum",
        specs_by_dir=f_specs,
        dir_var="d_i",
    )
    if input_forget == 1:
        ctx.lines.append("          float f_gate = 1.0f - i_gate;")
    else:
        ctx.lines.append("          float f_gate = 0.0f;")
        emit_activation_assign(
            ctx,
            indent="          ",
            out_var="f_gate",
            pre_var="f_sum",
            specs_by_dir=f_specs,
            dir_var="d_i",
        )
    ctx.lines.append("          float g_gate = 0.0f;")
    emit_activation_assign(
        ctx,
        indent="          ",
        out_var="g_gate",
        pre_var="g_sum",
        specs_by_dir=g_specs,
        dir_var="d_i",
    )
    ctx.lines.append("          float c_new = f_gate * c_prev + i_gate * g_gate;")
    if p_name is not None:
        ctx.lines.append(
            f"          o_sum += {read_real_expr(ctx, p_name, f'd_i * {3 * h_size} + {h_size} + h_i')} * c_new;"
        )
    emit_clip(ctx, indent="          ", value_var="o_sum", clip_var=clip_sym)
    ctx.lines.append("          float o_gate = 0.0f;")
    emit_activation_assign(
        ctx,
        indent="          ",
        out_var="o_gate",
        pre_var="o_sum",
        specs_by_dir=f_specs,
        dir_var="d_i",
    )
    ctx.lines.append("          float c_act = c_new;")
    emit_clip(ctx, indent="          ", value_var="c_act", clip_var=clip_sym)
    ctx.lines.append("          float h_act = 0.0f;")
    emit_activation_assign(
        ctx,
        indent="          ",
        out_var="h_act",
        pre_var="c_act",
        specs_by_dir=h_specs,
        dir_var="d_i",
    )
    ctx.lines.append("          float h_new = o_gate * h_act;")
    ctx.lines.append(f"          {c_next}[h_i] = c_new;")
    ctx.lines.append(f"          {h_next}[h_i] = h_new;")
    ctx.lines.append("        }")
    ctx.lines.append(f"        for (size_t h_i = 0; h_i < {h_size}; ++h_i) {{")
    ctx.lines.append(
        f"          {c_state}[(d_i * {b_size} + b_i) * {h_size} + h_i] = {c_next}[h_i];"
    )
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
    if yc_name is not None:
        ctx.lines.append(f"  for (size_t d_i = 0; d_i < {num_dir}; ++d_i) {{")
        ctx.lines.append(f"    for (size_t b_i = 0; b_i < {b_size}; ++b_i) {{")
        ctx.lines.append(f"      for (size_t h_i = 0; h_i < {h_size}; ++h_i) {{")
        emit_store_real(
            ctx,
            indent="        ",
            tensor_name=yc_name,
            idx_expr=f"(d_i * {b_size} + b_i) * {h_size} + h_i",
            value_expr=f"{c_state}[(d_i * {b_size} + b_i) * {h_size} + h_i]",
        )
        ctx.lines.append("      }")
        ctx.lines.append("    }")
        ctx.lines.append("  }")
