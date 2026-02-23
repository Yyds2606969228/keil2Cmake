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

    for name in (x_name, w_name, r_name, y_name):
        if ctx.dtype(name) != "float32":
            raise ValueError("LSTM currently supports float32 tensors only.")
    if b_name is not None and ctx.dtype(b_name) != "float32":
        raise ValueError("LSTM bias dtype must be float32.")
    if h0_name is not None and ctx.dtype(h0_name) != "float32":
        raise ValueError("LSTM initial_h dtype must be float32.")
    if c0_name is not None and ctx.dtype(c0_name) != "float32":
        raise ValueError("LSTM initial_c dtype must be float32.")
    if yh_name is not None and ctx.dtype(yh_name) != "float32":
        raise ValueError("LSTM Y_h dtype must be float32.")
    if yc_name is not None and ctx.dtype(yc_name) != "float32":
        raise ValueError("LSTM Y_c dtype must be float32.")
    if seq_lens_name is not None and ctx.dtype(seq_lens_name) not in ("int64", "int32"):
        raise ValueError("LSTM sequence_lens dtype must be int64/int32.")
    if p_name is not None:
        raise ValueError("LSTM currently does not support peephole input P.")

    direction = _decode_attr_str(node.attrs.get("direction", "forward"), "forward")
    if direction != "forward":
        raise ValueError("LSTM currently supports direction='forward' only.")
    acts = node.attrs.get("activations", None)
    if acts is not None:
        act_list: list[str] = []
        for v in acts:
            if isinstance(v, bytes):
                act_list.append(v.decode("utf-8", errors="ignore").lower())
            else:
                act_list.append(str(v).lower())
        if len(act_list) >= 3 and (act_list[0] != "sigmoid" or act_list[1] != "tanh" or act_list[2] != "tanh"):
            raise ValueError("LSTM currently supports activations=[Sigmoid,Tanh,Tanh] only.")

    x_shape = [int(v) for v in ctx.shape(x_name)]
    w_shape = [int(v) for v in ctx.shape(w_name)]
    r_shape = [int(v) for v in ctx.shape(r_name)]
    y_shape = [int(v) for v in ctx.shape(y_name)]
    if len(x_shape) != 3 or len(w_shape) != 3 or len(r_shape) != 3:
        raise ValueError("LSTM expects X/W/R rank=3.")
    t_size, b_size, i_size = x_shape
    num_dir, wh, i_w = w_shape
    num_dir_r, rh0, rh1 = r_shape
    if num_dir != 1 or num_dir_r != 1:
        raise ValueError("LSTM currently supports num_directions=1 only.")
    if wh % 4 != 0 or rh0 % 4 != 0:
        raise ValueError("LSTM W/R gate dimension must be 4*hidden_size.")
    h_size = wh // 4
    if i_w != i_size or rh0 != 4 * h_size or rh1 != h_size:
        raise ValueError("LSTM W/R shape mismatch.")
    if y_shape != [t_size, 1, b_size, h_size]:
        raise ValueError("LSTM Y output shape mismatch.")
    if b_name is not None:
        b_shape = [int(v) for v in ctx.shape(b_name)]
        if b_shape != [1, 8 * h_size]:
            raise ValueError("LSTM bias shape must be [1,8*hidden_size].")
    if h0_name is not None:
        h0_shape = [int(v) for v in ctx.shape(h0_name)]
        if h0_shape != [1, b_size, h_size]:
            raise ValueError("LSTM initial_h shape mismatch.")
    if c0_name is not None:
        c0_shape = [int(v) for v in ctx.shape(c0_name)]
        if c0_shape != [1, b_size, h_size]:
            raise ValueError("LSTM initial_c shape mismatch.")
    if yh_name is not None:
        yh_shape = [int(v) for v in ctx.shape(yh_name)]
        if yh_shape != [1, b_size, h_size]:
            raise ValueError("LSTM Y_h shape mismatch.")
    if yc_name is not None:
        yc_shape = [int(v) for v in ctx.shape(yc_name)]
        if yc_shape != [1, b_size, h_size]:
            raise ValueError("LSTM Y_c shape mismatch.")
    if seq_lens_name is not None:
        seq_t = ctx.model.tensors.get(seq_lens_name)
        if seq_t is None or seq_t.data is None or len(seq_t.data) != b_size:
            raise ValueError("LSTM currently requires constant sequence_lens of length batch_size.")
        for v in seq_t.data:
            if int(v) != t_size:
                raise ValueError("LSTM currently requires sequence_lens equal to seq_len.")

    x = ctx.map_ptr(x_name)
    w = ctx.map_ptr(w_name)
    r = ctx.map_ptr(r_name)
    b = ctx.map_ptr(b_name) if b_name is not None else None
    h0 = ctx.map_ptr(h0_name) if h0_name is not None else None
    c0 = ctx.map_ptr(c0_name) if c0_name is not None else None
    y = ctx.map_ptr(y_name)
    yh = ctx.map_ptr(yh_name) if yh_name is not None else None
    yc = ctx.map_ptr(yc_name) if yc_name is not None else None

    h_state = ctx.next_symbol("k2c_lstm_h")
    h_next = ctx.next_symbol("k2c_lstm_hn")
    c_state = ctx.next_symbol("k2c_lstm_c")
    c_next = ctx.next_symbol("k2c_lstm_cn")

    ctx.lines.append(f"  float {h_state}[{b_size} * {h_size}];")
    ctx.lines.append(f"  float {h_next}[{b_size} * {h_size}];")
    ctx.lines.append(f"  float {c_state}[{b_size} * {h_size}];")
    ctx.lines.append(f"  float {c_next}[{b_size} * {h_size}];")
    if h0 is not None:
        ctx.lines.append(f"  for (size_t b_i = 0; b_i < {b_size}; ++b_i) {{")
        ctx.lines.append(f"    for (size_t h_i = 0; h_i < {h_size}; ++h_i) {{")
        ctx.lines.append(f"      {h_state}[b_i * {h_size} + h_i] = {h0}[(0 * {b_size} + b_i) * {h_size} + h_i];")
        ctx.lines.append("    }")
        ctx.lines.append("  }")
    else:
        ctx.lines.append(f"  for (size_t i = 0; i < {b_size * h_size}; ++i) {h_state}[i] = 0.0f;")
    if c0 is not None:
        ctx.lines.append(f"  for (size_t b_i = 0; b_i < {b_size}; ++b_i) {{")
        ctx.lines.append(f"    for (size_t h_i = 0; h_i < {h_size}; ++h_i) {{")
        ctx.lines.append(f"      {c_state}[b_i * {h_size} + h_i] = {c0}[(0 * {b_size} + b_i) * {h_size} + h_i];")
        ctx.lines.append("    }")
        ctx.lines.append("  }")
    else:
        ctx.lines.append(f"  for (size_t i = 0; i < {b_size * h_size}; ++i) {c_state}[i] = 0.0f;")

    ctx.lines.append(f"  for (size_t t_i = 0; t_i < {t_size}; ++t_i) {{")
    ctx.lines.append(f"    for (size_t b_i = 0; b_i < {b_size}; ++b_i) {{")
    ctx.lines.append(f"      for (size_t h_i = 0; h_i < {h_size}; ++h_i) {{")
    ctx.lines.append("        float i_sum = 0.0f, o_sum = 0.0f, f_sum = 0.0f, g_sum = 0.0f;")
    ctx.lines.append(f"        for (size_t i_i = 0; i_i < {i_size}; ++i_i) {{")
    ctx.lines.append(f"          float xv = {x}[(t_i * {b_size} + b_i) * {i_size} + i_i];")
    ctx.lines.append(f"          i_sum += xv * {w}[((0 * {4 * h_size} + h_i) * {i_size}) + i_i];")
    ctx.lines.append(f"          o_sum += xv * {w}[((0 * {4 * h_size} + {h_size} + h_i) * {i_size}) + i_i];")
    ctx.lines.append(f"          f_sum += xv * {w}[((0 * {4 * h_size} + {2 * h_size} + h_i) * {i_size}) + i_i];")
    ctx.lines.append(f"          g_sum += xv * {w}[((0 * {4 * h_size} + {3 * h_size} + h_i) * {i_size}) + i_i];")
    ctx.lines.append("        }")
    ctx.lines.append(f"        for (size_t hh = 0; hh < {h_size}; ++hh) {{")
    ctx.lines.append(f"          float hv = {h_state}[b_i * {h_size} + hh];")
    ctx.lines.append(f"          i_sum += hv * {r}[((0 * {4 * h_size} + h_i) * {h_size}) + hh];")
    ctx.lines.append(f"          o_sum += hv * {r}[((0 * {4 * h_size} + {h_size} + h_i) * {h_size}) + hh];")
    ctx.lines.append(f"          f_sum += hv * {r}[((0 * {4 * h_size} + {2 * h_size} + h_i) * {h_size}) + hh];")
    ctx.lines.append(f"          g_sum += hv * {r}[((0 * {4 * h_size} + {3 * h_size} + h_i) * {h_size}) + hh];")
    ctx.lines.append("        }")
    if b is not None:
        ctx.lines.append(f"        i_sum += {b}[0 * {8 * h_size} + h_i] + {b}[0 * {8 * h_size} + {4 * h_size} + h_i];")
        ctx.lines.append(f"        o_sum += {b}[0 * {8 * h_size} + {h_size} + h_i] + {b}[0 * {8 * h_size} + {5 * h_size} + h_i];")
        ctx.lines.append(f"        f_sum += {b}[0 * {8 * h_size} + {2 * h_size} + h_i] + {b}[0 * {8 * h_size} + {6 * h_size} + h_i];")
        ctx.lines.append(f"        g_sum += {b}[0 * {8 * h_size} + {3 * h_size} + h_i] + {b}[0 * {8 * h_size} + {7 * h_size} + h_i];")
    ctx.lines.append("        float i_gate = 1.0f / (1.0f + expf(-i_sum));")
    ctx.lines.append("        float o_gate = 1.0f / (1.0f + expf(-o_sum));")
    ctx.lines.append("        float f_gate = 1.0f / (1.0f + expf(-f_sum));")
    ctx.lines.append("        float g_gate = tanhf(g_sum);")
    ctx.lines.append(f"        float c_prev = {c_state}[b_i * {h_size} + h_i];")
    ctx.lines.append("        float c_new = f_gate * c_prev + i_gate * g_gate;")
    ctx.lines.append("        float h_new = o_gate * tanhf(c_new);")
    ctx.lines.append(f"        {c_next}[b_i * {h_size} + h_i] = c_new;")
    ctx.lines.append(f"        {h_next}[b_i * {h_size} + h_i] = h_new;")
    ctx.lines.append(f"        {y}[((t_i * 1 + 0) * {b_size} + b_i) * {h_size} + h_i] = h_new;")
    ctx.lines.append("      }")
    ctx.lines.append("    }")
    ctx.lines.append(f"    for (size_t i = 0; i < {b_size * h_size}; ++i) {{ {h_state}[i] = {h_next}[i]; {c_state}[i] = {c_next}[i]; }}")
    ctx.lines.append("  }")

    if yh is not None:
        ctx.lines.append(f"  for (size_t b_i = 0; b_i < {b_size}; ++b_i) {{")
        ctx.lines.append(f"    for (size_t h_i = 0; h_i < {h_size}; ++h_i) {{")
        ctx.lines.append(f"      {yh}[(0 * {b_size} + b_i) * {h_size} + h_i] = {h_state}[b_i * {h_size} + h_i];")
        ctx.lines.append("    }")
        ctx.lines.append("  }")
    if yc is not None:
        ctx.lines.append(f"  for (size_t b_i = 0; b_i < {b_size}; ++b_i) {{")
        ctx.lines.append(f"    for (size_t h_i = 0; h_i < {h_size}; ++h_i) {{")
        ctx.lines.append(f"      {yc}[(0 * {b_size} + b_i) * {h_size} + h_i] = {c_state}[b_i * {h_size} + h_i];")
        ctx.lines.append("    }")
        ctx.lines.append("  }")
