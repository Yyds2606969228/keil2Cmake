# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any

import numpy as np

from ...converter.ir import ModelIR
from .utils import _decode_attr_str, _dequantize_int, _qparams, _quantize_float, _tensor_dtype

def _direction_and_count(op_name: str, direction_attr: Any) -> tuple[str, int]:
    direction = _decode_attr_str(direction_attr, "forward")
    if direction == "bidirectional":
        return direction, 2
    if direction in ("forward", "reverse"):
        return direction, 1
    raise ValueError(f"{op_name} direction must be forward/reverse/bidirectional.")


def _activation_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8", errors="ignore").lower())
        else:
            out.append(str(item).lower())
    return out


def _float_list(raw: Any) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [float(v) for v in raw]
    return [float(raw)]


_SUPPORTED_REC_ACTS = {
    "relu",
    "tanh",
    "sigmoid",
    "affine",
    "leakyrelu",
    "thresholdedrelu",
    "scaledtanh",
    "hardsigmoid",
    "elu",
    "softsign",
    "softplus",
}


def _rec_default_alpha(act: str) -> float:
    if act == "leakyrelu":
        return 0.01
    if act == "thresholdedrelu":
        return 1.0
    if act == "scaledtanh":
        return 1.0
    if act == "hardsigmoid":
        return 0.2
    if act == "elu":
        return 1.0
    if act == "affine":
        return 1.0
    return 0.0


def _rec_default_beta(act: str) -> float:
    if act == "scaledtanh":
        return 1.0
    if act == "hardsigmoid":
        return 0.5
    if act == "affine":
        return 0.0
    return 0.0


def _rec_activation_specs(
    op_name: str,
    node,
    *,
    expected_count: int,
    default_cycle: list[str],
) -> list[tuple[str, float, float]]:
    acts = _activation_list(node.attrs.get("activations", None))
    if not acts:
        reps = expected_count // len(default_cycle)
        if reps * len(default_cycle) != expected_count:
            raise ValueError(f"{op_name} invalid default activation cycle.")
        acts = default_cycle * reps
    if len(acts) != expected_count:
        raise ValueError(f"{op_name} activations count mismatch.")
    for act in acts:
        if act not in _SUPPORTED_REC_ACTS:
            raise ValueError(f"{op_name} activation '{act}' is unsupported.")
    alphas = _float_list(node.attrs.get("activation_alpha", None))
    betas = _float_list(node.attrs.get("activation_beta", None))
    specs: list[tuple[str, float, float]] = []
    for idx, act in enumerate(acts):
        alpha = alphas[idx] if idx < len(alphas) else _rec_default_alpha(act)
        beta = betas[idx] if idx < len(betas) else _rec_default_beta(act)
        specs.append((act, float(alpha), float(beta)))
    return specs


def _rec_apply_activation(x: np.ndarray, spec: tuple[str, float, float]) -> np.ndarray:
    act, alpha, beta = spec
    if act == "relu":
        return np.maximum(x, 0.0).astype(np.float32)
    if act == "tanh":
        return np.tanh(x).astype(np.float32)
    if act == "sigmoid":
        return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)
    if act == "affine":
        return (alpha * x + beta).astype(np.float32)
    if act == "leakyrelu":
        return np.where(x >= 0.0, x, alpha * x).astype(np.float32)
    if act == "thresholdedrelu":
        return np.where(x > alpha, x, 0.0).astype(np.float32)
    if act == "scaledtanh":
        return (alpha * np.tanh(beta * x)).astype(np.float32)
    if act == "hardsigmoid":
        return np.clip(alpha * x + beta, 0.0, 1.0).astype(np.float32)
    if act == "elu":
        return np.where(x >= 0.0, x, alpha * (np.exp(x) - 1.0)).astype(np.float32)
    if act == "softsign":
        return (x / (1.0 + np.abs(x))).astype(np.float32)
    if act == "softplus":
        return np.log1p(np.exp(x)).astype(np.float32)
    raise ValueError(f"Unsupported recurrent activation '{act}'.")


def _rec_apply_clip(x: np.ndarray, clip: float | None) -> np.ndarray:
    if clip is None:
        return x
    return np.clip(x, -clip, clip).astype(np.float32)


def _resolve_sequence_lens(
    seq_data: np.ndarray | None,
    batch_size: int,
    t_size: int,
    op_name: str,
) -> np.ndarray:
    if seq_data is None:
        return np.full((batch_size,), t_size, dtype=np.int64)
    seq = seq_data.astype(np.int64, copy=False).reshape(-1)
    if seq.size != batch_size:
        raise ValueError(f"{op_name} sequence_lens size mismatch.")
    seq = np.clip(seq, 0, t_size).astype(np.int64, copy=False)
    return seq


def _rec_tensor_to_real(
    model: ModelIR,
    tensors: dict[str, np.ndarray],
    name: str,
    *,
    op_name: str,
    role: str,
) -> np.ndarray:
    dtype = _tensor_dtype(model.tensors[name])
    if dtype == "float32":
        return tensors[name].astype(np.float32, copy=False)
    if dtype in ("int8", "int16"):
        scale, zero = _qparams(model, name)
        return _dequantize_int(tensors[name], scale, zero).astype(np.float32, copy=False)
    raise ValueError(f"{op_name} {role} dtype must be float32/int8/int16.")


def _rec_real_to_tensor(
    model: ModelIR,
    name: str,
    data: np.ndarray,
    *,
    op_name: str,
    role: str,
) -> np.ndarray:
    dtype = _tensor_dtype(model.tensors[name])
    out = data.astype(np.float32, copy=False)
    if dtype == "float32":
        return out
    if dtype in ("int8", "int16"):
        scale, zero = _qparams(model, name)
        return _quantize_float(out, scale, zero, dtype)
    raise ValueError(f"{op_name} {role} dtype must be float32/int8/int16.")


def _eval_rnn_node(model: ModelIR, node, tensors: dict[str, np.ndarray]) -> None:
    if len(node.inputs) < 3:
        raise ValueError("RNN expects at least 3 inputs: X, W, R.")
    if len(node.outputs) < 1 or len(node.outputs) > 2:
        raise ValueError("RNN expects 1 or 2 outputs: Y, [Y_h].")

    x_name = node.inputs[0]
    w_name = node.inputs[1]
    r_name = node.inputs[2]
    b_name = node.inputs[3] if len(node.inputs) >= 4 and node.inputs[3] else None
    seq_name = node.inputs[4] if len(node.inputs) >= 5 and node.inputs[4] else None
    h0_name = node.inputs[5] if len(node.inputs) >= 6 and node.inputs[5] else None
    y_name = node.outputs[0]
    yh_name = node.outputs[1] if len(node.outputs) >= 2 and node.outputs[1] else None

    for name, role in ((x_name, "X"), (w_name, "W"), (r_name, "R"), (y_name, "Y")):
        if _tensor_dtype(model.tensors[name]) not in ("float32", "int8", "int16"):
            raise ValueError(f"RNN {role} dtype must be float32/int8/int16.")
    if b_name is not None and _tensor_dtype(model.tensors[b_name]) not in ("float32", "int8", "int16"):
        raise ValueError("RNN B dtype must be float32/int8/int16.")
    if h0_name is not None and _tensor_dtype(model.tensors[h0_name]) not in ("float32", "int8", "int16"):
        raise ValueError("RNN initial_h dtype must be float32/int8/int16.")
    if yh_name is not None and _tensor_dtype(model.tensors[yh_name]) not in ("float32", "int8", "int16"):
        raise ValueError("RNN Y_h dtype must be float32/int8/int16.")
    if seq_name is not None and _tensor_dtype(model.tensors[seq_name]) not in ("int32", "int64"):
        raise ValueError("RNN sequence_lens dtype must be int64/int32.")

    x = _rec_tensor_to_real(model, tensors, x_name, op_name="RNN", role="X")
    w = _rec_tensor_to_real(model, tensors, w_name, op_name="RNN", role="W")
    r = _rec_tensor_to_real(model, tensors, r_name, op_name="RNN", role="R")
    b_arr = _rec_tensor_to_real(model, tensors, b_name, op_name="RNN", role="B") if b_name is not None else None
    seq_arr = tensors[seq_name] if seq_name is not None else None
    h0 = _rec_tensor_to_real(model, tensors, h0_name, op_name="RNN", role="initial_h") if h0_name is not None else None

    direction, expect_dirs = _direction_and_count("RNN", node.attrs.get("direction", "forward"))
    if x.ndim != 3 or w.ndim != 3 or r.ndim != 3:
        raise ValueError("RNN expects X/W/R rank=3.")
    t_size, b_size, i_size = (int(v) for v in x.shape)
    num_dir, h_size, i_w = (int(v) for v in w.shape)
    num_dir_r, h_r0, h_r1 = (int(v) for v in r.shape)
    if num_dir != expect_dirs or num_dir_r != expect_dirs:
        raise ValueError("RNN num_directions does not match direction attribute.")
    if i_w != i_size or h_r0 != h_size or h_r1 != h_size:
        raise ValueError("RNN W/R shape mismatch.")
    if b_arr is not None and tuple(int(v) for v in b_arr.shape) != (num_dir, 2 * h_size):
        raise ValueError("RNN bias shape mismatch.")
    if h0 is not None and tuple(int(v) for v in h0.shape) != (num_dir, b_size, h_size):
        raise ValueError("RNN initial_h shape mismatch.")

    act_specs = _rec_activation_specs(
        "RNN",
        node,
        expected_count=num_dir,
        default_cycle=["tanh"],
    )
    clip_attr = node.attrs.get("clip", None)
    clip_val = None if clip_attr is None else float(clip_attr)
    if clip_val is not None and clip_val < 0.0:
        raise ValueError("RNN clip must be non-negative.")

    seq_lens = _resolve_sequence_lens(seq_arr, b_size, t_size, "RNN")
    y = np.zeros((t_size, num_dir, b_size, h_size), dtype=np.float32)
    h_state = np.zeros((num_dir, b_size, h_size), dtype=np.float32)
    if h0 is not None:
        h_state[...] = h0
    for d_i in range(num_dir):
        dir_rev = (direction == "reverse") or (direction == "bidirectional" and d_i == 1)
        for b_i in range(b_size):
            seq_len = int(seq_lens[b_i])
            for step_i in range(seq_len):
                t_src = (seq_len - 1 - step_i) if dir_rev else step_i
                h_prev = h_state[d_i, b_i, :]
                pre = np.matmul(w[d_i, :, :], x[t_src, b_i, :]) + np.matmul(r[d_i, :, :], h_prev)
                if b_arr is not None:
                    pre = pre + b_arr[d_i, :h_size] + b_arr[d_i, h_size:]
                pre = _rec_apply_clip(pre.astype(np.float32, copy=False), clip_val)
                h_new = _rec_apply_activation(pre, act_specs[d_i]).astype(np.float32, copy=False)
                h_state[d_i, b_i, :] = h_new
                y[t_src, d_i, b_i, :] = h_new

    tensors[y_name] = _rec_real_to_tensor(model, y_name, y, op_name="RNN", role="Y")
    if yh_name is not None:
        tensors[yh_name] = _rec_real_to_tensor(
            model,
            yh_name,
            h_state.astype(np.float32, copy=False),
            op_name="RNN",
            role="Y_h",
        )


def _eval_gru_node(model: ModelIR, node, tensors: dict[str, np.ndarray]) -> None:
    if len(node.inputs) < 3:
        raise ValueError("GRU expects at least 3 inputs: X, W, R.")
    if len(node.outputs) < 1 or len(node.outputs) > 2:
        raise ValueError("GRU expects 1 or 2 outputs: Y, [Y_h].")

    x_name = node.inputs[0]
    w_name = node.inputs[1]
    r_name = node.inputs[2]
    b_name = node.inputs[3] if len(node.inputs) >= 4 and node.inputs[3] else None
    seq_name = node.inputs[4] if len(node.inputs) >= 5 and node.inputs[4] else None
    h0_name = node.inputs[5] if len(node.inputs) >= 6 and node.inputs[5] else None
    y_name = node.outputs[0]
    yh_name = node.outputs[1] if len(node.outputs) >= 2 and node.outputs[1] else None

    for name, role in ((x_name, "X"), (w_name, "W"), (r_name, "R"), (y_name, "Y")):
        if _tensor_dtype(model.tensors[name]) not in ("float32", "int8", "int16"):
            raise ValueError(f"GRU {role} dtype must be float32/int8/int16.")
    if b_name is not None and _tensor_dtype(model.tensors[b_name]) not in ("float32", "int8", "int16"):
        raise ValueError("GRU B dtype must be float32/int8/int16.")
    if h0_name is not None and _tensor_dtype(model.tensors[h0_name]) not in ("float32", "int8", "int16"):
        raise ValueError("GRU initial_h dtype must be float32/int8/int16.")
    if yh_name is not None and _tensor_dtype(model.tensors[yh_name]) not in ("float32", "int8", "int16"):
        raise ValueError("GRU Y_h dtype must be float32/int8/int16.")
    if seq_name is not None and _tensor_dtype(model.tensors[seq_name]) not in ("int32", "int64"):
        raise ValueError("GRU sequence_lens dtype must be int64/int32.")

    x = _rec_tensor_to_real(model, tensors, x_name, op_name="GRU", role="X")
    w = _rec_tensor_to_real(model, tensors, w_name, op_name="GRU", role="W")
    r = _rec_tensor_to_real(model, tensors, r_name, op_name="GRU", role="R")
    b_arr = _rec_tensor_to_real(model, tensors, b_name, op_name="GRU", role="B") if b_name is not None else None
    seq_arr = tensors[seq_name] if seq_name is not None else None
    h0 = _rec_tensor_to_real(model, tensors, h0_name, op_name="GRU", role="initial_h") if h0_name is not None else None

    direction, expect_dirs = _direction_and_count("GRU", node.attrs.get("direction", "forward"))
    linear_before_reset = int(node.attrs.get("linear_before_reset", 0))
    if linear_before_reset not in (0, 1):
        raise ValueError("GRU linear_before_reset must be 0 or 1.")
    if x.ndim != 3 or w.ndim != 3 or r.ndim != 3:
        raise ValueError("GRU expects X/W/R rank=3.")
    t_size, b_size, i_size = (int(v) for v in x.shape)
    num_dir, wh, i_w = (int(v) for v in w.shape)
    num_dir_r, rh0, rh1 = (int(v) for v in r.shape)
    if num_dir != expect_dirs or num_dir_r != expect_dirs:
        raise ValueError("GRU num_directions does not match direction attribute.")
    if wh % 3 != 0 or rh0 % 3 != 0:
        raise ValueError("GRU W/R gate dimension must be 3*hidden_size.")
    h_size = wh // 3
    if i_w != i_size or rh0 != 3 * h_size or rh1 != h_size:
        raise ValueError("GRU W/R shape mismatch.")
    if b_arr is not None and tuple(int(v) for v in b_arr.shape) != (num_dir, 6 * h_size):
        raise ValueError("GRU bias shape mismatch.")
    if h0 is not None and tuple(int(v) for v in h0.shape) != (num_dir, b_size, h_size):
        raise ValueError("GRU initial_h shape mismatch.")

    act_specs = _rec_activation_specs(
        "GRU",
        node,
        expected_count=2 * num_dir,
        default_cycle=["sigmoid", "tanh"],
    )
    f_specs = [act_specs[2 * d] for d in range(num_dir)]
    g_specs = [act_specs[2 * d + 1] for d in range(num_dir)]
    clip_attr = node.attrs.get("clip", None)
    clip_val = None if clip_attr is None else float(clip_attr)
    if clip_val is not None and clip_val < 0.0:
        raise ValueError("GRU clip must be non-negative.")

    seq_lens = _resolve_sequence_lens(seq_arr, b_size, t_size, "GRU")
    y = np.zeros((t_size, num_dir, b_size, h_size), dtype=np.float32)
    h_state = np.zeros((num_dir, b_size, h_size), dtype=np.float32)
    if h0 is not None:
        h_state[...] = h0

    for d_i in range(num_dir):
        dir_rev = (direction == "reverse") or (direction == "bidirectional" and d_i == 1)
        wb = rb = None
        if b_arr is not None:
            wb = b_arr[d_i, : 3 * h_size]
            rb = b_arr[d_i, 3 * h_size :]
        for b_i in range(b_size):
            seq_len = int(seq_lens[b_i])
            for step_i in range(seq_len):
                t_src = (seq_len - 1 - step_i) if dir_rev else step_i
                h_prev = h_state[d_i, b_i, :]
                x_proj = np.matmul(w[d_i, :, :], x[t_src, b_i, :])
                h_proj = np.matmul(r[d_i, :, :], h_prev)
                x_z, x_r, x_h = np.split(x_proj, 3)
                h_z, h_r, _ = np.split(h_proj, 3)
                if wb is not None and rb is not None:
                    wb_z, wb_r, wb_h = np.split(wb, 3)
                    rb_z, rb_r, rb_h = np.split(rb, 3)
                else:
                    wb_z = wb_r = wb_h = rb_z = rb_r = rb_h = 0.0
                z_pre = x_z + h_z + wb_z + rb_z
                r_pre = x_r + h_r + wb_r + rb_r
                z_pre = _rec_apply_clip(z_pre.astype(np.float32, copy=False), clip_val)
                r_pre = _rec_apply_clip(r_pre.astype(np.float32, copy=False), clip_val)
                z = _rec_apply_activation(z_pre, f_specs[d_i]).astype(np.float32, copy=False)
                rr = _rec_apply_activation(r_pre, f_specs[d_i]).astype(np.float32, copy=False)
                rec_h_mat = r[d_i, 2 * h_size : 3 * h_size, :]
                if linear_before_reset == 0:
                    h_pre = x_h + np.matmul(rec_h_mat, rr * h_prev) + wb_h + rb_h
                else:
                    h_pre = x_h + rr * (np.matmul(rec_h_mat, h_prev) + rb_h) + wb_h
                h_pre = _rec_apply_clip(h_pre.astype(np.float32, copy=False), clip_val)
                h_tilde = _rec_apply_activation(h_pre, g_specs[d_i]).astype(np.float32, copy=False)
                h_new = (1.0 - z) * h_tilde + z * h_prev
                h_state[d_i, b_i, :] = h_new.astype(np.float32, copy=False)
                y[t_src, d_i, b_i, :] = h_state[d_i, b_i, :]

    tensors[y_name] = _rec_real_to_tensor(model, y_name, y, op_name="GRU", role="Y")
    if yh_name is not None:
        tensors[yh_name] = _rec_real_to_tensor(
            model,
            yh_name,
            h_state.astype(np.float32, copy=False),
            op_name="GRU",
            role="Y_h",
        )


def _eval_lstm_node(model: ModelIR, node, tensors: dict[str, np.ndarray]) -> None:
    if len(node.inputs) < 3:
        raise ValueError("LSTM expects at least 3 inputs: X, W, R.")
    if len(node.outputs) < 1 or len(node.outputs) > 3:
        raise ValueError("LSTM expects 1..3 outputs: Y, [Y_h], [Y_c].")

    x_name = node.inputs[0]
    w_name = node.inputs[1]
    r_name = node.inputs[2]
    b_name = node.inputs[3] if len(node.inputs) >= 4 and node.inputs[3] else None
    seq_name = node.inputs[4] if len(node.inputs) >= 5 and node.inputs[4] else None
    h0_name = node.inputs[5] if len(node.inputs) >= 6 and node.inputs[5] else None
    c0_name = node.inputs[6] if len(node.inputs) >= 7 and node.inputs[6] else None
    p_name = node.inputs[7] if len(node.inputs) >= 8 and node.inputs[7] else None
    y_name = node.outputs[0]
    yh_name = node.outputs[1] if len(node.outputs) >= 2 and node.outputs[1] else None
    yc_name = node.outputs[2] if len(node.outputs) >= 3 and node.outputs[2] else None

    for name, role in ((x_name, "X"), (w_name, "W"), (r_name, "R"), (y_name, "Y")):
        if _tensor_dtype(model.tensors[name]) not in ("float32", "int8", "int16"):
            raise ValueError(f"LSTM {role} dtype must be float32/int8/int16.")
    if b_name is not None and _tensor_dtype(model.tensors[b_name]) not in ("float32", "int8", "int16"):
        raise ValueError("LSTM B dtype must be float32/int8/int16.")
    if h0_name is not None and _tensor_dtype(model.tensors[h0_name]) not in ("float32", "int8", "int16"):
        raise ValueError("LSTM initial_h dtype must be float32/int8/int16.")
    if c0_name is not None and _tensor_dtype(model.tensors[c0_name]) not in ("float32", "int8", "int16"):
        raise ValueError("LSTM initial_c dtype must be float32/int8/int16.")
    if yh_name is not None and _tensor_dtype(model.tensors[yh_name]) not in ("float32", "int8", "int16"):
        raise ValueError("LSTM Y_h dtype must be float32/int8/int16.")
    if yc_name is not None and _tensor_dtype(model.tensors[yc_name]) not in ("float32", "int8", "int16"):
        raise ValueError("LSTM Y_c dtype must be float32/int8/int16.")
    if seq_name is not None and _tensor_dtype(model.tensors[seq_name]) not in ("int32", "int64"):
        raise ValueError("LSTM sequence_lens dtype must be int64/int32.")
    if p_name is not None and _tensor_dtype(model.tensors[p_name]) not in ("float32", "int8", "int16"):
        raise ValueError("LSTM peephole dtype must be float32/int8/int16.")

    x = _rec_tensor_to_real(model, tensors, x_name, op_name="LSTM", role="X")
    w = _rec_tensor_to_real(model, tensors, w_name, op_name="LSTM", role="W")
    r = _rec_tensor_to_real(model, tensors, r_name, op_name="LSTM", role="R")
    b_arr = _rec_tensor_to_real(model, tensors, b_name, op_name="LSTM", role="B") if b_name is not None else None
    seq_arr = tensors[seq_name] if seq_name is not None else None
    h0 = _rec_tensor_to_real(model, tensors, h0_name, op_name="LSTM", role="initial_h") if h0_name is not None else None
    c0 = _rec_tensor_to_real(model, tensors, c0_name, op_name="LSTM", role="initial_c") if c0_name is not None else None
    p_arr = _rec_tensor_to_real(model, tensors, p_name, op_name="LSTM", role="P") if p_name is not None else None

    direction, expect_dirs = _direction_and_count("LSTM", node.attrs.get("direction", "forward"))
    input_forget = int(node.attrs.get("input_forget", 0))
    if input_forget not in (0, 1):
        raise ValueError("LSTM input_forget must be 0 or 1.")
    if x.ndim != 3 or w.ndim != 3 or r.ndim != 3:
        raise ValueError("LSTM expects X/W/R rank=3.")
    t_size, b_size, i_size = (int(v) for v in x.shape)
    num_dir, wh, i_w = (int(v) for v in w.shape)
    num_dir_r, rh0, rh1 = (int(v) for v in r.shape)
    if num_dir != expect_dirs or num_dir_r != expect_dirs:
        raise ValueError("LSTM num_directions does not match direction attribute.")
    if wh % 4 != 0 or rh0 % 4 != 0:
        raise ValueError("LSTM W/R gate dimension must be 4*hidden_size.")
    h_size = wh // 4
    if i_w != i_size or rh0 != 4 * h_size or rh1 != h_size:
        raise ValueError("LSTM W/R shape mismatch.")
    if b_arr is not None and tuple(int(v) for v in b_arr.shape) != (num_dir, 8 * h_size):
        raise ValueError("LSTM bias shape mismatch.")
    if h0 is not None and tuple(int(v) for v in h0.shape) != (num_dir, b_size, h_size):
        raise ValueError("LSTM initial_h shape mismatch.")
    if c0 is not None and tuple(int(v) for v in c0.shape) != (num_dir, b_size, h_size):
        raise ValueError("LSTM initial_c shape mismatch.")
    if p_arr is not None and tuple(int(v) for v in p_arr.shape) != (num_dir, 3 * h_size):
        raise ValueError("LSTM peephole shape mismatch.")

    act_specs = _rec_activation_specs(
        "LSTM",
        node,
        expected_count=3 * num_dir,
        default_cycle=["sigmoid", "tanh", "tanh"],
    )
    f_specs = [act_specs[3 * d] for d in range(num_dir)]
    g_specs = [act_specs[3 * d + 1] for d in range(num_dir)]
    h_specs = [act_specs[3 * d + 2] for d in range(num_dir)]
    clip_attr = node.attrs.get("clip", None)
    clip_val = None if clip_attr is None else float(clip_attr)
    if clip_val is not None and clip_val < 0.0:
        raise ValueError("LSTM clip must be non-negative.")

    seq_lens = _resolve_sequence_lens(seq_arr, b_size, t_size, "LSTM")
    y = np.zeros((t_size, num_dir, b_size, h_size), dtype=np.float32)
    h_state = np.zeros((num_dir, b_size, h_size), dtype=np.float32)
    c_state = np.zeros((num_dir, b_size, h_size), dtype=np.float32)
    if h0 is not None:
        h_state[...] = h0
    if c0 is not None:
        c_state[...] = c0

    for d_i in range(num_dir):
        dir_rev = (direction == "reverse") or (direction == "bidirectional" and d_i == 1)
        wb = rb = None
        if b_arr is not None:
            wb = b_arr[d_i, : 4 * h_size]
            rb = b_arr[d_i, 4 * h_size :]
        p_i = p_o = p_f = None
        if p_arr is not None:
            p_i = p_arr[d_i, :h_size]
            p_o = p_arr[d_i, h_size : 2 * h_size]
            p_f = p_arr[d_i, 2 * h_size :]
        for b_i in range(b_size):
            seq_len = int(seq_lens[b_i])
            for step_i in range(seq_len):
                t_src = (seq_len - 1 - step_i) if dir_rev else step_i
                h_prev = h_state[d_i, b_i, :]
                c_prev = c_state[d_i, b_i, :]
                x_proj = np.matmul(w[d_i, :, :], x[t_src, b_i, :])
                h_proj = np.matmul(r[d_i, :, :], h_prev)
                x_i, x_o, x_f, x_g = np.split(x_proj, 4)
                h_i, h_o, h_f, h_g = np.split(h_proj, 4)
                if wb is not None and rb is not None:
                    wb_i, wb_o, wb_f, wb_g = np.split(wb, 4)
                    rb_i, rb_o, rb_f, rb_g = np.split(rb, 4)
                else:
                    wb_i = wb_o = wb_f = wb_g = rb_i = rb_o = rb_f = rb_g = 0.0

                pre_i = x_i + h_i + wb_i + rb_i
                pre_o = x_o + h_o + wb_o + rb_o
                pre_f = x_f + h_f + wb_f + rb_f
                pre_g = x_g + h_g + wb_g + rb_g
                if p_i is not None and p_f is not None:
                    pre_i = pre_i + p_i * c_prev
                    pre_f = pre_f + p_f * c_prev
                pre_i = _rec_apply_clip(pre_i.astype(np.float32, copy=False), clip_val)
                pre_f = _rec_apply_clip(pre_f.astype(np.float32, copy=False), clip_val)
                pre_g = _rec_apply_clip(pre_g.astype(np.float32, copy=False), clip_val)
                i_gate = _rec_apply_activation(pre_i, f_specs[d_i]).astype(np.float32, copy=False)
                if input_forget == 1:
                    f_gate = (1.0 - i_gate).astype(np.float32, copy=False)
                else:
                    f_gate = _rec_apply_activation(pre_f, f_specs[d_i]).astype(np.float32, copy=False)
                g_gate = _rec_apply_activation(pre_g, g_specs[d_i]).astype(np.float32, copy=False)
                c_new = f_gate * c_prev + i_gate * g_gate
                if p_o is not None:
                    pre_o = pre_o + p_o * c_new
                pre_o = _rec_apply_clip(pre_o.astype(np.float32, copy=False), clip_val)
                o_gate = _rec_apply_activation(pre_o, f_specs[d_i]).astype(np.float32, copy=False)
                c_act = _rec_apply_clip(c_new.astype(np.float32, copy=False), clip_val)
                h_new = o_gate * _rec_apply_activation(c_act, h_specs[d_i]).astype(np.float32, copy=False)
                h_state[d_i, b_i, :] = h_new.astype(np.float32, copy=False)
                c_state[d_i, b_i, :] = c_new.astype(np.float32, copy=False)
                y[t_src, d_i, b_i, :] = h_state[d_i, b_i, :]

    tensors[y_name] = _rec_real_to_tensor(model, y_name, y, op_name="LSTM", role="Y")
    if yh_name is not None:
        tensors[yh_name] = _rec_real_to_tensor(
            model,
            yh_name,
            h_state.astype(np.float32, copy=False),
            op_name="LSTM",
            role="Y_h",
        )
    if yc_name is not None:
        tensors[yc_name] = _rec_real_to_tensor(
            model,
            yc_name,
            c_state.astype(np.float32, copy=False),
            op_name="LSTM",
            role="Y_c",
        )
