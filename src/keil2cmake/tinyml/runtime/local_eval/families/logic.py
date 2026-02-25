# -*- coding: utf-8 -*-

from __future__ import annotations

import math

import numpy as np
from onnx import TensorProto

from ....converter.ir import ModelIR
from ..recurrent import _eval_gru_node, _eval_lstm_node, _eval_rnn_node
from ..utils import (
    _attr_scalar,
    _const_from_constant_attrs,
    _const_ints,
    _const_scalar,
    _conv_out_dim,
    _dequantize_int,
    _qparams,
    _quantize_float,
    _reshape_like_onnx,
    _tensor_dtype,
)

def handle_logic_family(
    model: ModelIR,
    node,
    tensors: dict[str, np.ndarray],
    ins: list[np.ndarray | None],
    out_name: str,
    out_dtype: str,
) -> bool:
    op = node.op_type
    if op in ("Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"):
        a, b = ins[0], ins[1]
        if op == "Equal":
            out = np.equal(a, b)
        elif op == "Greater":
            out = np.greater(a, b)
        elif op == "Less":
            out = np.less(a, b)
        elif op == "GreaterOrEqual":
            out = np.greater_equal(a, b)
        else:
            out = np.less_equal(a, b)
        tensors[out_name] = out.astype(np.bool_)
        return True
    

    if op in ("And", "Or", "Xor"):
        a = ins[0] != 0
        b = ins[1] != 0
        if op == "And":
            out = np.logical_and(a, b)
        elif op == "Or":
            out = np.logical_or(a, b)
        else:
            out = np.logical_xor(a, b)
        tensors[out_name] = out.astype(np.bool_)
        return True
    

    if op == "Not":
        tensors[out_name] = np.logical_not(ins[0] != 0).astype(np.bool_)
        return True
    

    if op == "IsInf":
        a = ins[0].astype(np.float32)
        detect_negative = int(node.attrs.get("detect_negative", 1))
        detect_positive = int(node.attrs.get("detect_positive", 1))
        mask = np.isinf(a)
        if detect_negative == 0:
            mask = np.logical_and(mask, np.logical_not(a < 0.0))
        if detect_positive == 0:
            mask = np.logical_and(mask, np.logical_not(a > 0.0))
        tensors[out_name] = mask.astype(np.bool_)
        return True
    

    if op == "IsNaN":
        a = ins[0].astype(np.float32)
        tensors[out_name] = np.isnan(a).astype(np.bool_)
        return True
    

    if op == "BitShift":
        if out_dtype not in ("int8", "int16", "int32", "int64"):
            raise ValueError("BitShift output dtype is unsupported.")
        a, b = ins[0], ins[1]
        direction = node.attrs.get("direction", "RIGHT")
        if isinstance(direction, bytes):
            direction = direction.decode("utf-8", errors="ignore")
        direction = str(direction).upper()
        if direction not in ("LEFT", "RIGHT"):
            raise ValueError("BitShift direction must be LEFT/RIGHT.")
        if out_dtype == "int8":
            sh = np.clip(b.astype(np.int64), 0, 7).astype(np.uint8)
            au = a.astype(np.uint8)
            ru = np.left_shift(au, sh) if direction == "LEFT" else np.right_shift(au, sh)
            tensors[out_name] = ru.view(np.int8)
        elif out_dtype == "int16":
            sh = np.clip(b.astype(np.int64), 0, 15).astype(np.uint16)
            au = a.astype(np.uint16)
            ru = np.left_shift(au, sh) if direction == "LEFT" else np.right_shift(au, sh)
            tensors[out_name] = ru.view(np.int16)
        elif out_dtype == "int32":
            sh = np.clip(b.astype(np.int64), 0, 31).astype(np.uint32)
            au = a.astype(np.uint32)
            ru = np.left_shift(au, sh) if direction == "LEFT" else np.right_shift(au, sh)
            tensors[out_name] = ru.view(np.int32)
        else:
            sh = np.clip(b.astype(np.int64), 0, 63).astype(np.uint64)
            au = a.astype(np.uint64)
            ru = np.left_shift(au, sh) if direction == "LEFT" else np.right_shift(au, sh)
            tensors[out_name] = ru.view(np.int64)
        return True
    

    return False
