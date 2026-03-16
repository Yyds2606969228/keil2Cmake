# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any

import numpy as np
from onnx import numpy_helper

from ...converter.ir import ModelIR

def _round_away_from_zero(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _quantize_float(x: np.ndarray, scale: float, zero: int, dtype: str) -> np.ndarray:
    q = _round_away_from_zero(x / scale) + float(zero)
    if dtype == "uint8":
        q = np.clip(q, 0, 255).astype(np.uint8)
    elif dtype == "int8":
        q = np.clip(q, -128, 127).astype(np.int8)
    elif dtype == "int16":
        q = np.clip(q, -32768, 32767).astype(np.int16)
    else:
        raise ValueError("Unsupported quant dtype.")
    return q


def _dequantize_int(x: np.ndarray, scale: float, zero: int) -> np.ndarray:
    return (x.astype(np.float32) - float(zero)) * float(scale)


def _tensor_dtype(tensor) -> str:
    if tensor.dtype == "float32":
        return "float32"
    if tensor.dtype == "bool":
        return "bool"
    if tensor.dtype == "uint8":
        return "uint8"
    if tensor.dtype == "int8":
        return "int8"
    if tensor.dtype == "int16":
        return "int16"
    if tensor.dtype == "int32":
        return "int32"
    if tensor.dtype == "int64":
        return "int64"
    return "unknown"


def _qparams(model: ModelIR, name: str) -> tuple[float, int]:
    tensor = model.tensors.get(name)
    if tensor is None or tensor.qscale is None or tensor.qzero is None:
        raise ValueError(f"Missing quantization params for '{name}'.")
    return float(tensor.qscale), int(tensor.qzero)


def _const_scalar(tensors: dict[str, np.ndarray], name: str) -> float:
    if name not in tensors:
        raise ValueError(f"Missing const tensor '{name}'.")
    data = tensors[name]
    if data.size == 0:
        raise ValueError(f"Const tensor '{name}' has no data.")
    return float(data.reshape(-1)[0])


def _const_ints(tensors: dict[str, np.ndarray], name: str) -> list[int]:
    if name not in tensors:
        raise ValueError(f"Missing const tensor '{name}'.")
    data = tensors[name]
    return [int(v) for v in data.reshape(-1).tolist()]


def _attr_scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (int, float, bool)):
        return float(value)
    if isinstance(value, (list, tuple)):
        if not value:
            return float(default)
        return float(value[0])
    arr = numpy_helper.to_array(value)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def _decode_attr_str(value: Any, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").lower()
    return str(value).lower()



def _conv_out_dim(in_dim: int, kernel: int, stride: int, pad0: int, pad1: int, dilation: int) -> int:
    return (in_dim + pad0 + pad1 - dilation * (kernel - 1) - 1) // stride + 1

def _const_from_constant_attrs(attrs: dict[str, Any]) -> np.ndarray:
    if "value" in attrs:
        arr = numpy_helper.to_array(attrs["value"])
        if arr.dtype == np.bool_:
            return arr.astype(np.bool_)
        if arr.dtype.kind in ("f", "c"):
            return arr.astype(np.float32)
        if arr.dtype == np.int8:
            return arr.astype(np.int8)
        if arr.dtype == np.int16:
            return arr.astype(np.int16)
        if arr.dtype == np.int32:
            return arr.astype(np.int32)
        return arr.astype(np.int64)
    if "value_float" in attrs:
        return np.array(float(attrs["value_float"]), dtype=np.float32)
    if "value_int" in attrs:
        return np.array(int(attrs["value_int"]), dtype=np.int64)
    if "value_floats" in attrs:
        return np.array([float(v) for v in attrs["value_floats"]], dtype=np.float32)
    if "value_ints" in attrs:
        return np.array([int(v) for v in attrs["value_ints"]], dtype=np.int64)
    raise ValueError("Constant requires value/value_float/value_int/value_floats/value_ints.")


def _reshape_like_onnx(data: np.ndarray, shape_vals: list[int]) -> np.ndarray:
    in_shape = list(data.shape)
    out: list[int] = []
    unknown = None
    known_product = 1
    for idx, dim in enumerate(shape_vals):
        dim = int(dim)
        if dim == 0:
            if idx >= len(in_shape):
                raise ValueError("Reshape 0 uses out-of-range input dim.")
            dim = in_shape[idx]
        if dim == -1:
            if unknown is not None:
                raise ValueError("Reshape has multiple -1.")
            unknown = idx
            out.append(-1)
        else:
            if dim <= 0:
                raise ValueError("Reshape has invalid dim.")
            out.append(dim)
            known_product *= dim
    if unknown is not None:
        in_size = int(np.prod(in_shape)) if in_shape else 1
        if known_product == 0 or in_size % known_product != 0:
            raise ValueError("Reshape size mismatch.")
        out[unknown] = int(in_size / known_product)
    return data.reshape(out)
