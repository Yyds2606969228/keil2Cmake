# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

from ..converter.ir import ModelIR
from .c_runner import run_generated_c_model

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None

try:
    from onnx.reference import ReferenceEvaluator
except Exception:  # pragma: no cover - optional fallback
    ReferenceEvaluator = None


@dataclass
class ValidationResult:
    status: str
    reason: str = ""
    max_abs: float = 0.0
    max_rel: float = 0.0
    engine: str = ""


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


def _eval_model(model: ModelIR, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    for name, tensor in model.tensors.items():
        if tensor.data is None:
            continue
        dtype = tensor.dtype
        if dtype == "float32":
            arr = np.array(tensor.data, dtype=np.float32)
        elif dtype == "bool":
            arr = np.array(tensor.data, dtype=np.bool_)
        elif dtype == "uint8":
            arr = np.array(tensor.data, dtype=np.uint8)
        elif dtype == "int8":
            arr = np.array(tensor.data, dtype=np.int8)
        elif dtype == "int16":
            arr = np.array(tensor.data, dtype=np.int16)
        elif dtype == "int32":
            arr = np.array(tensor.data, dtype=np.int32)
        elif dtype == "int64":
            arr = np.array(tensor.data, dtype=np.int64)
        else:
            raise ValueError("Unsupported const dtype.")
        shape = list(tensor.shape)
        if len(shape) == 0:
            arr = arr.reshape(())
        else:
            arr = arr.reshape(shape)
        tensors[name] = arr
    for name, arr in inputs.items():
        tensors[name] = arr

    for node in model.nodes:
        op = node.op_type
        ins = [tensors[name] for name in node.inputs]
        out_name = node.outputs[0]
        out_dtype = model.tensors[out_name].dtype

        if op == "DynamicQuantizeLinear":
            if len(ins) < 1 or len(node.outputs) != 3:
                raise ValueError("DynamicQuantizeLinear expects 1 input and 3 outputs.")
            x = ins[0].astype(np.float32, copy=False)
            if x.size == 0:
                raise ValueError("DynamicQuantizeLinear input must be non-empty.")
            x_min = float(np.min(x))
            x_max = float(np.max(x))
            scale = (x_max - x_min) / 255.0
            if scale <= 0.0:
                scale = 1.0
            zero = int(np.round(-x_min / scale))
            zero = max(0, min(255, zero))
            y_name, y_scale_name, y_zero_name = node.outputs
            tensors[y_name] = _quantize_float(x, scale, zero, "uint8")
            tensors[y_scale_name] = np.array(scale, dtype=np.float32).reshape(())
            tensors[y_zero_name] = np.array(zero, dtype=np.uint8).reshape(())
            continue

        if op == "QuantizeLinear":
            scale = _const_scalar(tensors, node.inputs[1])
            zero = 0
            if len(node.inputs) >= 3:
                zero_vals = _const_ints(tensors, node.inputs[2])
                if len(zero_vals) != 1:
                    raise ValueError("QuantizeLinear supports scalar zero_point only.")
                zero = int(zero_vals[0])
            out = _quantize_float(ins[0].astype(np.float32), float(scale), int(zero), out_dtype)
            tensors[out_name] = out
            continue

        if op == "DequantizeLinear":
            scale = _const_scalar(tensors, node.inputs[1])
            zero = 0
            if len(node.inputs) >= 3:
                zero_vals = _const_ints(tensors, node.inputs[2])
                if len(zero_vals) != 1:
                    raise ValueError("DequantizeLinear supports scalar zero_point only.")
                zero = int(zero_vals[0])
            tensors[out_name] = _dequantize_int(ins[0], float(scale), int(zero))
            continue

        if op == "Einsum":
            if len(ins) != 2:
                raise ValueError("Einsum expects 2 inputs.")
            eq = node.attrs.get("equation", "")
            if isinstance(eq, bytes):
                eq = eq.decode("utf-8", errors="ignore")
            eq = str(eq).replace(" ", "")
            if out_dtype in ("int8", "int16"):
                in0_dtype = model.tensors[node.inputs[0]].dtype
                in1_dtype = model.tensors[node.inputs[1]].dtype
                if in0_dtype != out_dtype or in1_dtype != out_dtype:
                    raise ValueError("Quantized Einsum requires matching input/output dtypes.")
                sa, za = _qparams(model, node.inputs[0])
                sb, zb = _qparams(model, node.inputs[1])
                so, zo = _qparams(model, out_name)
                ra = _dequantize_int(ins[0], sa, za)
                rb = _dequantize_int(ins[1], sb, zb)
                ro = np.einsum(eq, ra, rb).astype(np.float32)
                tensors[out_name] = _quantize_float(ro, so, zo, out_dtype)
            elif out_dtype == "float32":
                tensors[out_name] = np.einsum(eq, ins[0], ins[1]).astype(np.float32)
            else:
                raise ValueError("Einsum output dtype must be float32/int8/int16.")
            continue

        if op in ("Add", "Sub", "Mul", "Div", "Max", "Min", "Pow"):
            a, b = ins[0], ins[1]
            if out_dtype in ("int8", "int16"):
                if a.shape != b.shape:
                    raise ValueError(f"Quantized {op} requires equal shapes.")
                sa, za = _qparams(model, node.inputs[0])
                sb, zb = _qparams(model, node.inputs[1])
                so, zo = _qparams(model, out_name)
                ra = _dequantize_int(a, sa, za)
                rb = _dequantize_int(b, sb, zb)
                if op == "Add":
                    ro = ra + rb
                elif op == "Sub":
                    ro = ra - rb
                elif op == "Mul":
                    ro = ra * rb
                elif op == "Div":
                    ro = ra / rb
                elif op == "Max":
                    ro = np.maximum(ra, rb)
                elif op == "Min":
                    ro = np.minimum(ra, rb)
                else:
                    ro = np.power(ra, rb)
                tensors[out_name] = _quantize_float(ro, so, zo, out_dtype)
            else:
                if op == "Add":
                    out = a + b
                elif op == "Sub":
                    out = a - b
                elif op == "Mul":
                    out = a * b
                elif op == "Div":
                    out = a / b
                elif op == "Max":
                    out = np.maximum(a, b)
                elif op == "Min":
                    out = np.minimum(a, b)
                else:
                    out = np.power(a, b)
                tensors[out_name] = out
            continue

        if op in ("Sum", "Mean"):
            if len(ins) < 1:
                raise ValueError(f"{op} expects at least one input.")
            if out_dtype in ("int8", "int16"):
                base_shape = ins[0].shape
                for arr in ins:
                    if arr.shape != base_shape:
                        raise ValueError(f"Quantized {op} requires equal shapes.")
                so, zo = _qparams(model, out_name)
                ro = np.zeros(base_shape, dtype=np.float32)
                for idx, arr in enumerate(ins):
                    si, zi = _qparams(model, node.inputs[idx])
                    ro += _dequantize_int(arr, si, zi)
                if op == "Mean":
                    ro = ro / float(len(ins))
                tensors[out_name] = _quantize_float(ro, so, zo, out_dtype)
            else:
                out = ins[0]
                for arr in ins[1:]:
                    out = out + arr
                if op == "Mean":
                    out = out / float(len(ins))
                tensors[out_name] = out
            continue

        if op == "Split":
            data = ins[0]
            rank = data.ndim
            axis = int(node.attrs.get("axis", 0))
            if axis < 0:
                axis += rank
            if axis < 0 or axis >= rank:
                raise ValueError("Split axis out of range.")
            if len(node.outputs) <= 0:
                raise ValueError("Split expects at least one output.")
            split_vals = None
            if len(node.inputs) >= 2 and node.inputs[1]:
                split_vals = _const_ints(tensors, node.inputs[1])
            if split_vals is None:
                split_attr = node.attrs.get("split")
                if split_attr is not None:
                    split_vals = [int(v) for v in split_attr]
            if split_vals is None:
                dim = int(data.shape[axis])
                if dim % len(node.outputs) != 0:
                    raise ValueError("Split cannot divide axis evenly.")
                each = dim // len(node.outputs)
                split_vals = [each for _ in node.outputs]
            if len(split_vals) != len(node.outputs):
                raise ValueError("Split output count mismatch.")
            if sum(int(v) for v in split_vals) != int(data.shape[axis]):
                raise ValueError("Split sizes do not match axis dimension.")
            start = 0
            for out_tensor_name, part in zip(node.outputs, split_vals):
                part_i = int(part)
                sl = [slice(None)] * rank
                sl[axis] = slice(start, start + part_i)
                tensors[out_tensor_name] = data[tuple(sl)].copy()
                start += part_i
            continue

        if op == "Shape":
            shape_arr = np.array(ins[0].shape, dtype=np.int64)
            if out_dtype == "int32":
                tensors[out_name] = shape_arr.astype(np.int32)
            elif out_dtype == "float32":
                tensors[out_name] = shape_arr.astype(np.float32)
            else:
                tensors[out_name] = shape_arr.astype(np.int64)
            continue

        if op == "Size":
            size_v = int(np.prod(ins[0].shape, dtype=np.int64))
            if out_dtype == "int32":
                tensors[out_name] = np.array(size_v, dtype=np.int32)
            elif out_dtype == "float32":
                tensors[out_name] = np.array(float(size_v), dtype=np.float32)
            else:
                tensors[out_name] = np.array(size_v, dtype=np.int64)
            continue

        if op == "Constant":
            const_val = _const_from_constant_attrs(node.attrs)
            expect_dtype = _tensor_dtype(model.tensors[out_name])
            if expect_dtype == "float32":
                out = const_val.astype(np.float32)
            elif expect_dtype == "bool":
                out = const_val.astype(np.bool_)
            elif expect_dtype == "int8":
                out = const_val.astype(np.int8)
            elif expect_dtype == "int16":
                out = const_val.astype(np.int16)
            elif expect_dtype == "int32":
                out = const_val.astype(np.int32)
            elif expect_dtype == "int64":
                out = const_val.astype(np.int64)
            else:
                raise ValueError("Constant output dtype unsupported.")
            expected_shape = tuple(int(v) for v in model.tensors[out_name].shape)
            if tuple(out.shape) != expected_shape:
                if expected_shape == () and out.size == 1:
                    out = out.reshape(())
                else:
                    raise ValueError("Constant output shape mismatch.")
            tensors[out_name] = out
            continue

        if op == "ConstantOfShape":
            shape_vals = ins[0].astype(np.int64).reshape(-1)
            out_shape = tuple(int(v) for v in shape_vals.tolist())
            if any(v < 0 for v in out_shape):
                raise ValueError("ConstantOfShape has negative dimension.")
            fill_v = _attr_scalar(node.attrs.get("value"), 0.0)
            if out_dtype == "float32":
                tensors[out_name] = np.full(out_shape, fill_v, dtype=np.float32)
            elif out_dtype == "int8":
                tensors[out_name] = np.full(out_shape, int(round(fill_v)), dtype=np.int8)
            elif out_dtype == "int16":
                tensors[out_name] = np.full(out_shape, int(round(fill_v)), dtype=np.int16)
            elif out_dtype == "int32":
                tensors[out_name] = np.full(out_shape, int(round(fill_v)), dtype=np.int32)
            elif out_dtype == "int64":
                tensors[out_name] = np.full(out_shape, int(round(fill_v)), dtype=np.int64)
            elif out_dtype == "bool":
                tensors[out_name] = np.full(out_shape, bool(fill_v), dtype=np.bool_)
            else:
                raise ValueError("ConstantOfShape output dtype unsupported.")
            continue

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
            continue

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
            continue

        if op == "Not":
            tensors[out_name] = np.logical_not(ins[0] != 0).astype(np.bool_)
            continue

        if op == "Dropout":
            data = ins[0]
            out = data.copy()
            if out_dtype in ("int8", "int16"):
                si, zi = _qparams(model, node.inputs[0])
                so, zo = _qparams(model, out_name)
                if abs(si - so) > 1e-12 or zi != zo:
                    out_f = _dequantize_int(out, si, zi)
                    out = _quantize_float(out_f, so, zo, out_dtype)
            tensors[out_name] = out
            continue

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
            continue

        if op == "IsNaN":
            a = ins[0].astype(np.float32)
            tensors[out_name] = np.isnan(a).astype(np.bool_)
            continue

        if op == "Mod":
            a, b = ins[0], ins[1]
            fmod_mode = int(node.attrs.get("fmod", 0))
            if out_dtype in ("int8", "int16"):
                sa, za = _qparams(model, node.inputs[0])
                sb, zb = _qparams(model, node.inputs[1])
                so, zo = _qparams(model, out_name)
                ra = _dequantize_int(a, sa, za)
                rb = _dequantize_int(b, sb, zb)
                if fmod_mode == 1:
                    ro = np.fmod(ra, rb)
                else:
                    ro = np.mod(ra, rb)
                tensors[out_name] = _quantize_float(ro, so, zo, out_dtype)
            elif fmod_mode == 1:
                tensors[out_name] = np.fmod(a, b)
            else:
                tensors[out_name] = np.mod(a, b)
            continue

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
            continue

        if op == "PRelu":
            a, slope = ins[0], ins[1]
            if out_dtype in ("int8", "int16"):
                sa, za = _qparams(model, node.inputs[0])
                so, zo = _qparams(model, out_name)
                ra = _dequantize_int(a, sa, za)
                slope_dtype = _tensor_dtype(model.tensors[node.inputs[1]])
                if slope_dtype in ("int8", "int16"):
                    ss, zs = _qparams(model, node.inputs[1])
                    rs = _dequantize_int(slope, ss, zs)
                else:
                    rs = slope.astype(np.float32, copy=False)
                ro = np.where(ra >= 0.0, ra, ra * rs)
                tensors[out_name] = _quantize_float(ro, so, zo, out_dtype)
            else:
                tensors[out_name] = np.where(a >= 0.0, a, a * slope)
            continue

        if op == "LpNormalization":
            if out_dtype in ("int8", "int16"):
                raise ValueError("LpNormalization quantized mode is not supported.")
            x = ins[0].astype(np.float32)
            if x.ndim <= 0:
                raise ValueError("LpNormalization expects rank >= 1.")
            axis = int(node.attrs.get("axis", -1))
            if axis < 0:
                axis += x.ndim
            if axis < 0 or axis >= x.ndim:
                raise ValueError("LpNormalization axis out of range.")
            p = int(node.attrs.get("p", 2))
            if p <= 0:
                raise ValueError("LpNormalization p must be positive.")
            norm = np.sum(np.power(np.abs(x), p), axis=axis, keepdims=True)
            norm = np.power(norm, 1.0 / float(p))
            tensors[out_name] = np.divide(x, norm, out=np.zeros_like(x), where=norm > 0.0)
            continue

        if op == "MeanVarianceNormalization":
            if out_dtype in ("int8", "int16"):
                raise ValueError("MeanVarianceNormalization quantized mode is not supported.")
            x = ins[0].astype(np.float32)
            if x.ndim != 4:
                raise ValueError("MeanVarianceNormalization expects 4D input.")
            axes = node.attrs.get("axes", [0, 2, 3])
            if not isinstance(axes, (list, tuple)):
                raise ValueError("MeanVarianceNormalization axes must be a list.")
            norm_axes: list[int] = []
            for axis in axes:
                ax = int(axis)
                if ax < 0:
                    ax += x.ndim
                if ax < 0 or ax >= x.ndim:
                    raise ValueError("MeanVarianceNormalization axis out of range.")
                norm_axes.append(ax)
            if set(norm_axes) != {0, 2, 3}:
                raise ValueError("MeanVarianceNormalization currently supports axes=[0,2,3] only.")
            mean = np.mean(x, axis=tuple(norm_axes), keepdims=True)
            var = np.mean(np.square(x - mean), axis=tuple(norm_axes), keepdims=True)
            tensors[out_name] = (x - mean) / np.sqrt(var + 1e-12)
            continue

        if op in (
            "Abs",
            "Neg",
            "Exp",
            "Sign",
            "Erf",
            "Elu",
            "Celu",
            "Selu",
            "Sin",
            "Cos",
            "Tan",
            "Asin",
            "Acos",
            "Atan",
            "Sinh",
            "Cosh",
            "Asinh",
            "Acosh",
            "Atanh",
            "Log",
            "Reciprocal",
            "Sqrt",
            "Floor",
            "Ceil",
            "Round",
            "LeakyRelu",
            "ThresholdedRelu",
            "HardSigmoid",
            "Sigmoid",
            "Tanh",
            "Softplus",
            "Softsign",
            "Clip",
        ):
            a = ins[0]
            if out_dtype in ("int8", "int16"):
                if op == "Sign":
                    ai = a.astype(np.int64)
                    si = np.where(ai > 0, 1, np.where(ai < 0, -1, 0))
                    if out_dtype == "int8":
                        tensors[out_name] = si.astype(np.int8)
                    else:
                        tensors[out_name] = si.astype(np.int16)
                    continue
                sa, za = _qparams(model, node.inputs[0])
                so, zo = _qparams(model, out_name)
                ra = _dequantize_int(a, sa, za)
                if op == "Abs":
                    ro = np.abs(ra)
                elif op == "Neg":
                    ro = -ra
                elif op == "Exp":
                    ro = np.exp(ra)
                elif op == "Erf":
                    if hasattr(np, "erf"):
                        ro = np.erf(ra)
                    else:
                        ro = np.vectorize(math.erf)(ra).astype(np.float32)
                elif op == "Elu":
                    alpha = float(node.attrs.get("alpha", 1.0))
                    ro = np.where(ra >= 0.0, ra, alpha * (np.exp(ra) - 1.0))
                elif op == "Celu":
                    alpha = float(node.attrs.get("alpha", 1.0))
                    if alpha <= 0.0:
                        raise ValueError("Celu alpha must be > 0.")
                    ro = np.where(ra > 0.0, ra, alpha * (np.exp(ra / alpha) - 1.0))
                elif op == "Selu":
                    alpha = float(node.attrs.get("alpha", 1.6732631921768188))
                    gamma = float(node.attrs.get("gamma", 1.0507010221481323))
                    ro = np.where(ra > 0.0, gamma * ra, gamma * alpha * (np.exp(ra) - 1.0))
                elif op == "Sin":
                    ro = np.sin(ra)
                elif op == "Cos":
                    ro = np.cos(ra)
                elif op == "Tan":
                    ro = np.tan(ra)
                elif op == "Asin":
                    ro = np.arcsin(ra)
                elif op == "Acos":
                    ro = np.arccos(ra)
                elif op == "Atan":
                    ro = np.arctan(ra)
                elif op == "Sinh":
                    ro = np.sinh(ra)
                elif op == "Cosh":
                    ro = np.cosh(ra)
                elif op == "Asinh":
                    ro = np.arcsinh(ra)
                elif op == "Acosh":
                    ro = np.arccosh(ra)
                elif op == "Atanh":
                    ro = np.arctanh(ra)
                elif op == "Log":
                    ro = np.log(ra)
                elif op == "Reciprocal":
                    ro = 1.0 / ra
                elif op == "Sqrt":
                    ro = np.sqrt(ra)
                elif op == "Floor":
                    ro = np.floor(ra)
                elif op == "Ceil":
                    ro = np.ceil(ra)
                elif op == "Round":
                    ro = np.rint(ra)
                elif op == "LeakyRelu":
                    alpha = float(node.attrs.get("alpha", 0.01))
                    ro = np.where(ra >= 0.0, ra, alpha * ra)
                elif op == "ThresholdedRelu":
                    alpha = float(node.attrs.get("alpha", 1.0))
                    ro = np.where(ra > alpha, ra, 0.0)
                elif op == "HardSigmoid":
                    alpha = float(node.attrs.get("alpha", 0.2))
                    beta = float(node.attrs.get("beta", 0.5))
                    ro = np.clip(alpha * ra + beta, 0.0, 1.0)
                elif op == "Sigmoid":
                    ro = 1.0 / (1.0 + np.exp(-ra))
                elif op == "Tanh":
                    ro = np.tanh(ra)
                elif op == "Softplus":
                    ro = np.log1p(np.exp(ra))
                elif op == "Softsign":
                    ro = ra / (1.0 + np.abs(ra))
                else:
                    min_v = float(node.attrs["min"])
                    max_v = float(node.attrs["max"])
                    ro = np.clip(ra, min_v, max_v)
                tensors[out_name] = _quantize_float(ro, so, zo, out_dtype)
            else:
                if op == "Abs":
                    out = np.abs(a)
                elif op == "Neg":
                    out = -a
                elif op == "Exp":
                    out = np.exp(a)
                elif op == "Sign":
                    out = np.sign(a)
                elif op == "Erf":
                    if hasattr(np, "erf"):
                        out = np.erf(a)
                    else:
                        out = np.vectorize(math.erf)(a).astype(np.float32)
                elif op == "Elu":
                    alpha = float(node.attrs.get("alpha", 1.0))
                    out = np.where(a >= 0.0, a, alpha * (np.exp(a) - 1.0))
                elif op == "Celu":
                    alpha = float(node.attrs.get("alpha", 1.0))
                    if alpha <= 0.0:
                        raise ValueError("Celu alpha must be > 0.")
                    out = np.where(a > 0.0, a, alpha * (np.exp(a / alpha) - 1.0))
                elif op == "Selu":
                    alpha = float(node.attrs.get("alpha", 1.6732631921768188))
                    gamma = float(node.attrs.get("gamma", 1.0507010221481323))
                    out = np.where(a > 0.0, gamma * a, gamma * alpha * (np.exp(a) - 1.0))
                elif op == "Sin":
                    out = np.sin(a)
                elif op == "Cos":
                    out = np.cos(a)
                elif op == "Tan":
                    out = np.tan(a)
                elif op == "Asin":
                    out = np.arcsin(a)
                elif op == "Acos":
                    out = np.arccos(a)
                elif op == "Atan":
                    out = np.arctan(a)
                elif op == "Sinh":
                    out = np.sinh(a)
                elif op == "Cosh":
                    out = np.cosh(a)
                elif op == "Asinh":
                    out = np.arcsinh(a)
                elif op == "Acosh":
                    out = np.arccosh(a)
                elif op == "Atanh":
                    out = np.arctanh(a)
                elif op == "Log":
                    out = np.log(a)
                elif op == "Reciprocal":
                    out = 1.0 / a
                elif op == "Sqrt":
                    out = np.sqrt(a)
                elif op == "Floor":
                    out = np.floor(a)
                elif op == "Ceil":
                    out = np.ceil(a)
                elif op == "Round":
                    out = np.rint(a)
                elif op == "LeakyRelu":
                    alpha = float(node.attrs.get("alpha", 0.01))
                    out = np.where(a >= 0.0, a, alpha * a)
                elif op == "ThresholdedRelu":
                    alpha = float(node.attrs.get("alpha", 1.0))
                    out = np.where(a > alpha, a, 0.0)
                elif op == "HardSigmoid":
                    alpha = float(node.attrs.get("alpha", 0.2))
                    beta = float(node.attrs.get("beta", 0.5))
                    out = np.clip(alpha * a + beta, 0.0, 1.0)
                elif op == "Sigmoid":
                    out = 1.0 / (1.0 + np.exp(-a))
                elif op == "Tanh":
                    out = np.tanh(a)
                elif op == "Softplus":
                    out = np.log1p(np.exp(a))
                elif op == "Softsign":
                    out = a / (1.0 + np.abs(a))
                else:
                    min_v = float(node.attrs["min"])
                    max_v = float(node.attrs["max"])
                    out = np.clip(a, min_v, max_v)
                tensors[out_name] = out
            continue

        if op == "Relu":
            a = ins[0]
            if out_dtype in ("int8", "int16"):
                _, zo = _qparams(model, out_name)
                out = np.maximum(a.astype(np.int32), int(zo))
                if out_dtype == "int8":
                    out = out.astype(np.int8)
                else:
                    out = out.astype(np.int16)
                tensors[out_name] = out
            else:
                tensors[out_name] = np.maximum(a, 0.0)
            continue

        if op == "Identity":
            tensors[out_name] = ins[0].copy()
            continue

        if op == "Cast":
            target = int(node.attrs.get("to", 0))
            if target == TensorProto.FLOAT:
                tensors[out_name] = ins[0].astype(np.float32)
            elif target == TensorProto.INT8:
                tensors[out_name] = ins[0].astype(np.int8)
            elif target == TensorProto.INT16:
                tensors[out_name] = ins[0].astype(np.int16)
            elif target == TensorProto.INT32:
                tensors[out_name] = ins[0].astype(np.int32)
            elif target == TensorProto.INT64:
                tensors[out_name] = ins[0].astype(np.int64)
            else:
                raise ValueError("Cast target dtype is unsupported.")
            continue

        if op == "MatMul":
            a, b = ins[0], ins[1]
            if out_dtype in ("int8", "int16"):
                sa, za = _qparams(model, node.inputs[0])
                sb, zb = _qparams(model, node.inputs[1])
                so, zo = _qparams(model, out_name)
                ra = _dequantize_int(a, sa, za)
                rb = _dequantize_int(b, sb, zb)
                out = np.matmul(ra, rb)
                tensors[out_name] = _quantize_float(out, so, zo, out_dtype)
            else:
                tensors[out_name] = np.matmul(a, b)
            continue

        if op == "MatMulInteger":
            if len(ins) < 2:
                raise ValueError("MatMulInteger expects at least 2 inputs.")
            a = ins[0].astype(np.int64)
            b = ins[1].astype(np.int64)
            a_zero = int(ins[2].reshape(-1)[0]) if len(ins) >= 3 else 0
            b_zero = int(ins[3].reshape(-1)[0]) if len(ins) >= 4 else 0
            out = np.matmul(a - a_zero, b - b_zero)
            if out_dtype == "int32":
                tensors[out_name] = out.astype(np.int32)
            elif out_dtype == "int64":
                tensors[out_name] = out.astype(np.int64)
            else:
                raise ValueError("MatMulInteger output dtype must be int32/int64.")
            continue

        if op == "QLinearMatMul":
            if len(ins) < 8:
                raise ValueError("QLinearMatMul expects 8 inputs.")
            a = ins[0].astype(np.int64)
            a_scale = float(ins[1].reshape(-1)[0])
            a_zero = int(ins[2].reshape(-1)[0])
            b = ins[3].astype(np.int64)
            b_scale = float(ins[4].reshape(-1)[0])
            b_zero = int(ins[5].reshape(-1)[0])
            y_scale = float(ins[6].reshape(-1)[0])
            y_zero = int(ins[7].reshape(-1)[0])
            if y_scale == 0.0:
                raise ValueError("QLinearMatMul y_scale must be non-zero.")
            acc = np.matmul(a - a_zero, b - b_zero).astype(np.float32)
            out_f = acc * (a_scale * b_scale)
            q = np.round(out_f / y_scale).astype(np.int64) + y_zero
            if out_dtype == "int8":
                q = np.clip(q, -128, 127)
                tensors[out_name] = q.astype(np.int8)
            elif out_dtype == "int16":
                q = np.clip(q, -32768, 32767)
                tensors[out_name] = q.astype(np.int16)
            else:
                raise ValueError("QLinearMatMul output dtype must be int8/int16.")
            continue

        if op in ("ArgMax", "ArgMin"):
            a = ins[0]
            axis = int(node.attrs.get("axis", 0))
            if axis < 0:
                axis += a.ndim
            if axis < 0 or axis >= a.ndim:
                raise ValueError("Arg op axis out of range.")
            keepdims = int(node.attrs.get("keepdims", 1))
            if keepdims not in (0, 1):
                raise ValueError("Arg op keepdims must be 0 or 1.")
            select_last = int(node.attrs.get("select_last_index", 0))
            if select_last not in (0, 1):
                raise ValueError("Arg op select_last_index must be 0 or 1.")
            if op == "ArgMax":
                if select_last == 1:
                    rev = np.flip(a, axis=axis)
                    idx = np.argmax(rev, axis=axis)
                    idx = a.shape[axis] - 1 - idx
                else:
                    idx = np.argmax(a, axis=axis)
            else:
                if select_last == 1:
                    rev = np.flip(a, axis=axis)
                    idx = np.argmin(rev, axis=axis)
                    idx = a.shape[axis] - 1 - idx
                else:
                    idx = np.argmin(a, axis=axis)
            if keepdims == 1:
                idx = np.expand_dims(idx, axis=axis)
            if out_dtype == "int32":
                tensors[out_name] = idx.astype(np.int32)
            else:
                tensors[out_name] = idx.astype(np.int64)
            continue

        if op == "Gemm":
            a, b = ins[0], ins[1]
            c = ins[2] if len(ins) >= 3 else None
            if out_dtype in ("int8", "int16"):
                sa, za = _qparams(model, node.inputs[0])
                sb, zb = _qparams(model, node.inputs[1])
                so, zo = _qparams(model, out_name)
                ra = _dequantize_int(a, sa, za)
                rb = _dequantize_int(b, sb, zb)
                out = np.matmul(ra, rb)
                if c is not None:
                    c_dtype = _tensor_dtype(model.tensors[node.inputs[2]])
                    if c_dtype == "float32":
                        out = out + c
                    elif c_dtype in ("int32", "int64"):
                        out = out + (c.astype(np.float32) * (sa * sb))
                    else:
                        sc, zc = _qparams(model, node.inputs[2])
                        out = out + _dequantize_int(c, sc, zc)
                tensors[out_name] = _quantize_float(out, so, zo, out_dtype)
            else:
                out = np.matmul(a, b)
                if c is not None:
                    if c.size == out.shape[1]:
                        out = out + c
                    else:
                        out = out + c.reshape(out.shape)
                tensors[out_name] = out
            continue

        if op == "Softmax":
            if out_dtype in ("int8", "int16"):
                si, zi = _qparams(model, node.inputs[0])
                so, zo = _qparams(model, out_name)
                a = _dequantize_int(ins[0], si, zi)
            else:
                a = ins[0].astype(np.float32)
            axis = int(node.attrs.get("axis", -1))
            if axis < 0:
                axis += a.ndim
            if axis < 0 or axis >= a.ndim:
                raise ValueError("Softmax axis out of range.")
            max_v = np.max(a, axis=axis, keepdims=True)
            e = np.exp(a - max_v)
            out = e / np.sum(e, axis=axis, keepdims=True)
            if out_dtype in ("int8", "int16"):
                tensors[out_name] = _quantize_float(out, so, zo, out_dtype)
            else:
                tensors[out_name] = out
            continue

        if op == "LogSoftmax":
            if out_dtype in ("int8", "int16"):
                si, zi = _qparams(model, node.inputs[0])
                so, zo = _qparams(model, out_name)
                a = _dequantize_int(ins[0], si, zi)
            else:
                a = ins[0].astype(np.float32)
            axis = int(node.attrs.get("axis", 1))
            if axis < 0:
                axis += a.ndim
            if axis < 0 or axis >= a.ndim:
                raise ValueError("LogSoftmax axis out of range.")
            max_v = np.max(a, axis=axis, keepdims=True)
            e = np.exp(a - max_v)
            sum_e = np.sum(e, axis=axis, keepdims=True)
            out = (a - max_v) - np.log(sum_e)
            if out_dtype in ("int8", "int16"):
                tensors[out_name] = _quantize_float(out, so, zo, out_dtype)
            else:
                tensors[out_name] = out
            continue

        if op == "Hardmax":
            a = ins[0]
            axis = int(node.attrs.get("axis", 1))
            if axis < 0:
                axis += a.ndim
            if axis < 0 or axis >= a.ndim:
                raise ValueError("Hardmax axis out of range.")
            idx = np.argmax(a, axis=axis, keepdims=True)
            out = np.zeros_like(a)
            if out_dtype in ("int8", "int16"):
                so, zo = _qparams(model, out_name)
                q1 = _quantize_float(np.array([1.0], dtype=np.float32), so, zo, out_dtype)[0]
                q0 = _quantize_float(np.array([0.0], dtype=np.float32), so, zo, out_dtype)[0]
                out.fill(q0)
                np.put_along_axis(out, idx, q1, axis=axis)
            else:
                np.put_along_axis(out, idx, 1, axis=axis)
            tensors[out_name] = out
            continue

        if op == "Range":
            if len(ins) != 3:
                raise ValueError("Range expects 3 inputs.")
            out_shape = tuple(int(v) for v in model.tensors[out_name].shape)
            if len(out_shape) != 1:
                raise ValueError("Range output must be 1D.")
            out_size = out_shape[0]
            if out_size <= 0:
                raise ValueError("Range output size must be positive.")
            dtype = _tensor_dtype(model.tensors[out_name])
            if dtype == "float32":
                start = float(ins[0].reshape(-1)[0])
                delta = float(ins[2].reshape(-1)[0])
                if delta == 0.0:
                    raise ValueError("Range delta must be non-zero.")
                out = start + np.arange(out_size, dtype=np.float32) * delta
                tensors[out_name] = out.astype(np.float32)
                continue
            if dtype in ("int8", "int16", "int32", "int64"):
                start = int(ins[0].reshape(-1)[0])
                delta = int(ins[2].reshape(-1)[0])
                if delta == 0:
                    raise ValueError("Range delta must be non-zero.")
                out = start + np.arange(out_size, dtype=np.int64) * delta
                if dtype == "int8":
                    tensors[out_name] = out.astype(np.int8)
                elif dtype == "int16":
                    tensors[out_name] = out.astype(np.int16)
                elif dtype == "int32":
                    tensors[out_name] = out.astype(np.int32)
                else:
                    tensors[out_name] = out.astype(np.int64)
                continue
            raise ValueError("Range output dtype unsupported.")

        if op == "TopK":
            if len(node.outputs) != 2:
                raise ValueError("TopK expects 2 outputs.")
            x = ins[0]
            k_vals = ins[1].reshape(-1)
            if k_vals.size < 1:
                raise ValueError("TopK K input is invalid.")
            k = int(k_vals[0])
            if k <= 0:
                raise ValueError("TopK K must be positive.")
            axis = int(node.attrs.get("axis", -1))
            if axis < 0:
                axis += x.ndim
            if axis < 0 or axis >= x.ndim:
                raise ValueError("TopK axis out of range.")
            if k > x.shape[axis]:
                raise ValueError("TopK K exceeds axis dimension.")
            largest = int(node.attrs.get("largest", 1))
            sorted_flag = int(node.attrs.get("sorted", 1))
            if largest not in (0, 1):
                raise ValueError("TopK largest must be 0/1.")
            if sorted_flag not in (0, 1):
                raise ValueError("TopK sorted must be 0/1.")

            if largest == 1:
                part = np.argpartition(-x, k - 1, axis=axis)
            else:
                part = np.argpartition(x, k - 1, axis=axis)
            k_idx = np.arange(k, dtype=np.int64)
            idx = np.take(part, k_idx, axis=axis)
            vals = np.take_along_axis(x, idx, axis=axis)

            if sorted_flag == 1:
                if largest == 1:
                    order = np.argsort(-vals, axis=axis, kind="stable")
                else:
                    order = np.argsort(vals, axis=axis, kind="stable")
                idx = np.take_along_axis(idx, order, axis=axis)
                vals = np.take_along_axis(vals, order, axis=axis)

            idx_name = node.outputs[1]
            idx_dtype = _tensor_dtype(model.tensors[idx_name])
            if idx_dtype == "int32":
                tensors[idx_name] = idx.astype(np.int32)
            else:
                tensors[idx_name] = idx.astype(np.int64)
            tensors[node.outputs[0]] = vals.astype(ins[0].dtype, copy=False)
            continue

        if op == "ConvInteger":
            if len(ins) < 2:
                raise ValueError("ConvInteger expects at least 2 inputs.")
            if len(ins) > 4:
                raise ValueError("ConvInteger supports at most 4 inputs.")
            if out_dtype not in ("int32", "int64"):
                raise ValueError("ConvInteger output must be int32/int64.")
            x = ins[0]
            w = ins[1]
            if x.ndim != 4 or w.ndim != 4:
                raise ValueError("ConvInteger expects 4D tensors (NCHW).")
            n, c_in, h, w_in = x.shape
            m, c_per_g, k_h, k_w = w.shape
            strides = list(node.attrs.get("strides", [1, 1]))
            pads = list(node.attrs.get("pads", [0, 0, 0, 0]))
            dilations = list(node.attrs.get("dilations", [1, 1]))
            groups = int(node.attrs.get("group", 1))
            if groups <= 0:
                raise ValueError("ConvInteger group must be positive.")
            if c_in != c_per_g * groups:
                raise ValueError("ConvInteger channel mismatch.")
            if m % groups != 0:
                raise ValueError("ConvInteger output channels must be divisible by groups.")
            if len(pads) == 2:
                pads = [pads[0], pads[1], pads[0], pads[1]]
            stride_h, stride_w = [int(v) for v in strides]
            pad_h0, pad_w0, pad_h1, pad_w1 = [int(v) for v in pads]
            dil_h, dil_w = [int(v) for v in dilations]
            out_h = (h + pad_h0 + pad_h1 - dil_h * (k_h - 1) - 1) // stride_h + 1
            out_w = (w_in + pad_w0 + pad_w1 - dil_w * (k_w - 1) - 1) // stride_w + 1
            oc_per_group = m // groups

            x_zero = 0
            if len(ins) >= 3:
                x_zp_vals = ins[2].reshape(-1).astype(np.int64)
                if x_zp_vals.size != 1:
                    raise ValueError("ConvInteger x_zero_point must be scalar.")
                x_zero = int(x_zp_vals[0])
            if len(ins) >= 4:
                w_zp_vals = ins[3].reshape(-1).astype(np.int64)
                if w_zp_vals.size not in (1, m):
                    raise ValueError("ConvInteger w_zero_point must be scalar or per-output-channel.")
            else:
                w_zp_vals = np.array([0], dtype=np.int64)

            out64 = np.zeros((n, m, out_h, out_w), dtype=np.int64)
            for ni in range(n):
                for oc in range(m):
                    wz = int(w_zp_vals[0]) if w_zp_vals.size == 1 else int(w_zp_vals[oc])
                    g = oc // oc_per_group
                    ic_begin = g * c_per_g
                    for oh in range(out_h):
                        for ow in range(out_w):
                            acc = np.int64(0)
                            for ic_local in range(c_per_g):
                                ic = ic_begin + ic_local
                                for kh in range(k_h):
                                    for kw in range(k_w):
                                        in_h = oh * stride_h + kh * dil_h - pad_h0
                                        in_w = ow * stride_w + kw * dil_w - pad_w0
                                        if 0 <= in_h < h and 0 <= in_w < w_in:
                                            xv = int(x[ni, ic, in_h, in_w]) - x_zero
                                            wv = int(w[oc, ic_local, kh, kw]) - wz
                                            acc += np.int64(xv * wv)
                            out64[ni, oc, oh, ow] = acc
            if out_dtype == "int32":
                tensors[out_name] = np.clip(out64, -2147483648, 2147483647).astype(np.int32)
            else:
                tensors[out_name] = out64
            continue

        if op == "QLinearConv":
            if len(ins) < 8:
                raise ValueError("QLinearConv expects at least 8 inputs.")
            if len(ins) > 9:
                raise ValueError("QLinearConv supports at most 9 inputs.")
            if out_dtype not in ("int8", "int16"):
                raise ValueError("QLinearConv output must be int8/int16.")
            x = ins[0]
            w = ins[3]
            if x.ndim != 4 or w.ndim != 4:
                raise ValueError("QLinearConv expects 4D tensors (NCHW).")
            x_scale_vals = ins[1].reshape(-1).astype(np.float32)
            x_zero_vals = ins[2].reshape(-1).astype(np.int64)
            w_scale_vals = ins[4].reshape(-1).astype(np.float32)
            w_zero_vals = ins[5].reshape(-1).astype(np.int64)
            y_scale_vals = ins[6].reshape(-1).astype(np.float32)
            y_zero_vals = ins[7].reshape(-1).astype(np.int64)
            if x_scale_vals.size != 1 or x_zero_vals.size != 1:
                raise ValueError("QLinearConv x_scale/x_zero_point must be scalar.")
            if y_scale_vals.size != 1 or y_zero_vals.size != 1:
                raise ValueError("QLinearConv y_scale/y_zero_point must be scalar.")
            x_scale = float(x_scale_vals[0])
            x_zero = int(x_zero_vals[0])
            y_scale = float(y_scale_vals[0])
            y_zero = int(y_zero_vals[0])
            if y_scale == 0.0:
                raise ValueError("QLinearConv y_scale must be non-zero.")

            n, c_in, h, w_in = x.shape
            m, c_per_g, k_h, k_w = w.shape
            if w_scale_vals.size not in (1, m):
                raise ValueError("QLinearConv w_scale must be scalar or per-output-channel.")
            if w_zero_vals.size not in (1, m):
                raise ValueError("QLinearConv w_zero_point must be scalar or per-output-channel.")

            strides = list(node.attrs.get("strides", [1, 1]))
            pads = list(node.attrs.get("pads", [0, 0, 0, 0]))
            dilations = list(node.attrs.get("dilations", [1, 1]))
            groups = int(node.attrs.get("group", 1))
            if groups <= 0:
                raise ValueError("QLinearConv group must be positive.")
            if c_in != c_per_g * groups:
                raise ValueError("QLinearConv channel mismatch.")
            if m % groups != 0:
                raise ValueError("QLinearConv output channels must be divisible by groups.")
            if len(pads) == 2:
                pads = [pads[0], pads[1], pads[0], pads[1]]
            stride_h, stride_w = [int(v) for v in strides]
            pad_h0, pad_w0, pad_h1, pad_w1 = [int(v) for v in pads]
            dil_h, dil_w = [int(v) for v in dilations]
            out_h = (h + pad_h0 + pad_h1 - dil_h * (k_h - 1) - 1) // stride_h + 1
            out_w = (w_in + pad_w0 + pad_w1 - dil_w * (k_w - 1) - 1) // stride_w + 1
            oc_per_group = m // groups
            bias = ins[8] if len(ins) >= 9 else None

            out_f = np.zeros((n, m, out_h, out_w), dtype=np.float32)
            for ni in range(n):
                for oc in range(m):
                    ws = float(w_scale_vals[0]) if w_scale_vals.size == 1 else float(w_scale_vals[oc])
                    wz = int(w_zero_vals[0]) if w_zero_vals.size == 1 else int(w_zero_vals[oc])
                    acc_bias = 0.0
                    if bias is not None:
                        if np.issubdtype(bias.dtype, np.floating):
                            acc_bias = float(bias[oc])
                        else:
                            acc_bias = float(bias[oc]) * x_scale * ws
                    g = oc // oc_per_group
                    ic_begin = g * c_per_g
                    for oh in range(out_h):
                        for ow in range(out_w):
                            acc = float(acc_bias)
                            for ic_local in range(c_per_g):
                                ic = ic_begin + ic_local
                                for kh in range(k_h):
                                    for kw in range(k_w):
                                        in_h = oh * stride_h + kh * dil_h - pad_h0
                                        in_w = ow * stride_w + kw * dil_w - pad_w0
                                        if 0 <= in_h < h and 0 <= in_w < w_in:
                                            rx = (float(x[ni, ic, in_h, in_w]) - x_zero) * x_scale
                                            rw = (float(w[oc, ic_local, kh, kw]) - wz) * ws
                                            acc += rx * rw
                            out_f[ni, oc, oh, ow] = acc

            q = np.round(out_f / y_scale).astype(np.int64) + y_zero
            if out_dtype == "int8":
                tensors[out_name] = np.clip(q, -128, 127).astype(np.int8)
            else:
                tensors[out_name] = np.clip(q, -32768, 32767).astype(np.int16)
            continue

        if op == "Reshape":
            shape_vals = _const_ints(tensors, node.inputs[1])
            tensors[out_name] = _reshape_like_onnx(ins[0], shape_vals)
            continue

        if op == "NonMaxSuppression":
            if len(ins) < 2:
                raise ValueError("NonMaxSuppression expects at least 2 inputs.")
            boxes = ins[0].astype(np.float32, copy=False)
            scores = ins[1].astype(np.float32, copy=False)
            if boxes.ndim != 3 or scores.ndim != 3:
                raise ValueError("NonMaxSuppression expects boxes/scores rank=3.")
            batch, spatial, four = boxes.shape
            if four != 4:
                raise ValueError("NonMaxSuppression boxes last dimension must be 4.")
            if scores.shape[0] != batch or scores.shape[2] != spatial:
                raise ValueError("NonMaxSuppression shape mismatch.")
            classes = int(scores.shape[1])
            max_output = 0
            if len(ins) >= 3:
                max_vals = ins[2].reshape(-1).astype(np.int64)
                if max_vals.size != 1:
                    raise ValueError("NonMaxSuppression max_output_boxes_per_class must be scalar.")
                max_output = int(max_vals[0])
            iou_threshold = 0.0
            if len(ins) >= 4:
                iou_vals = ins[3].reshape(-1).astype(np.float32)
                if iou_vals.size != 1:
                    raise ValueError("NonMaxSuppression iou_threshold must be scalar.")
                iou_threshold = float(iou_vals[0])
            score_threshold = -3.402823466e38
            if len(ins) >= 5:
                sc_vals = ins[4].reshape(-1).astype(np.float32)
                if sc_vals.size != 1:
                    raise ValueError("NonMaxSuppression score_threshold must be scalar.")
                score_threshold = float(sc_vals[0])
            center_point_box = int(node.attrs.get("center_point_box", 0))
            if center_point_box not in (0, 1):
                raise ValueError("NonMaxSuppression center_point_box must be 0/1.")

            out_shape = tuple(int(v) for v in model.tensors[out_name].shape)
            if len(out_shape) != 2 or out_shape[1] != 3:
                raise ValueError("NonMaxSuppression output shape must be [N,3].")
            out_cap = int(out_shape[0])
            out_dtype = _tensor_dtype(model.tensors[out_name])
            if out_dtype not in ("int64", "int32"):
                raise ValueError("NonMaxSuppression output dtype must be int64/int32.")
            out = np.full((out_cap, 3), -1, dtype=np.int64)
            if max_output <= 0:
                tensors[out_name] = out.astype(np.int32) if out_dtype == "int32" else out
                continue
            per_class_limit = min(max_output, spatial)

            def _to_corners(raw_box: np.ndarray) -> tuple[float, float, float, float]:
                if center_point_box == 0:
                    y1, x1, y2, x2 = [float(v) for v in raw_box]
                else:
                    x_center, y_center, w_val, h_val = [float(v) for v in raw_box]
                    x1 = x_center - 0.5 * w_val
                    y1 = y_center - 0.5 * h_val
                    x2 = x_center + 0.5 * w_val
                    y2 = y_center + 0.5 * h_val
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                return x1, y1, x2, y2

            def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
                ax1, ay1, ax2, ay2 = _to_corners(box_a)
                bx1, by1, bx2, by2 = _to_corners(box_b)
                inter_x1 = max(ax1, bx1)
                inter_y1 = max(ay1, by1)
                inter_x2 = min(ax2, bx2)
                inter_y2 = min(ay2, by2)
                inter_w = max(0.0, inter_x2 - inter_x1)
                inter_h = max(0.0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h
                area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
                area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                denom = area_a + area_b - inter_area
                if denom <= 0.0:
                    return 0.0
                return float(inter_area / denom)

            out_pos = 0
            for b in range(batch):
                for c in range(classes):
                    class_scores = scores[b, c, :]
                    order = np.argsort(-class_scores, kind="stable")
                    selected_indices: list[int] = []
                    for idx in order.tolist():
                        score = float(class_scores[idx])
                        if score < score_threshold:
                            continue
                        keep = True
                        cur_box = boxes[b, idx, :]
                        for prev_idx in selected_indices:
                            if _iou(cur_box, boxes[b, prev_idx, :]) > iou_threshold:
                                keep = False
                                break
                        if not keep:
                            continue
                        selected_indices.append(int(idx))
                        if out_pos < out_cap:
                            out[out_pos, 0] = int(b)
                            out[out_pos, 1] = int(c)
                            out[out_pos, 2] = int(idx)
                            out_pos += 1
                        if len(selected_indices) >= per_class_limit:
                            break
            tensors[out_name] = out.astype(np.int32) if out_dtype == "int32" else out
            continue

        if op == "RoiAlign":
            if len(ins) != 3:
                raise ValueError("RoiAlign expects 3 inputs.")
            x = ins[0].astype(np.float32, copy=False)
            rois = ins[1].astype(np.float32, copy=False)
            batch_indices = ins[2].astype(np.int64, copy=False).reshape(-1)
            if x.ndim != 4:
                raise ValueError("RoiAlign X must be 4D NCHW.")
            if rois.ndim != 2 or rois.shape[1] != 4:
                raise ValueError("RoiAlign rois must be [num_rois,4].")
            if batch_indices.ndim != 1:
                raise ValueError("RoiAlign batch_indices must be 1D.")
            num_rois = int(rois.shape[0])
            if batch_indices.shape[0] != num_rois:
                raise ValueError("RoiAlign rois/batch_indices mismatch.")
            out_shape = tuple(int(v) for v in model.tensors[out_name].shape)
            if len(out_shape) != 4:
                raise ValueError("RoiAlign output must be 4D.")
            if out_shape[0] != num_rois or out_shape[1] != x.shape[1]:
                raise ValueError("RoiAlign output shape mismatch.")
            out_h = int(node.attrs.get("output_height", 1))
            out_w = int(node.attrs.get("output_width", 1))
            if out_h <= 0 or out_w <= 0:
                raise ValueError("RoiAlign output_height/output_width must be positive.")
            if out_shape[2] != out_h or out_shape[3] != out_w:
                raise ValueError("RoiAlign output shape mismatch with attrs.")
            spatial_scale = float(node.attrs.get("spatial_scale", 1.0))
            sampling_ratio = int(node.attrs.get("sampling_ratio", 0))
            if sampling_ratio < 0:
                raise ValueError("RoiAlign sampling_ratio must be >= 0.")
            mode = node.attrs.get("mode", "avg")
            if isinstance(mode, bytes):
                mode = mode.decode("utf-8", errors="ignore")
            mode = str(mode).lower()
            if mode not in ("avg", "max"):
                raise ValueError("RoiAlign mode must be avg/max.")
            n, c, h, w_in = x.shape
            center_point_box = 0

            def _to_corners(raw_box: np.ndarray) -> tuple[float, float, float, float]:
                if center_point_box == 0:
                    x1 = float(raw_box[0]) * spatial_scale
                    y1 = float(raw_box[1]) * spatial_scale
                    x2 = float(raw_box[2]) * spatial_scale
                    y2 = float(raw_box[3]) * spatial_scale
                else:
                    cx = float(raw_box[0]) * spatial_scale
                    cy = float(raw_box[1]) * spatial_scale
                    rw = float(raw_box[2]) * spatial_scale
                    rh = float(raw_box[3]) * spatial_scale
                    x1 = cx - 0.5 * rw
                    y1 = cy - 0.5 * rh
                    x2 = cx + 0.5 * rw
                    y2 = cy + 0.5 * rh
                return x1, y1, x2, y2

            def _sample_bilinear(feat: np.ndarray, yy: float, xx: float) -> float:
                if yy < -1.0 or yy > float(h) or xx < -1.0 or xx > float(w_in):
                    return 0.0
                yy = min(max(yy, 0.0), float(h - 1))
                xx = min(max(xx, 0.0), float(w_in - 1))
                y0 = int(np.floor(yy))
                x0 = int(np.floor(xx))
                y1 = min(y0 + 1, h - 1)
                x1 = min(x0 + 1, w_in - 1)
                ly = yy - float(y0)
                lx = xx - float(x0)
                hy = 1.0 - ly
                hx = 1.0 - lx
                v00 = float(feat[y0, x0])
                v01 = float(feat[y0, x1])
                v10 = float(feat[y1, x0])
                v11 = float(feat[y1, x1])
                return v00 * hy * hx + v01 * hy * lx + v10 * ly * hx + v11 * ly * lx

            out = np.zeros((num_rois, c, out_h, out_w), dtype=np.float32)
            for r in range(num_rois):
                b = int(batch_indices[r])
                if b < 0:
                    b = 0
                if b >= n:
                    b = n - 1
                x1, y1, x2, y2 = _to_corners(rois[r])
                roi_w = max(x2 - x1, 1.0)
                roi_h = max(y2 - y1, 1.0)
                bin_h = roi_h / float(out_h)
                bin_w = roi_w / float(out_w)
                samp_h = sampling_ratio if sampling_ratio > 0 else int(np.ceil(roi_h / float(out_h)))
                samp_w = sampling_ratio if sampling_ratio > 0 else int(np.ceil(roi_w / float(out_w)))
                if samp_h < 1:
                    samp_h = 1
                if samp_w < 1:
                    samp_w = 1
                for ch in range(c):
                    feat = x[b, ch]
                    for ph in range(out_h):
                        for pw in range(out_w):
                            if mode == "max":
                                acc = -3.402823466e38
                            else:
                                acc = 0.0
                            for iy in range(samp_h):
                                yy = y1 + (float(ph) + (float(iy) + 0.5) / float(samp_h)) * bin_h
                                for ix in range(samp_w):
                                    xx = x1 + (float(pw) + (float(ix) + 0.5) / float(samp_w)) * bin_w
                                    val = _sample_bilinear(feat, yy, xx)
                                    if mode == "max":
                                        if val > acc:
                                            acc = val
                                    else:
                                        acc += val
                            if mode == "avg":
                                acc /= float(samp_h * samp_w)
                            out[r, ch, ph, pw] = acc
            tensors[out_name] = out
            continue

        if op == "Expand":
            shape_vals = _const_ints(tensors, node.inputs[1])
            target = tuple(int(v) for v in shape_vals)
            tensors[out_name] = np.broadcast_to(ins[0], target).copy()
            continue

        if op == "Tile":
            reps = _const_ints(tensors, node.inputs[1])
            tensors[out_name] = np.tile(ins[0], tuple(int(v) for v in reps))
            continue

        if op == "Upsample":
            x = ins[0]
            if x.ndim != 4:
                raise ValueError("Upsample currently supports 4D NCHW only.")
            mode = node.attrs.get("mode", "nearest")
            if isinstance(mode, bytes):
                mode = mode.decode("utf-8", errors="ignore")
            mode = str(mode).lower()
            if mode != "nearest":
                raise ValueError("Upsample currently supports nearest only.")

            scales = None
            if len(node.inputs) >= 2 and node.inputs[1]:
                if node.inputs[1] not in tensors:
                    raise ValueError("Upsample scales tensor is missing.")
                scales_arr = tensors[node.inputs[1]].reshape(-1)
                scales = [float(v) for v in scales_arr.tolist()]
            elif "scales" in node.attrs:
                scales = [float(v) for v in node.attrs["scales"]]
            if scales is None or len(scales) != x.ndim:
                raise ValueError("Upsample scales are invalid.")

            out_shape = tuple(int(v) for v in model.tensors[out_name].shape)
            if len(out_shape) != 4:
                raise ValueError("Upsample output shape is invalid.")
            _, _, in_h, in_w = x.shape
            _, _, out_h, out_w = out_shape
            scale_h = float(scales[2])
            scale_w = float(scales[3])
            if scale_h <= 0.0 or scale_w <= 0.0:
                raise ValueError("Upsample scales must be positive.")

            out = np.zeros(out_shape, dtype=x.dtype)
            for n_i in range(out_shape[0]):
                for c_i in range(out_shape[1]):
                    for oh_i in range(out_h):
                        ih = int(np.floor(float(oh_i) / scale_h))
                        if ih < 0:
                            ih = 0
                        if ih > (in_h - 1):
                            ih = in_h - 1
                        for ow_i in range(out_w):
                            iw = int(np.floor(float(ow_i) / scale_w))
                            if iw < 0:
                                iw = 0
                            if iw > (in_w - 1):
                                iw = in_w - 1
                            out[n_i, c_i, oh_i, ow_i] = x[n_i, c_i, ih, iw]
            if out_dtype in ("int8", "int16"):
                si, zi = _qparams(model, node.inputs[0])
                so, zo = _qparams(model, out_name)
                if abs(si - so) > 1e-12 or zi != zo:
                    out_f = _dequantize_int(out, si, zi)
                    out = _quantize_float(out_f, so, zo, out_dtype)
            tensors[out_name] = out
            continue

        if op == "Resize":
            x = ins[0]
            if x.ndim != 4:
                raise ValueError("Resize currently supports 4D NCHW only.")
            mode = node.attrs.get("mode", "nearest")
            if isinstance(mode, bytes):
                mode = mode.decode("utf-8", errors="ignore")
            mode = str(mode).lower()
            if mode != "nearest":
                raise ValueError("Resize currently supports nearest only.")
            coord_mode = node.attrs.get("coordinate_transformation_mode", "half_pixel")
            if isinstance(coord_mode, bytes):
                coord_mode = coord_mode.decode("utf-8", errors="ignore")
            coord_mode = str(coord_mode).lower()
            nearest_mode = node.attrs.get("nearest_mode", "round_prefer_floor")
            if isinstance(nearest_mode, bytes):
                nearest_mode = nearest_mode.decode("utf-8", errors="ignore")
            nearest_mode = str(nearest_mode).lower()

            _, _, in_h, in_w = x.shape
            out_shape = tuple(int(v) for v in model.tensors[out_name].shape)
            _, _, out_h, out_w = out_shape

            def _coord(out_idx: int, in_size: int, out_size: int) -> float:
                if coord_mode == "asymmetric":
                    return float(out_idx) * float(in_size) / float(out_size)
                if coord_mode == "half_pixel":
                    return ((float(out_idx) + 0.5) * float(in_size) / float(out_size)) - 0.5
                if coord_mode == "align_corners":
                    if out_size <= 1:
                        return 0.0
                    return float(out_idx) * float(in_size - 1) / float(out_size - 1)
                if coord_mode == "pytorch_half_pixel":
                    if out_size <= 1:
                        return 0.0
                    return ((float(out_idx) + 0.5) * float(in_size) / float(out_size)) - 0.5
                raise ValueError("Resize coordinate_transformation_mode unsupported.")

            def _nearest_index(coord: float, limit: int) -> int:
                if nearest_mode == "floor":
                    idx = int(np.floor(coord))
                elif nearest_mode == "ceil":
                    idx = int(np.ceil(coord))
                elif nearest_mode == "round_prefer_ceil":
                    idx = int(np.ceil(coord - 0.5))
                else:
                    idx = int(np.floor(coord + 0.5))
                if idx < 0:
                    return 0
                if idx > (limit - 1):
                    return limit - 1
                return idx

            out = np.zeros(out_shape, dtype=x.dtype)
            for n_i in range(out_shape[0]):
                for c_i in range(out_shape[1]):
                    for oh_i in range(out_h):
                        ih = _nearest_index(_coord(oh_i, in_h, out_h), in_h)
                        for ow_i in range(out_w):
                            iw = _nearest_index(_coord(ow_i, in_w, out_w), in_w)
                            out[n_i, c_i, oh_i, ow_i] = x[n_i, c_i, ih, iw]

            if out_dtype in ("int8", "int16"):
                si, zi = _qparams(model, node.inputs[0])
                so, zo = _qparams(model, out_name)
                if abs(si - so) > 1e-12 or zi != zo:
                    out_f = _dequantize_int(out, si, zi)
                    out = _quantize_float(out_f, so, zo, out_dtype)
            tensors[out_name] = out
            continue

        if op == "Flatten":
            axis = int(node.attrs.get("axis", 1))
            data = ins[0]
            rank = data.ndim
            if axis < 0:
                axis += rank
            if axis < 0 or axis > rank:
                raise ValueError("Flatten axis out of range.")
            dim0 = int(np.prod(data.shape[:axis])) if axis > 0 else 1
            dim1 = int(np.prod(data.shape[axis:])) if axis < rank else 1
            tensors[out_name] = data.reshape((dim0, dim1))
            continue

        if op == "Conv":
            x = ins[0]
            w = ins[1]
            b = ins[2] if len(ins) > 2 else None
            strides = list(node.attrs.get("strides", [1, 1]))
            pads = list(node.attrs.get("pads", [0, 0, 0, 0]))
            dilations = list(node.attrs.get("dilations", [1, 1]))
            groups = int(node.attrs.get("group", 1))
            if x.ndim != 4 or w.ndim != 4:
                raise ValueError("Conv expects 4D tensors (NCHW).")
            n, c_in, h, w_in = x.shape
            m, c_per_g, k_h, k_w = w.shape
            if groups <= 0:
                raise ValueError("Conv group must be positive.")
            if c_in != c_per_g * groups:
                raise ValueError("Conv channel mismatch.")
            if m % groups != 0:
                raise ValueError("Conv output channels must be divisible by groups.")
            oc_per_group = m // groups
            if len(pads) == 2:
                pads = [pads[0], pads[1], pads[0], pads[1]]
            stride_h, stride_w = strides
            pad_h0, pad_w0, pad_h1, pad_w1 = pads
            dil_h, dil_w = dilations
            out_h = (h + pad_h0 + pad_h1 - dil_h * (k_h - 1) - 1) // stride_h + 1
            out_w = (w_in + pad_w0 + pad_w1 - dil_w * (k_w - 1) - 1) // stride_w + 1
            out = np.zeros((n, m, out_h, out_w), dtype=np.float32)
            if out_dtype in ("int8", "int16"):
                sx, zx = _qparams(model, node.inputs[0])
                sw, zw = _qparams(model, node.inputs[1])
                so, zo = _qparams(model, out_name)
                for ni in range(n):
                    for oc in range(m):
                        for oh in range(out_h):
                            for ow in range(out_w):
                                acc = 0.0
                                if b is not None:
                                    b_dtype = _tensor_dtype(model.tensors[node.inputs[2]])
                                    if b_dtype == "float32":
                                        acc = float(b[oc])
                                    elif b_dtype in ("int32", "int64"):
                                        acc = float(b[oc]) * (sx * sw)
                                    else:
                                        sb, zb = _qparams(model, node.inputs[2])
                                        acc = float(b[oc] - zb) * sb
                                g = oc // oc_per_group
                                ic_begin = g * c_per_g
                                for ic_local in range(c_per_g):
                                    ic = ic_begin + ic_local
                                    for kh in range(k_h):
                                        for kw in range(k_w):
                                            in_h = oh * stride_h + kh * dil_h - pad_h0
                                            in_w = ow * stride_w + kw * dil_w - pad_w0
                                            if 0 <= in_h < h and 0 <= in_w < w_in:
                                                rx = (float(x[ni, ic, in_h, in_w]) - zx) * sx
                                                rw = (float(w[oc, ic_local, kh, kw]) - zw) * sw
                                                acc += rx * rw
                                out[ni, oc, oh, ow] = acc
                tensors[out_name] = _quantize_float(out, so, zo, out_dtype)
            else:
                for ni in range(n):
                    for oc in range(m):
                        for oh in range(out_h):
                            for ow in range(out_w):
                                acc = float(b[oc]) if b is not None else 0.0
                                g = oc // oc_per_group
                                ic_begin = g * c_per_g
                                for ic_local in range(c_per_g):
                                    ic = ic_begin + ic_local
                                    for kh in range(k_h):
                                        for kw in range(k_w):
                                            in_h = oh * stride_h + kh * dil_h - pad_h0
                                            in_w = ow * stride_w + kw * dil_w - pad_w0
                                            if 0 <= in_h < h and 0 <= in_w < w_in:
                                                acc += x[ni, ic, in_h, in_w] * w[oc, ic_local, kh, kw]
                                out[ni, oc, oh, ow] = acc
                tensors[out_name] = out
            continue

        if op == "ConvTranspose":
            x = ins[0]
            w = ins[1]
            b = ins[2] if len(ins) > 2 else None
            if out_dtype != "float32":
                raise ValueError("ConvTranspose quantized mode is not supported.")
            strides = list(node.attrs.get("strides", [1, 1]))
            pads = list(node.attrs.get("pads", [0, 0, 0, 0]))
            dilations = list(node.attrs.get("dilations", [1, 1]))
            output_padding = list(node.attrs.get("output_padding", [0, 0]))
            groups = int(node.attrs.get("group", 1))
            if groups != 1:
                raise ValueError("ConvTranspose currently supports group=1 only.")
            if x.ndim != 4 or w.ndim != 4:
                raise ValueError("ConvTranspose expects 4D tensors (NCHW).")
            n, c_in, h, w_in = x.shape
            wc_in, c_out_per_group, k_h, k_w = w.shape
            if wc_in != c_in:
                raise ValueError("ConvTranspose channel mismatch.")
            if len(pads) == 2:
                pads = [pads[0], pads[1], pads[0], pads[1]]
            if len(strides) != 2 or len(dilations) != 2 or len(pads) != 4 or len(output_padding) != 2:
                raise ValueError("ConvTranspose attributes shape mismatch.")
            stride_h, stride_w = [int(v) for v in strides]
            dil_h, dil_w = [int(v) for v in dilations]
            pad_h0, pad_w0, pad_h1, pad_w1 = [int(v) for v in pads]
            out_pad_h, out_pad_w = [int(v) for v in output_padding]
            out_h = (h - 1) * stride_h - pad_h0 - pad_h1 + dil_h * (k_h - 1) + out_pad_h + 1
            out_w = (w_in - 1) * stride_w - pad_w0 - pad_w1 + dil_w * (k_w - 1) + out_pad_w + 1
            c_out = c_out_per_group * groups
            out = np.zeros((n, c_out, out_h, out_w), dtype=np.float32)
            if b is not None:
                out += b.reshape(1, c_out, 1, 1).astype(np.float32)
            for ni in range(n):
                for ic in range(c_in):
                    for ih in range(h):
                        for iw in range(w_in):
                            xv = float(x[ni, ic, ih, iw])
                            for kh in range(k_h):
                                for kw in range(k_w):
                                    oh = ih * stride_h + kh * dil_h - pad_h0
                                    ow = iw * stride_w + kw * dil_w - pad_w0
                                    if 0 <= oh < out_h and 0 <= ow < out_w:
                                        out[ni, :, oh, ow] += xv * w[ic, :, kh, kw]
            tensors[out_name] = out
            continue

        if op in ("MaxPool", "AveragePool", "LpPool"):
            x = ins[0]
            if x.ndim < 3:
                raise ValueError("Pool expects rank >= 3 tensors.")
            spatial = x.ndim - 2
            kernel = node.attrs.get("kernel_shape")
            if kernel is None or len(kernel) != spatial:
                raise ValueError("Pool kernel_shape rank mismatch.")
            kernel = [int(v) for v in kernel]
            strides = [int(v) for v in node.attrs.get("strides", [1] * spatial)]
            if len(strides) != spatial:
                raise ValueError("Pool strides rank mismatch.")
            pads = [int(v) for v in node.attrs.get("pads", [0] * (spatial * 2))]
            if len(pads) == spatial:
                pads = pads + pads
            if len(pads) != spatial * 2:
                raise ValueError("Pool pads rank mismatch.")
            dilations = [int(v) for v in node.attrs.get("dilations", [1] * spatial)]
            if len(dilations) != spatial:
                raise ValueError("Pool dilations rank mismatch.")
            p = int(node.attrs.get("p", 2))
            if op == "LpPool" and p <= 0:
                raise ValueError("LpPool p must be positive.")
            count_include_pad = int(node.attrs.get("count_include_pad", 0))
            if op == "AveragePool" and count_include_pad not in (0, 1):
                raise ValueError("AveragePool count_include_pad must be 0 or 1.")

            n, c = x.shape[0], x.shape[1]
            spatial_in = [int(v) for v in x.shape[2:]]
            spatial_out: list[int] = []
            for i in range(spatial):
                out_dim = _conv_out_dim(
                    int(spatial_in[i]),
                    int(kernel[i]),
                    int(strides[i]),
                    int(pads[i]),
                    int(pads[i + spatial]),
                    int(dilations[i]),
                )
                if out_dim <= 0:
                    raise ValueError("Pool output shape mismatch.")
                spatial_out.append(out_dim)
            out_shape = (int(n), int(c), *spatial_out)
            out_dtype = model.tensors[out_name].dtype
            kernel_size = int(np.prod(kernel)) if kernel else 1
            if out_dtype in ("int8", "int16"):
                sa, za = _qparams(model, node.inputs[0])
                so, zo = _qparams(model, out_name)
                x_f = _dequantize_int(x, sa, za)
                out_f = np.zeros(out_shape, dtype=np.float32)
                for ni in range(n):
                    for ch in range(c):
                        for out_coord in np.ndindex(*spatial_out):
                                if op == "MaxPool":
                                    acc = -3.402823466e38
                                else:
                                    acc = 0.0
                                count = 0
                                for k_coord in np.ndindex(*kernel):
                                    in_coord = [
                                        int(out_coord[s]) * int(strides[s])
                                        + int(k_coord[s]) * int(dilations[s])
                                        - int(pads[s])
                                        for s in range(spatial)
                                    ]
                                    if not all(0 <= in_coord[s] < spatial_in[s] for s in range(spatial)):
                                        continue
                                    idx = (ni, ch, *in_coord)
                                    v = float(x_f[idx])
                                    if op == "MaxPool":
                                        if v > acc:
                                            acc = v
                                    elif op == "AveragePool":
                                        acc += v
                                        count += 1
                                    else:
                                        acc += abs(v) ** p
                                        count += 1
                                if op == "AveragePool":
                                    denom = float(kernel_size if count_include_pad == 1 else count)
                                    acc = acc / denom if denom > 0 else 0.0
                                elif op == "LpPool":
                                    acc = acc ** (1.0 / float(p))
                                out_f[(ni, ch, *out_coord)] = acc
                tensors[out_name] = _quantize_float(out_f, so, zo, out_dtype)
                continue
            out = np.zeros(out_shape, dtype=np.float32)
            for ni in range(n):
                for ch in range(c):
                    for out_coord in np.ndindex(*spatial_out):
                            if op == "MaxPool":
                                acc = -3.402823466e38
                            else:
                                acc = 0.0
                            count = 0
                            for k_coord in np.ndindex(*kernel):
                                in_coord = [
                                    int(out_coord[s]) * int(strides[s])
                                    + int(k_coord[s]) * int(dilations[s])
                                    - int(pads[s])
                                    for s in range(spatial)
                                ]
                                if not all(0 <= in_coord[s] < spatial_in[s] for s in range(spatial)):
                                    continue
                                idx = (ni, ch, *in_coord)
                                v = float(x[idx])
                                if op == "MaxPool":
                                    if v > acc:
                                        acc = v
                                elif op == "AveragePool":
                                    acc += v
                                    count += 1
                                else:
                                    acc += abs(v) ** p
                                    count += 1
                            if op == "AveragePool":
                                denom = float(kernel_size if count_include_pad == 1 else count)
                                acc = acc / denom if denom > 0 else 0.0
                            elif op == "LpPool":
                                acc = acc ** (1.0 / float(p))
                            out[(ni, ch, *out_coord)] = acc
            tensors[out_name] = out
            continue

        if op == "GlobalAveragePool":
            x = ins[0]
            if x.ndim < 3:
                raise ValueError("GlobalAveragePool expects rank >= 3 tensors.")
            n, c = x.shape[0], x.shape[1]
            axes = tuple(range(2, x.ndim))
            out_dtype = model.tensors[out_name].dtype
            if out_dtype in ("int8", "int16"):
                sa, za = _qparams(model, node.inputs[0])
                so, zo = _qparams(model, out_name)
                x_f = _dequantize_int(x, sa, za)
                out_f = np.mean(x_f, axis=axes, keepdims=True).astype(np.float32)
                tensors[out_name] = _quantize_float(out_f, so, zo, out_dtype)
                continue
            out = np.mean(x.astype(np.float32), axis=axes, keepdims=True).astype(np.float32)
            tensors[out_name] = out
            continue

        if op == "GlobalMaxPool":
            x = ins[0]
            if x.ndim < 3:
                raise ValueError("GlobalMaxPool expects rank >= 3 tensors.")
            axes = tuple(range(2, x.ndim))
            out_dtype = model.tensors[out_name].dtype
            if out_dtype in ("int8", "int16"):
                sa, za = _qparams(model, node.inputs[0])
                so, zo = _qparams(model, out_name)
                x_f = _dequantize_int(x, sa, za)
                out_f = np.max(x_f, axis=axes, keepdims=True).astype(np.float32)
                tensors[out_name] = _quantize_float(out_f, so, zo, out_dtype)
                continue
            out = np.max(x.astype(np.float32), axis=axes, keepdims=True).astype(np.float32)
            tensors[out_name] = out
            continue

        if op == "GlobalLpPool":
            x = ins[0]
            if x.ndim < 3:
                raise ValueError("GlobalLpPool expects rank >= 3 tensors.")
            p = int(node.attrs.get("p", 2))
            if p <= 0:
                raise ValueError("GlobalLpPool p must be positive.")
            axes = tuple(range(2, x.ndim))
            if out_dtype in ("int8", "int16"):
                sa, za = _qparams(model, node.inputs[0])
                so, zo = _qparams(model, out_name)
                x_f = _dequantize_int(x, sa, za)
                out_f = np.sum(np.power(np.abs(x_f.astype(np.float32)), p), axis=axes, keepdims=True)
                out_f = np.power(out_f, 1.0 / float(p)).astype(np.float32)
                tensors[out_name] = _quantize_float(out_f, so, zo, out_dtype)
                continue
            out = np.sum(np.power(np.abs(x.astype(np.float32)), p), axis=axes, keepdims=True)
            out = np.power(out, 1.0 / float(p)).astype(np.float32)
            tensors[out_name] = out
            continue

        if op == "BatchNormalization":
            x = ins[0]
            scale, bias, mean, var = ins[1], ins[2], ins[3], ins[4]
            eps = float(node.attrs.get("epsilon", 1e-5))
            if x.ndim < 2:
                raise ValueError("BatchNormalization expects rank >= 2 input.")
            # Validation follows inference semantics for converted models.
            mean_used = mean
            var_used = var
            dim_ones = (1,) * (len(x.shape) - 2)
            s = scale.reshape(-1, *dim_ones)
            b = bias.reshape(-1, *dim_ones)
            m = mean_used.reshape(-1, *dim_ones)
            v = var_used.reshape(-1, *dim_ones)
            out_dtype_bn = model.tensors[out_name].dtype
            if out_dtype_bn in ("int8", "int16"):
                sx, zx = _qparams(model, node.inputs[0])
                so, zo = _qparams(model, out_name)
                x_f = _dequantize_int(x, sx, zx)
                y_f = s * (x_f - m) / np.sqrt(v + eps) + b
                tensors[out_name] = _quantize_float(y_f.astype(np.float32), so, zo, out_dtype_bn)
                continue
            y = s * (x.astype(np.float32) - m) / np.sqrt(v + eps) + b
            tensors[out_name] = y.astype(np.float32)
            continue

        if op == "InstanceNormalization":
            x = ins[0]
            scale = ins[1]
            bias = ins[2]
            eps = float(node.attrs.get("epsilon", 1e-5))
            if x.ndim < 3:
                raise ValueError("InstanceNormalization expects rank >= 3 input.")
            if out_dtype != "float32":
                raise ValueError("InstanceNormalization quantized mode is not supported.")
            axes = tuple(range(2, x.ndim))
            mean = np.mean(x, axis=axes, keepdims=True)
            var = np.var(x, axis=axes, keepdims=True)
            rs = (1,) * (x.ndim - 2)
            s = scale.reshape(1, -1, *rs)
            b = bias.reshape(1, -1, *rs)
            y = s * (x - mean) / np.sqrt(var + eps) + b
            tensors[out_name] = y.astype(np.float32)
            continue

        if op == "LRN":
            x = ins[0]
            if x.ndim < 3:
                raise ValueError("LRN expects rank >= 3 input.")
            if out_dtype != "float32":
                raise ValueError("LRN quantized mode is not supported.")
            size = int(node.attrs.get("size", 0))
            if size <= 0:
                raise ValueError("LRN requires positive size.")
            alpha = float(node.attrs.get("alpha", 1e-4))
            beta = float(node.attrs.get("beta", 0.75))
            bias = float(node.attrs.get("bias", 1.0))
            n, c = x.shape[0], x.shape[1]
            radius = size // 2
            y = np.empty_like(x, dtype=np.float32)
            x32 = x.astype(np.float32, copy=False)
            sq = x32 * x32
            for ni in range(n):
                for ch in range(c):
                    c0 = max(0, ch - radius)
                    c1 = min(c - 1, ch + size - radius - 1)
                    sq_sum = np.sum(sq[ni, c0 : c1 + 1, ...], axis=0)
                    norm = np.power(bias + (alpha / float(size)) * sq_sum, beta)
                    y[ni, ch, ...] = x32[ni, ch, ...] / norm
            tensors[out_name] = y
            continue

        if op == "Concat":
            axis = int(node.attrs.get("axis", 0))
            tensors[out_name] = np.concatenate(ins, axis=axis)
            continue

        if op == "Gather":
            axis = int(node.attrs.get("axis", 0))
            data = ins[0]
            indices = ins[1].astype(np.int64)
            tensors[out_name] = np.take(data, indices, axis=axis)
            continue

        if op == "GatherND":
            data = ins[0]
            indices = ins[1].astype(np.int64)
            if indices.ndim <= 0:
                raise ValueError("GatherND requires rank >= 1 indices.")
            batch_dims = int(node.attrs.get("batch_dims", 0))
            if batch_dims != 0:
                raise ValueError("GatherND currently supports batch_dims=0 only.")
            k = int(indices.shape[-1])
            if k < 0 or k > data.ndim:
                raise ValueError("GatherND indices last dim out of range.")
            out_shape = tuple(indices.shape[:-1]) + tuple(data.shape[k:])
            tail_size = int(np.prod(data.shape[k:])) if k < data.ndim else 1
            out = np.empty(out_shape, dtype=data.dtype)
            out_flat = out.reshape(-1, tail_size)
            idx_flat = indices.reshape(-1, k)
            for i in range(idx_flat.shape[0]):
                coord: list[int] = []
                for j in range(k):
                    v = int(idx_flat[i, j])
                    dim = int(data.shape[j])
                    if v < 0:
                        v += dim
                    if v < 0 or v >= dim:
                        raise ValueError("GatherND index out of range.")
                    coord.append(v)
                sub = data[tuple(coord)] if k > 0 else data
                out_flat[i, :] = np.asarray(sub, dtype=data.dtype).reshape(-1)
            tensors[out_name] = out
            continue

        if op == "GatherElements":
            axis = int(node.attrs.get("axis", 0))
            data = ins[0]
            indices = ins[1].astype(np.int64)
            if data.ndim != indices.ndim or data.ndim == 0:
                raise ValueError("GatherElements requires equal rank >= 1.")
            if axis < 0:
                axis += data.ndim
            if axis < 0 or axis >= data.ndim:
                raise ValueError("GatherElements axis out of range.")
            axis_dim = data.shape[axis]
            adj = np.where(indices < 0, indices + axis_dim, indices)
            if np.any(adj < 0) or np.any(adj >= axis_dim):
                raise ValueError("GatherElements index out of range.")
            tensors[out_name] = np.take_along_axis(data, adj, axis=axis)
            continue

        if op == "ReverseSequence":
            data = ins[0]
            seq = ins[1].astype(np.int64).reshape(-1)
            if data.ndim < 2:
                raise ValueError("ReverseSequence requires rank >= 2.")
            batch_axis = int(node.attrs.get("batch_axis", 1))
            time_axis = int(node.attrs.get("time_axis", 0))
            if batch_axis < 0:
                batch_axis += data.ndim
            if time_axis < 0:
                time_axis += data.ndim
            if batch_axis < 0 or batch_axis >= data.ndim:
                raise ValueError("ReverseSequence batch_axis out of range.")
            if time_axis < 0 or time_axis >= data.ndim:
                raise ValueError("ReverseSequence time_axis out of range.")
            if batch_axis == time_axis:
                raise ValueError("ReverseSequence batch_axis and time_axis must differ.")
            batch_dim = int(data.shape[batch_axis])
            time_dim = int(data.shape[time_axis])
            if seq.size != batch_dim:
                raise ValueError("ReverseSequence sequence_lens size mismatch.")

            perm = [time_axis, batch_axis] + [i for i in range(data.ndim) if i not in (time_axis, batch_axis)]
            inv_perm = [0] * data.ndim
            for i, p in enumerate(perm):
                inv_perm[p] = i
            data_tb = np.transpose(data, axes=perm)
            out_tb = np.array(data_tb, copy=True)
            for b in range(batch_dim):
                slen = int(seq[b])
                if slen < 0:
                    slen = 0
                if slen > time_dim:
                    slen = time_dim
                if slen > 1:
                    out_tb[:slen, b, ...] = data_tb[:slen, b, ...][::-1, ...]
            tensors[out_name] = np.transpose(out_tb, axes=inv_perm)
            continue

        if op == "Det":
            data = ins[0]
            if out_dtype in ("int8", "int16"):
                raise ValueError("Det quantized mode is not supported.")
            if data.ndim != 2:
                raise ValueError("Det currently supports 2D input only.")
            if data.shape[0] != data.shape[1]:
                raise ValueError("Det requires square matrix.")
            tensors[out_name] = np.array(np.linalg.det(data.astype(np.float32)), dtype=np.float32)
            continue

        if op == "Scatter":
            reduction = node.attrs.get("reduction", "none")
            if isinstance(reduction, bytes):
                reduction = reduction.decode("utf-8", errors="ignore")
            if str(reduction).strip().lower() != "none":
                raise ValueError("Scatter does not support reduction; use ScatterElements/ScatterND.")

        if op in ("ScatterElements", "Scatter"):
            data = ins[0]
            indices = ins[1].astype(np.int64)
            updates = ins[2]
            if data.ndim <= 0 or indices.ndim != data.ndim or updates.ndim != data.ndim:
                raise ValueError("ScatterElements requires same rank >= 1.")
            if indices.shape != updates.shape:
                raise ValueError("ScatterElements requires indices/updates same shape.")
            axis = int(node.attrs.get("axis", 0))
            if axis < 0:
                axis += data.ndim
            if axis < 0 or axis >= data.ndim:
                raise ValueError("ScatterElements axis out of range.")
            for dim_i in range(data.ndim):
                if indices.shape[dim_i] > data.shape[dim_i]:
                    raise ValueError("ScatterElements indices shape exceeds data shape.")
            if updates.dtype != data.dtype:
                updates = updates.astype(data.dtype)
            reduction = node.attrs.get("reduction", "none")
            if isinstance(reduction, bytes):
                reduction = reduction.decode("utf-8", errors="ignore")
            reduction = str(reduction).strip().lower()
            if reduction not in ("none", "add", "mul", "max", "min"):
                raise ValueError("ScatterElements reduction must be none/add/mul/max/min.")
            if data.dtype == np.bool_ and reduction != "none":
                raise ValueError("ScatterElements bool dtype supports reduction=none only.")
            out = np.array(data, copy=True)
            axis_dim = data.shape[axis]
            adj = np.where(indices < 0, indices + axis_dim, indices)
            if np.any(adj < 0) or np.any(adj >= axis_dim):
                raise ValueError("ScatterElements index out of range.")
            for idx_t in np.ndindex(indices.shape):
                dst = list(idx_t)
                dst[axis] = int(adj[idx_t])
                if reduction == "none":
                    out[tuple(dst)] = updates[idx_t]
                elif reduction == "add":
                    out[tuple(dst)] = out[tuple(dst)] + updates[idx_t]
                elif reduction == "mul":
                    out[tuple(dst)] = out[tuple(dst)] * updates[idx_t]
                elif reduction == "max":
                    out[tuple(dst)] = np.maximum(out[tuple(dst)], updates[idx_t])
                else:
                    out[tuple(dst)] = np.minimum(out[tuple(dst)], updates[idx_t])
            tensors[out_name] = out
            continue

        if op == "ScatterND":
            data = ins[0]
            indices = ins[1].astype(np.int64)
            updates = ins[2]
            if data.ndim <= 0 or indices.ndim <= 0:
                raise ValueError("ScatterND requires non-scalar data/indices.")
            k = int(indices.shape[-1])
            if k < 0 or k > data.ndim:
                raise ValueError("ScatterND indices last dim out of range.")
            expected_updates = tuple(indices.shape[:-1]) + tuple(data.shape[k:])
            if tuple(updates.shape) != expected_updates:
                raise ValueError("ScatterND updates shape mismatch.")
            if updates.dtype != data.dtype:
                updates = updates.astype(data.dtype)
            reduction = node.attrs.get("reduction", "none")
            if isinstance(reduction, bytes):
                reduction = reduction.decode("utf-8", errors="ignore")
            reduction = str(reduction).strip().lower()
            if reduction not in ("none", "add", "mul", "max", "min"):
                raise ValueError("ScatterND reduction must be none/add/mul/max/min.")
            if data.dtype == np.bool_ and reduction != "none":
                raise ValueError("ScatterND bool dtype supports reduction=none only.")
            out = np.array(data, copy=True)
            tail_size = int(np.prod(data.shape[k:])) if k < data.ndim else 1
            tuple_count = int(np.prod(indices.shape[:-1])) if indices.ndim > 1 else 1
            idx_flat = indices.reshape(tuple_count, k)
            upd_flat = updates.reshape(tuple_count, tail_size)
            strides = [1] * data.ndim
            acc = 1
            for i in range(data.ndim - 1, -1, -1):
                strides[i] = acc
                acc *= int(data.shape[i])
            out_flat = out.reshape(-1)
            for i in range(tuple_count):
                base = 0
                for j in range(k):
                    v = int(idx_flat[i, j])
                    dim = int(data.shape[j])
                    if v < 0:
                        v += dim
                    if v < 0 or v >= dim:
                        raise ValueError("ScatterND index out of range.")
                    base += v * strides[j]
                if reduction == "none":
                    out_flat[base : base + tail_size] = upd_flat[i]
                elif reduction == "add":
                    out_flat[base : base + tail_size] += upd_flat[i]
                elif reduction == "mul":
                    out_flat[base : base + tail_size] *= upd_flat[i]
                elif reduction == "max":
                    out_flat[base : base + tail_size] = np.maximum(out_flat[base : base + tail_size], upd_flat[i])
                else:
                    out_flat[base : base + tail_size] = np.minimum(out_flat[base : base + tail_size], upd_flat[i])
            tensors[out_name] = out
            continue

        if op == "NonZero":
            data = ins[0]
            idx = np.array(np.nonzero(data), dtype=np.int64)
            expected_shape = tuple(int(v) for v in model.tensors[out_name].shape)
            if tuple(idx.shape) != expected_shape:
                raise ValueError("NonZero output shape mismatch.")
            if out_dtype == "int32":
                tensors[out_name] = idx.astype(np.int32)
            else:
                tensors[out_name] = idx.astype(np.int64)
            continue

        if op == "Where":
            cond = ins[0]
            x = ins[1]
            y = ins[2]
            cond_mask = cond if cond.dtype == np.bool_ else (cond != 0)
            out = np.where(cond_mask, x, y)
            out_dtype = model.tensors[out_name].dtype
            if out_dtype == "float32":
                out = out.astype(np.float32)
            elif out_dtype == "int8":
                out = out.astype(np.int8)
            elif out_dtype == "int16":
                out = out.astype(np.int16)
            tensors[out_name] = out
            continue

        if op == "Transpose":
            perm = node.attrs.get("perm")
            if perm is None:
                perm = list(reversed(range(ins[0].ndim)))
            tensors[out_name] = np.transpose(ins[0], axes=perm)
            continue

        if op == "SpaceToDepth":
            x = ins[0]
            if x.ndim != 4:
                raise ValueError("SpaceToDepth expects 4D tensors.")
            block = int(node.attrs.get("blocksize", 0))
            if block <= 0:
                raise ValueError("SpaceToDepth requires positive blocksize.")
            n, c, h, w_in = x.shape
            if h % block != 0 or w_in % block != 0:
                raise ValueError("SpaceToDepth shape mismatch.")
            out = (
                x.reshape(n, c, h // block, block, w_in // block, block)
                .transpose(0, 1, 3, 5, 2, 4)
                .reshape(n, c * block * block, h // block, w_in // block)
            )
            tensors[out_name] = out
            continue

        if op == "DepthToSpace":
            x = ins[0]
            if x.ndim != 4:
                raise ValueError("DepthToSpace expects 4D tensors.")
            block = int(node.attrs.get("blocksize", 0))
            if block <= 0:
                raise ValueError("DepthToSpace requires positive blocksize.")
            mode = node.attrs.get("mode", "DCR")
            if isinstance(mode, bytes):
                mode = mode.decode("utf-8", errors="ignore")
            mode = str(mode).upper()
            n, c, h, w_in = x.shape
            if c % (block * block) != 0:
                raise ValueError("DepthToSpace shape mismatch.")
            out_c = c // (block * block)
            if mode == "DCR":
                out = (
                    x.reshape(n, out_c, block, block, h, w_in)
                    .transpose(0, 1, 4, 2, 5, 3)
                    .reshape(n, out_c, h * block, w_in * block)
                )
            elif mode == "CRD":
                out = (
                    x.reshape(n, block, block, out_c, h, w_in)
                    .transpose(0, 3, 4, 1, 5, 2)
                    .reshape(n, out_c, h * block, w_in * block)
                )
            else:
                raise ValueError("DepthToSpace mode must be DCR or CRD.")
            tensors[out_name] = out
            continue

        if op == "Pad":
            pads = node.attrs.get("pads")
            if pads is None and len(node.inputs) >= 2:
                pads = _const_ints(tensors, node.inputs[1])
            if pads is None:
                raise ValueError("Pad requires pads.")
            value = float(node.attrs.get("value", 0.0))
            rank = ins[0].ndim
            if len(pads) != rank * 2:
                raise ValueError("Pad pads length mismatch.")
            pad_begin = pads[:rank]
            pad_end = pads[rank:]
            pad_width = [(pad_begin[i], pad_end[i]) for i in range(rank)]
            tensors[out_name] = np.pad(ins[0], pad_width, mode="constant", constant_values=value)
            continue

        if op == "Slice":
            starts = node.attrs.get("starts")
            ends = node.attrs.get("ends")
            axes = node.attrs.get("axes")
            steps = node.attrs.get("steps")
            if starts is None and len(node.inputs) >= 2:
                starts = _const_ints(tensors, node.inputs[1])
            if ends is None and len(node.inputs) >= 3:
                ends = _const_ints(tensors, node.inputs[2])
            if axes is None and len(node.inputs) >= 4:
                axes = _const_ints(tensors, node.inputs[3])
            if steps is None and len(node.inputs) >= 5:
                steps = _const_ints(tensors, node.inputs[4])
            if starts is None or ends is None:
                raise ValueError("Slice requires starts/ends.")
            if steps is not None and any(int(v) != 1 for v in steps):
                raise ValueError("Slice steps != 1 not supported.")
            data = ins[0]
            rank = data.ndim
            if axes is None:
                axes = list(range(rank))
            slices = [slice(None)] * rank
            for idx, axis in enumerate(axes):
                axis = int(axis)
                s = int(starts[idx])
                e = int(ends[idx])
                slices[axis] = slice(s, e, 1)
            tensors[out_name] = data[tuple(slices)]
            continue

        if op in (
            "ReduceMean",
            "ReduceSum",
            "ReduceLogSum",
            "ReduceLogSumExp",
            "ReduceMax",
            "ReduceMin",
            "ReduceProd",
            "ReduceL1",
            "ReduceL2",
            "ReduceSumSquare",
        ):
            axes = node.attrs.get("axes")
            if axes is None and len(node.inputs) >= 2:
                axes = _const_ints(tensors, node.inputs[1])
            keepdims = int(node.attrs.get("keepdims", 1))
            data = ins[0]
            q_out = out_dtype in ("int8", "int16")
            if q_out:
                si, zi = _qparams(model, node.inputs[0])
                so, zo = _qparams(model, out_name)
                data = _dequantize_int(data, si, zi)
            if axes is None:
                axes = list(range(data.ndim))
            axes = tuple(int(a) for a in axes)
            if op == "ReduceMean":
                out = np.mean(data, axis=axes, keepdims=bool(keepdims))
            elif op == "ReduceSum":
                out = np.sum(data, axis=axes, keepdims=bool(keepdims))
            elif op == "ReduceLogSum":
                out = np.log(np.sum(data, axis=axes, keepdims=bool(keepdims)))
            elif op == "ReduceLogSumExp":
                out = np.log(np.sum(np.exp(data), axis=axes, keepdims=bool(keepdims)))
            elif op == "ReduceMax":
                out = np.max(data, axis=axes, keepdims=bool(keepdims))
            elif op == "ReduceMin":
                out = np.min(data, axis=axes, keepdims=bool(keepdims))
            elif op == "ReduceProd":
                out = np.prod(data, axis=axes, keepdims=bool(keepdims))
            elif op == "ReduceL1":
                out = np.sum(np.abs(data), axis=axes, keepdims=bool(keepdims))
            elif op == "ReduceL2":
                out = np.sqrt(np.sum(np.square(data), axis=axes, keepdims=bool(keepdims)))
            else:
                out = np.sum(np.square(data), axis=axes, keepdims=bool(keepdims))
            if q_out:
                tensors[out_name] = _quantize_float(out.astype(np.float32), so, zo, out_dtype)
            else:
                tensors[out_name] = out
            continue

        if op == "Squeeze":
            axes = node.attrs.get("axes")
            if axes is None and len(node.inputs) >= 2:
                axes = _const_ints(tensors, node.inputs[1])
            data = ins[0]
            if axes is None:
                tensors[out_name] = np.squeeze(data)
            else:
                tensors[out_name] = np.squeeze(data, axis=tuple(int(a) for a in axes))
            continue

        if op == "Unsqueeze":
            data = ins[0]
            axes = node.attrs.get("axes")
            if axes is None and len(node.inputs) >= 2:
                axes = _const_ints(tensors, node.inputs[1])
            if axes is None:
                raise ValueError("Unsqueeze requires axes.")
            axes = sorted(int(a) for a in axes)
            out = data
            for axis in axes:
                out = np.expand_dims(out, axis=axis)
            tensors[out_name] = out
            continue

        raise ValueError(f"Validation: unsupported op {op}.")

    return tensors


def _run_reference_output(
    model_path: str,
    input_name: str,
    input_data: np.ndarray,
    *,
    allow_reference_fallback: bool = True,
) -> tuple[np.ndarray | None, str, str]:
    ort_reason = ""
    if ort is not None:
        try:
            sess = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
            outputs = sess.run(None, {input_name: input_data})
            if outputs:
                return np.array(outputs[0]), "onnxruntime", ""
            return None, "onnxruntime", "reference output missing"
        except Exception as exc:
            ort_reason = f"onnxruntime error: {exc}"
    else:
        ort_reason = "onnxruntime unavailable"

    if not allow_reference_fallback:
        return None, "onnxruntime", ort_reason or "onnxruntime output missing"

    if ReferenceEvaluator is not None:
        try:
            onnx_model = onnx.load(model_path)
            ref_eval = ReferenceEvaluator(onnx_model)
            outputs = ref_eval.run(None, {input_name: input_data})
            if outputs:
                return np.array(outputs[0]), "onnx.reference", ""
            return None, "onnx.reference", "reference output missing"
        except Exception as exc:
            if ort_reason:
                return None, "onnx.reference", f"{ort_reason}; reference eval error: {exc}"
            return None, "onnx.reference", f"reference eval error: {exc}"
    return None, "", ort_reason


def validate_model_consistency(
    model: ModelIR,
    model_path: str,
    *,
    source_path: str = "",
    header_path: str = "",
    seed: int = 0,
    max_input_elems: int = 200000,
    rtol: float = 1e-3,
    atol: float = 1e-4,
    int8_atol: float = 1.0,
    int16_atol: float = 1.0,
    allow_reference_fallback: bool = True,
) -> ValidationResult:
    if ort is None and ReferenceEvaluator is None:
        return ValidationResult(status="skipped", reason="onnxruntime/reference evaluator unavailable")
    if len(model.inputs) != 1 or len(model.outputs) != 1:
        return ValidationResult(status="skipped", reason="only single input/output supported")

    input_tensor = model.inputs[0]
    input_shape = list(input_tensor.shape)
    if any(dim <= 0 for dim in input_shape):
        return ValidationResult(status="skipped", reason="input shape unknown")
    input_elems = int(np.prod(input_shape)) if input_shape else 1
    if input_elems > max_input_elems:
        return ValidationResult(status="skipped", reason="input too large for validation")

    rng = np.random.default_rng(seed)
    if input_tensor.dtype == "float32":
        input_data = rng.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)
    elif input_tensor.dtype == "int8":
        input_data = rng.integers(-128, 128, size=input_shape, dtype=np.int8)
    elif input_tensor.dtype == "int16":
        input_data = rng.integers(-32768, 32768, size=input_shape, dtype=np.int16)
    else:
        return ValidationResult(status="skipped", reason="input dtype unsupported")

    ref, ref_engine, ref_reason = _run_reference_output(
        model_path,
        input_tensor.name,
        input_data,
        allow_reference_fallback=allow_reference_fallback,
    )
    if ref is None:
        return ValidationResult(status="skipped", reason=ref_reason)

    pred: np.ndarray
    pred_engine: str
    if source_path and header_path:
        c_run = run_generated_c_model(model, source_path, header_path, input_data)
        if not c_run.ok or c_run.output is None:
            return ValidationResult(status="skipped", reason=f"generated-c run skipped: {c_run.reason}")
        pred = c_run.output
        pred_engine = "generated-c"
    else:
        try:
            pred_tensors = _eval_model(model, {input_tensor.name: input_data})
        except Exception as exc:
            return ValidationResult(status="skipped", reason=f"predict eval error: {exc}")
        out_name = model.outputs[0].name
        if out_name not in pred_tensors:
            return ValidationResult(status="skipped", reason="predict output missing")
        pred = pred_tensors[out_name]
        pred_engine = "python-eval"

    if model.outputs[0].dtype in ("uint8", "int8", "int16"):
        if pred.shape != ref.shape:
            return ValidationResult(status="failed", reason="output shape mismatch", engine=f"{pred_engine} vs {ref_engine}")
        pred_i64 = pred.astype(np.int64)
        ref_i64 = ref.astype(np.int64)
        diff = np.abs(pred_i64 - ref_i64)
        max_diff = float(np.max(diff)) if diff.size else 0.0
        tol = float(int8_atol if model.outputs[0].dtype in ("uint8", "int8") else int16_atol)
        if max_diff > tol:
            return ValidationResult(
                status="failed",
                reason="output mismatch",
                max_abs=max_diff,
                engine=f"{pred_engine} vs {ref_engine}",
            )
        return ValidationResult(
            status="passed",
            max_abs=max_diff,
            engine=f"{pred_engine} vs {ref_engine}",
        )

    pred_f = pred.astype(np.float32)
    ref_f = np.array(ref, dtype=np.float32)
    if pred_f.shape != ref_f.shape:
        return ValidationResult(status="failed", reason="output shape mismatch", engine=f"{pred_engine} vs {ref_engine}")
    if not np.allclose(pred_f, ref_f, rtol=rtol, atol=atol):
        abs_err = np.max(np.abs(pred_f - ref_f))
        rel_err = np.max(np.abs(pred_f - ref_f) / (np.abs(ref_f) + 1e-8))
        return ValidationResult(
            status="failed",
            reason="output mismatch",
            max_abs=float(abs_err),
            max_rel=float(rel_err),
            engine=f"{pred_engine} vs {ref_engine}",
        )
    abs_err = float(np.max(np.abs(pred_f - ref_f))) if pred_f.size else 0.0
    rel_err = float(np.max(np.abs(pred_f - ref_f) / (np.abs(ref_f) + 1e-8))) if pred_f.size else 0.0
    return ValidationResult(
        status="passed",
        max_abs=abs_err,
        max_rel=rel_err,
        engine=f"{pred_engine} vs {ref_engine}",
    )
