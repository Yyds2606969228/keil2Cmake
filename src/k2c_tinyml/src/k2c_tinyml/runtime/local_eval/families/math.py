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

def handle_math_family(
    model: ModelIR,
    node,
    tensors: dict[str, np.ndarray],
    ins: list[np.ndarray | None],
    out_name: str,
    out_dtype: str,
) -> bool:
    op = node.op_type
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
        return True
    

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
        return True
    

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
        return True
    

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
        return True
    

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
        return True
    

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
                return True
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
        return True
    

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
        return True
    

    if op == "Identity":
        tensors[out_name] = ins[0].copy()
        return True
    

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
        return True
    

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
        return True
    

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
        return True
    

    if op == "Gemm":
        a, b = ins[0], ins[1]
        c = ins[2] if len(ins) >= 3 else None
        trans_a = int(node.attrs.get("transA", 0))
        trans_b = int(node.attrs.get("transB", 0))
        if trans_a not in (0, 1) or trans_b not in (0, 1):
            raise ValueError("Gemm transA/transB must be 0 or 1.")
        alpha = float(node.attrs.get("alpha", 1.0))
        beta = float(node.attrs.get("beta", 1.0))
    
        def _gemm_broadcast_c(c_arr: np.ndarray, m_dim: int, n_dim: int) -> np.ndarray:
            if c_arr.ndim == 0:
                return c_arr.reshape(())
            if c_arr.ndim == 1:
                if c_arr.shape[0] == n_dim:
                    return c_arr.reshape(1, n_dim)
                if c_arr.shape[0] == m_dim:
                    return c_arr.reshape(m_dim, 1)
                if c_arr.shape[0] == m_dim * n_dim:
                    return c_arr.reshape(m_dim, n_dim)
                raise ValueError("Gemm C shape is not broadcastable.")
            if c_arr.ndim == 2:
                return c_arr
            raise ValueError("Gemm C rank > 2 is not supported.")
    
        if out_dtype in ("int8", "int16"):
            sa, za = _qparams(model, node.inputs[0])
            sb, zb = _qparams(model, node.inputs[1])
            so, zo = _qparams(model, out_name)
            ra = _dequantize_int(a, sa, za)
            rb = _dequantize_int(b, sb, zb)
            ra_m = ra.T if trans_a == 1 else ra
            rb_m = rb.T if trans_b == 1 else rb
            out = np.matmul(ra_m, rb_m)
            out = alpha * out
            if c is not None:
                c_dtype = _tensor_dtype(model.tensors[node.inputs[2]])
                c_arr = _gemm_broadcast_c(c, out.shape[0], out.shape[1])
                if c_dtype == "float32":
                    out = out + beta * c_arr.astype(np.float32)
                elif c_dtype in ("int32", "int64"):
                    out = out + beta * (c_arr.astype(np.float32) * (sa * sb))
                else:
                    sc, zc = _qparams(model, node.inputs[2])
                    out = out + beta * _dequantize_int(c_arr, sc, zc)
            tensors[out_name] = _quantize_float(out, so, zo, out_dtype)
        else:
            af = a.astype(np.float32, copy=False)
            bf = b.astype(np.float32, copy=False)
            af_m = af.T if trans_a == 1 else af
            bf_m = bf.T if trans_b == 1 else bf
            out = np.matmul(af_m, bf_m)
            out = alpha * out
            if c is not None:
                c_arr = _gemm_broadcast_c(c, out.shape[0], out.shape[1]).astype(np.float32)
                out = out + beta * c_arr
            tensors[out_name] = out.astype(np.float32)
        return True
    

    return False
