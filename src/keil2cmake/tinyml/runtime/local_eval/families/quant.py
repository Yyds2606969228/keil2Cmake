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

def handle_quant_family(
    model: ModelIR,
    node,
    tensors: dict[str, np.ndarray],
    ins: list[np.ndarray | None],
    out_name: str,
    out_dtype: str,
) -> bool:
    op = node.op_type
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
        return True
    

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
        return True
    

    if op == "DequantizeLinear":
        scale = _const_scalar(tensors, node.inputs[1])
        zero = 0
        if len(node.inputs) >= 3:
            zero_vals = _const_ints(tensors, node.inputs[2])
            if len(zero_vals) != 1:
                raise ValueError("DequantizeLinear supports scalar zero_point only.")
            zero = int(zero_vals[0])
        tensors[out_name] = _dequantize_int(ins[0], float(scale), int(zero))
        return True
    

    if op == "MatMulInteger":
        if len(ins) < 2:
            raise ValueError("MatMulInteger expects at least 2 inputs.")
        if ins[0] is None or ins[1] is None:
            raise ValueError("MatMulInteger expects valid A/B inputs.")
        a = ins[0].astype(np.int64)
        b = ins[1].astype(np.int64)
        a_zero = int(ins[2].reshape(-1)[0]) if len(ins) >= 3 and ins[2] is not None else 0
        b_zero = int(ins[3].reshape(-1)[0]) if len(ins) >= 4 and ins[3] is not None else 0
        out = np.matmul(a - a_zero, b - b_zero)
        if out_dtype == "int32":
            tensors[out_name] = out.astype(np.int32)
        elif out_dtype == "int64":
            tensors[out_name] = out.astype(np.int64)
        else:
            raise ValueError("MatMulInteger output dtype must be int32/int64.")
        return True
    

    if op == "QLinearMatMul":
        if len(ins) < 8:
            raise ValueError("QLinearMatMul expects 8 inputs.")
        if any(v is None for v in ins[:8]):
            raise ValueError("QLinearMatMul expects valid non-empty inputs.")
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
        return True
    

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
        return True
    

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
        return True
    

    return False
