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

def handle_nn_family(
    model: ModelIR,
    node,
    tensors: dict[str, np.ndarray],
    ins: list[np.ndarray | None],
    out_name: str,
    out_dtype: str,
) -> bool:
    op = node.op_type
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
        return True
    

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
        return True
    

    if op == "MeanVarianceNormalization":
        x = ins[0]
        if x.ndim <= 0:
            raise ValueError("MeanVarianceNormalization expects rank >= 1.")
        axes = node.attrs.get("axes", [0, 2, 3])
        if isinstance(axes, (list, tuple)):
            axes_raw = [int(v) for v in axes]
        else:
            axes_raw = [int(axes)]
        norm_axes: list[int] = []
        seen_axes: set[int] = set()
        for axis in axes_raw:
            ax = int(axis)
            if ax < 0:
                ax += x.ndim
            if ax < 0 or ax >= x.ndim:
                raise ValueError("MeanVarianceNormalization axis out of range.")
            if ax in seen_axes:
                continue
            seen_axes.add(ax)
            norm_axes.append(ax)
        axes_tuple = tuple(norm_axes)
        eps = float(node.attrs.get("epsilon", 1e-12))
        if out_dtype in ("int8", "int16"):
            sx, zx = _qparams(model, node.inputs[0])
            so, zo = _qparams(model, out_name)
            x_f = _dequantize_int(x, sx, zx)
            mean = np.mean(x_f, axis=axes_tuple, keepdims=True)
            var = np.mean(np.square(x_f - mean), axis=axes_tuple, keepdims=True)
            y_f = (x_f - mean) / np.sqrt(var + eps)
            tensors[out_name] = _quantize_float(y_f.astype(np.float32), so, zo, out_dtype)
        else:
            x_f = x.astype(np.float32)
            mean = np.mean(x_f, axis=axes_tuple, keepdims=True)
            var = np.mean(np.square(x_f - mean), axis=axes_tuple, keepdims=True)
            tensors[out_name] = ((x_f - mean) / np.sqrt(var + eps)).astype(np.float32)
        return True
    

    if op == "RNN":
        _eval_rnn_node(model, node, tensors)
        return True
    

    if op == "GRU":
        _eval_gru_node(model, node, tensors)
        return True
    

    if op == "LSTM":
        _eval_lstm_node(model, node, tensors)
        return True
    

    if op == "NegativeLogLikelihoodLoss":
        x = ins[0]
        target = ins[1].astype(np.int64, copy=False)
        weight = ins[2] if len(ins) >= 3 and node.inputs[2] else None
        if x.ndim < 2:
            raise ValueError("NegativeLogLikelihoodLoss expects input rank >= 2.")
        n_size = int(x.shape[0])
        c_size = int(x.shape[1])
        spatial_shape = tuple(int(v) for v in x.shape[2:])
        expected_target = (n_size, *spatial_shape)
        if tuple(int(v) for v in target.shape) != expected_target:
            raise ValueError("NegativeLogLikelihoodLoss target shape mismatch.")
        x_dtype = _tensor_dtype(model.tensors[node.inputs[0]])
        x_tensor = model.tensors[node.inputs[0]]
        if x_dtype in ("int8", "int16") and x_tensor.qscale is not None and x_tensor.qzero is not None:
            sx, zx = _qparams(model, node.inputs[0])
            x_f = _dequantize_int(x, sx, zx)
        else:
            x_f = x.astype(np.float32, copy=False)
        weight_f = None
        if weight is not None:
            w_dtype = _tensor_dtype(model.tensors[node.inputs[2]])
            w_tensor = model.tensors[node.inputs[2]]
            if w_dtype in ("int8", "int16") and w_tensor.qscale is not None and w_tensor.qzero is not None:
                sw, zw = _qparams(model, node.inputs[2])
                weight_f = _dequantize_int(weight, sw, zw)
            else:
                weight_f = weight.astype(np.float32, copy=False)
            if weight_f.ndim != 1 or int(weight_f.shape[0]) != c_size:
                raise ValueError("NegativeLogLikelihoodLoss weight shape mismatch.")
    
        reduction = node.attrs.get("reduction", "mean")
        if isinstance(reduction, bytes):
            reduction = reduction.decode("utf-8", errors="ignore")
        reduction = str(reduction).lower()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("NegativeLogLikelihoodLoss reduction must be none/mean/sum.")
        ignore_index = int(node.attrs.get("ignore_index", -100))
    
        inner = int(np.prod(spatial_shape, dtype=np.int64)) if spatial_shape else 1
        x_r = x_f.reshape(n_size, c_size, inner)
        t_r = target.reshape(n_size, inner)
        loss = np.zeros((n_size, inner), dtype=np.float32)
        total_loss = 0.0
        total_weight = 0.0
        for ni in range(n_size):
            for pi in range(inner):
                cls = int(t_r[ni, pi])
                if cls == ignore_index or cls < 0 or cls >= c_size:
                    continue
                v = -float(x_r[ni, cls, pi])
                if weight_f is not None:
                    wv = float(weight_f[cls])
                    v *= wv
                    total_weight += wv
                else:
                    total_weight += 1.0
                loss[ni, pi] = v
                total_loss += v
    
        if reduction == "none":
            tensors[out_name] = loss.reshape(expected_target).astype(np.float32)
        elif reduction == "sum":
            tensors[out_name] = np.array(total_loss, dtype=np.float32).reshape(())
        else:
            mean_v = (total_loss / total_weight) if total_weight > 0.0 else 0.0
            tensors[out_name] = np.array(mean_v, dtype=np.float32).reshape(())
        return True
    

    if op == "SoftmaxCrossEntropyLoss":
        x = ins[0]
        target = ins[1].astype(np.int64, copy=False)
        weight = ins[2] if len(ins) >= 3 and node.inputs[2] else None
        loss_name = node.outputs[0]
        logp_name = node.outputs[1] if len(node.outputs) == 2 and node.outputs[1] else None
        if x.ndim < 2:
            raise ValueError("SoftmaxCrossEntropyLoss expects logits rank >= 2.")
        n_size = int(x.shape[0])
        c_size = int(x.shape[1])
        spatial_shape = tuple(int(v) for v in x.shape[2:])
        expected_target = (n_size, *spatial_shape)
        if tuple(int(v) for v in target.shape) != expected_target:
            raise ValueError("SoftmaxCrossEntropyLoss target shape mismatch.")
        x_dtype = _tensor_dtype(model.tensors[node.inputs[0]])
        x_tensor = model.tensors[node.inputs[0]]
        if x_dtype in ("int8", "int16") and x_tensor.qscale is not None and x_tensor.qzero is not None:
            sx, zx = _qparams(model, node.inputs[0])
            x_f = _dequantize_int(x, sx, zx)
        else:
            x_f = x.astype(np.float32, copy=False)
    
        weight_f = None
        if weight is not None:
            w_dtype = _tensor_dtype(model.tensors[node.inputs[2]])
            w_tensor = model.tensors[node.inputs[2]]
            if w_dtype in ("int8", "int16") and w_tensor.qscale is not None and w_tensor.qzero is not None:
                sw, zw = _qparams(model, node.inputs[2])
                weight_f = _dequantize_int(weight, sw, zw)
            else:
                weight_f = weight.astype(np.float32, copy=False)
            if weight_f.ndim != 1 or int(weight_f.shape[0]) != c_size:
                raise ValueError("SoftmaxCrossEntropyLoss weight shape mismatch.")
    
        reduction = node.attrs.get("reduction", "mean")
        if isinstance(reduction, bytes):
            reduction = reduction.decode("utf-8", errors="ignore")
        reduction = str(reduction).lower()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("SoftmaxCrossEntropyLoss reduction must be none/mean/sum.")
        ignore_index = int(node.attrs.get("ignore_index", -100))
    
        inner = int(np.prod(spatial_shape, dtype=np.int64)) if spatial_shape else 1
        x_r = x_f.reshape(n_size, c_size, inner)
        t_r = target.reshape(n_size, inner)
        logp = np.empty_like(x_r, dtype=np.float32) if logp_name is not None else None
        loss = np.zeros((n_size, inner), dtype=np.float32)
        total_loss = 0.0
        total_weight = 0.0
        for ni in range(n_size):
            for pi in range(inner):
                logits = x_r[ni, :, pi].astype(np.float32, copy=False)
                max_v = float(np.max(logits))
                shifted = logits - max_v
                exp_sum = float(np.sum(np.exp(shifted)))
                log_sum = float(np.log(exp_sum))
                lp = logits - max_v - log_sum
                if logp is not None:
                    logp[ni, :, pi] = lp
                cls = int(t_r[ni, pi])
                if cls == ignore_index or cls < 0 or cls >= c_size:
                    continue
                sample = -float(lp[cls])
                if weight_f is not None:
                    wv = float(weight_f[cls])
                    sample *= wv
                    total_weight += wv
                else:
                    total_weight += 1.0
                loss[ni, pi] = sample
                total_loss += sample
    
        if reduction == "none":
            tensors[loss_name] = loss.reshape(expected_target).astype(np.float32)
        elif reduction == "sum":
            tensors[loss_name] = np.array(total_loss, dtype=np.float32).reshape(())
        else:
            mean_v = (total_loss / total_weight) if total_weight > 0.0 else 0.0
            tensors[loss_name] = np.array(mean_v, dtype=np.float32).reshape(())
        if logp is not None:
            tensors[logp_name] = logp.reshape(x.shape).astype(np.float32)
        return True
    

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
        return True
    

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
        return True
    

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
        return True
    

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
        return True
    

    if op == "ConvTranspose":
        x = ins[0]
        w = ins[1]
        b = ins[2] if len(ins) > 2 else None
        strides = list(node.attrs.get("strides", [1, 1]))
        pads = list(node.attrs.get("pads", [0, 0, 0, 0]))
        dilations = list(node.attrs.get("dilations", [1, 1]))
        output_padding = list(node.attrs.get("output_padding", [0, 0]))
        groups = int(node.attrs.get("group", 1))
        if groups <= 0:
            raise ValueError("ConvTranspose group must be positive.")
        if x.ndim != 4 or w.ndim != 4:
            raise ValueError("ConvTranspose expects 4D tensors (NCHW).")
        n, c_in, h, w_in = x.shape
        wc_in, c_out_per_group, k_h, k_w = w.shape
        if c_in % groups != 0:
            raise ValueError("ConvTranspose input channels must be divisible by group.")
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
        ic_per_group = c_in // groups
        out = np.zeros((n, c_out, out_h, out_w), dtype=np.float32)
        quant_mode = out_dtype in ("int8", "int16")
        if quant_mode:
            if _tensor_dtype(model.tensors[node.inputs[0]]) != out_dtype:
                raise ValueError("ConvTranspose quantized input dtype mismatch.")
            if _tensor_dtype(model.tensors[node.inputs[1]]) != out_dtype:
                raise ValueError("ConvTranspose quantized weight dtype mismatch.")
            sx, zx = _qparams(model, node.inputs[0])
            sw, zw = _qparams(model, node.inputs[1])
            so, zo = _qparams(model, out_name)
            x_f = _dequantize_int(x, sx, zx)
            w_f = _dequantize_int(w, sw, zw)
        else:
            x_f = x.astype(np.float32, copy=False)
            w_f = w.astype(np.float32, copy=False)
    
        if b is not None:
            b_dtype = _tensor_dtype(model.tensors[node.inputs[2]])
            if quant_mode:
                if b_dtype == "float32":
                    b_f = b.astype(np.float32)
                elif b_dtype in ("int32", "int64"):
                    b_f = b.astype(np.float32) * (sx * sw)
                elif b_dtype in ("int8", "int16"):
                    sb, zb = _qparams(model, node.inputs[2])
                    b_f = _dequantize_int(b, sb, zb)
                else:
                    raise ValueError("ConvTranspose quantized bias dtype is unsupported.")
            else:
                b_f = b.astype(np.float32)
            out += b_f.reshape(1, c_out, 1, 1)
        for ni in range(n):
            for ic in range(c_in):
                g = ic // ic_per_group
                oc0 = g * c_out_per_group
                oc1 = oc0 + c_out_per_group
                for ih in range(h):
                    for iw in range(w_in):
                        xv = float(x_f[ni, ic, ih, iw])
                        for kh in range(k_h):
                            for kw in range(k_w):
                                oh = ih * stride_h + kh * dil_h - pad_h0
                                ow = iw * stride_w + kw * dil_w - pad_w0
                                if 0 <= oh < out_h and 0 <= ow < out_w:
                                    out[ni, oc0:oc1, oh, ow] += xv * w_f[ic, :, kh, kw]
        if quant_mode:
            tensors[out_name] = _quantize_float(out, so, zo, out_dtype)
        else:
            tensors[out_name] = out.astype(np.float32)
        return True
    

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
            return True
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
        return True
    

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
            return True
        out = np.mean(x.astype(np.float32), axis=axes, keepdims=True).astype(np.float32)
        tensors[out_name] = out
        return True
    

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
            return True
        out = np.max(x.astype(np.float32), axis=axes, keepdims=True).astype(np.float32)
        tensors[out_name] = out
        return True
    

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
            return True
        out = np.sum(np.power(np.abs(x.astype(np.float32)), p), axis=axes, keepdims=True)
        out = np.power(out, 1.0 / float(p)).astype(np.float32)
        tensors[out_name] = out
        return True
    

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
            return True
        y = s * (x.astype(np.float32) - m) / np.sqrt(v + eps) + b
        tensors[out_name] = y.astype(np.float32)
        return True
    

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
        return True
    

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
        return True
    

    return False
