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

def handle_index_family(
    model: ModelIR,
    node,
    tensors: dict[str, np.ndarray],
    ins: list[np.ndarray | None],
    out_name: str,
    out_dtype: str,
) -> bool:
    op = node.op_type
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
        return True
    

    if op == "Gather":
        axis = int(node.attrs.get("axis", 0))
        data = ins[0]
        indices = ins[1].astype(np.int64)
        tensors[out_name] = np.take(data, indices, axis=axis)
        return True
    

    if op == "GatherND":
        data = ins[0]
        indices = ins[1].astype(np.int64)
        if indices.ndim <= 0:
            raise ValueError("GatherND requires rank >= 1 indices.")
        batch_dims = int(node.attrs.get("batch_dims", 0))
        if batch_dims < 0:
            batch_dims += min(data.ndim, indices.ndim - 1)
        if batch_dims < 0 or batch_dims >= data.ndim or batch_dims >= indices.ndim:
            raise ValueError("GatherND batch_dims out of range.")
        k = int(indices.shape[-1])
        if k < 0 or k > (data.ndim - batch_dims):
            raise ValueError("GatherND indices last dim out of range.")
        idx_suffix_rank = indices.ndim - batch_dims - 1
        tail_rank = data.ndim - batch_dims - k
        out_shape = tuple(data.shape[:batch_dims]) + tuple(indices.shape[batch_dims:-1]) + tuple(
            data.shape[batch_dims + k :]
        )
        out = np.empty(out_shape, dtype=data.dtype)
        for out_idx in np.ndindex(out_shape if out_shape else (1,)):
            out_idx_eff = () if out_shape == () else tuple(int(v) for v in out_idx)
            batch_coord = list(out_idx_eff[:batch_dims])
            idx_suffix_coord = list(out_idx_eff[batch_dims : batch_dims + idx_suffix_rank])
            tail_coord = list(out_idx_eff[batch_dims + idx_suffix_rank :])
            data_coord = list(batch_coord)
            for j in range(k):
                idx_coord = tuple(batch_coord + idx_suffix_coord + [j])
                v = int(indices[idx_coord])
                dim = int(data.shape[batch_dims + j])
                if v < 0:
                    v += dim
                if v < 0 or v >= dim:
                    raise ValueError("GatherND index out of range.")
                data_coord.append(v)
            if tail_rank > 0:
                data_coord.extend(tail_coord)
            value = data[tuple(data_coord)]
            if out_shape == ():
                out = np.array(value, dtype=data.dtype).reshape(())
            else:
                out[out_idx_eff] = value
        tensors[out_name] = out
        return True
    

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
        return True
    

    if op == "OneHot":
        indices = ins[0].astype(np.int64, copy=False)
        depth_arr = ins[1].reshape(-1)
        if depth_arr.size != 1:
            raise ValueError("OneHot depth must be scalar.")
        depth = int(depth_arr[0])
        if depth <= 0:
            raise ValueError("OneHot depth must be positive.")
        values = ins[2].reshape(-1)
        if values.size != 2:
            raise ValueError("OneHot values must contain [off, on].")
        out_dtype_np = {
            "float32": np.float32,
            "int8": np.int8,
            "int16": np.int16,
            "int32": np.int32,
            "int64": np.int64,
            "bool": np.bool_,
        }.get(out_dtype)
        if out_dtype_np is None:
            raise ValueError("OneHot output dtype is unsupported.")
        out_rank = indices.ndim + 1
        axis = int(node.attrs.get("axis", -1))
        if axis < 0:
            axis += out_rank
        if axis < 0 or axis >= out_rank:
            raise ValueError("OneHot axis out of range.")
        out_shape = list(indices.shape)
        out_shape.insert(axis, depth)
        out = np.full(out_shape, values[0], dtype=out_dtype_np)
        if indices.ndim == 0:
            cls = int(indices.reshape(()))
            cls %= depth
            if cls < 0:
                cls += depth
            out[(cls,)] = values[1]
        else:
            for idx_t in np.ndindex(indices.shape):
                cls = int(indices[idx_t])
                cls %= depth
                if cls < 0:
                    cls += depth
                dst = list(idx_t)
                dst.insert(axis, cls)
                out[tuple(dst)] = values[1]
        tensors[out_name] = out.astype(out_dtype_np, copy=False)
        return True
    

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
        return True
    

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
        return True
    

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
        return True
    

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
        return True
    

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
        return True
    

    return False
