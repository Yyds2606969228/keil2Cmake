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

def handle_shape_family(
    model: ModelIR,
    node,
    tensors: dict[str, np.ndarray],
    ins: list[np.ndarray | None],
    out_name: str,
    out_dtype: str,
) -> bool:
    op = node.op_type
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
        return True
    

    if op == "Shape":
        shape_arr = np.array(ins[0].shape, dtype=np.int64)
        if out_dtype == "int32":
            tensors[out_name] = shape_arr.astype(np.int32)
        elif out_dtype == "float32":
            tensors[out_name] = shape_arr.astype(np.float32)
        else:
            tensors[out_name] = shape_arr.astype(np.int64)
        return True
    

    if op == "Size":
        size_v = int(np.prod(ins[0].shape, dtype=np.int64))
        if out_dtype == "int32":
            tensors[out_name] = np.array(size_v, dtype=np.int32)
        elif out_dtype == "float32":
            tensors[out_name] = np.array(float(size_v), dtype=np.float32)
        else:
            tensors[out_name] = np.array(size_v, dtype=np.int64)
        return True
    

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
        return True
    

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
        return True
    

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
            return True
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
            return True
        raise ValueError("Range output dtype unsupported.")
    

    if op == "Reshape":
        shape_vals = _const_ints(tensors, node.inputs[1])
        tensors[out_name] = _reshape_like_onnx(ins[0], shape_vals)
        return True
    

    if op == "Expand":
        target = tuple(int(v) for v in model.tensors[out_name].shape)
        tensors[out_name] = np.broadcast_to(ins[0], target).copy()
        return True
    

    if op == "CumSum":
        if len(ins) < 2:
            raise ValueError("CumSum expects data and axis.")
        x = ins[0]
        axis_arr = ins[1].reshape(-1)
        if axis_arr.size != 1:
            raise ValueError("CumSum axis input must be scalar.")
        axis = int(axis_arr[0])
        if axis < 0:
            axis += x.ndim
        if axis < 0 or axis >= x.ndim:
            raise ValueError("CumSum axis out of range.")
        exclusive = int(node.attrs.get("exclusive", 0))
        reverse = int(node.attrs.get("reverse", 0))
        if exclusive not in (0, 1) or reverse not in (0, 1):
            raise ValueError("CumSum exclusive/reverse must be 0 or 1.")
        x_work = np.flip(x, axis=axis) if reverse == 1 else x
        if x.dtype == np.float32:
            out = np.cumsum(x_work.astype(np.float32), axis=axis, dtype=np.float32)
            if exclusive == 1:
                out = out - x_work.astype(np.float32)
            if reverse == 1:
                out = np.flip(out, axis=axis)
            tensors[out_name] = out.astype(np.float32)
            return True
        acc = np.cumsum(x_work.astype(np.int64), axis=axis, dtype=np.int64)
        if exclusive == 1:
            acc = acc - x_work.astype(np.int64)
        if reverse == 1:
            acc = np.flip(acc, axis=axis)
        if out_dtype == "int8":
            tensors[out_name] = np.clip(acc, -128, 127).astype(np.int8)
        elif out_dtype == "int16":
            tensors[out_name] = np.clip(acc, -32768, 32767).astype(np.int16)
        elif out_dtype == "int32":
            tensors[out_name] = np.clip(acc, -2147483648, 2147483647).astype(np.int32)
        else:
            tensors[out_name] = acc.astype(np.int64)
        return True
    

    if op == "Tile":
        reps = _const_ints(tensors, node.inputs[1])
        tensors[out_name] = np.tile(ins[0], tuple(int(v) for v in reps))
        return True
    

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
        return True
    

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
        return True
    

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
        return True
    

    if op == "Concat":
        axis = int(node.attrs.get("axis", 0))
        tensors[out_name] = np.concatenate(ins, axis=axis)
        return True
    

    if op == "Transpose":
        perm = node.attrs.get("perm")
        if perm is None:
            perm = list(reversed(range(ins[0].ndim)))
        tensors[out_name] = np.transpose(ins[0], axes=perm)
        return True
    

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
        return True
    

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
        return True
    

    if op == "Pad":
        pads = node.attrs.get("pads")
        if pads is None and len(node.inputs) >= 2:
            pads = _const_ints(tensors, node.inputs[1])
        if pads is None:
            raise ValueError("Pad requires pads.")
        mode = node.attrs.get("mode", "constant")
        if isinstance(mode, bytes):
            mode = mode.decode("utf-8", errors="ignore")
        mode = str(mode).lower()
        value = float(node.attrs.get("value", 0.0))
        if len(node.inputs) >= 3 and node.inputs[2]:
            v_arr = tensors[node.inputs[2]].reshape(-1)
            if v_arr.size > 0:
                value = float(v_arr[0])
        rank = ins[0].ndim
        if len(pads) != rank * 2:
            raise ValueError("Pad pads length mismatch.")
        pad_begin = pads[:rank]
        pad_end = pads[rank:]
        pad_width = [(pad_begin[i], pad_end[i]) for i in range(rank)]
        if mode == "constant":
            if out_dtype in ("int8", "int16"):
                so, zo = _qparams(model, out_name)
                qmin, qmax = (-128, 127) if out_dtype == "int8" else (-32768, 32767)
                qv = int(round(value / so) + zo)
                qv = min(max(qv, qmin), qmax)
                tensors[out_name] = np.pad(ins[0], pad_width, mode="constant", constant_values=qv)
            else:
                tensors[out_name] = np.pad(ins[0], pad_width, mode="constant", constant_values=value)
        elif mode == "reflect":
            tensors[out_name] = np.pad(ins[0], pad_width, mode="reflect")
        elif mode == "edge":
            tensors[out_name] = np.pad(ins[0], pad_width, mode="edge")
        else:
            raise ValueError("Pad mode must be constant/reflect/edge.")
        return True
    

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
        data = ins[0]
        rank = data.ndim
        if axes is None:
            axes = list(range(rank))
        if steps is None:
            steps = [1] * len(axes)
        if len(axes) != len(starts) or len(axes) != len(ends) or len(axes) != len(steps):
            raise ValueError("Slice axes/starts/ends/steps length mismatch.")
        slices = [slice(None)] * rank
        for idx, axis_v in enumerate(axes):
            axis = int(axis_v)
            if axis < 0:
                axis += rank
            if axis < 0 or axis >= rank:
                raise ValueError("Slice axis out of range.")
            s = int(starts[idx])
            e = int(ends[idx])
            st = int(steps[idx])
            if st == 0:
                raise ValueError("Slice step must be non-zero.")
            slices[axis] = slice(s, e, st)
        tensors[out_name] = data[tuple(slices)]
        return True
    

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
        return True
    

    if op == "Squeeze":
        axes = node.attrs.get("axes")
        if axes is None and len(node.inputs) >= 2:
            axes = _const_ints(tensors, node.inputs[1])
        data = ins[0]
        if axes is None:
            tensors[out_name] = np.squeeze(data)
        else:
            tensors[out_name] = np.squeeze(data, axis=tuple(int(a) for a in axes))
        return True
    

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
        return True
    

    return False
