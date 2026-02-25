# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Iterable

from onnx import TensorProto, numpy_helper

from .ir import NodeInfo, TensorInfo
from .onnx_loader_quant_utils import _dtype_from_tensorproto

def _shape_from_value(value_info) -> list[int]:
    dims = []
    tensor_type = value_info.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.dim_value > 0:
            dims.append(int(dim.dim_value))
        else:
            dims.append(-1)
    return dims


def _tensor_dtype(value_info) -> int:
    return int(value_info.type.tensor_type.elem_type)


def _is_initializer(name: str, initializer_names: set[str]) -> bool:
    return name in initializer_names


def _tensor_size(shape: Iterable[int]) -> int:
    size = 1
    for dim in shape:
        if dim <= 0:
            raise ValueError("Tensor shape contains unknown or invalid dimension.")
        size *= int(dim)
    return int(size)


def _shape_known(shape: list[int] | None) -> bool:
    if shape is None:
        return False
    return all(dim > 0 for dim in shape)


def _get_shape(tensors: dict[str, TensorInfo], name: str) -> list[int] | None:
    tensor = tensors.get(name)
    if tensor is None:
        return None
    shape = list(tensor.shape)
    if not _shape_known(shape):
        return None
    return shape


def _get_const_ints(tensors: dict[str, TensorInfo], name: str) -> list[int] | None:
    tensor = tensors.get(name)
    if tensor is None or tensor.data is None:
        return None
    return [int(v) for v in tensor.data]


def _get_const_floats(tensors: dict[str, TensorInfo], name: str) -> list[float] | None:
    tensor = tensors.get(name)
    if tensor is None or tensor.data is None:
        return None
    return [float(v) for v in tensor.data]


def _broadcast_shape(a: list[int] | None, b: list[int] | None) -> list[int] | None:
    if a is None or b is None:
        return None
    if not _shape_known(a) or not _shape_known(b):
        return None
    ra = len(a)
    rb = len(b)
    out_rev: list[int] = []
    for i in range(max(ra, rb)):
        dim_a = a[ra - 1 - i] if i < ra else 1
        dim_b = b[rb - 1 - i] if i < rb else 1
        if dim_a == dim_b or dim_a == 1 or dim_b == 1:
            out_rev.append(max(dim_a, dim_b))
        else:
            return None
    return list(reversed(out_rev))


def _infer_unary_shape(tensors: dict[str, TensorInfo], name: str) -> list[int] | None:
    return _get_shape(tensors, name)


def _infer_binary_broadcast_shape(
    tensors: dict[str, TensorInfo], a_name: str, b_name: str
) -> list[int] | None:
    return _broadcast_shape(_get_shape(tensors, a_name), _get_shape(tensors, b_name))


def _infer_ternary_broadcast_shape(
    tensors: dict[str, TensorInfo], a_name: str, b_name: str, c_name: str
) -> list[int] | None:
    ab = _broadcast_shape(_get_shape(tensors, a_name), _get_shape(tensors, b_name))
    if ab is None:
        return None
    return _broadcast_shape(ab, _get_shape(tensors, c_name))


def _infer_variadic_broadcast_shape(
    tensors: dict[str, TensorInfo], names: list[str]
) -> list[int] | None:
    if not names:
        return None
    out = _get_shape(tensors, names[0])
    for name in names[1:]:
        out = _broadcast_shape(out, _get_shape(tensors, name))
        if out is None:
            return None
    return out


def _infer_matmul_shape(
    tensors: dict[str, TensorInfo], a_name: str, b_name: str
) -> list[int] | None:
    a_shape = _get_shape(tensors, a_name)
    b_shape = _get_shape(tensors, b_name)
    if a_shape is None or b_shape is None:
        return None
    if len(a_shape) != 2 or len(b_shape) != 2:
        return None
    if a_shape[1] != b_shape[0]:
        return None
    return [a_shape[0], b_shape[1]]


def _einsum_equation(attrs: dict) -> str:
    eq = attrs.get("equation", "")
    if isinstance(eq, bytes):
        return eq.decode("utf-8", errors="ignore").replace(" ", "")
    return str(eq).replace(" ", "")


def _infer_einsum_shape(
    tensors: dict[str, TensorInfo], a_name: str, b_name: str, attrs: dict
) -> list[int] | None:
    a_shape = _get_shape(tensors, a_name)
    b_shape = _get_shape(tensors, b_name)
    if a_shape is None or b_shape is None:
        return None
    eq = _einsum_equation(attrs)
    if eq == "ij,jk->ik":
        if len(a_shape) != 2 or len(b_shape) != 2:
            return None
        if a_shape[1] != b_shape[0]:
            return None
        return [a_shape[0], b_shape[1]]
    if eq == "bij,bjk->bik":
        if len(a_shape) != 3 or len(b_shape) != 3:
            return None
        if a_shape[0] != b_shape[0] or a_shape[2] != b_shape[1]:
            return None
        return [a_shape[0], a_shape[1], b_shape[2]]
    if eq == "bij,jk->bik":
        if len(a_shape) != 3 or len(b_shape) != 2:
            return None
        if a_shape[2] != b_shape[0]:
            return None
        return [a_shape[0], a_shape[1], b_shape[1]]
    if eq == "ij,bjk->bik":
        if len(a_shape) != 2 or len(b_shape) != 3:
            return None
        if a_shape[1] != b_shape[1]:
            return None
        return [b_shape[0], a_shape[0], b_shape[2]]
    return None


def _infer_gemm_shape(
    tensors: dict[str, TensorInfo], a_name: str, b_name: str, attrs: dict
) -> list[int] | None:
    a_shape = _get_shape(tensors, a_name)
    b_shape = _get_shape(tensors, b_name)
    if a_shape is None or b_shape is None:
        return None
    if len(a_shape) != 2 or len(b_shape) != 2:
        return None
    trans_a = int(attrs.get("transA", 0))
    trans_b = int(attrs.get("transB", 0))
    if trans_a not in (0, 1) or trans_b not in (0, 1):
        return None
    if trans_a:
        a_shape = [a_shape[1], a_shape[0]]
    if trans_b:
        b_shape = [b_shape[1], b_shape[0]]
    if a_shape[1] != b_shape[0]:
        return None
    return [a_shape[0], b_shape[1]]


def _infer_flatten_shape(tensors: dict[str, TensorInfo], name: str, attrs: dict) -> list[int] | None:
    in_shape = _get_shape(tensors, name)
    if in_shape is None:
        return None
    axis = int(attrs.get("axis", 1))
    rank = len(in_shape)
    if axis < 0:
        axis += rank
    if axis < 0 or axis > rank:
        return None
    dim0 = 1
    for v in in_shape[:axis]:
        dim0 *= v
    dim1 = 1
    for v in in_shape[axis:]:
        dim1 *= v
    return [dim0, dim1]


def _infer_reshape_shape(
    tensors: dict[str, TensorInfo], data_name: str, shape_name: str
) -> list[int] | None:
    in_shape = _get_shape(tensors, data_name)
    shape_vals = _get_const_ints(tensors, shape_name)
    if in_shape is None or shape_vals is None:
        return None
    out: list[int] = []
    unknown = None
    known_product = 1
    for idx, dim in enumerate(shape_vals):
        dim = int(dim)
        if dim == 0:
            if idx >= len(in_shape):
                return None
            dim = in_shape[idx]
        if dim == -1:
            if unknown is not None:
                return None
            unknown = idx
            out.append(-1)
        else:
            if dim <= 0:
                return None
            out.append(dim)
            known_product *= dim
    if unknown is not None:
        in_size = 1
        for v in in_shape:
            in_size *= v
        if known_product == 0 or in_size % known_product != 0:
            return None
        out[unknown] = int(in_size / known_product)
    if not _shape_known(out):
        return None
    return out


def _conv_out_dim(in_dim: int, kernel: int, stride: int, pad0: int, pad1: int, dilation: int) -> int:
    return (in_dim + pad0 + pad1 - dilation * (kernel - 1) - 1) // stride + 1


def _infer_conv_shape(
    tensors: dict[str, TensorInfo], x_name: str, w_name: str, attrs: dict
) -> list[int] | None:
    x_shape = _get_shape(tensors, x_name)
    w_shape = _get_shape(tensors, w_name)
    if x_shape is None or w_shape is None:
        return None
    if len(x_shape) != 4 or len(w_shape) != 4:
        return None
    n, c_in, h, w_in = x_shape
    m, c_per_g, k_h, k_w = w_shape
    groups = int(attrs.get("group", 1))
    if groups <= 0 or c_per_g * groups != c_in:
        return None
    strides = attrs.get("strides", [1, 1])
    if len(strides) != 2:
        return None
    pads = attrs.get("pads", [0, 0, 0, 0])
    if len(pads) == 2:
        pads = [pads[0], pads[1], pads[0], pads[1]]
    if len(pads) != 4:
        return None
    dilations = attrs.get("dilations", [1, 1])
    if len(dilations) != 2:
        return None
    out_h = _conv_out_dim(h, k_h, int(strides[0]), int(pads[0]), int(pads[2]), int(dilations[0]))
    out_w = _conv_out_dim(w_in, k_w, int(strides[1]), int(pads[1]), int(pads[3]), int(dilations[1]))
    if out_h <= 0 or out_w <= 0:
        return None
    return [n, m, out_h, out_w]


def _infer_conv_transpose_shape(
    tensors: dict[str, TensorInfo], x_name: str, w_name: str, attrs: dict
) -> list[int] | None:
    x_shape = _get_shape(tensors, x_name)
    w_shape = _get_shape(tensors, w_name)
    if x_shape is None or w_shape is None:
        return None
    if len(x_shape) != 4 or len(w_shape) != 4:
        return None
    n, c_in, h, w_in = x_shape
    wc_in, c_out_per_g, k_h, k_w = w_shape
    groups = int(attrs.get("group", 1))
    if groups <= 0:
        return None
    if wc_in != c_in or c_in % groups != 0:
        return None

    strides = attrs.get("strides", [1, 1])
    if len(strides) != 2:
        return None
    dilations = attrs.get("dilations", [1, 1])
    if len(dilations) != 2:
        return None
    pads = attrs.get("pads", [0, 0, 0, 0])
    if len(pads) == 2:
        pads = [pads[0], pads[1], pads[0], pads[1]]
    if len(pads) != 4:
        return None
    out_pad = attrs.get("output_padding", [0, 0])
    if len(out_pad) != 2:
        return None

    stride_h, stride_w = int(strides[0]), int(strides[1])
    dil_h, dil_w = int(dilations[0]), int(dilations[1])
    pad_h0, pad_w0, pad_h1, pad_w1 = [int(v) for v in pads]
    out_pad_h, out_pad_w = int(out_pad[0]), int(out_pad[1])
    out_c = c_out_per_g * groups

    if "output_shape" in attrs:
        output_shape = [int(v) for v in attrs["output_shape"]]
        if len(output_shape) != 2:
            return None
        out_h, out_w = output_shape
    else:
        out_h = (h - 1) * stride_h - pad_h0 - pad_h1 + dil_h * (k_h - 1) + out_pad_h + 1
        out_w = (w_in - 1) * stride_w - pad_w0 - pad_w1 + dil_w * (k_w - 1) + out_pad_w + 1
    if out_h <= 0 or out_w <= 0:
        return None
    return [n, out_c, out_h, out_w]


def _infer_pool_shape(
    tensors: dict[str, TensorInfo], x_name: str, attrs: dict
) -> list[int] | None:
    x_shape = _get_shape(tensors, x_name)
    if x_shape is None or len(x_shape) < 3:
        return None
    rank = len(x_shape)
    spatial = rank - 2
    n, c = x_shape[0], x_shape[1]
    spatial_in = x_shape[2:]
    kernel = attrs.get("kernel_shape")
    if kernel is None or len(kernel) != spatial:
        return None
    kernel = [int(v) for v in kernel]
    strides = attrs.get("strides", [1] * spatial)
    if len(strides) != spatial:
        return None
    strides = [int(v) for v in strides]
    pads = attrs.get("pads", [0] * (spatial * 2))
    if len(pads) == spatial:
        pads = list(pads) + list(pads)
    if len(pads) != spatial * 2:
        return None
    pads = [int(v) for v in pads]
    dilations = attrs.get("dilations", [1] * spatial)
    if len(dilations) != spatial:
        return None
    dilations = [int(v) for v in dilations]
    out_spatial: list[int] = []
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
            return None
        out_spatial.append(out_dim)
    return [n, c] + out_spatial


def _infer_concat_shape(tensors: dict[str, TensorInfo], inputs: list[str], attrs: dict) -> list[int] | None:
    if not inputs:
        return None
    shapes = [_get_shape(tensors, name) for name in inputs]
    if any(shape is None for shape in shapes):
        return None
    base = list(shapes[0])
    rank = len(base)
    axis = int(attrs.get("axis", 0))
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        return None
    total = 0
    for shape in shapes:
        if len(shape) != rank:
            return None
        for idx in range(rank):
            if idx == axis:
                continue
            if shape[idx] != base[idx]:
                return None
        total += shape[axis]
    base[axis] = total
    return base


def _infer_gather_shape(
    tensors: dict[str, TensorInfo], data_name: str, idx_name: str, attrs: dict
) -> list[int] | None:
    data_shape = _get_shape(tensors, data_name)
    idx_shape = _get_shape(tensors, idx_name)
    if data_shape is None or idx_shape is None:
        return None
    rank = len(data_shape)
    axis = int(attrs.get("axis", 0))
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        return None
    out = []
    out.extend(data_shape[:axis])
    out.extend(idx_shape)
    out.extend(data_shape[axis + 1 :])
    return out


def _infer_gather_elements_shape(
    tensors: dict[str, TensorInfo], data_name: str, idx_name: str, attrs: dict
) -> list[int] | None:
    data_shape = _get_shape(tensors, data_name)
    idx_shape = _get_shape(tensors, idx_name)
    if data_shape is None or idx_shape is None:
        return None
    rank = len(data_shape)
    if rank <= 0 or rank != len(idx_shape):
        return None
    axis = int(attrs.get("axis", 0))
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        return None
    for dim_i in range(rank):
        if dim_i == axis:
            continue
        if idx_shape[dim_i] > data_shape[dim_i]:
            return None
    return list(idx_shape)


def _infer_reverse_sequence_shape(
    tensors: dict[str, TensorInfo], data_name: str, seq_name: str, attrs: dict
) -> list[int] | None:
    data_shape = _get_shape(tensors, data_name)
    seq_shape = _get_shape(tensors, seq_name)
    if data_shape is None or seq_shape is None:
        return None
    rank = len(data_shape)
    if rank < 2 or len(seq_shape) != 1:
        return None
    batch_axis = int(attrs.get("batch_axis", 1))
    time_axis = int(attrs.get("time_axis", 0))
    if batch_axis < 0:
        batch_axis += rank
    if time_axis < 0:
        time_axis += rank
    if batch_axis < 0 or batch_axis >= rank:
        return None
    if time_axis < 0 or time_axis >= rank:
        return None
    if batch_axis == time_axis:
        return None
    if int(seq_shape[0]) != int(data_shape[batch_axis]):
        return None
    return list(data_shape)


def _infer_det_shape(tensors: dict[str, TensorInfo], data_name: str) -> list[int] | None:
    data_shape = _get_shape(tensors, data_name)
    if data_shape is None:
        return None
    if len(data_shape) != 2:
        return None
    if int(data_shape[0]) != int(data_shape[1]):
        return None
    return []


def _infer_non_max_suppression_shape(
    tensors: dict[str, TensorInfo],
    boxes_name: str,
    scores_name: str,
    max_name: str | None,
) -> list[int] | None:
    boxes_shape = _get_shape(tensors, boxes_name)
    scores_shape = _get_shape(tensors, scores_name)
    if boxes_shape is None or scores_shape is None:
        return None
    if len(boxes_shape) != 3 or len(scores_shape) != 3:
        return None
    batch = int(boxes_shape[0])
    spatial = int(boxes_shape[1])
    if int(boxes_shape[2]) != 4:
        return None
    if batch != int(scores_shape[0]) or spatial != int(scores_shape[2]):
        return None
    classes = int(scores_shape[1])
    if batch <= 0 or classes <= 0 or spatial <= 0:
        return None
    max_output_boxes = 0
    if max_name:
        max_vals = _get_const_ints(tensors, max_name)
        if max_vals is None or len(max_vals) != 1:
            return None
        max_output_boxes = int(max_vals[0])
    if max_output_boxes <= 0:
        return None
    per_class = min(max_output_boxes, spatial)
    upper = batch * classes * per_class
    if upper <= 0:
        return None
    return [int(upper), 3]


def _infer_dynamic_quantize_linear_shapes(
    tensors: dict[str, TensorInfo],
    x_name: str,
) -> list[list[int] | None]:
    x_shape = _get_shape(tensors, x_name)
    if x_shape is None:
        return [None, None, None]
    return [list(x_shape), [], []]


def _infer_roi_align_shape(
    tensors: dict[str, TensorInfo],
    x_name: str,
    rois_name: str,
    batch_name: str,
    attrs: dict,
) -> list[int] | None:
    x_shape = _get_shape(tensors, x_name)
    rois_shape = _get_shape(tensors, rois_name)
    batch_shape = _get_shape(tensors, batch_name)
    if x_shape is None or rois_shape is None or batch_shape is None:
        return None
    if len(x_shape) != 4:
        return None
    if len(rois_shape) != 2 or int(rois_shape[1]) != 4:
        return None
    if len(batch_shape) != 1:
        return None
    num_rois = int(rois_shape[0])
    if int(batch_shape[0]) != num_rois:
        return None
    out_h = int(attrs.get("output_height", 1))
    out_w = int(attrs.get("output_width", 1))
    if out_h <= 0 or out_w <= 0:
        return None
    return [num_rois, int(x_shape[1]), out_h, out_w]


def _infer_gather_nd_shape(
    tensors: dict[str, TensorInfo], data_name: str, idx_name: str, attrs: dict
) -> list[int] | None:
    data_shape = _get_shape(tensors, data_name)
    idx_shape = _get_shape(tensors, idx_name)
    if data_shape is None or idx_shape is None:
        return None
    if len(idx_shape) <= 0:
        return None
    batch_dims = int(attrs.get("batch_dims", 0))
    if batch_dims != 0:
        return None
    k = int(idx_shape[-1])
    if k < 0 or k > len(data_shape):
        return None
    out = list(idx_shape[:-1])
    out.extend(data_shape[k:])
    return out


def _infer_scatter_elements_shape(
    tensors: dict[str, TensorInfo],
    data_name: str,
    idx_name: str,
    upd_name: str,
    attrs: dict,
) -> list[int] | None:
    data_shape = _get_shape(tensors, data_name)
    idx_shape = _get_shape(tensors, idx_name)
    upd_shape = _get_shape(tensors, upd_name)
    if data_shape is None or idx_shape is None or upd_shape is None:
        return None
    rank = len(data_shape)
    if rank <= 0 or len(idx_shape) != rank or len(upd_shape) != rank:
        return None
    axis = int(attrs.get("axis", 0))
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        return None
    if idx_shape != upd_shape:
        return None
    for dim_i in range(rank):
        if idx_shape[dim_i] > data_shape[dim_i]:
            return None
    return list(data_shape)


def _infer_onehot_shape(
    tensors: dict[str, TensorInfo], idx_name: str, depth_name: str, attrs: dict
) -> list[int] | None:
    idx_shape = _get_shape(tensors, idx_name)
    depth_vals = _get_const_ints(tensors, depth_name)
    if idx_shape is None or depth_vals is None or len(depth_vals) != 1:
        return None
    if len(idx_shape) <= 0:
        return None
    depth = int(depth_vals[0])
    if depth <= 0:
        return None
    out_rank = len(idx_shape) + 1
    axis = int(attrs.get("axis", -1))
    if axis < 0:
        axis += out_rank
    if axis < 0 or axis >= out_rank:
        return None
    out = list(idx_shape)
    out.insert(axis, depth)
    return out


def _infer_compress_shape(
    tensors: dict[str, TensorInfo], data_name: str, cond_name: str, attrs: dict
) -> list[int] | None:
    data_shape = _get_shape(tensors, data_name)
    cond_vals = _get_const_ints(tensors, cond_name)
    if data_shape is None or cond_vals is None:
        return None
    if len(data_shape) <= 0 or len(cond_vals) <= 0:
        return None
    cond_true = [int(v) != 0 for v in cond_vals]
    axis = attrs.get("axis", None)
    if axis is None:
        total = int(_tensor_size(data_shape))
        limit = min(total, len(cond_true))
        count = sum(1 for i in range(limit) if cond_true[i])
        if count <= 0:
            return None
        return [count]
    axis_i = int(axis)
    rank = len(data_shape)
    if axis_i < 0:
        axis_i += rank
    if axis_i < 0 or axis_i >= rank:
        return None
    axis_dim = int(data_shape[axis_i])
    limit = min(axis_dim, len(cond_true))
    count = sum(1 for i in range(limit) if cond_true[i])
    if count <= 0:
        return None
    out = list(data_shape)
    out[axis_i] = count
    return out


def _infer_scatter_nd_shape(
    tensors: dict[str, TensorInfo],
    data_name: str,
    idx_name: str,
    upd_name: str,
) -> list[int] | None:
    data_shape = _get_shape(tensors, data_name)
    idx_shape = _get_shape(tensors, idx_name)
    upd_shape = _get_shape(tensors, upd_name)
    if data_shape is None or idx_shape is None or upd_shape is None:
        return None
    if len(data_shape) <= 0 or len(idx_shape) <= 0:
        return None
    k = int(idx_shape[-1])
    if k < 0 or k > len(data_shape):
        return None
    expected_upd = list(idx_shape[:-1]) + list(data_shape[k:])
    if upd_shape != expected_upd:
        return None
    return list(data_shape)


def _infer_range_shape(
    tensors: dict[str, TensorInfo], start_name: str, limit_name: str, delta_name: str
) -> list[int] | None:
    start_vals = _get_const_floats(tensors, start_name)
    limit_vals = _get_const_floats(tensors, limit_name)
    delta_vals = _get_const_floats(tensors, delta_name)
    if start_vals is None or limit_vals is None or delta_vals is None:
        return None
    if len(start_vals) != 1 or len(limit_vals) != 1 or len(delta_vals) != 1:
        return None
    start = float(start_vals[0])
    limit = float(limit_vals[0])
    delta = float(delta_vals[0])
    if delta == 0.0:
        return None
    if delta > 0.0 and start >= limit:
        return None
    if delta < 0.0 and start <= limit:
        return None
    span = (limit - start) / delta
    length = int(math.ceil(span - 1e-12))
    if length <= 0:
        return None
    return [length]


def _infer_nonzero_shape(tensors: dict[str, TensorInfo], input_name: str) -> list[int] | None:
    tensor = tensors.get(input_name)
    if tensor is None:
        return None
    in_shape = _get_shape(tensors, input_name)
    if in_shape is None:
        return None
    rank = len(in_shape)
    if rank <= 0:
        return None
    if tensor.data is None:
        return None
    nnz = 0
    if tensor.dtype == "bool":
        for v in tensor.data:
            if int(v) != 0:
                nnz += 1
    else:
        for v in tensor.data:
            if float(v) != 0.0:
                nnz += 1
    return [rank, int(nnz)]


def _infer_topk_shapes(
    tensors: dict[str, TensorInfo], data_name: str, k_name: str, attrs: dict
) -> list[list[int] | None]:
    data_shape = _get_shape(tensors, data_name)
    if data_shape is None:
        return [None, None]
    k_vals = _get_const_ints(tensors, k_name)
    if k_vals is None or len(k_vals) != 1:
        return [None, None]
    k = int(k_vals[0])
    if k <= 0:
        return [None, None]
    rank = len(data_shape)
    if rank <= 0:
        return [None, None]
    axis = int(attrs.get("axis", -1))
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        return [None, None]
    if k > int(data_shape[axis]):
        return [None, None]
    out = list(data_shape)
    out[axis] = k
    return [out, list(out)]


def _infer_constant_meta(attrs: dict) -> tuple[str, list[int]] | None:
    if "value" in attrs:
        value = attrs["value"]
        dtype = _dtype_from_tensorproto(int(value.data_type))
        try:
            arr = numpy_helper.to_array(value)
        except Exception:
            return None
        if dtype == "unknown":
            if str(arr.dtype) == "bool":
                dtype = "bool"
            elif str(arr.dtype).startswith("int8"):
                dtype = "int8"
            elif str(arr.dtype).startswith("int16"):
                dtype = "int16"
            elif str(arr.dtype).startswith("int32"):
                dtype = "int32"
            elif str(arr.dtype).startswith("int64"):
                dtype = "int64"
            elif arr.dtype.kind in ("f", "c"):
                dtype = "float32"
            else:
                return None
        return dtype, list(arr.shape)
    if "value_float" in attrs:
        return "float32", []
    if "value_int" in attrs:
        return "int64", []
    if "value_floats" in attrs:
        vals = attrs["value_floats"]
        return "float32", [len(vals)]
    if "value_ints" in attrs:
        vals = attrs["value_ints"]
        return "int64", [len(vals)]
    return None


def _infer_constant_shape(attrs: dict) -> list[int] | None:
    meta = _infer_constant_meta(attrs)
    if meta is None:
        return None
    _, shape = meta
    return list(shape)


def _infer_transpose_shape(tensors: dict[str, TensorInfo], name: str, attrs: dict) -> list[int] | None:
    in_shape = _get_shape(tensors, name)
    if in_shape is None:
        return None
    rank = len(in_shape)
    perm = attrs.get("perm")
    if perm is None:
        perm = list(reversed(range(rank)))
    if len(perm) != rank:
        return None
    return [in_shape[int(p)] for p in perm]


def _infer_expand_shape(
    tensors: dict[str, TensorInfo], data_name: str, shape_name: str
) -> list[int] | None:
    data_shape = _get_shape(tensors, data_name)
    shape_vals = _get_const_ints(tensors, shape_name)
    if data_shape is None or shape_vals is None:
        return None
    out_shape = [int(v) for v in shape_vals]
    if any(v <= 0 for v in out_shape):
        return None
    if len(data_shape) > len(out_shape):
        return None
    aligned = [1] * (len(out_shape) - len(data_shape)) + list(data_shape)
    for i, in_dim in enumerate(aligned):
        out_dim = out_shape[i]
        if in_dim != out_dim and in_dim != 1:
            return None
    return out_shape


def _infer_tile_shape(
    tensors: dict[str, TensorInfo], data_name: str, reps_name: str
) -> list[int] | None:
    in_shape = _get_shape(tensors, data_name)
    reps = _get_const_ints(tensors, reps_name)
    if in_shape is None or reps is None:
        return None
    if len(reps) != len(in_shape):
        return None
    out = []
    for dim, rep in zip(in_shape, reps):
        if rep <= 0:
            return None
        out.append(int(dim) * int(rep))
    if not _shape_known(out):
        return None
    return out


def _infer_resize_shape(
    tensors: dict[str, TensorInfo],
    data_name: str,
    scales_name: str | None,
    sizes_name: str | None,
) -> list[int] | None:
    in_shape = _get_shape(tensors, data_name)
    if in_shape is None:
        return None
    if sizes_name is not None:
        sizes = _get_const_ints(tensors, sizes_name)
        if sizes is None or len(sizes) != len(in_shape):
            return None
        out = [int(v) for v in sizes]
        if not _shape_known(out):
            return None
        return out
    if scales_name is not None:
        scales = _get_const_floats(tensors, scales_name)
        if scales is None or len(scales) != len(in_shape):
            return None
        out: list[int] = []
        for dim, scale in zip(in_shape, scales):
            if scale <= 0:
                return None
            out_dim = int(float(dim) * float(scale))
            if out_dim <= 0:
                return None
            out.append(out_dim)
        if not _shape_known(out):
            return None
        return out
    return None


def _infer_upsample_shape(
    tensors: dict[str, TensorInfo],
    data_name: str,
    scales_name: str | None,
    attrs: dict,
) -> list[int] | None:
    in_shape = _get_shape(tensors, data_name)
    if in_shape is None:
        return None
    scales: list[float] | None = None
    if scales_name is not None:
        scales = _get_const_floats(tensors, scales_name)
    elif "scales" in attrs:
        scales = [float(v) for v in attrs["scales"]]
    if scales is None:
        return None
    if len(scales) != len(in_shape):
        return None
    out: list[int] = []
    for dim, scale in zip(in_shape, scales):
        if scale <= 0:
            return None
        out_dim = int(float(dim) * float(scale))
        if out_dim <= 0:
            return None
        out.append(out_dim)
    if not _shape_known(out):
        return None
    return out


def _infer_space_to_depth_shape(
    tensors: dict[str, TensorInfo], data_name: str, attrs: dict
) -> list[int] | None:
    in_shape = _get_shape(tensors, data_name)
    if in_shape is None or len(in_shape) != 4:
        return None
    n, c, h, w_in = in_shape
    block = int(attrs.get("blocksize", 0))
    if block <= 0:
        return None
    if h % block != 0 or w_in % block != 0:
        return None
    return [n, c * block * block, h // block, w_in // block]


def _infer_depth_to_space_shape(
    tensors: dict[str, TensorInfo], data_name: str, attrs: dict
) -> list[int] | None:
    in_shape = _get_shape(tensors, data_name)
    if in_shape is None or len(in_shape) != 4:
        return None
    n, c, h, w_in = in_shape
    block = int(attrs.get("blocksize", 0))
    if block <= 0:
        return None
    if c % (block * block) != 0:
        return None
    return [n, c // (block * block), h * block, w_in * block]


def _infer_pad_shape(
    tensors: dict[str, TensorInfo], data_name: str, pads_name: str | None, attrs: dict
) -> list[int] | None:
    in_shape = _get_shape(tensors, data_name)
    if in_shape is None:
        return None
    pads = attrs.get("pads")
    if pads is None and pads_name is not None:
        pads = _get_const_ints(tensors, pads_name)
    if pads is None:
        return None
    rank = len(in_shape)
    if len(pads) != rank * 2:
        return None
    out = []
    for i in range(rank):
        out.append(int(in_shape[i]) + int(pads[i]) + int(pads[i + rank]))
    if not _shape_known(out):
        return None
    return out


def _infer_slice_shape(
    tensors: dict[str, TensorInfo],
    data_name: str,
    starts_name: str | None,
    ends_name: str | None,
    axes_name: str | None,
    steps_name: str | None,
    attrs: dict,
) -> list[int] | None:
    in_shape = _get_shape(tensors, data_name)
    if in_shape is None:
        return None
    starts = attrs.get("starts")
    ends = attrs.get("ends")
    axes = attrs.get("axes")
    steps = attrs.get("steps")
    if starts is None and starts_name is not None:
        starts = _get_const_ints(tensors, starts_name)
    if ends is None and ends_name is not None:
        ends = _get_const_ints(tensors, ends_name)
    if axes is None and axes_name is not None:
        axes = _get_const_ints(tensors, axes_name)
    if steps is None and steps_name is not None:
        steps = _get_const_ints(tensors, steps_name)
    if starts is None or ends is None:
        return None
    rank = len(in_shape)
    axes = list(range(rank)) if axes is None else [int(v) for v in axes]
    starts = [int(v) for v in starts]
    ends = [int(v) for v in ends]
    if steps is not None:
        steps = [int(v) for v in steps]
        if any(v != 1 for v in steps):
            return None
    if len(axes) != len(starts) or len(axes) != len(ends):
        return None
    out_shape = list(in_shape)
    for idx, axis in enumerate(axes):
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            return None
        dim = in_shape[axis]
        s = starts[idx]
        e = ends[idx]
        if s < 0:
            s += dim
        if e < 0:
            e += dim
        if s < 0:
            s = 0
        if e > dim:
            e = dim
        if e < s:
            e = s
        out_shape[axis] = e - s
    if not _shape_known(out_shape):
        return None
    return out_shape


def _infer_reduce_shape(
    tensors: dict[str, TensorInfo], name: str, axes_name: str | None, attrs: dict
) -> list[int] | None:
    in_shape = _get_shape(tensors, name)
    if in_shape is None:
        return None
    rank = len(in_shape)
    axes = attrs.get("axes")
    if axes is None and axes_name is not None:
        axes = _get_const_ints(tensors, axes_name)
    keepdims = int(attrs.get("keepdims", 1))
    if axes is None:
        axes = list(range(rank))
    axes = [int(v) for v in axes]
    norm_axes = []
    for axis in axes:
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            return None
        norm_axes.append(axis)
    if keepdims not in (0, 1):
        return None
    if keepdims == 1:
        out = []
        for i in range(rank):
            out.append(1 if i in norm_axes else in_shape[i])
        return out
    out = [in_shape[i] for i in range(rank) if i not in norm_axes]
    return out


def _infer_squeeze_shape(
    tensors: dict[str, TensorInfo], name: str, axes_name: str | None, attrs: dict
) -> list[int] | None:
    in_shape = _get_shape(tensors, name)
    if in_shape is None:
        return None
    axes = attrs.get("axes")
    if axes is None and axes_name is not None:
        axes = _get_const_ints(tensors, axes_name)
    if axes is None:
        out = [dim for dim in in_shape if dim != 1]
        return out
    axes = [int(v) for v in axes]
    rank = len(in_shape)
    norm_axes = []
    for axis in axes:
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            return None
        norm_axes.append(axis)
    out = []
    for idx, dim in enumerate(in_shape):
        if idx in norm_axes:
            if dim != 1:
                return None
            continue
        out.append(dim)
    return out


def _infer_unsqueeze_shape(
    tensors: dict[str, TensorInfo], name: str, axes_name: str | None, attrs: dict
) -> list[int] | None:
    in_shape = _get_shape(tensors, name)
    if in_shape is None:
        return None
    axes = attrs.get("axes")
    if axes is None and axes_name is not None:
        axes = _get_const_ints(tensors, axes_name)
    if axes is None:
        return None
    axes = [int(v) for v in axes]
    out_rank = len(in_shape) + len(axes)
    norm_axes = []
    for axis in axes:
        if axis < 0:
            axis += out_rank
        if axis < 0 or axis >= out_rank:
            return None
        norm_axes.append(axis)
    if len(set(norm_axes)) != len(norm_axes):
        return None
    out = [None] * out_rank
    for axis in norm_axes:
        out[axis] = 1
    in_iter = iter(in_shape)
    for i in range(out_rank):
        if out[i] is None:
            out[i] = next(in_iter)
    return [int(v) for v in out]


def _infer_arg_shape(tensors: dict[str, TensorInfo], name: str, attrs: dict) -> list[int] | None:
    in_shape = _get_shape(tensors, name)
    if in_shape is None:
        return None
    rank = len(in_shape)
    if rank <= 0:
        return None
    axis = int(attrs.get("axis", 0))
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        return None
    keepdims = int(attrs.get("keepdims", 1))
    if keepdims not in (0, 1):
        return None
    if keepdims == 1:
        out = list(in_shape)
        out[axis] = 1
        return out
    return [in_shape[i] for i in range(rank) if i != axis]


def _infer_shape_output_shape(tensors: dict[str, TensorInfo], name: str) -> list[int] | None:
    in_shape = _get_shape(tensors, name)
    if in_shape is None:
        return None
    return [len(in_shape)]


def _infer_size_output_shape(tensors: dict[str, TensorInfo], name: str) -> list[int] | None:
    in_shape = _get_shape(tensors, name)
    if in_shape is None:
        return None
    return []


def _infer_constant_of_shape_shape(tensors: dict[str, TensorInfo], shape_name: str) -> list[int] | None:
    shape_vals = _get_const_ints(tensors, shape_name)
    if shape_vals is None:
        return None
    out = [int(v) for v in shape_vals]
    if any(v < 0 for v in out):
        return None
    return out


def _infer_split_shapes(
    tensors: dict[str, TensorInfo], node: NodeInfo
) -> list[list[int] | None]:
    if not node.inputs:
        return [None for _ in node.outputs]
    data_shape = _get_shape(tensors, node.inputs[0])
    if data_shape is None:
        return [None for _ in node.outputs]
    rank = len(data_shape)
    axis = int(node.attrs.get("axis", 0))
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        return [None for _ in node.outputs]
    out_count = len(node.outputs)
    if out_count <= 0:
        return [None]

    split_vals = None
    if len(node.inputs) >= 2 and node.inputs[1]:
        split_vals = _get_const_ints(tensors, node.inputs[1])
    if split_vals is None:
        split_attr = node.attrs.get("split")
        if split_attr is not None:
            split_vals = [int(v) for v in split_attr]
    if split_vals is None:
        dim = int(data_shape[axis])
        if dim % out_count != 0:
            return [None for _ in node.outputs]
        each = dim // out_count
        split_vals = [each for _ in node.outputs]
    if len(split_vals) != out_count:
        return [None for _ in node.outputs]
    if any(int(v) <= 0 for v in split_vals):
        return [None for _ in node.outputs]
    if sum(int(v) for v in split_vals) != int(data_shape[axis]):
        return [None for _ in node.outputs]

    out_shapes: list[list[int] | None] = []
    for split_dim in split_vals:
        s = list(data_shape)
        s[axis] = int(split_dim)
        out_shapes.append(s)
    return out_shapes


def _infer_node_output_shape(
    node: NodeInfo, tensors: dict[str, TensorInfo]
) -> list[list[int] | None]:
    op = node.op_type
    inputs = node.inputs
    attrs = node.attrs
    if op in (
        "Relu",
        "ThresholdedRelu",
        "LeakyRelu",
        "Elu",
        "Celu",
        "Selu",
        "Dropout",
        "Hardmax",
        "IsInf",
        "IsNaN",
        "Not",
        "Sign",
        "Sigmoid",
        "Tanh",
        "Clip",
        "Abs",
        "Neg",
        "Exp",
        "Erf",
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
        "HardSigmoid",
        "LogSoftmax",
        "Softplus",
        "Softsign",
        "Shrink",
        "Identity",
        "Cast",
        "QuantizeLinear",
        "DequantizeLinear",
        "Softmax",
        "BatchNormalization",
        "InstanceNormalization",
        "LRN",
        "LpNormalization",
        "MeanVarianceNormalization",
    ):
        return [_infer_unary_shape(tensors, inputs[0])]
    if op in (
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Mod",
        "BitShift",
        "Max",
        "Min",
        "Pow",
        "PRelu",
        "Equal",
        "Greater",
        "Less",
        "GreaterOrEqual",
        "LessOrEqual",
        "And",
        "Or",
        "Xor",
    ):
        return [_infer_binary_broadcast_shape(tensors, inputs[0], inputs[1])]
    if op == "DynamicQuantizeLinear":
        if len(inputs) < 1:
            return [None, None, None]
        return _infer_dynamic_quantize_linear_shapes(tensors, inputs[0])
    if op in ("Sum", "Mean"):
        return [_infer_variadic_broadcast_shape(tensors, inputs)]
    if op == "Where":
        if len(inputs) < 3:
            return [None]
        return [_infer_ternary_broadcast_shape(tensors, inputs[0], inputs[1], inputs[2])]
    if op == "MatMul":
        return [_infer_matmul_shape(tensors, inputs[0], inputs[1])]
    if op == "Einsum":
        if len(inputs) < 2:
            return [None]
        return [_infer_einsum_shape(tensors, inputs[0], inputs[1], attrs)]
    if op == "MatMulInteger":
        if len(inputs) < 2:
            return [None]
        return [_infer_matmul_shape(tensors, inputs[0], inputs[1])]
    if op == "QLinearMatMul":
        if len(inputs) < 5:
            return [None]
        return [_infer_matmul_shape(tensors, inputs[0], inputs[3])]
    if op in ("ArgMax", "ArgMin"):
        return [_infer_arg_shape(tensors, inputs[0], attrs)]
    if op == "Gemm":
        return [_infer_gemm_shape(tensors, inputs[0], inputs[1], attrs)]
    if op == "Reshape":
        if len(inputs) < 2:
            return [None]
        return [_infer_reshape_shape(tensors, inputs[0], inputs[1])]
    if op == "Shape":
        return [_infer_shape_output_shape(tensors, inputs[0])]
    if op == "Size":
        return [_infer_size_output_shape(tensors, inputs[0])]
    if op == "ConstantOfShape":
        return [_infer_constant_of_shape_shape(tensors, inputs[0])]
    if op == "Constant":
        return [_infer_constant_shape(attrs)]
    if op == "Split":
        return _infer_split_shapes(tensors, node)
    if op == "Flatten":
        return [_infer_flatten_shape(tensors, inputs[0], attrs)]
    if op == "Conv":
        if len(inputs) < 2:
            return [None]
        return [_infer_conv_shape(tensors, inputs[0], inputs[1], attrs)]
    if op == "ConvInteger":
        if len(inputs) < 2:
            return [None]
        return [_infer_conv_shape(tensors, inputs[0], inputs[1], attrs)]
    if op == "QLinearConv":
        if len(inputs) < 6:
            return [None]
        return [_infer_conv_shape(tensors, inputs[0], inputs[3], attrs)]
    if op == "ConvTranspose":
        if len(inputs) < 2:
            return [None]
        return [_infer_conv_transpose_shape(tensors, inputs[0], inputs[1], attrs)]
    if op in ("MaxPool", "AveragePool", "LpPool"):
        return [_infer_pool_shape(tensors, inputs[0], attrs)]
    if op == "GlobalAveragePool":
        in_shape = _get_shape(tensors, inputs[0])
        if in_shape is None or len(in_shape) < 3:
            return [None]
        return [[in_shape[0], in_shape[1]] + [1] * (len(in_shape) - 2)]
    if op == "GlobalMaxPool":
        in_shape = _get_shape(tensors, inputs[0])
        if in_shape is None or len(in_shape) < 3:
            return [None]
        return [[in_shape[0], in_shape[1]] + [1] * (len(in_shape) - 2)]
    if op == "GlobalLpPool":
        in_shape = _get_shape(tensors, inputs[0])
        if in_shape is None or len(in_shape) < 3:
            return [None]
        return [[in_shape[0], in_shape[1]] + [1] * (len(in_shape) - 2)]
    if op == "Gather":
        if len(inputs) < 2:
            return [None]
        return [_infer_gather_shape(tensors, inputs[0], inputs[1], attrs)]
    if op == "GatherND":
        if len(inputs) < 2:
            return [None]
        return [_infer_gather_nd_shape(tensors, inputs[0], inputs[1], attrs)]
    if op == "GatherElements":
        if len(inputs) < 2:
            return [None]
        return [_infer_gather_elements_shape(tensors, inputs[0], inputs[1], attrs)]
    if op == "ReverseSequence":
        if len(inputs) < 2:
            return [None]
        return [_infer_reverse_sequence_shape(tensors, inputs[0], inputs[1], attrs)]
    if op == "Compress":
        if len(inputs) < 2:
            return [None]
        return [_infer_compress_shape(tensors, inputs[0], inputs[1], attrs)]
    if op == "ScatterElements":
        if len(inputs) < 3:
            return [None]
        return [_infer_scatter_elements_shape(tensors, inputs[0], inputs[1], inputs[2], attrs)]
    if op == "Scatter":
        if len(inputs) < 3:
            return [None]
        return [_infer_scatter_elements_shape(tensors, inputs[0], inputs[1], inputs[2], attrs)]
    if op == "ScatterND":
        if len(inputs) < 3:
            return [None]
        return [_infer_scatter_nd_shape(tensors, inputs[0], inputs[1], inputs[2])]
    if op == "OneHot":
        if len(inputs) < 2:
            return [None]
        return [_infer_onehot_shape(tensors, inputs[0], inputs[1], attrs)]
    if op == "Det":
        return [_infer_det_shape(tensors, inputs[0])]
    if op == "NonMaxSuppression":
        if len(inputs) < 2:
            return [None]
        max_name = inputs[2] if len(inputs) >= 3 and inputs[2] else None
        return [_infer_non_max_suppression_shape(tensors, inputs[0], inputs[1], max_name)]
    if op == "RoiAlign":
        if len(inputs) < 3:
            return [None]
        return [_infer_roi_align_shape(tensors, inputs[0], inputs[1], inputs[2], attrs)]
    if op == "Concat":
        return [_infer_concat_shape(tensors, inputs, attrs)]
    if op == "Transpose":
        return [_infer_transpose_shape(tensors, inputs[0], attrs)]
    if op == "Expand":
        if len(inputs) < 2:
            return [None]
        return [_infer_expand_shape(tensors, inputs[0], inputs[1])]
    if op == "Tile":
        if len(inputs) < 2:
            return [None]
        return [_infer_tile_shape(tensors, inputs[0], inputs[1])]
    if op == "Resize":
        scales_name = inputs[2] if len(inputs) >= 3 and inputs[2] else None
        sizes_name = inputs[3] if len(inputs) >= 4 and inputs[3] else None
        return [_infer_resize_shape(tensors, inputs[0], scales_name, sizes_name)]
    if op == "Upsample":
        scales_name = inputs[1] if len(inputs) >= 2 and inputs[1] else None
        return [_infer_upsample_shape(tensors, inputs[0], scales_name, attrs)]
    if op == "Range":
        if len(inputs) < 3:
            return [None]
        return [_infer_range_shape(tensors, inputs[0], inputs[1], inputs[2])]
    if op == "CumSum":
        if len(inputs) < 2:
            return [None]
        return [_infer_unary_shape(tensors, inputs[0])]
    if op == "NonZero":
        return [_infer_nonzero_shape(tensors, inputs[0])]
    if op == "EyeLike":
        return [_infer_unary_shape(tensors, inputs[0])]
    if op == "TopK":
        if len(inputs) < 2:
            return [None, None]
        return _infer_topk_shapes(tensors, inputs[0], inputs[1], attrs)
    if op == "Pad":
        pads_name = inputs[1] if len(inputs) >= 2 else None
        return [_infer_pad_shape(tensors, inputs[0], pads_name, attrs)]
    if op == "Slice":
        starts_name = inputs[1] if len(inputs) >= 2 else None
        ends_name = inputs[2] if len(inputs) >= 3 else None
        axes_name = inputs[3] if len(inputs) >= 4 else None
        steps_name = inputs[4] if len(inputs) >= 5 else None
        return [
            _infer_slice_shape(tensors, inputs[0], starts_name, ends_name, axes_name, steps_name, attrs)
        ]
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
        axes_name = inputs[1] if len(inputs) >= 2 and inputs[1] else None
        return [_infer_reduce_shape(tensors, inputs[0], axes_name, attrs)]
    if op == "Squeeze":
        axes_name = inputs[1] if len(inputs) >= 2 and inputs[1] else None
        return [_infer_squeeze_shape(tensors, inputs[0], axes_name, attrs)]
    if op == "Unsqueeze":
        axes_name = inputs[1] if len(inputs) >= 2 else None
        return [_infer_unsqueeze_shape(tensors, inputs[0], axes_name, attrs)]
    if op == "SpaceToDepth":
        return [_infer_space_to_depth_shape(tensors, inputs[0], attrs)]
    if op == "DepthToSpace":
        return [_infer_depth_to_space_shape(tensors, inputs[0], attrs)]
    return [None]


def _infer_shapes(tensors: dict[str, TensorInfo], nodes: list[NodeInfo]) -> None:
    changed = True
    while changed:
        changed = False
        for node in nodes:
            out_shapes = _infer_node_output_shape(node, tensors)
            if not out_shapes:
                continue
            for idx, out_name in enumerate(node.outputs):
                if idx >= len(out_shapes):
                    continue
                shape = out_shapes[idx]
                if shape is None:
                    continue
                existing = tensors.get(out_name)
                if existing is not None:
                    if existing.data is not None:
                        continue
                    if _shape_known(list(existing.shape)):
                        continue
                    tensors[out_name] = TensorInfo(
                        name=out_name,
                        shape=shape,
                        dtype=existing.dtype,
                        data=existing.data,
                    )
                    changed = True
                else:
                    dtype = "float32"
                    if node.op_type == "MatMulInteger":
                        dtype = "int32"
                    elif node.op_type == "ConvInteger":
                        dtype = "int32"
                    elif node.op_type == "NonMaxSuppression":
                        dtype = "int64"
                    elif node.op_type == "DynamicQuantizeLinear":
                        if idx == 0:
                            dtype = "uint8"
                        elif idx == 1:
                            dtype = "float32"
                        else:
                            dtype = "uint8"
                    elif node.op_type == "QLinearConv":
                        if len(node.inputs) >= 8 and node.inputs[7] in tensors:
                            dtype = tensors[node.inputs[7]].dtype
                        elif node.inputs and node.inputs[0] in tensors:
                            dtype = tensors[node.inputs[0]].dtype
                    elif node.op_type == "QLinearMatMul":
                        if len(node.inputs) >= 8 and node.inputs[7] in tensors:
                            dtype = tensors[node.inputs[7]].dtype
                        elif node.inputs and node.inputs[0] in tensors:
                            dtype = tensors[node.inputs[0]].dtype
                    tensors[out_name] = TensorInfo(name=out_name, shape=shape, dtype=dtype)
                    changed = True


