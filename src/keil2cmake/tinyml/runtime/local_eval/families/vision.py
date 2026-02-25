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

def handle_vision_family(
    model: ModelIR,
    node,
    tensors: dict[str, np.ndarray],
    ins: list[np.ndarray | None],
    out_name: str,
    out_dtype: str,
) -> bool:
    op = node.op_type
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
            return True
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
        return True
    

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
        return True
    

    if op == "Det":
        data = ins[0]
        if data.ndim < 2:
            raise ValueError("Det expects input rank >= 2.")
        if data.shape[-2] != data.shape[-1]:
            raise ValueError("Det requires square matrix.")
        x_dtype = _tensor_dtype(model.tensors[node.inputs[0]])
        if x_dtype in ("int8", "int16"):
            x_tensor = model.tensors[node.inputs[0]]
            if x_tensor.qscale is not None and x_tensor.qzero is not None:
                sx, zx = _qparams(model, node.inputs[0])
                data_f = _dequantize_int(data, sx, zx)
            else:
                data_f = data.astype(np.float32, copy=False)
        else:
            data_f = data.astype(np.float32, copy=False)
        n = int(data.shape[-1])
        mats = data_f.reshape(-1, n, n)
        det_vals = np.linalg.det(mats).astype(np.float32).reshape(data.shape[:-2])
        if out_dtype in ("int8", "int16"):
            so, zo = _qparams(model, out_name)
            tensors[out_name] = _quantize_float(det_vals, so, zo, out_dtype)
        else:
            tensors[out_name] = det_vals.astype(np.float32)
        return True
    

    return False
