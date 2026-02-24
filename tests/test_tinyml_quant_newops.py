# -*- coding: utf-8 -*-

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

import sys

sys.path.insert(0, str(SRC))

from keil2cmake.tinyml.codegen import generate_c_code
from keil2cmake.tinyml.converter import load_onnx_model


def _save_model(
    path: str,
    nodes: list[onnx.NodeProto],
    inputs: list[onnx.ValueInfoProto],
    outputs: list[onnx.ValueInfoProto],
    initializers: list[onnx.TensorProto] | None = None,
) -> None:
    graph = helper.make_graph(
        nodes,
        "tinyml_quant_newops",
        inputs,
        outputs,
        list(initializers or []),
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    onnx.save(model, path)


def _build_random_uniform_int8_model(path: str) -> None:
    dummy = helper.make_tensor_value_info("dummy", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.INT8, [2, 3])
    node = helper.make_node(
        "RandomUniform",
        inputs=[],
        outputs=["y"],
        shape=[2, 3],
        low=-8.0,
        high=8.0,
        seed=1.25,
    )
    _save_model(path, [node], [dummy], [y])


def _build_random_normal_like_int16_model(path: str) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.INT16, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.INT16, [2, 3])
    node = helper.make_node(
        "RandomNormalLike",
        inputs=["x"],
        outputs=["y"],
        mean=0.0,
        scale=2.0,
        seed=2.5,
    )
    _save_model(path, [node], [x], [y])


def _build_multinomial_int8_model(path: str) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.INT8, [2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.INT32, [2, 2])
    node = helper.make_node("Multinomial", inputs=["x"], outputs=["y"], sample_size=2, seed=3.0)
    _save_model(path, [node], [x], [y])


def _build_nllloss_int16_model(path: str) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.INT16, [2, 3])
    t = helper.make_tensor_value_info("t", TensorProto.INT64, [2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
    node = helper.make_node(
        "NegativeLogLikelihoodLoss",
        inputs=["x", "t"],
        outputs=["y"],
        reduction="none",
    )
    _save_model(path, [node], [x, t], [y])


def _build_softmax_ce_int8_model(path: str) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.INT8, [2, 3])
    t = helper.make_tensor_value_info("t", TensorProto.INT64, [2])
    loss = helper.make_tensor_value_info("loss", TensorProto.FLOAT, [])
    logp = helper.make_tensor_value_info("logp", TensorProto.FLOAT, [2, 3])
    node = helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=["x", "t"],
        outputs=["loss", "logp"],
        reduction="mean",
    )
    _save_model(path, [node], [x, t], [loss, logp])


def _build_max_roi_pool_int16_model(path: str) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.INT16, [1, 2, 4, 4])
    rois = helper.make_tensor_value_info("rois", TensorProto.FLOAT, [1, 5])
    y = helper.make_tensor_value_info("y", TensorProto.INT16, [1, 2, 2, 2])
    node = helper.make_node(
        "MaxRoiPool",
        inputs=["x", "rois"],
        outputs=["y"],
        pooled_shape=[2, 2],
        spatial_scale=1.0,
    )
    _save_model(path, [node], [x, rois], [y])


def _build_conv_transpose_group_model(path: str) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 3, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 6, 5, 5])
    w = numpy_helper.from_array(np.random.randn(4, 3, 3, 3).astype(np.float32), name="W")
    b = numpy_helper.from_array(np.random.randn(6).astype(np.float32), name="B")
    node = helper.make_node(
        "ConvTranspose",
        inputs=["x", "W", "B"],
        outputs=["y"],
        group=2,
        strides=[1, 1],
        pads=[0, 0, 0, 0],
    )
    _save_model(path, [node], [x], [y], [w, b])


def _build_mvn_axes_general_model(path: str) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4, 5])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4, 5])
    node = helper.make_node(
        "MeanVarianceNormalization",
        inputs=["x"],
        outputs=["y"],
        axes=[1, 3],
    )
    _save_model(path, [node], [x], [y])


def _build_nllloss_rank3_model(path: str) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.INT16, [2, 3, 4])
    t = helper.make_tensor_value_info("t", TensorProto.INT64, [2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
    node = helper.make_node(
        "NegativeLogLikelihoodLoss",
        inputs=["x", "t"],
        outputs=["y"],
        reduction="none",
    )
    _save_model(path, [node], [x, t], [y])


def _build_softmax_ce_rank3_model(path: str) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.INT8, [2, 3, 4])
    t = helper.make_tensor_value_info("t", TensorProto.INT64, [2, 4])
    loss = helper.make_tensor_value_info("loss", TensorProto.FLOAT, [])
    logp = helper.make_tensor_value_info("logp", TensorProto.FLOAT, [2, 3, 4])
    node = helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=["x", "t"],
        outputs=["loss", "logp"],
        reduction="mean",
    )
    _save_model(path, [node], [x, t], [loss, logp])


def _build_gemm_full_attrs_model(path: str) -> None:
    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [3, 2])
    b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 3])
    c = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2, 1])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])
    node = helper.make_node(
        "Gemm",
        inputs=["A", "B", "C"],
        outputs=["Y"],
        transA=1,
        transB=1,
        alpha=0.75,
        beta=0.25,
    )
    _save_model(path, [node], [a, b, c], [y])


def _build_expand_runtime_shape_model(path: str) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 1])
    shape = helper.make_tensor_value_info("shape", TensorProto.INT64, [3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])
    node = helper.make_node("Expand", inputs=["x", "shape"], outputs=["y"])
    _save_model(path, [node], [x, shape], [y])


def _build_cumsum_runtime_axis_model(path: str) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.INT16, [2, 3, 4])
    axis = helper.make_tensor_value_info("axis", TensorProto.INT64, [1])
    y = helper.make_tensor_value_info("y", TensorProto.INT16, [2, 3, 4])
    node = helper.make_node("CumSum", inputs=["x", "axis"], outputs=["y"], exclusive=1, reverse=1)
    _save_model(path, [node], [x, axis], [y])


def _build_gather_nd_indices_model(path: str) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])
    idx = helper.make_tensor_value_info("idx", TensorProto.INT64, [2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2, 2, 4])
    node = helper.make_node("Gather", inputs=["x", "idx"], outputs=["y"], axis=1)
    _save_model(path, [node], [x, idx], [y])


def _build_gathernd_batch_dims_model(path: str) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])
    idx = helper.make_tensor_value_info("idx", TensorProto.INT64, [2, 5, 1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 5, 4])
    node = helper.make_node("GatherND", inputs=["x", "idx"], outputs=["y"], batch_dims=1)
    _save_model(path, [node], [x, idx], [y])


def _build_onehot_dynamic_io_model(path: str) -> None:
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [])
    depth = helper.make_tensor_value_info("depth", TensorProto.INT64, [1])
    values = helper.make_tensor_value_info("values", TensorProto.FLOAT, [2])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [4])
    node = helper.make_node("OneHot", inputs=["indices", "depth", "values"], outputs=["out"], axis=-1)
    _save_model(path, [node], [indices, depth, values], [out])


def _build_pad_reflect_model(path: str) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 5, 6])
    node = helper.make_node("Pad", inputs=["x"], outputs=["y"], mode="reflect", pads=[0, 1, 1, 0, 1, 1])
    _save_model(path, [node], [x], [y])


def _build_slice_steps_model(path: str) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 8])
    starts = numpy_helper.from_array(np.array([0, 0, 7], dtype=np.int64), name="starts")
    ends = numpy_helper.from_array(np.array([2, 3, -9], dtype=np.int64), name="ends")
    axes = numpy_helper.from_array(np.array([0, 1, 2], dtype=np.int64), name="axes")
    steps = numpy_helper.from_array(np.array([1, 1, -2], dtype=np.int64), name="steps")
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])
    node = helper.make_node("Slice", inputs=["x", "starts", "ends", "axes", "steps"], outputs=["y"])
    _save_model(path, [node], [x], [y], [starts, ends, axes, steps])


def _build_det_batched_model(path: str) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
    node = helper.make_node("Det", inputs=["x"], outputs=["y"])
    _save_model(path, [node], [x], [y])


class TestTinyMlQuantizedNewOps(unittest.TestCase):
    def _assert_codegen_ok(self, builder) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, "model.onnx")
            out_dir = os.path.join(td, "out")
            os.makedirs(out_dir, exist_ok=True)
            builder(model_path)
            model = load_onnx_model(model_path)
            result = generate_c_code(model, out_dir, "model", "flash")
            self.assertTrue(os.path.exists(result["source"]))
            self.assertTrue(os.path.exists(result["header"]))

    def test_random_uniform_int8_codegen(self) -> None:
        self._assert_codegen_ok(_build_random_uniform_int8_model)

    def test_random_normal_like_int16_codegen(self) -> None:
        self._assert_codegen_ok(_build_random_normal_like_int16_model)

    def test_multinomial_int8_input_codegen(self) -> None:
        self._assert_codegen_ok(_build_multinomial_int8_model)

    def test_negative_log_likelihood_loss_int16_input_codegen(self) -> None:
        self._assert_codegen_ok(_build_nllloss_int16_model)

    def test_softmax_cross_entropy_loss_int8_logits_codegen(self) -> None:
        self._assert_codegen_ok(_build_softmax_ce_int8_model)

    def test_max_roi_pool_int16_codegen(self) -> None:
        self._assert_codegen_ok(_build_max_roi_pool_int16_model)

    def test_conv_transpose_group_codegen(self) -> None:
        self._assert_codegen_ok(_build_conv_transpose_group_model)

    def test_mvn_axes_general_codegen(self) -> None:
        self._assert_codegen_ok(_build_mvn_axes_general_model)

    def test_nllloss_rank3_codegen(self) -> None:
        self._assert_codegen_ok(_build_nllloss_rank3_model)

    def test_softmax_cross_entropy_rank3_codegen(self) -> None:
        self._assert_codegen_ok(_build_softmax_ce_rank3_model)

    def test_gemm_full_attrs_codegen(self) -> None:
        self._assert_codegen_ok(_build_gemm_full_attrs_model)

    def test_expand_runtime_shape_codegen(self) -> None:
        self._assert_codegen_ok(_build_expand_runtime_shape_model)

    def test_cumsum_runtime_axis_codegen(self) -> None:
        self._assert_codegen_ok(_build_cumsum_runtime_axis_model)

    def test_gather_nd_indices_codegen(self) -> None:
        self._assert_codegen_ok(_build_gather_nd_indices_model)

    def test_gathernd_batch_dims_codegen(self) -> None:
        self._assert_codegen_ok(_build_gathernd_batch_dims_model)

    def test_onehot_dynamic_io_codegen(self) -> None:
        self._assert_codegen_ok(_build_onehot_dynamic_io_model)

    def test_pad_reflect_codegen(self) -> None:
        self._assert_codegen_ok(_build_pad_reflect_model)

    def test_slice_steps_codegen(self) -> None:
        self._assert_codegen_ok(_build_slice_steps_model)

    def test_det_batched_codegen(self) -> None:
        self._assert_codegen_ok(_build_det_batched_model)
