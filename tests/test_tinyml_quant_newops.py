# -*- coding: utf-8 -*-

import os
import tempfile
import unittest
from pathlib import Path

import onnx
from onnx import TensorProto, helper

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
) -> None:
    graph = helper.make_graph(nodes, "tinyml_quant_newops", inputs, outputs, [])
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
