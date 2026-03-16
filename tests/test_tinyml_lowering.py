# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import os
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

raise unittest.SkipTest("tinyml 已抽离到 k2c_tinyml 子项目，主仓库不再运行该测试。")

if importlib.util.find_spec('numpy') is None or importlib.util.find_spec('onnx') is None:
    raise unittest.SkipTest('tinyml optional dependencies numpy/onnx are missing')

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

import sys

sys.path.insert(0, str(SRC))

from keil2cmake.tinyml.codegen import generate_c_code
from keil2cmake.tinyml.converter import load_onnx_model
from keil2cmake.tinyml.runtime.local_evaluator import _eval_model


@contextmanager
def _workspace_temp_dir():
    td = os.path.join(os.getcwd(), f".tmp_tinyml_lower_{uuid.uuid4().hex[:8]}")
    os.makedirs(td, exist_ok=False)
    try:
        yield td
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _save_model(
    path: str,
    nodes: list[onnx.NodeProto],
    inputs: list[onnx.ValueInfoProto],
    outputs: list[onnx.ValueInfoProto],
    initializers: list[onnx.TensorProto] | None = None,
) -> None:
    graph = helper.make_graph(
        nodes=nodes,
        name="tinyml_lowering",
        inputs=inputs,
        outputs=outputs,
        initializer=list(initializers or []),
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    onnx.save(model, path)


def _build_if_model(path: str) -> None:
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [2])

    then_out = helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [2])
    else_out = helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [2])
    then_graph = helper.make_graph(
        [helper.make_node("Add", inputs=["x", "y"], outputs=["then_out"])],
        "then_branch",
        [],
        [then_out],
    )
    else_graph = helper.make_graph(
        [helper.make_node("Sub", inputs=["x", "y"], outputs=["else_out"])],
        "else_branch",
        [],
        [else_out],
    )

    node = helper.make_node(
        "If",
        inputs=["cond"],
        outputs=["z"],
        then_branch=then_graph,
        else_branch=else_graph,
    )
    _save_model(path, [node], [cond, x, y], [z])


def _build_concat_from_sequence_model(path: str) -> None:
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [2])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 2])

    seq = helper.make_node("SequenceConstruct", inputs=["a", "b"], outputs=["seq"])
    cat = helper.make_node("ConcatFromSequence", inputs=["seq"], outputs=["z"], axis=0, new_axis=1)
    _save_model(path, [seq, cat], [a, b], [z])


def _build_split_sequence_model(path: str) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])
    length = helper.make_tensor_value_info("length", TensorProto.INT64, [])

    split = numpy_helper.from_array(np.array([2, 2], dtype=np.int64), name="split")
    idx = numpy_helper.from_array(np.array(1, dtype=np.int64), name="idx")

    split_seq = helper.make_node(
        "SplitToSequence",
        inputs=["x", "split"],
        outputs=["seq"],
        axis=1,
        keepdims=1,
    )
    seq_at = helper.make_node("SequenceAt", inputs=["seq", "idx"], outputs=["y"])
    seq_len = helper.make_node("SequenceLength", inputs=["seq"], outputs=["length"])
    _save_model(path, [split_seq, seq_at, seq_len], [x], [y, length], [split, idx])


class TestTinyMlLowering(unittest.TestCase):
    def test_if_is_lowered_to_tensor_ops(self) -> None:
        with _workspace_temp_dir() as td:
            model_path = os.path.join(td, "if.onnx")
            out_dir = os.path.join(td, "out")
            _build_if_model(model_path)

            model = load_onnx_model(model_path)
            ops = [node.op_type for node in model.nodes]
            self.assertNotIn("If", ops)
            self.assertIn("Where", ops)

            inputs_true = {
                "cond": np.array(True, dtype=np.bool_),
                "x": np.array([1.0, 2.0], dtype=np.float32),
                "y": np.array([3.0, 4.0], dtype=np.float32),
            }
            out_true = _eval_model(model, inputs_true)["z"]
            np.testing.assert_allclose(out_true, np.array([4.0, 6.0], dtype=np.float32), rtol=1e-5, atol=1e-6)

            inputs_false = dict(inputs_true)
            inputs_false["cond"] = np.array(False, dtype=np.bool_)
            out_false = _eval_model(model, inputs_false)["z"]
            np.testing.assert_allclose(out_false, np.array([-2.0, -2.0], dtype=np.float32), rtol=1e-5, atol=1e-6)

            result = generate_c_code(model, out_dir, "if_lowered", "flash")
            self.assertTrue(os.path.exists(result["source"]))
            self.assertTrue(os.path.exists(result["header"]))

    def test_concat_from_sequence_is_lowered(self) -> None:
        with _workspace_temp_dir() as td:
            model_path = os.path.join(td, "seq_concat.onnx")
            out_dir = os.path.join(td, "out")
            _build_concat_from_sequence_model(model_path)

            model = load_onnx_model(model_path)
            ops = [node.op_type for node in model.nodes]
            self.assertNotIn("SequenceConstruct", ops)
            self.assertNotIn("ConcatFromSequence", ops)
            self.assertIn("Concat", ops)

            inputs = {
                "a": np.array([1.0, 2.0], dtype=np.float32),
                "b": np.array([3.0, 4.0], dtype=np.float32),
            }
            out = _eval_model(model, inputs)["z"]
            expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)

            result = generate_c_code(model, out_dir, "seq_concat_lowered", "flash")
            self.assertTrue(os.path.exists(result["source"]))
            self.assertTrue(os.path.exists(result["header"]))

    def test_split_to_sequence_and_at_and_length_are_lowered(self) -> None:
        with _workspace_temp_dir() as td:
            model_path = os.path.join(td, "seq_split.onnx")
            out_dir = os.path.join(td, "out")
            _build_split_sequence_model(model_path)

            model = load_onnx_model(model_path)
            ops = [node.op_type for node in model.nodes]
            self.assertNotIn("SplitToSequence", ops)
            self.assertNotIn("SequenceAt", ops)
            self.assertNotIn("SequenceLength", ops)
            self.assertIn("Split", ops)
            self.assertIn("Identity", ops)

            x = np.arange(8, dtype=np.float32).reshape(2, 4)
            out_map = _eval_model(model, {"x": x})
            np.testing.assert_allclose(out_map["y"], x[:, 2:4], rtol=1e-5, atol=1e-6)
            self.assertEqual(int(np.array(out_map["length"]).reshape(-1)[0]), 2)

            result = generate_c_code(model, out_dir, "seq_split_lowered", "flash")
            self.assertTrue(os.path.exists(result["source"]))
            self.assertTrue(os.path.exists(result["header"]))
