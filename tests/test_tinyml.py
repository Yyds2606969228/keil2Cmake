# -*- coding: utf-8 -*-

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'

import sys
sys.path.insert(0, str(SRC))

from keil2cmake.tinyml.project import generate_tinyml_project as _generate_tinyml_project
from keil2cmake.tinyml.converter import load_onnx_model
from keil2cmake.tinyml.runtime import validate_model_consistency
from keil2cmake.tinyml.runtime.c_runner import run_generated_c_model
from keil2cmake.tinyml.runtime.validator import _eval_model


def generate_tinyml_project(*args, **kwargs):
    # Keep default test behavior stable: strict validation is covered in dedicated tests.
    kwargs.setdefault('strict_validation', False)
    return _generate_tinyml_project(*args, **kwargs)


def _build_simple_gemm_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3])

    w = np.random.rand(4, 3).astype(np.float32)
    b = np.random.rand(3).astype(np.float32)
    init_w = numpy_helper.from_array(w, name='W')
    init_b = numpy_helper.from_array(b, name='B')

    node_gemm = helper.make_node('Gemm', inputs=['input', 'W', 'B'], outputs=['z'])
    node_relu = helper.make_node('Relu', inputs=['z'], outputs=['output'])

    graph = helper.make_graph(
        [node_gemm, node_relu],
        'tinyml_test',
        [x],
        [y],
        [init_w, init_b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_gemm_softmax_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3])

    w = np.random.rand(4, 3).astype(np.float32)
    b = np.random.rand(3).astype(np.float32)
    init_w = numpy_helper.from_array(w, name='W')
    init_b = numpy_helper.from_array(b, name='B')

    node_gemm = helper.make_node('Gemm', inputs=['input', 'W', 'B'], outputs=['z'])
    node_softmax = helper.make_node('Softmax', inputs=['z'], outputs=['output'], axis=1)

    graph = helper.make_graph(
        [node_gemm, node_softmax],
        'tinyml_fallback_reason_test',
        [x],
        [y],
        [init_w, init_b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_conv_pool_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 1, 1])
    w = np.random.rand(1, 1, 3, 3).astype(np.float32)
    b = np.random.rand(1).astype(np.float32)
    init_w = numpy_helper.from_array(w, name='W')
    init_b = numpy_helper.from_array(b, name='B')
    node_conv = helper.make_node(
        'Conv',
        inputs=['input', 'W', 'B'],
        outputs=['c1'],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
    )
    node_relu = helper.make_node('Relu', inputs=['c1'], outputs=['r1'])
    node_pool = helper.make_node(
        'MaxPool',
        inputs=['r1'],
        outputs=['output'],
        kernel_shape=[2, 2],
        strides=[2, 2],
        pads=[0, 0, 0, 0],
    )
    graph = helper.make_graph(
        [node_conv, node_relu, node_pool],
        'conv_pool',
        [x],
        [y],
        [init_w, init_b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_group_conv_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4, 4, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 6, 2, 2])
    w = np.random.rand(6, 2, 3, 3).astype(np.float32)
    b = np.random.rand(6).astype(np.float32)
    init_w = numpy_helper.from_array(w, name='W')
    init_b = numpy_helper.from_array(b, name='B')
    node_conv = helper.make_node(
        'Conv',
        inputs=['input', 'W', 'B'],
        outputs=['output'],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
        group=2,
    )
    graph = helper.make_graph(
        [node_conv],
        'group_conv',
        [x],
        [y],
        [init_w, init_b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_batchnorm_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2, 2, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2, 2, 2])
    scale = numpy_helper.from_array(np.random.rand(2).astype(np.float32), name='scale')
    bias = numpy_helper.from_array(np.random.rand(2).astype(np.float32), name='bias')
    mean = numpy_helper.from_array(np.random.rand(2).astype(np.float32), name='mean')
    var = numpy_helper.from_array(np.random.rand(2).astype(np.float32), name='var')
    node_bn = helper.make_node(
        'BatchNormalization',
        inputs=['input', 'scale', 'bias', 'mean', 'var'],
        outputs=['output'],
        epsilon=1e-5,
    )
    graph = helper.make_graph(
        [node_bn],
        'bn_test',
        [x],
        [y],
        [scale, bias, mean, var],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_conv_pool_n2_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 1, 4, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 1, 1, 1])
    w = np.random.rand(1, 1, 3, 3).astype(np.float32)
    b = np.random.rand(1).astype(np.float32)
    init_w = numpy_helper.from_array(w, name='W')
    init_b = numpy_helper.from_array(b, name='B')
    node_conv = helper.make_node(
        'Conv',
        inputs=['input', 'W', 'B'],
        outputs=['c1'],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
    )
    node_pool = helper.make_node(
        'MaxPool',
        inputs=['c1'],
        outputs=['output'],
        kernel_shape=[2, 2],
        strides=[2, 2],
        pads=[0, 0, 0, 0],
    )
    graph = helper.make_graph(
        [node_conv, node_pool],
        'conv_pool_n2',
        [x],
        [y],
        [init_w, init_b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_batchnorm_n2_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 2, 2, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 2, 2, 2])
    scale = numpy_helper.from_array(np.random.rand(2).astype(np.float32), name='scale')
    bias = numpy_helper.from_array(np.random.rand(2).astype(np.float32), name='bias')
    mean = numpy_helper.from_array(np.random.rand(2).astype(np.float32), name='mean')
    var = numpy_helper.from_array(np.random.rand(2).astype(np.float32), name='var')
    node_bn = helper.make_node(
        'BatchNormalization',
        inputs=['input', 'scale', 'bias', 'mean', 'var'],
        outputs=['output'],
        epsilon=1e-5,
    )
    graph = helper.make_graph(
        [node_bn],
        'bn_n2_test',
        [x],
        [y],
        [scale, bias, mean, var],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_batchnorm_nd_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3, 2, 2, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3, 2, 2, 2])
    scale = numpy_helper.from_array(np.random.rand(3).astype(np.float32), name='scale')
    bias = numpy_helper.from_array(np.random.rand(3).astype(np.float32), name='bias')
    mean = numpy_helper.from_array(np.random.rand(3).astype(np.float32), name='mean')
    var = numpy_helper.from_array(np.random.rand(3).astype(np.float32), name='var')
    node_bn = helper.make_node(
        'BatchNormalization',
        inputs=['input', 'scale', 'bias', 'mean', 'var'],
        outputs=['output'],
        epsilon=1e-5,
    )
    graph = helper.make_graph(
        [node_bn],
        'bn_nd_test',
        [x],
        [y],
        [scale, bias, mean, var],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_instance_norm_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4, 3, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4, 3, 3])
    scale = numpy_helper.from_array(np.random.rand(4).astype(np.float32), name='scale')
    bias = numpy_helper.from_array(np.random.rand(4).astype(np.float32), name='bias')
    node = helper.make_node(
        'InstanceNormalization',
        inputs=['input', 'scale', 'bias'],
        outputs=['output'],
        epsilon=1e-5,
    )
    graph = helper.make_graph([node], 'instancenorm_test', [x], [y], [scale, bias])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_lrn_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 5, 4, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 5, 4, 4])
    node = helper.make_node(
        'LRN',
        inputs=['input'],
        outputs=['output'],
        size=3,
        alpha=1e-4,
        beta=0.75,
        bias=1.0,
    )
    graph = helper.make_graph([node], 'lrn_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_lp_normalization_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])
    node = helper.make_node('LpNormalization', inputs=['input'], outputs=['output'], axis=1, p=2)
    graph = helper.make_graph([node], 'lp_normalization_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_mean_variance_normalization_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2, 2, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2, 2, 2])
    node = helper.make_node(
        'MeanVarianceNormalization',
        inputs=['input'],
        outputs=['output'],
        axes=[0, 2, 3],
    )
    graph = helper.make_graph([node], 'mean_variance_normalization_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_lppool_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 2, 2])
    node = helper.make_node(
        'LpPool',
        inputs=['input'],
        outputs=['output'],
        kernel_shape=[2, 2],
        strides=[2, 2],
        p=2,
    )
    graph = helper.make_graph([node], 'lppool_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_lppool_nd_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 3, 4, 5])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 2, 2, 2])
    node = helper.make_node(
        'LpPool',
        inputs=['input'],
        outputs=['output'],
        kernel_shape=[2, 2, 2],
        strides=[1, 2, 2],
        p=2,
    )
    graph = helper.make_graph([node], 'lppool_nd_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_global_lppool_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2, 3, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2, 1, 1])
    node = helper.make_node('GlobalLpPool', inputs=['input'], outputs=['output'], p=2)
    graph = helper.make_graph([node], 'global_lppool_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_global_lppool_nd_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 2, 2, 2, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 2, 1, 1, 1])
    node = helper.make_node('GlobalLpPool', inputs=['input'], outputs=['output'], p=2)
    graph = helper.make_graph([node], 'global_lppool_nd_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_global_avgpool_n2_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 2, 3, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 2, 1, 1])
    node = helper.make_node('GlobalAveragePool', inputs=['input'], outputs=['output'])
    graph = helper.make_graph([node], 'global_avgpool_n2_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_avgpool_nd_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2, 3, 4, 5])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2, 2, 2, 2])
    node = helper.make_node(
        'AveragePool',
        inputs=['input'],
        outputs=['output'],
        kernel_shape=[2, 2, 2],
        strides=[1, 2, 2],
        pads=[0, 0, 0, 0, 0, 0],
    )
    graph = helper.make_graph([node], 'avgpool_nd_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_global_avgpool_nd_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3, 2, 2, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3, 1, 1, 1])
    node = helper.make_node('GlobalAveragePool', inputs=['input'], outputs=['output'])
    graph = helper.make_graph([node], 'global_avgpool_nd_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_add_broadcast_n2_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3, 4, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3, 4, 4])
    c = np.random.rand(3, 1, 1).astype(np.float32)
    init_c = numpy_helper.from_array(c, name='C')
    node = helper.make_node('Add', inputs=['input', 'C'], outputs=['output'])
    graph = helper.make_graph([node], 'add_broadcast_n2_test', [x], [y], [init_c])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_resnet_like_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 8, 8, 8])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])

    w1 = numpy_helper.from_array(np.random.randn(8, 8, 3, 3).astype(np.float32), name='W1')
    b1 = numpy_helper.from_array(np.random.randn(8).astype(np.float32), name='B1')
    s1 = numpy_helper.from_array(np.random.rand(8).astype(np.float32), name='S1')
    bb1 = numpy_helper.from_array(np.random.rand(8).astype(np.float32), name='BB1')
    m1 = numpy_helper.from_array(np.random.rand(8).astype(np.float32), name='M1')
    v1 = numpy_helper.from_array((np.random.rand(8) + 0.1).astype(np.float32), name='V1')

    w2 = numpy_helper.from_array(np.random.randn(8, 8, 3, 3).astype(np.float32), name='W2')
    b2 = numpy_helper.from_array(np.random.randn(8).astype(np.float32), name='B2')
    s2 = numpy_helper.from_array(np.random.rand(8).astype(np.float32), name='S2')
    bb2 = numpy_helper.from_array(np.random.rand(8).astype(np.float32), name='BB2')
    m2 = numpy_helper.from_array(np.random.rand(8).astype(np.float32), name='M2')
    v2 = numpy_helper.from_array((np.random.rand(8) + 0.1).astype(np.float32), name='V2')

    w3 = numpy_helper.from_array(np.random.randn(8, 4).astype(np.float32), name='W3')
    b3 = numpy_helper.from_array(np.random.randn(4).astype(np.float32), name='B3')

    nodes = [
        helper.make_node('Conv', inputs=['input', 'W1', 'B1'], outputs=['c1'], pads=[1, 1, 1, 1]),
        helper.make_node('BatchNormalization', inputs=['c1', 'S1', 'BB1', 'M1', 'V1'], outputs=['bn1'], epsilon=1e-5),
        helper.make_node('Relu', inputs=['bn1'], outputs=['r1']),
        helper.make_node('Conv', inputs=['r1', 'W2', 'B2'], outputs=['c2'], pads=[1, 1, 1, 1]),
        helper.make_node('BatchNormalization', inputs=['c2', 'S2', 'BB2', 'M2', 'V2'], outputs=['bn2'], epsilon=1e-5),
        helper.make_node('Add', inputs=['bn2', 'input'], outputs=['add']),
        helper.make_node('Relu', inputs=['add'], outputs=['r2']),
        helper.make_node('GlobalAveragePool', inputs=['r2'], outputs=['gap']),
        helper.make_node('Flatten', inputs=['gap'], outputs=['flat'], axis=1),
        helper.make_node('Gemm', inputs=['flat', 'W3', 'B3'], outputs=['output']),
    ]
    graph = helper.make_graph(
        nodes,
        'resnet_like',
        [x],
        [y],
        [w1, b1, s1, bb1, m1, v1, w2, b2, s2, bb2, m2, v2, w3, b3],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_mobilenetv2_like_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 8, 8, 8])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 8, 8, 8])

    w_exp = numpy_helper.from_array(np.random.randn(16, 8, 1, 1).astype(np.float32), name='W_EXP')
    b_exp = numpy_helper.from_array(np.random.randn(16).astype(np.float32), name='B_EXP')
    w_dw = numpy_helper.from_array(np.random.randn(16, 1, 3, 3).astype(np.float32), name='W_DW')
    b_dw = numpy_helper.from_array(np.random.randn(16).astype(np.float32), name='B_DW')
    w_prj = numpy_helper.from_array(np.random.randn(8, 16, 1, 1).astype(np.float32), name='W_PRJ')
    b_prj = numpy_helper.from_array(np.random.randn(8).astype(np.float32), name='B_PRJ')

    nodes = [
        helper.make_node('Conv', inputs=['input', 'W_EXP', 'B_EXP'], outputs=['exp']),
        helper.make_node('Clip', inputs=['exp'], outputs=['exp6'], min=0.0, max=6.0),
        helper.make_node('Conv', inputs=['exp6', 'W_DW', 'B_DW'], outputs=['dw'], group=16, pads=[1, 1, 1, 1]),
        helper.make_node('Clip', inputs=['dw'], outputs=['dw6'], min=0.0, max=6.0),
        helper.make_node('Conv', inputs=['dw6', 'W_PRJ', 'B_PRJ'], outputs=['prj']),
        helper.make_node('Add', inputs=['prj', 'input'], outputs=['output']),
    ]
    graph = helper.make_graph(nodes, 'mobilenetv2_like', [x], [y], [w_exp, b_exp, w_dw, b_dw, w_prj, b_prj])
    # Use opset 10 so Clip min/max attributes are ONNX-runtime valid.
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 10)])
    onnx.save(model, path)


def _build_softmax2d_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])
    node = helper.make_node('Softmax', inputs=['input'], outputs=['output'], axis=1)
    graph = helper.make_graph([node], 'softmax2d', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_softmax4d_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 16, 4, 10])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 4, 10])
    node = helper.make_node('Softmax', inputs=['input'], outputs=['output'], axis=1)
    graph = helper.make_graph([node], 'softmax4d', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_einsum_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3, 5])
    w = numpy_helper.from_array(np.random.randn(2, 4, 5).astype(np.float32), name='W')
    node = helper.make_node('Einsum', inputs=['input', 'W'], outputs=['output'], equation='bij,bjk->bik')
    graph = helper.make_graph([node], 'einsum_test', [x], [y], [w])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_einsum_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3, 5])
    w = numpy_helper.from_array(np.random.randn(2, 4, 5).astype(np.float32), name='W')
    scale, zero = _qdq_params(qdtype)

    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    qw = helper.make_node('QuantizeLinear', inputs=['W', 'scale', 'zero'], outputs=['qw'])
    einsum = helper.make_node('Einsum', inputs=['qx', 'qw'], outputs=['qy'], equation='bij,bjk->bik')
    dq = helper.make_node('DequantizeLinear', inputs=['qy', 'scale', 'zero'], outputs=['output'])

    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [2, 3, 4]),
        helper.make_tensor_value_info('qw', qdtype, [2, 4, 5]),
        helper.make_tensor_value_info('qy', qdtype, [2, 3, 5]),
    ]

    graph = helper.make_graph(
        [qx, qw, einsum, dq],
        'qdq_einsum_test',
        [x],
        [y],
        [w, scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_unsupported_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])
    node = helper.make_node('CustomUnsupportedOp', inputs=['input'], outputs=['output'])
    graph = helper.make_graph([node], 'custom_unsupported_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_conv_transpose_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2, 4, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 6, 6])
    w = np.random.randn(2, 3, 3, 3).astype(np.float32)
    b = np.random.randn(3).astype(np.float32)
    init_w = numpy_helper.from_array(w, name='W')
    init_b = numpy_helper.from_array(b, name='B')
    node = helper.make_node('ConvTranspose', inputs=['input', 'W', 'B'], outputs=['output'])
    graph = helper.make_graph([node], 'conv_transpose_test', [x], [y], [init_w, init_b])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_concat_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    c = np.array([[1.0, 2.0]], dtype=np.float32)
    init_c = numpy_helper.from_array(c, name='C')
    node = helper.make_node('Concat', inputs=['input', 'C'], outputs=['output'], axis=1)
    graph = helper.make_graph([node], 'concat_test', [x], [y], [init_c])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_transpose_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [3, 2])
    node = helper.make_node('Transpose', inputs=['input'], outputs=['output'], perm=[1, 0])
    graph = helper.make_graph([node], 'transpose_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_transpose5d_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2, 3, 4, 5])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4, 3, 5, 2])
    node = helper.make_node('Transpose', inputs=['input'], outputs=['output'], perm=[0, 3, 2, 4, 1])
    graph = helper.make_graph([node], 'transpose5d_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_pad_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [3, 4])
    node = helper.make_node('Pad', inputs=['input'], outputs=['output'], pads=[1, 1, 1, 1], value=0.5)
    graph = helper.make_graph([node], 'pad_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_pad5d_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2, 3, 2, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 4, 3, 3])
    pads = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
    node = helper.make_node('Pad', inputs=['input'], outputs=['output'], pads=pads, value=1.25)
    graph = helper.make_graph([node], 'pad5d_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_slice_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 2])
    starts = numpy_helper.from_array(np.array([1, 1], dtype=np.int64), name='starts')
    ends = numpy_helper.from_array(np.array([3, 3], dtype=np.int64), name='ends')
    axes = numpy_helper.from_array(np.array([0, 1], dtype=np.int64), name='axes')
    node = helper.make_node('Slice', inputs=['input', 'starts', 'ends', 'axes'], outputs=['output'])
    graph = helper.make_graph([node], 'slice_test', [x], [y], [starts, ends, axes])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_slice5d_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 4, 5, 6, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3, 4, 4, 3])
    starts = numpy_helper.from_array(np.array([0, 1, 1, 2, 0], dtype=np.int64), name='starts')
    ends = numpy_helper.from_array(np.array([2, 4, 5, 6, 3], dtype=np.int64), name='ends')
    axes = numpy_helper.from_array(np.array([0, 1, 2, 3, 4], dtype=np.int64), name='axes')
    node = helper.make_node('Slice', inputs=['input', 'starts', 'ends', 'axes'], outputs=['output'])
    graph = helper.make_graph([node], 'slice5d_test', [x], [y], [starts, ends, axes])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_cumsum_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])
    axis = numpy_helper.from_array(np.array(1, dtype=np.int64), name='axis')
    node = helper.make_node('CumSum', inputs=['input', 'axis'], outputs=['output'])
    graph = helper.make_graph([node], 'cumsum_test', [x], [y], [axis])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_shrink_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 6])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 6])
    node = helper.make_node('Shrink', inputs=['input'], outputs=['output'], lambd=0.5, bias=0.1)
    graph = helper.make_graph([node], 'shrink_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_eyelike_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [3, 4])
    node = helper.make_node('EyeLike', inputs=['input'], outputs=['output'], k=1)
    graph = helper.make_graph([node], 'eyelike_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_onehot_model(path: str) -> None:
    idx = helper.make_tensor_value_info('indices', TensorProto.INT16, [2, 3])
    out = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3, 4])
    depth = numpy_helper.from_array(np.array(4, dtype=np.int64), name='depth')
    values = numpy_helper.from_array(np.array([0.0, 1.0], dtype=np.float32), name='values')
    node = helper.make_node('OneHot', inputs=['indices', 'depth', 'values'], outputs=['output'], axis=-1)
    graph = helper.make_graph([node], 'onehot_test', [idx], [out], [depth, values])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_scatter_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])
    idx = numpy_helper.from_array(np.array([[2, 1, 0], [0, 1, 2]], dtype=np.int64), name='indices')
    upd = numpy_helper.from_array(np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]], dtype=np.float32), name='updates')
    node = helper.make_node('Scatter', inputs=['input', 'indices', 'updates'], outputs=['output'], axis=1)
    graph = helper.make_graph([node], 'scatter_test', [x], [y], [idx, upd])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 11)])
    onnx.save(model, path)


def _build_scatter_compat_model(path: str, op_type: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])
    idx = numpy_helper.from_array(np.array([[2, 1, 0], [0, 1, 2]], dtype=np.int64), name='indices')
    upd = numpy_helper.from_array(np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]], dtype=np.float32), name='updates')
    node = helper.make_node(op_type, inputs=['input', 'indices', 'updates'], outputs=['output'], axis=1)
    graph = helper.make_graph([node], f'scatter_compat_{op_type.lower()}', [x], [y], [idx, upd])
    opset = 11 if op_type == 'Scatter' else 13
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', opset)])
    onnx.save(model, path)


def _build_scatter_invalid_reduction_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])
    idx = numpy_helper.from_array(np.array([[2, 1, 0], [0, 1, 2]], dtype=np.int64), name='indices')
    upd = numpy_helper.from_array(np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]], dtype=np.float32), name='updates')
    node = helper.make_node(
        'Scatter',
        inputs=['input', 'indices', 'updates'],
        outputs=['output'],
        axis=1,
        reduction='add',
    )
    graph = helper.make_graph([node], 'scatter_invalid_reduction_test', [x], [y], [idx, upd])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 11)])
    onnx.save(model, path)


def _build_det_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [])
    node = helper.make_node('Det', inputs=['input'], outputs=['output'])
    graph = helper.make_graph([node], 'det_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_reverse_sequence_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [4, 3])
    seq = numpy_helper.from_array(np.array([4, 2, 3], dtype=np.int64), name='seq')
    node = helper.make_node(
        'ReverseSequence',
        inputs=['input', 'seq'],
        outputs=['output'],
        time_axis=0,
        batch_axis=1,
    )
    graph = helper.make_graph([node], 'reverse_sequence_test', [x], [y], [seq])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_non_max_suppression_model(path: str) -> None:
    boxes = helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [1, 4, 4])
    output = helper.make_tensor_value_info('output', TensorProto.INT64, [3, 3])
    scores_init = numpy_helper.from_array(
        np.array([[[0.90, 0.80, 0.75, 0.45]]], dtype=np.float32),
        name='scores',
    )
    max_output = numpy_helper.from_array(np.array([3], dtype=np.int64), name='max_output')
    iou_threshold = numpy_helper.from_array(np.array([0.5], dtype=np.float32), name='iou_threshold')
    score_threshold = numpy_helper.from_array(np.array([0.4], dtype=np.float32), name='score_threshold')
    node = helper.make_node(
        'NonMaxSuppression',
        inputs=['boxes', 'scores', 'max_output', 'iou_threshold', 'score_threshold'],
        outputs=['output'],
        center_point_box=0,
    )
    graph = helper.make_graph(
        [node],
        'nms_test',
        [boxes],
        [output],
        [scores_init, max_output, iou_threshold, score_threshold],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 12)])
    onnx.save(model, path)


def _build_non_max_suppression_dynamic_scalars_model(path: str) -> None:
    boxes = helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [1, 4, 4])
    output = helper.make_tensor_value_info('output', TensorProto.INT64, [3, 3])
    scores_init = numpy_helper.from_array(
        np.array([[[0.90, 0.80, 0.75, 0.45]]], dtype=np.float32),
        name='scores',
    )
    gather_index = numpy_helper.from_array(np.array([1], dtype=np.int64), name='gather_index')
    iou_threshold_init = numpy_helper.from_array(np.array([0.5], dtype=np.float32), name='iou_threshold_init')
    score_threshold_init = numpy_helper.from_array(np.array([0.4], dtype=np.float32), name='score_threshold_init')
    shape_node = helper.make_node(
        'Shape',
        inputs=['boxes'],
        outputs=['boxes_shape'],
    )
    max_output_gather = helper.make_node(
        'Gather',
        inputs=['boxes_shape', 'gather_index'],
        outputs=['max_output'],
    )
    iou_identity = helper.make_node(
        'Identity',
        inputs=['iou_threshold_init'],
        outputs=['iou_threshold'],
    )
    score_identity = helper.make_node(
        'Identity',
        inputs=['score_threshold_init'],
        outputs=['score_threshold'],
    )
    nms = helper.make_node(
        'NonMaxSuppression',
        inputs=['boxes', 'scores', 'max_output', 'iou_threshold', 'score_threshold'],
        outputs=['output'],
        center_point_box=0,
    )
    graph = helper.make_graph(
        [shape_node, max_output_gather, iou_identity, score_identity, nms],
        'nms_dynamic_scalar_test',
        [boxes],
        [output],
        [scores_init, gather_index, iou_threshold_init, score_threshold_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 12)])
    onnx.save(model, path)


def _build_dynamic_quantize_linear_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 8])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 8])
    n1 = helper.make_node(
        'DynamicQuantizeLinear',
        inputs=['input'],
        outputs=['q', 'q_scale', 'q_zero'],
    )
    n2 = helper.make_node(
        'DequantizeLinear',
        inputs=['q', 'q_scale', 'q_zero'],
        outputs=['output'],
    )
    value_info = [
        helper.make_tensor_value_info('q', TensorProto.UINT8, [1, 8]),
        helper.make_tensor_value_info('q_scale', TensorProto.FLOAT, []),
        helper.make_tensor_value_info('q_zero', TensorProto.UINT8, []),
    ]
    graph = helper.make_graph([n1, n2], 'dynamic_quantize_linear_test', [x], [y], value_info=value_info)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_roi_align_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 5, 5])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 1, 2, 2])
    rois = numpy_helper.from_array(
        np.array(
            [
                [0.0, 0.0, 4.0, 4.0],
                [1.0, 1.0, 4.0, 4.0],
            ],
            dtype=np.float32,
        ),
        name='rois',
    )
    batch_indices = numpy_helper.from_array(np.array([0, 0], dtype=np.int64), name='batch_indices')
    node = helper.make_node(
        'RoiAlign',
        inputs=['input', 'rois', 'batch_indices'],
        outputs=['output'],
        output_height=2,
        output_width=2,
        sampling_ratio=2,
        spatial_scale=1.0,
        mode='avg',
    )
    graph = helper.make_graph([node], 'roi_align_test', [x], [y], [rois, batch_indices])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 12)])
    onnx.save(model, path)


def _build_bitshift_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.INT16, [2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.INT16, [2, 3])
    shift = numpy_helper.from_array(np.array([1], dtype=np.int16), name='shift')
    node = helper.make_node('BitShift', inputs=['input', 'shift'], outputs=['output'], direction='RIGHT')
    graph = helper.make_graph([node], 'bitshift_test', [x], [y], [shift])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_compress_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 2])
    cond = numpy_helper.from_array(np.array([True, False, True, False], dtype=np.bool_), name='cond')
    node = helper.make_node('Compress', inputs=['input', 'cond'], outputs=['output'], axis=1)
    graph = helper.make_graph([node], 'compress_test', [x], [y], [cond])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_matmul_integer_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.INT8, [2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 2])
    w = numpy_helper.from_array(np.array([[1, -2], [3, 4], [-1, 2]], dtype=np.int8), name='W')
    a_zp = numpy_helper.from_array(np.array([0], dtype=np.int8), name='a_zero')
    b_zp = numpy_helper.from_array(np.array([0], dtype=np.int8), name='b_zero')
    node_mm = helper.make_node(
        'MatMulInteger',
        inputs=['input', 'W', 'a_zero', 'b_zero'],
        outputs=['mm_i32'],
    )
    node_cast = helper.make_node('Cast', inputs=['mm_i32'], outputs=['output'], to=TensorProto.FLOAT)
    value_info = [helper.make_tensor_value_info('mm_i32', TensorProto.INT32, [2, 2])]
    graph = helper.make_graph([node_mm, node_cast], 'matmul_integer_test', [x], [y], [w, a_zp, b_zp], value_info=value_info)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qlinear_matmul_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.INT8, [2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.INT8, [2, 2])
    w = numpy_helper.from_array(np.array([[1, -2], [3, 4], [-1, 2]], dtype=np.int8), name='W')
    a_scale = numpy_helper.from_array(np.array([0.125], dtype=np.float32), name='a_scale')
    b_scale = numpy_helper.from_array(np.array([0.125], dtype=np.float32), name='b_scale')
    y_scale = numpy_helper.from_array(np.array([0.25], dtype=np.float32), name='y_scale')
    a_zero = numpy_helper.from_array(np.array([0], dtype=np.int8), name='a_zero')
    b_zero = numpy_helper.from_array(np.array([0], dtype=np.int8), name='b_zero')
    y_zero = numpy_helper.from_array(np.array([0], dtype=np.int8), name='y_zero')
    node = helper.make_node(
        'QLinearMatMul',
        inputs=['input', 'a_scale', 'a_zero', 'W', 'b_scale', 'b_zero', 'y_scale', 'y_zero'],
        outputs=['output'],
    )
    graph = helper.make_graph(
        [node],
        'qlinear_matmul_test',
        [x],
        [y],
        [w, a_scale, b_scale, y_scale, a_zero, b_zero, y_zero],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_matmul_integer_batched_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.INT8, [2, 2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 2, 4])
    w = numpy_helper.from_array(
        np.array(
            [
                [[1, -2, 0, 3], [2, 1, -1, 0], [-1, 2, 1, 1]],
            ],
            dtype=np.int8,
        ),
        name='W',
    )
    a_zero = numpy_helper.from_array(np.array([0], dtype=np.int8), name='a_zero')
    b_zero = numpy_helper.from_array(np.array([0], dtype=np.int8), name='b_zero')
    node_mm = helper.make_node(
        'MatMulInteger',
        inputs=['input', 'W', 'a_zero', 'b_zero'],
        outputs=['mm_i32'],
    )
    node_cast = helper.make_node('Cast', inputs=['mm_i32'], outputs=['output'], to=TensorProto.FLOAT)
    value_info = [helper.make_tensor_value_info('mm_i32', TensorProto.INT32, [2, 2, 4])]
    graph = helper.make_graph(
        [node_mm, node_cast],
        'matmul_integer_batched_test',
        [x],
        [y],
        [w, a_zero, b_zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qlinear_matmul_batched_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.INT8, [2, 2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.INT8, [2, 2, 4])
    w = numpy_helper.from_array(
        np.array(
            [
                [[1, -2, 0, 3], [2, 1, -1, 0], [-1, 2, 1, 1]],
            ],
            dtype=np.int8,
        ),
        name='W',
    )
    a_scale = numpy_helper.from_array(np.array([0.125], dtype=np.float32), name='a_scale')
    b_scale = numpy_helper.from_array(np.array([0.0625], dtype=np.float32), name='b_scale')
    y_scale = numpy_helper.from_array(np.array([0.25], dtype=np.float32), name='y_scale')
    a_zero = numpy_helper.from_array(np.array([0], dtype=np.int8), name='a_zero')
    b_zero = numpy_helper.from_array(np.array([0], dtype=np.int8), name='b_zero')
    y_zero = numpy_helper.from_array(np.array([0], dtype=np.int8), name='y_zero')
    node = helper.make_node(
        'QLinearMatMul',
        inputs=['input', 'a_scale', 'a_zero', 'W', 'b_scale', 'b_zero', 'y_scale', 'y_zero'],
        outputs=['output'],
    )
    graph = helper.make_graph(
        [node],
        'qlinear_matmul_batched_test',
        [x],
        [y],
        [w, a_scale, b_scale, y_scale, a_zero, b_zero, y_zero],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_rnn_reverse_seq_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4, 1, 1])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [4, 1, 1, 1])
    w = numpy_helper.from_array(np.array([[[1.0]]], dtype=np.float32), name='W')
    r = numpy_helper.from_array(np.array([[[0.5]]], dtype=np.float32), name='R')
    b = numpy_helper.from_array(np.array([[0.0, 0.0]], dtype=np.float32), name='B')
    seq_lens = numpy_helper.from_array(np.array([2], dtype=np.int32), name='seq_lens')
    node = helper.make_node(
        'RNN',
        inputs=['input', 'W', 'R', 'B', 'seq_lens'],
        outputs=['output'],
        direction='reverse',
        hidden_size=1,
    )
    graph = helper.make_graph([node], 'rnn_reverse_seq_test', [x], [y], [w, r, b, seq_lens])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_rnn_bidirectional_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4, 1, 1])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [4, 2, 1, 1])
    w = numpy_helper.from_array(np.array([[[1.0]], [[1.0]]], dtype=np.float32), name='W')
    r = numpy_helper.from_array(np.array([[[0.5]], [[-0.25]]], dtype=np.float32), name='R')
    b = numpy_helper.from_array(np.zeros((2, 2), dtype=np.float32), name='B')
    node = helper.make_node(
        'RNN',
        inputs=['input', 'W', 'R', 'B'],
        outputs=['output'],
        direction='bidirectional',
        hidden_size=1,
    )
    graph = helper.make_graph([node], 'rnn_bidirectional_test', [x], [y], [w, r, b])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_gru_reverse_lbr1_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4, 1, 1])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [4, 1, 1, 1])
    w_data = np.zeros((1, 3, 1), dtype=np.float32)
    r_data = np.zeros((1, 3, 1), dtype=np.float32)
    b_data = np.zeros((1, 6), dtype=np.float32)
    w_data[0, 2, 0] = 1.0
    r_data[0, 2, 0] = 1.0
    b_data[0, 0] = -100.0  # force z gate ~0 to expose candidate branch
    b_data[0, 5] = 1.0     # recurrent h bias, affects linear_before_reset path
    h0 = np.array([[[1.0]]], dtype=np.float32)
    seq_lens = np.array([2], dtype=np.int32)
    w = numpy_helper.from_array(w_data, name='W')
    r = numpy_helper.from_array(r_data, name='R')
    b = numpy_helper.from_array(b_data, name='B')
    init_h = numpy_helper.from_array(h0, name='initial_h')
    seq = numpy_helper.from_array(seq_lens, name='seq_lens')
    node = helper.make_node(
        'GRU',
        inputs=['input', 'W', 'R', 'B', 'seq_lens', 'initial_h'],
        outputs=['output'],
        direction='reverse',
        hidden_size=1,
        linear_before_reset=1,
    )
    graph = helper.make_graph([node], 'gru_reverse_lbr1_test', [x], [y], [w, r, b, seq, init_h])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_lstm_reverse_peephole_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4, 1, 1])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [4, 1, 1, 1])
    w_data = np.zeros((1, 4, 1), dtype=np.float32)
    r_data = np.zeros((1, 4, 1), dtype=np.float32)
    b_data = np.zeros((1, 8), dtype=np.float32)
    h0_data = np.zeros((1, 1, 1), dtype=np.float32)
    c0_data = np.ones((1, 1, 1), dtype=np.float32)
    p_data = np.array([[0.2, 0.3, 0.4]], dtype=np.float32)  # [i, o, f]
    seq_lens = np.array([2], dtype=np.int32)
    w_data[0, 3, 0] = 1.0  # candidate gate from input
    w = numpy_helper.from_array(w_data, name='W')
    r = numpy_helper.from_array(r_data, name='R')
    b = numpy_helper.from_array(b_data, name='B')
    h0 = numpy_helper.from_array(h0_data, name='initial_h')
    c0 = numpy_helper.from_array(c0_data, name='initial_c')
    p = numpy_helper.from_array(p_data, name='P')
    seq = numpy_helper.from_array(seq_lens, name='seq_lens')
    node = helper.make_node(
        'LSTM',
        inputs=['input', 'W', 'R', 'B', 'seq_lens', 'initial_h', 'initial_c', 'P'],
        outputs=['output'],
        direction='reverse',
        hidden_size=1,
    )
    graph = helper.make_graph([node], 'lstm_reverse_peephole_test', [x], [y], [w, r, b, seq, h0, c0, p])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_rnn_relu_clip_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4, 1, 1])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [4, 1, 1, 1])
    w = numpy_helper.from_array(np.array([[[1.0]]], dtype=np.float32), name='W')
    r = numpy_helper.from_array(np.array([[[0.25]]], dtype=np.float32), name='R')
    b = numpy_helper.from_array(np.array([[0.0, 0.0]], dtype=np.float32), name='B')
    node = helper.make_node(
        'RNN',
        inputs=['input', 'W', 'R', 'B'],
        outputs=['output'],
        activations=['Relu'],
        clip=0.5,
        hidden_size=1,
    )
    graph = helper.make_graph([node], 'rnn_relu_clip_test', [x], [y], [w, r, b])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_gru_hardsigmoid_relu_clip_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4, 1, 1])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [4, 1, 1, 1])
    w_data = np.zeros((1, 3, 1), dtype=np.float32)
    r_data = np.zeros((1, 3, 1), dtype=np.float32)
    b_data = np.zeros((1, 6), dtype=np.float32)
    w_data[0, 2, 0] = 1.0
    r_data[0, 2, 0] = 0.5
    w = numpy_helper.from_array(w_data, name='W')
    r = numpy_helper.from_array(r_data, name='R')
    b = numpy_helper.from_array(b_data, name='B')
    node = helper.make_node(
        'GRU',
        inputs=['input', 'W', 'R', 'B'],
        outputs=['output'],
        activations=['HardSigmoid', 'Relu'],
        clip=0.5,
        hidden_size=1,
    )
    graph = helper.make_graph([node], 'gru_hardsigmoid_relu_clip_test', [x], [y], [w, r, b])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_lstm_input_forget_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4, 1, 1])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [4, 1, 1, 1])
    w_data = np.zeros((1, 4, 1), dtype=np.float32)
    r_data = np.zeros((1, 4, 1), dtype=np.float32)
    b_data = np.zeros((1, 8), dtype=np.float32)
    w_data[0, 3, 0] = 1.0
    w = numpy_helper.from_array(w_data, name='W')
    r = numpy_helper.from_array(r_data, name='R')
    b = numpy_helper.from_array(b_data, name='B')
    node = helper.make_node(
        'LSTM',
        inputs=['input', 'W', 'R', 'B'],
        outputs=['output'],
        activations=['HardSigmoid', 'Tanh', 'Relu'],
        input_forget=1,
        clip=0.5,
        hidden_size=1,
    )
    graph = helper.make_graph([node], 'lstm_input_forget_test', [x], [y], [w, r, b])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_rnn_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4, 1, 1])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [4, 1, 1, 1])
    w = numpy_helper.from_array(np.array([[[1.0]]], dtype=np.float32), name='W')
    r = numpy_helper.from_array(np.array([[[0.5]]], dtype=np.float32), name='R')
    b = numpy_helper.from_array(np.array([[0.0, 0.0]], dtype=np.float32), name='B')
    scale, zero = _qdq_params(qdtype)

    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    rnn = helper.make_node('RNN', inputs=['qx', 'W', 'R', 'B'], outputs=['qy'], hidden_size=1)
    dq = helper.make_node('DequantizeLinear', inputs=['qy', 'scale', 'zero'], outputs=['output'])

    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [4, 1, 1]),
        helper.make_tensor_value_info('qy', qdtype, [4, 1, 1, 1]),
    ]
    graph = helper.make_graph(
        [qx, rnn, dq],
        'qdq_rnn_test',
        [x],
        [y],
        [w, r, b, scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_gru_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4, 1, 1])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [4, 1, 1, 1])
    w_data = np.zeros((1, 3, 1), dtype=np.float32)
    r_data = np.zeros((1, 3, 1), dtype=np.float32)
    b_data = np.zeros((1, 6), dtype=np.float32)
    w_data[0, 2, 0] = 1.0
    r_data[0, 2, 0] = 0.5
    w = numpy_helper.from_array(w_data, name='W')
    r = numpy_helper.from_array(r_data, name='R')
    b = numpy_helper.from_array(b_data, name='B')
    scale, zero = _qdq_params(qdtype)

    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    gru = helper.make_node('GRU', inputs=['qx', 'W', 'R', 'B'], outputs=['qy'], hidden_size=1)
    dq = helper.make_node('DequantizeLinear', inputs=['qy', 'scale', 'zero'], outputs=['output'])

    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [4, 1, 1]),
        helper.make_tensor_value_info('qy', qdtype, [4, 1, 1, 1]),
    ]
    graph = helper.make_graph(
        [qx, gru, dq],
        'qdq_gru_test',
        [x],
        [y],
        [w, r, b, scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_lstm_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4, 1, 1])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [4, 1, 1, 1])
    w_data = np.zeros((1, 4, 1), dtype=np.float32)
    r_data = np.zeros((1, 4, 1), dtype=np.float32)
    b_data = np.zeros((1, 8), dtype=np.float32)
    w_data[0, 3, 0] = 1.0
    w = numpy_helper.from_array(w_data, name='W')
    r = numpy_helper.from_array(r_data, name='R')
    b = numpy_helper.from_array(b_data, name='B')
    scale, zero = _qdq_params(qdtype)

    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    lstm = helper.make_node('LSTM', inputs=['qx', 'W', 'R', 'B'], outputs=['qy'], hidden_size=1)
    dq = helper.make_node('DequantizeLinear', inputs=['qy', 'scale', 'zero'], outputs=['output'])

    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [4, 1, 1]),
        helper.make_tensor_value_info('qy', qdtype, [4, 1, 1, 1]),
    ]
    graph = helper.make_graph(
        [qx, lstm, dq],
        'qdq_lstm_test',
        [x],
        [y],
        [w, r, b, scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_conv_integer_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.INT8, [1, 2, 4, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 2, 2])
    w = numpy_helper.from_array(
        np.array(
            [
                [[[1, -1, 2], [0, 1, -2], [1, 0, 1]], [[-1, 1, 0], [2, -1, 1], [0, 1, -1]]],
                [[[2, 0, -1], [1, -2, 1], [0, 1, 1]], [[1, -1, 2], [0, 1, 0], [1, -2, 1]]],
                [[[0, 1, 1], [-1, 2, 0], [1, -1, 2]], [[2, 1, -1], [1, 0, 1], [-2, 1, 0]]],
            ],
            dtype=np.int8,
        ),
        name='W',
    )
    x_zero = numpy_helper.from_array(np.array([0], dtype=np.int8), name='x_zero')
    w_zero = numpy_helper.from_array(np.array([0], dtype=np.int8), name='w_zero')
    node_conv = helper.make_node(
        'ConvInteger',
        inputs=['input', 'W', 'x_zero', 'w_zero'],
        outputs=['conv_i32'],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
    )
    node_cast = helper.make_node('Cast', inputs=['conv_i32'], outputs=['output'], to=TensorProto.FLOAT)
    value_info = [helper.make_tensor_value_info('conv_i32', TensorProto.INT32, [1, 3, 2, 2])]
    graph = helper.make_graph(
        [node_conv, node_cast],
        'conv_integer_test',
        [x],
        [y],
        [w, x_zero, w_zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qlinear_conv_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.INT8, [1, 2, 4, 4])
    y = helper.make_tensor_value_info('output', TensorProto.INT8, [1, 3, 2, 2])
    w = numpy_helper.from_array(
        np.array(
            [
                [[[1, -1, 2], [0, 1, -2], [1, 0, 1]], [[-1, 1, 0], [2, -1, 1], [0, 1, -1]]],
                [[[2, 0, -1], [1, -2, 1], [0, 1, 1]], [[1, -1, 2], [0, 1, 0], [1, -2, 1]]],
                [[[0, 1, 1], [-1, 2, 0], [1, -1, 2]], [[2, 1, -1], [1, 0, 1], [-2, 1, 0]]],
            ],
            dtype=np.int8,
        ),
        name='W',
    )
    x_scale = numpy_helper.from_array(np.array([0.0625], dtype=np.float32), name='x_scale')
    w_scale = numpy_helper.from_array(np.array([0.05, 0.07, 0.09], dtype=np.float32), name='w_scale')
    y_scale = numpy_helper.from_array(np.array([0.125], dtype=np.float32), name='y_scale')
    x_zero = numpy_helper.from_array(np.array([0], dtype=np.int8), name='x_zero')
    w_zero = numpy_helper.from_array(np.array([0, 0, 0], dtype=np.int8), name='w_zero')
    y_zero = numpy_helper.from_array(np.array([0], dtype=np.int8), name='y_zero')
    bias = numpy_helper.from_array(np.array([3, -2, 5], dtype=np.int32), name='B')
    node = helper.make_node(
        'QLinearConv',
        inputs=['input', 'x_scale', 'x_zero', 'W', 'w_scale', 'w_zero', 'y_scale', 'y_zero', 'B'],
        outputs=['output'],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
    )
    graph = helper.make_graph(
        [node],
        'qlinear_conv_test',
        [x],
        [y],
        [w, x_scale, x_zero, w_scale, w_zero, y_scale, y_zero, bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_reduce_mean_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [])
    node = helper.make_node('ReduceMean', inputs=['input'], outputs=['output'], axes=[0, 1], keepdims=0)
    graph = helper.make_graph([node], 'reduce_mean_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_cast_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    n1 = helper.make_node('Cast', inputs=['input'], outputs=['x_i16'], to=TensorProto.INT16)
    n2 = helper.make_node('Cast', inputs=['x_i16'], outputs=['output'], to=TensorProto.FLOAT)
    value_info = [helper.make_tensor_value_info('x_i16', TensorProto.INT16, [1, 4])]
    graph = helper.make_graph([n1, n2], 'cast_test', [x], [y], value_info=value_info)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_gather_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])
    idx = numpy_helper.from_array(np.array([0, 2], dtype=np.int64), name='idx')
    node = helper.make_node('Gather', inputs=['input', 'idx'], outputs=['output'], axis=0)
    graph = helper.make_graph([node], 'gather_test', [x], [y], [idx])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_gathernd_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 4])
    idx = numpy_helper.from_array(
        np.array([[0, 1], [1, 2]], dtype=np.int64),
        name='idx',
    )
    node = helper.make_node('GatherND', inputs=['input', 'idx'], outputs=['output'])
    graph = helper.make_graph([node], 'gathernd_test', [x], [y], [idx])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_global_maxpool_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 1, 1])
    node = helper.make_node('GlobalMaxPool', inputs=['input'], outputs=['output'])
    graph = helper.make_graph([node], 'global_maxpool_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_maxpool_nd_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2, 3, 4, 5])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2, 2, 2, 2])
    node = helper.make_node(
        'MaxPool',
        inputs=['input'],
        outputs=['output'],
        kernel_shape=[2, 2, 2],
        strides=[1, 2, 2],
        pads=[0, 0, 0, 0, 0, 0],
    )
    graph = helper.make_graph([node], 'maxpool_nd_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_global_maxpool_nd_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 2, 2, 2, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 2, 1, 1, 1])
    node = helper.make_node('GlobalMaxPool', inputs=['input'], outputs=['output'])
    graph = helper.make_graph([node], 'global_maxpool_nd_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_expand_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 2])
    shape = numpy_helper.from_array(np.array([2, 2], dtype=np.int64), name='shape')
    node = helper.make_node('Expand', inputs=['input', 'shape'], outputs=['output'])
    graph = helper.make_graph([node], 'expand_test', [x], [y], [shape])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_where_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 2])
    cond = numpy_helper.from_array(np.array([[True, False], [False, True]], dtype=np.bool_), name='cond')
    c = numpy_helper.from_array(np.array([[1.0], [2.0]], dtype=np.float32), name='C')
    node = helper.make_node('Where', inputs=['cond', 'input', 'C'], outputs=['output'])
    graph = helper.make_graph([node], 'where_test', [x], [y], [cond, c])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_space_to_depth_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4, 2, 2])
    node = helper.make_node('SpaceToDepth', inputs=['input'], outputs=['output'], blocksize=2)
    graph = helper.make_graph([node], 'space_to_depth_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_depth_to_space_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4, 2, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 4, 4])
    node = helper.make_node('DepthToSpace', inputs=['input'], outputs=['output'], blocksize=2, mode='DCR')
    graph = helper.make_graph([node], 'depth_to_space_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_tile_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 4])
    reps = numpy_helper.from_array(np.array([2, 2], dtype=np.int64), name='reps')
    node = helper.make_node('Tile', inputs=['input', 'reps'], outputs=['output'])
    graph = helper.make_graph([node], 'tile_test', [x], [y], [reps])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_resize_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 2, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 4, 4])
    sizes = numpy_helper.from_array(np.array([1, 1, 4, 4], dtype=np.int64), name='sizes')
    node = helper.make_node(
        'Resize',
        inputs=['input', '', '', 'sizes'],
        outputs=['output'],
        mode='nearest',
        coordinate_transformation_mode='asymmetric',
        nearest_mode='floor',
    )
    graph = helper.make_graph([node], 'resize_test', [x], [y], [sizes])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_elu_selu_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    n1 = helper.make_node('Elu', inputs=['input'], outputs=['x1'], alpha=1.0)
    n2 = helper.make_node('Selu', inputs=['x1'], outputs=['output'])
    graph = helper.make_graph([n1, n2], 'elu_selu_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_sign_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    node = helper.make_node('Sign', inputs=['input'], outputs=['output'])
    graph = helper.make_graph([node], 'sign_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_compare_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    c = numpy_helper.from_array(np.array([[0.0, -1.0, 0.0, 1.0]], dtype=np.float32), name='C')
    n1 = helper.make_node('Equal', inputs=['input', 'C'], outputs=['eq'])
    n2 = helper.make_node('Greater', inputs=['input', 'C'], outputs=['gt'])
    n3 = helper.make_node('Less', inputs=['input', 'C'], outputs=['lt'])
    n4 = helper.make_node('GreaterOrEqual', inputs=['input', 'C'], outputs=['ge'])
    n5 = helper.make_node('LessOrEqual', inputs=['input', 'C'], outputs=['le'])
    c1 = helper.make_node('Cast', inputs=['eq'], outputs=['eqf'], to=TensorProto.FLOAT)
    c2 = helper.make_node('Cast', inputs=['gt'], outputs=['gtf'], to=TensorProto.FLOAT)
    c3 = helper.make_node('Cast', inputs=['lt'], outputs=['ltf'], to=TensorProto.FLOAT)
    c4 = helper.make_node('Cast', inputs=['ge'], outputs=['gef'], to=TensorProto.FLOAT)
    c5 = helper.make_node('Cast', inputs=['le'], outputs=['lef'], to=TensorProto.FLOAT)
    a1 = helper.make_node('Add', inputs=['eqf', 'gtf'], outputs=['s1'])
    a2 = helper.make_node('Add', inputs=['ltf', 'gef'], outputs=['s2'])
    a3 = helper.make_node('Add', inputs=['s1', 's2'], outputs=['s3'])
    a4 = helper.make_node('Add', inputs=['s3', 'lef'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('eq', TensorProto.BOOL, [1, 4]),
        helper.make_tensor_value_info('gt', TensorProto.BOOL, [1, 4]),
        helper.make_tensor_value_info('lt', TensorProto.BOOL, [1, 4]),
        helper.make_tensor_value_info('ge', TensorProto.BOOL, [1, 4]),
        helper.make_tensor_value_info('le', TensorProto.BOOL, [1, 4]),
        helper.make_tensor_value_info('eqf', TensorProto.FLOAT, [1, 4]),
        helper.make_tensor_value_info('gtf', TensorProto.FLOAT, [1, 4]),
        helper.make_tensor_value_info('ltf', TensorProto.FLOAT, [1, 4]),
        helper.make_tensor_value_info('gef', TensorProto.FLOAT, [1, 4]),
        helper.make_tensor_value_info('lef', TensorProto.FLOAT, [1, 4]),
        helper.make_tensor_value_info('s1', TensorProto.FLOAT, [1, 4]),
        helper.make_tensor_value_info('s2', TensorProto.FLOAT, [1, 4]),
        helper.make_tensor_value_info('s3', TensorProto.FLOAT, [1, 4]),
    ]
    graph = helper.make_graph(
        [n1, n2, n3, n4, n5, c1, c2, c3, c4, c5, a1, a2, a3, a4],
        'compare_test',
        [x],
        [y],
        [c],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_logic_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    c0 = numpy_helper.from_array(np.array([[0.0, 0.5, -0.5, 1.5]], dtype=np.float32), name='C0')
    c1 = numpy_helper.from_array(np.array([[0.0, -0.5, 0.5, 1.0]], dtype=np.float32), name='C1')
    gt = helper.make_node('Greater', inputs=['input', 'C0'], outputs=['gt'])
    le = helper.make_node('LessOrEqual', inputs=['input', 'C1'], outputs=['le'])
    nt = helper.make_node('Not', inputs=['le'], outputs=['nle'])
    a1 = helper.make_node('And', inputs=['gt', 'nle'], outputs=['b1'])
    o1 = helper.make_node('Or', inputs=['gt', 'le'], outputs=['b2'])
    x1 = helper.make_node('Xor', inputs=['b1', 'b2'], outputs=['b3'])
    c3 = helper.make_node('Cast', inputs=['b3'], outputs=['output'], to=TensorProto.FLOAT)
    value_info = [
        helper.make_tensor_value_info('gt', TensorProto.BOOL, [1, 4]),
        helper.make_tensor_value_info('le', TensorProto.BOOL, [1, 4]),
        helper.make_tensor_value_info('nle', TensorProto.BOOL, [1, 4]),
        helper.make_tensor_value_info('b1', TensorProto.BOOL, [1, 4]),
        helper.make_tensor_value_info('b2', TensorProto.BOOL, [1, 4]),
        helper.make_tensor_value_info('b3', TensorProto.BOOL, [1, 4]),
    ]
    graph = helper.make_graph(
        [gt, le, nt, a1, o1, x1, c3],
        'logic_test',
        [x],
        [y],
        [c0, c1],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_erf_round_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    n1 = helper.make_node('Erf', inputs=['input'], outputs=['x1'])
    n2 = helper.make_node('Round', inputs=['x1'], outputs=['output'])
    graph = helper.make_graph([n1, n2], 'erf_round_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_trig_hyper_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    one = numpy_helper.from_array(np.ones((1, 4), dtype=np.float32), name='ONE')

    n_tanh = helper.make_node('Tanh', inputs=['input'], outputs=['x_tanh'])
    n_tan = helper.make_node('Tan', inputs=['input'], outputs=['x_tan'])
    n_atan = helper.make_node('Atan', inputs=['x_tan'], outputs=['x_atan'])
    n_asin = helper.make_node('Asin', inputs=['x_tanh'], outputs=['x_asin'])
    n_acos = helper.make_node('Acos', inputs=['x_tanh'], outputs=['x_acos'])
    n_clip = helper.make_node('Clip', inputs=['input'], outputs=['x_clip'], min=-0.9, max=0.9)
    n_atanh = helper.make_node('Atanh', inputs=['x_clip'], outputs=['x_atanh'])
    n_abs = helper.make_node('Abs', inputs=['input'], outputs=['x_abs'])
    n_shift = helper.make_node('Add', inputs=['x_abs', 'ONE'], outputs=['x_shift'])
    n_acosh = helper.make_node('Acosh', inputs=['x_shift'], outputs=['x_acosh'])
    n_asinh = helper.make_node('Asinh', inputs=['input'], outputs=['x_asinh'])
    n_cosh = helper.make_node('Cosh', inputs=['x_tanh'], outputs=['x_cosh'])
    n_sinh = helper.make_node('Sinh', inputs=['x_tanh'], outputs=['x_sinh'])

    a1 = helper.make_node('Add', inputs=['x_atan', 'x_asin'], outputs=['s1'])
    a2 = helper.make_node('Add', inputs=['s1', 'x_acos'], outputs=['s2'])
    a3 = helper.make_node('Add', inputs=['s2', 'x_atanh'], outputs=['s3'])
    a4 = helper.make_node('Add', inputs=['s3', 'x_acosh'], outputs=['s4'])
    a5 = helper.make_node('Add', inputs=['s4', 'x_asinh'], outputs=['s5'])
    a6 = helper.make_node('Add', inputs=['s5', 'x_cosh'], outputs=['s6'])
    a7 = helper.make_node('Add', inputs=['s6', 'x_sinh'], outputs=['output'])

    graph = helper.make_graph(
        [
            n_tanh,
            n_tan,
            n_atan,
            n_asin,
            n_acos,
            n_clip,
            n_atanh,
            n_abs,
            n_shift,
            n_acosh,
            n_asinh,
            n_cosh,
            n_sinh,
            a1,
            a2,
            a3,
            a4,
            a5,
            a6,
            a7,
        ],
        'trig_hyper_test',
        [x],
        [y],
        [one],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_tan_sinh_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])

    scale, zero = _qdq_params(qdtype)
    q1 = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['q1'])
    q2 = helper.make_node('Tan', inputs=['q1'], outputs=['q2'])
    q3 = helper.make_node('Atan', inputs=['q2'], outputs=['q3'])
    q4 = helper.make_node('Sinh', inputs=['q3'], outputs=['q4'])
    q5 = helper.make_node('Cosh', inputs=['q4'], outputs=['q5'])
    dq = helper.make_node('DequantizeLinear', inputs=['q5', 'scale', 'zero'], outputs=['output'])

    value_info = [
        helper.make_tensor_value_info('q1', qdtype, [1, 4]),
        helper.make_tensor_value_info('q2', qdtype, [1, 4]),
        helper.make_tensor_value_info('q3', qdtype, [1, 4]),
        helper.make_tensor_value_info('q4', qdtype, [1, 4]),
        helper.make_tensor_value_info('q5', qdtype, [1, 4]),
    ]

    graph = helper.make_graph(
        [q1, q2, q3, q4, q5, dq],
        'qdq_tan_sinh_test',
        [x],
        [y],
        [scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_argmax_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 4])
    n1 = helper.make_node('ArgMax', inputs=['input'], outputs=['idx'], axis=1, keepdims=1)
    n2 = helper.make_node('Cast', inputs=['idx'], outputs=['output'], to=TensorProto.FLOAT)
    value_info = [helper.make_tensor_value_info('idx', TensorProto.INT64, [1, 1, 4])]
    graph = helper.make_graph([n1, n2], 'argmax_test', [x], [y], value_info=value_info)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_argmin_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3])
    n1 = helper.make_node('ArgMin', inputs=['input'], outputs=['idx'], axis=2, keepdims=0, select_last_index=1)
    n2 = helper.make_node('Cast', inputs=['idx'], outputs=['output'], to=TensorProto.FLOAT)
    value_info = [helper.make_tensor_value_info('idx', TensorProto.INT64, [1, 3])]
    graph = helper.make_graph([n1, n2], 'argmin_test', [x], [y], value_info=value_info)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_reduce_axes_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1])
    n1 = helper.make_node('ReduceMean', inputs=['input'], outputs=['r1'], axes=[1], keepdims=1)
    n2 = helper.make_node('ReduceSum', inputs=['r1'], outputs=['output'], axes=[2], keepdims=0)
    value_info = [helper.make_tensor_value_info('r1', TensorProto.FLOAT, [1, 1, 3])]
    graph = helper.make_graph([n1, n2], 'reduce_axes_test', [x], [y], value_info=value_info)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_reduce_extreme_axes_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1])
    n1 = helper.make_node('ReduceMax', inputs=['input'], outputs=['r1'], axes=[2], keepdims=0)
    n2 = helper.make_node('ReduceMin', inputs=['r1'], outputs=['output'], axes=[1], keepdims=1)
    value_info = [helper.make_tensor_value_info('r1', TensorProto.FLOAT, [1, 2])]
    graph = helper.make_graph([n1, n2], 'reduce_extreme_axes_test', [x], [y], value_info=value_info)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_reduce_extended_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1])
    n1 = helper.make_node('ReduceProd', inputs=['input'], outputs=['r1'], axes=[2], keepdims=1)
    n2 = helper.make_node('ReduceL1', inputs=['r1'], outputs=['r2'], axes=[1], keepdims=1)
    n3 = helper.make_node('ReduceL2', inputs=['r2'], outputs=['r3'], axes=[2], keepdims=0)
    n4 = helper.make_node('ReduceSumSquare', inputs=['input'], outputs=['r4'], axes=[1], keepdims=0)
    n5 = helper.make_node('ReduceSum', inputs=['r4'], outputs=['r5'], axes=[1], keepdims=1)
    n6 = helper.make_node('Add', inputs=['r3', 'r5'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('r1', TensorProto.FLOAT, [1, 2, 1]),
        helper.make_tensor_value_info('r2', TensorProto.FLOAT, [1, 1, 1]),
        helper.make_tensor_value_info('r3', TensorProto.FLOAT, [1, 1]),
        helper.make_tensor_value_info('r4', TensorProto.FLOAT, [1, 3]),
        helper.make_tensor_value_info('r5', TensorProto.FLOAT, [1, 1]),
    ]
    graph = helper.make_graph([n1, n2, n3, n4, n5, n6], 'reduce_extended_test', [x], [y], value_info=value_info)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_reduce_extended_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1])
    scale, zero = _qdq_params(qdtype)
    q1 = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['q1'])
    q2 = helper.make_node('ReduceProd', inputs=['q1'], outputs=['q2'], axes=[2], keepdims=1)
    q3 = helper.make_node('ReduceL1', inputs=['q2'], outputs=['q3'], axes=[1], keepdims=1)
    q4 = helper.make_node('ReduceL2', inputs=['q3'], outputs=['q4'], axes=[2], keepdims=0)
    dq = helper.make_node('DequantizeLinear', inputs=['q4', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('q1', qdtype, [1, 2, 3]),
        helper.make_tensor_value_info('q2', qdtype, [1, 2, 1]),
        helper.make_tensor_value_info('q3', qdtype, [1, 1, 1]),
        helper.make_tensor_value_info('q4', qdtype, [1, 1]),
    ]
    graph = helper.make_graph(
        [q1, q2, q3, q4, dq],
        'qdq_reduce_extended_test',
        [x],
        [y],
        [scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_reduce_sum_square_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1])
    scale, zero = _qdq_params(qdtype)
    q1 = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['q1'])
    q2 = helper.make_node('ReduceSumSquare', inputs=['q1'], outputs=['q2'], axes=[2], keepdims=0)
    q3 = helper.make_node('ReduceSumSquare', inputs=['q2'], outputs=['q3'], axes=[1], keepdims=1)
    dq = helper.make_node('DequantizeLinear', inputs=['q3', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('q1', qdtype, [1, 2, 3]),
        helper.make_tensor_value_info('q2', qdtype, [1, 2]),
        helper.make_tensor_value_info('q3', qdtype, [1, 1]),
    ]
    graph = helper.make_graph(
        [q1, q2, q3, dq],
        'qdq_reduce_sum_square_test',
        [x],
        [y],
        [scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_squeeze_unsqueeze_axes_input_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 2, 3])
    sq_axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name='sq_axes')
    us_axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name='us_axes')
    n1 = helper.make_node('Squeeze', inputs=['input', 'sq_axes'], outputs=['s1'])
    n2 = helper.make_node('Unsqueeze', inputs=['s1', 'us_axes'], outputs=['u1'])
    n3 = helper.make_node('Add', inputs=['u1', 'input'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('s1', TensorProto.FLOAT, [1, 2, 3]),
        helper.make_tensor_value_info('u1', TensorProto.FLOAT, [1, 1, 2, 3]),
    ]
    graph = helper.make_graph(
        [n1, n2, n3],
        'squeeze_unsqueeze_axes_input_test',
        [x],
        [y],
        [sq_axes, us_axes],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_squeeze_unsqueeze_axes_input_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 2, 3])
    scale, zero = _qdq_params(qdtype)
    sq_axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name='sq_axes')
    us_axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name='us_axes')
    q1 = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['q1'])
    q2 = helper.make_node('Squeeze', inputs=['q1', 'sq_axes'], outputs=['q2'])
    q3 = helper.make_node('Unsqueeze', inputs=['q2', 'us_axes'], outputs=['q3'])
    dq = helper.make_node('DequantizeLinear', inputs=['q3', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('q1', qdtype, [1, 1, 2, 3]),
        helper.make_tensor_value_info('q2', qdtype, [1, 2, 3]),
        helper.make_tensor_value_info('q3', qdtype, [1, 1, 2, 3]),
    ]
    graph = helper.make_graph(
        [q1, q2, q3, dq],
        'qdq_squeeze_unsqueeze_axes_input_test',
        [x],
        [y],
        [scale, zero, sq_axes, us_axes],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_reduce_log_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1])
    n1 = helper.make_node('Abs', inputs=['input'], outputs=['x_abs'])
    n2 = helper.make_node('ReduceLogSumExp', inputs=['x_abs'], outputs=['r1'], axes=[1], keepdims=1)
    n3 = helper.make_node('ReduceLogSum', inputs=['r1'], outputs=['output'], axes=[2], keepdims=0)
    value_info = [helper.make_tensor_value_info('r1', TensorProto.FLOAT, [1, 1, 3])]
    graph = helper.make_graph([n1, n2, n3], 'reduce_log_test', [x], [y], value_info=value_info)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_shape_size_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [])
    n1 = helper.make_node('Shape', inputs=['input'], outputs=['sh'])
    n2 = helper.make_node('Size', inputs=['input'], outputs=['sz'])
    n3 = helper.make_node('Cast', inputs=['sh'], outputs=['shf'], to=TensorProto.FLOAT)
    n4 = helper.make_node('Cast', inputs=['sz'], outputs=['szf'], to=TensorProto.FLOAT)
    n5 = helper.make_node('ReduceSum', inputs=['shf'], outputs=['shsum'], axes=[0], keepdims=0)
    n6 = helper.make_node('Add', inputs=['shsum', 'szf'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('sh', TensorProto.INT64, [4]),
        helper.make_tensor_value_info('sz', TensorProto.INT64, []),
        helper.make_tensor_value_info('shf', TensorProto.FLOAT, [4]),
        helper.make_tensor_value_info('szf', TensorProto.FLOAT, []),
        helper.make_tensor_value_info('shsum', TensorProto.FLOAT, []),
    ]
    graph = helper.make_graph([n1, n2, n3, n4, n5, n6], 'shape_size_test', [x], [y], value_info=value_info)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_constant_of_shape_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 2, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 2, 2])
    value = helper.make_tensor('const_value', TensorProto.FLOAT, [1], [0.25])
    n1 = helper.make_node('Shape', inputs=['input'], outputs=['shape'])
    n2 = helper.make_node('ConstantOfShape', inputs=['shape'], outputs=['k'], value=value)
    n3 = helper.make_node('Add', inputs=['k', 'input'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('shape', TensorProto.INT64, [4]),
        helper.make_tensor_value_info('k', TensorProto.FLOAT, [1, 1, 2, 2]),
    ]
    graph = helper.make_graph([n1, n2, n3], 'constant_of_shape_test', [x], [y], value_info=value_info)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_split_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 6])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 6])
    n1 = helper.make_node('Split', inputs=['input'], outputs=['a', 'b'], axis=1, split=[2, 4])
    n2 = helper.make_node('Concat', inputs=['b', 'a'], outputs=['output'], axis=1)
    value_info = [
        helper.make_tensor_value_info('a', TensorProto.FLOAT, [1, 2]),
        helper.make_tensor_value_info('b', TensorProto.FLOAT, [1, 4]),
    ]
    graph = helper.make_graph([n1, n2], 'split_test', [x], [y], value_info=value_info)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_mean_sum_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    c1 = numpy_helper.from_array(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32), name='C1')
    c2 = numpy_helper.from_array(np.array([[0.5, 1.5, 2.5, 3.5]], dtype=np.float32), name='C2')
    n1 = helper.make_node('Sum', inputs=['input', 'C1', 'C2'], outputs=['s1'])
    n2 = helper.make_node('Mean', inputs=['s1', 'C1'], outputs=['output'])
    value_info = [helper.make_tensor_value_info('s1', TensorProto.FLOAT, [1, 4])]
    graph = helper.make_graph([n1, n2], 'mean_sum_test', [x], [y], [c1, c2], value_info=value_info)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_add_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])

    scale, zero = _qdq_params(qdtype)
    c_float = numpy_helper.from_array(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32), name='c_float')

    q1 = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['q1'])
    q2 = helper.make_node('QuantizeLinear', inputs=['c_float', 'scale', 'zero'], outputs=['q2'])
    add = helper.make_node('Add', inputs=['q1', 'q2'], outputs=['q3'])
    dq = helper.make_node('DequantizeLinear', inputs=['q3', 'scale', 'zero'], outputs=['output'])

    value_info = [
        helper.make_tensor_value_info('q1', qdtype, [1, 4]),
        helper.make_tensor_value_info('q2', qdtype, [1, 4]),
        helper.make_tensor_value_info('q3', qdtype, [1, 4]),
    ]

    graph = helper.make_graph(
        [q1, q2, add, dq],
        'qdq_add_test',
        [x],
        [y],
        [scale, zero, c_float],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_split_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 6])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 6])
    scale, zero = _qdq_params(qdtype)
    n1 = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['q1'])
    n2 = helper.make_node('Split', inputs=['q1'], outputs=['qa', 'qb'], axis=1, split=[2, 4])
    n3 = helper.make_node('Concat', inputs=['qb', 'qa'], outputs=['q2'], axis=1)
    n4 = helper.make_node('DequantizeLinear', inputs=['q2', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('q1', qdtype, [1, 6]),
        helper.make_tensor_value_info('qa', qdtype, [1, 2]),
        helper.make_tensor_value_info('qb', qdtype, [1, 4]),
        helper.make_tensor_value_info('q2', qdtype, [1, 6]),
    ]
    graph = helper.make_graph([n1, n2, n3, n4], 'qdq_split_test', [x], [y], [scale, zero], value_info=value_info)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_mean_sum_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])

    scale, zero = _qdq_params(qdtype)
    c1_float = numpy_helper.from_array(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32), name='c1_float')
    c2_float = numpy_helper.from_array(np.array([[0.5, 1.5, 2.5, 3.5]], dtype=np.float32), name='c2_float')
    q1 = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['q1'])
    q2 = helper.make_node('QuantizeLinear', inputs=['c1_float', 'scale', 'zero'], outputs=['q2'])
    q3 = helper.make_node('QuantizeLinear', inputs=['c2_float', 'scale', 'zero'], outputs=['q3'])
    q4 = helper.make_node('Sum', inputs=['q1', 'q2', 'q3'], outputs=['q4'])
    q5 = helper.make_node('Mean', inputs=['q4', 'q2'], outputs=['q5'])
    dq = helper.make_node('DequantizeLinear', inputs=['q5', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('q1', qdtype, [1, 4]),
        helper.make_tensor_value_info('q2', qdtype, [1, 4]),
        helper.make_tensor_value_info('q3', qdtype, [1, 4]),
        helper.make_tensor_value_info('q4', qdtype, [1, 4]),
        helper.make_tensor_value_info('q5', qdtype, [1, 4]),
    ]
    graph = helper.make_graph(
        [q1, q2, q3, q4, q5, dq],
        'qdq_mean_sum_test',
        [x],
        [y],
        [scale, zero, c1_float, c2_float],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_mul_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])

    scale, zero = _qdq_params(qdtype)
    c_float = numpy_helper.from_array(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32), name='c_float')

    q1 = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['q1'])
    q2 = helper.make_node('QuantizeLinear', inputs=['c_float', 'scale', 'zero'], outputs=['q2'])
    mul = helper.make_node('Mul', inputs=['q1', 'q2'], outputs=['q3'])
    dq = helper.make_node('DequantizeLinear', inputs=['q3', 'scale', 'zero'], outputs=['output'])

    value_info = [
        helper.make_tensor_value_info('q1', qdtype, [1, 4]),
        helper.make_tensor_value_info('q2', qdtype, [1, 4]),
        helper.make_tensor_value_info('q3', qdtype, [1, 4]),
    ]

    graph = helper.make_graph(
        [q1, q2, mul, dq],
        'qdq_mul_test',
        [x],
        [y],
        [scale, zero, c_float],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_unary_binary_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])

    scale = numpy_helper.from_array(np.array([0.1], dtype=np.float32), name='scale')
    zero = numpy_helper.from_array(np.array([0], dtype=np.int8), name='zero')
    c_float = numpy_helper.from_array(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32), name='c_float')

    q1 = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['q1'])
    q2 = helper.make_node('QuantizeLinear', inputs=['c_float', 'scale', 'zero'], outputs=['q2'])
    qabs = helper.make_node('Abs', inputs=['q1'], outputs=['qabs'])
    qneg = helper.make_node('Neg', inputs=['qabs'], outputs=['qneg'])
    qclip = helper.make_node('Clip', inputs=['qneg'], outputs=['qclip'], min=-1.0, max=1.0)
    qmax = helper.make_node('Max', inputs=['qclip', 'q2'], outputs=['qmax'])
    qmin = helper.make_node('Min', inputs=['qmax', 'q2'], outputs=['qmin'])
    dq = helper.make_node('DequantizeLinear', inputs=['qmin', 'scale', 'zero'], outputs=['output'])

    graph = helper.make_graph(
        [q1, q2, qabs, qneg, qclip, qmax, qmin, dq],
        'qdq_unary_bin_test',
        [x],
        [y],
        [scale, zero, c_float],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)

def _build_qdq_unary_extra_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])

    scale = numpy_helper.from_array(np.array([0.1], dtype=np.float32), name='scale')
    zero = numpy_helper.from_array(np.array([0], dtype=np.int8), name='zero')

    q1 = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['q1'])
    q2 = helper.make_node('LeakyRelu', inputs=['q1'], outputs=['q2'], alpha=0.1)
    q3 = helper.make_node('Sigmoid', inputs=['q2'], outputs=['q3'])
    q4 = helper.make_node('Tanh', inputs=['q3'], outputs=['q4'])
    q5 = helper.make_node('Exp', inputs=['q4'], outputs=['q5'])
    q6 = helper.make_node('Log', inputs=['q5'], outputs=['q6'])
    q7 = helper.make_node('Sqrt', inputs=['q6'], outputs=['q7'])
    q8 = helper.make_node('Floor', inputs=['q7'], outputs=['q8'])
    q9 = helper.make_node('Ceil', inputs=['q8'], outputs=['q9'])
    dq = helper.make_node('DequantizeLinear', inputs=['q9', 'scale', 'zero'], outputs=['output'])

    value_info = [
        helper.make_tensor_value_info(name, TensorProto.INT8, [1, 4])
        for name in ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9']
    ]

    graph = helper.make_graph(
        [q1, q2, q3, q4, q5, q6, q7, q8, q9, dq],
        'qdq_unary_extra_test',
        [x],
        [y],
        [scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)

def _build_qdq_pow_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])

    scale = numpy_helper.from_array(np.array([0.1], dtype=np.float32), name='scale')
    zero = numpy_helper.from_array(np.array([0], dtype=np.int8), name='zero')
    exp_float = numpy_helper.from_array(np.array([[2.0, 2.0, 2.0, 2.0]], dtype=np.float32), name='exp_float')

    q1 = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['q1'])
    q2 = helper.make_node('QuantizeLinear', inputs=['exp_float', 'scale', 'zero'], outputs=['q2'])
    q3 = helper.make_node('Pow', inputs=['q1', 'q2'], outputs=['q3'])
    dq = helper.make_node('DequantizeLinear', inputs=['q3', 'scale', 'zero'], outputs=['output'])

    value_info = [
        helper.make_tensor_value_info(name, TensorProto.INT8, [1, 4])
        for name in ['q1', 'q2', 'q3']
    ]

    graph = helper.make_graph(
        [q1, q2, q3, dq],
        'qdq_pow_test',
        [x],
        [y],
        [scale, zero, exp_float],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)

def _qdq_params(dtype: int):
    scale = numpy_helper.from_array(np.array([0.1], dtype=np.float32), name='scale')
    if dtype == TensorProto.INT8:
        zero = numpy_helper.from_array(np.array([0], dtype=np.int8), name='zero')
    elif dtype == TensorProto.INT16:
        zero = numpy_helper.from_array(np.array([0], dtype=np.int16), name='zero')
    else:
        raise ValueError("Unsupported Q/DQ dtype.")
    return scale, zero


def _build_qdq_conv_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 2, 2])

    w = np.random.rand(1, 1, 3, 3).astype(np.float32)
    b = np.random.rand(1).astype(np.float32)
    init_w = numpy_helper.from_array(w, name='W')
    init_b = numpy_helper.from_array(b, name='B')
    scale, zero = _qdq_params(qdtype)

    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    qw = helper.make_node('QuantizeLinear', inputs=['W', 'scale', 'zero'], outputs=['qw'])
    conv = helper.make_node(
        'Conv',
        inputs=['qx', 'qw', 'B'],
        outputs=['qy'],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
    )
    dq = helper.make_node('DequantizeLinear', inputs=['qy', 'scale', 'zero'], outputs=['output'])

    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [1, 1, 4, 4]),
        helper.make_tensor_value_info('qw', qdtype, [1, 1, 3, 3]),
        helper.make_tensor_value_info('qy', qdtype, [1, 1, 2, 2]),
    ]

    graph = helper.make_graph(
        [qx, qw, conv, dq],
        'qdq_conv_test',
        [x],
        [y],
        [init_w, init_b, scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)

def _build_qdq_depthwise_conv_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2, 4, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2, 2, 2])

    w = np.random.rand(2, 1, 3, 3).astype(np.float32)
    b = np.random.rand(2).astype(np.float32)
    init_w = numpy_helper.from_array(w, name='W')
    init_b = numpy_helper.from_array(b, name='B')
    scale, zero = _qdq_params(qdtype)

    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    qw = helper.make_node('QuantizeLinear', inputs=['W', 'scale', 'zero'], outputs=['qw'])
    conv = helper.make_node(
        'Conv',
        inputs=['qx', 'qw', 'B'],
        outputs=['qy'],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
        group=2,
    )
    dq = helper.make_node('DequantizeLinear', inputs=['qy', 'scale', 'zero'], outputs=['output'])

    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [1, 2, 4, 4]),
        helper.make_tensor_value_info('qw', qdtype, [2, 1, 3, 3]),
        helper.make_tensor_value_info('qy', qdtype, [1, 2, 2, 2]),
    ]

    graph = helper.make_graph(
        [qx, qw, conv, dq],
        'qdq_dwconv_test',
        [x],
        [y],
        [init_w, init_b, scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_group_conv_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4, 4, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 6, 2, 2])

    w = np.random.rand(6, 2, 3, 3).astype(np.float32)
    b = np.random.rand(6).astype(np.float32)
    init_w = numpy_helper.from_array(w, name='W')
    init_b = numpy_helper.from_array(b, name='B')
    scale, zero = _qdq_params(qdtype)

    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    qw = helper.make_node('QuantizeLinear', inputs=['W', 'scale', 'zero'], outputs=['qw'])
    conv = helper.make_node(
        'Conv',
        inputs=['qx', 'qw', 'B'],
        outputs=['qy'],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
        group=2,
    )
    dq = helper.make_node('DequantizeLinear', inputs=['qy', 'scale', 'zero'], outputs=['output'])

    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [1, 4, 4, 4]),
        helper.make_tensor_value_info('qw', qdtype, [6, 2, 3, 3]),
        helper.make_tensor_value_info('qy', qdtype, [1, 6, 2, 2]),
    ]

    graph = helper.make_graph(
        [qx, qw, conv, dq],
        'qdq_group_conv_test',
        [x],
        [y],
        [init_w, init_b, scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_matmul_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])

    w = np.random.rand(3, 2).astype(np.float32)
    init_w = numpy_helper.from_array(w, name='W')
    scale, zero = _qdq_params(qdtype)

    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    qw = helper.make_node('QuantizeLinear', inputs=['W', 'scale', 'zero'], outputs=['qw'])
    mm = helper.make_node('MatMul', inputs=['qx', 'qw'], outputs=['qy'])
    dq = helper.make_node('DequantizeLinear', inputs=['qy', 'scale', 'zero'], outputs=['output'])

    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [1, 3]),
        helper.make_tensor_value_info('qw', qdtype, [3, 2]),
        helper.make_tensor_value_info('qy', qdtype, [1, 2]),
    ]

    graph = helper.make_graph(
        [qx, qw, mm, dq],
        'qdq_matmul_test',
        [x],
        [y],
        [init_w, scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)

def _build_qdq_gemm_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])

    w = np.random.rand(3, 2).astype(np.float32)
    b = np.random.rand(2).astype(np.float32)
    init_w = numpy_helper.from_array(w, name='W')
    init_b = numpy_helper.from_array(b, name='B')
    scale, zero = _qdq_params(qdtype)

    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    qw = helper.make_node('QuantizeLinear', inputs=['W', 'scale', 'zero'], outputs=['qw'])
    gemm = helper.make_node('Gemm', inputs=['qx', 'qw', 'B'], outputs=['qy'])
    dq = helper.make_node('DequantizeLinear', inputs=['qy', 'scale', 'zero'], outputs=['output'])

    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [1, 3]),
        helper.make_tensor_value_info('qw', qdtype, [3, 2]),
        helper.make_tensor_value_info('qy', qdtype, [1, 2]),
    ]

    graph = helper.make_graph(
        [qx, qw, gemm, dq],
        'qdq_gemm_test',
        [x],
        [y],
        [init_w, init_b, scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_maxpool_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 2, 2])
    scale, zero = _qdq_params(qdtype)
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    pool = helper.make_node(
        'MaxPool',
        inputs=['qx'],
        outputs=['qy'],
        kernel_shape=[2, 2],
        strides=[2, 2],
        pads=[0, 0, 0, 0],
    )
    dq = helper.make_node('DequantizeLinear', inputs=['qy', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [1, 1, 4, 4]),
        helper.make_tensor_value_info('qy', qdtype, [1, 1, 2, 2]),
    ]
    graph = helper.make_graph(
        [qx, pool, dq],
        'qdq_maxpool_test',
        [x],
        [y],
        [scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_maxpool_nd_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 3, 4, 5])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 2, 2, 2])
    scale, zero = _qdq_params(qdtype)
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    pool = helper.make_node(
        'MaxPool',
        inputs=['qx'],
        outputs=['qy'],
        kernel_shape=[2, 2, 2],
        strides=[1, 2, 2],
        pads=[0, 0, 0, 0, 0, 0],
    )
    dq = helper.make_node('DequantizeLinear', inputs=['qy', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [1, 1, 3, 4, 5]),
        helper.make_tensor_value_info('qy', qdtype, [1, 1, 2, 2, 2]),
    ]
    graph = helper.make_graph(
        [qx, pool, dq],
        'qdq_maxpool_nd_test',
        [x],
        [y],
        [scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_avgpool_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 2, 2])
    scale, zero = _qdq_params(qdtype)
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    pool = helper.make_node(
        'AveragePool',
        inputs=['qx'],
        outputs=['qy'],
        kernel_shape=[2, 2],
        strides=[2, 2],
        pads=[0, 0, 0, 0],
    )
    dq = helper.make_node('DequantizeLinear', inputs=['qy', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [1, 1, 4, 4]),
        helper.make_tensor_value_info('qy', qdtype, [1, 1, 2, 2]),
    ]
    graph = helper.make_graph(
        [qx, pool, dq],
        'qdq_avgpool_test',
        [x],
        [y],
        [scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_avgpool_nd_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 3, 4, 5])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 2, 2, 2])
    scale, zero = _qdq_params(qdtype)
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    pool = helper.make_node(
        'AveragePool',
        inputs=['qx'],
        outputs=['qy'],
        kernel_shape=[2, 2, 2],
        strides=[1, 2, 2],
        pads=[0, 0, 0, 0, 0, 0],
    )
    dq = helper.make_node('DequantizeLinear', inputs=['qy', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [1, 1, 3, 4, 5]),
        helper.make_tensor_value_info('qy', qdtype, [1, 1, 2, 2, 2]),
    ]
    graph = helper.make_graph(
        [qx, pool, dq],
        'qdq_avgpool_nd_test',
        [x],
        [y],
        [scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_global_avgpool_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 1, 1])
    scale, zero = _qdq_params(qdtype)
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    pool = helper.make_node('GlobalAveragePool', inputs=['qx'], outputs=['qy'])
    dq = helper.make_node('DequantizeLinear', inputs=['qy', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [1, 1, 4, 4]),
        helper.make_tensor_value_info('qy', qdtype, [1, 1, 1, 1]),
    ]
    graph = helper.make_graph(
        [qx, pool, dq],
        'qdq_global_avgpool_test',
        [x],
        [y],
        [scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_global_avgpool_nd_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2, 2, 2, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2, 1, 1, 1])
    scale, zero = _qdq_params(qdtype)
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    pool = helper.make_node('GlobalAveragePool', inputs=['qx'], outputs=['qy'])
    dq = helper.make_node('DequantizeLinear', inputs=['qy', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [1, 2, 2, 2, 2]),
        helper.make_tensor_value_info('qy', qdtype, [1, 2, 1, 1, 1]),
    ]
    graph = helper.make_graph(
        [qx, pool, dq],
        'qdq_global_avgpool_nd_test',
        [x],
        [y],
        [scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_global_maxpool_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 1, 1])
    scale, zero = _qdq_params(qdtype)
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    pool = helper.make_node('GlobalMaxPool', inputs=['qx'], outputs=['qy'])
    dq = helper.make_node('DequantizeLinear', inputs=['qy', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [1, 1, 4, 4]),
        helper.make_tensor_value_info('qy', qdtype, [1, 1, 1, 1]),
    ]
    graph = helper.make_graph(
        [qx, pool, dq],
        'qdq_global_maxpool_test',
        [x],
        [y],
        [scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_global_maxpool_nd_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2, 2, 2, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2, 1, 1, 1])
    scale, zero = _qdq_params(qdtype)
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    pool = helper.make_node('GlobalMaxPool', inputs=['qx'], outputs=['qy'])
    dq = helper.make_node('DequantizeLinear', inputs=['qy', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [1, 2, 2, 2, 2]),
        helper.make_tensor_value_info('qy', qdtype, [1, 2, 1, 1, 1]),
    ]
    graph = helper.make_graph(
        [qx, pool, dq],
        'qdq_global_maxpool_nd_test',
        [x],
        [y],
        [scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_relu_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    scale, zero = _qdq_params(qdtype)
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    relu = helper.make_node('Relu', inputs=['qx'], outputs=['qy'])
    dq = helper.make_node('DequantizeLinear', inputs=['qy', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [1, 4]),
        helper.make_tensor_value_info('qy', qdtype, [1, 4]),
    ]
    graph = helper.make_graph(
        [qx, relu, dq],
        'qdq_relu_test',
        [x],
        [y],
        [scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_softmax_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    scale, zero = _qdq_params(qdtype)
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    sm = helper.make_node('Softmax', inputs=['qx'], outputs=['qy'], axis=1)
    dq = helper.make_node('DequantizeLinear', inputs=['qy', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [1, 4]),
        helper.make_tensor_value_info('qy', qdtype, [1, 4]),
    ]
    graph = helper.make_graph(
        [qx, sm, dq],
        'qdq_softmax_test',
        [x],
        [y],
        [scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_logsoftmax_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    scale, zero = _qdq_params(qdtype)
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    sm = helper.make_node('LogSoftmax', inputs=['qx'], outputs=['qy'], axis=1)
    dq = helper.make_node('DequantizeLinear', inputs=['qy', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [1, 4]),
        helper.make_tensor_value_info('qy', qdtype, [1, 4]),
    ]
    graph = helper.make_graph(
        [qx, sm, dq],
        'qdq_logsoftmax_test',
        [x],
        [y],
        [scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_prelu_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    slope = numpy_helper.from_array(np.array([0.25], dtype=np.float32), name='slope')
    scale, zero = _qdq_params(qdtype)
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    prelu = helper.make_node('PRelu', inputs=['qx', 'slope'], outputs=['qy'])
    dq = helper.make_node('DequantizeLinear', inputs=['qy', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [1, 4]),
        helper.make_tensor_value_info('qy', qdtype, [1, 4]),
    ]
    graph = helper.make_graph(
        [qx, prelu, dq],
        'qdq_prelu_test',
        [x],
        [y],
        [slope, scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_mod_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    b = numpy_helper.from_array(np.array([[0.6, 0.7, 0.8, 0.9]], dtype=np.float32), name='B')
    scale, zero = _qdq_params(qdtype)
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    qb = helper.make_node('QuantizeLinear', inputs=['B', 'scale', 'zero'], outputs=['qb'])
    mod = helper.make_node('Mod', inputs=['qx', 'qb'], outputs=['qy'], fmod=1)
    dq = helper.make_node('DequantizeLinear', inputs=['qy', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [1, 4]),
        helper.make_tensor_value_info('qb', qdtype, [1, 4]),
        helper.make_tensor_value_info('qy', qdtype, [1, 4]),
    ]
    graph = helper.make_graph(
        [qx, qb, mod, dq],
        'qdq_mod_test',
        [x],
        [y],
        [b, scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_reduce_basic_model(path: str, op_type: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2, 2, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2, 1, 1])
    scale, zero = _qdq_params(qdtype)
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    rd = helper.make_node(op_type, inputs=['qx'], outputs=['qy'], axes=[2, 3], keepdims=1)
    dq = helper.make_node('DequantizeLinear', inputs=['qy', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [1, 2, 2, 2]),
        helper.make_tensor_value_info('qy', qdtype, [1, 2, 1, 1]),
    ]
    graph = helper.make_graph(
        [qx, rd, dq],
        'qdq_reduce_basic_test',
        [x],
        [y],
        [scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_rearrange_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 4, 4])
    scale, zero = _qdq_params(qdtype)
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    s2d = helper.make_node('SpaceToDepth', inputs=['qx'], outputs=['q1'], blocksize=2)
    d2s = helper.make_node('DepthToSpace', inputs=['q1'], outputs=['q2'], blocksize=2, mode='DCR')
    dq = helper.make_node('DequantizeLinear', inputs=['q2', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [1, 1, 4, 4]),
        helper.make_tensor_value_info('q1', qdtype, [1, 4, 2, 2]),
        helper.make_tensor_value_info('q2', qdtype, [1, 1, 4, 4]),
    ]
    graph = helper.make_graph(
        [qx, s2d, d2s, dq],
        'qdq_rearrange_test',
        [x],
        [y],
        [scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_gathernd_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 2])
    scale, zero = _qdq_params(qdtype)
    idx = numpy_helper.from_array(
        np.array([[0, 1], [1, 2]], dtype=np.int64),
        name='idx',
    )
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    gnd = helper.make_node('GatherND', inputs=['qx', 'idx'], outputs=['q1'])
    dq = helper.make_node('DequantizeLinear', inputs=['q1', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [2, 3, 2]),
        helper.make_tensor_value_info('q1', qdtype, [2, 2]),
    ]
    graph = helper.make_graph(
        [qx, gnd, dq],
        'qdq_gathernd_test',
        [x],
        [y],
        [scale, zero, idx],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_scatter_elements_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])
    scale, zero = _qdq_params(qdtype)
    idx = numpy_helper.from_array(np.array([[0, 2], [1, 0]], dtype=np.int64), name='idx')
    upd_f = numpy_helper.from_array(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), name='upd_f')
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    qu = helper.make_node('QuantizeLinear', inputs=['upd_f', 'scale', 'zero'], outputs=['qu'])
    sc = helper.make_node('ScatterElements', inputs=['qx', 'idx', 'qu'], outputs=['q1'], axis=1)
    dq = helper.make_node('DequantizeLinear', inputs=['q1', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [2, 3]),
        helper.make_tensor_value_info('qu', qdtype, [2, 2]),
        helper.make_tensor_value_info('q1', qdtype, [2, 3]),
    ]
    graph = helper.make_graph(
        [qx, qu, sc, dq],
        'qdq_scatter_elements_test',
        [x],
        [y],
        [scale, zero, idx, upd_f],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_scatter_nd_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])
    scale, zero = _qdq_params(qdtype)
    idx = numpy_helper.from_array(np.array([[0], [1]], dtype=np.int64), name='idx')
    upd_f = numpy_helper.from_array(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32), name='upd_f')
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    qu = helper.make_node('QuantizeLinear', inputs=['upd_f', 'scale', 'zero'], outputs=['qu'])
    sc = helper.make_node('ScatterND', inputs=['qx', 'idx', 'qu'], outputs=['q1'])
    dq = helper.make_node('DequantizeLinear', inputs=['q1', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [2, 3]),
        helper.make_tensor_value_info('qu', qdtype, [2, 3]),
        helper.make_tensor_value_info('q1', qdtype, [2, 3]),
    ]
    graph = helper.make_graph(
        [qx, qu, sc, dq],
        'qdq_scatter_nd_test',
        [x],
        [y],
        [scale, zero, idx, upd_f],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_scatter_elements_reduction_model(
    path: str,
    qdtype: int = TensorProto.INT8,
    reduction: str = 'add',
) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])
    scale, zero = _qdq_params(qdtype)
    idx = numpy_helper.from_array(np.array([[0, 0, 2], [1, 1, 1]], dtype=np.int64), name='idx')
    upd_f = numpy_helper.from_array(np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]], dtype=np.float32), name='upd_f')
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    qu = helper.make_node('QuantizeLinear', inputs=['upd_f', 'scale', 'zero'], outputs=['qu'])
    sc = helper.make_node(
        'ScatterElements',
        inputs=['qx', 'idx', 'qu'],
        outputs=['q1'],
        axis=1,
        reduction=reduction,
    )
    dq = helper.make_node('DequantizeLinear', inputs=['q1', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [2, 3]),
        helper.make_tensor_value_info('qu', qdtype, [2, 3]),
        helper.make_tensor_value_info('q1', qdtype, [2, 3]),
    ]
    graph = helper.make_graph(
        [qx, qu, sc, dq],
        f'qdq_scatter_elements_{reduction}_test',
        [x],
        [y],
        [scale, zero, idx, upd_f],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 18)])
    onnx.save(model, path)


def _build_qdq_scatter_nd_reduction_model(
    path: str,
    qdtype: int = TensorProto.INT8,
    reduction: str = 'add',
) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [4])
    scale, zero = _qdq_params(qdtype)
    idx = numpy_helper.from_array(np.array([[1], [1], [2]], dtype=np.int64), name='idx')
    upd_f = numpy_helper.from_array(np.array([0.3, 0.4, 0.5], dtype=np.float32), name='upd_f')
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    qu = helper.make_node('QuantizeLinear', inputs=['upd_f', 'scale', 'zero'], outputs=['qu'])
    sc = helper.make_node(
        'ScatterND',
        inputs=['qx', 'idx', 'qu'],
        outputs=['q1'],
        reduction=reduction,
    )
    dq = helper.make_node('DequantizeLinear', inputs=['q1', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [4]),
        helper.make_tensor_value_info('qu', qdtype, [3]),
        helper.make_tensor_value_info('q1', qdtype, [4]),
    ]
    graph = helper.make_graph(
        [qx, qu, sc, dq],
        f'qdq_scatter_nd_{reduction}_test',
        [x],
        [y],
        [scale, zero, idx, upd_f],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 18)])
    onnx.save(model, path)


def _build_qdq_reverse_sequence_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [4, 3])
    scale, zero = _qdq_params(qdtype)
    seq = numpy_helper.from_array(np.array([4, 2, 3], dtype=np.int64), name='seq')
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    rs = helper.make_node(
        'ReverseSequence',
        inputs=['qx', 'seq'],
        outputs=['q1'],
        time_axis=0,
        batch_axis=1,
    )
    dq = helper.make_node('DequantizeLinear', inputs=['q1', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [4, 3]),
        helper.make_tensor_value_info('q1', qdtype, [4, 3]),
    ]
    graph = helper.make_graph(
        [qx, rs, dq],
        'qdq_reverse_sequence_test',
        [x],
        [y],
        [scale, zero, seq],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_tile_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 4])
    scale, zero = _qdq_params(qdtype)
    reps = numpy_helper.from_array(np.array([2, 2], dtype=np.int64), name='reps')
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    tile = helper.make_node('Tile', inputs=['qx', 'reps'], outputs=['qy'])
    dq = helper.make_node('DequantizeLinear', inputs=['qy', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [1, 2]),
        helper.make_tensor_value_info('qy', qdtype, [2, 4]),
    ]
    graph = helper.make_graph(
        [qx, tile, dq],
        'qdq_tile_test',
        [x],
        [y],
        [scale, zero, reps],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_resize_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 2, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 4, 4])
    scale, zero = _qdq_params(qdtype)
    sizes = numpy_helper.from_array(np.array([1, 1, 4, 4], dtype=np.int64), name='sizes')
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    resize = helper.make_node(
        'Resize',
        inputs=['qx', '', '', 'sizes'],
        outputs=['qy'],
        mode='nearest',
        coordinate_transformation_mode='asymmetric',
        nearest_mode='floor',
    )
    dq = helper.make_node('DequantizeLinear', inputs=['qy', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [1, 1, 2, 2]),
        helper.make_tensor_value_info('qy', qdtype, [1, 1, 4, 4]),
    ]
    graph = helper.make_graph(
        [qx, resize, dq],
        'qdq_resize_test',
        [x],
        [y],
        [scale, zero, sizes],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_elu_selu_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    scale, zero = _qdq_params(qdtype)
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    n1 = helper.make_node('Elu', inputs=['qx'], outputs=['q1'], alpha=1.0)
    n2 = helper.make_node('Selu', inputs=['q1'], outputs=['q2'])
    dq = helper.make_node('DequantizeLinear', inputs=['q2', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [1, 4]),
        helper.make_tensor_value_info('q1', qdtype, [1, 4]),
        helper.make_tensor_value_info('q2', qdtype, [1, 4]),
    ]
    graph = helper.make_graph(
        [qx, n1, n2, dq],
        'qdq_elu_selu_test',
        [x],
        [y],
        [scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_sign_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    scale, zero = _qdq_params(qdtype)
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    sign = helper.make_node('Sign', inputs=['qx'], outputs=['q1'])
    dq = helper.make_node('DequantizeLinear', inputs=['q1', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [1, 4]),
        helper.make_tensor_value_info('q1', qdtype, [1, 4]),
    ]
    graph = helper.make_graph(
        [qx, sign, dq],
        'qdq_sign_test',
        [x],
        [y],
        [scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_erf_round_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    scale, zero = _qdq_params(qdtype)
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    n1 = helper.make_node('Erf', inputs=['qx'], outputs=['q1'])
    n2 = helper.make_node('Round', inputs=['q1'], outputs=['q2'])
    dq = helper.make_node('DequantizeLinear', inputs=['q2', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [1, 4]),
        helper.make_tensor_value_info('q1', qdtype, [1, 4]),
        helper.make_tensor_value_info('q2', qdtype, [1, 4]),
    ]
    graph = helper.make_graph(
        [qx, n1, n2, dq],
        'qdq_erf_round_test',
        [x],
        [y],
        [scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_unary_chain_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3])
    node_abs = helper.make_node('Abs', inputs=['input'], outputs=['a1'])
    node_exp = helper.make_node('Exp', inputs=['a1'], outputs=['a2'])
    node_log = helper.make_node('Log', inputs=['a2'], outputs=['a3'])
    node_sqrt = helper.make_node('Sqrt', inputs=['a3'], outputs=['a4'])
    node_floor = helper.make_node('Floor', inputs=['a4'], outputs=['a5'])
    node_ceil = helper.make_node('Ceil', inputs=['a5'], outputs=['a6'])
    node_neg = helper.make_node('Neg', inputs=['a6'], outputs=['output'])
    graph = helper.make_graph(
        [node_abs, node_exp, node_log, node_sqrt, node_floor, node_ceil, node_neg],
        'unary_chain_test',
        [x],
        [y],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_unary_extended_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    n1 = helper.make_node('Sin', inputs=['input'], outputs=['u1'])
    n2 = helper.make_node('Cos', inputs=['u1'], outputs=['u2'])
    n3 = helper.make_node('Softplus', inputs=['u2'], outputs=['u3'])
    n4 = helper.make_node('Softsign', inputs=['u3'], outputs=['u4'])
    n5 = helper.make_node('HardSigmoid', inputs=['u4'], outputs=['u5'], alpha=0.2, beta=0.5)
    n6 = helper.make_node('Reciprocal', inputs=['u5'], outputs=['output'])
    graph = helper.make_graph(
        [n1, n2, n3, n4, n5, n6],
        'unary_extended_test',
        [x],
        [y],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_qdq_unary_extended_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    scale, zero = _qdq_params(qdtype)
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['qx'])
    n1 = helper.make_node('Sin', inputs=['qx'], outputs=['q1'])
    n2 = helper.make_node('Cos', inputs=['q1'], outputs=['q2'])
    n3 = helper.make_node('Softplus', inputs=['q2'], outputs=['q3'])
    n4 = helper.make_node('Softsign', inputs=['q3'], outputs=['q4'])
    n5 = helper.make_node('HardSigmoid', inputs=['q4'], outputs=['q5'])
    n6 = helper.make_node('Reciprocal', inputs=['q5'], outputs=['q6'])
    dq = helper.make_node('DequantizeLinear', inputs=['q6', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('qx', qdtype, [1, 4]),
        helper.make_tensor_value_info('q1', qdtype, [1, 4]),
        helper.make_tensor_value_info('q2', qdtype, [1, 4]),
        helper.make_tensor_value_info('q3', qdtype, [1, 4]),
        helper.make_tensor_value_info('q4', qdtype, [1, 4]),
        helper.make_tensor_value_info('q5', qdtype, [1, 4]),
        helper.make_tensor_value_info('q6', qdtype, [1, 4]),
    ]
    graph = helper.make_graph(
        [qx, n1, n2, n3, n4, n5, n6, dq],
        'qdq_unary_extended_test',
        [x],
        [y],
        [scale, zero],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_pow_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    exp = numpy_helper.from_array(np.array([2.0], dtype=np.float32), name='exp')
    node = helper.make_node('Pow', inputs=['input', 'exp'], outputs=['output'])
    graph = helper.make_graph([node], 'pow_test', [x], [y], [exp])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_max_min_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])
    c = numpy_helper.from_array(np.array([[0.5, 2.0]], dtype=np.float32), name='C')
    node_max = helper.make_node('Max', inputs=['input', 'C'], outputs=['m1'])
    node_min = helper.make_node('Min', inputs=['m1', 'C'], outputs=['output'])
    graph = helper.make_graph([node_max, node_min], 'max_min_test', [x], [y], [c])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_reduce_max_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [])
    node_rmax = helper.make_node('ReduceMax', inputs=['input'], outputs=['output'], axes=[0, 1], keepdims=0)
    graph = helper.make_graph([node_rmax], 'reduce_max_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_reduce_min_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [])
    node_rmin = helper.make_node('ReduceMin', inputs=['input'], outputs=['output'], axes=[0, 1], keepdims=0)
    graph = helper.make_graph([node_rmin], 'reduce_min_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_prelu_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 2, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 2, 2])
    slope = numpy_helper.from_array(
        np.array([[[[0.10]], [[0.20]], [[0.30]]]], dtype=np.float32),
        name='slope',
    )
    node = helper.make_node('PRelu', inputs=['input', 'slope'], outputs=['output'])
    graph = helper.make_graph([node], 'prelu_test', [x], [y], [slope])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_mod_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    div = numpy_helper.from_array(np.array([[0.8, 0.7, 0.5, 0.6]], dtype=np.float32), name='div')
    node = helper.make_node('Mod', inputs=['input', 'div'], outputs=['output'], fmod=1)
    graph = helper.make_graph([node], 'mod_test', [x], [y], [div])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_dropout_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 6])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 6])
    node = helper.make_node('Dropout', inputs=['input'], outputs=['output'])
    graph = helper.make_graph([node], 'dropout_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_upsample_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 2, 2])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 4, 6])
    scales = numpy_helper.from_array(np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32), name='scales')
    node = helper.make_node('Upsample', inputs=['input', 'scales'], outputs=['output'], mode='nearest')
    graph = helper.make_graph([node], 'upsample_test', [x], [y], [scales])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 9)])
    onnx.save(model, path)


def _build_qdq_identity_reshape_model(path: str, qdtype: int = TensorProto.INT8) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 2])
    scale, zero = _qdq_params(qdtype)
    shape = numpy_helper.from_array(np.array([2, 2], dtype=np.int64), name='shape')
    qx = helper.make_node('QuantizeLinear', inputs=['input', 'scale', 'zero'], outputs=['q0'])
    ident = helper.make_node('Identity', inputs=['q0'], outputs=['q1'])
    reshape = helper.make_node('Reshape', inputs=['q1', 'shape'], outputs=['q2'])
    dq = helper.make_node('DequantizeLinear', inputs=['q2', 'scale', 'zero'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('q0', qdtype, [1, 4]),
        helper.make_tensor_value_info('q1', qdtype, [1, 4]),
        helper.make_tensor_value_info('q2', qdtype, [2, 2]),
    ]
    graph = helper.make_graph(
        [qx, ident, reshape, dq],
        'qdq_identity_reshape_test',
        [x],
        [y],
        [scale, zero, shape],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_constant_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    cval = numpy_helper.from_array(
        np.array([[0.10, -0.20, 0.30, 0.40]], dtype=np.float32),
        name='k',
    )
    node_const = helper.make_node('Constant', inputs=[], outputs=['k1'], value=cval)
    node_add = helper.make_node('Add', inputs=['input', 'k1'], outputs=['output'])
    graph = helper.make_graph([node_const, node_add], 'constant_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_hardmax_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])
    node = helper.make_node('Hardmax', inputs=['input'], outputs=['output'], axis=1)
    graph = helper.make_graph([node], 'hardmax_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_logsoftmax_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])
    node = helper.make_node('LogSoftmax', inputs=['input'], outputs=['output'], axis=1)
    graph = helper.make_graph([node], 'logsoftmax_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_thresholded_relu_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 6])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 6])
    node = helper.make_node('ThresholdedRelu', inputs=['input'], outputs=['output'], alpha=0.2)
    graph = helper.make_graph([node], 'thresholded_relu_test', [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_range_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [5])
    start = numpy_helper.from_array(np.array([0.0], dtype=np.float32), name='start')
    limit = numpy_helper.from_array(np.array([5.0], dtype=np.float32), name='limit')
    delta = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name='delta')
    node = helper.make_node('Range', inputs=['start', 'limit', 'delta'], outputs=['output'])
    graph = helper.make_graph([node], 'range_test', [x], [y], [start, limit, delta])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_gather_elements_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])
    idx = numpy_helper.from_array(
        np.array([[2, 1, 0], [0, 1, 2]], dtype=np.int64),
        name='idx',
    )
    node = helper.make_node('GatherElements', inputs=['input', 'idx'], outputs=['output'], axis=1)
    graph = helper.make_graph([node], 'gather_elements_test', [x], [y], [idx])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_scatter_elements_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])
    idx = numpy_helper.from_array(
        np.array([[0, 2], [1, 0]], dtype=np.int64),
        name='idx',
    )
    upd = numpy_helper.from_array(
        np.array([[9.0, 8.0], [7.0, 6.0]], dtype=np.float32),
        name='upd',
    )
    node = helper.make_node('ScatterElements', inputs=['input', 'idx', 'upd'], outputs=['output'], axis=1)
    graph = helper.make_graph([node], 'scatter_elements_test', [x], [y], [idx, upd])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_scatter_nd_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])
    idx = numpy_helper.from_array(
        np.array([[0], [1]], dtype=np.int64),
        name='idx',
    )
    upd = numpy_helper.from_array(
        np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=np.float32),
        name='upd',
    )
    node = helper.make_node('ScatterND', inputs=['input', 'idx', 'upd'], outputs=['output'])
    graph = helper.make_graph([node], 'scatter_nd_test', [x], [y], [idx, upd])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_scatter_elements_reduction_model(path: str, reduction: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])
    idx = numpy_helper.from_array(
        np.array([[0, 0, 2], [1, 1, 1]], dtype=np.int64),
        name='idx',
    )
    upd = numpy_helper.from_array(
        np.array([[10.0, 20.0, 30.0], [2.0, 3.0, 4.0]], dtype=np.float32),
        name='upd',
    )
    node = helper.make_node(
        'ScatterElements',
        inputs=['input', 'idx', 'upd'],
        outputs=['output'],
        axis=1,
        reduction=reduction,
    )
    graph = helper.make_graph([node], f'scatter_elements_{reduction}_test', [x], [y], [idx, upd])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 18)])
    onnx.save(model, path)


def _build_scatter_nd_reduction_model(path: str, reduction: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [4])
    idx = numpy_helper.from_array(
        np.array([[1], [1], [2]], dtype=np.int64),
        name='idx',
    )
    upd = numpy_helper.from_array(
        np.array([3.0, 9.0, 8.0], dtype=np.float32),
        name='upd',
    )
    node = helper.make_node(
        'ScatterND',
        inputs=['input', 'idx', 'upd'],
        outputs=['output'],
        reduction=reduction,
    )
    graph = helper.make_graph([node], f'scatter_nd_{reduction}_test', [x], [y], [idx, upd])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 18)])
    onnx.save(model, path)


def _build_isinf_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 6])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 6])
    node_inf = helper.make_node(
        'IsInf',
        inputs=['input'],
        outputs=['mask'],
        detect_negative=1,
        detect_positive=1,
    )
    node_cast = helper.make_node('Cast', inputs=['mask'], outputs=['output'], to=TensorProto.FLOAT)
    value_info = [
        helper.make_tensor_value_info('mask', TensorProto.BOOL, [1, 6]),
    ]
    graph = helper.make_graph([node_inf, node_cast], 'isinf_test', [x], [y], value_info=value_info)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_isnan_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 6])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 6])
    node_nan = helper.make_node('IsNaN', inputs=['input'], outputs=['mask'])
    node_cast = helper.make_node('Cast', inputs=['mask'], outputs=['output'], to=TensorProto.FLOAT)
    value_info = [
        helper.make_tensor_value_info('mask', TensorProto.BOOL, [1, 6]),
    ]
    graph = helper.make_graph([node_nan, node_cast], 'isnan_test', [x], [y], value_info=value_info)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_nonzero_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])
    const_x = numpy_helper.from_array(
        np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]], dtype=np.float32),
        name='const_x',
    )
    node_nz = helper.make_node('NonZero', inputs=['const_x'], outputs=['nz'])
    node_cast = helper.make_node('Cast', inputs=['nz'], outputs=['output'], to=TensorProto.FLOAT)
    value_info = [
        helper.make_tensor_value_info('nz', TensorProto.INT64, [2, 3]),
    ]
    graph = helper.make_graph(
        [node_nz, node_cast],
        'nonzero_test',
        [x],
        [y],
        [const_x],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_topk_model(path: str) -> None:
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 5])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])
    k = numpy_helper.from_array(np.array([3], dtype=np.int64), name='k')
    node_topk = helper.make_node('TopK', inputs=['input', 'k'], outputs=['values', 'indices'], axis=-1)
    node_out = helper.make_node('Identity', inputs=['values'], outputs=['output'])
    value_info = [
        helper.make_tensor_value_info('values', TensorProto.FLOAT, [2, 3]),
        helper.make_tensor_value_info('indices', TensorProto.INT64, [2, 3]),
    ]
    graph = helper.make_graph(
        [node_topk, node_out],
        'topk_test',
        [x],
        [y],
        [k],
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


def _build_topk_int32_precision_model(path: str) -> None:
    dummy = helper.make_tensor_value_info('dummy', TensorProto.FLOAT, [1])
    values = helper.make_tensor_value_info('values', TensorProto.INT32, [1])
    indices = helper.make_tensor_value_info('indices', TensorProto.INT64, [1])

    const_x = numpy_helper.from_array(
        np.array([16777216, 16777217], dtype=np.int32),
        name='const_x',
    )
    k = numpy_helper.from_array(np.array([1], dtype=np.int64), name='k')

    node_topk = helper.make_node(
        'TopK',
        inputs=['const_x', 'k'],
        outputs=['values', 'indices'],
        axis=0,
        largest=1,
        sorted=1,
    )
    node_dummy = helper.make_node('Identity', inputs=['dummy'], outputs=['dummy_out'])

    graph = helper.make_graph(
        [node_topk, node_dummy],
        'topk_int32_precision_test',
        [dummy],
        [values, indices],
        [const_x, k],
        value_info=[helper.make_tensor_value_info('dummy_out', TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 13)])
    onnx.save(model, path)


class TestTinyMlProject(unittest.TestCase):
    def _assert_model_consistency_regression(
        self,
        model_path: str,
        result: dict[str, object],
        *,
        seeds: tuple[int, ...] = (0, 1, 7),
        rtol: float = 1e-3,
        atol: float = 1e-4,
        int8_atol: float = 1.0,
        int16_atol: float = 1.0,
        require_onnxruntime: bool = True,
    ) -> None:
        model = load_onnx_model(model_path)
        source_path = str(result['source'])
        header_path = str(result['header'])
        for seed in seeds:
            validation = validate_model_consistency(
                model,
                model_path,
                source_path=source_path,
                header_path=header_path,
                seed=seed,
                rtol=rtol,
                atol=atol,
                int8_atol=int8_atol,
                int16_atol=int16_atol,
                allow_reference_fallback=not require_onnxruntime,
            )
            if validation.status == 'skipped':
                self.skipTest(
                    f"consistency regression skipped: seed={seed}, reason={validation.reason}"
                )
            self.assertEqual(
                validation.status,
                'passed',
                msg=f"seed={seed}, reason={validation.reason}, engine={validation.engine}, "
                f"max_abs={validation.max_abs}, max_rel={validation.max_rel}",
            )
            self.assertIn('generated-c vs ', validation.engine)
            if require_onnxruntime:
                self.assertIn('onnxruntime', validation.engine)
            else:
                self.assertTrue(
                    ('onnxruntime' in validation.engine) or ('onnx.reference' in validation.engine),
                    msg=f"unexpected reference engine: {validation.engine}",
                )

    def _assert_quant_recurrent_codegen_and_run(
        self,
        model_path: str,
        result: dict[str, object],
        *,
        qdtype: int,
    ) -> None:
        model = load_onnx_model(model_path)
        source = Path(result['source']).read_text(encoding='utf-8')
        if qdtype == TensorProto.INT8:
            self.assertIn('int8_t', source)
        elif qdtype == TensorProto.INT16:
            self.assertIn('int16_t', source)
        else:
            raise ValueError('Unsupported qdtype for recurrent test.')
        in_data = np.array([[[0.1]], [[0.2]], [[0.3]], [[0.4]]], dtype=np.float32)
        py_out = _eval_model(model, {'input': in_data})[model.outputs[0].name]
        c_run = run_generated_c_model(
            model,
            str(result['source']),
            str(result['header']),
            {'input': in_data},
        )
        self.assertTrue(c_run.ok, msg=c_run.reason)
        assert c_run.outputs is not None
        c_out = c_run.outputs[model.outputs[0].name]
        np.testing.assert_allclose(c_out, py_out, rtol=1e-4, atol=1e-4)

    def test_generate_tinyml_project(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'model.onnx')
            _build_simple_gemm_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')

            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )

            self.assertTrue(os.path.exists(result['header']))
            self.assertTrue(os.path.exists(result['source']))
            self.assertTrue(os.path.exists(result['manifest']))
            self.assertEqual(result['library'], '')

            header = Path(result['header']).read_text(encoding='utf-8')
            self.assertIn('k2c_forward', header)
            self.assertIn('k2c_prepare', header)

    def test_weights_ram_codegen(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'model.onnx')
            _build_simple_gemm_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')

            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='ram',
                emit='c',
            )

            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('k2c_prepare', source)
            self.assertIn('memcpy', source)

    def test_int8_quant_codegen(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'model.onnx')
            _build_qdq_add_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')

            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )

            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('int8_t', source)
            self.assertIn('QuantizeLinear', manifest)

    def test_int8_quant_unary_binary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'model.onnx')
            _build_qdq_unary_binary_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')

            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )

            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('Abs', Path(result['manifest']).read_text(encoding='utf-8'))
            self.assertIn('int8_t', source)

    def test_int8_quant_unary_extra(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'model.onnx')
            _build_qdq_unary_extra_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')

            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )

            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('LeakyRelu', manifest)
            self.assertIn('int8_t', source)

    def test_int8_quant_pow(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'model.onnx')
            _build_qdq_pow_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')

            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )

            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('Pow', manifest)
            self.assertIn('int8_t', source)

    def test_int8_quant_conv(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'model.onnx')
            _build_qdq_conv_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')

            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )

            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('Conv', manifest)
            self.assertIn('int8_t', source)

    def test_int8_quant_group_conv(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'group_conv.onnx')
            _build_qdq_group_conv_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')

            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )

            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('size_t g = oc / 3;', source)
            self.assertIn('size_t ic_begin = g * 2;', source)

    def test_consistency_regression_group_conv_c(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'group_conv_consistency.onnx')
            _build_group_conv_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            self._assert_model_consistency_regression(
                model_path,
                result,
                seeds=(0, 1),
                rtol=1e-3,
                atol=1e-4,
            )

    def test_int8_quant_global_avgpool_nd_c(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'global_avgpool_nd_qdq.onnx')
            _build_qdq_global_avgpool_nd_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('GlobalAveragePool', Path(result['manifest']).read_text(encoding='utf-8'))
            self.assertIn('for (size_t i = 0; i < 8; ++i)', source)

    def test_int8_quant_global_maxpool_nd_c(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'global_maxpool_nd_qdq.onnx')
            _build_qdq_global_maxpool_nd_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('GlobalMaxPool', Path(result['manifest']).read_text(encoding='utf-8'))
            self.assertIn('for (size_t i = 0; i < 8; ++i)', source)

    def test_int8_quant_avgpool_nd_c(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'avgpool_nd_qdq.onnx')
            _build_qdq_avgpool_nd_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('AveragePool', manifest)

    def test_int8_quant_maxpool_nd_c(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'maxpool_nd_qdq.onnx')
            _build_qdq_maxpool_nd_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('MaxPool', manifest)

    def test_int16_quant_conv(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'model.onnx')
            _build_qdq_conv_model(model_path, TensorProto.INT16)
            out_root = os.path.join(td, 'onnx-for-mcu')

            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )

            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('Conv', manifest)
            self.assertIn('int16_t', source)

    def test_int8_quant_matmul(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'model.onnx')
            _build_qdq_matmul_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')

            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )

            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('MatMul', manifest)
            self.assertIn('int8_t', source)

    def test_int16_quant_matmul(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'model.onnx')
            _build_qdq_matmul_model(model_path, TensorProto.INT16)
            out_root = os.path.join(td, 'onnx-for-mcu')

            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )

            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('MatMul', manifest)
            self.assertIn('int16_t', source)

    def test_int8_quant_einsum(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'model.onnx')
            _build_qdq_einsum_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')

            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )

            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('Einsum', manifest)
            self.assertIn('int8_t', source)

    def test_int16_quant_einsum(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'model.onnx')
            _build_qdq_einsum_model(model_path, TensorProto.INT16)
            out_root = os.path.join(td, 'onnx-for-mcu')

            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )

            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('Einsum', manifest)
            self.assertIn('int16_t', source)

    def test_int8_quant_softmax(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'model.onnx')
            _build_qdq_softmax_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')

            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )

            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('Softmax', manifest)
            self.assertIn('int8_t', source)

    def test_int8_quant_logsoftmax(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'model.onnx')
            _build_qdq_logsoftmax_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')

            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )

            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('LogSoftmax', manifest)
            self.assertIn('int8_t', source)

    def test_int8_quant_prelu(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'model.onnx')
            _build_qdq_prelu_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')

            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )

            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('PRelu', manifest)
            self.assertIn('int8_t', source)

    def test_int8_quant_mod(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'model.onnx')
            _build_qdq_mod_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')

            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )

            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('Mod', manifest)
            self.assertIn('int8_t', source)

    def test_int8_quant_reduce_basic(self) -> None:
        for op in ("ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceLogSum", "ReduceLogSumExp"):
            with tempfile.TemporaryDirectory() as td:
                model_path = os.path.join(td, f'{op}.onnx')
                _build_qdq_reduce_basic_model(model_path, op, TensorProto.INT8)
                out_root = os.path.join(td, 'onnx-for-mcu')

                result = generate_tinyml_project(
                    model_path,
                    out_root,
                    weights='flash',
                    emit='c',
                )

                manifest = Path(result['manifest']).read_text(encoding='utf-8')
                source = Path(result['source']).read_text(encoding='utf-8')
                self.assertIn(op, manifest)
                self.assertIn('int8_t', source)

    def test_int8_quant_gemm(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'model.onnx')
            _build_qdq_gemm_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')

            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )

            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('Gemm', manifest)
            self.assertIn('int8_t', source)

    def test_int16_quant_gemm(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'model.onnx')
            _build_qdq_gemm_model(model_path, TensorProto.INT16)
            out_root = os.path.join(td, 'onnx-for-mcu')

            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )

            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('Gemm', manifest)
            self.assertIn('int16_t', source)

    def test_unsupported_operator(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'model.onnx')
            _build_unsupported_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            with self.assertRaises(ValueError):
                generate_tinyml_project(
                    model_path,
                    out_root,
                    weights='flash',
                    emit='c',
                )

    def test_conv_pool_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'conv.onnx')
            _build_conv_pool_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            self.assertTrue(os.path.exists(result['source']))

    def test_conv_transpose_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'conv_transpose.onnx')
            _build_conv_transpose_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('ConvTranspose', manifest)

    def test_group_conv_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'group_conv.onnx')
            _build_group_conv_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('size_t g = oc / 3;', source)
            self.assertIn('size_t ic_begin = g * 2;', source)

    def test_batchnorm_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'bn.onnx')
            _build_batchnorm_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            self.assertTrue(os.path.exists(result['source']))

    def test_instance_norm_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'instancenorm.onnx')
            _build_instance_norm_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('InstanceNormalization', manifest)

    def test_lrn_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'lrn.onnx')
            _build_lrn_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('LRN', manifest)

    def test_lp_normalization_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'lp_normalization.onnx')
            _build_lp_normalization_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('LpNormalization', manifest)
            self.assertIn('norm <= 0.0f', source)

    def test_mean_variance_normalization_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'mean_variance_normalization.onnx')
            _build_mean_variance_normalization_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('MeanVarianceNormalization', manifest)
            self.assertIn('/ sqrtf(', source)

    def test_lppool_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'lppool.onnx')
            _build_lppool_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('LpPool', manifest)
            self.assertIn('sqrtf(acc)', source)

    def test_lppool_nd_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'lppool_nd.onnx')
            _build_lppool_nd_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('LpPool', manifest)

    def test_global_lppool_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'global_lppool.onnx')
            _build_global_lppool_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('GlobalLpPool', manifest)
            self.assertIn('sqrtf(acc)', source)

    def test_global_lppool_nd_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'global_lppool_nd.onnx')
            _build_global_lppool_nd_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('for (size_t i = 0; i < 8; ++i)', source)

    def test_conv_pool_n2_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'conv_n2.onnx')
            _build_conv_pool_n2_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            self.assertTrue(os.path.exists(result['source']))

    def test_batchnorm_n2_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'bn_n2.onnx')
            _build_batchnorm_n2_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            self.assertTrue(os.path.exists(result['source']))

    def test_batchnorm_nd_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'bn_nd.onnx')
            _build_batchnorm_nd_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('for (size_t i = 0; i < 8; ++i)', source)

    def test_global_avgpool_n2_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'global_avgpool_n2.onnx')
            _build_global_avgpool_n2_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            self.assertTrue(os.path.exists(result['source']))

    def test_avgpool_nd_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'avgpool_nd.onnx')
            _build_avgpool_nd_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('AveragePool', manifest)

    def test_global_avgpool_nd_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'global_avgpool_nd.onnx')
            _build_global_avgpool_nd_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('for (size_t i = 0; i < 8; ++i)', source)

    def test_add_broadcast_n2_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'add_broadcast_n2.onnx')
            _build_add_broadcast_n2_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            self.assertTrue(os.path.exists(result['source']))

    def test_resnet_like_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'resnet_like.onnx')
            _build_resnet_like_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('BatchNormalization', manifest)
            self.assertIn('GlobalAveragePool', manifest)
            self.assertIn('Gemm', manifest)

    def test_consistency_regression_resnet_like_c(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'resnet_like_consistency.onnx')
            _build_resnet_like_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            self._assert_model_consistency_regression(
                model_path,
                result,
                seeds=(0, 1),
                rtol=1e-3,
                atol=1e-4,
            )

    def test_mobilenetv2_like_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'mobilenetv2_like.onnx')
            _build_mobilenetv2_like_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Conv', manifest)
            self.assertIn('Clip', manifest)
            self.assertIn('Add', manifest)

    def test_consistency_regression_mobilenet_like_c(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'mobilenet_like_consistency.onnx')
            _build_mobilenetv2_like_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            self._assert_model_consistency_regression(
                model_path,
                result,
                seeds=(0, 1),
                rtol=1e-3,
                atol=1e-4,
            )

    def test_softmax2d_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'softmax.onnx')
            _build_softmax2d_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            self.assertTrue(os.path.exists(result['source']))

    def test_softmax4d_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'softmax4d.onnx')
            _build_softmax4d_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            self.assertTrue(os.path.exists(result['source']))

    def test_einsum_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'einsum.onnx')
            _build_einsum_model(model_path)
            model = load_onnx_model(model_path)
            a = np.random.randn(2, 3, 4).astype(np.float32)
            b = np.array(model.tensors['W'].data, dtype=np.float32).reshape(2, 4, 5)
            out = _eval_model(model, {'input': a})[model.outputs[0].name]
            ref = np.einsum('bij,bjk->bik', a, b).astype(np.float32)
            np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)

            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Einsum', manifest)
            self.assertIn('b_i', source)

    def test_concat_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'concat.onnx')
            _build_concat_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            self.assertTrue(os.path.exists(result['source']))

    def test_transpose_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'transpose.onnx')
            _build_transpose_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            self.assertTrue(os.path.exists(result['source']))

    def test_transpose5d_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'transpose5d.onnx')
            _build_transpose5d_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            self.assertTrue(os.path.exists(result['source']))

    def test_cast_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'cast.onnx')
            _build_cast_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('Cast', Path(result['manifest']).read_text(encoding='utf-8'))
            self.assertIn('int16_t', source)

    def test_gather_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'gather.onnx')
            _build_gather_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('Gather', Path(result['manifest']).read_text(encoding='utf-8'))
            self.assertIn('idx_v', source)

    def test_gathernd_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'gathernd.onnx')
            _build_gathernd_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('GatherND', manifest)

    def test_global_maxpool_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'global_maxpool.onnx')
            _build_global_maxpool_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('GlobalMaxPool', Path(result['manifest']).read_text(encoding='utf-8'))
            self.assertIn('acc = -3.402823466e+38F', source)

    def test_maxpool_nd_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'maxpool_nd.onnx')
            _build_maxpool_nd_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('MaxPool', manifest)

    def test_global_maxpool_nd_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'global_maxpool_nd.onnx')
            _build_global_maxpool_nd_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('GlobalMaxPool', Path(result['manifest']).read_text(encoding='utf-8'))
            self.assertIn('for (size_t i = 0; i < 8; ++i)', source)

    def test_expand_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'expand.onnx')
            _build_expand_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('Expand', Path(result['manifest']).read_text(encoding='utf-8'))
            self.assertIn('k2c_expand_in_strides', source)

    def test_where_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'where.onnx')
            _build_where_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('Where', Path(result['manifest']).read_text(encoding='utf-8'))
            self.assertIn('k2c_where_cond_strides', source)

    def test_space_to_depth_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'space_to_depth.onnx')
            _build_space_to_depth_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('SpaceToDepth', Path(result['manifest']).read_text(encoding='utf-8'))
            self.assertIn('oc = c_i * 4 + bh * 2 + bw', source)

    def test_depth_to_space_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'depth_to_space.onnx')
            _build_depth_to_space_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('DepthToSpace', Path(result['manifest']).read_text(encoding='utf-8'))
            self.assertIn('ic = oc * 4 + bh * 2 + bw', source)

    def test_tile_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'tile.onnx')
            _build_tile_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Tile', manifest)
            self.assertIn('k2c_tile_out_dims', source)

    def test_resize_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'resize.onnx')
            _build_resize_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Resize', manifest)
            self.assertIn('src_hf', source)

    def test_elu_selu_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'elu_selu.onnx')
            _build_elu_selu_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Elu', manifest)
            self.assertIn('Selu', manifest)
            self.assertIn('expf(v)', source)

    def test_sign_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'sign.onnx')
            _build_sign_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Sign', manifest)
            self.assertIn('v > 0.0f ? 1.0f', source)

    def test_compare_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'compare.onnx')
            _build_compare_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Equal', manifest)
            self.assertIn('GreaterOrEqual', manifest)
            self.assertIn('LessOrEqual', manifest)
            self.assertIn('k2c_cmp_out_dims', source)

    def test_logic_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'logic.onnx')
            _build_logic_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Not', manifest)
            self.assertIn('And', manifest)
            self.assertIn('Or', manifest)
            self.assertIn('Xor', manifest)
            self.assertIn('k2c_logic_out_dims', source)

    def test_erf_round_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'erf_round.onnx')
            _build_erf_round_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Erf', manifest)
            self.assertIn('Round', manifest)
            self.assertIn('erff', source)
            self.assertIn('nearbyintf', source)

    def test_argmax_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'argmax.onnx')
            _build_argmax_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('ArgMax', manifest)
            self.assertIn('best_k', source)

    def test_argmin_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'argmin.onnx')
            _build_argmin_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('ArgMin', manifest)
            self.assertIn('best_k', source)

    def test_reduce_axes_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'reduce_axes.onnx')
            _build_reduce_axes_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('ReduceMean', manifest)
            self.assertIn('ReduceSum', manifest)
            self.assertIn('k2c_reduce_mask', source)

    def test_reduce_extreme_axes_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'reduce_extreme_axes.onnx')
            _build_reduce_extreme_axes_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('ReduceMax', manifest)
            self.assertIn('ReduceMin', manifest)
            self.assertIn('k2c_reduce_mask', source)

    def test_pad_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'pad.onnx')
            _build_pad_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            self.assertTrue(os.path.exists(result['source']))

    def test_pad5d_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'pad5d.onnx')
            _build_pad5d_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Pad', manifest)
            self.assertIn('k2c_pad_in_dims', source)

    def test_slice_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'slice.onnx')
            _build_slice_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            self.assertTrue(os.path.exists(result['source']))

    def test_slice5d_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'slice5d.onnx')
            _build_slice5d_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Slice', manifest)
            self.assertIn('k2c_slice_step', source)

    def test_cumsum_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'cumsum.onnx')
            _build_cumsum_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('CumSum', manifest)
            self.assertIn('src_axis', source)

    def test_shrink_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'shrink.onnx')
            _build_shrink_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Shrink', manifest)
            self.assertIn('v < -0.50000000f', source)

    def test_eyelike_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'eyelike.onnx')
            _build_eyelike_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('EyeLike', manifest)
            self.assertIn('on_diag', source)

    def test_onehot_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'onehot.onnx')
            _build_onehot_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('OneHot', manifest)
            self.assertIn('class_v', source)

    def test_scatter_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'scatter.onnx')
            _build_scatter_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Scatter', manifest)
            self.assertIn('idx_v', source)

    def test_scatter_vs_scatter_elements_compat_none(self) -> None:
        in_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            model_sc = os.path.join(td, 'scatter_compat_scatter.onnx')
            model_se = os.path.join(td, 'scatter_compat_scatter_elements.onnx')
            _build_scatter_compat_model(model_sc, 'Scatter')
            _build_scatter_compat_model(model_se, 'ScatterElements')

            sc_ir = load_onnx_model(model_sc)
            se_ir = load_onnx_model(model_se)
            out_sc = _eval_model(sc_ir, {'input': in_data})[sc_ir.outputs[0].name]
            out_se = _eval_model(se_ir, {'input': in_data})[se_ir.outputs[0].name]
            np.testing.assert_allclose(out_sc, out_se, rtol=0.0, atol=0.0)

    def test_scatter_reduction_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'scatter_invalid_reduction.onnx')
            _build_scatter_invalid_reduction_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            with self.assertRaisesRegex(ValueError, 'Scatter does not support reduction'):
                generate_tinyml_project(
                    model_path,
                    out_root,
                    weights='flash',
                    emit='c',
                )

    def test_det_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'det.onnx')
            _build_det_model(model_path)
            model = load_onnx_model(model_path)
            in_data = np.array(
                [[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 2.0, 1.0]],
                dtype=np.float32,
            )
            out = _eval_model(model, {'input': in_data})[model.outputs[0].name]
            exp = np.array(np.linalg.det(in_data), dtype=np.float32)
            np.testing.assert_allclose(out, exp, rtol=1e-5, atol=1e-5)

            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Det', manifest)
            self.assertIn('det_v', source)

    def test_reverse_sequence_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'reverse_sequence.onnx')
            _build_reverse_sequence_model(model_path)
            model = load_onnx_model(model_path)
            in_data = np.array(
                [
                    [1.0, 10.0, 100.0],
                    [2.0, 20.0, 200.0],
                    [3.0, 30.0, 300.0],
                    [4.0, 40.0, 400.0],
                ],
                dtype=np.float32,
            )
            out = _eval_model(model, {'input': in_data})[model.outputs[0].name]
            exp = np.array(
                [
                    [4.0, 20.0, 300.0],
                    [3.0, 10.0, 200.0],
                    [2.0, 30.0, 100.0],
                    [1.0, 40.0, 400.0],
                ],
                dtype=np.float32,
            )
            np.testing.assert_allclose(out, exp, rtol=0.0, atol=0.0)

            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('ReverseSequence', manifest)
            self.assertIn('src_t', source)

    def test_bitshift_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'bitshift.onnx')
            _build_bitshift_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('BitShift', manifest)
            self.assertIn('k2c_bshift_out_dims', source)

    def test_compress_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'compress.onnx')
            _build_compress_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Compress', manifest)
            self.assertIn('k2c_comp_axis_idx', source)

    def test_matmul_integer_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'matmul_integer.onnx')
            _build_matmul_integer_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('MatMulInteger', manifest)
            self.assertIn('int64_t acc', source)

    def test_qlinear_matmul_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'qlinear_matmul.onnx')
            _build_qlinear_matmul_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('QLinearMatMul', manifest)
            self.assertIn('real_v', source)

    def test_matmul_integer_batched_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'matmul_integer_batched.onnx')
            _build_matmul_integer_batched_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('MatMulInteger', manifest)
            self.assertIn('k2c_mmi_batch_dims', source)
            self._assert_model_consistency_regression(model_path, result)

    def test_qlinear_matmul_batched_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'qlinear_matmul_batched.onnx')
            _build_qlinear_matmul_batched_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('QLinearMatMul', manifest)
            self.assertIn('k2c_qmm_batch_dims', source)
            self._assert_model_consistency_regression(model_path, result, int8_atol=2.0)

    def test_rnn_reverse_sequence_lens_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'rnn_reverse_seq.onnx')
            _build_rnn_reverse_seq_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('RNN', manifest)
            self.assertIn('seq_len', source)
            self._assert_model_consistency_regression(model_path, result)

    def test_rnn_bidirectional_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'rnn_bidirectional.onnx')
            _build_rnn_bidirectional_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('RNN', manifest)
            self.assertIn('dir_rev', source)
            self._assert_model_consistency_regression(model_path, result)

    def test_gru_reverse_lbr1_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'gru_reverse_lbr1.onnx')
            _build_gru_reverse_lbr1_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('GRU', manifest)
            self.assertIn('rec_sum', source)
            self._assert_model_consistency_regression(model_path, result)

    def test_lstm_reverse_peephole_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'lstm_reverse_peephole.onnx')
            _build_lstm_reverse_peephole_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('LSTM', manifest)
            self.assertIn('c_prev', source)
            self._assert_model_consistency_regression(model_path, result)

    def test_rnn_relu_clip_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'rnn_relu_clip.onnx')
            _build_rnn_relu_clip_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('RNN', manifest)
            self.assertIn('k2c_rnn_clip', source)
            self._assert_model_consistency_regression(model_path, result)

    def test_gru_hardsigmoid_relu_clip_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'gru_hardsigmoid_relu_clip.onnx')
            _build_gru_hardsigmoid_relu_clip_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('GRU', manifest)
            self.assertIn('k2c_gru_clip', source)
            self._assert_model_consistency_regression(model_path, result)

    def test_lstm_input_forget_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'lstm_input_forget.onnx')
            _build_lstm_input_forget_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('LSTM', manifest)
            self.assertIn('1.0f - i_gate', source)
            self.assertIn('k2c_lstm_clip', source)
            self._assert_model_consistency_regression(model_path, result)

    def test_quant_recurrent_rnn(self) -> None:
        cases = (
            (TensorProto.INT8, 'qdq_rnn_int8.onnx'),
            (TensorProto.INT16, 'qdq_rnn_int16.onnx'),
        )
        for qdtype, filename in cases:
            with self.subTest(qdtype=qdtype):
                with tempfile.TemporaryDirectory() as td:
                    model_path = os.path.join(td, filename)
                    _build_qdq_rnn_model(model_path, qdtype)
                    out_root = os.path.join(td, 'onnx-for-mcu')
                    result = generate_tinyml_project(
                        model_path,
                        out_root,
                        weights='flash',
                        emit='c',
                    )
                    manifest = Path(result['manifest']).read_text(encoding='utf-8')
                    self.assertIn('RNN', manifest)
                    self._assert_quant_recurrent_codegen_and_run(
                        model_path,
                        result,
                        qdtype=qdtype,
                    )

    def test_quant_recurrent_gru(self) -> None:
        cases = (
            (TensorProto.INT8, 'qdq_gru_int8.onnx'),
            (TensorProto.INT16, 'qdq_gru_int16.onnx'),
        )
        for qdtype, filename in cases:
            with self.subTest(qdtype=qdtype):
                with tempfile.TemporaryDirectory() as td:
                    model_path = os.path.join(td, filename)
                    _build_qdq_gru_model(model_path, qdtype)
                    out_root = os.path.join(td, 'onnx-for-mcu')
                    result = generate_tinyml_project(
                        model_path,
                        out_root,
                        weights='flash',
                        emit='c',
                    )
                    manifest = Path(result['manifest']).read_text(encoding='utf-8')
                    self.assertIn('GRU', manifest)
                    self._assert_quant_recurrent_codegen_and_run(
                        model_path,
                        result,
                        qdtype=qdtype,
                    )

    def test_quant_recurrent_lstm(self) -> None:
        cases = (
            (TensorProto.INT8, 'qdq_lstm_int8.onnx'),
            (TensorProto.INT16, 'qdq_lstm_int16.onnx'),
        )
        for qdtype, filename in cases:
            with self.subTest(qdtype=qdtype):
                with tempfile.TemporaryDirectory() as td:
                    model_path = os.path.join(td, filename)
                    _build_qdq_lstm_model(model_path, qdtype)
                    out_root = os.path.join(td, 'onnx-for-mcu')
                    result = generate_tinyml_project(
                        model_path,
                        out_root,
                        weights='flash',
                        emit='c',
                    )
                    manifest = Path(result['manifest']).read_text(encoding='utf-8')
                    self.assertIn('LSTM', manifest)
                    self._assert_quant_recurrent_codegen_and_run(
                        model_path,
                        result,
                        qdtype=qdtype,
                    )

    def test_conv_integer_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'conv_integer.onnx')
            _build_conv_integer_model(model_path)
            model = load_onnx_model(model_path)
            in_data = np.arange(1, 1 + 1 * 2 * 4 * 4, dtype=np.int8).reshape(1, 2, 4, 4)
            out = _eval_model(model, {'input': in_data})[model.outputs[0].name]
            self.assertEqual(out.shape, (1, 3, 2, 2))
            self.assertEqual(out.dtype, np.float32)

            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('ConvInteger', manifest)
            self.assertIn('int64_t acc', source)

    def test_qlinear_conv_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'qlinear_conv.onnx')
            _build_qlinear_conv_model(model_path)
            model = load_onnx_model(model_path)
            in_data = (np.arange(1, 1 + 1 * 2 * 4 * 4, dtype=np.int16).reshape(1, 2, 4, 4) - 10).astype(np.int8)
            out = _eval_model(model, {'input': in_data})[model.outputs[0].name]
            self.assertEqual(out.shape, (1, 3, 2, 2))
            self.assertEqual(out.dtype, np.int8)

            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('QLinearConv', manifest)
            self.assertIn('k2c_qconv_w_scale', source)

    def test_reduce_mean_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'reduce.onnx')
            _build_reduce_mean_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            self.assertTrue(os.path.exists(result['source']))

    def test_unary_chain_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'unary.onnx')
            _build_unary_chain_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            self.assertTrue(os.path.exists(result['source']))

    def test_unary_extended_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'unary_extended.onnx')
            _build_unary_extended_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('HardSigmoid', manifest)
            self.assertIn('Softplus', manifest)
            self.assertIn('1.0f /', source)

    def test_int8_quant_unary_extended(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'qdq_unary_extended.onnx')
            _build_qdq_unary_extended_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('int8_t', source)
            self.assertIn('HardSigmoid', manifest)

    def test_int8_quant_rearrange(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'qdq_rearrange.onnx')
            _build_qdq_rearrange_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('SpaceToDepth', manifest)
            self.assertIn('DepthToSpace', manifest)
            self.assertIn('int8_t', source)

    def test_int8_quant_gathernd(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'qdq_gathernd.onnx')
            _build_qdq_gathernd_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('GatherND', manifest)
            self.assertIn('int8_t', source)

    def test_int8_quant_scatter_elements(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'qdq_scatter_elements.onnx')
            _build_qdq_scatter_elements_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('ScatterElements', manifest)
            self.assertIn('int8_t', source)

    def test_int8_quant_scatter_nd(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'qdq_scatter_nd.onnx')
            _build_qdq_scatter_nd_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('ScatterND', manifest)
            self.assertIn('int8_t', source)

    def test_int8_quant_reverse_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'qdq_reverse_sequence.onnx')
            _build_qdq_reverse_sequence_model(model_path, TensorProto.INT8)
            model = load_onnx_model(model_path)
            _ = _eval_model(
                model,
                {
                    'input': np.array(
                        [
                            [0.1, 0.2, 0.3],
                            [0.2, 0.3, 0.4],
                            [0.3, 0.4, 0.5],
                            [0.4, 0.5, 0.6],
                        ],
                        dtype=np.float32,
                    )
                },
            )

            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('ReverseSequence', manifest)
            self.assertIn('int8_t', source)

    def test_int8_quant_scatter_reduction(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_root = os.path.join(td, 'onnx-for-mcu')
            for name, builder in (
                ('qdq_scatter_elements_add_int8.onnx', _build_qdq_scatter_elements_reduction_model),
                ('qdq_scatter_nd_add_int8.onnx', _build_qdq_scatter_nd_reduction_model),
            ):
                model_path = os.path.join(td, name)
                builder(model_path, TensorProto.INT8, 'add')
                model = load_onnx_model(model_path)
                _ = _eval_model(model, {'input': np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]], dtype=np.float32) if 'elements' in name else np.array([0.5, 0.6, 0.1, 0.0], dtype=np.float32)})
                result = generate_tinyml_project(
                    model_path,
                    out_root,
                    weights='flash',
                    emit='c',
                )
                source = Path(result['source']).read_text(encoding='utf-8')
                manifest = Path(result['manifest']).read_text(encoding='utf-8')
                self.assertIn('int8_t', source)
                self.assertTrue('ScatterElements' in manifest or 'ScatterND' in manifest)

    def test_int16_quant_scatter_reduction(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_root = os.path.join(td, 'onnx-for-mcu')
            for name, builder in (
                ('qdq_scatter_elements_add_int16.onnx', _build_qdq_scatter_elements_reduction_model),
                ('qdq_scatter_nd_add_int16.onnx', _build_qdq_scatter_nd_reduction_model),
            ):
                model_path = os.path.join(td, name)
                builder(model_path, TensorProto.INT16, 'add')
                model = load_onnx_model(model_path)
                _ = _eval_model(model, {'input': np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]], dtype=np.float32) if 'elements' in name else np.array([0.5, 0.6, 0.1, 0.0], dtype=np.float32)})
                result = generate_tinyml_project(
                    model_path,
                    out_root,
                    weights='flash',
                    emit='c',
                )
                source = Path(result['source']).read_text(encoding='utf-8')
                manifest = Path(result['manifest']).read_text(encoding='utf-8')
                self.assertIn('int16_t', source)
                self.assertTrue('ScatterElements' in manifest or 'ScatterND' in manifest)

    def test_int8_quant_tile(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'qdq_tile.onnx')
            _build_qdq_tile_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Tile', manifest)
            self.assertIn('int8_t', source)

    def test_int8_quant_resize(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'qdq_resize.onnx')
            _build_qdq_resize_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Resize', manifest)
            self.assertIn('int8_t', source)

    def test_int8_quant_elu_selu(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'qdq_elu_selu.onnx')
            _build_qdq_elu_selu_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Elu', manifest)
            self.assertIn('Selu', manifest)
            self.assertIn('int8_t', source)

    def test_int8_quant_sign(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'qdq_sign.onnx')
            _build_qdq_sign_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Sign', manifest)
            self.assertIn('int8_t', source)

    def test_int8_quant_erf_round(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'qdq_erf_round.onnx')
            _build_qdq_erf_round_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Erf', manifest)
            self.assertIn('Round', manifest)
            self.assertIn('int8_t', source)

    def test_unary_trig_hyper_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'trig_hyper.onnx')
            _build_trig_hyper_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Acosh', manifest)
            self.assertIn('Atanh', manifest)
            self.assertIn('Cosh', manifest)
            self.assertIn('Sinh', manifest)

    def test_int8_quant_tan_sinh(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'qdq_tan_sinh.onnx')
            _build_qdq_tan_sinh_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Tan', manifest)
            self.assertIn('Cosh', manifest)
            self.assertIn('int8_t', source)

    def test_mean_sum_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'mean_sum.onnx')
            _build_mean_sum_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Mean', manifest)
            self.assertIn('Sum', manifest)

    def test_shape_size_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'shape_size.onnx')
            _build_shape_size_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Shape', manifest)
            self.assertIn('Size', manifest)

    def test_constant_of_shape_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'constant_of_shape.onnx')
            _build_constant_of_shape_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('ConstantOfShape', manifest)

    def test_split_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'split.onnx')
            _build_split_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Split', manifest)

    def test_int8_quant_mean_sum(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'qdq_mean_sum.onnx')
            _build_qdq_mean_sum_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('Mean', manifest)
            self.assertIn('Sum', manifest)
            self.assertIn('int8_t', source)

    def test_int8_quant_reduce_extended(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'qdq_reduce_extended.onnx')
            _build_qdq_reduce_extended_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('ReduceProd', manifest)
            self.assertIn('ReduceL1', manifest)
            self.assertIn('ReduceL2', manifest)
            self.assertIn('int8_t', source)

    def test_int8_quant_reduce_sum_square(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'qdq_reduce_sum_square.onnx')
            _build_qdq_reduce_sum_square_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('ReduceSumSquare', manifest)
            self.assertIn('int8_t', source)

    def test_int8_quant_squeeze_unsqueeze_axes_input(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'qdq_squeeze_unsqueeze_axes_input.onnx')
            _build_qdq_squeeze_unsqueeze_axes_input_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('Squeeze', manifest)
            self.assertIn('Unsqueeze', manifest)
            self.assertIn('int8_t', source)

    def test_int8_quant_split(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'qdq_split.onnx')
            _build_qdq_split_model(model_path, TensorProto.INT8)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertIn('Split', manifest)
            self.assertIn('int8_t', source)

    def test_pow_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'pow.onnx')
            _build_pow_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            self.assertTrue(os.path.exists(result['source']))

    def test_max_min_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'maxmin.onnx')
            _build_max_min_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            self.assertTrue(os.path.exists(result['source']))

    def test_reduce_max_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'reduce_max.onnx')
            _build_reduce_max_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            self.assertTrue(os.path.exists(result['source']))

    def test_reduce_min_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'reduce_min.onnx')
            _build_reduce_min_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            self.assertTrue(os.path.exists(result['source']))

    def test_reduce_extended_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'reduce_extended.onnx')
            _build_reduce_extended_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('ReduceProd', manifest)
            self.assertIn('ReduceL1', manifest)
            self.assertIn('ReduceL2', manifest)
            self.assertIn('ReduceSumSquare', manifest)

    def test_squeeze_unsqueeze_axes_input_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'squeeze_unsqueeze_axes_input.onnx')
            _build_squeeze_unsqueeze_axes_input_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Squeeze', manifest)
            self.assertIn('Unsqueeze', manifest)

    def test_reduce_log_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'reduce_log.onnx')
            _build_reduce_log_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('ReduceLogSum', manifest)
            self.assertIn('ReduceLogSumExp', manifest)

    def test_prelu_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'prelu.onnx')
            _build_prelu_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('PRelu', manifest)

    def test_mod_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'mod.onnx')
            _build_mod_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Mod', manifest)

    def test_dropout_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'dropout.onnx')
            _build_dropout_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Dropout', manifest)

    def test_upsample_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'upsample.onnx')
            _build_upsample_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Upsample', manifest)

    def test_constant_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'constant.onnx')
            _build_constant_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Constant', manifest)

    def test_hardmax_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'hardmax.onnx')
            _build_hardmax_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Hardmax', manifest)

    def test_logsoftmax_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'logsoftmax.onnx')
            _build_logsoftmax_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('LogSoftmax', manifest)

    def test_thresholded_relu_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'thresholded_relu.onnx')
            _build_thresholded_relu_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('ThresholdedRelu', manifest)

    def test_range_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'range.onnx')
            _build_range_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('Range', manifest)

    def test_gather_elements_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'gather_elements.onnx')
            _build_gather_elements_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('GatherElements', manifest)

    def test_scatter_elements_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'scatter_elements.onnx')
            _build_scatter_elements_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('ScatterElements', manifest)

    def test_scatter_nd_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'scatter_nd.onnx')
            _build_scatter_nd_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('ScatterND', manifest)

    def test_scatter_elements_reduction_modes(self) -> None:
        expected: dict[str, np.ndarray] = {
            'none': np.array([[20.0, 2.0, 30.0], [4.0, 4.0, 6.0]], dtype=np.float32),
            'add': np.array([[31.0, 2.0, 33.0], [4.0, 14.0, 6.0]], dtype=np.float32),
            'mul': np.array([[200.0, 2.0, 90.0], [4.0, 120.0, 6.0]], dtype=np.float32),
            'max': np.array([[20.0, 2.0, 30.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            'min': np.array([[1.0, 2.0, 3.0], [4.0, 2.0, 6.0]], dtype=np.float32),
        }
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            out_root = os.path.join(td, 'onnx-for-mcu')
            for reduction, target in expected.items():
                model_path = os.path.join(td, f'scatter_elements_{reduction}.onnx')
                _build_scatter_elements_reduction_model(model_path, reduction)
                model = load_onnx_model(model_path)
                out = _eval_model(model, {'input': data})[model.outputs[0].name]
                np.testing.assert_allclose(out, target, rtol=0.0, atol=0.0)
                result = generate_tinyml_project(
                    model_path,
                    out_root,
                    weights='flash',
                    emit='c',
                )
                manifest = Path(result['manifest']).read_text(encoding='utf-8')
                self.assertIn('ScatterElements', manifest)

    def test_scatter_nd_reduction_modes(self) -> None:
        expected: dict[str, np.ndarray] = {
            'none': np.array([5.0, 9.0, 8.0, 0.0], dtype=np.float32),
            'add': np.array([5.0, 18.0, 9.0, 0.0], dtype=np.float32),
            'mul': np.array([5.0, 162.0, 8.0, 0.0], dtype=np.float32),
            'max': np.array([5.0, 9.0, 8.0, 0.0], dtype=np.float32),
            'min': np.array([5.0, 3.0, 1.0, 0.0], dtype=np.float32),
        }
        data = np.array([5.0, 6.0, 1.0, 0.0], dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            out_root = os.path.join(td, 'onnx-for-mcu')
            for reduction, target in expected.items():
                model_path = os.path.join(td, f'scatter_nd_{reduction}.onnx')
                _build_scatter_nd_reduction_model(model_path, reduction)
                model = load_onnx_model(model_path)
                out = _eval_model(model, {'input': data})[model.outputs[0].name]
                np.testing.assert_allclose(out, target, rtol=0.0, atol=0.0)
                result = generate_tinyml_project(
                    model_path,
                    out_root,
                    weights='flash',
                    emit='c',
                )
                manifest = Path(result['manifest']).read_text(encoding='utf-8')
                self.assertIn('ScatterND', manifest)

    def test_isinf_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'isinf.onnx')
            _build_isinf_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('IsInf', manifest)

    def test_isnan_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'isnan.onnx')
            _build_isnan_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('IsNaN', manifest)

    def test_nonzero_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'nonzero.onnx')
            _build_nonzero_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('NonZero', manifest)

    def test_topk_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'topk.onnx')
            _build_topk_model(model_path)
            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('TopK', manifest)

    def test_topk_int32_precision_consistency(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'topk_int32_precision.onnx')
            _build_topk_int32_precision_model(model_path)
            model = load_onnx_model(model_path)
            out = _eval_model(model, {'dummy': np.array([0.0], dtype=np.float32)})
            np.testing.assert_array_equal(out['values'], np.array([16777217], dtype=np.int32))
            np.testing.assert_array_equal(out['indices'], np.array([1], dtype=np.int64))

            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            self.assertNotIn('float cur = (float)', source)
            self._assert_model_consistency_regression(
                model_path,
                result,
                seeds=(0,),
                rtol=0.0,
                atol=0.0,
                int8_atol=0.0,
                int16_atol=0.0,
                require_onnxruntime=False,
            )

    def test_non_max_suppression_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'nms.onnx')
            _build_non_max_suppression_model(model_path)
            model = load_onnx_model(model_path)
            boxes = np.array(
                [
                    [
                        [0.0, 0.0, 1.0, 1.0],
                        [0.1, 0.1, 1.1, 1.1],
                        [2.0, 2.0, 3.0, 3.0],
                        [0.0, 0.0, 0.5, 0.5],
                    ]
                ],
                dtype=np.float32,
            )
            out = _eval_model(model, {'boxes': boxes})[model.outputs[0].name]
            expected = np.array([[0, 0, 0], [0, 0, 2], [0, 0, 3]], dtype=np.int64)
            np.testing.assert_array_equal(out, expected)

            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('NonMaxSuppression', manifest)
            self.assertIn('k2c_nms_out_pos', source)

    def test_non_max_suppression_dynamic_scalars_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'nms_dynamic_scalar.onnx')
            _build_non_max_suppression_dynamic_scalars_model(model_path)
            model = load_onnx_model(model_path)
            boxes = np.array(
                [
                    [
                        [0.0, 0.0, 1.0, 1.0],
                        [0.1, 0.1, 1.1, 1.1],
                        [2.0, 2.0, 3.0, 3.0],
                        [0.0, 0.0, 0.5, 0.5],
                    ]
                ],
                dtype=np.float32,
            )
            out = _eval_model(model, {'boxes': boxes})[model.outputs[0].name]
            expected = np.array([[0, 0, 0], [0, 0, 2], [0, 0, 3]], dtype=np.int64)
            np.testing.assert_array_equal(out, expected)

            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('NonMaxSuppression', manifest)
            self.assertIn('k2c_nms_max_output', source)
            self.assertIn('k2c_nms_iou_threshold', source)
            self.assertIn('k2c_nms_score_threshold', source)

    def test_dynamic_quantize_linear_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'dynamic_quantize_linear.onnx')
            _build_dynamic_quantize_linear_model(model_path)
            model = load_onnx_model(model_path)
            x = np.array([[-1.5, -0.75, -0.3, 0.0, 0.2, 0.7, 1.2, 1.8]], dtype=np.float32)
            out = _eval_model(model, {'input': x})[model.outputs[0].name]
            self.assertEqual(out.shape, x.shape)
            self.assertTrue(np.all(np.isfinite(out)))

            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('DynamicQuantizeLinear', manifest)
            self.assertIn('k2c_dq_scale', source)

    def test_roi_align_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_path = os.path.join(td, 'roi_align.onnx')
            _build_roi_align_model(model_path)
            model = load_onnx_model(model_path)
            in_data = np.arange(25, dtype=np.float32).reshape(1, 1, 5, 5)
            out = _eval_model(model, {'input': in_data})[model.outputs[0].name]
            self.assertEqual(out.shape, (2, 1, 2, 2))
            self.assertTrue(np.all(np.isfinite(out)))

            out_root = os.path.join(td, 'onnx-for-mcu')
            result = generate_tinyml_project(
                model_path,
                out_root,
                weights='flash',
                emit='c',
            )
            source = Path(result['source']).read_text(encoding='utf-8')
            manifest = Path(result['manifest']).read_text(encoding='utf-8')
            self.assertIn('RoiAlign', manifest)
            self.assertIn('samp_h', source)

