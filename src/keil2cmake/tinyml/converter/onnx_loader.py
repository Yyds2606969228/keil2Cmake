# -*- coding: utf-8 -*-

from __future__ import annotations

import onnx
from onnx import numpy_helper, shape_inference

from .ir import ModelIR, NodeInfo, TensorInfo
from .lowering import constant_tensor_from_attrs, lower_placeholder_ops
from .onnx_loader_helpers import (
    _apply_qparams,
    _dtype_from_tensorproto,
    _infer_shapes,
    _is_initializer,
    _propagate_qparams,
    _shape_from_value,
    _shape_known,
    _tensor_dtype,
    _tensor_size,
)


def load_onnx_model(path: str) -> ModelIR:
    model = onnx.load(path)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    graph = model.graph
    opset = 0
    if model.opset_import:
        opset = int(model.opset_import[0].version)

    initializer_names = {init.name for init in graph.initializer}
    tensors: dict[str, TensorInfo] = {}

    for init in graph.initializer:
        array = numpy_helper.to_array(init)
        dtype = _dtype_from_tensorproto(int(init.data_type))
        if dtype == "float32":
            tensors[init.name] = TensorInfo(
                name=init.name,
                shape=list(array.shape),
                dtype=dtype,
                data=array.flatten().astype("float32").tolist(),
            )
        elif dtype == "bool":
            tensors[init.name] = TensorInfo(
                name=init.name,
                shape=list(array.shape),
                dtype=dtype,
                data=array.flatten().astype("int64").tolist(),
            )
        elif dtype in ("uint8", "int64", "int32", "int8", "int16"):
            tensors[init.name] = TensorInfo(
                name=init.name,
                shape=list(array.shape),
                dtype=dtype,
                data=array.flatten().astype("int64").tolist(),
            )
        else:
            raise ValueError("Unsupported initializer tensor type.")

    inputs = []
    for inp in graph.input:
        if _is_initializer(inp.name, initializer_names):
            continue
        dtype = _dtype_from_tensorproto(_tensor_dtype(inp))
        if dtype not in ("float32", "uint8", "int8", "int16", "bool", "int32", "int64"):
            raise ValueError("Only FLOAT/UINT8/INT8/INT16/BOOL/INT32/INT64 inputs are supported in this version.")
        shape = _shape_from_value(inp)
        _tensor_size(shape)
        tensor = TensorInfo(name=inp.name, shape=shape, dtype=dtype)
        tensors[inp.name] = tensor
        inputs.append(tensor)

    outputs = []
    output_names = []
    for out in graph.output:
        dtype = _dtype_from_tensorproto(_tensor_dtype(out))
        if dtype not in ("float32", "uint8", "int8", "int16", "bool", "int32", "int64"):
            raise ValueError("Only FLOAT/UINT8/INT8/INT16/BOOL/INT32/INT64 outputs are supported in this version.")
        shape = _shape_from_value(out)
        tensor = TensorInfo(name=out.name, shape=shape, dtype=dtype)
        tensors[out.name] = tensor
        outputs.append(tensor)
        output_names.append(out.name)

    for val in graph.value_info:
        if val.name in tensors:
            continue
        dtype = _dtype_from_tensorproto(_tensor_dtype(val))
        if dtype not in ("float32", "uint8", "int8", "int16", "bool", "int32", "int64"):
            continue
        shape = _shape_from_value(val)
        tensors[val.name] = TensorInfo(name=val.name, shape=shape, dtype=dtype)

    nodes = []
    for node in graph.node:
        attrs = {}
        for attr in node.attribute:
            attrs[attr.name] = onnx.helper.get_attribute_value(attr)
        node_info = NodeInfo(
            op_type=node.op_type,
            inputs=list(node.input),
            outputs=list(node.output),
            attrs=attrs,
        )
        nodes.append(node_info)
        if node_info.op_type == "Constant" and node_info.outputs:
            out_name = node_info.outputs[0]
            const_tensor = constant_tensor_from_attrs(out_name, node_info.attrs)
            existing = tensors.get(out_name)
            if const_tensor is None:
                continue
            if existing is None:
                tensors[out_name] = const_tensor
                continue
            shape = existing.shape if _shape_known(existing.shape) else list(const_tensor.shape)
            dtype = existing.dtype if existing.dtype != "unknown" else const_tensor.dtype
            tensors[out_name] = TensorInfo(
                name=existing.name,
                shape=shape,
                dtype=dtype,
                data=const_tensor.data,
                qscale=existing.qscale,
                qzero=existing.qzero,
            )

    _infer_shapes(tensors, nodes)
    nodes = lower_placeholder_ops(nodes, tensors)
    _apply_qparams(tensors, nodes)
    _infer_shapes(tensors, nodes)
    _propagate_qparams(tensors, nodes)
    outputs = [tensors[name] for name in output_names]
    for out in outputs:
        _tensor_size(out.shape)

    model_name = model.graph.name or "model"
    return ModelIR(
        name=model_name,
        opset=opset,
        inputs=inputs,
        outputs=outputs,
        tensors=tensors,
        nodes=nodes,
    )
