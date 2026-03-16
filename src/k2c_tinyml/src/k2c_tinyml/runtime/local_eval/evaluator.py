# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np

from ...converter.ir import ModelIR
from .families import (
    handle_index_family,
    handle_logic_family,
    handle_math_family,
    handle_nn_family,
    handle_quant_family,
    handle_shape_family,
    handle_vision_family,
)


def _eval_model(model: ModelIR, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    for name, tensor in model.tensors.items():
        if tensor.data is None:
            continue
        dtype = tensor.dtype
        if dtype == "float32":
            arr = np.array(tensor.data, dtype=np.float32)
        elif dtype == "bool":
            arr = np.array(tensor.data, dtype=np.bool_)
        elif dtype == "uint8":
            arr = np.array(tensor.data, dtype=np.uint8)
        elif dtype == "int8":
            arr = np.array(tensor.data, dtype=np.int8)
        elif dtype == "int16":
            arr = np.array(tensor.data, dtype=np.int16)
        elif dtype == "int32":
            arr = np.array(tensor.data, dtype=np.int32)
        elif dtype == "int64":
            arr = np.array(tensor.data, dtype=np.int64)
        else:
            raise ValueError("Unsupported const dtype.")
        shape = list(tensor.shape)
        if len(shape) == 0:
            arr = arr.reshape(())
        else:
            arr = arr.reshape(shape)
        tensors[name] = arr
    for name, arr in inputs.items():
        tensors[name] = arr

    handlers = (
        handle_quant_family,
        handle_logic_family,
        handle_math_family,
        handle_nn_family,
        handle_vision_family,
        handle_index_family,
        handle_shape_family,
    )

    for node in model.nodes:
        out_name = node.outputs[0]
        out_dtype = model.tensors[out_name].dtype
        ins = [tensors[name] if name else None for name in node.inputs]

        handled = False
        for handle in handlers:
            if handle(model, node, tensors, ins, out_name, out_dtype):
                handled = True
                break
        if not handled:
            raise ValueError(f"Validation: unsupported op {node.op_type}.")

    return tensors
