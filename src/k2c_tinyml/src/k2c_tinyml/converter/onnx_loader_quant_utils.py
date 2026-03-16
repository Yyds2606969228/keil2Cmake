# -*- coding: utf-8 -*-

from __future__ import annotations

from onnx import TensorProto

from .ir import NodeInfo, TensorInfo

def _dtype_from_tensorproto(dtype: int) -> str:
    if dtype == TensorProto.FLOAT:
        return "float32"
    if dtype == TensorProto.BOOL:
        return "bool"
    if dtype == TensorProto.UINT8:
        return "uint8"
    if dtype == TensorProto.INT8:
        return "int8"
    if dtype == TensorProto.INT16:
        return "int16"
    if dtype == TensorProto.INT64:
        return "int64"
    if dtype == TensorProto.INT32:
        return "int32"
    return "unknown"


def _const_scalar_float(tensors: dict[str, TensorInfo], name: str) -> float | None:
    tensor = tensors.get(name)
    if tensor is None or tensor.data is None or len(tensor.data) == 0:
        return None
    return float(tensor.data[0])


def _const_scalar_int(tensors: dict[str, TensorInfo], name: str) -> int | None:
    tensor = tensors.get(name)
    if tensor is None or tensor.data is None or len(tensor.data) == 0:
        return None
    return int(tensor.data[0])


def _apply_qparams(tensors: dict[str, TensorInfo], nodes: list[NodeInfo]) -> None:
    for node in nodes:
        if node.op_type == "QuantizeLinear":
            if len(node.inputs) < 2 or len(node.outputs) < 1:
                continue
            scale = _const_scalar_float(tensors, node.inputs[1])
            if scale is None:
                continue
            zero = 0
            if len(node.inputs) >= 3:
                zero_val = _const_scalar_int(tensors, node.inputs[2])
                if zero_val is None:
                    continue
                zero = zero_val
            out_name = node.outputs[0]
            qdtype = "int8"
            if len(node.inputs) >= 3:
                zp_tensor = tensors.get(node.inputs[2])
                if zp_tensor is not None and zp_tensor.dtype in ("uint8", "int8", "int16"):
                    qdtype = zp_tensor.dtype
            out_tensor = tensors.get(out_name)
            if out_tensor is None:
                tensors[out_name] = TensorInfo(
                    name=out_name,
                    shape=[],
                    dtype=qdtype,
                    qscale=scale,
                    qzero=zero,
                )
            else:
                tensors[out_name] = TensorInfo(
                    name=out_tensor.name,
                    shape=out_tensor.shape,
                    dtype=qdtype,
                    data=out_tensor.data,
                    qscale=scale,
                    qzero=zero,
                )
        elif node.op_type == "DequantizeLinear":
            if len(node.inputs) < 2:
                continue
            scale = _const_scalar_float(tensors, node.inputs[1])
            if scale is None:
                continue
            zero = 0
            if len(node.inputs) >= 3:
                zero_val = _const_scalar_int(tensors, node.inputs[2])
                if zero_val is None:
                    continue
                zero = zero_val
            in_name = node.inputs[0]
            in_tensor = tensors.get(in_name)
            if in_tensor is None:
                continue
            qdtype = in_tensor.dtype
            if qdtype not in ("uint8", "int8", "int16"):
                if len(node.inputs) >= 3:
                    zp_tensor = tensors.get(node.inputs[2])
                    if zp_tensor is not None and zp_tensor.dtype in ("uint8", "int8", "int16"):
                        qdtype = zp_tensor.dtype
            tensors[in_name] = TensorInfo(
                name=in_tensor.name,
                shape=in_tensor.shape,
                dtype=qdtype,
                data=in_tensor.data,
                qscale=scale,
                qzero=zero,
            )
        elif node.op_type in ("QLinearMatMul", "QLinearConv"):
            if len(node.inputs) < 8 or len(node.outputs) < 1:
                continue
            scale = _const_scalar_float(tensors, node.inputs[6])
            zero = _const_scalar_int(tensors, node.inputs[7])
            if scale is None or zero is None:
                continue
            out_name = node.outputs[0]
            qdtype = "int8"
            zp_tensor = tensors.get(node.inputs[7])
            if zp_tensor is not None and zp_tensor.dtype in ("uint8", "int8", "int16"):
                qdtype = zp_tensor.dtype
            out_tensor = tensors.get(out_name)
            if out_tensor is None:
                tensors[out_name] = TensorInfo(
                    name=out_name,
                    shape=[],
                    dtype=qdtype,
                    qscale=scale,
                    qzero=zero,
                )
            else:
                out_dtype = out_tensor.dtype if out_tensor.dtype in ("uint8", "int8", "int16") else qdtype
                tensors[out_name] = TensorInfo(
                    name=out_tensor.name,
                    shape=out_tensor.shape,
                    dtype=out_dtype,
                    data=out_tensor.data,
                    qscale=scale,
                    qzero=zero,
                )


def _propagate_qparams(tensors: dict[str, TensorInfo], nodes: list[NodeInfo]) -> None:
    quant_ops = {
        "Add",
        "Sum",
        "Mean",
        "Sub",
        "Mul",
        "Div",
        "And",
        "Or",
        "Xor",
        "Not",
        "Relu",
        "ThresholdedRelu",
        "LeakyRelu",
        "Elu",
        "Celu",
        "Selu",
        "Dropout",
        "Sign",
        "Sigmoid",
        "Tanh",
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
        "Softplus",
        "Softsign",
        "Pow",
        "Identity",
        "Abs",
        "Neg",
        "Clip",
        "Max",
        "Min",
        "Where",
        "Conv",
        "MatMul",
        "Gemm",
        "Einsum",
        "Det",
        "PRelu",
        "BatchNormalization",
        "InstanceNormalization",
        "MeanVarianceNormalization",
        "LRN",
        "LpNormalization",
        "LpPool",
        "GlobalLpPool",
        "ConvTranspose",
        "Reshape",
        "Gather",
        "GatherND",
        "GatherElements",
        "ScatterElements",
        "ScatterND",
        "Squeeze",
        "Unsqueeze",
        "Concat",
        "Split",
        "Softmax",
        "LogSoftmax",
        "Hardmax",
        "TopK",
        "Pad",
        "Slice",
        "Flatten",
        "Transpose",
        "Range",
        "Mod",
        "ReduceMean",
        "ReduceSum",
        "ReduceLogSum",
        "ReduceLogSumExp",
        "ReduceMax",
        "ReduceMin",
        "IsInf",
        "IsNaN",
        "NonZero",
        "NonMaxSuppression",
        "RoiAlign",
        "GlobalMaxPool",
        "ReduceProd",
        "ReduceL1",
        "ReduceL2",
        "ReduceSumSquare",
        "Expand",
        "Tile",
        "Resize",
        "Upsample",
        "SpaceToDepth",
        "DepthToSpace",
        "ReverseSequence",
        "RNN",
        "GRU",
        "LSTM",
    }
    for node in nodes:
        if node.op_type not in quant_ops:
            continue
        base = None
        for name in node.inputs:
            tensor = tensors.get(name)
            if tensor is None or tensor.dtype not in ("uint8", "int8", "int16"):
                continue
            if tensor.qscale is None or tensor.qzero is None:
                continue
            base = (float(tensor.qscale), int(tensor.qzero), tensor.dtype)
            break
        if base is None:
            continue
        base_scale, base_zero, base_dtype = base
        for out_name in node.outputs:
            out_tensor = tensors.get(out_name)
            if out_tensor is None:
                tensors[out_name] = TensorInfo(
                    name=out_name,
                    shape=[],
                    dtype=base_dtype,
                    qscale=base_scale,
                    qzero=base_zero,
                )
                continue
            if out_tensor.dtype not in ("uint8", "int8", "int16"):
                if out_tensor.dtype == "unknown":
                    tensors[out_name] = TensorInfo(
                        name=out_tensor.name,
                        shape=out_tensor.shape,
                        dtype=base_dtype,
                        data=out_tensor.data,
                        qscale=base_scale,
                        qzero=base_zero,
                    )
                continue
            if out_tensor.qscale is not None and out_tensor.qzero is not None:
                continue
            tensors[out_name] = TensorInfo(
                name=out_tensor.name,
                shape=out_tensor.shape,
                dtype=out_tensor.dtype,
                data=out_tensor.data,
                qscale=base_scale,
                qzero=base_zero,
            )


