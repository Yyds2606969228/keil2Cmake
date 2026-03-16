# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import re

from .template_engine import write_template
from .backends import get_backend
from .ir import ModelIR
from .operators import EmitContext
from .operators.utils import get_shape, tensor_size


_C_IDENTIFIER_RE = re.compile(r"[^0-9a-zA-Z_]")


def _sanitize(name: str) -> str:
    cleaned = _C_IDENTIFIER_RE.sub("_", name)
    if not cleaned:
        cleaned = "tensor"
    if cleaned[0].isdigit():
        cleaned = f"tensor_{cleaned}"
    return cleaned


def _validate_io(model: ModelIR) -> tuple[list[str], list[str]]:
    if not model.inputs:
        raise ValueError("Model has no inputs.")
    if not model.outputs:
        raise ValueError("Model has no outputs.")
    return [t.name for t in model.inputs], [t.name for t in model.outputs]


def _align_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


def generate_c_code(
    model: ModelIR,
    output_dir: str,
    model_name: str,
    weights: str,
) -> dict[str, str]:
    input_names, output_names = _validate_io(model)

    os.makedirs(output_dir, exist_ok=True)
    backend_impl = get_backend()

    consts: dict[str, str] = {}
    buffers: dict[str, str] = {}
    weights_map: dict[str, str] = {}
    weights_ram = weights == "ram"

    weight_names: list[str] = []
    for name, tensor in model.tensors.items():
        if tensor.data is not None:
            weight_names.append(name)
            consts[name] = f"const_{_sanitize(name)}"

    buffer_names: list[str] = []
    output_name_set = set(output_names)
    for node in model.nodes:
        for out_name in node.outputs:
            if out_name in output_name_set:
                continue
            if out_name not in buffer_names:
                buffer_names.append(out_name)

    buffer_offsets: list[int] = []
    weight_offsets: list[int] = []
    weight_sizes: list[int] = []
    offset = 0
    def _dtype_size(dtype: str) -> int:
        if dtype == "uint8":
            return 1
        if dtype == "int8":
            return 1
        if dtype == "bool":
            return 1
        if dtype == "int16":
            return 2
        if dtype == "int32":
            return 4
        if dtype == "int64":
            return 8
        return 4

    for name in buffer_names:
        shape = get_shape(model, name)
        dtype = model.tensors[name].dtype
        size_bytes = tensor_size(shape) * _dtype_size(dtype)
        offset = _align_up(offset, 4)
        buffer_offsets.append(offset)
        offset += size_bytes
    if weights_ram:
        for name in weight_names:
            shape = get_shape(model, name)
            dtype = model.tensors[name].dtype
            size_bytes = tensor_size(shape) * _dtype_size(dtype)
            offset = _align_up(offset, 4)
            weight_offsets.append(offset)
            weight_sizes.append(size_bytes)
            offset += size_bytes
    arena_bytes = _align_up(offset, 4)

    for idx, name in enumerate(buffer_names):
        buffers[name] = f"ctx->buffers[{idx}]"
    if weights_ram:
        for idx, name in enumerate(weight_names):
            weights_map[name] = f"ctx->weights[{idx}]"

    input_shapes = [get_shape(model, name) for name in input_names]
    output_shapes = [get_shape(model, name) for name in output_names]
    input_dtypes = [model.tensors[name].dtype for name in input_names]
    output_dtypes = [model.tensors[name].dtype for name in output_names]
    input_sizes = [tensor_size(shape) for shape in input_shapes]
    output_sizes = [tensor_size(shape) for shape in output_shapes]
    input_total_size = sum(input_sizes)
    output_total_size = sum(output_sizes)

    header_name = f"{model_name}.h"
    source_name = f"{model_name}.c"

    header_path = os.path.join(output_dir, header_name)
    source_path = os.path.join(output_dir, source_name)
    arena_words = max(1, (arena_bytes + 3) // 4)
    guard = f"K2C_{_sanitize(model_name).upper()}_H"

    const_decls: list[dict[str, object]] = []
    for name, tensor in model.tensors.items():
        if tensor.data is None:
            continue
        size = tensor_size(tensor.shape)
        dtype = tensor.dtype
        if dtype == "float32":
            data = ", ".join(f"{v:.8f}f" for v in tensor.data)
            const_decls.append({"ctype": "float", "name": consts[name], "size": size, "values": data})
        elif dtype == "bool":
            data = ", ".join("1" if int(v) else "0" for v in tensor.data)
            const_decls.append({"ctype": "uint8_t", "name": consts[name], "size": size, "values": data})
        elif dtype == "uint8":
            data = ", ".join(str(int(v)) for v in tensor.data)
            const_decls.append({"ctype": "uint8_t", "name": consts[name], "size": size, "values": data})
        elif dtype == "int8":
            data = ", ".join(str(int(v)) for v in tensor.data)
            const_decls.append({"ctype": "int8_t", "name": consts[name], "size": size, "values": data})
        elif dtype == "int16":
            data = ", ".join(str(int(v)) for v in tensor.data)
            const_decls.append({"ctype": "int16_t", "name": consts[name], "size": size, "values": data})
        elif dtype == "int32":
            data = ", ".join(str(int(v)) for v in tensor.data)
            const_decls.append({"ctype": "int32_t", "name": consts[name], "size": size, "values": data})
        elif dtype == "int64":
            data = ", ".join(str(int(v)) for v in tensor.data)
            const_decls.append({"ctype": "int64_t", "name": consts[name], "size": size, "values": data})
        else:
            raise ValueError(f"Unsupported const dtype: {dtype}")

    input_shape_decls: list[str] = []
    output_shape_decls: list[str] = []
    input_descs: list[dict[str, object]] = []
    output_descs: list[dict[str, object]] = []

    def _ctype_size(dtype: str) -> str:
        if dtype == "uint8":
            return "sizeof(uint8_t)"
        if dtype == "int8":
            return "sizeof(int8_t)"
        if dtype == "bool":
            return "sizeof(uint8_t)"
        if dtype == "int16":
            return "sizeof(int16_t)"
        if dtype == "int32":
            return "sizeof(int32_t)"
        if dtype == "int64":
            return "sizeof(int64_t)"
        return "sizeof(float)"

    for idx, name in enumerate(input_names):
        shape = input_shapes[idx]
        ptr = f"k2c_input_shape_{idx}"
        if len(shape) == 0:
            input_shape_decls.append(f"static const int {ptr}[1] = {{ 1 }};")
            ptr = "NULL"
        else:
            values = ", ".join(str(v) for v in shape)
            input_shape_decls.append(f"static const int {ptr}[{len(shape)}] = {{ {values} }};")
        input_descs.append(
            {
                "name": name,
                "shape_ptr": ptr,
                "rank": len(shape),
                "elem_size": _ctype_size(input_dtypes[idx]),
                "size": input_sizes[idx],
            }
        )

    for idx, name in enumerate(output_names):
        shape = output_shapes[idx]
        ptr = f"k2c_output_shape_{idx}"
        if len(shape) == 0:
            output_shape_decls.append(f"static const int {ptr}[1] = {{ 1 }};")
            ptr = "NULL"
        else:
            values = ", ".join(str(v) for v in shape)
            output_shape_decls.append(f"static const int {ptr}[{len(shape)}] = {{ {values} }};")
        output_descs.append(
            {
                "name": name,
                "shape_ptr": ptr,
                "rank": len(shape),
                "elem_size": _ctype_size(output_dtypes[idx]),
                "size": output_sizes[idx],
            }
        )

    includes = [
        "#include <math.h>",
        "#include <string.h>",
        "#include <stdint.h>",
    ] + backend_impl.extra_includes

    invoke_lines: list[str] = []
    invoke_lines.append("  if (!ctx || !input_ptrs || !output_ptrs) return -1;")
    def _io_ctype(dtype: str) -> str:
        if dtype == "float32":
            return "float"
        if dtype == "bool":
            return "uint8_t"
        if dtype == "uint8":
            return "uint8_t"
        if dtype == "int8":
            return "int8_t"
        if dtype == "int16":
            return "int16_t"
        if dtype == "int32":
            return "int32_t"
        if dtype == "int64":
            return "int64_t"
        raise ValueError(f"Unsupported IO dtype: {dtype}")

    input_ptrs: dict[str, str] = {}
    output_ptrs: dict[str, str] = {}
    for idx, name in enumerate(input_names):
        invoke_lines.append(f"  if (!input_ptrs[{idx}]) return -1;")
        ctype = _io_ctype(input_dtypes[idx])
        var = f"input_{idx}"
        invoke_lines.append(f"  const {ctype}* {var} = (const {ctype}*)input_ptrs[{idx}];")
        input_ptrs[name] = var
    for idx, name in enumerate(output_names):
        invoke_lines.append(f"  if (!output_ptrs[{idx}]) return -1;")
        ctype = _io_ctype(output_dtypes[idx])
        var = f"output_{idx}"
        invoke_lines.append(f"  {ctype}* {var} = ({ctype}*)output_ptrs[{idx}];")
        output_ptrs[name] = var

    ctx = EmitContext(
        lines=invoke_lines,
        model=model,
        input_ptrs=input_ptrs,
        output_ptrs=output_ptrs,
        buffers=buffers,
        consts=consts,
        weights=weights_map,
    )
    unsupported_ops: list[str] = []
    for node in model.nodes:
        if backend_impl.get_handler(node.op_type) is None:
            unsupported_ops.append(node.op_type)
    if unsupported_ops:
        missing = ", ".join(sorted(set(unsupported_ops)))
        raise ValueError(f"Unsupported operators for backend '{backend_impl.name}': {missing}")
    quant_ops = {
        "QuantizeLinear",
        "DequantizeLinear",
        "DynamicQuantizeLinear",
        "Cast",
        "Gather",
        "GatherND",
        "Scatter",
        "ScatterElements",
        "ScatterND",
        "Reshape",
        "Add",
        "Sum",
        "Mean",
        "Split",
        "Shape",
        "Size",
        "Constant",
        "ConstantOfShape",
        "EyeLike",
        "Compress",
        "Concat",
        "GatherElements",
        "Sub",
        "Mul",
        "Div",
        "BitShift",
        "And",
        "Or",
        "Xor",
        "Not",
        "Equal",
        "Greater",
        "Less",
        "GreaterOrEqual",
        "LessOrEqual",
        "Relu",
        "ThresholdedRelu",
        "Shrink",
        "LeakyRelu",
        "Elu",
        "Celu",
        "Selu",
        "Dropout",
        "CumSum",
        "Sigmoid",
        "Tanh",
        "Exp",
        "Sign",
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
        "ArgMax",
        "ArgMin",
        "Where",
        "Conv",
        "MatMul",
        "ConvInteger",
        "MatMulInteger",
        "QLinearConv",
        "QLinearMatMul",
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
        "MaxPool",
        "AveragePool",
        "GlobalAveragePool",
        "GlobalMaxPool",
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
        "RandomUniform",
        "RandomUniformLike",
        "RandomNormal",
        "RandomNormalLike",
        "Multinomial",
        "Unique",
        "NegativeLogLikelihoodLoss",
        "SoftmaxCrossEntropyLoss",
        "MaxRoiPool",
        "MaxUnpool",
        "RNN",
        "GRU",
        "LSTM",
        "If",
        "Loop",
        "Scan",
        "SequenceConstruct",
        "SequenceEmpty",
        "SequenceAt",
        "SequenceInsert",
        "SequenceErase",
        "SequenceLength",
        "SplitToSequence",
        "ConcatFromSequence",
        "StringNormalizer",
        "TfIdfVectorizer",
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
        "Expand",
        "Tile",
        "Resize",
        "Upsample",
        "SpaceToDepth",
        "DepthToSpace",
        "ReverseSequence",
        "Squeeze",
        "Unsqueeze",
        "OneHot",
        "ReduceProd",
        "ReduceL1",
        "ReduceL2",
        "ReduceSumSquare",
    }
    op_backends: list[dict[str, str]] = []
    backend_stats: dict[str, int] = {}
    fallback_stats: dict[str, int] = {}
    for node in model.nodes:
        if node.op_type not in quant_ops:
            for name in node.inputs + node.outputs:
                tensor = model.tensors.get(name)
                if tensor is None:
                    continue
                if tensor.dtype in ("uint8", "int8", "int16"):
                    raise ValueError(f"Quantized tensor used in unsupported op: {node.op_type}")
        handler = backend_impl.get_handler(node.op_type)
        if handler is None:
            raise ValueError(f"Unsupported operator: {node.op_type}")
        multi_output_ops = {
            "Split",
            "TopK",
            "DynamicQuantizeLinear",
            "Unique",
            "SoftmaxCrossEntropyLoss",
            "RNN",
            "GRU",
            "LSTM",
            "If",
            "Loop",
            "Scan",
        }
        if len(node.outputs) != 1 and node.op_type not in multi_output_ops:
            raise ValueError(f"Operator {node.op_type} with multiple outputs is not supported.")
        handler(ctx, node)
        op_entry: dict[str, str] = {"op": node.op_type, "backend": backend_impl.name}
        op_backends.append(op_entry)
        backend_stats[backend_impl.name] = backend_stats.get(backend_impl.name, 0) + 1
        invoke_lines.append("")
    if not model.nodes:
        invoke_lines.append("  (void)input_ptrs;")
        invoke_lines.append("  (void)output_ptrs;")
    invoke_lines.append("  return 0;")

    header_context = {
        "guard": guard,
        "num_inputs": len(input_names),
        "num_outputs": len(output_names),
        "input_sizes": input_sizes,
        "output_sizes": output_sizes,
        "input_total_size": input_total_size,
        "output_total_size": output_total_size,
        "num_buffers": len(buffer_names),
        "num_weights": len(weight_names) if weights_ram else 0,
        "arena_bytes": arena_bytes,
        "arena_words": arena_words,
    }
    write_template("tinyml/model.h.j2", header_context, header_path, encoding="utf-8")

    weight_copies = []
    if weights_ram and weight_offsets:
        for idx, name in enumerate(weight_names):
            weight_copies.append(
                {"index": idx, "const_name": consts[name], "size": weight_sizes[idx]}
            )

    source_context = {
        "header_name": header_name,
        "includes": includes,
        "const_decls": const_decls,
        "buffer_offsets": [str(v) for v in buffer_offsets],
        "weight_offsets": [str(v) for v in weight_offsets] if weights_ram else [],
        "input_shape_decls": input_shape_decls,
        "output_shape_decls": output_shape_decls,
        "input_descs": input_descs,
        "output_descs": output_descs,
        "weights_ram": weights_ram,
        "weight_copies": weight_copies,
        "invoke_body": "\n".join(invoke_lines),
        "num_inputs": len(input_names),
        "num_outputs": len(output_names),
        "input_total_size": input_total_size,
        "output_total_size": output_total_size,
        "arena_bytes": arena_bytes,
    }
    write_template("tinyml/model.c.j2", source_context, source_path, encoding="utf-8")

    return {
        "header": header_path,
        "source": source_path,
        "input_size": str(input_total_size),
        "output_size": str(output_total_size),
        "arena_bytes": str(arena_bytes),
        "op_backends": op_backends,
        "backend_stats": backend_stats,
        "fallback_stats": fallback_stats,
    }


def generate_manifest(
    model: ModelIR,
    output_dir: str,
    backend: str,
    weights: str,
    arena_bytes: int,
    op_backends: list[dict[str, str]],
    backend_stats: dict[str, int],
    fallback_stats: dict[str, int],
) -> str:
    ops = [node.op_type for node in model.nodes]
    manifest = {
        "name": model.name,
        "opset": model.opset,
        "backend": backend,
        "inputs": [t.name for t in model.inputs],
        "outputs": [t.name for t in model.outputs],
        "input_shapes": {t.name: t.shape for t in model.inputs},
        "output_shapes": {t.name: t.shape for t in model.outputs},
        "input_dtypes": {t.name: t.dtype for t in model.inputs},
        "output_dtypes": {t.name: t.dtype for t in model.outputs},
        "ops": ops,
        "arena_bytes": arena_bytes,
        "weights": weights,
        "op_backends": op_backends,
        "backend_stats": backend_stats,
        "fallback_stats": fallback_stats,
    }
    path = os.path.join(output_dir, "model.manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path
