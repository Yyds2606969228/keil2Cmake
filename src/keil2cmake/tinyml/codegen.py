# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import re

from ..template_engine import write_template
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


def _unique_name(base: str, used: set[str]) -> str:
    name = base
    idx = 1
    while name in used:
        name = f"{base}_{idx}"
        idx += 1
    used.add(name)
    return name


def _build_cmsis_bias_values(
    model: ModelIR,
    a_name: str,
    b_name: str,
    bias_tensor,
    out_channels: int,
) -> list[int] | None:
    if bias_tensor is None or bias_tensor.data is None:
        return None
    if len(bias_tensor.data) < out_channels:
        return None
    if bias_tensor.dtype in ("int32", "int64") and bias_tensor.qscale is None:
        return [int(bias_tensor.data[i]) for i in range(out_channels)]
    a_tensor = model.tensors.get(a_name)
    b_tensor = model.tensors.get(b_name)
    if a_tensor is None or b_tensor is None:
        return None
    if a_tensor.qscale is None or b_tensor.qscale is None:
        return None
    bias_scale = float(a_tensor.qscale) * float(b_tensor.qscale)
    if bias_scale == 0.0:
        return None
    if bias_tensor.dtype == "float32":
        return [
            int(round(float(bias_tensor.data[i]) / bias_scale))
            for i in range(out_channels)
        ]
    if bias_tensor.dtype in ("int8", "int16"):
        if bias_tensor.qscale is None or bias_tensor.qzero is None:
            return None
        sc = float(bias_tensor.qscale)
        zc = int(bias_tensor.qzero)
        return [
            int(round((float(bias_tensor.data[i]) - zc) * sc / bias_scale))
            for i in range(out_channels)
        ]
    if bias_tensor.dtype in ("int32", "int64"):
        sc = bias_tensor.qscale
        zc = bias_tensor.qzero
        if sc is None:
            return [int(bias_tensor.data[i]) for i in range(out_channels)]
        zc_val = int(zc) if zc is not None else 0
        sc_val = float(sc)
        return [
            int(round((float(bias_tensor.data[i]) - zc_val) * sc_val / bias_scale))
            for i in range(out_channels)
        ]
    return None


def _build_cmsis_nn_tables(
    model: ModelIR,
    backend: str,
    used_names: set[str],
) -> tuple[list[tuple[str, str, list[int]]], dict[str, str], dict[str, str], dict[int, str]]:
    extra_consts: list[tuple[str, str, list[int]]] = []
    weights_t: dict[str, str] = {}
    kernel_sums: dict[str, str] = {}
    bias_map: dict[int, str] = {}
    zero_bias_cache: dict[int, str] = {}
    if backend != "cmsis-nn":
        return extra_consts, weights_t, kernel_sums, bias_map
    for node in model.nodes:
        if node.op_type not in ("MatMul", "Gemm"):
            continue
        if len(node.inputs) < 2 or not node.outputs:
            continue
        a_name = node.inputs[0]
        b_name = node.inputs[1]
        b_tensor = model.tensors.get(b_name)
        if b_tensor is None or b_tensor.dtype != "int8":
            continue
        b_shape = get_shape(model, b_name)
        if len(b_shape) != 2 or any(dim <= 0 for dim in b_shape):
            continue
        k_dim, n_dim = b_shape
        if b_tensor.data is not None and b_name not in weights_t:
            t_name = _unique_name(f"const_{_sanitize(b_name)}_t", used_names)
            t_data: list[int] = []
            for col in range(n_dim):
                for row in range(k_dim):
                    t_data.append(int(b_tensor.data[row * n_dim + col]))
            weights_t[b_name] = t_name
            extra_consts.append(("int8_t", t_name, t_data))
            ks_name = _unique_name(f"const_{_sanitize(b_name)}_ks", used_names)
            kernel_sums[b_name] = ks_name
            ks_data: list[int] = []
            for row in range(n_dim):
                acc = 0
                base = row * k_dim
                for col in range(k_dim):
                    acc += int(t_data[base + col])
                ks_data.append(int(acc))
            extra_consts.append(("int32_t", ks_name, ks_data))
        bias_name = None
        if node.op_type == "Gemm":
            c_name = node.inputs[2] if len(node.inputs) >= 3 else None
            if c_name:
                c_tensor = model.tensors.get(c_name)
                if c_tensor is None or c_tensor.data is None:
                    continue
                c_shape = get_shape(model, c_name)
                if tensor_size(c_shape) != n_dim:
                    continue
                bias_vals = _build_cmsis_bias_values(
                    model, a_name, b_name, c_tensor, n_dim
                )
                if bias_vals is None:
                    continue
                bias_name = _unique_name(f"const_{_sanitize(c_name)}_bias", used_names)
                extra_consts.append(("int32_t", bias_name, bias_vals))
            else:
                bias_name = zero_bias_cache.get(n_dim)
                if bias_name is None:
                    bias_name = _unique_name(f"const_k2c_zero_bias_{n_dim}", used_names)
                    extra_consts.append(("int32_t", bias_name, [0] * n_dim))
                    zero_bias_cache[n_dim] = bias_name
        else:
            bias_name = zero_bias_cache.get(n_dim)
            if bias_name is None:
                bias_name = _unique_name(f"const_k2c_zero_bias_{n_dim}", used_names)
                extra_consts.append(("int32_t", bias_name, [0] * n_dim))
                zero_bias_cache[n_dim] = bias_name
        if bias_name:
            bias_map[id(node)] = bias_name
    return extra_consts, weights_t, kernel_sums, bias_map


def _validate_single_io(model: ModelIR) -> tuple[str, str]:
    if len(model.inputs) != 1 or len(model.outputs) != 1:
        raise ValueError("Only single-input and single-output models are supported.")
    return model.inputs[0].name, model.outputs[0].name


def _align_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


def generate_c_code(
    model: ModelIR,
    output_dir: str,
    model_name: str,
    weights: str,
    quant: str,
    backend: str,
) -> dict[str, str]:
    input_name, output_name = _validate_single_io(model)

    os.makedirs(output_dir, exist_ok=True)
    backend_impl = get_backend(backend)

    consts: dict[str, str] = {}
    buffers: dict[str, str] = {}
    weights_map: dict[str, str] = {}
    weights_ram = weights == "ram"

    weight_names: list[str] = []
    for name, tensor in model.tensors.items():
        if tensor.data is not None:
            weight_names.append(name)
            consts[name] = f"const_{_sanitize(name)}"

    used_names = set(consts.values())
    cmsis_extra_consts, cmsis_weights_t, cmsis_kernel_sums, cmsis_biases = _build_cmsis_nn_tables(
        model, backend, used_names
    )

    buffer_names: list[str] = []
    for node in model.nodes:
        for out_name in node.outputs:
            if out_name == output_name:
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

    input_shape = get_shape(model, input_name)
    output_shape = get_shape(model, output_name)
    input_dtype = model.tensors[input_name].dtype
    output_dtype = model.tensors[output_name].dtype
    input_size = tensor_size(input_shape)
    output_size = tensor_size(output_shape)

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
    for ctype, name, data in cmsis_extra_consts:
        data_str = ", ".join(str(int(v)) for v in data)
        const_decls.append({"ctype": ctype, "name": name, "size": len(data), "values": data_str})

    input_shape_vals = ", ".join(str(v) for v in input_shape)
    output_shape_vals = ", ".join(str(v) for v in output_shape)
    input_shape_ptr = "k2c_input_shape"
    output_shape_ptr = "k2c_output_shape"
    if len(input_shape) == 0:
        input_shape_decl = "static const int k2c_input_shape[1] = { 1 };"
        input_shape_ptr = "NULL"
    else:
        input_shape_decl = f"static const int k2c_input_shape[{len(input_shape)}] = {{ {input_shape_vals} }};"
    if len(output_shape) == 0:
        output_shape_decl = "static const int k2c_output_shape[1] = { 1 };"
        output_shape_ptr = "NULL"
    else:
        output_shape_decl = f"static const int k2c_output_shape[{len(output_shape)}] = {{ {output_shape_vals} }};"

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

    includes = [
        "#include <math.h>",
        "#include <string.h>",
        "#include <stdint.h>",
    ] + backend_impl.extra_includes

    invoke_lines: list[str] = []
    invoke_lines.append("  if (!ctx || !input_ptr || !output_ptr) return -1;")
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

    input_ctype = _io_ctype(input_dtype)
    output_ctype = _io_ctype(output_dtype)
    invoke_lines.append(f"  const {input_ctype}* input = (const {input_ctype}*)input_ptr;")
    invoke_lines.append(f"  {output_ctype}* output = ({output_ctype}*)output_ptr;")

    ctx = EmitContext(
        lines=invoke_lines,
        model=model,
        input_name=input_name,
        output_name=output_name,
        buffers=buffers,
        consts=consts,
        weights=weights_map,
        backend=backend_impl.name,
        cmsis_weights_t=cmsis_weights_t,
        cmsis_kernel_sums=cmsis_kernel_sums,
        cmsis_biases=cmsis_biases,
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
    cmsis_handler_get = None
    if backend_impl.name == "cmsis-nn":
        from .backends.cmsis_nn.ops import get_handler as cmsis_handler_get
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
        if len(node.outputs) != 1 and node.op_type not in ("Split", "TopK", "DynamicQuantizeLinear"):
            raise ValueError(f"Operator {node.op_type} with multiple outputs is not supported.")
        has_cmsis_native = bool(cmsis_handler_get and cmsis_handler_get(node.op_type))
        ctx.backend_used = backend_impl.name
        ctx.fallback_reason = None
        handler(ctx, node)
        used_backend = ctx.backend_used or backend_impl.name
        if backend_impl.name == "cmsis-nn" and not has_cmsis_native:
            used_backend = "c"
        fallback_reason = None
        if backend_impl.name == "cmsis-nn" and used_backend == "c":
            if ctx.fallback_reason:
                fallback_reason = ctx.fallback_reason
            else:
                if has_cmsis_native:
                    fallback_reason = "cmsis-nn constraints not satisfied, fallback to c"
                else:
                    fallback_reason = "cmsis-nn op not implemented, fallback to c"
        op_entry: dict[str, str] = {"op": node.op_type, "backend": used_backend}
        if fallback_reason:
            op_entry["fallback_reason"] = fallback_reason
            fallback_stats[fallback_reason] = fallback_stats.get(fallback_reason, 0) + 1
        op_backends.append(op_entry)
        backend_stats[used_backend] = backend_stats.get(used_backend, 0) + 1
        ctx.backend_used = None
        ctx.fallback_reason = None
        invoke_lines.append("")
    if not model.nodes:
        invoke_lines.append("  (void)input;")
        invoke_lines.append("  (void)output;")
    invoke_lines.append("  return 0;")

    header_context = {
        "guard": guard,
        "input_size": input_size,
        "output_size": output_size,
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
        "input_shape_decl": input_shape_decl,
        "output_shape_decl": output_shape_decl,
        "input_name": input_name,
        "output_name": output_name,
        "input_shape_ptr": input_shape_ptr,
        "output_shape_ptr": output_shape_ptr,
        "input_rank": len(input_shape),
        "output_rank": len(output_shape),
        "input_elem_size": _ctype_size(input_dtype),
        "output_elem_size": _ctype_size(output_dtype),
        "weights_ram": weights_ram,
        "weight_copies": weight_copies,
        "invoke_body": "\n".join(invoke_lines),
        "input_size": input_size,
        "output_size": output_size,
        "arena_bytes": arena_bytes,
    }
    write_template("tinyml/model.c.j2", source_context, source_path, encoding="utf-8")

    return {
        "header": header_path,
        "source": source_path,
        "input_size": str(input_size),
        "output_size": str(output_size),
        "arena_bytes": str(arena_bytes),
        "op_backends": op_backends,
        "backend_stats": backend_stats,
        "fallback_stats": fallback_stats,
    }


def generate_manifest(
    model: ModelIR,
    output_dir: str,
    backend: str,
    quant: str,
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
        "quant": quant,
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
