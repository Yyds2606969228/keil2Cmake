# -*- coding: utf-8 -*-

from __future__ import annotations

from onnx import TensorProto, numpy_helper

from ....ir import NodeInfo
from ....operators.context import EmitContext
from ....operators.utils import tensor_size
from .registry import register_op


def _dtype_from_tensorproto(dtype: int) -> str:
    if dtype in (TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE):
        return "float32"
    if dtype == TensorProto.BOOL:
        return "bool"
    if dtype == TensorProto.INT8:
        return "int8"
    if dtype == TensorProto.INT16:
        return "int16"
    if dtype == TensorProto.INT32:
        return "int32"
    if dtype == TensorProto.INT64:
        return "int64"
    raise ValueError("Constant value dtype is unsupported.")


def _parse_constant(node: NodeInfo) -> tuple[str, list[int], list[float | int]]:
    if "value" in node.attrs:
        value = node.attrs["value"]
        dtype = _dtype_from_tensorproto(int(value.data_type))
        arr = numpy_helper.to_array(value)
        shape = list(arr.shape)
        flat = arr.reshape(-1)
        if dtype == "float32":
            data = [float(v) for v in flat.astype("float32").tolist()]
        elif dtype == "bool":
            data = [1 if bool(v) else 0 for v in flat.tolist()]
        else:
            data = [int(v) for v in flat.tolist()]
        return dtype, shape, data

    if "value_float" in node.attrs:
        return "float32", [], [float(node.attrs["value_float"])]
    if "value_int" in node.attrs:
        return "int64", [], [int(node.attrs["value_int"])]
    if "value_floats" in node.attrs:
        vals = [float(v) for v in node.attrs["value_floats"]]
        return "float32", [len(vals)], vals
    if "value_ints" in node.attrs:
        vals = [int(v) for v in node.attrs["value_ints"]]
        return "int64", [len(vals)], vals

    raise ValueError("Constant requires value/value_float/value_int/value_floats/value_ints.")


def _ctype(dtype: str) -> str:
    if dtype == "float32":
        return "float"
    if dtype == "bool":
        return "uint8_t"
    if dtype == "int8":
        return "int8_t"
    if dtype == "int16":
        return "int16_t"
    if dtype == "int32":
        return "int32_t"
    if dtype == "int64":
        return "int64_t"
    raise ValueError("Constant output dtype is unsupported.")


def _literal(dtype: str, value: float | int) -> str:
    if dtype == "float32":
        return f"{float(value):.8f}f"
    if dtype == "bool":
        return "1u" if int(value) != 0 else "0u"
    return str(int(value))


@register_op("Constant")
def emit_constant(ctx: EmitContext, node: NodeInfo) -> None:
    if node.inputs:
        raise ValueError("Constant expects 0 input.")
    out_name = node.outputs[0]
    out_dtype = ctx.dtype(out_name)
    out_shape = ctx.shape(out_name)
    size = tensor_size(out_shape)

    const_dtype, const_shape, const_data = _parse_constant(node)
    if out_dtype != const_dtype:
        raise ValueError("Constant output dtype mismatch.")
    if out_shape != const_shape:
        raise ValueError("Constant output shape mismatch.")
    if len(const_data) != size:
        raise ValueError("Constant value size mismatch.")

    out = ctx.map_ptr(out_name)
    ctype = _ctype(out_dtype)
    if size == 1:
        ctx.lines.append(f"  {out}[0] = ({ctype}){_literal(out_dtype, const_data[0])};")
        return

    sym = ctx.next_symbol("k2c_const_data")
    values = ", ".join(_literal(out_dtype, v) for v in const_data)
    ctx.lines.append(f"  static const {ctype} {sym}[{size}] = {{ {values} }};")
    ctx.lines.append(f"  for (size_t i = 0; i < {size}; ++i) {{")
    ctx.lines.append(f"    {out}[i] = {sym}[i];")
    ctx.lines.append("  }")
