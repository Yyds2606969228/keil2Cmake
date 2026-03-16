# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import onnx
from onnx import numpy_helper

from .ir import NodeInfo, TensorInfo
from .onnx_loader_helpers import _dtype_from_tensorproto, _shape_from_value, _tensor_dtype


def _shape_known(shape: Iterable[int]) -> bool:
    return all(int(dim) > 0 for dim in shape)


def _tensor_info_from_proto(name: str, tensor_proto: onnx.TensorProto) -> TensorInfo | None:
    dtype = _dtype_from_tensorproto(int(tensor_proto.data_type))
    if dtype not in ("float32", "bool", "uint8", "int8", "int16", "int32", "int64"):
        return None
    try:
        arr = numpy_helper.to_array(tensor_proto)
    except (TypeError, ValueError, RuntimeError):
        return None
    shape = list(arr.shape)
    if dtype == "float32":
        data = arr.flatten().astype("float32").tolist()
    elif dtype == "bool":
        data = arr.flatten().astype("int64").tolist()
    else:
        data = arr.flatten().astype("int64").tolist()
    return TensorInfo(name=name, shape=shape, dtype=dtype, data=data)


def constant_tensor_from_attrs(name: str, attrs: dict[str, object]) -> TensorInfo | None:
    if "value" in attrs:
        value = attrs["value"]
        if isinstance(value, onnx.TensorProto):
            return _tensor_info_from_proto(name, value)
        return None
    if "value_float" in attrs:
        return TensorInfo(name=name, shape=[], dtype="float32", data=[float(attrs["value_float"])])
    if "value_int" in attrs:
        return TensorInfo(name=name, shape=[], dtype="int64", data=[int(attrs["value_int"])])
    if "value_floats" in attrs:
        vals = attrs["value_floats"]
        data = [float(v) for v in vals]
        return TensorInfo(name=name, shape=[len(data)], dtype="float32", data=data)
    if "value_ints" in attrs:
        vals = attrs["value_ints"]
        data = [int(v) for v in vals]
        return TensorInfo(name=name, shape=[len(data)], dtype="int64", data=data)
    return None


@dataclass
class _NameGenerator:
    used: set[str]
    index: int = 0

    def new(self, prefix: str) -> str:
        safe = prefix.replace(":", "_").replace("/", "_")
        while True:
            name = f"{safe}_{self.index}"
            self.index += 1
            if name not in self.used:
                self.used.add(name)
                return name


class _Lowerer:
    def __init__(self, tensors: dict[str, TensorInfo], nodes: list[NodeInfo]):
        used = set(tensors.keys())
        for node in nodes:
            for out in node.outputs:
                if out:
                    used.add(out)
        self._name_gen = _NameGenerator(used=used)
        self._tensors = tensors

    def lower_nodes(self, nodes: list[NodeInfo]) -> list[NodeInfo]:
        lowered: list[NodeInfo] = []
        sequence_values: dict[str, list[str]] = {}
        for node in nodes:
            op = node.op_type
            if op == "If":
                self._lower_if(node, lowered)
                continue
            if op == "Loop":
                raise ValueError(
                    "Loop requires lowering before C codegen. "
                    "Current converter supports direct lowering for If/Sequence family only."
                )
            if op == "Scan":
                raise ValueError(
                    "Scan requires lowering before C codegen. "
                    "Current converter supports direct lowering for If/Sequence family only."
                )
            if op in (
                "SequenceConstruct",
                "SequenceEmpty",
                "SequenceInsert",
                "SequenceErase",
                "SequenceLength",
                "SequenceAt",
                "SplitToSequence",
                "ConcatFromSequence",
            ):
                self._lower_sequence(node, sequence_values, lowered)
                continue

            for in_name in node.inputs:
                if in_name and in_name in sequence_values:
                    raise ValueError(
                        f"{op} received sequence-typed input '{in_name}' without valid sequence lowering."
                    )

            lowered.append(node)
            if op == "Constant" and node.outputs:
                out_name = node.outputs[0]
                if out_name:
                    const_tensor = constant_tensor_from_attrs(out_name, node.attrs)
                    if const_tensor is not None:
                        self._tensors[out_name] = const_tensor

            for out_name in node.outputs:
                if out_name in sequence_values:
                    del sequence_values[out_name]
        return lowered

    def _const_scalar_int(self, name: str) -> int | None:
        tensor = self._tensors.get(name)
        if tensor is None or tensor.data is None or len(tensor.data) == 0:
            return None
        return int(tensor.data[0])

    def _const_ints(self, name: str) -> list[int] | None:
        tensor = self._tensors.get(name)
        if tensor is None or tensor.data is None:
            return None
        if tensor.dtype not in ("float32", "int8", "int16", "int32", "int64"):
            return None
        return [int(v) for v in tensor.data]

    def _clone_tensor_meta(self, src_name: str, dst_name: str) -> None:
        src = self._tensors.get(src_name)
        if src is None:
            return
        self._tensors[dst_name] = TensorInfo(
            name=dst_name,
            shape=list(src.shape),
            dtype=src.dtype,
            data=None,
            qscale=src.qscale,
            qzero=src.qzero,
        )

    @staticmethod
    def _normalize_axis(axis: int, rank: int, *, allow_end: bool = False) -> int:
        adjust = rank + 1 if allow_end else rank
        if axis < 0:
            axis += adjust
        upper = rank if allow_end else (rank - 1)
        if axis < 0 or axis > upper:
            raise ValueError("Axis out of range.")
        return axis

    @staticmethod
    def _normalize_index(index: int, length: int, *, for_insert: bool = False) -> int:
        idx = index
        if idx < 0:
            idx += length if not for_insert else (length + 1)
        if for_insert:
            if idx < 0:
                idx = 0
            if idx > length:
                idx = length
            return idx
        if idx < 0 or idx >= length:
            raise ValueError("Sequence index out of range.")
        return idx

    def _lower_if(self, node: NodeInfo, lowered: list[NodeInfo]) -> None:
        if len(node.inputs) < 1:
            raise ValueError("If expects 1 input (condition).")
        cond_name = node.inputs[0]
        then_graph = node.attrs.get("then_branch")
        else_graph = node.attrs.get("else_branch")
        if not isinstance(then_graph, onnx.GraphProto) or not isinstance(else_graph, onnx.GraphProto):
            raise ValueError("If expects then_branch/else_branch graph attributes.")

        then_prefix = self._name_gen.new("if_then")
        else_prefix = self._name_gen.new("if_else")

        then_nodes_raw, then_outs = self._inline_graph(then_graph, then_prefix)
        else_nodes_raw, else_outs = self._inline_graph(else_graph, else_prefix)
        then_nodes = self.lower_nodes(then_nodes_raw)
        else_nodes = self.lower_nodes(else_nodes_raw)

        lowered.extend(then_nodes)
        lowered.extend(else_nodes)

        if len(node.outputs) != len(then_outs) or len(node.outputs) != len(else_outs):
            raise ValueError("If output count mismatch between parent node and branch graphs.")

        for out_name, then_name, else_name in zip(node.outputs, then_outs, else_outs):
            if not out_name:
                continue
            then_tensor = self._tensors.get(then_name)
            else_tensor = self._tensors.get(else_name)
            if then_tensor is None or else_tensor is None:
                raise ValueError("If branch output tensor metadata is missing.")
            if then_tensor.dtype != else_tensor.dtype:
                raise ValueError("If requires then/else outputs with matching dtypes.")
            if _shape_known(then_tensor.shape) and _shape_known(else_tensor.shape):
                if [int(v) for v in then_tensor.shape] != [int(v) for v in else_tensor.shape]:
                    raise ValueError("If requires then/else outputs with matching shapes.")
            out_tensor = self._tensors.get(out_name)
            if out_tensor is None:
                self._tensors[out_name] = TensorInfo(
                    name=out_name,
                    shape=list(then_tensor.shape),
                    dtype=then_tensor.dtype,
                    data=None,
                    qscale=then_tensor.qscale,
                    qzero=then_tensor.qzero,
                )
            lowered.append(NodeInfo(op_type="Where", inputs=[cond_name, then_name, else_name], outputs=[out_name], attrs={}))

    def _register_value_info(self, onnx_name: str, mapped_name: str, value_info: onnx.ValueInfoProto) -> None:
        try:
            dtype = _dtype_from_tensorproto(_tensor_dtype(value_info))
            shape = _shape_from_value(value_info)
        except (AttributeError, TypeError, ValueError):
            return
        if dtype == "unknown":
            return
        existing = self._tensors.get(mapped_name)
        if existing is None:
            self._tensors[mapped_name] = TensorInfo(name=mapped_name, shape=shape, dtype=dtype)
            return
        if existing.data is not None:
            return
        if _shape_known(existing.shape):
            return
        self._tensors[mapped_name] = TensorInfo(
            name=existing.name,
            shape=shape,
            dtype=existing.dtype if existing.dtype != "unknown" else dtype,
            data=existing.data,
            qscale=existing.qscale,
            qzero=existing.qzero,
        )

    def _inline_graph(self, graph: onnx.GraphProto, prefix: str) -> tuple[list[NodeInfo], list[str]]:
        name_map: dict[str, str] = {}
        inlined_nodes: list[NodeInfo] = []

        for init in graph.initializer:
            init_name = init.name
            if not init_name:
                continue
            mapped_name = self._name_gen.new(f"{prefix}_{init_name}")
            name_map[init_name] = mapped_name
            tensor = _tensor_info_from_proto(mapped_name, init)
            if tensor is not None:
                self._tensors[mapped_name] = tensor

        for value_info in list(graph.input) + list(graph.value_info) + list(graph.output):
            raw_name = value_info.name
            if not raw_name:
                continue
            mapped_name = name_map.get(raw_name, raw_name)
            self._register_value_info(raw_name, mapped_name, value_info)
            if raw_name not in name_map and mapped_name in self._tensors:
                name_map[raw_name] = mapped_name

        for sub_node in graph.node:
            attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in sub_node.attribute}
            in_names = [name_map.get(name, name) if name else "" for name in sub_node.input]
            out_names: list[str] = []
            for out_name in sub_node.output:
                if not out_name:
                    out_names.append("")
                    continue
                mapped_name = self._name_gen.new(f"{prefix}_{out_name}")
                name_map[out_name] = mapped_name
                out_names.append(mapped_name)
            node_info = NodeInfo(op_type=sub_node.op_type, inputs=in_names, outputs=out_names, attrs=attrs)
            inlined_nodes.append(node_info)
            if node_info.op_type == "Constant" and out_names and out_names[0]:
                const_tensor = constant_tensor_from_attrs(out_names[0], attrs)
                if const_tensor is not None:
                    self._tensors[out_names[0]] = const_tensor

        output_names: list[str] = []
        for out in graph.output:
            mapped = name_map.get(out.name, out.name)
            self._register_value_info(out.name, mapped, out)
            output_names.append(mapped)
        return inlined_nodes, output_names

    def _lower_sequence(
        self,
        node: NodeInfo,
        sequence_values: dict[str, list[str]],
        lowered: list[NodeInfo],
    ) -> None:
        op = node.op_type
        if op == "SequenceConstruct":
            if len(node.outputs) != 1:
                raise ValueError("SequenceConstruct expects 1 output.")
            sequence_values[node.outputs[0]] = [name for name in node.inputs if name]
            return
        if op == "SequenceEmpty":
            if len(node.outputs) != 1:
                raise ValueError("SequenceEmpty expects 1 output.")
            sequence_values[node.outputs[0]] = []
            return
        if op == "SequenceInsert":
            if len(node.inputs) < 2 or len(node.outputs) != 1:
                raise ValueError("SequenceInsert expects inputs [sequence, tensor, optional position].")
            seq_name = node.inputs[0]
            if seq_name not in sequence_values:
                raise ValueError("SequenceInsert input is not a lowered sequence value.")
            elems = list(sequence_values[seq_name])
            insert_tensor = node.inputs[1]
            if not insert_tensor:
                raise ValueError("SequenceInsert tensor input is empty.")
            index = len(elems)
            if len(node.inputs) >= 3 and node.inputs[2]:
                idx = self._const_scalar_int(node.inputs[2])
                if idx is None:
                    raise ValueError("SequenceInsert position must be a constant scalar.")
                index = idx
            index = self._normalize_index(index, len(elems), for_insert=True)
            elems.insert(index, insert_tensor)
            sequence_values[node.outputs[0]] = elems
            return
        if op == "SequenceErase":
            if len(node.inputs) < 1 or len(node.outputs) != 1:
                raise ValueError("SequenceErase expects inputs [sequence, optional position].")
            seq_name = node.inputs[0]
            if seq_name not in sequence_values:
                raise ValueError("SequenceErase input is not a lowered sequence value.")
            elems = list(sequence_values[seq_name])
            if not elems:
                raise ValueError("SequenceErase cannot erase from an empty sequence.")
            index = len(elems) - 1
            if len(node.inputs) >= 2 and node.inputs[1]:
                idx = self._const_scalar_int(node.inputs[1])
                if idx is None:
                    raise ValueError("SequenceErase position must be a constant scalar.")
                index = idx
            index = self._normalize_index(index, len(elems), for_insert=False)
            del elems[index]
            sequence_values[node.outputs[0]] = elems
            return
        if op == "SequenceLength":
            if len(node.inputs) != 1 or len(node.outputs) != 1:
                raise ValueError("SequenceLength expects 1 input and 1 output.")
            seq_name = node.inputs[0]
            if seq_name not in sequence_values:
                raise ValueError("SequenceLength input is not a lowered sequence value.")
            out_name = node.outputs[0]
            length_val = len(sequence_values[seq_name])
            out_dtype = "int64"
            out_existing = self._tensors.get(out_name)
            if out_existing is not None and out_existing.dtype in ("int64", "int32", "float32"):
                out_dtype = out_existing.dtype
            if out_dtype == "float32":
                data = [float(length_val)]
            else:
                data = [int(length_val)]
            self._tensors[out_name] = TensorInfo(name=out_name, shape=[], dtype=out_dtype, data=data)
            return
        if op == "SequenceAt":
            if len(node.inputs) != 2 or len(node.outputs) != 1:
                raise ValueError("SequenceAt expects 2 inputs and 1 output.")
            seq_name = node.inputs[0]
            if seq_name not in sequence_values:
                raise ValueError("SequenceAt input is not a lowered sequence value.")
            elems = sequence_values[seq_name]
            if not elems:
                raise ValueError("SequenceAt cannot index into an empty sequence.")
            idx = self._const_scalar_int(node.inputs[1])
            if idx is None:
                raise ValueError("SequenceAt index must be a constant scalar.")
            index = self._normalize_index(idx, len(elems), for_insert=False)
            src_name = elems[index]
            out_name = node.outputs[0]
            self._clone_tensor_meta(src_name, out_name)
            lowered.append(NodeInfo(op_type="Identity", inputs=[src_name], outputs=[out_name], attrs={}))
            return
        if op == "SplitToSequence":
            if len(node.inputs) < 1 or len(node.outputs) != 1:
                raise ValueError("SplitToSequence expects data input and one sequence output.")
            data_name = node.inputs[0]
            data_tensor = self._tensors.get(data_name)
            if data_tensor is None:
                raise ValueError("SplitToSequence data tensor metadata is missing.")
            in_shape = [int(v) for v in data_tensor.shape]
            if not _shape_known(in_shape):
                raise ValueError("SplitToSequence requires statically known input shape.")
            rank = len(in_shape)
            axis = self._normalize_axis(int(node.attrs.get("axis", 0)), rank)
            axis_dim = int(in_shape[axis])
            keepdims = int(node.attrs.get("keepdims", 1))
            if keepdims not in (0, 1):
                raise ValueError("SplitToSequence keepdims must be 0 or 1.")

            split_vals: list[int] | None = None
            if len(node.inputs) >= 2 and node.inputs[1]:
                split_vals = self._const_ints(node.inputs[1])
            if split_vals is None:
                split_attr = node.attrs.get("split")
                if isinstance(split_attr, (list, tuple)):
                    split_vals = [int(v) for v in split_attr]
                elif split_attr is not None:
                    split_vals = [int(split_attr)]
            if split_vals is None:
                split_vals = [1 for _ in range(axis_dim)]
            if len(split_vals) == 1 and split_vals[0] > 0 and split_vals[0] < axis_dim and axis_dim % split_vals[0] == 0:
                split_vals = [int(split_vals[0]) for _ in range(axis_dim // int(split_vals[0]))]
            if any(int(v) <= 0 for v in split_vals):
                raise ValueError("SplitToSequence split sizes must be positive.")
            if sum(int(v) for v in split_vals) != axis_dim:
                raise ValueError("SplitToSequence split sizes must sum to axis dimension.")

            split_outs: list[str] = []
            for idx in range(len(split_vals)):
                split_outs.append(self._name_gen.new(f"{node.outputs[0]}_split_{idx}"))
            lowered.append(NodeInfo(op_type="Split", inputs=[data_name], outputs=split_outs, attrs={"axis": axis}))

            seq_elems: list[str] = []
            for split_out, split_dim in zip(split_outs, split_vals):
                out_shape = list(in_shape)
                out_shape[axis] = int(split_dim)
                self._tensors[split_out] = TensorInfo(
                    name=split_out,
                    shape=out_shape,
                    dtype=data_tensor.dtype,
                    data=None,
                    qscale=data_tensor.qscale,
                    qzero=data_tensor.qzero,
                )
                if keepdims == 1:
                    seq_elems.append(split_out)
                    continue
                squeezed_name = self._name_gen.new(f"{split_out}_sq")
                squeezed_shape = list(out_shape[:axis] + out_shape[axis + 1 :])
                self._tensors[squeezed_name] = TensorInfo(
                    name=squeezed_name,
                    shape=squeezed_shape,
                    dtype=data_tensor.dtype,
                    data=None,
                    qscale=data_tensor.qscale,
                    qzero=data_tensor.qzero,
                )
                lowered.append(
                    NodeInfo(
                        op_type="Squeeze",
                        inputs=[split_out],
                        outputs=[squeezed_name],
                        attrs={"axes": [axis]},
                    )
                )
                seq_elems.append(squeezed_name)
            sequence_values[node.outputs[0]] = seq_elems
            return
        if op == "ConcatFromSequence":
            if len(node.inputs) != 1 or len(node.outputs) != 1:
                raise ValueError("ConcatFromSequence expects 1 sequence input and 1 tensor output.")
            seq_name = node.inputs[0]
            if seq_name not in sequence_values:
                raise ValueError("ConcatFromSequence input is not a lowered sequence value.")
            elems = list(sequence_values[seq_name])
            if not elems:
                raise ValueError("ConcatFromSequence cannot concatenate an empty sequence.")

            out_name = node.outputs[0]
            axis = int(node.attrs.get("axis", 0))
            new_axis = int(node.attrs.get("new_axis", 0))
            if new_axis not in (0, 1):
                raise ValueError("ConcatFromSequence new_axis must be 0 or 1.")

            first = self._tensors.get(elems[0])
            if first is None:
                raise ValueError("ConcatFromSequence input tensor metadata is missing.")
            in_shapes = []
            for elem in elems:
                tensor = self._tensors.get(elem)
                if tensor is None:
                    raise ValueError("ConcatFromSequence input tensor metadata is missing.")
                if tensor.dtype != first.dtype:
                    raise ValueError("ConcatFromSequence requires all sequence elements to have same dtype.")
                in_shapes.append([int(v) for v in tensor.shape])
            rank = len(in_shapes[0])
            if not _shape_known(in_shapes[0]):
                raise ValueError("ConcatFromSequence requires statically known input shapes.")

            concat_inputs = elems
            concat_axis = self._normalize_axis(axis, rank if new_axis == 0 else rank, allow_end=(new_axis == 1))
            out_shape: list[int]
            if new_axis == 0:
                for shape in in_shapes:
                    if len(shape) != rank:
                        raise ValueError("ConcatFromSequence element rank mismatch.")
                out_shape = list(in_shapes[0])
                total_dim = 0
                for shape in in_shapes:
                    for idx, dim in enumerate(shape):
                        if idx == concat_axis:
                            continue
                        if int(dim) != int(out_shape[idx]):
                            raise ValueError("ConcatFromSequence element shape mismatch.")
                    total_dim += int(shape[concat_axis])
                out_shape[concat_axis] = total_dim
                lowered.append(NodeInfo(op_type="Concat", inputs=concat_inputs, outputs=[out_name], attrs={"axis": concat_axis}))
            else:
                unsqueezed_inputs: list[str] = []
                out_rank = rank + 1
                concat_axis = self._normalize_axis(axis, out_rank, allow_end=True)
                for elem_name, shape in zip(elems, in_shapes):
                    unsq_name = self._name_gen.new(f"{elem_name}_unsq")
                    unsq_shape = list(shape)
                    unsq_shape.insert(concat_axis, 1)
                    tensor = self._tensors[elem_name]
                    self._tensors[unsq_name] = TensorInfo(
                        name=unsq_name,
                        shape=unsq_shape,
                        dtype=tensor.dtype,
                        data=None,
                        qscale=tensor.qscale,
                        qzero=tensor.qzero,
                    )
                    lowered.append(
                        NodeInfo(
                            op_type="Unsqueeze",
                            inputs=[elem_name],
                            outputs=[unsq_name],
                            attrs={"axes": [concat_axis]},
                        )
                    )
                    unsqueezed_inputs.append(unsq_name)
                concat_inputs = unsqueezed_inputs
                out_shape = list(self._tensors[concat_inputs[0]].shape)
                out_shape[concat_axis] = len(concat_inputs)
                lowered.append(NodeInfo(op_type="Concat", inputs=concat_inputs, outputs=[out_name], attrs={"axis": concat_axis}))

            out_tensor = self._tensors.get(out_name)
            if out_tensor is None:
                self._tensors[out_name] = TensorInfo(
                    name=out_name,
                    shape=out_shape,
                    dtype=first.dtype,
                    data=None,
                    qscale=first.qscale,
                    qzero=first.qzero,
                )
            return
        raise ValueError(f"Unexpected sequence op during lowering: {op}")


def lower_placeholder_ops(nodes: list[NodeInfo], tensors: dict[str, TensorInfo]) -> list[NodeInfo]:
    lowerer = _Lowerer(tensors=tensors, nodes=nodes)
    return lowerer.lower_nodes(nodes)
