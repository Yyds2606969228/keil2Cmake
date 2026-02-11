# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass

from ..ir import ModelIR, NodeInfo
from .utils import get_shape


@dataclass
class EmitContext:
    lines: list[str]
    model: ModelIR
    input_name: str
    output_name: str
    buffers: dict[str, str]
    consts: dict[str, str]
    weights: dict[str, str]
    backend: str
    cmsis_weights_t: dict[str, str]
    cmsis_kernel_sums: dict[str, str]
    cmsis_biases: dict[int, str]
    backend_used: str | None = None
    fallback_reason: str | None = None
    symbol_index: int = 0

    def next_symbol(self, prefix: str) -> str:
        name = f"{prefix}_{self.symbol_index}"
        self.symbol_index += 1
        return name

    def map_ptr(self, name: str) -> str:
        dtype = self.dtype(name)
        ctype = "float"
        if dtype == "uint8":
            ctype = "uint8_t"
        elif dtype == "int8":
            ctype = "int8_t"
        elif dtype == "int16":
            ctype = "int16_t"
        elif dtype == "int32":
            ctype = "int32_t"
        elif dtype == "int64":
            ctype = "int64_t"
        elif dtype == "bool":
            ctype = "uint8_t"
        if name == self.input_name:
            return f"(({ctype}*)input)"
        if name == self.output_name:
            return f"(({ctype}*)output)"
        if name in self.weights:
            return f"(({ctype}*){self.weights[name]})"
        if name in self.consts:
            return f"(({ctype}*){self.consts[name]})"
        if name in self.buffers:
            return f"(({ctype}*){self.buffers[name]})"
        raise ValueError(f"Unknown tensor mapping for '{name}'.")

    def shape(self, name: str) -> list[int]:
        return get_shape(self.model, name)

    def dtype(self, name: str) -> str:
        if name not in self.model.tensors:
            raise ValueError(f"Missing tensor dtype for '{name}'.")
        return self.model.tensors[name].dtype

    def qparams(self, name: str) -> tuple[float, int]:
        if name not in self.model.tensors:
            raise ValueError(f"Missing tensor for '{name}'.")
        tensor = self.model.tensors[name]
        if tensor.qscale is None or tensor.qzero is None:
            raise ValueError(f"Missing quantization params for '{name}'.")
        return float(tensor.qscale), int(tensor.qzero)

    def qparams_optional(self, name: str) -> tuple[float, int] | None:
        if name not in self.model.tensors:
            return None
        tensor = self.model.tensors[name]
        if tensor.qscale is None or tensor.qzero is None:
            return None
        return float(tensor.qscale), int(tensor.qzero)

    def cmsis_weight_t(self, name: str) -> str | None:
        return self.cmsis_weights_t.get(name)

    def cmsis_kernel_sum(self, name: str) -> str | None:
        return self.cmsis_kernel_sums.get(name)

    def cmsis_bias(self, node: NodeInfo) -> str | None:
        return self.cmsis_biases.get(id(node))
