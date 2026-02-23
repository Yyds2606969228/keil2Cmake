# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass

from ..ir import ModelIR
from .utils import get_shape


@dataclass
class EmitContext:
    lines: list[str]
    model: ModelIR
    input_ptrs: dict[str, str]
    output_ptrs: dict[str, str]
    buffers: dict[str, str]
    consts: dict[str, str]
    weights: dict[str, str]
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
        if name in self.input_ptrs:
            return f"(({ctype}*){self.input_ptrs[name]})"
        if name in self.output_ptrs:
            return f"(({ctype}*){self.output_ptrs[name]})"
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
