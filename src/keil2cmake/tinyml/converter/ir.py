# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: list[int]
    dtype: str
    data: list[float] | None = None
    qscale: float | None = None
    qzero: int | None = None


@dataclass(frozen=True)
class NodeInfo:
    op_type: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any]


@dataclass(frozen=True)
class ModelIR:
    name: str
    opset: int
    inputs: list[TensorInfo]
    outputs: list[TensorInfo]
    tensors: dict[str, TensorInfo]
    nodes: list[NodeInfo]
