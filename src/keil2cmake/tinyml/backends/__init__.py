# -*- coding: utf-8 -*-

from __future__ import annotations

from .base import Backend
from .c.backend import create_backend as create_c_backend
from .cmsis_nn.backend import create_backend as create_cmsis_nn_backend


def get_backend(name: str) -> Backend:
    if name == "cmsis-nn":
        return create_cmsis_nn_backend()
    if name == "c":
        return create_c_backend()
    raise ValueError(f"Unsupported backend: {name}")


__all__ = ["Backend", "get_backend"]
