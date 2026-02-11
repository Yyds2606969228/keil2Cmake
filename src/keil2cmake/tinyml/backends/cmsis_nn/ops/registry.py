# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Callable

from ....ir import NodeInfo
from ....operators.context import EmitContext


OP_HANDLERS: dict[str, Callable[[EmitContext, NodeInfo], None]] = {}


def register_op(name: str) -> Callable[[Callable[[EmitContext, NodeInfo], None]], Callable[[EmitContext, NodeInfo], None]]:
    def decorator(func: Callable[[EmitContext, NodeInfo], None]) -> Callable[[EmitContext, NodeInfo], None]:
        if name in OP_HANDLERS:
            raise ValueError(f"Duplicate operator registration: {name}")
        OP_HANDLERS[name] = func
        return func

    return decorator


def get_handler(name: str) -> Callable[[EmitContext, NodeInfo], None] | None:
    return OP_HANDLERS.get(name)
