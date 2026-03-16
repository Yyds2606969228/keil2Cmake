# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..ir import NodeInfo
from ..operators.context import EmitContext


@dataclass(frozen=True)
class Backend:
    name: str
    extra_includes: list[str]
    handler_provider: Callable[[str], Callable[[EmitContext, NodeInfo], None] | None]

    def get_handler(self, op_name: str) -> Callable[[EmitContext, NodeInfo], None] | None:
        return self.handler_provider(op_name)
