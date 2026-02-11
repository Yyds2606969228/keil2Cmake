# -*- coding: utf-8 -*-

from __future__ import annotations

from ..base import Backend
from ..c.ops import get_handler as c_get_handler
from .ops import get_handler as cmsis_get_handler


def _get_handler(op_name: str):
    handler = cmsis_get_handler(op_name)
    if handler is not None:
        return handler
    return c_get_handler(op_name)


def create_backend() -> Backend:
    return Backend(
        name="cmsis-nn",
        extra_includes=[
            "#ifdef K2C_USE_CMSIS_NN",
            '#include "arm_nnfunctions.h"',
            '#include "arm_math.h"',
            "#endif",
        ],
        handler_provider=_get_handler,
    )
