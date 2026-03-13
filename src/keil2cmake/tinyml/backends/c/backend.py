# -*- coding: utf-8 -*-

from __future__ import annotations

from ..base import Backend
from .ops import get_handler as c_get_handler


def create_backend() -> Backend:
    return Backend(name="c", extra_includes=[], handler_provider=c_get_handler)
