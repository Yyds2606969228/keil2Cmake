# -*- coding: utf-8 -*-

from __future__ import annotations

from .base import Backend
from .c.backend import create_backend as create_c_backend


def get_backend() -> Backend:
    return create_c_backend()


__all__ = ["Backend", "get_backend"]
