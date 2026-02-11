# -*- coding: utf-8 -*-

from __future__ import annotations

from .registry import get_handler

# Import operator modules to register handlers.
from . import conv  # noqa: F401
from . import add  # noqa: F401
from . import mul  # noqa: F401
from . import matmul  # noqa: F401
from . import gemm  # noqa: F401
from . import maxpool  # noqa: F401
from . import averagepool  # noqa: F401
from . import global_average_pool  # noqa: F401
from . import global_max_pool  # noqa: F401
from . import relu  # noqa: F401
from . import identity  # noqa: F401
from . import reshape  # noqa: F401

__all__ = ["get_handler"]

