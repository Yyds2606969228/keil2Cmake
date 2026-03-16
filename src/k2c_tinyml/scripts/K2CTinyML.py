#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TinyML ONNX -> C generator (modular entrypoint)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from k2c_tinyml.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

