#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Keil uVision -> CMake converter (modular entrypoint)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))

from keil2cmake.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
