#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Keil uVision -> CMake converter (modular entrypoint).

Thin wrapper that delegates to [keil2cmake_cli.py](keil2cmake_cli.py).
"""

from keil2cmake_cli import main


if __name__ == "__main__":
    raise SystemExit(main())