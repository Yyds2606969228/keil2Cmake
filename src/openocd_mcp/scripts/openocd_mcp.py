#!/usr/bin/env python3
"""openocd-mcp entrypoint."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))

from openocd_mcp.server import main


if __name__ == "__main__":
    main()

