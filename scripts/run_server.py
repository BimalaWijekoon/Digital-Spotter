#!/usr/bin/env python3
"""
scripts/run_server.py
Purpose: Production server launcher for Raspberry Pi.
Author: bimalawijekoon
Version: 1.0.0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from integration.main_loop import main

if __name__ == "__main__":
    main()
