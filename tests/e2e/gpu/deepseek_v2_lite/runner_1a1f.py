#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Backward-compatible entrypoint for the DeepSeekV2 AFD 1A1F smoke test."""

from __future__ import annotations

import sys

from tests.e2e.gpu.deepseek_v2_lite.runner import main

if __name__ == "__main__":
    sys.exit(main())
