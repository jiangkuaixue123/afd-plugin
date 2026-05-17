# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Isolated monkey-patch namespace.

Patches in this package must remain idempotent, version-aware, documented, and
covered by CPU-safe tests whenever possible.
"""

from afd_plugin.compat.patches.engine_core import apply_engine_core_patch

__all__ = ["apply_engine_core_patch"]
