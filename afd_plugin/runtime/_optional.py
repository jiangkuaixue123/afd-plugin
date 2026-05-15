# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Optional vLLM imports for Phase 1 runtime class-path placeholders."""

from __future__ import annotations

import importlib
from typing import Any


def optional_class(
    module_name: str,
    class_name: str,
) -> tuple[type[Any], BaseException | None]:
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except Exception as exc:
        return object, exc
    if not isinstance(cls, type):
        return object, TypeError(f"{module_name}.{class_name} is not a class")
    return cls, None


def phase1_placeholder_error(
    class_name: str,
    base_error: BaseException | None,
) -> NotImplementedError:
    message = (
        f"{class_name} is a Phase 1 class-path placeholder. Real AFD runtime "
        "behavior is scheduled for Phase 2/3; this class is currently only "
        "safe to import and resolve."
    )
    error = NotImplementedError(message)
    if base_error is not None:
        error.__cause__ = base_error
    return error
