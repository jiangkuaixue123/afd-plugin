# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU runtime classes loadable by explicit vLLM class paths."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AFDNPUAttentionModelRunner": (
        "afd_plugin.v1.worker.ascend.attention_model_runner"
    ),
    "AFDNPUAttentionWorker": "afd_plugin.v1.worker.ascend.attention_worker",
    "AFDNPUFFNModelRunner": "afd_plugin.v1.worker.ascend.ffn_model_runner",
    "AFDNPUFFNWorker": "afd_plugin.v1.worker.ascend.ffn_worker",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)


__all__ = list(_EXPORTS)
