# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU runtime classes loadable by explicit vLLM class paths."""

from afd_plugin.v1.worker.ascend.attention_model_runner import (
    AFDNPUAttentionModelRunner,
)
from afd_plugin.v1.worker.ascend.attention_worker import AFDNPUAttentionWorker
from afd_plugin.v1.worker.ascend.ffn_model_runner import AFDNPUFFNModelRunner
from afd_plugin.v1.worker.ascend.ffn_worker import AFDNPUFFNWorker
from afd_plugin.v1.worker.ascend.ubatch_wrapper import AFDNPUUBatchWrapper

__all__ = [
    "AFDNPUAttentionModelRunner",
    "AFDNPUAttentionWorker",
    "AFDNPUFFNModelRunner",
    "AFDNPUFFNWorker",
    "AFDNPUUBatchWrapper",
]
