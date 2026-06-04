# vllm-afd-plugin

**vllm-afd-plugin** is a [vLLM](https://github.com/vllm-project/vllm)
external plugin for **Attention-FFN Disaggregation (AFD)**. The package
provides AFD runtime adapters, connector logic, model wrappers, configuration
validation, and GPU-gated integration tests for vLLM deployments.

The target runtime is **vLLM `v0.19.1`**. The plugin avoids modifying the vLLM
source tree: AFD behavior is provided by this package, `vllm.general_plugins`,
`--worker-cls`, `--additional-config`, plugin-owned model wrappers, and narrow
compatibility shims where no public extension point exists.

## Architecture

![vLLM AFD plugin architecture](docs/assets/vllm-afd-plugin-architecture.svg)

## Current Status

The repository currently contains the AFD plugin skeleton plus Attention, FFN,
P2P, DBO, and CUDA graph MVP runtime paths.

Core runtime support:

- Python package metadata for the `vllm-afd-plugin` distribution.
- `vllm.general_plugins` entry point named `afd`, implemented by
  `afd_plugin:register_afd`.
- CPU-safe imports, config parsing, stack validation, and class-path
  resolution tests.
- Plugin-owned `AFDConfig`, parsed from vLLM
  `additional_config["afd"]`.
- Attention runtime adapter:
  `afd_plugin.v1.worker.AFDAttentionWorker`.
- FFN runtime adapter:
  `afd_plugin.v1.worker.AFDFFNWorker`.
- Attention and FFN model runners that exchange AFD metadata and hidden states
  through the plugin connector contract.
- Two-way DBO/ubatching metadata support for the current AFD paths.
- `FULL_DECODE_ONLY` CUDA graph support for the current Attention/FFN runtime
  shape, including FFN graph-keyed capture/replay.

Model support:

| Model family | Registered architectures | Status | Notes |
| --- | --- | --- | --- |
| DeepSeekV2 / DeepSeekV3 / GLM MoE DSA | `DeepseekForCausalLM`, `DeepseekV2ForCausalLM`, `DeepseekV3ForCausalLM`, `GlmMoeDsaForCausalLM` | Supported for AFD smoke and E2E validation | Uses `afd_plugin.model_executor.models.deepseek_v2` wrappers. Attention and FFN sides currently load full model weights. |
| Other model families | Not registered by this plugin | Not supported yet | Add a plugin-owned model wrapper before using AFD-specific model forward behavior. |

Connector support:

| Connector | Status | Platform | Ubatch support | Graph support | Notes |
| --- | --- | --- | --- | --- | --- |
| `p2pconnector` | Supported | CUDA | DBO supported | `FULL_DECODE_ONLY` | Uses FFN ranks first, followed by Attention ranks. `num_attention_servers` must be greater than or equal to `num_ffn_servers` and divisible by it. |
| Other AFD connectors | Not supported yet | N/A | N/A | N/A | Connector implementations should be added under `afd_plugin.connectors` and registered through the connector factory. |

Known gaps remain important:

- vLLM versions other than `0.19.1` are not claimed as supported.
- Only the vLLM v1 model runner path is supported; the v2 model runner is not
  supported yet.
- Role-based weight pruning is not implemented yet; Attention and FFN sides
  still load full DeepSeekV2 weights.
- GPU end-to-end tests are opt-in and currently focus on request success rather
  than token-by-token eager/graph output comparison.
- CUDA graph support is limited to vLLM `FULL_DECODE_ONLY`; other graph modes
  fail fast.
- DBO plus CUDA graph is limited to two ubatches.
- TP/PP, ratio topologies beyond the current validated cases, and non-DeepSeekV2
  model paths still need hardening.

## Roadmap

Near-term development focuses on:

- Ascend NPU platform support, including connector, worker, model runner, and
  related runtime compatibility work.
- Broader model support through additional plugin-owned model wrappers and
  AFD-specific forward-path integration.
- Role-based weight loading and pruning, so Attention workers load only
  Attention-side weights and FFN workers load only FFN-side weights.

## Install

Requires Python **3.10-3.13** and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/vllm-project/afd-plugin.git
cd afd-plugin
uv sync --group dev
```

`vllm` is an optional runtime extra so CPU-only or macOS development
environments can still run import/config tests without a CUDA wheel:

```bash
# Linux / CUDA-capable environments
uv sync --group dev --extra vllm
```

The optional extra pins `vllm==0.19.1`.

## Using the Plugin

Install or sync the distribution as `vllm-afd-plugin`. Python imports and CLI
class paths use the `afd_plugin` package name.

Ask vLLM to load the plugin entry point:

```bash
export VLLM_PLUGINS=afd
unset VLLM_USE_V2_MODEL_RUNNER
```

AFD is configured through the native vLLM `--additional-config` channel. There
is no separate `--afd-config` flag. The current runtime adapters target vLLM's
v1 GPU model-runner path.

Minimal Attention-side shape:

```bash
vllm serve /path/to/DeepSeek-V2-Lite \
  --worker-cls afd_plugin.v1.worker.AFDAttentionWorker \
  --served-model-name deepseek-v2-lite-afd-attention \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --enforce-eager \
  --host 127.0.0.1 \
  --port 18000 \
  --additional-config '{"afd":{"enabled":true,"role":"attention","connector":"p2pconnector","host":"127.0.0.1","port":6239,"num_attention_servers":1,"num_ffn_servers":1}}'
```

Minimal FFN-side shape:

```bash
vllm serve /path/to/DeepSeek-V2-Lite \
  --worker-cls afd_plugin.v1.worker.AFDFFNWorker \
  --served-model-name deepseek-v2-lite-afd-ffn \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --enforce-eager \
  --host 127.0.0.1 \
  --port 18001 \
  --additional-config '{"afd":{"enabled":true,"role":"ffn","connector":"p2pconnector","host":"127.0.0.1","port":6239,"num_attention_servers":1,"num_ffn_servers":1}}'
```

Start the FFN side first, then start the Attention side and send requests to
the Attention API server. The FFN worker is connector-driven; scheduler-driven
FFN `execute_model()` calls intentionally fail fast.

For repeatable local GPU smoke testing, prefer the bundled runner:

```bash
uv run python tests/e2e/gpu/deepseek_v2_lite/runner.py \
  --model /path/to/DeepSeek-V2-Lite \
  --num-attention-servers 1 \
  --num-ffn-servers 1 \
  --attention-gpus 0 \
  --ffn-gpus 1 \
  --api-port-base 18000 \
  --afd-port 6239 \
  --common-vllm-arg=--trust-remote-code
```

## AFD Config

The canonical config shape is:

```json
{
  "afd": {
    "enabled": true,
    "role": "attention",
    "connector": "p2pconnector",
    "host": "127.0.0.1",
    "port": 1239,
    "num_attention_servers": 2,
    "num_ffn_servers": 1,
    "afd_server_rank": 0
  }
}
```

`role` must be either `attention` or `ffn`. The plugin also accepts selected
compatibility field aliases such as `afd_role`, `afd_connector`, `afd_host`,
`afd_port`, and `afd_extra_config`.

## Development

Run the default CPU-safe checks:

```bash
uv run pytest
uv run ruff check .
```

Opt-in GPU E2E tests require a CUDA-capable vLLM environment and a DeepSeekV2
Lite model path:

```bash
AFD_GPU_E2E_MODEL=/path/to/DeepSeek-V2-Lite uv run pytest -q -m gpu
```

The GPU tests default to `AFD_GPU_E2E_GPUS=0,1,2,3` and cover eager
`1A1F`/`2A2F`, `FULL_DECODE_ONLY` CUDA graph `1A1F`/`2A2F`, and
`FULL_DECODE_ONLY` `2A2F` with DBO ubatch replay.

## Docs

- [docs/PHASE0_COMPATIBILITY_INVENTORY.md](docs/PHASE0_COMPATIBILITY_INVENTORY.md)
  - vLLM `0.19.1` extension-point and compatibility inventory.
- [docs/ATTENTION_RUNTIME_DESIGN.md](docs/ATTENTION_RUNTIME_DESIGN.md) -
  Attention worker and model-runner design.
- [docs/FFN_RUNTIME_DESIGN.md](docs/FFN_RUNTIME_DESIGN.md) - FFN worker,
  daemon loop, and connector-driven execution design.
- [docs/PHASE4_P2P_DESIGN.md](docs/PHASE4_P2P_DESIGN.md) - P2P connector
  topology, rank mapping, and current gaps.
- [docs/PHASE6_CUDA_GRAPH_IMPLEMENTATION.md](docs/PHASE6_CUDA_GRAPH_IMPLEMENTATION.md)
  - CUDA graph policy, implementation notes, and remaining work.
- [docs/GPU_E2E_TESTS.md](docs/GPU_E2E_TESTS.md) - opt-in GPU pytest and
  manual runner commands.

## License

Apache License 2.0 - see [LICENSE](LICENSE).
