# NPU E2E Test Matrix

NPU E2E tests are opt-in because they require a vLLM-Ascend runtime, Ascend
devices, a local DeepSeekV2-Lite model, and the AFD Ascend custom ops.

Run the full matrix with:

```bash
AFD_NPU_E2E_MODEL=/path/to/DeepSeek-V2-Lite \
AFD_NPU_E2E_DEVICES=0,1,2,3 \
uv run pytest -q -m npu tests/e2e/npu/deepseek_v2_lite
```

Engine logs are written per case under
`/tmp/afd_npu_e2e_logs/<case-id-topology-ubatch-mode>/` by default:

```text
/tmp/afd_npu_e2e_logs/NPU-E2E-001-1A1F-no-ubatch-eager/ffn.log
/tmp/afd_npu_e2e_logs/NPU-E2E-001-1A1F-no-ubatch-eager/attention.log
```

Set `AFD_NPU_E2E_LOG_DIR=/path/to/log-root` to change the root directory.

The pytest wrapper shells out to:

```bash
python tests/e2e/npu/deepseek_v2_lite/runner.py ... --log-dir /path/to/case-log-dir
```

FULL graph cases automatically enable `AFDDecodeBenchConnector` on the
Attention side and send two same-shape request rounds so the first round can
capture and the second round can replay. Ubatch cases enable two-way DBO
ubatching with `num_ubatches=2`.

## Case Matrix

| Case ID | Topology | Attention ranks | FFN ranks | Ubatch | Mode | AFD connector | KV connector | Acceptance focus |
| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |
| NPU-E2E-001 | 1A1F | 1 | 1 | Off | eager | `camp2pconnector` | None | Basic AFD A2E/E2A serving loop |
| NPU-E2E-002 | 1A1F | 1 | 1 | On | eager | `camp2pconnector` | None | Two ubatch stages are sent and processed |
| NPU-E2E-003 | 1A1F | 1 | 1 | Off | FULL graph | `camp2pconnector` | `AFDDecodeBenchConnector` | ACL graph capture and replay |
| NPU-E2E-004 | 1A1F | 1 | 1 | On | FULL graph | `camp2pconnector` | `AFDDecodeBenchConnector` | Ubatch split plus graph capture/replay |
| NPU-E2E-005 | 2A2F | 2 | 2 | Off | eager | `camp2pconnector` | None | DP rank routing across 2 Attention and 2 FFN ranks |
| NPU-E2E-006 | 2A2F | 2 | 2 | On | eager | `camp2pconnector` | None | Both FFN ranks process ubatch stages 0 and 1 |
| NPU-E2E-007 | 2A2F | 2 | 2 | Off | FULL graph | `camp2pconnector` | `AFDDecodeBenchConnector` | Both FFN ranks capture and replay |
| NPU-E2E-008 | 2A2F | 2 | 2 | On | FULL graph | `camp2pconnector` | `AFDDecodeBenchConnector` | 2 ranks x 2 ubatch stages capture/replay |
| NPU-E2E-009 | 2A1F | 2 | 1 | Off | eager | `camp2pconnector` | None | Two Attention ranks route to one FFN rank |
| NPU-E2E-010 | 2A1F | 2 | 1 | On | eager | `camp2pconnector` | None | Single FFN rank processes both ubatch stages |
| NPU-E2E-011 | 2A1F | 2 | 1 | Off | FULL graph | `camp2pconnector` | `AFDDecodeBenchConnector` | Single FFN rank capture/replay with 2 Attention ranks |
| NPU-E2E-012 | 2A1F | 2 | 1 | On | FULL graph | `camp2pconnector` | `AFDDecodeBenchConnector` | Single FFN rank captures/replays both ubatch stages |

## Acceptance Criteria

| Dimension | Applies to | Pass criteria |
| --- | --- | --- |
| Startup | All cases | FFN and Attention processes are launched in the same runner startup phase before readiness checks, and `/v1/models` is reachable |
| Request success | All cases | `/v1/completions` returns HTTP 200 with valid non-empty JSON output |
| AFD communication | All cases | No HCCL, CAM, A2E, E2A, or connector exceptions are emitted |
| Topology routing | All cases | Logs and request success match the configured `1A1F`, `2A2F`, or `2A1F` rank layout |
| Ubatch split | Ubatch cases | Attention reports two ubatch DP metadata slices and FFN reports two ubatch stages |
| FULL graph capture | FULL graph cases | FFN reports ACL graph capture |
| FULL graph replay | FULL graph cases | FFN reports ACL graph replay during the second same-shape request round |
| FULL graph plus ubatch | NPU-E2E-004, 008, 012 | Ubatch split markers and graph capture/replay markers are all present |
| Cleanup | All cases | Runner terminates Attention and FFN processes after success or failure |
