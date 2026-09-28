# Async combine recv prefix optimization

## Change

Previously every AIV scanned the routing table from token zero to its partition
end, including tokens owned by preceding cores. This reconstructed each expert's
occurrence index before receiving and accumulating the local tokens.

Dispatch send already stores inclusive per-AIV expert counts in the Attention
rank's communication window. For batches larger than the AIV count, combine recv
loads the preceding AIV's counts and starts scanning at its own token partition.
Core zero starts with zero counts. The temporary load reuses the expert-prefix
buffer before its final contents are populated, so the UB layout does not grow.

The optimization requires dispatch send and combine recv to use the same AIV
count and token partition rule. Changes to either partition must preserve this
contract. Top-k accumulation order, output conversion and communication state
publication/cleanup are unchanged.

Small batches retain tiling keys 100/101 (BF16/FP16); larger batches select
102/103. A boolean template parameter removes the prefix path at compile time
for small batches. In the tested build, the executable function bytes for keys
100/101 are identical to the baseline. No other communication operator changes.

## Experiment setup

Measured on 2026-09-28 in `jcz_afd2`: Ascend910_9382, CANN 9.0.1,
PyTorch 2.10.0+cpu and torch-npu 2.10.0.post2. The topology uses four devices:
two Attention ranks (TP=2), two MoE ranks, and 48 AIVs per device. Baseline source
revision: `205bd770113aee7c9c098c55068ca2ce7c766d9c`.

Both revisions use the unchanged communication precision/performance harness.
Native compilation and installation succeeded. Installed host tiling libraries,
device binaries and source hashes were checked separately. Runtime profiling
confirmed key 100 for small BF16 batches, key 102 for large BF16 batches, mixed
100/102 for lengths 49/48, and key 103 for large FP16 batches.

Multi-rank `msprof` records device task duration. Each performance profile has
one precision preflight, five warmups and five measured iterations. For each
measured iteration, take the maximum combine recv duration across Attention
ranks, then average those five maxima. The committed
[experiment data](experiments/combine_recv_prefix_20260928.json) includes
per-rank samples, the resulting maxima, correctness configurations and the
small-key machine-code hashes.

## Device task results

Lengths are per Attention rank; experts are the total across both MoE ranks.
Quantization describes the dispatch input path.

| Lengths | Hidden | Experts | Top-k | Dtype / quant | Baseline (µs) | Optimized (µs) | Change |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 1025/1024 | 256 | 8 | 3 | BF16 / 1 | 140.036 | 39.560 | −71.75% |
| 1025/1024 | 256 | 256 | 2 | BF16 / 0 | 105.668 | 42.064 | −60.19% |
| 49/48 | 256 | 8 | 3 | BF16 / 1 | 17.592 | 16.676 | −5.21% |
| 7/5, pair 1 | 7168 | 8 | 2 | BF16 / 0 | 14.368 | 14.768 | +2.78% |
| 7/5, pair 2 | 7168 | 8 | 2 | BF16 / 0 | 14.392 | 13.788 | −4.20% |

A second independent 256-expert baseline measured 104.360 µs, compared with the
same optimized profile's 42.064 µs. Each optimized large-shape result above is
one profile group, not a distribution across many independent profile runs.
Small-batch A/B differences change direction, with unchanged executable bytes;
no stable small-batch regression was observed. These measurements do not imply
an end-to-end model speedup or qualify every supported shape.

The regular performance program also passed its precision preflight. Its event
timing includes waiting and runtime effects and was visibly noisier than device
task timing; it is not used for the kernel improvement percentages above.

## Precision coverage

All cases below passed with zero mismatches using the existing tolerances.
Hidden size is 256. The performance harness additionally validated the
1025/1024 BF16 quantized and hidden-7168 cases before measurement.

| Dtype / quant | Lengths | Top-k | Experts/rank | Routing | Capacity | Rounds |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| BF16 / 0 | 97/65 | 3 | 4 | sparse | 256 | 2 |
| BF16 / 1 | 97/65 | 3 | 4 | balanced | 128 | 2 |
| BF16 / 1 | 49/48 | 3 | 4 | balanced | 64 | 2 |
| BF16 / 0 | 1025/1024 | 2 | 128 | balanced | 2048 | 1 |
| FP16 / 0 | 7/5 | 2 | 4 | balanced | 8 | 2 |
| FP16 / 1 | 97/65 | 3 | 4 | balanced | 128 | 2 |
| FP16 / 1 | 7/5 | 2 | 4 | sparse | 16 | 2 |

This covers both specialized dtypes, empty experts, multiple receive chunks,
window reuse, uneven token partitions, and both sides of the specialization
threshold in one run. No tolerance or test-harness changes are included.

## Reproduce

Use separate baseline and optimized checkouts/installations and fresh processes
to avoid mixing host tiling libraries and device binaries. From each checkout,
with its matching CANN environment loaded, build using the
[native setup instructions](../../tests/npu/README.md#native-setup).

```bash
torchrun --standalone --nproc-per-node=4 tests/npu/async_cam_precision.py \
  --lengths 97,65 --top-k 3 --capacity 128 --dtype fp16 \
  --dynamic-quant 1 --iterations 2 --output precision.json

msprof --output=./profile-large --application="torchrun --standalone \
  --nproc-per-node=4 tests/npu/async_cam_performance.py \
  --measure-mode recv --warmup 5 --iterations 5 --lengths 1025,1024 \
  --top-k 3 --capacity 1024 --dtype bf16 --dynamic-quant 1 \
  --output performance.json"
```

For the 256-expert case, set `--experts-per-rank 128 --top-k 2 --capacity 2048
--dynamic-quant 0`. For the threshold case, use `--lengths 49,48 --top-k 3
--capacity 64 --dtype bf16 --dynamic-quant 1`. For the small-batch control, use
`--lengths 7,5 --hidden-size 7168 --top-k 2 --capacity 16 --dtype bf16
--dynamic-quant 0`. The four-device topology otherwise uses harness defaults.
Apply the same warmup and cross-rank aggregation to both revisions' exported
`op_summary` CSV files.
