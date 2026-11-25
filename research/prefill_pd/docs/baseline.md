# B0: reproducible baseline preparation

## Scope and inspected revision

Baseline: `355553ac22dca4080929ea8f88a446f15df1f3ba`.
Development host is Apple Silicon; execute serving and benchmarks only on a Linux
NVIDIA server. No performance or correctness results are available yet.

Source findings (paths relative to repository root):

| Feature | Evidence and implication |
| --- | --- |
| Chunked prefill | `vllm/v1/core/sched/scheduler.py` schedules token budgets and partial prefills. This is not proof of the paper's layer-wise hybrid prefilling. |
| Prefix caching | `vllm/v1/core/kv_cache_manager.py`, `block_pool.py`, and `single_type_kv_cache_manager.py` already manage reusable blocks. Keep caching enabled in B0 and record actual hit rates. |
| CPU KV offload | `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`, `simple_cpu_offload_connector.py`, and `vllm/v1/kv_offload/` already exist. `tests/v1/kv_connector/nixl_integration/run_multi_connector_accuracy_test.sh` covers NIXL plus offloading. M1 must distinguish existing cache offloading from proposed active-layer offloading. |
| Scheduling | `vllm/v1/core/sched/request_queue.py` exposes FCFS and priority queues. These do not implement cache-aware JCT prediction and continuous recalibration. No such implementation was identified in the inspected scheduler paths. |
| Hybrid prefilling | No general implementation of the paper's non-attention-layer chunking was identified in the inspected scheduler/config/compilation paths. This is a scoped finding, not a repository-wide absence proof. |
| Transfer lifetime | Scheduler `_free_request` consults connector completion; blocks can remain held until `finished_sending`. Preserve this contract. NIXL also uses leases and heartbeats. |
| PD routing | Use `tests/v1/kv_connector/nixl_integration/toy_proxy_server.py`, as specified in `docs/features/nixl_connector_usage.md`; a generic sequential proxy is not interchangeable. |

The [PrefillOnly paper](https://arxiv.org/html/2505.07203v1) chunks non-attention
operations within a forward pass and recalibrates scheduling estimates as prefix
cache contents change. Its experiments discard unneeded suffix KV; CPU offloading
is discussed as an alternative. PD decoding needs that KV, so discarding it is
not an applicable optimization. See also the
[PD RFC](https://github.com/vllm-project/vllm/issues/19038).

## Freeze the server configuration

1. Use two exclusive GPUs on one Linux host, one per worker, TP=1. Select a dense
   causal model supported by NIXL that fits both GPUs. Hardware/model selection
   remains open until a server is available. The supplied memory/token budgets
   are starting values, not validated capacity claims.
2. Check out the pinned commit. Follow repository environment setup with `uv venv
   --python 3.12`, `uv pip install -r requirements/lint.txt`, and
   `.venv/bin/pre-commit install`. Install vLLM using
   `VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto` and the
   applicable NIXL requirement from `requirements/kv_connectors.txt`. Install
   `requirements/test/cuda.in` for integration validation. Verify wheel/source
   compatibility; preserve installation logs and a reusable container digest or
   wheel archive, in addition to the captured package freeze.
3. Download model and tokenizer to an immutable local snapshot. Preserve their
   repository revision and file SHA256 manifest. Do not use a moving model name.
4. Copy `configs/b0.env.example`, fill all values, and source it in every terminal.
   Archive the completed config. Use the same config for all compared runs.
   Explicit controls are BF16, FCFS, prefix caching, chunked prefill, seed, batch
   budget, concurrency budget, and GPU memory fraction. Compilation uses this
   revision's defaults; archive both worker startup logs, including resolved
   configuration. Do not add eager mode to only one experimental condition.
5. Start each in a separate terminal and save its log to a unique session folder:

   ```bash
   bash research/prefill_pd/scripts/serve-b0.sh prefill
   bash research/prefill_pd/scripts/serve-b0.sh decode
   bash research/prefill_pd/scripts/serve-b0.sh proxy
   ```

   Wait for both worker health endpoints and the proxy startup log. Archive OS,
   CPU/RAM, driver/CUDA, GPU topology, transport configuration, clocks/power limits,
   worker/proxy logs, and a copy/hash of these research scripts alongside runs.
   Pin the load-generator host and networking as well.

## Correctness gate before measurement

Use the existing NIXL `test_accuracy.py` and its `run_accuracy_test.sh` as the
reference for standalone-versus-PD validation with the selected model. The shell
harness uses bare Python and broad process cleanup: do not execute it unchanged;
use a research-local adaptation with `.venv/bin/python` and owned process IDs.
Install dependencies through uv. Preserve commands, output, model/task details,
seeds and tolerance. Compare deterministic standalone and PD generations for
short/long inputs and 1/128/512 outputs; exercise repeated prefixes, concurrency,
and transfer failure. Require successful requests and actual connector transfers,
not merely HTTP health. This GPU validation is pending and is a prerequisite to
accepting B0 results. Output-length 1 still exercises the PD proxy and is not a
standalone PrefillOnly measurement.

## Measurements

Restart both workers and proxy before **each** condition/repeat to prevent cache
carryover. Wait for readiness. Each invocation runs 16 warmups and the benchmark's
initial probe; these warm the first prompt/cache. This is a warm-cache protocol,
not a cold-cache latency test. Use the same repeat index across configurations to
replay the same synthetic workload. Repeat indices 1, 2, 3 provide independent
seeds. Save failures; never reuse or edit result directories.

```bash
mkdir -p research/prefill_pd/results
bash research/prefill_pd/scripts/bench-b0.sh \
  research/prefill_pd/results/B0-input4096-r1 4096 128 0.5 0 1 1000
```

The 0.5 QPS and 1000 requests above are examples, not measured sustainable rates.
The runner refuses existing output directories, wrong revisions, tracked edits
in serving-related paths, non-Linux execution, and contexts beyond the config.
Inspect git-status.txt for untracked source files as an additional acceptance gate.
A FINISHED marker means command execution ended, not that every request succeeded.
Require completed=requested, no transfer errors, and expected token lengths.

| Workload | Conditions |
| --- | --- |
| Input scaling | 512, 2048, 4096, 8192, 16384, optionally 32768; output 128; prefix 0; fixed low QPS verified to avoid queue buildup |
| Output scaling | Input 4096; outputs 1, 128, 512; prefix 0; same low QPS |
| Rate scaling | Input 4096/output 128/prefix 0; 0.25, 0.50, 0.75, 1.00, 1.25 times calibrated sustainable QPS |
| Prefix scaling | Input 4096/output 128; prefix percentages 0, 25, 50, 75; fixed rate, e.g. 0.5 times sustainable QPS |

For calibration, bracket capacity by increasing finite offered QPS, then refine
between passing/failing rates. Before running, choose TTFT/ITL SLOs and observation
window. Suggested minimum: 5 minutes and 1000 requests per candidate, three
repeats; increase request count to cover the window. A sustainable candidate must
meet the SLOs with all requests successful, achieved rate close to offered rate
(predeclare tolerance), and no growing worker waiting queue. Sample /metrics
throughout calibration, not only at endpoints. Select the highest rate passing
all repeats. Archive calibration runs separately and freeze this rate for all
comparisons. Never normalize each optimization by its own newly measured capacity.
Tail estimates from 1000 requests are preliminary; increase sample size and report
between-repeat variability. Do not average per-run P99s into a pooled P99.

Prefix percentage means the fraction of input tokens intended to be shared,
not the fraction of requests that hit cache. The runner subtracts the shared
prefix from random-input-len because this CLI adds random-prefix-len to it.
Random text is decoded/re-tokenized: inspect generated token lengths and verify
token-identical shared prefixes with the frozen tokenizer before accepting the
prefix experiment. Block alignment and accidental sharing affect actual hits.
The built-in generator is a reproducible initial workload, not a validated
multi-prefix fairness trace. Fairness evaluation needs request IDs, per-request
queue delay and a mixed-length/multiple-prefix workload in a later experiment.

## Metric interpretation and outstanding instrumentation

- `result.json` includes detailed client latencies and success/throughput data;
  compare P50/P99 TTFT and P99 ITL. ITL is undefined for a one-token response.
- Worker `/metrics` snapshots include prefill/queue histograms and cache usage.
  Before/after deltas include the probe and warmups in this runner; do not present
  them as measurement-only request statistics. Histograms are not raw GPU timings.
- Client TTFT includes routing, queueing, prefill, transfer and decoder work.
  `vllm:request_prefill_time_seconds` is a worker request-stage metric, not isolated
  CUDA compute time. Separate profiling runs with GPU timelines are required for
  the primary computation-latency metric.
- GPU samples are device-wide memory.used at 200 ms intervals. They can miss peaks
  and include weights, reserved allocator/KV pools and other allocations. They
  are **not peak tensor allocation**. Use an isolated profiler/memory trace in a
  separately labeled run to measure transient allocation; do not mix profiled
  throughput with ordinary B0 throughput.
- Inspect `vllm/distributed/kv_transfer/kv_connector/v1/nixl/stats.py` and actual
  exported metrics for connector transfer measurements. Do not derive transfer
  latency by subtracting independent TTFT summaries. CPU RSS and continuous queue
  sampling require external process monitoring; these are not captured by this
  initial runner.
- Maximum input length must leave space for output (32768+512=33280). Distinguish
  model context limits, startup allocation failures, and runtime OOM. A grid gives
  the largest tested successful length, not an exact maximum. Refine the boundary
  with fixed concurrency/output/config and preserve failed trials.

## Local verification

`bash -n` passes for both scripts. CUDA execution, connector compatibility,
CLI runtime, model correctness, token-prefix validation, capacity calibration and
performance measurements remain pending on the GPU server. No system Python or
local CUDA benchmark was used. Neither uv nor a project .venv was available on
this development host during preparation.
