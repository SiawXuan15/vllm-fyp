# Experimental Design

## Project

Optimizations on Prefill Tasks in PD Disaggregation

## Research Question

Can optimization techniques proposed in PrefillOnly improve the performance of dedicated prefill workers in vLLM's PD-disaggregated architecture?

## Experiment Configurations

| ID | Configuration |
| --- | --- |
| B0 | Unmodified vLLM PD baseline |
| H1 | vLLM PD + hybrid prefilling |
| M1 | vLLM PD + CPU KV-cache offloading |
| S1 | vLLM PD + JCT-aware scheduling |
| C1 | vLLM PD + selected optimizations combined |

## Evaluation Metrics

- Prefill computation latency (milliseconds)
- Peak GPU memory allocation (GB)
- Maximum supported input length (tokens)
- Request throughput (requests/second)
- P50 and P99 TTFT (milliseconds)
- P99 inter-token latency (milliseconds)
- KV-cache transfer latency (milliseconds)
- CPU memory usage, where applicable
- Output correctness and request success rate

## Workload 1: Input Length Scaling

Input lengths: 512, 2048, 4096, 8192, 16384, 32768 tokens.

Output length: 128 tokens.

Measure prefill latency and GPU memory usage at a controlled low arrival rate.

## Workload 2: Output Length Scaling

Input length: 4096 tokens.

Output lengths: 1, 128, 512 tokens.

Evaluate downstream decoding effects.

## Workload 3: Request Rate Scaling

Input length: 4096 tokens.

Output length: 128 tokens.

Offered rates: 25%, 50%, 75%, 100%, and 125% of measured sustainable baseline QPS.

Measure throughput, TTFT and tail latency.

## Workload 4: Prefix Cache Reuse

Shared-prefix percentages: 0%, 25%, 50%, 75%.

Use repeated token-identical prefixes and varying suffixes.

Measure latency, cache reuse and scheduling fairness.

## Methodology

1. Verify the correctness of the unmodified vLLM PD serving setup.
2. Freeze the model checkpoint, hardware configuration, vLLM baseline commit, KV connector and workload definitions.
3. Conduct warm-up runs.
4. Run the baseline experiments and save raw metrics.
5. Implement selected PrefillOnly optimizations separately.
6. Rerun the same workloads with identical configurations.
7. Repeat each measured condition at least three times.
8. Perform an ablation study where multiple optimizations are implemented.
9. Validate correctness and report performance regressions as well as improvements.

## Hardware Requirements

- Linux NVIDIA GPU server
- Two GPUs for the initial PD setup
- Model and context lengths selected based on available VRAM
- MacBook Air used for local development only

## Expected Outputs

- Baseline result files
- Optimized result files
- Latency and throughput curves
- GPU memory plots
- Maximum context-length comparisons
- Analysis of KV-transfer overhead
- Reproducibility documentation
- Potential upstream vLLM pull request
