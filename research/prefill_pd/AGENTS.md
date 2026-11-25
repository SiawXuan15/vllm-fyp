# Project: Optimizations on Prefill Tasks in PD Disaggregation

## Research Background

This project investigates whether optimization techniques proposed in the PrefillOnly paper (arXiv:2505.07203) can improve dedicated prefill workers in vLLM's Prefill–Decode (PD) disaggregated inference system.

Paper: <https://arxiv.org/pdf/2505.07203>

vLLM repository: <https://github.com/vllm-project/vllm>

Relevant RFC: <https://github.com/vllm-project/vllm/issues/19038>

## Important Technical Context

- vLLM already supports PD disaggregation.
- PrefillOnly is built on vLLM and targets single-token inference workloads.
- The PrefillOnly paper proposes hybrid prefilling, KV-cache management, and job-completion-time-aware (JCT-aware) scheduling.
- The paper suggests extending PrefillOnly to dedicated prefill workers in PD disaggregation but leaves that extension to future work.
- PD disaggregation requires preserving and transferring the KV cache needed for subsequent decoding.
- Do not discard required KV caches before successful transfer to the decode worker.

## Research Objectives

1. Establish a baseline using unmodified vLLM PD disaggregation.
2. Re-evaluate relevant PrefillOnly optimization techniques.
3. Identify techniques not already implemented in the selected vLLM version.
4. Implement or adapt selected optimizations in vLLM.
5. Compare baseline and modified implementations under controlled conditions.
6. Prepare a well-tested contribution to the upstream vLLM repository.

## Evaluation Metrics

Primary:

- Prefill computation latency
- Peak prefill GPU memory usage
- Maximum supported input length
- Request throughput
- P50 and P99 time to first token (TTFT)

Secondary:

- KV-cache transfer latency
- Inter-token latency (ITL)
- CPU RAM usage
- Queueing delay and scheduling fairness
- Inference correctness

## Experimental Configurations

B0: Unmodified vLLM PD baseline.

H1: PD with hybrid prefilling.

M1: PD with CPU KV-cache offloading.

S1: PD with JCT-aware scheduling.

C1: PD with multiple applicable optimizations combined.

These are proposed configurations, not confirmed existing implementations.

## Workloads

Input token lengths: 512, 2048, 4096, 8192, 16384, and optionally 32768.

Output token lengths: 1, 128, and 512.

Arrival rates: 25%, 50%, 75%, 100%, and 125% of the measured sustainable baseline request rate.

Prefix reuse: 0%, 25%, 50%, and 75%.

## Experimental Controls

Keep the model, tokenizer, GPU hardware, base vLLM revision, KV connector, precision, sampling settings, and workloads consistent when comparing baseline and optimized implementations.

Perform warm-up runs and repeat measurements to evaluate variability.

## Current Environment

Development computer: MacBook Air with Apple Silicon.

Preferred editor: VS Code.

GPU server: Not yet available.

Do not attempt CUDA benchmarking locally on the MacBook. Prepare scripts and configurations that can later run on a compatible Linux NVIDIA GPU server.

## Instructions for Codex

- Read the PrefillOnly paper and relevant vLLM implementation before suggesting changes.
- Inspect the actual checked-out vLLM repository before assuming file paths or APIs.
- Do not assume that an optimization is absent from current vLLM.
- Keep baseline and optimized experiments reproducible.
- Clearly distinguish implemented functionality from proposed functionality.
- Prefer focused, testable changes over large modifications.
- Do not fabricate experimental results.
- Do not modify the original baseline results.
- Explain code changes and required tests.
- Do not create a GitHub pull request unless explicitly requested.
- Preserve KV-cache correctness when implementing PD-specific optimizations.
