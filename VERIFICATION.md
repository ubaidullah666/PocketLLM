# Verification record — on-demand update

Date: 2026-10-06

37 Python unittest checks passed in Linux, with the optional streaming dependencies installed.

New checks cover:
- An estimated 100 GiB checkpoint can pass a 4 GiB working-set budget when the largest component and runtime fit; an oversized component is refused. This is an estimator test, not a 100 GiB inference benchmark.
- Tiny randomly initialized real Llama, Qwen2 and Qwen3 models produce CPU logits matching conventional resident execution (rtol 1e-5, atol 1e-6).
- Next-token forward with retained KV cache matches resident execution.
- Tied input/output embedding weights work.
- Parameters start on meta without reading weights; they return to meta after forward.
- Active weight bytes return to zero; peak resident weight bytes stay at or below the largest component.
- Cancellation before reads, budget rejection, checkpoint modification and forward-exception cleanup.
- A real worker subprocess loads a tiny Safetensors checkpoint with a local tokenizer, performs generation through HTTP/SSE, emits paging counters and releases its weights.

The prior 21 checks also pass. Those include malformed GGUF rejection, conventional allocation, local HTTP access controls and watchdog behavior; their llama.cpp subprocess uses a protocol fixture, not real llama.cpp inference.

Python compilation, frontend JavaScript syntax and static UI-ID references checked. Native Windows rendering, rendered browser UI, CUDA execution and performance with large trained checkpoints remain untested. No benchmark speed or zero-OOM guarantee is claimed.

Smart speed update: 5 additional tests cover cache size bounds, stable admission under cyclic traversal, pressure eviction, exact-prefix consumption, actual read reduction with matching model outputs, and prefix-KV equivalence. The cache tests count as three methods and real-model tests as two. A three-run median tiny CPU benchmark is included under benchmarks/. Smart speed completed the measured workload in 0.175 s versus 0.282 s for strict streaming. These are development microbenchmark results, not large-model predictions.
