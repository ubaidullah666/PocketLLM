# PocketLLM — Smart speed edition

A desktop app that can **load a model component, use it, release its weights, and repeat**. On-demand mode is the default. The model is not first loaded into a full in-memory state dictionary.

## Two clearly different modes

| Mode | Model format | Weight behavior |
|---|---|---|
| **On-demand — default** | Local Hugging Face Safetensors folder | Components are loaded on demand. Smart speed retains a bounded reusable subset; Minimum memory releases all weights after each operation. |
| Resident | Single GGUF file + llama-server | Fixed CPU/GPU layer allocation, as in the first version. No per-component release. |

**GGUF files do not work in the new streaming backend.** Select a local model folder containing `config.json`, tokenizer files including a chat template, and either `model.safetensors` or `model.safetensors.index.json` with its shards. Supported architectures: dense Llama, Qwen2 and Qwen3; F32, F16 or BF16 weights. Quantized Safetensors, MoE, multimodal and custom-code models are rejected. Model weights are not bundled and the app does not download them.

## Windows setup

Install 64-bit Python 3.11 or 3.12, extract the project, and open PowerShell in this folder. For CPU execution:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-streaming.txt
.\.venv\Scripts\python.exe run.py
```

Use `-3.11` if that is your Python version. After setup, double-click `start-windows.bat` to launch again. A Windows native window also needs Edge WebView2.

For optional NVIDIA CUDA execution, use this PyTorch installation command **instead of the CPU installation above**, then install `requirements-streaming.txt`:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

An appropriate NVIDIA driver is required. If you previously installed CPU PyTorch, uninstall it first with `python -m pip uninstall torch`, using the same virtual environment interpreter. On-demand mode supports CUDA only. Resident GGUF mode supports devices exposed by your llama.cpp build, including AMD and Intel GPUs through Vulkan. No paid API or Ollama is required. On-demand mode does not need llama-server.

If you want only the original GGUF mode, install `requirements.txt`, choose **Resident** in the app, and select a recent llama.cpp `llama-server` executable. For NVIDIA offload, use its CUDA build with the required engine flags. Keep its DLLs next to the executable.

The app also finds an engine installed in `tools/llama.cpp/` automatically when the executable field is blank. Entering a `.gguf` model path selects Resident mode automatically. On this machine, `../models/qwen2.5-3b.gguf` is available; use its full path in the interface. The local CPU engine is llama.cpp build b11429.

## Run a model

1. Leave **Loading behavior** on **On-demand · load, use, release**.
2. Choose your local Safetensors model folder.
3. Choose **Smart speed** (default) or **Minimum memory**, then set RAM and optional VRAM budgets. Setting VRAM to zero selects CPU execution.
4. Choose a context size; click **Inspect model**. The estimate uses the largest component, temporary conversion memory, KV cache, activations and runtime overhead, rather than the full checkpoint size.
5. Click **Load model**. This creates an empty model structure, reads metadata and loads the tokenizer. It does not materialize the weight tensors. When READY, send a prompt.
6. Watch the live component name/device, active and peak weight memory, and load/release counters. Stop cancels generation; Unload terminates the worker.

Budgets can be changed after unloading. Existing files from the previous version can be replaced by this extracted project; install the additional streaming dependencies in your virtual environment.

## Speed improvements

Smart speed is now the default. It uses only spare memory **inside** your cap after the working set, context cache and runtime reserves have been accounted for. The weight cache is capped at the smaller of available spare memory, 30% of your RAM cap and 50% of checkpoint weights in their execution representation. Nothing is preloaded in full. Cache entries are admitted as components are first used. A stable admitted subset avoids the repeated misses that ordinary LRU can produce when every layer is revisited cyclically. Entries are evicted under memory pressure; remaining weights continue to load/use/release normally.

Other changes: larger memory-aware prompt batches, exact conversation-prefix KV reuse, physical-core-aware CPU threading, fewer GPU allocator flushes, no unused forward pass after the final generated token, and live first-token latency/tokens-per-second counters. Prefix reuse is skipped whenever tokenization or conversation content does not match. It never substitutes an old answer. Cache admission and reuse do not change the model's weights or decoding strategy.

If minimum RAM use matters more, select **Minimum memory** for the previous immediate-release behavior. If your model already fits in RAM/VRAM, the resident GGUF mode may be more appropriate than repeatedly streaming Safetensors. The app does not silently switch model formats or truncate your conversation to appear faster.

Measured development microbenchmark (tiny random Llama, CPU, warm OS file cache, 128 prompt tokens plus 8 decode passes; median of 3 runs):

| Measure | Minimum memory | Smart speed |
|---|---:|---:|
| Prompt processing | 0.101 s | 0.025 s |
| Total | 0.282 s | 0.175 s |
| Logical weight reads | 34.7 MB | 22.0 MB |

Outputs matched within the test's numerical tolerance. This is **not a large-model benchmark or a promised speedup on your PC**. Run `python benchmarks/compare_speed.py` to repeat the tiny-model comparison locally. The exact development result is in `benchmarks/development_result.json`.

## How load/use/release works

- A Transformers model skeleton is built with Accelerate's `init_empty_weights`. All parameters initially live on the meta device and have no weight storage.
- Each supported leaf module gets a pre-forward and post-forward hook. A “component” is a parameter-owning module such as an attention projection, normalization, embedding, or feed-forward projection; it is usually smaller than a full transformer layer.
- Immediately before execution, the pre-hook checks available memory and loads only that component's tensors through `safe_open`. The tensors are copied independently of the source mapping. No full checkpoint dictionary, full shard, or full transformer layer is loaded into Python first.
- CPU computation uses F32 weights. Where CUDA and the budget allow it, eligible floating-input components temporarily execute on GPU. Otherwise they execute on CPU. The choice is made again for each component using live memory headroom.
- After execution, the output returns to CPU and module parameters return to meta placeholders. Smart speed can keep the underlying CPU weight tensor in a bounded reuse cache; Minimum memory releases it immediately. GPU allocator reuse is allowed during smart generation, with unused CUDA cache released at response completion. No speculative prefetch is used.
- Small activations, model buffers, tokenizer data and the conversation KV cache remain in RAM. KV memory is part of the estimate and grows with context. Minimum memory releases it after the response. Smart speed retains one conversation prefix cache and consumes it only if a later prompt starts with exactly the same tokens. Mismatched or cancelled requests discard it; unloading releases it.
- Prompt prefill uses 32 tokens in Minimum memory mode. Smart speed selects 32, 64, 128 or 256 based on the available RAM estimate, charging additional activation memory before allocating the weight cache. Each generated token repeats the necessary model operations. Local token output is streamed into the chat; this backend currently uses greedy decoding.
- A separate inference process lets the desktop remain responsive and reclaim inference memory by terminating the worker.

**Dense models still need all their layers for every output token.** This mode releases weights between operations; it does not identify a small topic-specific subset of the model. Repeated disk reads and CPU/GPU transfers can make it much slower than keeping weights resident. An SSD can help. It may run a checkpoint larger than available RAM if its largest component and runtime state fit, but it cannot make arbitrary model sizes practical.

The operating system may retain previously read file pages in its reclaimable file cache. “Released” means the app no longer keeps that weight tensor resident; it does not mean every corresponding byte is forcibly erased from the OS cache. The displayed logical weight-read counter includes reads served by the OS cache and is not a physical SSD traffic measurement. F16/BF16 checkpoint components are converted to F32 during execution; the planner includes this larger working representation.

## Memory and GPU limits

Budgets are monitored limits, **not hard allocator quotas or zero-OOM guarantees**. The pager checks before each component, and a parent watchdog also monitors memory approximately every 350 ms. Allocations, source mappings, drivers and other apps can still create peaks between samples. The largest component plus temporary memory must fit; this version does not split a single tensor into matrix tiles.

RAM telemetry includes the app's process tree. Shared pages may be counted more than once. In browser mode, an independently running browser is outside this accounting. OS file caches are managed by the OS. On-demand GPU use is temporary per component, so the resident GPU/RAM layer counts are both zero and all layers are shown as disk-backed. Smart speed may retain individual tensors in its RAM cache; the separate cached-memory indicator reports these rather than claiming that complete layers are pinned. This differs from the resident GGUF mode's fixed split.

Choose **Inference device** to use CPU, a dedicated GPU, or an integrated GPU. Click **Refresh GPUs** after changing the engine. Resident mode uses llama.cpp device IDs, with GPU capacities reported by the selected engine. The bundled Vulkan engine is preferred when present in `tools/llama-vulkan/`; the original engine remains in `tools/llama.cpp/`. NVIDIA CUDA devices use NVML telemetry and UUID isolation. On-demand mode requires CUDA-enabled PyTorch and rejects a nonzero GPU budget when CUDA is unavailable. Integrated GPU allocations share system RAM, so the resident planner does not subtract their weights from the RAM estimate. Vulkan device usage is an engine discovery snapshot, not live allocator telemetry; the RAM watchdog remains active. CUDA guard accounting uses total device change from the pre-load baseline, so another app's allocations may cause a conservative unload.

## Linux and macOS

Create a virtual environment and install the corresponding PyTorch 2.6 build and `requirements-streaming.txt`. Run `python run.py` from that environment. See pywebview's platform dependencies: https://pywebview.flowrl.com/guide/installation.html . On Linux, a supported GTK/WebKit or Qt runtime is required.

`python run.py --browser` uses the same local backend in a browser. Paste full model paths there; the native file/folder picker is only available in desktop mode. AMD/Intel acceleration is available in Resident mode with a compatible Vulkan llama.cpp build. On-demand mode supports CPU or NVIDIA CUDA. This archive is a source application, not a prebuilt or signed installer. Frozen executable packaging for the new subprocess backend has not been implemented.

## Privacy

Inference stays local. Transformers runs offline with `local_files_only=True`, remote code disabled and telemetry disabled. The API binds only to loopback and requires a random per-launch token; cross-origin requests are rejected. The UI loads no third-party assets. Chats stay in memory and are discarded on exit. Installing dependencies and separately obtaining a model require internet access.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

**37 tests passed** in the Linux development environment. Tiny, randomly initialized real Llama, Qwen2 and Qwen3 networks produced matching CPU logits under resident and streamed loading. Tests also checked autoregressive KV reuse, tied embeddings, zero active weights after execution, bounded retained weight-cache memory, maximum component residency, cancellation, memory rejection and exception cleanup. A real local HTTP worker generated a response using a tiny checkpoint and tokenizer. These tests validate mechanics, not language quality or large-model performance.

The original llama.cpp integration tests still use their clearly labeled fake protocol engine. NVIDIA inference, large trained checkpoints and Windows native rendering were not tested. JavaScript syntax and referenced UI element IDs were checked. See `VERIFICATION.md`.

## Project map

- `streaming_model.py`: header-only checkpoint inspection and working-set estimation.
- `streaming_runtime.py`: actual per-component load/use/release hooks.
- `streaming_worker.py`: isolated local inference server, tokenization and generation.
- `engine.py`: worker lifecycle, monitoring, streaming transport and resident llama.cpp mode.
- `app.py`: authenticated local API and native window/file picker.
- `gguf.py`, `planner.py`: resident GGUF inspection and layer allocation.
- `hardware.py`: RAM/CPU telemetry, GPU inventory and engine device discovery.
- `ui/`: desktop frontend and live paging indicators.
- `tests/`: parser, planner, API, process and actual streaming-inference tests.

All Python modules above are inside `pocketllm/`.

## Upstream references

- Empty model initialization: https://huggingface.co/docs/accelerate/package_reference/big_modeling
- Safetensors selective reads: https://huggingface.co/docs/safetensors/index
- Transformers 4.57.1: https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/model
- PyTorch 2.6 installation variants: https://pytorch.org/get-started/previous-versions/
- Resident llama.cpp server: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

Performance references: https://huggingface.co/docs/transformers/v4.57.1/en/kv_cache .

GPU update verification: the local Intel UHD Graphics 620 and AMD Radeon 550X are exposed as separate Vulkan devices. Use Resident mode with your GGUF model, choose the device, set its graphics memory budget, and inspect again before loading.

## Automatic placement on low-VRAM GPUs

Resident GGUF mode keeps the maximum fitting suffix of transformer layers on the selected GPU. Remaining layers, embeddings/output tensors and KV cache stay in system RAM. If the graphics budget cannot fit one complete layer and runtime overhead, the model automatically uses RAM/CPU instead of refusing to load. The RAM estimate must still fit the user budget. Layers are needed for every token; CPU layers are executed, rather than skipped as unimportant.

On-demand CUDA mode makes a fresh placement decision for each active component using its weights, input size, the GPU budget and current free memory. Smaller eligible operations can use GPU even with less than 512 MiB budget; components exceeding available headroom run on CPU. After an operation, its GPU weights are released. Smart speed retains reusable inactive CPU weights within its bounded RAM cache; uncached weights remain on disk. The live panel reports GPU/CPU operation counts and memory-based RAM fallbacks. This component paging currently requires CUDA; Intel/AMD Vulkan GPUs use the resident layer split.
