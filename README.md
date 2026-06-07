# qwen3.6-edge

A patched fork of [llama.cpp](https://github.com/ggerganov/llama.cpp) via [EsmaeelNabil/llama.cpp-mtp-turbo-quant](https://github.com/EsmaeelNabil/llama.cpp-mtp-turbo-quant), enabling Qwen3.6-35B-A3B (MTP variant) to run at full 128K context on consumer hardware.

## Quick Start

```bash
# Docker (NVIDIA)
docker compose up --build
curl http://localhost:8081/v1/models

# Docker (AMD ROCm)
docker compose -f docker-compose-rocm.yml up --build

# Build from source
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_CUDA=ON
cmake --build build --target llama-server -j$(nproc)

# Run
./build/bin/llama-server \
  -m /path/to/Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf \
  -ngl 99 --n-cpu-moe 22 -c 131072 --no-mmap \
  --cache-type-k turbo3 --cache-type-v turbo3 \
  --spec-type mtp --spec-draft-n-max 2 --spec-draft-n-cpu-moe 0 \
  --host 0.0.0.0 --port 8081
```

---

## Patches on Top of Base Fork

### Patch 1: sm_120 FlashAttention Occupancy Fallback

**File:** `ggml/src/ggml-cuda/fattn-common.cuh:1425-1435`

`cudaOccupancyMaxActiveBlocksPerMultiprocessor` returns `cudaErrorSharedObjectInitFailed` spuriously on sm_120 (Blackwell) when shared memory usage is zero. This crashes the FlashAttention kernel on RTX 5060 Ti.

The fix catches the failed occupancy query and falls back to the kernel's `__launch_bounds__(128, 2)` annotation, which provides a safe occupancy estimate of 2 blocks per SM.

### Patch 2: MTP Head Tensor Loading

**Files:** `src/models/qwen35moe-mtp.cpp`, `src/models/models.h`

Qwen3.6-35B-A3B has `n_layer=41` (40 trunk layers + 1 MTP head). The upstream MTP model loader iterated over all 41 layers, then tried to load trunk-only tensors using indices intended for the draft head, causing buffer allocation failures.

The fix tracks the original `n_layer` (41) in `mtp_trunk_n_layer`. In `load_arch_tensors`, it skips trunk layer indices. For the MTP block, `tensor_il = i` so GGUF tensor names match the model's block indices.

### Patch 3: MTP Head Q+Gate Fused Tensor

**File:** `src/models/qwen35moe-mtp.cpp:148-170`

Qwen3.6's attention uses an "output gate" — the Q projection's output is `2 * n_head * head_dim` (Q + sigmoid gate interleaved). The upstream MTP loader was creating Q with the wrong dimension, causing shape mismatches in the attention graph.

The fix creates Q with the correct fused dimension (`n_embd_head * n_head * 2`), splits Q and gate via `ggml_view_3d` with a stride of `n_embd_head * 2`, and applies `sigmoid(gate) * attn_out` before the output projection, matching the trunk's full-attention layers.

### Patch 4: split_sum Guard

**File:** `src/llama-model.cpp:1228-1240`

When loading a second model after the first has consumed all GPU memory, device memory queries return 0, causing `split_sum` to be zero/NaN. This leads to division-by-zero in split normalization.

The fix guards against invalid `split_sum` with `if (!(split_sum > 0.0f))`, falling back to all-on-one-device.

### Patch 5: layer_gpu OOB Clamp

**File:** `src/llama-model.cpp:1249-1251`

`std::upper_bound` can return an index past `devices.size() - 1` in edge cases. The subsequent `devices.at(layer_gpu)` throws an out-of-bounds exception.

The fix clamps `layer_gpu` to valid range: `std::min((int)(devices.size() - 1), std::max(0, layer_gpu_raw))`.

---

## Docker

### NVIDIA (CUDA)

`Dockerfile` builds a multi-stage image:
1. **Builder stage** — clones this repo, compiles `llama-server` with CUDA 13.2 (sm_120 target)
2. **Runtime stage** — Ubuntu 24.04 base with CUDA runtime, copies the binary, installs Python + hermes-agent

`docker-compose.yml` mounts `../models` as a volume and exposes port 8081.

### AMD (ROCm)

`Dockerfile.rocm` follows the same multi-stage pattern but targets ROCm 6.2.4. Supports all RDNA 3 (RX 7000 series) and CDNA 2/3 (MI210, MI250, MI300X) architectures via the `AMDGPU_TARGETS` build arg.

`docker-compose-rocm.yml` adds `/dev/kfd` and `/dev/dri` device mappings with `video` and `render` group access.

### Environment Variables

Both Dockerfiles accept these environment variables (set via `docker-compose.yml` or `-e`):

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `/models/Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf` | Model file path |
| `PORT` | `8081` | Server port |
| `HOST` | `0.0.0.0` | Bind address |
| `CTX_SIZE` | `131072` | Context window size |
| `N_CPU_MOE` | `22` | Number of MoE layers on CPU |
| `THREADS` | `8` | CPU threads |
| `PARALLEL` | `1` | Parallel requests |
| `CACHE_TYPE_K` | `turbo3` | KV cache type for keys |
| `CACHE_TYPE_V` | `turbo3` | KV cache type for values |
| `SPEC_DRAFT_N_MAX` | `2` | Max MTP draft tokens |
| `NO_WARMUP` | `true` | Skip server warmup |

---

## Upstream Forks

| Repo | What it provides |
|---|---|
| [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) | Upstream llama.cpp |
| [EsmaeelNabil/llama.cpp-mtp-turbo-quant](https://github.com/EsmaeelNabil/llama.cpp-mtp-turbo-quant) | TurboQuant KV compression + MTP speculative decoding |
| [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) | Q4_K_M GGUF model |

---

# llama.cpp

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp)](https://github.com/ggml-org/llama.cpp/releases)
[![Server](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)

[Manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md)

LLM inference in C/C++

## Recent API changes

- [Changelog for `libllama` API](https://github.com/ggml-org/llama.cpp/issues/9289)
- [Changelog for `llama-server` REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

## Hot topics

- **Hugging Face cache migration: models downloaded with `-hf` are now stored in the standard Hugging Face cache directory, enabling sharing with other HF tools.**
- **[guide : using the new WebUI of llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/16938)**
- [guide : running gpt-oss with llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/15396)
- [[FEEDBACK] Better packaging for llama.cpp to support downstream consumers](https://github.com/ggml-org/llama.cpp/discussions/15313)
- Support for the `gpt-oss` model with native MXFP4 format has been added | [PR](https://github.com/ggml-org/llama.cpp/pull/15091) | [Collaboration with NVIDIA](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss) | [Comment](https://github.com/ggml-org/llama.cpp/discussions/15095)
- Multimodal support arrived in `llama-server`: [#12898](https://github.com/ggml-org/llama.cpp/pull/12898) | [documentation](./docs/multimodal.md)
- VS Code extension for FIM completions: https://github.com/ggml-org/llama.vscode
- Vim/Neovim plugin for FIM completions: https://github.com/ggml-org/llama.vim
- Hugging Face Inference Endpoints now support GGUF out of the box! https://github.com/ggml-org/llama.cpp/discussions/9669
- Hugging Face GGUF editor: [discussion](https://github.com/ggml-org/llama.cpp/discussions/9268) | [tool](https://huggingface.co/spaces/CISCai/gguf-editor)

----

## Quick start (upstream)

Getting started with llama.cpp is straightforward. Here are several ways to install it on your machine:

- Install `llama.cpp` using [brew, nix or winget](docs/install.md)
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed, you'll need a model to work with. Head to the [Obtaining and quantizing models](#obtaining-and-quantizing-models) section to learn more.

Example command:

```sh
# Use a local model file
llama-cli -m my_model.gguf

# Or download and run a model directly from Hugging Face
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF

# Launch OpenAI-compatible API server
llama-server -hf ggml-org/gemma-3-1b-it-GGUF
```

## Description

The main goal of `llama.cpp` is to enable LLM inference with minimal setup and state-of-the-art performance on a wide
range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is the main playground for developing new features for the [ggml](https://github.com/ggml-org/ggml) library.

<details>
<summary>Models</summary>

Typically finetunes of the base models below are supported as well.

Instructions for adding support for new models: [HOWTO-add-model.md](docs/development/HOWTO-add-model.md)

#### Text-only
