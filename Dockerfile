# ── qwen3.6-edge ──────────────────────────────────────────────────────────────
# Qwen3.6-35B-A3B Q4_K_M at 54+ tok/s on RTX 5060 Ti 16GB
#
# Multi-stage build:
#   Stage 1 (builder): compile llama-server from our patched llama.cpp fork
#   Stage 2 (runtime): slim CUDA runtime + binary + startup script
#
# Build:
#   docker build -t qwen3.6-edge .
#
# Run (requires nvidia-docker / --gpus all):
#   docker run --gpus all -p 8081:8081 \
#     -v C:\llama\models:/models:ro \
#     qwen3.6-edge
#
# The model file must be pre-downloaded into the host directory you mount
# at /models.  The default filename is:
#   /models/Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf
#
# To download the model into that directory before starting the container:
#   pip install huggingface_hub
#   huggingface-cli download unsloth/Qwen3.6-35B-A3B-MTP-GGUF \
#     Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf \
#     --local-dir C:\llama\models

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1: Build llama-server from source
# ═══════════════════════════════════════════════════════════════════════════════
FROM nvidia/cuda:13.2.0-devel-ubuntu24.04 AS builder

# Prevent interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# Build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    ca-certificates \
    pkg-config \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone our patched fork (main branch)
RUN git clone --depth 1 --branch main \
    https://github.com/lhqezio/qwen3.6-edge.git /src

WORKDIR /src

# Build llama-server — targeting RTX 5060 Ti (sm_120)
# -DCMAKE_CUDA_ARCHITECTURES=120: Blackwell architecture
# -DGGML_CUDA=ON: enable CUDA backend
# -DCMAKE_BUILD_TYPE=Release: optimized build
# -DGGML_NATIVE=OFF: don't use host CPU optimizations (Docker compatibility)
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=OFF \
    -DLLAMA_CURL=ON \
    && cmake --build build --target llama-server -j$(nproc)

# Verify the binary exists
RUN ls -la build/bin/llama-server

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2: Runtime image
# ═══════════════════════════════════════════════════════════════════════════════
FROM nvidia/cuda:13.2.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Runtime dependencies
# - libgomp1: OpenMP (required by llama.cpp)
# - libcurl4: HTTP client for model downloads via --model-url (if supported)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libcurl4 \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Copy the compiled binary from builder
COPY --from=builder /src/build/bin/llama-server /usr/local/bin/llama-server

# Model directory — mount your pre-downloaded GGUF here
RUN mkdir -p /models
VOLUME /models

# Working directory
WORKDIR /app

# Copy startup script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Default model path (override with -e MODEL=/models/other.gguf)
ENV MODEL=/models/Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf
ENV PORT=8081
ENV HOST=0.0.0.0
ENV CTX_SIZE=131072
ENV N_CPU_MOE=22
ENV PARALLEL=1
ENV THREADS=8
ENV SPEC_DRAFT_N_MAX=2
ENV SPEC_DRAFT_N_CPU_MOE=0
ENV CACHE_TYPE_K=turbo3
ENV CACHE_TYPE_V=turbo3
ENV NO_WARMUP=true

# llama-server port
EXPOSE 8081

ENTRYPOINT ["/app/entrypoint.sh"]
