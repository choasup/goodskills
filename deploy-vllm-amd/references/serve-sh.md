# serve.sh Template

Copy-paste ready. Replace the `WORK_DIR`, `MODEL_PATH`, and `TRITON_HIP_LLD_PATH` values.

```bash
#!/bin/bash
# ============================================================
# vLLM Inference Server — AMD MI300X (ROCm)
# Model:    Kimi K2.5 / DeepSeek-V3 family (INT4 compressed-tensors)
# Hardware: 8× AMD MI300X, 192GB HBM3 each
# Stack:    vLLM 0.11.0 + ROCm 6.3 + PyTorch 2.8.0+rocm6.3
# ============================================================

# ── Paths ────────────────────────────────────────────────────
WORK_DIR="/home/tione/notebook/research/choasliu/005_claude"
MODEL_PATH="${WORK_DIR}/models/Kimi-K2.5"
LOG_DIR="${WORK_DIR}/logs"
LOG_FILE="${LOG_DIR}/vllm_$(date +%Y%m%d_%H%M%S).log"
PORT=8000
HOST="0.0.0.0"

mkdir -p "${LOG_DIR}"

if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "[ERROR] Model not found at ${MODEL_PATH}"
    exit 1
fi

# ── GPU visibility ───────────────────────────────────────────
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ── Library paths ────────────────────────────────────────────
TORCH_LIB=/usr/local/lib/python3.12/dist-packages/torch/lib
CUDA_LIB=/usr/local/cuda/lib64
CUDA_COMPAT=/usr/local/cuda-12.8/compat       # provides libcuda.so.1
CUBLAS=/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib

export LD_LIBRARY_PATH="${CUDA_COMPAT}:${TORCH_LIB}:${CUDA_LIB}:${CUBLAS}:${LD_LIBRARY_PATH}"

# cuBLAS MUST be preloaded (LD_LIBRARY_PATH alone is insufficient for cublasGemmEx)
export LD_PRELOAD="${CUBLAS}/libcublas.so.12:${CUBLAS}/libcublasLt.so.12"

# Triton AMD linker — find with: find / -name 'ld.lld' 2>/dev/null
# Common when ROCm not system-installed: inside a conda triton package
export TRITON_HIP_LLD_PATH="/path/to/triton/backends/amd/llvm/bin/ld.lld"

echo "======================================================="
echo "  vLLM serving ${MODEL_PATH}"
echo "  Endpoint: http://${HOST}:${PORT}"
echo "  Log: ${LOG_FILE}"
echo "======================================================="

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "Kimi-K2.5" \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 \
    --enable-chunked-prefill \
    --disable-custom-all-reduce \
    --tool-call-parser kimi_k2 \
    --host "${HOST}" \
    --port "${PORT}" \
    2>&1 | tee "${LOG_FILE}"
```

## Flag Notes

| Flag | Reason |
|------|--------|
| `--disable-custom-all-reduce` | CUDA custom allreduce crashes on AMD; RCCL handles it |
| `--dtype bfloat16` | MI300X native, avoids FP16 overflow on large MoE |
| `--trust-remote-code` | Kimi K2.5 config/tokenizer uses custom code |
| `--tool-call-parser kimi_k2` | Enables Kimi K2.5's tool-call format parsing |
| `--enable-chunked-prefill` | Better throughput for long-context requests |
| `--gpu-memory-utilization 0.90` | Leave 10% headroom; INT4 model fits well within 90% |
| `--max-model-len 65536` | 64K context; increase to 131072 if needed (uses more KV cache) |

## Starting in Background

```bash
nohup bash scripts/serve.sh > /tmp/vllm_out.log 2>&1 &
echo "PID=$!"

# Watch for errors
tail -f /tmp/vllm_out.log | grep -E 'ERROR|Completed|Uvicorn|health'

# Poll until ready
while ! curl -sf http://localhost:8000/health > /dev/null; do
    echo "waiting..."; sleep 30
done
echo "API is up!"
```
