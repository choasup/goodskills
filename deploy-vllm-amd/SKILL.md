---
name: deploy-vllm-amd
description: |
  Complete runbook for deploying large language models (Kimi K2.5, DeepSeek-V3 family,
  INT4 compressed-tensors quantized) on AMD MI300X GPU servers using vLLM + ROCm.
  Covers the full pipeline: environment setup, 9 known compatibility patches, model
  adapter registration, serve.sh configuration, and OpenAI-compatible API usage.

  Use this skill whenever the user is:
  - Deploying vLLM on AMD GPUs (MI300X, MI250, CDNA architecture)
  - Hitting vLLM + ROCm compatibility errors (CUDA symbol mismatches, Marlin kernel failures, flashinfer hangs, missing linker)
  - Setting up Kimi K2.5 / DeepSeek-V3 / MoE models with INT4 compressed-tensors on AMD
  - Writing or debugging a serve.sh for vLLM on ROCm
  - Getting "libcuda.so.1 not found", "libtorch_cuda.so not found", "cublasGemmEx undefined", "max_shared_mem" errors
  - Setting up an OpenAI-compatible inference endpoint on AMD hardware
---

# Deploy vLLM on AMD MI300X (ROCm)

Deployment guide for **Kimi K2.5 / DeepSeek-V3 family** (INT4 compressed-tensors) on
**8× AMD MI300X** using **vLLM 0.11.0 + ROCm 6.3 + PyTorch 2.8.0+rocm6.3**.

> The core challenge: vLLM was compiled against CUDA-named PyTorch, but this environment
> has ROCm PyTorch (HIP-named symbols). Multiple layers of patching are required.

---

## Quick-Start Checklist

Run these steps in order on a fresh environment. Each is idempotent.

### Step 1 — Install missing Python deps
```bash
pip install blobfile          # needed by tiktoken tokenizer
```

### Step 2 — Compile CUDA→HIP symbol shim
vLLM's `_C.abi3.so` expects `c10::cuda::` symbols; ROCm PyTorch only exports `c10::hip::`.
See **`references/cuda-hip-shim.md`** for the full C source and explanation.

```bash
TORCH_LIB=/usr/local/lib/python3.12/dist-packages/torch/lib

cat > /tmp/cuda_hip_shim.c << 'CEOF'
#define _GNU_SOURCE
#include <stdint.h>
#include <string.h>
#include <dlfcn.h>

char _cuda_allocator_buf[8]
    __asm__("_ZN3c104cuda20CUDACachingAllocator9allocatorE")
    __attribute__((visibility("default")));

uint64_t __attribute__((visibility("default")))
_ZN3c104cuda20getCurrentCUDAStreamEa(signed char d) {
    typedef uint64_t (*fn_t)(signed char);
    static fn_t fn = NULL;
    if (!fn) {
        void* lib = dlopen("libc10_hip.so", RTLD_LAZY | RTLD_NOLOAD);
        if (!lib) lib = dlopen("libc10_hip.so", RTLD_LAZY | RTLD_GLOBAL);
        if (lib) fn = (fn_t)dlsym(lib, "_ZN3c103hip19getCurrentHIPStreamEa");
    }
    return fn ? fn(d) : 0;
}

__attribute__((constructor(101))) static void cuda_hip_compat_init(void) {
    void* lib = dlopen("libc10_hip.so", RTLD_LAZY | RTLD_NOLOAD);
    if (!lib) lib = dlopen("libc10_hip.so", RTLD_LAZY | RTLD_GLOBAL);
    if (lib) {
        void* hip_alloc = dlsym(lib, "_ZN3c103hip19HIPCachingAllocator9allocatorE");
        if (hip_alloc) memcpy(_cuda_allocator_buf, hip_alloc, 8);
    }
}
CEOF

gcc -shared -fPIC -o ${TORCH_LIB}/libc10_cuda.so /tmp/cuda_hip_shim.c \
    -L${TORCH_LIB} -lc10_hip -ldl \
    -Wl,-rpath,${TORCH_LIB} -Wl,--allow-shlib-undefined
echo "✓ libc10_cuda.so compiled"
```

### Step 3 — Create libtorch_cuda.so symlink
```bash
TORCH_LIB=/usr/local/lib/python3.12/dist-packages/torch/lib
ln -sf ${TORCH_LIB}/libtorch_hip.so ${TORCH_LIB}/libtorch_cuda.so
echo "✓ libtorch_cuda.so → libtorch_hip.so"
```

### Step 4 — Patch Marlin kernel (disable on ROCm)
Marlin is a CUDA-only kernel; on AMD it crashes with `Expected max_shared_mem > 0`.
```bash
python3 - << 'PYEOF'
path = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/marlin_utils.py'
with open(path, 'r') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if 'def check_moe_marlin_supports_layer' in line:
        rocm_check = '    from vllm.platforms import current_platform\n    if current_platform.is_rocm():\n        return False\n'
        lines.insert(i+2, rocm_check)
        print(f"✓ Patched marlin_utils.py at line {i}")
        break
with open(path, 'w') as f:
    f.writelines(lines)
PYEOF
```
This makes vLLM fall back to `CompressedTensorsWNA16MoEMethod` (Triton-based, ROCm compatible).

### Step 5 — Patch flashinfer autotuner (skip on ROCm)
AMD MI300X reports `has_device_capability(90) = True`, which incorrectly triggers
flashinfer's CUDA JIT autotuner. It hangs indefinitely, causing NCCL timeout.
```bash
python3 - << 'PYEOF'
path = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/warmup/kernel_warmup.py'
with open(path, 'r') as f:
    content = f.read()
old = 'has_flashinfer() and current_platform.has_device_capability(90):'
new = 'has_flashinfer() and current_platform.has_device_capability(90) and not current_platform.is_rocm():'
if old in content:
    with open(path, 'w') as f:
        f.write(content.replace(old, new))
    print("✓ Patched kernel_warmup.py")
else:
    print("Already patched or line not found")
PYEOF
```

### Step 6 — Find the ROCm linker
Triton's AMD backend needs `ld.lld`. If `/opt/rocm/llvm/bin/ld.lld` doesn't exist:
```bash
find / -name 'ld.lld' 2>/dev/null
# Common location when ROCm is not system-installed:
# /path/to/miniconda/envs/<env>/lib/python3.x/site-packages/triton/backends/amd/llvm/bin/ld.lld
```
Set `TRITON_HIP_LLD_PATH` to the found path in your serve.sh (see Step 8).

### Step 7 — Register Kimi K2.5 model adapter
Only needed if deploying `KimiK25ForConditionalGeneration` (Kimi K2.5).
See **`references/model-adapter.md`** for full code.

**Create the adapter** at:
`/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/kimi_k25.py`

**Register it** in:
`/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/registry.py`
Add this line to the model dict (around line 244):
```python
"KimiK25ForConditionalGeneration": ("kimi_k25", "KimiK25ForConditionalGeneration"),
```

### Step 8 — Write serve.sh
See **`references/serve-sh.md`** for the full template. Key environment variables:

```bash
# Library paths
TORCH_LIB=/usr/local/lib/python3.12/dist-packages/torch/lib
CUDA_COMPAT=/usr/local/cuda-12.8/compat           # provides libcuda.so.1
CUBLAS=/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib

export LD_LIBRARY_PATH="${CUDA_COMPAT}:${TORCH_LIB}:${CUDA_LIB}:${CUBLAS}:${LD_LIBRARY_PATH}"
export LD_PRELOAD="${CUBLAS}/libcublas.so.12:${CUBLAS}/libcublasLt.so.12"  # must preload, not just in path
export TRITON_HIP_LLD_PATH="/path/to/ld.lld"       # from Step 6
```

Key vLLM flags for AMD:
```
--tensor-parallel-size 8          # one per GPU
--disable-custom-all-reduce        # CUDA custom allreduce crashes on AMD
--dtype bfloat16
--trust-remote-code
```

### Step 9 — Launch and verify
```bash
# Start in background
nohup bash scripts/serve.sh > /tmp/vllm_out.log 2>&1 &

# Monitor (loading takes ~60 min for 555GB model)
tail -f /tmp/vllm_out.log | grep -E 'shard|ERROR|Uvicorn|health'

# Health check (ready when this returns {"status":"ok"})
curl http://localhost:8000/health
```

---

## Expected Timeline

| Phase | Duration | Log signal |
|-------|----------|------------|
| Worker init + model graph | ~1 min | `Using Triton MLA backend`, `Using CompressedTensorsWNA16MoEMethod` |
| Weight loading (64 shards) | ~60 min | `Loading safetensors checkpoint shards: X%` |
| torch.compile | ~60 sec | `torch.compile takes 59.xx s` |
| KV cache sizing | instant | `GPU KV cache size: 1,43x,xxx tokens` |
| API ready | — | `Uvicorn running on http://0.0.0.0:8000` |

---

## Error → Fix Quick Reference

| Error message | Fix |
|---------------|-----|
| `libcuda.so.1: cannot open shared object file` | Add `/usr/local/cuda-12.8/compat` to `LD_LIBRARY_PATH` |
| `libtorch_cuda.so: cannot open shared object file` | Step 3: symlink `libtorch_cuda.so → libtorch_hip.so` |
| `undefined symbol: cublasGemmEx` | `LD_PRELOAD` cublas + cublasLt (not just LD_LIBRARY_PATH) |
| `Model architectures ['KimiK25...'] are not supported` | Step 7: create adapter + register in registry.py |
| `blobfile is not installed` | `pip install blobfile` |
| `allocate_shared_buffer_and_handle` crash | Add `--disable-custom-all-reduce` flag |
| `Expected max_shared_mem > 0 to be true` | Step 4: patch marlin_utils.py |
| `ROCm linker /opt/rocm/llvm/bin/ld.lld not found` | Step 6: set `TRITON_HIP_LLD_PATH` |
| `flashinfer.jit: Autotuning process starts` then hangs | Step 5: patch kernel_warmup.py |
| `RuntimeError: cancelled` (after autotuner hang) | Same as above — NCCL timeout caused by flashinfer hang |

---

## API Usage After Deployment

```bash
# Health
curl http://SERVER:8000/health
# → {"status":"ok"}

# List models
curl http://SERVER:8000/v1/models

# Chat completion (curl)
curl http://SERVER:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Kimi-K2.5",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 512,
    "temperature": 0.6
  }'
```

```python
# Python (OpenAI client)
from openai import OpenAI
client = OpenAI(base_url="http://SERVER:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="Kimi-K2.5",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=512,
    temperature=0.6,
)
print(response.choices[0].message.content)
```

---

## Reference Files

- **`references/cuda-hip-shim.md`** — Full C source for the CUDA→HIP shim + technical explanation
- **`references/model-adapter.md`** — Full kimi_k25.py adapter code + registry patch
- **`references/serve-sh.md`** — Complete serve.sh template (copy-paste ready)
