# CUDA→HIP Symbol Shim (libc10_cuda.so)

## Why This Is Needed

vLLM 0.11.0 was compiled against **CUDA PyTorch**, so its `_C.abi3.so` extension
references symbols in the `c10::cuda::` namespace:
- `_ZN3c104cuda20CUDACachingAllocator9allocatorE` (data symbol)
- `_ZN3c104cuda20getCurrentCUDAStreamEa` (function)

ROCm PyTorch exports these same concepts under `c10::hip::`:
- `_ZN3c103hip19HIPCachingAllocator9allocatorE`
- `_ZN3c103hip19getCurrentHIPStreamEa`

The shim library provides CUDA-named symbols that forward to HIP implementations,
allowing vLLM's C extension to load without modification.

## Full C Source

```c
#define _GNU_SOURCE
#include <stdint.h>
#include <string.h>
#include <dlfcn.h>

/* ── Data symbol: CUDACachingAllocator::allocator ─────────────────────────
   The __asm__ attribute gives the variable the mangled CUDA name at link time.
   The constructor (below) copies the actual HIP allocator pointer into it.    */
char _cuda_allocator_buf[8]
    __asm__("_ZN3c104cuda20CUDACachingAllocator9allocatorE")
    __attribute__((visibility("default")));

/* ── Function symbol: getCurrentCUDAStream ────────────────────────────── */
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

/* ── Constructor: copy HIP allocator pointer into the CUDA-named buffer ── */
__attribute__((constructor(101))) static void cuda_hip_compat_init(void) {
    void* lib = dlopen("libc10_hip.so", RTLD_LAZY | RTLD_NOLOAD);
    if (!lib) lib = dlopen("libc10_hip.so", RTLD_LAZY | RTLD_GLOBAL);
    if (lib) {
        void* hip_alloc = dlsym(lib,
            "_ZN3c103hip19HIPCachingAllocator9allocatorE");
        if (hip_alloc) memcpy(_cuda_allocator_buf, hip_alloc, 8);
    }
}
```

## Compilation Command

```bash
TORCH_LIB=/usr/local/lib/python3.12/dist-packages/torch/lib

gcc -shared -fPIC -o ${TORCH_LIB}/libc10_cuda.so /tmp/cuda_hip_shim.c \
    -L${TORCH_LIB} -lc10_hip -ldl \
    -Wl,-rpath,${TORCH_LIB} \
    -Wl,--allow-shlib-undefined
```

**Important notes:**
- `__asm__` attribute MUST come before `__attribute__((visibility(...)))` in the
  variable declaration, otherwise GCC reports a syntax error
- `-Wl,--allow-shlib-undefined` is needed because `libc10_hip.so` may not be fully
  resolved at shim compile time
- The shim must live in `${TORCH_LIB}` so it's found via the rpath

## Verification

```bash
# Confirm the CUDA-named symbols are exported
nm -D ${TORCH_LIB}/libc10_cuda.so | grep cuda

# Test vllm._C imports cleanly (requires LD_PRELOAD for cuBLAS too — see serve.sh)
LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat:${TORCH_LIB} \
  python3 -c "import vllm._C; print('OK')"
```
