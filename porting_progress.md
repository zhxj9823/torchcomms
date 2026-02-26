# NCCL CPU-Emulation Port: Progress & Change Analysis

This document describes the porting work done to run NCCL's device-side collective
algorithms on a CPU emulator (the NEX/accvm GPU-emulation stack).  It compares two
parallel tracks:

- **nccl/** — a standalone upstream NCCL (v2.22-ish) with CPU-emulation patches applied
  directly.  Base commit `d07cb69` ("Add the original nccl library"), HEAD `f1bd0a8`.
- **torchcomms/comms/ncclx/v2_27/** — the Meta `ncclx` fork (v2.27) with the same
  CPU-emulation strategy already baked in (the reference implementation used to guide
  the nccl port).

Both compile NCCL's device-side C++/CUDA code with **g++ instead of nvcc**, emulating
the GPU thread model using **Boost.Fiber** cooperative fibers.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [The Fiber-Based GPU Thread Emulator (`cuda_emulator.hh`)](#2-the-fiber-based-gpu-thread-emulator)
3. [Build System Changes](#3-build-system-changes)
4. [Device Shared Memory: `__shared__` → `thread_local shared_memory<T>`](#4-device-shared-memory)
5. [CUDA Intrinsics & Built-ins Replaced](#5-cuda-intrinsics--built-ins-replaced)
6. [Collective Algorithm Files (`all_reduce`, `all_gather`, etc.)](#6-collective-algorithm-files)
7. [Host-Side / Infrastructure Changes](#7-host-side--infrastructure-changes)
8. [ncclx-Exclusive Features Not Yet in nccl Port](#8-ncclx-exclusive-features-not-yet-in-nccl-port)
9. [Runtime Environment: `run.mk` vs `run_ncclx.mk`](#9-runtime-environment)
10. [Commit-by-Commit History (nccl port)](#10-commit-by-commit-history)
11. [Remaining Gaps & Next Steps](#11-remaining-gaps--next-steps)
12. [File-by-File Comparison: nccl port vs ncclx/v2_27](#12-file-by-file-comparison-nccl-port-vs-ncclxv2_27)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  nccl-test binary (all_reduce_perf, etc.)               │
│  ↓  dlopen  libnccl.so                                  │
├─────────────────────────────────────────────────────────┤
│  libnccl.so                                             │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Host-side NCCL  (init, channels, transport,    │    │
│  │  enqueue, proxy threads)                        │    │
│  │  compiled normally with g++                     │    │
│  ├─────────────────────────────────────────────────┤    │
│  │  Device-side algorithms (all_reduce.h, etc.)   │    │
│  │  compiled as C++ with g++ via cuda_emulator.hh │    │
│  │  Each "kernel launch" → CudaThread fibers       │    │
│  └─────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────┤
│  NEX launcher  (./nex <cmd>)                            │
│    injects accvm.so + nex_cuda.so via LD_PRELOAD        │
│    intercepts CUDA API calls → emulated GPU             │
└─────────────────────────────────────────────────────────┘
```

The ported nccl and ncclx v2.27 follow the **same strategy**.  The main structural
difference is that ncclx v2.27 additionally carries the full ctran extension
(custom transport algorithms) and Meta-internal features (sparse all-reduce, RAS, etc.).

---

## 2. The Fiber-Based GPU Thread Emulator

**File:** `src/include/cuda_emulator.hh`  
*(Identical copy in both `nccl/src/include/` and `torchcomms/comms/ncclx/v2_27/src/include/`)*

### Design

Each "CUDA kernel" is executed as a set of **Boost.Fiber cooperatively-scheduled
fibers**, one per CUDA thread.  Each fiber stores its thread identity in a
fiber-local `ThreadContext`:

```cpp
struct ThreadContext {
    dim3 threadIdx;   // this fiber's simulated thread index
    dim3 blockIdx;
    dim3 blockDim;
    dim3 gridDim;
};

// accessed via fiber-local storage:
extern thread_local
  boost::fibers::fiber_specific_ptr<ThreadContext> current_thread_ctx;
```

### Emulated CUDA intrinsics

| CUDA built-in | Emulator implementation |
|---|---|
| `threadIdx()` / `blockIdx()` / … | Read from `current_thread_ctx` |
| `__syncthreads()` | Fiber barrier — all fibers in the block must arrive |
| `barrier_sync(name)` | Named barrier using `NamedBarrierManager` |
| `barrier_red_or(vote, name)` | Named barrier with OR reduction |
| `__ballot_sync(mask, pred)` | Collects predicate across active fibers |
| `__any_sync(mask, pred)` | Reduction across active fibers |
| `__syncwarp()` | Warp-level fiber sync |
| `__threadfence()` / `__threadfence_block()` | `std::atomic_thread_fence` |
| `coop_thread_yield(spins)` | `boost::this_fiber::yield()` |

### Shared memory emulation

`shared_memory<T>` is a heap-allocated wrapper:
- Accessed via `operator->()` and `operator+()` (pointer arithmetic on the raw buffer)
- Stored as `thread_local` to give each "kernel invocation" its own instance
- This is why every `ncclShmem.field` in the original CUDA code became `ncclShmem->field`
  after the port

---

## 3. Build System Changes

### 3a. `src/device/Makefile`

This is the most important build change — device code is now compiled with **g++** instead of **nvcc**.

| Aspect | Original (CUDA) | nccl port | ncclx v2.27 |
|---|---|---|---|
| Compiler | `nvcc` | `$(CXX)` (g++) | `$(CXX)` (g++) |
| C++ standard | implicit | `-std=c++17` | `-std=c++2a` |
| `.cu` → `.cc` | `.cu` files | renamed to `.cc` | both `.cu` and `.cc` kept |
| Device link | `nvcc -dlink` | `$(CXX) -r` (relocatable) | `$(CXX) -r` |
| `MANIFEST` dep | `LIB_OBJS + DEVGLUE_OBJ` | `LIB_OBJS` only | `LIB_OBJS` only |
| `COMPILE_SYM` | `nvcc $(NVCUFLAGS_SYM)` | `$(call COMPILE$(suffix $2),…)` | same as nccl port |
| Codegen extra | `generate.py` only | `generate.py` only | `generate.py` + `genctran.py` |
| Extra SRCS | `common.cu onerank.cu` | `common.cc onerank.cc` | `+ ctran_cpu_stubs.cc`, `+ all_reduce_sparse_block.cc` |
| CUDART version | from nvcc | not set | `-DCUDART_VERSION=12010` |
| Extra defines | — | — | `-DGLOG_USE_GLOG_EXPORT -DMOCK_SCUBA_DATA` |

**nccl port diff (key lines):**
```makefile
# Before:
CXXFLAGS += $(INCFLAGS)
SRCS = common.cu onerank.cu
$(DEVGLUE_OBJ): $(LIB_OBJS)
    $(NVCC) $(NVCUFLAGS) -dlink $^ -o $@
$(MANIFEST): $(LIB_OBJS) $(DEVGLUE_OBJ)

# After:
CXXFLAGS += $(INCFLAGS) -O3 -std=c++17 -Wno-unknown-pragmas -fvisibility=hidden \
            -Wno-reorder -Wno-uninitialized -Wno-comment -Wno-unused-variable \
            -Wno-unused-but-set-variable -Wno-unused-local-typedefs -Wno-int-in-bool-context
SRCS = common.cc onerank.cc
$(DEVGLUE_OBJ): $(LIB_OBJS)
    $(CXX) $(CXXFLAGS) -r $^ -o $@
$(MANIFEST): $(LIB_OBJS)
```

### 3b. `src/device/generate.py`

All generated file extensions changed from `.cu` to `.cc`:

```diff
-with open(os.path.join(gensrc, "device_table.cu"), "w") as f:
+with open(os.path.join(gensrc, "device_table.cc"), "w") as f:

-def impl_filename(...): return "%s.cu" % paste(...)
+def impl_filename(...): return "%s.cc" % paste(...)

-names = impl_names + ["host_table.cc", "device_table.cu"]
+names = impl_names + ["host_table.cc", "device_table.cc"]
```

ncclx v2.27 has the same changes plus the `genctran.py` call which generates ctran
algorithm CPU stubs — this step does not exist in the standalone nccl port.

### 3c. Top-level `src/Makefile`

```diff
# Added CUDA stub lib for link-time (no real CUDA driver at build time):
-LDFLAGS += -L${CUDA_LIB} -l$(CUDARTLIB) -lcuda -lpthread -lrt -ldl
+LDFLAGS += -L$(CUDA_STUB_LIB) -lcuda -L${CUDA_LIB} -l$(CUDARTLIB) -lpthread -lrt -ldl

# New: optional mock topology support
+ifeq ($(MOCK_TOPOLOGY), 1)
+  NCCL_CPPFLAGS += -DNCCL_MOCK_TOPOLOGY
+  LIBSRCFILES += misc/mock_syscalls.cc
+endif
```

---

## 4. Device Shared Memory

This is the **single most pervasive change** — touching every collective algorithm file.

### Original CUDA

```cpp
// Declared as GPU shared memory:
extern __shared__ ncclShmemData ncclShmem;
extern __shared__ ulong2 ncclShmemPerWarp[...];

// Accessed as struct member:
ncclRing *ring = &ncclShmem.channel.ring;
const int nranks = ncclShmem.comm.nRanks;
ncclShmem.aborted = 1;
```

### After port (both nccl and ncclx v2.27)

```cpp
// Declared as thread-local heap object:
extern thread_local shared_memory<ncclShmemData> ncclShmem;
extern thread_local shared_memory<ulong2> ncclShmemPerWarp;

// Accessed via pointer dereference:
ncclRing *ring = &ncclShmem->channel.ring;
const int nranks = ncclShmem->comm.nRanks;
ncclShmem->aborted = 1;
```

The `__shared__` macro is redefined:
```cpp
#undef __shared__
#define __shared__ thread_local
```

And `ncclScratchForWarp` pointer arithmetic is fixed for the wrapper type:
```cpp
// Before:
return (char*)ncclShmemPerWarp + warp*ncclShmemScratchWarpSize();
// After:
return (char*)(ncclShmemPerWarp + warp*ncclShmemScratchWarpSize());
```

**Definition moved from `common.cu` → `common.cc`:**
```cpp
// nccl port: common.cc (new file)
thread_local shared_memory<ncclShmemData> ncclShmem(sizeof(ncclShmemData));
thread_local shared_memory<ulong2> ncclShmemPerWarp(
    ncclShmemScratchWarpSize() * (NCCL_MAX_NTHREADS/WARP_SIZE)
);
```

---

## 5. CUDA Intrinsics & Built-ins Replaced

### 5a. `src/device/common.h` — CPU shim header

The guard `NCCL_DEVICE_COMMON_H_` is renamed to `NCCL_DEVICE_COMMON_CPU_H_` to
signal this is the CPU variant.

**New includes:**
```cpp
#include <cstdint>
#include <cstring>
#include <chrono>
#include <thread>
#include <atomic>
#include <cassert>
#include <threads.h>
#include "cuda_emulator.hh"
```

**CUDA qualifiers stubbed out:**
```cpp
#ifndef __device__    #define __device__
#ifndef __host__      #define __host__
#ifndef __global__    #define __global__
#ifndef __launch_bounds__(m,b)  (no-op)
#ifndef __forceinline__         #define __forceinline__
```

**Thread-index functions declared** (implemented in `cuda_emulator.hh`):
```cpp
extern dim3 threadIdx();
extern dim3 blockIdx();
extern dim3 blockDim();
extern dim3 gridDim();
extern void __syncthreads();
extern void __syncwarp();
extern void barrier_sync(int);
extern void barrier_sync(int, int);
extern bool barrier_red_or(bool, int);
extern bool barrier_red_or(bool, int, int);
extern int  __any_sync(unsigned, int);
extern unsigned __ballot_sync(unsigned, int);
extern void __threadfence_block();
extern void __threadfence_system();
extern void __threadfence();
extern void coop_thread_yield(int spins);
```

**CPU fallbacks for GPU intrinsics:**
```cpp
inline void __trap() { assert(0); }
inline int  __popc(uint x) { return __builtin_popcount(x); }
inline uint64_t atomicMax(uint64_t *addr, uint64_t val) {
    std::atomic<uint64_t>* a = reinterpret_cast<std::atomic<uint64_t>*>(addr);
    // CAS loop ...
}
inline long long int atomicMax(long long int *addr, long long int val) { /* CAS loop */ }
```

**PTX inline asm removed from `barrier_sync` / `barrier_red_or`:**
```cpp
// Removed:
__device__ inline void barrier_sync(int name) {
    asm volatile("barrier.sync.aligned %0;" :: "r"(name) : "memory");
}
// Now extern (implemented in cuda_emulator.hh via Boost.Fiber)
```

**`copyToShmem16`: PTX vectorised store → `std::memcpy`:**
```cpp
// Before: PTX ld.v2.u64, __cvta_generic_to_shared, st.shared.v2.u64
// After:
std::memcpy((uint8_t*)dst + offset, (const uint8_t*)src + offset, 16);
```

**`loadWorkBatchToShmem`: struct access changed to pointer access:**
```cpp
// Before:
struct ncclDevWorkBatch batch = ((struct ncclDevWorkBatch*)(args+1))[batchIx];
int tid = threadIdx.x;
if (tid < sizeof(batch)/16) copyToShmem16(tid, ...);

// After:
struct ncclDevWorkBatch* batch = &((struct ncclDevWorkBatch*)(args+1))[batchIx];
int tid = threadIdx().x;
if (tid < sizeof(*batch)/16) copyToShmem16(tid, ...);
```

**`ncclDevFuncPtr_t` moved to `extern "C"` linkage** (needed for function table
to be visible from the plain C++ host side without name mangling):
```cpp
typedef void(*ncclDevFuncPtr_t)();
extern "C" {
  extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable[];
}
```

### 5b. `src/device/common_kernel.h`

**`loadInt`: PTX volatile load → plain dereference:**
```cpp
// Before:
inline __device__ int loadInt(int* ptr) {
    int v;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(ptr));
    return v;
}
// After:
inline __device__ __host__ int loadInt(const int* ptr) {
    return *(ptr);
}
```

**`#pragma unroll` disabled throughout `reduceCopyPacks`:**
```cpp
// All occurrences:
// #pragma unroll  ← commented out (g++ does not support CUDA unroll pragmas)
```

**`__trap()` → `assert(0)`:**
```cpp
// Before:
if (BytePerPack == 0) __trap();
// After:
if (BytePerPack == 0) assert(0);
```

**`std::max`/`std::min` added where ADL was previously pulling in CUDA overloads:**
```cpp
// Before: max(ssize_t, ssize_t)  — used CUDA device max
// After:  std::max(ssize_t, ssize_t)
```

### 5c. `src/device/primitives.h`

**Cooperative yield added to the spin-wait abort check:**
```cpp
// Before:
if (++spins < NCCL_SPINS_BEFORE_CHECK_ABORT) return 0;

// After:
if (++spins < NCCL_SPINS_BEFORE_CHECK_ABORT) {
    coop_thread_yield(spins);   // ← yields to other fibers while spinning
    return 0;
}
```

This is critical for correctness: without yielding, a fiber waiting for a buffer slot
would starve the sender fibers and deadlock.

### 5d. `src/device/sendrecv.h`

**`__ballot_sync` result explicitly captured before `__popc`** (compiler ordering):
```cpp
// Before:
int nWarpPerWork = __popc(__ballot_sync(~0u, nWorks*(lane+1) <= nWarps));

// After:
__syncwarp();
uint32_t result2 = __ballot_sync(~0u, nWorks*(lane+1) <= nWarps);
int nWarpPerWork = __popc(result2);
```

### 5e. `src/device/onerank.cu` → `onerank.cc`

File renamed `.cu` → `.cc`.  Only change: direct struct-access CUDA built-ins:
```cpp
// Before:
int tid = threadIdx.x;  int tn = blockDim.x;
int bid = blockIdx.x;   int bn = gridDim.x;

// After:
int tid = threadIdx().x;  int tn = blockDim().x;
int bid = blockIdx().x;   int bn = gridDim().x;
```

### 5f. `src/device/common.cc` (new file)

The `ncclDevKernel_Generic` entry point is now a plain C++ function (not `__global__`):
```cpp
// common.cc — new file
void ncclDevKernel_Generic(ncclDevKernelArgs4K const args4K) {
    ncclKernelMain<-1, RunWorkNop>(&args4K.args);
}
__device__ void ncclDevFunc_Nop() {}
```

ncclx v2.27 has an identical `common.cc` with the same structure.

---

## 6. Collective Algorithm Files

All collective headers received the same mechanical substitution: **`ncclShmem.` → `ncclShmem->`**.  The table below lists any *additional* changes beyond that.

| File | `ncclShmem.→->` | Other changes in nccl port | Status vs ncclx v2.27 |
|---|---|---|---|
| `src/device/all_reduce.h` | ✓ all algos | `__trap()→assert(0)` in CollNet direct | Identical after substitution |
| `src/device/all_gather.h` | ✓ all algos | Commented-out debug `printf` lines | Identical after substitution |
| `src/device/reduce_scatter.h` | ✓ all algos | `max/min→std::max/min` for `ssize_t`; `__trap()→assert(0)` | Identical after substitution |
| `src/device/reduce.h` | ✓ all algos | None | Identical after substitution |
| `src/device/broadcast.h` | ✓ | Minor | Identical after substitution |
| `src/device/sendrecv.h` | ✓ | `__ballot_sync` explicit capture; `threadIdx.x→threadIdx().x`; debug LOG comments | Identical after substitution |
| `src/device/onerank.cc` | n/a | `threadIdx.x→threadIdx().x` (×4) | Identical |
| `src/device/primitives.h` | ✓ | `coop_thread_yield(spins)` in `checkAbort` | Identical |
| `src/device/common_kernel.h` | n/a | `loadInt` PTX→deref; `#pragma unroll` disabled; `std::max/min`; `assert(0)` | Identical |
| `src/device/common.h` | n/a | Full CPU shim (see §5a) | Identical in structure; ncclx uses `c++2a` |
| `src/device/common.cc` | n/a | New entry-point file | Identical |

### Files in nccl port not in ncclx v2.27 device dir

- `collectives.h` — local override copy (ncclx references upstream path directly)
- `Makefile.test`, `scripts/`, `.cache/` — development/tooling artifacts

### Files in ncclx v2.27 not yet in nccl port

- `all_reduce_sparse_block.cc/.cu/.cuh` — sparse block all-reduce (Meta internal feature)
- `ctran_cpu_stubs.cc` — CPU compatibility stubs for ctran transport algorithms
- `onerank.cu` / `common.cu` — ncclx keeps the original `.cu` alongside `.cc` for reference
- `symmetric/*.cuh` — ncclx retains the original CUDA `.cuh` files (`.hh` versions added for CPU path); nccl port only has the `.hh` renamed versions

---

## 7. Host-Side / Infrastructure Changes

### 7a. `src/include/device.h`

**Original entire body commented out.**  The file now only exposes the kernel
function-pointer tables over `extern "C"`:

```cpp
// ~627 lines of original content commented out with //
// (ncclConnInfo, ncclRing, ncclTree, ncclDirect, ncclNvls,
//  ncclDevWorkBatch, ncclShmemData, etc.)
// These are now provided via the ncclx include path.

extern "C" {
  extern int   const ncclDevKernelCount;
  extern void* const ncclDevKernelList[];
  extern int   const ncclDevFuncRowToId[];
  extern void* const ncclDevKernelForFunc[];
  extern bool  const ncclDevKernelForFuncIsSpecialized[];
}
```

The actual struct definitions are inherited from the ncclx header set (included via
`-I$(PROJECT_DIR)/torchcomms/comms/ncclx/v2_27/src/include` at build time).

### 7b. `src/include/nccl_common.h`

Single addition:
```cpp
+#include "cuda_emulator.hh"
```

The `cuda_emulator.hh` exposes `shared_memory<T>`, `ThreadContext`, and all the
`extern dim3 threadIdx()` declarations that device code needs.

### 7c. `src/enqueue.cc`

| Change | Before | After | Reason |
|---|---|---|---|
| `cudaGetFuncBySymbol` | `CUDACHECKGOTO(cudaGetFuncBySymbol(&fn, sym), …)` | `fn = (CUfunction)sym;` | No CUDA runtime; symbol IS the function pointer |
| Persistent work upload | Returns `ncclSystemError` | `fprintf(stderr, …); exit(1)` | Not supported in emulator |
| INFO logs in hot path | Multiple `INFO(NCCL_TUNING, …)` | Removed | Reduce log noise |
| Shared memory check | `return ncclSystemError` | `//return ncclSystemError` (commented) | Avoid false failure |

### 7d. `src/transport/net_ib.cc`

| Change | Type | Detail |
|---|---|---|
| `TRACE→INFO` for QP create/RTR | Verbosity | Makes IB path visible in `NCCL_DEBUG=INFO` |
| IB device port enumeration logs | New INFO | `"Device %s has %d ports"` |
| `qpAttr.ah_attr.dlid = info->lid` | **Functional** | NEX override: force LID assignment regardless of subnet check (RoCE path) |
| IB recv entry log | New INFO | `"ncclIbIrecv n %d"` |
| `TRACE→INFO` for MR registration, completion | Verbosity | — |

The `dlid` override is the only functional change — it bypasses the subnet check for
NEX's virtual IB fabric.

### 7e. `src/misc/nvmlwrap.cc`

Two `INFO(NCCL_INIT, …)` calls in the hot NVML device-count path commented out to
reduce startup log noise.

### 7f. `src/graph/paths.cc`

Minor changes (part of revert commits) — reverts to standard NCCL path-finding with
no functional impact.

---

## 8. ncclx-Exclusive Features Not Yet in nccl Port

These features exist in ncclx v2.27 but have **not been ported** to `nccl/`:

### 8a. ctran — Custom Transport Algorithms

`torchcomms/comms/ctran/` provides alternative collective algorithms (AllGather, AllReduce,
AllToAll, Broadcast, SendRecv) implemented over custom transport backends (IB, NVLink).
In the ncclx device Makefile, ctran `.cu` files are listed (commented out — compiled
by the main Makefile) and `ctran_cpu_stubs.cc` bridges the CPU port.

The nccl port has no ctran equivalent.

### 8b. Sparse Block AllReduce

`all_reduce_sparse_block.cc/.cu/.cuh` — a Meta-specific sparse all-reduce optimization
that operates on non-contiguous blocks.  Not present in nccl port.

### 8c. `.cuh` symmetric kernel files

ncclx v2.27 retains the original CUDA `.cuh` symmetric kernel headers alongside the
CPU-ported `.hh` versions.  The nccl port only has the `.hh` renamed versions.

### 8d. `CUDART_VERSION` guard

ncclx explicitly sets `-DCUDART_VERSION=12010` in its Makefile to correctly gate
CUDA 12.x feature paths.  The nccl port does not set this, relying on whatever the
system headers define.

### 8e. Folly/glog runtime dependencies

ncclx v2.27's `libnccl.so` links against Folly, glog, gflags, fmt, double-conversion,
boost_fiber, etc. (visible in `CONDA_LIB_DIR` at runtime).  The standalone nccl port
does **not** require these; it only needs `libboost_fiber` + `libboost_context` for
`cuda_emulator.hh`.

---

## 9. Runtime Environment

### `run.mk` — ported nccl runner

```
NCCL lib:       nccl/build/lib/libnccl.so
NCCL_DEBUG:     WARN
NEX_GPU_PREDICT: 1   ← GPU timing prediction enabled
NEX_PREDICT_DB:  data/A100.db
LD_LIBRARY_PATH: IB_LIB_PATH : nccl/build/lib
No Conda libs needed
```

### `run_ncclx.mk` — ncclx v2.27 runner

```
NCCL lib:        torchcomms/build/ncclx_v2_27/lib/libnccl.so
NCCL_DEBUG:      INFO
NEX_GPU_PREDICT: 0   ← prediction disabled (uses actual emulator timing)
LD_LIBRARY_PATH: IB_LIB_PATH : ncclx_v2_27/lib : miniconda3/envs/torchcomms/lib
Conda lib dir:   /home/jcm/miniconda3/envs/torchcomms/lib  (for folly/glog/boost_fiber)
```

### Common to both runners

```
NCCL_IB_DISABLE=1       — use Socket transport, not real IB
NCCL_NET=Socket
NCCL_IB_HCA=nex0        — virtual IB device provided by NEX
NCCL_TOPO_FILE=config/topo_tap_nccl.xml
NCCL_SHM_DISABLE=1
NCCL_PROTO=Simple
NCCL_NET_SHARED_BUFFERS=0
NCCL_RAS_ENABLE=0
NCCL_NVLS_ENABLE=0
NCCL_CUMEM_ENABLE=0
NCCL_MAX_NCHANNELS=1
```

---

## 10. Commit-by-Commit History (nccl port)

| Commit | Message | Key changes |
|---|---|---|
| `c84d3d9` | changes for fiber | Initial Boost.Fiber integration into `cuda_emulator.hh`; `common.h` gets fiber externs |
| `74d5a33` | change device code inside nccl; emu is free of device code | Main port: all collective `.h` files get `ncclShmem.→ncclShmem->`; `onerank.cu→.cc`; `__shared__` macro; `threadIdx.→threadIdx().` |
| `8b1feaf` | update for freq and ib | `net_ib.cc` IB verbosity increases; `dlid` override for NEX; NVML log quieting |
| `082fd0a` | add -O2 | Compiler optimisation flags added to device Makefile |
| `f4af212` | update | Miscellaneous fixes post initial compile |
| `59014a3` | add reduce_kernel.h; change init | `reduce_kernel.h` added; init path fixes |
| `f1396e0` | checkpoint | `enqueue.cc` patches (`cudaGetFuncBySymbol→direct cast`, persistent work guard) |
| `e94f34b` | no-gpu fix | Fixes for non-GPU (pure-CPU) execution path |
| `aeecbfd` | revert more to original nccl | Reverts non-essential divergences from upstream NCCL to minimise diff |
| `0a87c35` | updated nccl tracing | Tracing/debug logging updates |
| `f1bd0a8` | improve perf bug | Performance correctness fix (spin-wait / cooperative yield tuning) |

---

## 11. Remaining Gaps & Next Steps

### Confirmed working (test results from ncclx v2.27 validation)

All nccl-test collectives pass with **0 wrong values** on 1–2 GPUs:

| Test category | Status |
|---|---|
| All collectives, 1 GPU, default dtype/redop, 8B–128M | ✓ PASS |
| All collectives, 2 GPU (all_reduce, all_gather, reduce_scatter, broadcast, reduce) | ✓ PASS |
| All 10 dtypes × all_reduce, 1+2 GPU | ✓ PASS |
| All 6 redops × all_reduce + reduce + reduce_scatter, 2 GPU | ✓ PASS |
| All 10 dtypes × all 6 redops × all_reduce, 2 GPU (1320 rows) | ✓ PASS |
| Extended iters (n=50, check=5) | ✓ PASS |
| All roots × broadcast + reduce, 2 GPU | ✓ PASS |
| Tiny sizes (1B–1024B step=32) | ✓ PASS |
| Large buffers (256M), 1+2 GPU | ✓ PASS |
| Fixed step sizes (128K–128M step=1M) | ✓ PASS |
| Parallel communicator init (`-p 1`), 2 GPU | ✓ PASS |

### Known limitations / gaps

1. **2-GPU P2P-class collectives (scatter, gather, sendrecv, alltoall) crash with NEX**
   when `-g 2` is used.  The crash is a segfault in `NEX/interpose.c:44` after the
   NCCL kernels launch successfully — this is a **NEX emulator P2P bug**, not an
   ncclx correctness issue.  The same ops pass cleanly with `-g 1`.

2. **Multi-thread mode (`-t 2 -g 1`) causes a deadlock** in NEX's kernel dispatch.
   Two parallel communicators across threads share the same process but NEX's
   fiber scheduler does not currently support concurrent kernel invocations.

3. **ctran not ported** — the nccl port does not include ctran's custom AllReduce /
   AllGather algorithms.  If needed, `ctran_cpu_stubs.cc` from ncclx v2.27 provides
   the needed CPU compatibility shims.

4. **`CUDART_VERSION` not set** in the nccl port Makefile.  Should add
   `-DCUDART_VERSION=12010` (or match the actual CUDA toolkit version) to ensure
   `#if CUDART_VERSION >= ...` feature guards behave correctly.

5. **Symmetric `.cuh` files** — nccl port only has the renamed `.hh` versions.
   Keep the originals around as reference until the port is fully stable.

6. **Persistent work buffers not supported** — `enqueue.cc` calls `exit(1)` if the
   host tries to upload persistent work.  This is acceptable for the current emulator
   (no persistent kernel support) but should be documented.

---

## 12. File-by-File Comparison: nccl port vs ncclx/v2\_27

This section compares every CPU-emulation-relevant file that was modified in the
**nccl port** (`nccl/`) against its counterpart in the **ncclx v2.27 reference**
(`torchcomms/comms/ncclx/v2_27/`).  Upstream NCCL vs ncclx feature differences
(ctran, Scuba, folly, cvars, etc.) are OUT OF SCOPE — we focus only on the
Boost-Fiber GPU-thread emulation layer.

---

### Core Methodology — Both Are Identical

Both repos share the exact same fundamental CPU-emulation architecture:

- **Boost.Fiber** cooperative coroutines simulate CUDA threads
- `CudaEmulator` class creates one fiber per logical thread, runs them all within
  a single OS thread (per block)
- `NamedBarrierManager` + `boost::fibers::barrier` replaces `__syncthreads()` /
  `barrier_sync()`
- `SharedMemory` class (indexed by blockIdx) replaces GPU shared memory
- `shared_memory<T>` template wraps `SharedMemory` lookups
- `SharedMemoryLayout` (`thread_local`) tracks static shm offsets
- `current_thread_ctx` (`fiber_specific_ptr`) provides per-fiber
  `threadIdx`/`blockIdx`
- `#define __shared__ thread_local` (in `common.h`) handles static `__shared__`
  declarations

---

### File-by-File Comparison Table

| File | nccl | ncclx/v2\_27 | CPU-Emu Verdict | Specific Difference |
|------|------|-------------|-----------------|---------------------|
| `src/include/cuda_emulator.hh` | ✅ | ✅ | **DIFFERS** | (1) nccl has forward decls for `__syncthreads()` and dim functions (`dim3 threadIdx()` etc.); ncclx defines them inline. (2) ncclx adds inline wrappers `{ return current_thread_ctx->threadIdx; }` directly in the header. |
| `src/include/cuda_compat.hh` | ❌ MISSING | ✅ (stashed) | **ncclx-only** | Full CUDA→CPU shim: inline `__syncthreads()`, full atomic set, warp shuffle pass-throughs, bit-intrinsics. Currently **intentionally not compiled** — definitions live in `common.h`/`cuda_emulator.hh` instead. Represents a planned future consolidation. |
| `src/device/common.h` | ✅ | ✅ | **IDENTICAL** | `#define __shared__ thread_local`, `__device__`/`__global__`/`__forceinline__` macros, `extern void __syncthreads()`, `__popc()` inline, `atomicMax()` CAS-loop, barrier_sync externals. |
| `src/device/common.cc` | ✅ | ✅ | **IDENTICAL** | Fiber-based `CudaEmulator` runtime implementation. |
| `src/device/common_kernel.h` | ✅ | ✅ | **IDENTICAL** | |
| `src/device/primitives.h` | ✅ | ✅ | **IDENTICAL** | |
| `src/device/prims_simple.h` | ✅ | ✅ | **IDENTICAL** | |
| `src/device/prims_ll.h` | ✅ | ✅ | **IDENTICAL** | |
| `src/device/prims_ll128.h` | ✅ | ✅ | **IDENTICAL** | |
| `src/device/reduce_kernel.h` | ✅ | ✅ | **IDENTICAL** | |
| `src/device/op128.h` | ✅ | ✅ | **IDENTICAL** | |
| `src/device/generate.py` | ✅ | ✅ | **IDENTICAL** | |
| `src/device/all_gather.h` | ✅ | ✅ | **IDENTICAL** | |
| `src/device/all_reduce.h` | ✅ | ✅ | **IDENTICAL** | |
| `src/device/broadcast.h` | ✅ | ✅ | **IDENTICAL** | |
| `src/device/reduce.h` | ✅ | ✅ | **IDENTICAL** | |
| `src/device/reduce_scatter.h` | ✅ | ✅ | **IDENTICAL** | |
| `src/device/sendrecv.h` | ✅ | ✅ | **IDENTICAL** | |
| `src/device/collectives.h` | ✅ (in `src/device/`) | ❌ (in `src/include/`) | **Location differs** | nccl added a copy in `src/device/` with an extra `log2Up()` helper; ncclx keeps it only in `src/include/`. Content otherwise equivalent. |
| `src/device/network/unpack/unpack.h` | ✅ | ✅ | **IDENTICAL** | |
| `src/device/symmetric/all_gather.hh` | ✅ | ✅ | **IDENTICAL** | |
| `src/device/symmetric/all_reduce.hh` | ✅ | ✅ | **IDENTICAL** | |
| `src/device/symmetric/generate.py` | ✅ | ✅ | **IDENTICAL** | |
| `src/device/symmetric/kernel.hh` | ✅ | ✅ | **IDENTICAL** | |
| `src/device/symmetric/primitives.hh` | ✅ | ✅ | **IDENTICAL** | |
| `src/device/symmetric/reduce_scatter.hh` | ✅ | ✅ | **IDENTICAL** | |
| `src/device/Makefile` | ✅ | ✅ | **DIFFERS (build)** | ncclx: explicit `$(CXX) -x c++` for `.cu` files, `-std=c++2a`, `-DCUDART_VERSION=12010`, `-DMOCK_SCUBA_DATA`, conda/CUDA include paths. nccl: retains `-std=c++17`, still references NVCC variable (unused). |
| `src/include/bitops.h` | ✅ | ✅ | **DIFFERS** | nccl: keeps `__host__ __device__` on all template functions; uses ternary for `minval`/`maxval`. ncclx: removes `__device__`, wraps `min()`/`max()` calls with `#if __CUDA_ARCH__` guards. |
| `src/include/device.h` | ✅ (1251 lines) | ✅ (631 lines) | **DIFFERS (structure)** | nccl keeps the original upstream content as a 624-line in-file comment block then the active ported code. ncclx has only active code. Functionally equivalent. |
| `src/include/timer.h` | ✅ | ✅ | **DIFFERS** | nccl adds `calibrate()` with `__rdtsc()` / `<x86intrin.h>` under `#if ENABLE_TIMER`. ncclx removed those lines. Both retain `gettime()` via `clock_gettime()`. |
| `src/include/nccl_common.h` | ✅ | ✅ | **Not CPU-emu** | ncclx shifts log levels, adds `func` param to logger callback. API diff, not emu-related. |
| `src/include/cudawrap.h` | ✅ | ✅ | **Not CPU-emu** | ncclx adds `ErrorStackTraceUtil`, `ncclGetCuMemSysSupported()`. NEX/LD_PRELOAD comment is identical. |
| `src/include/gdrwrap.h` | ✅ | ✅ | **Not CPU-emu** | ncclx adds memory logging. |
| `src/misc/mock_checks.h` | ✅ | ❌ MISSING | **nccl-only** | Mock-aware `SYSCHECK`/`SYSCHECKSYNC`/`SYSCHECKGOTO` macros gated on `NCCL_MOCK_TOPOLOGY`. ncclx uses its own infrastructure. |
| `src/misc/mock_syscalls.cc` | ✅ | ✅ | **IDENTICAL** | |
| `src/misc/ipcsocket.cc` | ✅ | ✅ | **IDENTICAL** | |
| `src/misc/cudawrap.cc` | ✅ | ✅ | **Not CPU-emu** | ncclx adds `ncclGetCuMemSysSupported()`, ctran init, cvar usage. |
| `src/misc/nvmlwrap.cc` | ✅ | ✅ | **Not CPU-emu** | ncclx adds Scuba sampling guard. |
| `src/mnnvl.cc` | ✅ | ✅ | **Not CPU-emu** | ncclx adds logMetaData, extra ncclCuMemAlloc parameter. |
| `src/proxy.cc` | ✅ | ✅ | **Not CPU-emu** | ncclx adds ProxyTrace, cvars, `NCCL_NAMED_THREAD_START`. |
| `src/ras/client_support.cc` | ✅ | ✅ | **Not CPU-emu** | ncclx uses cvars, adds `[[fallthrough]]` annotations. |
| `src/enqueue.cc` | ✅ | ✅ | **Not CPU-emu** | All diffs are Meta-specific features. |
| `src/init.cc` | ✅ | ✅ | **Not CPU-emu** | All diffs are Meta-specific features. |
| `src/transport/net_ib.cc` | ✅ | ✅ | **Not CPU-emu** | ncclx uses cvars, folly, adds ncclFuncToString. |
| `src/transport/nvls.cc` | ✅ | ✅ | **Not CPU-emu** | ncclx uses cvars, adds memory logging. |
| `src/transport/shm.cc` | ✅ | ✅ | **Not CPU-emu** | ncclx uses cvars, adds logMetaData. |
| `makefiles/common.mk` | ✅ | ✅ | **DIFFERS (build)** | ncclx uses `-std=c++2a`, adds `-lstdc++fs`. nccl uses `-std=c++17`. |
| `src/Makefile` | ✅ | ✅ | **DIFFERS (build)** | nccl uses `NCCL_CPPFLAGS` for `-DNCCL_MOCK_TOPOLOGY`; ncclx uses `CXXFLAGS` + adds `-DMOCK_SCUBA_DATA`. Many other diffs are ctran/colltrace source files. |

---

### CPU-Emu Methodology Differences — Summary

| Aspect | nccl port | ncclx/v2\_27 |
|--------|-----------|-------------|
| **`__syncthreads()` location** | Forward decl in `cuda_emulator.hh`; defined in `.cc` | Inline in `cuda_compat.hh` (stashed); active code identical via `common.h` |
| **Dim functions (`threadIdx()` etc.)** | Non-inline forward decls in `cuda_emulator.hh` | Inline `{ return current_thread_ctx->threadIdx; }` in header |
| **`#define __shared__`** | `thread_local` (in `common.h`) | Same in active code; `cuda_compat.hh` (stashed) uses empty macro |
| **Atomic operations** | Only `atomicMax` (CAS loop) in `common.h`; rest from CUDA headers | Full set (`atomicAdd`, `atomicMax`, `atomicMin`, `atomicXor`, `atomicOr`, `atomicCAS`, `atomicExch`) in `cuda_compat.hh` (stashed) |
| **Warp shuffle ops** | Not stubbed; relies on CUDA host headers | Pass-through stubs in `cuda_compat.hh` (return own value) — stashed |
| **Bit intrinsics** | `__popc` in `common.h`; others from CUDA headers | Full set via `__builtin_*` in `cuda_compat.hh` — stashed |
| **`__device__` on `bitops.h`** | Kept (`__host__ __device__`) | Removed; `min()`/`max()` guarded by `#if __CUDA_ARCH__` |
| **`device.h` structure** | Original as 624-line comment block + active code | Active code only (clean) |
| **TSC timer calibration** | Present (`calibrate()` + `<x86intrin.h>`) | Removed |
| **`mock_checks.h`** | Present | Absent (different infra) |
| **Device Makefile C++ standard** | `-std=c++17` | `-std=c++2a` |
| **`cuda_compat.hh` shim** | **Absent** | Present but **intentionally stashed** — planned future consolidation layer |

### Bottom Line

The **core fiber-based GPU-thread emulation is identical** between both ports.  All
device-kernel files (`prims_simple.h`, `prims_ll.h`, `prims_ll128.h`, `primitives.h`,
collective `.h` files, symmetric `.hh` files, `common.h`, `common.cc`) are
**byte-for-byte identical**.  Differences are:

1. **ncclx has `cuda_compat.hh`** as a more complete CUDA→CPU shim but it is
   **intentionally stashed** (not force-included).  It is a planned future
   consolidation point for all CUDA compat macros currently spread across
   `common.h` and `cuda_emulator.hh`.
2. **nccl** relies on CUDA host headers for the missing atomic/intrinsic stubs
   rather than having explicit CPU replacements.
3. **`bitops.h`**: ncclx is more conservative (removes `__device__`, guards
   `min()`/`max()` with `__CUDA_ARCH__`); nccl leaves original qualifiers.
4. Minor: `device.h` comment bloat in nccl, TSC calibration in `timer.h`,
   `mock_checks.h` presence, Makefile C++ standard (`c++17` vs `c++2a`).
