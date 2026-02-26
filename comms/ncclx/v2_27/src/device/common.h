
#ifndef NCCL_DEVICE_COMMON_CPU_H_
#define NCCL_DEVICE_COMMON_CPU_H_

#include <cstdint>
#include <cstring>
#include <chrono>
#include <thread>
#include <atomic>

#include "collectives.h"
#include "device.h"
#include "op128.h"
#include "reduce_kernel.h"
#include "network/unpack/unpack_defs.h"
#include <threads.h>
#include "cuda_emulator.hh"

//--- CPU-only qualifiers
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __global__
#define __global__
#endif

#ifndef __launch_bounds__
#define __launch_bounds__(maxThreads, minBlocks)
#endif

#ifndef __shared__
#define __shared__ thread_local
#else
#undef __shared__
#define __shared__ thread_local
#endif

#ifndef __forceinline__
#define __forceinline__
#endif

extern dim3 threadIdx();
extern dim3 blockIdx();
extern dim3 blockDim();
extern dim3 gridDim();
extern void __syncthreads();
extern void __syncwarp();
extern void coop_thread_yield();

// __device__ inline void barrier_sync(int) {}
// __device__ inline void barrier_sync(int, int) {}
// __device__ inline bool barrier_red_or(bool vote, int) { return vote; }
// __device__ inline bool barrier_red_or(bool vote, int, int) { return vote; }
extern void barrier_sync(int);
extern void barrier_sync(int, int);
extern bool barrier_red_or(bool, int);
extern bool barrier_red_or(bool, int, int);
extern int __any_sync(unsigned, int);
extern unsigned __ballot_sync(unsigned, int);
extern void __threadfence_block();
extern void __threadfence_system();
extern void __threadfence();


// artificially introduced functions
// extern void syncwall();
// extern void explicit_yield();
extern void coop_thread_yield(int spins);

inline void __trap(){assert(0);};
inline int __popc(uint x){return __builtin_popcount(x);}
inline uint64_t atomicMax(uint64_t *address, uint64_t val) {
    // Use atomic for thread safety
    std::atomic<uint64_t>* atomic_addr = reinterpret_cast<std::atomic<uint64_t>*>(address);
    uint64_t old = atomic_addr->load(std::memory_order_relaxed);
    while (val > old && !atomic_addr->compare_exchange_weak(old, val, std::memory_order_acq_rel, std::memory_order_relaxed)) {}
    return old;
}

inline long long int atomicMax(long long int *address, long long int val) {
    // Use atomic for thread safety
    std::atomic<long long int>* atomic_addr = reinterpret_cast<std::atomic<long long int>*>(address);
    long long int old = atomic_addr->load(std::memory_order_relaxed);
    while (val > old && !atomic_addr->compare_exchange_weak(old, val, std::memory_order_acq_rel, std::memory_order_relaxed)) {}
    return old;
}

#define COLL_UNROLL (ncclCollUnroll())
#define WARP_SIZE 32


struct ncclShmemGroup {
  ncclConnInfo *recvConns[NCCL_MAX_ARITY];
  ncclConnInfo *sendConns[NCCL_MAX_ARITY];
  void* userInput;
  void* userOutput;
  void* srcs[NCCL_MAX_ARITY+1];
  void* dsts[NCCL_MAX_ARITY+1];
  union {
    unpackGroupShmem unpack;
  } devicePlugin;
  int32_t dstSizes[NCCL_MAX_ARITY+1];
};

struct ncclShmemData {
  struct ncclDevKernelArgs args;
  int channelId;
  int aborted;
  alignas(16) struct ncclDevComm comm;
  alignas(16) struct ncclDevChannel channel;

  int batchIx, nextBatchIx;
  enum ncclDevWorkType workType;
  uint8_t directMode;
  uint16_t funcId;
  int nWorks;
  int workSize;
  uint64_t workCounter;
  bool profilerEnabled;
  struct ncclShmemGroup groups[NCCL_MAX_GROUPS];
  uint64_t redOpArgs[NCCL_MAX_NVLS_ARITY+1];

  alignas(16) char workStorage[1024];

  alignas(16) union {
    unpackShmem unpack;
  } devicePlugin;
};

extern thread_local shared_memory<ncclShmemData> ncclShmem;
extern thread_local shared_memory<ulong2> ncclShmemPerWarp;

__device__ inline void* ncclScratchForWarp(int warp) {
  return (char*)(ncclShmemPerWarp + warp*ncclShmemScratchWarpSize());
}

// extern __shared__ ncclShmemData ncclShmem[1];

// extern __shared__ ulong2 ncclShmemPerWarp[ncclShmemScratchWarpSize()*(NCCL_MAX_NTHREADS/WARP_SIZE)/sizeof(ulong2)];

// __device__ inline void* ncclScratchForWarp(int warp) {
//   return (char*)ncclShmemPerWarp + warp*ncclShmemScratchWarpSize();
// }


typedef void(*ncclDevFuncPtr_t)();

extern "C" {
  extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable[];
}

// Copy 16-byte aligned data
inline __device__ void copyToShmem16(int tid, void* dst, const void* src, int bytes) {
  int offset = 16*tid;
  if (offset < bytes) {
    std::memcpy((uint8_t*)dst + offset, (const uint8_t*)src + offset, 16);
  }
}

// Must run with at least 64 threads
__device__ __forceinline__ void loadWorkBatchToShmem(
    int tid, int tn, struct ncclDevKernelArgs const* args, int batchIx
  ) {
  int lane = tid%WARP_SIZE;
  int workCursor = 0; // num works written in previous loop iterations.
  while (true) {

    struct ncclDevWorkBatch* batch = &((struct ncclDevWorkBatch*)(args+1))[batchIx];

    // fnsOfBitset[n] = index of n'th set bit in batch->offsetBitset.
    // PTX has instruction "fns" (find n-th set) but it expands to a lot of SASS,
    // since we know all lanes will be querying the same bitmask we can compute
    // much faster using shared memory.
    uint8_t* fnsOfBitset = (uint8_t*)ncclScratchForWarp(threadIdx().x/WARP_SIZE);
    // printf("tid(%d) batch (%d) (%p) %p, offsetBitset: %ld, offsetBase %d\n", tid, batchIx, args+1, batch, batch->offsetBitset, batch->offsetBase);

    __syncwarp();
    if (uint32_t(batch->offsetBitset) & (1u<<lane)) {
      int nWorksBelow = __popc(uint32_t(batch->offsetBitset) & ((1u<<lane)-1));
      fnsOfBitset[nWorksBelow] = lane;
    }
    int nWorksLow32 = __popc(uint32_t(batch->offsetBitset)); // just of low 32 bits
    if (uint32_t(batch->offsetBitset>>32) & (1u<<lane)) {
      int nWorksBelow = nWorksLow32;
      nWorksBelow += __popc(uint32_t(batch->offsetBitset>>32) & ((1u<<lane)-1));
      fnsOfBitset[nWorksBelow] = 32 + lane;
    }
    int nWorks = nWorksLow32 + __popc(uint32_t(batch->offsetBitset>>32)); // add high 32 bits
    __syncwarp();

    int workSize;
    int nPacks; // total number of packs loaded, each pack is 16 bytes
    int packInWork; // my pack index within work struct
    int dstWork; // my work index in contiguous destination shmem
    switch (batch->workType) {
    case (int)ncclDevWorkTypeP2p:
      workSize = sizeof(struct ncclDevWorkP2p);
      nPacks = nWorks*(workSize/16);
      packInWork = tid%(workSize/16);
      dstWork = tid/(workSize/16);
      break;
    case (int)ncclDevWorkTypeColl:
      workSize = sizeof(struct ncclDevWorkColl);
      nPacks = nWorks*(workSize/16);
      packInWork = tid%(workSize/16);
      dstWork = tid/(workSize/16);
      break;
    case (int)ncclDevWorkTypeCollReg:
    default:
      workSize = sizeof(struct ncclDevWorkCollReg);
      nPacks = nWorks*(workSize/16);
      packInWork = tid%(workSize/16);
      dstWork = tid/(workSize/16);
      break;
    }
    if (tid == 0) {
      ncclShmem->workSize = workSize;
    }
    // We deliberately replicate these div and mod calculations into the case
    // blocks above so that they get constant divisor optimizations by the compiler.
    //   packInWork = tid%(workSize/16);
    //   dstWork = tid/(workSize/16);

    // We can only assume we have 64 threads, which means we can read at most 1024 bytes
    // here which is the per batch maximum.
    if (tid < nPacks) {
      int srcWork = fnsOfBitset[dstWork]; // find n'th set bit in batch->offsetBitset
      ulong2 tmp;
      // The loads done in these two cases must be kept separate since we are
      // relying on the compiler to use "ld.param" in the first one. The parameter
      // space is not generically addressable, so any attempt to load through
      // a pointer that *might* be parameter space backed will cause the
      // compiler to spill the parameter struct (4K!) to each thread's local space
      // before creating a pointer (to the spill) and decimate perf.
      //
      // An example of what not to do would be the following:
      //
      // if (condition) {
      //   // The compiler could spill parameter_variable to local space and take
      //   // the address of that, since when src is loaded below it could also
      //   // be global space.
      //   src = &parameter_variable;
      // } else {
      //   src = &global_variable;
      // }
      // memcpy(dst, src, n);
      if (ncclShmem->args.workStorageType == ncclDevWorkStorageTypeArgs) {
        char* src = (char*)args + (batch->offsetBase + srcWork*workSize + packInWork*16);
        tmp = *(ulong2*)src; // becomes ld.param.v2.u64
        // LOG(LOG_DEBUG, "tid(%d) workbatch src (ncclDevWorkStorageTypeArgs) (offsetbase %d, srcWork %d, workSize %d, packInWork %d) %p", tid, batch->offsetBase, srcWork, workSize, packInWork, src);
      } else {
        char* src = (char*)ncclShmem->args.workBuf + ((batch->offsetBase + srcWork*workSize + packInWork*16) & ncclShmem->args.workMask);
        tmp = *(ulong2*)src; // becomes ld.v2.u64
        // LOG(LOG_DEBUG, "tid(%d) workbatch src %p, (offsetbase %d, srcWork %d, workSize %d, packInWork %d) ", tid, src, batch->offsetBase, srcWork, workSize, packInWork);

      }
      char* dst = ncclShmem->workStorage;
      dst += (workCursor + dstWork)*workSize + packInWork*16;
      *(ulong2*)dst = tmp;
    }
    workCursor += nWorks;

    if (batch->nextExtends) {
      batchIx += batch->nextJump;
      tid -= 64; // Rotate threads so we use the next two warps for next batch struct.
      if (tid < 0) tid += tn;
    } else {
      if (tid == 0) {
        ncclShmem->batchIx = batchIx;
        ncclShmem->nextBatchIx = (batch->nextJump == 0) ? -1 : batchIx + batch->nextJump;
        ncclShmem->workType = (enum ncclDevWorkType)batch->workType;
        ncclShmem->nWorks = workCursor;
        ncclShmem->funcId = batch->funcId;
      }
      break;
    }
  }
}

// Global timer using std::chrono
__device__ __forceinline__ unsigned long long globaltimer() {
  auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
  return (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWorkColl {
  __device__ void run(int tid, int tn, struct ncclDevWorkColl* work) {
    // Put NOT IMPLEMENTED behavior here.
  }
};


template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWorkBatch;

// Specialized for P2p in sendrecv.h
template<typename T, typename RedOp>
struct RunWorkBatch<ncclFuncSendRecv, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE>;


// Specialized here for non-P2p (Coll and CollReg)
template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWorkBatch {
  // This __forceinline__ is necessary. The compiler was inserting a function call
  // here from the LL ncclKernel.
  __device__ __forceinline__ void run() {
    int tid = threadIdx().x;
    int tn = blockDim().x;

    if (RedOpArg<RedOp>::ArgUsed) {
      int nWorks = ncclShmem->nWorks;
      for (int w=tid; w < nWorks; w += tn) {
        struct ncclDevWorkColl* work = (ncclDevWorkColl*)(ncclShmem->workStorage + w*ncclShmem->workSize);
        if (work->redOpArgIsPtr) {
          work->redOpArg = RedOpArg<RedOp>::loadArg(reinterpret_cast<void*>(work->redOpArg));
        }
      }
      __syncthreads();
    }

    #pragma unroll 1
    for (int w=0; w < ncclShmem->nWorks; w++) {
      struct ncclDevWorkColl* work = (struct ncclDevWorkColl*)(ncclShmem->workStorage + w*ncclShmem->workSize);
      if (w != 0) {
        struct ncclDevWorkColl* workPrev = (struct ncclDevWorkColl*)(ncclShmem->workStorage + (w-1)*ncclShmem->workSize);
        if (work->nWarps != workPrev->nWarps) __syncthreads();
      }
      
      // LOG(LOG_DEBUG, "RunWorkBatch nwarps %d", work->nWarps);
      int subtn = work->nWarps*WARP_SIZE;
      // Coverity reports a possible thread divergence due to not all threads participating in the collective.
      // However, the code ensures that the participation is on a per-warp basis.
      // coverity[device_thread_diverged:FALSE]
      if (tid < subtn) RunWorkColl<Fn, T, RedOp, Algo, Proto>().run(tid, subtn, work);
    }
  }
};

#define START 0
#define STOP  1
#define FINI  2

__device__ __forceinline__ bool profilerEnabled(int workItemIdx) {
  return (ncclShmem->workType == ncclDevWorkTypeP2p) ?
    ((struct ncclDevWorkP2p*)ncclShmem->workStorage)[workItemIdx].profilerEnabled :
    ((struct ncclDevWorkColl*)ncclShmem->workStorage)[workItemIdx].profilerEnabled;
}

__device__ __forceinline__ void profiler(int action) {
  if (threadIdx().x == 0) {
    int idx = 0;
    uint64_t wc = ncclShmem->channel.workCounter + 1;
    if (action == START) {
      for (; wc <= ncclShmem->channel.workCounter + ncclShmem->nWorks; wc++) {
        if (!profilerEnabled(idx++)) continue;
        ncclShmem->comm.workStarted[ncclShmem->channelId].data[wc%MAX_PROFILER_EVENTS_PER_CHANNEL].timestamp = globaltimer();
        ncclShmem->comm.workStarted[ncclShmem->channelId].data[wc%MAX_PROFILER_EVENTS_PER_CHANNEL].counter = wc;
      }
    } else {
      for (; wc <= ncclShmem->channel.workCounter + ncclShmem->nWorks; wc++) {
        if (!profilerEnabled(idx++)) continue;
        ncclShmem->comm.workCompleted[ncclShmem->channelId].data[wc%MAX_PROFILER_EVENTS_PER_CHANNEL].timestamp = globaltimer();
        ncclShmem->comm.workCompleted[ncclShmem->channelId].data[wc%MAX_PROFILER_EVENTS_PER_CHANNEL].counter = wc;
      }
      ncclShmem->channel.workCounter += ncclShmem->nWorks;
      if (action == FINI) ((ncclDevCommAndChannels*)ncclShmem->args.comm)->channels[ncclShmem->channelId].workCounter = ncclShmem->channel.workCounter;
    }
  }
}


// Kernel main in CPU: sequential single-threaded
// Mirrors original CUDA logic in a single-threaded context
template<int SpecializedFnId, typename SpecializedRunWorkBatch>
__device__ __forceinline__ void ncclKernelMain(const struct ncclDevKernelArgs* args) {
  // Single-thread context
  int tid = threadIdx().x;
  int tn = blockDim().x;
   
  if (tid < sizeof(ncclDevKernelArgs)/sizeof(uint32_t)) {
    ((uint32_t*)&ncclShmem->args)[tid] = ((uint32_t*)args)[tid];
  }

  if (tid < MAXCHANNELS && (args->channelMask & (1ull<<tid))) {
    int n = __builtin_popcount(args->channelMask & ((1ull<<tid)-1));
    if (blockIdx().x == n) ncclShmem->channelId = tid;
  }

  __syncthreads(); // publish ncclShmem->{args, channelId}
  /* set abort flag to 0 */
  if (tid == 0) {
    ncclShmem->aborted = 0;
    ncclShmem->channel.workCounter = ((ncclDevCommAndChannels*)ncclShmem->args.comm)->channels[ncclShmem->channelId].workCounter;
  }

  // Use first 2 warps to load comm and channel, and remaining load work batch->
  switch (tid/WARP_SIZE) {
  case 0:
    { void* dst = &ncclShmem->comm;
      void* src = ncclShmem->args.comm;
      int bytes = sizeof(ncclDevComm);
      static_assert(sizeof(ncclDevComm) <= 16*WARP_SIZE, "ncclDevComm cannot be loaded by a single warp in one insn.");
      copyToShmem16(tid, dst, src, bytes);
    } break;
  case 1:
    { // Get address of channel without incurring indirect load from ncclDevComm::channels
      void* dst = &ncclShmem->channel;
      void* src = &((ncclDevCommAndChannels*)ncclShmem->args.comm)->channels[ncclShmem->channelId];
      int bytes = sizeof(ncclDevChannel);
      static_assert(sizeof(ncclDevChannel) <= 16*WARP_SIZE, "ncclDevChannel cannot be loaded by a single warp in one insn.");
      copyToShmem16(tid-WARP_SIZE, dst, src, bytes);
    } break;
  default:
    { int subtid = tid - 2*WARP_SIZE;
      int subtn = tn - 2*WARP_SIZE;
      // Coverity reports a possible thread divergence due to not all threads participating in the collective.
      // However, the code ensures that the participation is on a per-warp basis.
      // coverity[device_thread_diverged:FALSE]
      loadWorkBatchToShmem(subtid, subtn, args, /*batchIx=*/blockIdx().x);
    } break;
  }

  __syncthreads(); // publish ncclShmem

  while (ncclShmem->aborted == 0) {
    profiler(START);
    if (0 <= SpecializedFnId && ncclShmem->funcId == (unsigned)SpecializedFnId) {
      SpecializedRunWorkBatch().run();
      // LOG(LOG_DEBUG, "SpecializedRunWorkBatch[%d]", ncclShmem->funcId);
    } else {
      // LOG(LOG_DEBUG, "ncclDevFuncTable[%d]", ncclShmem->funcId);
      ncclDevFuncTable[ncclShmem->funcId]();
    }

    if (ncclShmem->nextBatchIx == -1) break;
    int batchIx = ncclShmem->nextBatchIx;
    __syncthreads();
    profiler(STOP);
    loadWorkBatchToShmem(tid, tn, args, batchIx);
    __syncthreads();
  }
  profiler(FINI);
}

__global__ void ncclDevKernel_Generic(ncclDevKernelArgs4K const args4K);
__device__ void ncclDevFunc_Nop();

#define DEFINE_ncclDevKernel(suffix, coll, redop, ty, algo, proto, specializedFnId) \
  __global__ void ncclDevKernel_##suffix(ncclDevKernelArgs4K const args4K) { \
    ncclKernelMain<specializedFnId, RunWorkBatch<coll, ty, redop<ty>, algo, proto>>(&args4K.args); \
  }

#define DEFINE_ncclDevKernel_nop(suffix, coll, redop, ty, algo, proto, specializedFnId) \
  __global__ void ncclDevKernel_##suffix(ncclDevKernelArgs4K const args4K) {}

#define DEFINE_ncclDevFunc(suffix, coll, redop, ty, algo, proto) \
  __device__ void ncclDevFunc_##suffix() { \
    RunWorkBatch<coll, ty, redop<ty>, algo, proto>().run(); \
  }

#endif
