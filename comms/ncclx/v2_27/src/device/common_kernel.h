/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COMMON_KERNEL_H_
#define NCCL_COMMON_KERNEL_H_

#include "device.h"
#include "op128.h"
#include "reduce_kernel.h"
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <type_traits>
#include <cstring>

#include <cuda_runtime.h>

extern float compressT_extreme();
extern void compressT(float ratio);

// Define min for ssize_t
inline __device__ int min(int a, ssize_t b) { return (a < b) ? a : b; }

// Translate GPU volatile int load to direct memory read on CPU
inline __device__ __host__  int loadInt(const int* ptr) {
  return *(ptr);
}

template<typename RedFn, typename T, int Unroll, int BytePerPack,
         int MultimemSrcs, int MinSrcs, int MaxSrcs,
         int MultimemDsts, int MinDsts, int MaxDsts, int PreOpSrcs,
         typename IntBytes, typename SrcPtrFn, typename DstPtrFn>
__device__ __forceinline__ void reduceCopyPacks(
    int nThreads, int &thread,
    uint64_t redArg, uint64_t *preOpArgs, bool postOp,
    int nSrcs, SrcPtrFn const &srcPtrFn, int nDsts, DstPtrFn const &dstPtrFn,
    IntBytes &nBytesBehind, IntBytes &nBytesAhead
  ) {
#if 0
printf("NCCL_DEVICE: WARNING, not using the original code at all.") 
using Scalar = T;
  RedFn redFn(redArg);
  int totalBytes = nBytesBehind + nBytesAhead;
  if (totalBytes <= 0 || nSrcs == 0 || nDsts == 0) {
    return;
  }

  Scalar* srcPtrs[MaxSrcs > 0 ? MaxSrcs : 1];
  for (int s = 0; s < nSrcs; ++s) {
    srcPtrs[s] = reinterpret_cast<Scalar*>(srcPtrFn(s));
  }

  Scalar* dstPtrs[MaxDsts > 0 ? MaxDsts : 1];
  for (int d = 0; d < nDsts; ++d) {
    dstPtrs[d] = reinterpret_cast<Scalar*>(dstPtrFn(d));
  }

  size_t totalElts = static_cast<size_t>(totalBytes / sizeof(Scalar));
  for (size_t idx = thread; idx < totalElts; idx += nThreads) {
    Scalar val = srcPtrs[0] ? srcPtrs[0][idx] : Scalar{};
    // if constexpr (PreOpSrcs > 0) {
      // val = ncclEmuPreOpElement(redFn, preOpArgs, 0, val);
    // }
    for (int s = 1; s < nSrcs; ++s) {
      if (!srcPtrs[s]) continue;
      Scalar tmp = srcPtrs[s][idx];
      // if constexpr (PreOpSrcs > 0) {
        // if (s < PreOpSrcs) tmp = ncclEmuPreOpElement(redFn, preOpArgs, s, tmp);
      // }
      val = tmp;
      // val = ncclEmuReduceElement(redFn, val, tmp);
    }
    if (postOp) {
      // val = ncclEmuPostOpElement(redFn, val);
    }
    for (int d = 0; d < nDsts; ++d) {
      if (dstPtrs[d]) dstPtrs[d][idx] = val;
    }
  }

  nBytesBehind += nBytesAhead;
  nBytesAhead = 0;
  return;
#else

  static_assert(std::is_signed<IntBytes>::value, "IntBytes must be a signed integral type.");
  if (BytePerPack == 0) assert(0);

  // A hunk is the amount of contiguous data a warp consumes per loop iteration
  // assuming all threads partake.
  constexpr int BytePerHunk = Unroll*WARP_SIZE*BytePerPack;
  int nWarps = nThreads/WARP_SIZE;
  int warp = thread/WARP_SIZE;
  int lane = thread%WARP_SIZE;

  // This thread's initial position.
  IntBytes threadBytesBehind = nBytesBehind + (warp*BytePerHunk + lane*BytePerPack);
  IntBytes threadBytesAhead = nBytesAhead - (warp*BytePerHunk + lane*BytePerPack);
  // Number of hunks to be consumed over all warps.
  IntBytes nHunksAhead = nBytesAhead/(BytePerHunk + !BytePerHunk);
  // Advance collective position.
  nBytesBehind += nHunksAhead*BytePerHunk;
  nBytesAhead -= nHunksAhead*BytePerHunk;
  if (Unroll==1 && BytePerPack <= nBytesAhead) {
    // Only Unroll=1 can do partial hunks (where not all threads partake).
    nHunksAhead += 1;
    nBytesBehind += nBytesAhead - (nBytesAhead%(BytePerPack + !BytePerPack));
    nBytesAhead = nBytesAhead%(BytePerPack + !BytePerPack);
  }
  nHunksAhead -= warp;

  RedFn redFn(redArg);
  uintptr_t minSrcs[MinSrcs + !MinSrcs];
  uintptr_t minDsts[MinDsts + !MinDsts];
  // #pragma unroll
  for (int s=0; s < MinSrcs; s++) {
    minSrcs[s] = cvta_to_global(srcPtrFn(s)) + threadBytesBehind;
  }

  // #pragma unroll
  for (int d=0; d < MinDsts; d++) {
    // Yes, for some template arguments this code will be unreachable.  That's fine.
    // coverity[dead_error_line]
    minDsts[d] = cvta_to_global(dstPtrFn(d)) + threadBytesBehind;
  }

  // if (thread == 0) {
  //   printf("reduceCopyPacks debug: nSrcs %d nDsts %d threadBytesBehind %lld threadBytesAhead %lld nBytesBehind %lld nBytesAhead %lld firstSrc %p firstDst %p\n",
  //          nSrcs, nDsts,
  //          (long long)threadBytesBehind, (long long)threadBytesAhead,
  //          (long long)nBytesBehind, (long long)nBytesAhead,
  //          (MinSrcs > 0) ? (void*)minSrcs[0] : nullptr,
  //          (MinDsts > 0) ? (void*)minDsts[0] : nullptr);
  // }

  // We dictate loop termination condition according to whether partial hunks
  // can be handled or not.
  while (Unroll==1 ? (BytePerPack <= threadBytesAhead) : (0 < nHunksAhead)) {
    BytePack<BytePerPack> acc[Unroll];

    // minSrcs[0] cannot be nullptr so we always process it
    { RedFn preFn(0 < PreOpSrcs ? preOpArgs[0] : 0);
      // #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        if (0 < MultimemSrcs) {
          // applyLoadMultimem uses relaxed semantics for same reason we use volatile below.
          acc[u] = applyLoadMultimem<RedFn, BytePerPack>(redFn, minSrcs[0]);
        } else {
          // Use volatile loads in case credits are polled for with volatile (instead of acquire).
          acc[u] = ld_volatile_global<BytePerPack>(minSrcs[0]);
          if (0 < PreOpSrcs) acc[u] = applyPreOp(preFn, acc[u]);
        }
        minSrcs[0] += WARP_SIZE*BytePerPack;
      }
    }

    // #pragma unroll (MinSrcs-1 + !(MinSrcs-1))
    for (int s=1; s < MinSrcs; s++) {
      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_begin]
      BytePack<BytePerPack> tmp[Unroll];
      // coverity[dead_error_line]
      RedFn preFn(s < PreOpSrcs ? preOpArgs[s] : 0);
      // #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        if (s < MultimemSrcs) {
          // applyLoadMultimem uses relaxed semantics for same reason we use volatile below.
          // coverity[dead_error_line]
          tmp[u] = applyLoadMultimem<RedFn, BytePerPack>(redFn, minSrcs[s]);
        } else {
          // Use volatile loads in case credits are polled for with volatile (instead of acquire).
          tmp[u] = ld_volatile_global<BytePerPack>(minSrcs[s]);
        }
        minSrcs[s] += WARP_SIZE*BytePerPack;
      }
      // #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        // coverity[dead_error_line]
        if (s < PreOpSrcs) tmp[u] = applyPreOp(preFn, tmp[u]);
        acc[u] = applyReduce(redFn, acc[u], tmp[u]);
      }
    }

    for (int s=MinSrcs; (MinSrcs < MaxSrcs) && (s < MaxSrcs) && (s < nSrcs); s++) {
      uintptr_t src = cvta_to_global(srcPtrFn(s)) + threadBytesBehind;
      BytePack<BytePerPack> tmp[Unroll];
      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_line]
      RedFn preFn(s < PreOpSrcs ? preOpArgs[s] : 0);
      // #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        // Use volatile loads in case credits are polled for with volatile (instead of acquire).
        tmp[u] = ld_volatile_global<BytePerPack>(src);
        src += WARP_SIZE*BytePerPack;
      }
      // #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        // Yes, for some template arguments this code will be unreachable.  That's fine.
        // coverity[dead_error_line]
        if (s < PreOpSrcs) tmp[u] = applyPreOp(preFn, tmp[u]);
        acc[u] = applyReduce(redFn, acc[u], tmp[u]);
      }
    }

    if (postOp) {
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++)
        acc[u] = applyPostOp(redFn, acc[u]);
    }

    #pragma unroll (MinDsts + !MinDsts)
    for (int d=0; d < MinDsts; d++) {
      #pragma unroll Unroll
      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_begin]
      for (int u=0; u < Unroll; u++) {
        // coverity[dead_error_condition]
        if (d < MultimemDsts) {
          multimem_st_global(minDsts[d], acc[u]);
        } else {
          st_global<BytePerPack>(minDsts[d], acc[u]);
        }
        minDsts[d] += WARP_SIZE*BytePerPack;
      }
    }
    for (int d=MinDsts; (MinDsts < MaxDsts) && (d < MaxDsts) && (d < nDsts); d++) {
      uintptr_t dstPtr = cvta_to_global(dstPtrFn(d));
      uintptr_t dst = dstPtr + threadBytesBehind;
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        st_global<BytePerPack>(dst, acc[u]);
        dst += WARP_SIZE*BytePerPack;
      }
    }

    nWarps = nThreads/WARP_SIZE;
    #pragma unroll
    for (int s=0; s < MinSrcs; s++) {
      minSrcs[s] += (nWarps-1)*BytePerHunk;
    }
    #pragma unroll
    // Yes, for some template arguments this code will be unreachable.  That's fine.
    // coverity[dead_error_line]
    for (int d=0; d < MinDsts; d++) {
      minDsts[d] += (nWarps-1)*BytePerHunk;
    }
    threadBytesBehind += nWarps*BytePerHunk;
    threadBytesAhead -= nWarps*BytePerHunk;
    nHunksAhead -= nWarps;
  }

  nWarps = nThreads/WARP_SIZE;
  warp = thread/WARP_SIZE;
  lane = thread%WARP_SIZE;
  // The last loop iteration could have been partial, i.e. not taken by all
  // threads. The threads that weren't included need an extra subtraction to
  // make the value warp uniform.
  if (Unroll==1 && nHunksAhead > 0) nHunksAhead -= nWarps;
  // Rotate warps so the warp which got the least work here will be warp 0.
  // This effectively assigns: warp = (warp-nHunks+nWarps)%nWarps;
  warp = -nHunksAhead;
  thread = warp*WARP_SIZE + lane;
#endif
}

template<int Unroll, typename RedFn, typename T,
         int MultimemSrcs, int MinSrcs, int MaxSrcs,
         int MultimemDsts, int MinDsts, int MaxDsts, int PreOpSrcs,
         typename IntBytes, typename SrcPtrFn, typename DstPtrFn>
__device__ __forceinline__ void reduceCopy(
    int thread, int nThreads,
    uint64_t redArg, uint64_t *preOpArgs, bool postOp,
    int nSrcs, SrcPtrFn const &srcPtrFn, int nDsts, DstPtrFn const &dstPtrFn,
    IntBytes nElts
  ) {

  static_assert(MultimemSrcs <= MinSrcs && MultimemDsts <= MinDsts, "Multimem pointers cannot exceed respective Min values.");
  //int nWarps = nThreads/WARP_SIZE;
  //int warp = thread/WARP_SIZE;
  int lane = thread%WARP_SIZE;
  // If a multimem src is present then our biggest pack size is limited to what
  // is supported for this redfn/type.
  constexpr int BigPackSize = (MultimemSrcs == 0) ? 16 : LoadMultimem_BigPackSize<RedFn>::BigPackSize;

  if (MaxDsts==0) return;
  if (MinDsts==0 && nDsts==0) return;

  IntBytes nBytesBehind = 0;
  IntBytes nBytesAhead = nElts*sizeof(T);

  #if __cpp_if_constexpr
  if constexpr (BigPackSize > sizeof(T)) {
  #else
  if (BigPackSize > sizeof(T)) {
  #endif
    // Check that all pointers are BigPackSize aligned.
    bool aligned = true;
    if (lane < nSrcs) aligned &= 0 == cvta_to_global(srcPtrFn(lane)) % (BigPackSize + !BigPackSize);
    if (lane < nDsts) aligned &= 0 == cvta_to_global(dstPtrFn(lane)) % (BigPackSize + !BigPackSize);
    aligned = __all_sync(~0u, aligned);
    if (aligned) {
      reduceCopyPacks<RedFn, T, Unroll, BigPackSize,
        MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
        (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
         nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead);
      if (nBytesAhead == 0) return;

      reduceCopyPacks<RedFn, T, /*Unroll=*/1, BigPackSize,
        MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
        (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
         nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead);
      if (nBytesAhead == 0) return;
    }
  }

  reduceCopyPacks<RedFn, T, Unroll*(16/sizeof(T))/2, /*BytePerPack=*/sizeof(T),
    MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
    (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
     nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead);
  if (nBytesAhead == 0) return;

  reduceCopyPacks<RedFn, T, /*Unroll=*/1, /*BytePerPack=*/sizeof(T),
    MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
    (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
     nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead);
}

template<int Unroll, typename RedFn, typename T,
         int MultimemSrcs, int MinSrcs, int MaxSrcs,
         int MultimemDsts, int MinDsts, int MaxDsts, int PreOpSrcs,
         typename IntBytes>
__device__ __forceinline__ void reduceCopy(
    int thread, int nThreads,
    uint64_t redArg, uint64_t *preOpArgs, bool postOp,
    int nSrcs, void** srcPtrs, int nDsts, void** dstPtrs,
    IntBytes nElts
  ) {

  reduceCopy<Unroll, RedFn, T,
             MultimemSrcs, MinSrcs, MaxSrcs,
             MultimemDsts, MinDsts, MaxDsts, PreOpSrcs, IntBytes>
    (thread, nThreads, redArg, preOpArgs, postOp,
     nSrcs, [=]__device__(int i) { return srcPtrs[i]; },
     nDsts, [=]__device__(int i) { return dstPtrs[i]; }, nElts);

}

#endif // COMMON_KERNEL_H_
