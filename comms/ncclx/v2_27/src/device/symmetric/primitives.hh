#ifndef NCCL_DEVICE_SYMMETRIC_PRIMITIVES_H_
#define NCCL_DEVICE_SYMMETRIC_PRIMITIVES_H_

#include "symmetric.h"
#include "bitops.h"
#include "collectives.h"
#include "op128.h"
#include "reduce_kernel.h"
#include <cstdint>

#if __CUDA_ARCH__ >= 700
// __grid_constant__ appears to break cuda-gdb
#define NCCL_GRID_CONSTANT __grid_constant__
#else
#define NCCL_GRID_CONSTANT
#endif

// flattenIx(pos0, dim0, pos1, dim1, pos2, dim2, ...)
// Given a position vector `pos` in a rectangular index space with lengths in the `dim`
// vector, flatten that down to a linear index. The fastest moving dimension is given first.
__device__ __forceinline__ int flattenIx() { return 0; }

template<typename Int0, typename Int1, typename ...Ints>
static __device__ Int0 flattenIx(Int0 pos, Int1 size, Ints ...more) {
  return pos + size*flattenIx(more...);
}

// Precomputed integer reciprocoals for denominator values 1..64 inclusive.
// Pass these to idivFast64() for fast division on the GPU.
static __device__ uint64_t idivRcp64_upto64(int x) {
  static constexpr uint64_t table[65] = {
    idivRcp64(0x01), idivRcp64(0x01), idivRcp64(0x02), idivRcp64(0x03),
    idivRcp64(0x04), idivRcp64(0x05), idivRcp64(0x06), idivRcp64(0x07),
    idivRcp64(0x08), idivRcp64(0x09), idivRcp64(0x0a), idivRcp64(0x0b),
    idivRcp64(0x0c), idivRcp64(0x0d), idivRcp64(0x0e), idivRcp64(0x0f),
    idivRcp64(0x10), idivRcp64(0x11), idivRcp64(0x12), idivRcp64(0x13),
    idivRcp64(0x14), idivRcp64(0x15), idivRcp64(0x16), idivRcp64(0x17),
    idivRcp64(0x18), idivRcp64(0x19), idivRcp64(0x1a), idivRcp64(0x1b),
    idivRcp64(0x1c), idivRcp64(0x1d), idivRcp64(0x1e), idivRcp64(0x1f),
    idivRcp64(0x20), idivRcp64(0x21), idivRcp64(0x22), idivRcp64(0x23),
    idivRcp64(0x24), idivRcp64(0x25), idivRcp64(0x26), idivRcp64(0x27),
    idivRcp64(0x28), idivRcp64(0x29), idivRcp64(0x2a), idivRcp64(0x2b),
    idivRcp64(0x2c), idivRcp64(0x2d), idivRcp64(0x2e), idivRcp64(0x2f),
    idivRcp64(0x30), idivRcp64(0x31), idivRcp64(0x32), idivRcp64(0x33),
    idivRcp64(0x34), idivRcp64(0x35), idivRcp64(0x36), idivRcp64(0x37),
    idivRcp64(0x38), idivRcp64(0x39), idivRcp64(0x3a), idivRcp64(0x3b),
    idivRcp64(0x3c), idivRcp64(0x3d), idivRcp64(0x3e), idivRcp64(0x3f),
    idivRcp64(0x40)
  };
  return table[x];
}

static __device__ uint32_t idivRcp32_upto64(int x) {
  return idivRcp64_upto64(x)>>32;
}

namespace {
struct ncclCoopCta {
  __device__ void sync() { __syncthreads(); }
  __device__ int self() { return threadIdx().x; }
  __device__ int count() { return blockDim().x; }
};
struct ncclCoopWarps {
  int log2_nWarps;
  __device__ void sync() {
  // PTX removed: barrier.sync to synchronize subgroups of size (32<<log2_nWarps)
  // Fallback: synchronize the whole CTA. Host emulator should map __syncthreads() appropriately.
  __syncthreads();
  }
  __device__ int self() { return threadIdx().x & ((32<<log2_nWarps)-1); }
  __device__ int count() { return 32<<log2_nWarps; }
};
struct ncclCoopWarp {
  __device__ void sync() { __syncwarp(); }
  __device__ int self() { return threadIdx().x%32; }
  __device__ int count() { return 32; }
};
}

namespace {
static constexpr int ncclSymPrims_UseBarrier = 1;
static constexpr int ncclSymPrims_UseLL = 2;
static constexpr int ncclSymPrims_UseMultimem = 4;
struct ncclSymPrims {
  int flags;
  int const &rank;
  int const &nRanks;
  uint32_t const &nRanks_rcp32;
  int block, nBlocks;
  uint32_t nBlocks_rcp32;
  uint32_t nBlocks_nWarps_rcp32;
  uint32_t nRanks_nBlocks_rcp32;
  uint32_t nWarpPerRank, nWarpPerRank_rcp32;
  struct ncclSymDevBase* const &base;
  uintptr_t offsetMc;

  uint32_t const &stride4G;
  uint32_t barEpoch;
  uint32_t llEpoch;

  __device__ ncclSymPrims(ncclSymDevComm const &comm, int flags):
    flags(flags),
    rank(comm.rank),
    nRanks(comm.nRanks),
    nRanks_rcp32(comm.nRanks_rcp32),
    block(blockIdx().x),
    nBlocks(gridDim().x),
    nBlocks_rcp32(idivRcp32_upto64(nBlocks)),
    nBlocks_nWarps_rcp32(imulRcp32(nBlocks, nBlocks_rcp32, blockDim().x/32, idivRcp32_upto64(blockDim().x/32))),
    nRanks_nBlocks_rcp32(imulRcp32(nRanks, nRanks_rcp32, gridDim().x, nBlocks_rcp32)),
    nWarpPerRank(idivFast32(nBlocks*blockDim().x/32, nRanks, nRanks_rcp32)),
    nWarpPerRank_rcp32(idivRcp32_upto64(nWarpPerRank)),
    base(comm.base),
    offsetMc((flags & ncclSymPrims_UseMultimem) ? (char*)comm.baseMc - (char*)base : 0x0),
    stride4G(comm.stride4G) {

    #if CUDART_VERSION >= 12030 && __CUDA_ARCH__ >= 900
      cudaGridDependencySynchronize();
    #endif

    if ((flags & ncclSymPrims_UseBarrier) && threadIdx().x < nRanks) {
      barEpoch = (flags & ncclSymPrims_UseMultimem) ? base->barEpochMc[block] : base->barEpochUc[block];
    }
    if (flags & ncclSymPrims_UseLL) llEpoch = base->llEpoch[block] + 2;
  }
  __device__  ~ncclSymPrims() {
    if (threadIdx().x == 0) {
      if (flags & ncclSymPrims_UseBarrier) {
        ((flags & ncclSymPrims_UseMultimem) ? base->barEpochMc : base->barEpochUc)[block] = barEpoch;
      }
      if (flags & ncclSymPrims_UseLL) base->llEpoch[block] = llEpoch - 2;
    }
  }

  template<typename T>
  __device__ T* peerPtr(int peer, T* selfPtr) {
    return reinterpret_cast<T*>(add4G((uintptr_t)selfPtr, (peer-rank)*stride4G));
  }

  template<typename T>
  __device__ T* multimemPtr(T* selfPtr) {
    return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(selfPtr) + offsetMc);
  }

  __device__  void barrierArrive(ncclCoopCta cta, bool release) {
    cta.sync();
    #if __CUDA_ARCH__ < 700
      if (release) {
        if (cta.self() == 0) __threadfence_system();
        cta.sync();
      }
    #endif
    if (flags & ncclSymPrims_UseMultimem) {
      // PTX removed: multimem.red.{release,relaxed}.sys.add.u32
      // Emulate with atomic fetch_add on host/emulator.
      if (cta.self() == 0) {
        uint32_t* inbox = &multimemPtr(base)->barInboxMc[block];
        if (release) {
          __atomic_fetch_add(inbox, 1u, __ATOMIC_RELEASE);
        } else {
          __atomic_fetch_add(inbox, 1u, __ATOMIC_RELAXED);
        }
      }
    } else {
      int r = cta.self();
      if (r != rank && r < nRanks) {
        uint32_t* inbox = &peerPtr(r, base)->barInboxPerPeer[block*nRanks + rank];
        // PTX removed: st.{release,relaxed}.sys.u32 / st.volatile.u32
        if (release) {
          __atomic_store_n(inbox, (uint32_t)(barEpoch+1), __ATOMIC_RELEASE);
        } else {
          __atomic_store_n(inbox, (uint32_t)(barEpoch+1), __ATOMIC_RELAXED);
        }
      }
    }
  }

  __device__  void barrierWait(ncclCoopCta cta, bool acquire) {
    if (flags & ncclSymPrims_UseMultimem) {
      // PTX removed: ld.{acquire,relaxed}.sys.u32
      if (cta.self() == 0) {
        uint32_t* inbox = &base->barInboxMc[block];
        while (true) {
          uint32_t got = acquire ? __atomic_load_n(inbox, __ATOMIC_ACQUIRE)
                                 : __atomic_load_n(inbox, __ATOMIC_RELAXED);
          if (got-(barEpoch+nRanks) <= (uint32_t(-1)>>1)) break;
        }
        barEpoch += nRanks;
      }
    } else {
      int r = cta.self();
      if (r != rank && r < nRanks) {
        uint32_t* inbox = &base->barInboxPerPeer[block*nRanks + r];
        while (true) {
          // PTX removed: ld.{acquire,relaxed}.sys.u32 / ld.volatile.u32
          uint32_t got = acquire ? __atomic_load_n(inbox, __ATOMIC_ACQUIRE)
                                 : __atomic_load_n(inbox, __ATOMIC_RELAXED);
          if (got-(barEpoch+1) <= uint32_t(-1)>>1) break;
        }
      }
      #if __CUDA_ARCH__ < 700
        if (acquire) {
          cta.sync();
          if (cta.self() == 0) __threadfence();
        }
      #endif
      barEpoch += 1;
    }
    cta.sync();
  }

  __device__ void endLL(ncclCoopCta cta) {
    if (__builtin_expect(llEpoch >= -2u, false)) {
      cta.sync();
      uint4* buf = ncclSymDevBase_getLLBuf(base, nRanks, block, llEpoch);
      int epochSize = ncclSymLLEpochSize(nRanks);
      #pragma unroll 4
      for (int i=cta.self(); i*16 < epochSize; i += cta.count()) {
        buf[i] = uint4{0, 0, 0, 0};
      }
    }
    cta.sync();
    llEpoch += (llEpoch == -1u) ? 3 : 1;
  }

  template<typename T>
  __device__ void sendLL(int peer, int slot, T val) {
    union { T tmp; uint32_t u32[divUp(sizeof(T),8)][2]; };
    tmp = val;
    uint4* buf = ncclSymDevBase_getLLBuf(peerPtr(peer, base), nRanks, block, llEpoch) + slot;
    #pragma unroll
    for (int u=0; u < divUp(sizeof(T),8); u++) {
      // PTX removed: st.volatile.v4.u32 of payload and epoch tags
      // Order: write payload (x,z) then publish tags (y,w) with release semantics.
      uint4* dst = buf + ncclSymLLMaxSlots(sizeof(T))*u;
      __atomic_store_n(&dst->x, u32[u][0], __ATOMIC_RELAXED);
      __atomic_store_n(&dst->z, u32[u][1], __ATOMIC_RELAXED);
      __atomic_store_n(&dst->y, (uint32_t)llEpoch, __ATOMIC_RELEASE);
      __atomic_store_n(&dst->w, (uint32_t)llEpoch, __ATOMIC_RELEASE);
    }
  }

  template<typename T>
  __device__ void bcastLL(int slot, T val) {
    if (flags & ncclSymPrims_UseMultimem) {
      union { T tmp; uint32_t u32[divUp(sizeof(T),8)][2]; };
      tmp = val;
      uint4* bufmc = ncclSymDevBase_getLLBuf(multimemPtr(base), nRanks, block, llEpoch) + slot;
      #pragma unroll
      for (int u=0; u < divUp(sizeof(T),8); u++) {
        // PTX removed: st.volatile.v4.u32 (multimem)
        uint4* dst = bufmc + ncclSymLLMaxSlots(sizeof(T))*u;
        __atomic_store_n(&dst->x, u32[u][0], __ATOMIC_RELAXED);
        __atomic_store_n(&dst->z, u32[u][1], __ATOMIC_RELAXED);
        __atomic_store_n(&dst->y, (uint32_t)llEpoch, __ATOMIC_RELEASE);
        __atomic_store_n(&dst->w, (uint32_t)llEpoch, __ATOMIC_RELEASE);
      }
    } else {
      union { T tmp; uint32_t u32[divUp(sizeof(T),8)][2]; };
      tmp = val;
      uint4* buf0 = ncclSymDevBase_getLLBuf(peerPtr(0, base), nRanks, block, llEpoch) + slot;
      int dr = 0;
      int r = rank;
      #pragma unroll 1
      for (; dr+8 <= nRanks; dr += 8) {
        #pragma unroll
        for (int ur=0; ur < 8; ur++) {
          uint4* buf = add4G(buf0, r*stride4G);
          #pragma unroll
          for (int u=0; u < divUp(sizeof(T),8); u++) {
            // PTX removed: st.volatile.v4.u32
            uint4* dst = buf + ncclSymLLMaxSlots(sizeof(T))*u;
            __atomic_store_n(&dst->x, u32[u][0], __ATOMIC_RELAXED);
            __atomic_store_n(&dst->z, u32[u][1], __ATOMIC_RELAXED);
            __atomic_store_n(&dst->y, (uint32_t)llEpoch, __ATOMIC_RELEASE);
            __atomic_store_n(&dst->w, (uint32_t)llEpoch, __ATOMIC_RELEASE);
          }
          r += 1;
          if (r == nRanks) r = 0;
        }
      }
      #pragma unroll
      for (int ur=0; ur < 8; ur++, dr++) {
        if (dr == nRanks) break;
        uint4* buf = add4G(buf0, r*stride4G);
        #pragma unroll
        for (int u=0; u < divUp(sizeof(T),8); u++) {
          // PTX removed: st.volatile.v4.u32
          uint4* dst = buf + ncclSymLLMaxSlots(sizeof(T))*u;
          __atomic_store_n(&dst->x, u32[u][0], __ATOMIC_RELAXED);
          __atomic_store_n(&dst->z, u32[u][1], __ATOMIC_RELAXED);
          __atomic_store_n(&dst->y, (uint32_t)llEpoch, __ATOMIC_RELEASE);
          __atomic_store_n(&dst->w, (uint32_t)llEpoch, __ATOMIC_RELEASE);
        }
        r += 1;
        if (r == nRanks) r = 0;
      }
    }
  }

  template<int nSlotsMin, int nSlotsMax, typename T>
  __device__ void recvLL(int slot0, int nSlots, int stride, T(&elts)[nSlotsMax]) {
    uint4* buf = ncclSymDevBase_getLLBuf(base, nRanks, block, llEpoch) + slot0;
    uint4 tmp[nSlotsMax][divUp(sizeof(T),8)];
    //int spins=0;
    while (true) {
      #pragma unroll
      for (int u=0; u < nSlotsMax; u++) {
        if (u < nSlotsMin || u < nSlots) {
          #pragma unroll
          for (int v=0; v < divUp(sizeof(T),8); v++) {
            // PTX removed: ld.volatile.v4.u32
            uint4* src = buf + u*stride + v*ncclSymLLMaxSlots(sizeof(T));
            tmp[u][v].x = __atomic_load_n(&src->x, __ATOMIC_RELAXED);
            tmp[u][v].y = __atomic_load_n(&src->y, __ATOMIC_RELAXED);
            tmp[u][v].z = __atomic_load_n(&src->z, __ATOMIC_RELAXED);
            tmp[u][v].w = __atomic_load_n(&src->w, __ATOMIC_RELAXED);
          }
        }
      }
      bool okAll = true;
      #pragma unroll
      for (int u=0; u < nSlotsMax; u++) {
        #pragma unroll
        for (int v=0; v < divUp(sizeof(T),8); v++) {
          if (u < nSlotsMin || u < nSlots) {
            bool ok = tmp[u][v].y == llEpoch &&
                      tmp[u][v].w == llEpoch;
            okAll &= ok;
          }
        }
      }
      if (__builtin_expect(okAll, true)) break;
      //if (spins++ == 10<<20) spins=0;
    }
    #pragma unroll
    for (int u=0; u < nSlotsMax; u++) {
      if (nSlotsMin <= u && u == nSlots) break;
      union { T val; uint32_t u32[divUp(sizeof(T),8)][2]; };
      #pragma unroll
      for (int v=0; v < divUp(sizeof(T),8); v++) {
        u32[v][0] = tmp[u][v].x;
        u32[v][1] = tmp[u][v].z;
      }
      elts[u] = val;
    }
  }

  template<typename Pack, typename T, typename Red, int Unroll=8>
  __device__ Pack recvReduceLL(int slot, int stride, Red red) {
    using Acc = typename Red::EltType;
    using AccPack = BytePack<sizeof(Pack)*sizeof(Acc)/sizeof(T)>;
    AccPack acc;
    bool first = true;
    int r = 0;
    #pragma unroll 1
    for (; r+Unroll <= nRanks; r += Unroll) {
      Pack got[Unroll];
      this->template recvLL</*Min=*/Unroll>(slot + r*stride, Unroll, stride, got);
      AccPack acc0 = applyCast<T, Acc>(got[0]);
      acc = first ? acc0 : applyReduce(red, acc, acc0);
      first = false;
      #pragma unroll
      for (int i=1; i < Unroll; i++) acc = applyReduce(red, acc, applyCast<T, Acc>(got[i]));
    }
    if (r < nRanks) {
      Pack got[Unroll];
      this->template recvLL</*Min=*/1>(slot + r*stride, nRanks-r, stride, got);
      AccPack acc0 = applyCast<T, Acc>(got[0]);
      acc = first ? acc0 : applyReduce(red, acc, acc0);
      #pragma unroll
      for (int i=1; i < Unroll-1; i++) {
        if (r+i < nRanks) acc = applyReduce(red, acc, applyCast<T, Acc>(got[i]));
      }
    }
    return applyCast<Acc, T>(acc);
  }

  template<typename T>
  __device__ T recvLL(int slot) {
    T one[1];
    this->template recvLL<1, 1, T>(slot, 1, 0, one);
    return one[0];
  }

  template<typename Coop, typename T>
  __device__ void coopRecvLL(Coop coop, int slot0, int nSlots, T* dst) {
    int me = coop.self();
    if (me < nSlots) {
      uint4* buf = ncclSymDevBase_getLLBuf(base, nRanks, block, llEpoch) + slot0 + me;
      uint4 got[divUp(sizeof(T), 8)];
      //int spins=0;
      #pragma unroll 1
      while (true) {
        #pragma unroll
        for (int u=0; u < divUp(sizeof(T), 8); u++) {
          // PTX removed: ld.volatile.v4.u32
          uint4* src = buf + u*ncclSymLLMaxSlots(sizeof(T));
          got[u].x = __atomic_load_n(&src->x, __ATOMIC_RELAXED);
          got[u].y = __atomic_load_n(&src->y, __ATOMIC_RELAXED);
          got[u].z = __atomic_load_n(&src->z, __ATOMIC_RELAXED);
          got[u].w = __atomic_load_n(&src->w, __ATOMIC_RELAXED);
        }
        bool ok = true;
        #pragma unroll
        for (int u=0; u < divUp(sizeof(T), 8); u++) {
          ok &= got[u].y == llEpoch;
          ok &= got[u].w == llEpoch;
        }
        if (__builtin_expect(ok, true)) break;
        //if (++spins == 10<<20) { spins=0; printf("r=%d LL spin @ ix=%d got=%d want=%d\n", rank, slot0+me, got[0].y, llEpoch); }
      }
      union { T val; uint32_t u32[divUp(sizeof(T), 8)][2]; };
      #pragma unroll
      for (int u=0; u < divUp(sizeof(T), 8); u++) {
        u32[u][0] = got[u].x;
        u32[u][1] = got[u].z;
      }
      dst[slot0 + me] = val;
    }
  }
};
}

template<template<typename> typename Red, typename T, bool nvls>
struct ncclSymAccumType { using Type = T; };

// Only Red's whose opArg is invariant w.r.t. the datatype can have a different
// accumulator type. At the moment this excludes integer min/max, sumpostdiv,
// and premulsum.
template<> struct ncclSymAccumType<FuncSum, __half, false> { using Type = float; };
#if defined(__CUDA_BF16_TYPES_EXIST__)
template<> struct ncclSymAccumType<FuncSum, __nv_bfloat16, false> { using Type = float; };
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__)
template<> struct ncclSymAccumType<FuncSum, __nv_fp8_e4m3, false> { using Type = float; };
template<> struct ncclSymAccumType<FuncSum, __nv_fp8_e5m2, false> { using Type = float; };
#endif
#endif
