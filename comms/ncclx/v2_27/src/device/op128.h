/*************************************************************************
 * CPU-port of op128.h
 *   – keeps the original API/identifiers so existing code still compiles
 *   – replaces CUDA/PTX and __device__ intrinsics with portable C++17
 *   – GPU-only helpers are stubbed (no-ops) unless they can be mapped
 *     directly to an equivalent host operation.
 ************************************************************************/
#ifndef OP128_CPU_H_
#define OP128_CPU_H_

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <algorithm>

/* --------------------------------------------------------------------- */
/*  Helper macros so the original qualifiers don’t break host builds     */
/* --------------------------------------------------------------------- */
#ifndef __device__
#define __device__
#endif
#ifndef __align__
#define __align__(x)
#endif

extern void __syncthreads();
extern void __syncwarp();
extern int __all_sync(unsigned mask, int predicate);
extern void global_memcpy(void* dst, const void* src, size_t size, int pos);
// extern void shared_memcpy(void* dst, const void* src, size_t size);

/* Funnel-shift helper (PTX’s __funnelshift_r replacement) */
static inline uint32_t __funnelshift_r(uint32_t lo, uint32_t hi, int shift)
{
  shift &= 31;
  return (lo >> shift) | (hi << (32 - shift));
}

/* --------------------------------------------------------------------- */
/*  128-bit load/store helpers                                           */
/* --------------------------------------------------------------------- */
inline __device__ void load128(const uint64_t* ptr,
                               uint64_t& v0,
                               uint64_t& v1)
{
  /* aligned CPU load */
  v0 = ptr[0];
  v1 = ptr[1];
}

inline __device__ void store128(uint64_t* ptr,
                                uint64_t  v0,
                                uint64_t  v1)
{
  /* aligned CPU store */
  ptr[0] = v0;
  ptr[1] = v1;
}

/* “Shared-memory” versions are aliases on the CPU build. */
inline __device__ uint64_t* shmemCvtPtr(volatile uint64_t* p)
{ return const_cast<uint64_t*>(p); }

inline __device__ void loadShmem128(uint64_t* p, uint64_t& v0, uint64_t& v1)
{ load128(p, v0, v1); }

inline __device__ void storeShmem128(uint64_t* p, uint64_t v0, uint64_t v1)
{ store128(p, v0, v1); }

/* Generic mis-aligned 128-bit load from an arbitrary pointer.            */
template<typename T>
inline __device__ void loadShmemMisaligned128(T* ptr,
                                              uint64_t& v0,
                                              uint64_t& v1)
{
  alignas(16) struct { uint64_t a, b; } tmp;
  std::memcpy(&tmp, ptr, sizeof(tmp));
  v0 = tmp.a;
  v1 = tmp.b;
}

/* --------------------------------------------------------------------- */
/*  Pointer “conversions” – just casts on the CPU                        */
/* --------------------------------------------------------------------- */
template<typename T>
inline __device__  uint32_t cvta_to_shared(T* p)
{ return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p)); }

template<typename T>
inline __device__  uintptr_t cvta_to_global(T* p)
{ return reinterpret_cast<uintptr_t>(p); }

template<typename T>
inline __device__  T* cvta_from_shared(uint32_t sh)
{ return reinterpret_cast<T*>(static_cast<uintptr_t>(sh)); }

template<typename T>
inline __device__  T* cvta_from_global(uintptr_t gp)
{ return reinterpret_cast<T*>(gp); }

/* --------------------------------------------------------------------- */
/*  BytePack unions (unchanged from original except for host-only tweaks)*/
/* --------------------------------------------------------------------- */
template<int Size> union BytePack;
template<> union BytePack<0> {};
template<> union BytePack<1> { uint8_t u8[1], native; };
template<> union BytePack<2> {
  BytePack<1> half[2]; BytePack<1> b1[2]; uint8_t u8[2];
  uint16_t u16[1], native;
};
template<> union BytePack<4> {
  BytePack<2> half[2]; BytePack<1> b1[4]; BytePack<2> b2[2];
  uint8_t u8[4]; uint16_t u16[2]; uint32_t u32[1], native;
};
template<> union BytePack<8> {
  BytePack<4> half[2]; BytePack<1> b1[8];
  BytePack<2> b2[4];  BytePack<4> b4[2];
  uint8_t u8[8]; uint16_t u16[4]; uint32_t u32[2]; uint64_t u64[1], native;
};
template<> union alignas(16) BytePack<16> {
  BytePack<8>  half[2]; BytePack<1>  b1[16];
  BytePack<2>  b2[8];  BytePack<4>  b4[4];
  BytePack<8>  b8[2];
  uint8_t  u8[16]; uint16_t u16[8]; uint32_t u32[4]; uint64_t u64[2];
  ulong2 ul2[1], native;
};
template<int Size> union BytePack   {
  BytePack<Size/2> half[2];
  BytePack<1>  b1[Size];
  BytePack<2>  b2[Size/2];
  BytePack<4>  b4[Size/4];
  BytePack<8>  b8[Size/8];
  BytePack<16> b16[Size/16];
  uint8_t  u8[Size];
  uint16_t u16[Size/2];
  uint32_t u32[Size/4];
  uint64_t u64[Size/8];
};

/* Mappers to/from packs. */
template<typename T>
struct BytePackOf { static constexpr int Size = sizeof(T); using Pack = BytePack<Size>; };
template<> struct BytePackOf<BytePack<0>> { static constexpr int Size = 0; using Pack = BytePack<0>; };

template<typename T>
inline __device__  typename BytePackOf<T>::Pack
toPack(T value)
{
  typename BytePackOf<T>::Pack p;
  std::memcpy(&p, &value, sizeof(T));
  return p;
}
template<typename T>
inline __device__  T
fromPack(typename BytePackOf<T>::Pack p)
{
  T v;
  std::memcpy(&v, &p, sizeof(T));
  return v;
}

/* --------------------------------------------------------------------- */
/*  Global/shared load/store helpers – CPU versions                      */
/* --------------------------------------------------------------------- */
template<int Size>
inline __device__  BytePack<Size> ld_global(uintptr_t addr)
{
  BytePack<Size> v;
  // std::memcpy(&v, reinterpret_cast<void*>(addr), Size);
  global_memcpy(&v, reinterpret_cast<void*>(addr), Size, 1);
  return v;
}

template<int Size>
inline __device__  __attribute__((no_sanitize("undefined"))) void st_global(uintptr_t addr,
                                                 BytePack<Size> value)
{
  if (Size > 0) global_memcpy(reinterpret_cast<void*>(addr), &value, Size, 0);
}

/* Shared‐space versions just forward to the same implementation          */
template<int Size> inline __device__ 
BytePack<Size> ld_shared(uint32_t addr) { 
  // return ld_global<Size>(addr); 
  BytePack<Size> v;
  std::memcpy(&v, reinterpret_cast<void*>(addr), Size);
  return v;
}

template<int Size> inline __device__ 
void st_shared(uint32_t addr, BytePack<Size> v) { 
  if (Size > 0) std::memcpy(reinterpret_cast<void*>(addr), &v, Size);
}

/* Volatile / relaxed variants map to the same thing on host builds       */
template<int Size> inline __device__ 
BytePack<Size> ld_volatile_global(uintptr_t a) { return ld_global<Size>(a); }
template<int Size> inline __device__ 
BytePack<Size> ld_volatile_shared(uint32_t a)  { return ld_shared<Size>(a); }
template<int Size> inline __device__ 
BytePack<Size> ld_relaxed_gpu_global(uintptr_t a) { return ld_global<Size>(a); }
template<int Size> inline __device__ 
void st_relaxed_gpu_global(uintptr_t a, BytePack<Size> v) { st_global<Size>(a,v); }

/* --------------------------------------------------------------------- */
/*  Atomic / fence helpers (best-effort host emulation)                   */
/* --------------------------------------------------------------------- */
inline __device__  uint64_t ld_volatile_global(uint64_t* p)
{ return *reinterpret_cast<volatile uint64_t*>(p); }

inline __device__  uint64_t ld_relaxed_sys_global(uint64_t* p)
{ return *p; }

inline __device__  uint64_t ld_relaxed_gpu_global(uint64_t* p)
{ return *p; }

inline __device__  uint64_t ld_acquire_sys_global(uint64_t* p)
{ return *p; }

inline __device__  void st_volatile_global(uint64_t* p, uint64_t v)
{ *reinterpret_cast<volatile uint64_t*>(p) = v; }

inline __device__  void st_relaxed_sys_global(uint64_t* p, uint64_t v)
{ *p = v; }

inline __device__  void st_release_sys_global(uint64_t* p, uint64_t v)
{ *p = v; }

inline __device__  void fence_acq_rel_sys()  { /* full compiler fence */ __sync_synchronize(); }
inline __device__  void fence_acq_rel_gpu()  { __sync_synchronize(); }

/* --------------------------------------------------------------------- */
/*  Multimem stubs – no multi-mem semantics on the CPU                   */
/* --------------------------------------------------------------------- */
template<int Size>
inline __device__  void multimem_st_global(uintptr_t, BytePack<Size>)
{ /* no-op on CPU */ }

/* --------------------------------------------------------------------- */
/*  Pack helpers – CPU implementation                                    */
/* --------------------------------------------------------------------- */
template<typename Pack, typename T>
inline __device__ 
Pack loadPack(T* ptr, int ix, int end)
{
  constexpr int Size = sizeof(Pack);
  Pack out{};
  int nElts = std::min<int>(end - ix, Size / sizeof(T));
  std::memcpy(&out, ptr + ix, nElts * sizeof(T));
  return out;
}

template<typename Pack, typename T>
inline __device__ 
void storePack(T* ptr, int ix, int end, Pack val)
{
  constexpr int Size = sizeof(Pack);
  int nElts = std::min<int>(end - ix, Size / sizeof(T));
  std::memcpy(ptr + ix, &val, nElts * sizeof(T));
}

/* --------------------------------------------------------------------- */
/*  copyGlobalShared_WarpUnrolled – stub (single-thread copy version)    */
/* --------------------------------------------------------------------- */
// template<int EltSize, int MaxBytes, bool Multimem, typename IntBytes>
// inline __device__ 
// void copyGlobalShared_WarpUnrolled(int /*lane*/,
//                                    uintptr_t dstAddr,
//                                    uint32_t  srcAddr,
//                                    IntBytes  nBytesAhead)
// {
//   if (nBytesAhead <= 0) return;
//   auto n = static_cast<size_t>(std::min<IntBytes>(nBytesAhead, MaxBytes));
//   std::memcpy(reinterpret_cast<void*>(dstAddr),
//               reinterpret_cast<void*>(static_cast<uintptr_t>(srcAddr)),
//               n);
// }

// Emulated warp-cooperative memory copy

template<int EltSize, int MaxBytes, bool Multimem, typename IntBytes>
inline __device__ 
void copyGlobalShared_WarpUnrolled(int lane,
                                   uintptr_t dstAddr,
                                   uint32_t  srcAddr,
                                   IntBytes  nBytesAhead)
{
    static_assert(std::is_signed_v<IntBytes>, "`IntBytes` must be a signed integral type.");
    
    int nBytes = std::min(nBytesAhead, (IntBytes)MaxBytes);
    int nFrontBytes = std::min(nBytes, (16 - int(dstAddr%16))%16);
    int nMiddleBytes = (nBytes-nFrontBytes) & -16;
    int nBackBytes = (nBytes-nFrontBytes) % 16;

    // Handle front and back bytes (unaligned parts)
    {
        int backLane = WARP_SIZE-1 - lane;
        bool hasFront = lane*EltSize < nFrontBytes;
        bool hasBack = backLane*EltSize < nBackBytes;
        int offset = hasFront ? lane*EltSize : (nBytes - (backLane+1)*EltSize);
        
        if (hasFront | hasBack) {
            BytePack<EltSize> tmp = ld_shared<EltSize>(srcAddr+offset);
            st_global<EltSize>(dstAddr+offset, tmp);
        }
    }

    srcAddr += nFrontBytes;
    int srcMisalign = EltSize < 4 ? (srcAddr%4) : 0;
    srcAddr += -srcMisalign + lane*16;
    dstAddr += nFrontBytes + lane*16;
    nMiddleBytes -= lane*16;
    
    // Handle aligned middle section in 16-byte chunks
    #pragma unroll
    for (int u=0; u < divUp(MaxBytes, WARP_SIZE*16); u++) {
        if (nMiddleBytes <= 0) break;
        
        union {
            BytePack<4> b4[4];
            BytePack<16> b16;
        };
        
        // Load 4x4-byte chunks
        b4[0] = ld_shared<4>(srcAddr + 0*4);
        b4[1] = ld_shared<4>(srcAddr + 1*4);
        b4[2] = ld_shared<4>(srcAddr + 2*4);
        b4[3] = ld_shared<4>(srcAddr + 3*4);
        
        // Handle misalignment with funnel shift
        if (srcMisalign != 0) {
            BytePack<4> b4_4 = ld_shared<4>(srcAddr + 4*4);
            b4[0].native = __funnelshift_r(b4[0].native, b4[1].native, srcMisalign*8);
            b4[1].native = __funnelshift_r(b4[1].native, b4[2].native, srcMisalign*8);
            b4[2].native = __funnelshift_r(b4[2].native, b4[3].native, srcMisalign*8);
            b4[3].native = __funnelshift_r(b4[3].native, b4_4.native, srcMisalign*8);
        }
        
        // Store 16-byte chunk
        if (Multimem) {
            multimem_st_global<16>(dstAddr, b16);
        } else {
            st_global<16>(dstAddr, b16);
        }

        srcAddr += WARP_SIZE*16;
        dstAddr += WARP_SIZE*16;
        nMiddleBytes -= WARP_SIZE*16;
    }
}

#endif /* OP128_CPU_H_ */
