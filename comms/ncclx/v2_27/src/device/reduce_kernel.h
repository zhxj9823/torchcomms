#ifndef NCCL_REDUCE_KERNEL_CPU_H_
#define NCCL_REDUCE_KERNEL_CPU_H_

#include <cstdint>
#include <type_traits>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <limits>

#include "op128.h"

#include <cuda_fp16.h>
#if defined(__CUDA_BF16_TYPES_EXIST__)
#include <cuda_bf16.h>
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__)
#include <cuda_fp8.h>
#endif

#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__
#define __forceinline__
#endif

////////////////////////////////////////////////////////////////////////////////
// Floating-point trait

template<typename T>
struct IsFloatingPoint : std::false_type {};
template<> struct IsFloatingPoint<__half> : std::true_type {};
#if defined(__CUDA_BF16_TYPES_EXIST__)
template<> struct IsFloatingPoint<__nv_bfloat16> : std::true_type {};
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__)
template<> struct IsFloatingPoint<__nv_fp8_e4m3> : std::true_type {};
template<> struct IsFloatingPoint<__nv_fp8_e5m2> : std::true_type {};
#endif
template<> struct IsFloatingPoint<float>  : std::true_type {};
template<> struct IsFloatingPoint<double> : std::true_type {};

////////////////////////////////////////////////////////////////////////////////
// Reduction function classes: no-op constructors

template<typename T>
struct FuncCopy { using EltType = T; __host__ __device__ FuncCopy(uint64_t=0){} };

template<typename T>
struct FuncSum { using EltType = T; __host__ __device__ FuncSum(uint64_t=0){} };

template<typename T>
struct FuncProd { using EltType = T; __host__ __device__ FuncProd(uint64_t=0){} };

template<typename T>
struct FuncMinMax {
  using EltType = T;
  uint64_t xormask;
  bool isMin;
  __host__ __device__ FuncMinMax(uint64_t op=0) : xormask(op), isMin((op & 1) == 0) {}
};

template<typename T>
struct FuncPreMulSum {
  using EltType = T;
  uint64_t raw;
  __host__ __device__ FuncPreMulSum(uint64_t op=0) : raw(op) {}
};

template<typename T>
struct FuncSumPostDiv {
  using EltType = T;
  uint32_t divisor;
  bool isSigned;
  __host__ __device__ FuncSumPostDiv(uint64_t op=0)
      : divisor(static_cast<uint32_t>(op >> 1)), isSigned((op & 1) != 0) {
    if (divisor == 0) divisor = 1;
  }
  __host__ __device__ T divide(T x) const {
    using Unsigned = std::conditional_t<sizeof(T)==8, uint64_t,
                    std::conditional_t<sizeof(T)==4, uint32_t,
                    std::conditional_t<sizeof(T)==2, uint16_t,
                    uint8_t>>>;
    using Signed = std::make_signed_t<Unsigned>;
    Unsigned u = static_cast<Unsigned>(x);
    if (!isSigned) {
      return static_cast<T>(static_cast<Unsigned>(u / divisor));
    } else {
      Signed s = static_cast<Signed>(u);
      Signed q = s / static_cast<Signed>(divisor);
      return static_cast<T>(static_cast<Unsigned>(q));
    }
  }
};

////////////////////////////////////////////////////////////////////////////////
// RedOpArg trait

template<typename Fn>
struct RedOpArg { static constexpr bool ArgUsed = false;
  __host__ __device__ static uint64_t loadArg(void*) { return 0; }
};

template<typename T>
struct RedOpArg<FuncMinMax<T>> {
  static constexpr bool ArgUsed = true;
  __host__ __device__ static uint64_t loadArg(void* ptr) {
    uint64_t raw = 0;
    std::memcpy(&raw, ptr, sizeof(T));
    return raw;
  }
};

template<typename T>
struct RedOpArg<FuncPreMulSum<T>> : RedOpArg<FuncMinMax<T>> {};

template<typename T>
struct RedOpArg<FuncSumPostDiv<T>> {
  static constexpr bool ArgUsed = true;
  __host__ __device__ static uint64_t loadArg(void* ptr) {
    uint64_t raw = 0;
    std::memcpy(&raw, ptr, sizeof(raw));
    return raw;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Packing and reduction trait stubs

template<typename A, typename B, int EltPerPack>
struct Apply_Cast {
  __host__ __device__ static BytePack<EltPerPack*sizeof(B)> cast(BytePack<EltPerPack*sizeof(A)> in) {
    BytePack<EltPerPack*sizeof(B)> out;
    A src[EltPerPack];
    B dst[EltPerPack];
    std::memcpy(src, in.u8, sizeof(src));
    for (int i = 0; i < EltPerPack; ++i) dst[i] = static_cast<B>(src[i]);
    std::memcpy(out.u8, dst, sizeof(dst));
    return out;
  }
};

template<typename A, typename B>
struct Apply_Cast<A, B, 0> {
  __host__ __device__ static BytePack<0> cast(BytePack<0>) { return {}; }
};

template<typename Fn, int EltPerPack>
struct Apply_Reduce {
  using T = typename Fn::EltType;
  __host__ __device__ static BytePack<EltPerPack*sizeof(T)>
  reduce(Fn fn, BytePack<EltPerPack*sizeof(T)> a, BytePack<EltPerPack*sizeof(T)> b) {
    T va[EltPerPack];
    T vb[EltPerPack];
    std::memcpy(va, a.u8, sizeof(va));
    std::memcpy(vb, b.u8, sizeof(vb));
    for (int i = 0; i < EltPerPack; ++i) {
      va[i] = ncclReduceScalar(fn, va[i], vb[i]);
    }
    BytePack<EltPerPack*sizeof(T)> out;
    std::memcpy(out.u8, va, sizeof(va));
    return out;
  }
};

template<typename Fn>
struct Apply_Reduce<Fn, 0> {
  using T = typename Fn::EltType;
  __host__ __device__ static BytePack<0> reduce(Fn, BytePack<0>, BytePack<0>) { return {}; }
};

template<typename Fn, int EltPerPack>
struct Apply_PreOp {
  static constexpr bool IsIdentity = true;
  __host__ __device__ static BytePack<EltPerPack*sizeof(typename Fn::EltType)>
  preOp(Fn, BytePack<EltPerPack*sizeof(typename Fn::EltType)> in) {
    return in;
  }
};

template<typename Fn>
struct Apply_PreOp<Fn, 0> {
  static constexpr bool IsIdentity = true;
  __host__ __device__ static BytePack<0> preOp(Fn, BytePack<0> in) { return in; }
};

template<typename Fn, int EltPerPack>
struct Apply_PostOp {
  static constexpr bool IsIdentity = true;
  __host__ __device__ static BytePack<EltPerPack*sizeof(typename Fn::EltType)>
  postOp(Fn, BytePack<EltPerPack*sizeof(typename Fn::EltType)> in) {
    return in;
  }
};

template<typename Fn>
struct Apply_PostOp<Fn, 0> {
  static constexpr bool IsIdentity = true;
  __host__ __device__ static BytePack<0> postOp(Fn, BytePack<0> in) { return in; }
};

template<typename Fn, int BytePerPack>
struct Apply_LoadMultimem {
  __host__ __device__ static BytePack<BytePerPack> load(Fn, uintptr_t addr) {
    return ld_global<BytePerPack>(addr);
  }
};

template<typename Fn>
struct LoadMultimem_BigPackSize { static constexpr int BigPackSize = 0; };

#if !defined(__CUDA_ARCH__)

template<typename T>
__host__ __device__ inline T ncclDecodeScalar(uint64_t raw) {
  return static_cast<T>(raw);
}

template<>
__host__ __device__ inline float ncclDecodeScalar<float>(uint64_t raw) {
  uint32_t bits = static_cast<uint32_t>(raw & 0xffffffffu);
  float val;
  std::memcpy(&val, &bits, sizeof(val));
  return val;
}

template<>
__host__ __device__ inline double ncclDecodeScalar<double>(uint64_t raw) {
  double val;
  std::memcpy(&val, &raw, sizeof(val));
  return val;
}

template<typename T>
__host__ __device__ inline T ncclMultiply(T a, T b) { return a * b; }

template<typename T>
__host__ __device__ inline float ncclToFloat(T v) { return static_cast<float>(v); }

template<typename T>
__host__ __device__ inline T ncclFromFloat(float v) { return static_cast<T>(v); }

template<typename T>
__host__ __device__ inline T ncclAdd(T a, T b) { return a + b; }

struct Fp8Format {
  int mantBits;
  int expBits;
  int bias;
};

__host__ __device__ inline float decodeFp8(uint8_t bits, Fp8Format fmt) {
  int sign = bits >> 7;
  int expMask = (1 << fmt.expBits) - 1;
  int mantMask = (1 << fmt.mantBits) - 1;
  int exp = (bits >> fmt.mantBits) & expMask;
  int mant = bits & mantMask;
  float value;
  if (exp == 0) {
    if (mant == 0) {
      value = 0.0f;
    } else {
      float frac = mant / static_cast<float>(1 << fmt.mantBits);
      value = std::ldexp(frac, 1 - fmt.bias);
    }
  } else if (exp == expMask) {
    value = mant ? std::numeric_limits<float>::quiet_NaN()
                 : std::numeric_limits<float>::infinity();
  } else {
    float frac = 1.0f + mant / static_cast<float>(1 << fmt.mantBits);
    value = std::ldexp(frac, exp - fmt.bias);
  }
  return sign ? -value : value;
}

__host__ __device__ inline uint8_t encodeFp8(float value, Fp8Format fmt) {
  if (std::isnan(value)) {
    return static_cast<uint8_t>(((1 << fmt.expBits) - 1) << fmt.mantBits | ((1 << fmt.mantBits) - 1));
  }
  bool sign = std::signbit(value);
  float absVal = std::fabs(value);
  int signBit = sign ? 0x80 : 0;
  int expMask = (1 << fmt.expBits) - 1;
  int mantMask = (1 << fmt.mantBits) - 1;
  if (absVal == 0.0f) return static_cast<uint8_t>(signBit);
  float maxNormal = std::ldexp(2.0f - std::ldexp(1.0f, -fmt.mantBits), expMask - 1 - fmt.bias);
  if (absVal >= maxNormal) {
    return static_cast<uint8_t>(signBit | (expMask << fmt.mantBits));
  }
  int exp;
  float mant = std::frexp(absVal, &exp); // absVal = mant * 2^{exp}, mant in [0.5,1)
  mant *= 2.0f;
  exp -= 1;
  int expField = exp + fmt.bias;
  if (expField <= 0) {
    float sub = absVal / std::ldexp(1.0f, 1 - fmt.bias);
    int mantInt = static_cast<int>(std::round(sub * (1 << fmt.mantBits)));
    if (mantInt < 1) mantInt = 1;
    if (mantInt > mantMask) mantInt = mantMask;
    return static_cast<uint8_t>(signBit | mantInt);
  }
  int mantInt = static_cast<int>(std::round((mant - 1.0f) * (1 << fmt.mantBits)));
  if (mantInt > mantMask) {
    mantInt = 0;
    expField += 1;
    if (expField >= expMask) {
      return static_cast<uint8_t>(signBit | (expMask << fmt.mantBits));
    }
  }
  return static_cast<uint8_t>(signBit | (expField << fmt.mantBits) | (mantInt & mantMask));
}

template<>
__host__ __device__ inline __half ncclDecodeScalar<__half>(uint64_t raw) {
  uint16_t bits = static_cast<uint16_t>(raw & 0xffff);
  return __ushort_as_half(bits);
}

template<>
__host__ __device__ inline float ncclToFloat(__half v) { return __half2float(v); }

template<>
__host__ __device__ inline __half ncclFromFloat<__half>(float v) { return __float2half(v); }

template<>
__host__ __device__ inline __half ncclMultiply(__half a, __half b) {
  return __float2half(__half2float(a) * __half2float(b));
}

template<>
__host__ __device__ inline __half ncclAdd(__half a, __half b) {
  return __float2half(__half2float(a) + __half2float(b));
}

#if defined(__CUDA_BF16_TYPES_EXIST__)
template<>
__host__ __device__ inline __nv_bfloat16 ncclDecodeScalar<__nv_bfloat16>(uint64_t raw) {
  __nv_bfloat16_raw hr;
  hr.x = static_cast<unsigned short>(raw & 0xffff);
  return __nv_bfloat16(hr);
}

template<>
__host__ __device__ inline float ncclToFloat(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template<>
__host__ __device__ inline __nv_bfloat16 ncclFromFloat<__nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}

template<>
__host__ __device__ inline __nv_bfloat16 ncclMultiply(__nv_bfloat16 a, __nv_bfloat16 b) {
  float res = __bfloat162float(a) * __bfloat162float(b);
  return __float2bfloat16_rn(res);
}

template<>
__host__ __device__ inline __nv_bfloat16 ncclAdd(__nv_bfloat16 a, __nv_bfloat16 b) {
  float res = __bfloat162float(a) + __bfloat162float(b);
  return __float2bfloat16_rn(res);
}
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)
template<>
__host__ __device__ inline __nv_fp8_e4m3 ncclDecodeScalar<__nv_fp8_e4m3>(uint64_t raw) {
  return static_cast<__nv_fp8_e4m3>(static_cast<uint8_t>(raw & 0xff));
}

template<>
__host__ __device__ inline __nv_fp8_e5m2 ncclDecodeScalar<__nv_fp8_e5m2>(uint64_t raw) {
  return static_cast<__nv_fp8_e5m2>(static_cast<uint8_t>(raw & 0xff));
}

template<>
__host__ __device__ inline float ncclToFloat(__nv_fp8_e4m3 v) {
  return decodeFp8(static_cast<uint8_t>(v), {3, 4, 7});
}

template<>
__host__ __device__ inline float ncclToFloat(__nv_fp8_e5m2 v) {
  return decodeFp8(static_cast<uint8_t>(v), {2, 5, 15});
}

template<>
__host__ __device__ inline __nv_fp8_e4m3 ncclFromFloat<__nv_fp8_e4m3>(float v) {
  uint8_t bits = encodeFp8(v, {3, 4, 7});
  return static_cast<__nv_fp8_e4m3>(bits);
}

template<>
__host__ __device__ inline __nv_fp8_e5m2 ncclFromFloat<__nv_fp8_e5m2>(float v) {
  uint8_t bits = encodeFp8(v, {2, 5, 15});
  return static_cast<__nv_fp8_e5m2>(bits);
}

template<>
__host__ __device__ inline __nv_fp8_e4m3 ncclMultiply(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b) {
  float res = ncclToFloat(a) * ncclToFloat(b);
  return ncclFromFloat<__nv_fp8_e4m3>(res);
}

template<>
__host__ __device__ inline __nv_fp8_e5m2 ncclMultiply(__nv_fp8_e5m2 a, __nv_fp8_e5m2 b) {
  float res = ncclToFloat(a) * ncclToFloat(b);
  return ncclFromFloat<__nv_fp8_e5m2>(res);
}

template<>
__host__ __device__ inline __nv_fp8_e4m3 ncclAdd(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b) {
  float res = ncclToFloat(a) + ncclToFloat(b);
  return ncclFromFloat<__nv_fp8_e4m3>(res);
}

template<>
__host__ __device__ inline __nv_fp8_e5m2 ncclAdd(__nv_fp8_e5m2 a, __nv_fp8_e5m2 b) {
  float res = ncclToFloat(a) + ncclToFloat(b);
  return ncclFromFloat<__nv_fp8_e5m2>(res);
}
#endif
template<typename Fn, typename T>
__host__ __device__ inline T ncclReduceScalar(Fn, T current, T value) {
  return value;
}

#define SKIP_COMP

template<typename T>
__host__ __device__ inline T ncclReduceScalar(FuncCopy<T>, T current, T) { return current; }
template<typename T>
__host__ __device__ inline T ncclReduceScalar(FuncSum<T>, T current, T value) {
  // Skip reduction computation - return current (first value wins)
#ifdef SKIP_COMP
  return current; 
#endif 

  return ncclAdd(current, value);
}
template<typename T>
__host__ __device__ inline T ncclReduceScalar(FuncProd<T>, T current, T value) {
  // Skip reduction computation - return current (first value wins)
#ifdef SKIP_COMP
  return current; 
#endif 

  return ncclMultiply(current, value);
}
template<typename T>
__host__ __device__ inline T ncclReduceScalar(FuncMinMax<T> fn, T current, T value) {
  // Skip reduction computation - return current (first value wins)
#ifdef SKIP_COMP
  return current; 
#endif 

  if constexpr (std::is_same_v<T, __half>
#if defined(__CUDA_BF16_TYPES_EXIST__)
                || std::is_same_v<T, __nv_bfloat16>
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__)
                || std::is_same_v<T, __nv_fp8_e4m3>
                || std::is_same_v<T, __nv_fp8_e5m2>
#endif
               ) {
    float c = ncclToFloat(current);
    float v = ncclToFloat(value);
    float res = fn.isMin ? (v < c ? v : c) : (v > c ? v : c);
    return ncclFromFloat<T>(res);
  } else {
    return fn.isMin ? (value < current ? value : current)
                    : (value > current ? value : current);
  }
}
template<typename T>
__host__ __device__ inline T ncclReduceScalar(FuncPreMulSum<T>, T current, T value) {
  // Skip reduction computation - return current (first value wins)
#ifdef SKIP_COMP
  return current; 
#endif 

  return ncclAdd(current, value);
}
template<typename T>
__host__ __device__ inline T ncclReduceScalar(FuncSumPostDiv<T>, T current, T value) {
  // Skip reduction computation - return current (first value wins)
#ifdef SKIP_COMP
  return current; 
#endif 

  return ncclAdd(current, value);
}

template<typename T, int EltPerPack>
struct Apply_PreOp<FuncPreMulSum<T>, EltPerPack> {
  static constexpr bool IsIdentity = false;
  __host__ __device__ static BytePack<EltPerPack*sizeof(T)>
  preOp(FuncPreMulSum<T> fn, BytePack<EltPerPack*sizeof(T)> in) {
    // skip
#ifdef SKIP_COMP
    return in;
#endif 

    T vals[EltPerPack];
    std::memcpy(vals, in.u8, sizeof(vals));
    T factor = ncclDecodeScalar<T>(fn.raw);
    for (int i = 0; i < EltPerPack; ++i) {
      vals[i] = ncclMultiply(vals[i], factor);
    }
    BytePack<EltPerPack*sizeof(T)> out;
    std::memcpy(out.u8, vals, sizeof(vals));
    return out;
  }
};

template<typename T, int EltPerPack>
struct Apply_PostOp<FuncSumPostDiv<T>, EltPerPack> {
  static constexpr bool IsIdentity = false;
  __host__ __device__ static BytePack<EltPerPack*sizeof(T)>
  postOp(FuncSumPostDiv<T> fn, BytePack<EltPerPack*sizeof(T)> in) {
    // Skip
#ifdef SKIP_COMP
    return in;
#endif 

    T vals[EltPerPack];
    std::memcpy(vals, in.u8, sizeof(vals));
    for (int i = 0; i < EltPerPack; ++i) {
      vals[i] = fn.divide(vals[i]);
    }
    BytePack<EltPerPack*sizeof(T)> out;
    std::memcpy(out.u8, vals, sizeof(vals));
    return out;
  }
};

#endif

////////////////////////////////////////////////////////////////////////////////
// Public API stubs

template<typename A, typename B, typename PackA>
__host__ __device__ inline BytePack<BytePackOf<PackA>::Size*sizeof(B)/sizeof(A)>
applyCast(PackA pack) {
  constexpr int elt = BytePackOf<PackA>::Size/sizeof(A);
  auto asPack = toPack(pack);
  return Apply_Cast<A,B,elt>::cast(asPack);
}

template<typename Fn, typename Pack>
__host__ __device__ inline Pack applyReduce(Fn fn, Pack a, Pack b) {
  // skip
#ifdef SKIP_COMP
  return a;
#endif 

  constexpr int elt = BytePackOf<Pack>::Size/sizeof(typename Fn::EltType);
  auto aPack = toPack(a);
  auto bPack = toPack(b);
  auto result = Apply_Reduce<Fn, elt>::reduce(fn, aPack, bPack);
  return fromPack<Pack>(result);
}

template<typename Fn, typename Pack>
__host__ __device__ inline Pack applyPreOp(Fn fn, Pack a) {
  // skip
#ifdef SKIP_COMP
  return a;
#endif

  constexpr int elt = BytePackOf<Pack>::Size/sizeof(typename Fn::EltType);
  auto packValue = toPack(a);
  auto result = Apply_PreOp<Fn, elt>::preOp(fn, packValue);
  return fromPack<Pack>(result);
}

template<typename Fn, typename Pack>
__host__ __device__ inline Pack applyPostOp(Fn fn, Pack a) {
   // skip
#ifdef SKIP_COMP
  return a;
#endif

  constexpr int elt = BytePackOf<Pack>::Size/sizeof(typename Fn::EltType);
  auto packValue = toPack(a);
  auto result = Apply_PostOp<Fn, elt>::postOp(fn, packValue);
  return fromPack<Pack>(result);
}

template<typename Fn, int BytePerPack>
__host__ __device__ inline BytePack<BytePerPack> applyLoadMultimem(Fn fn, uintptr_t addr) {
  return Apply_LoadMultimem<Fn, BytePerPack>::load(fn, addr);
}

#endif // NCCL_REDUCE_KERNEL_CPU_H_
