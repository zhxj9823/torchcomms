/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_CUDAWRAP_H_
#define NCCL_CUDAWRAP_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "checks.h"

#include "comms/ctran/utils/ErrorStackTraceUtil.h"

// External CUDA Driver API functions for NEX CPU emulation interception
// These are resolved at link/load time via nex_cuda.so (LD_PRELOAD) instead of
// using cudaGetDriverEntryPoint which doesn't work under NEX.
extern CUresult cuDeviceGet(CUdevice *device, int ordinal);
extern CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
extern CUresult cuGetErrorString(CUresult error, const char **pStr);
extern CUresult cuGetErrorName(CUresult error, const char **pStr);
extern CUresult cuMemGetAddressRange_v2(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr);
extern CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev);
extern CUresult cuCtxDestroy(CUcontext ctx);
extern CUresult cuCtxGetCurrent(CUcontext *pctx);
extern CUresult cuCtxSetCurrent(CUcontext ctx);
extern CUresult cuCtxGetDevice(CUdevice *device);
extern CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr);
extern CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                              unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                              unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
extern CUresult cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f, void **kernelParams, void **extra);
extern CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags);
extern CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size);
extern CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size, const CUmemAllocationProp *prop, unsigned long long flags);
extern CUresult cuMemGetAllocationGranularity(size_t *granularity, const CUmemAllocationProp *prop, CUmemAllocationGranularity_flags option);
extern CUresult cuMemExportToShareableHandle(void *shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags);
extern CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *handle, void *osHandle, CUmemAllocationHandleType shHandleType);
extern CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags);
extern CUresult cuMemRelease(CUmemGenericAllocationHandle handle);
extern CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *handle, void *addr);
extern CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc, size_t count);
extern CUresult cuMemUnmap(CUdeviceptr ptr, size_t size);
extern CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp *prop, CUmemGenericAllocationHandle handle);
extern CUresult cuMemGetHandleForAddressRange(void *handle, CUdeviceptr dptr, size_t size, CUmemRangeHandleType handleType, unsigned long long flags);
extern CUresult cuMulticastAddDevice(CUmemGenericAllocationHandle mcHandle, CUdevice dev);
extern CUresult cuMulticastBindMem(CUmemGenericAllocationHandle mcHandle, size_t mcOffset, CUmemGenericAllocationHandle memHandle, size_t memOffset, size_t size, unsigned long long flags);
extern CUresult cuMulticastBindAddr(CUmemGenericAllocationHandle mcHandle, size_t mcOffset, CUdeviceptr memptr, size_t size, unsigned long long flags);
extern CUresult cuMulticastCreate(CUmemGenericAllocationHandle *mcHandle, const void *prop);
extern CUresult cuMulticastGetGranularity(size_t *granularity, const void *prop, unsigned int option);
extern CUresult cuMulticastUnbind(CUmemGenericAllocationHandle mcHandle, CUdevice dev, size_t mcOffset, size_t size);

// Is cuMem API usage enabled
extern int ncclCuMemEnable();
extern int ncclCuMemHostEnable();

#if CUDART_VERSION >= 11030
#include <cudaTypedefs.h>

// Handle type used for cuMemCreate()
extern CUmemAllocationHandleType ncclCuMemHandleType;

#endif

#define CUPFN(symbol) symbol

// Check CUDA driver calls
#define CUCHECK(cmd) do {				      \
    CUresult err = cmd;					      \
    if( err != CUDA_SUCCESS ) {				      \
      const char *errStr;				      \
      (void) cuGetErrorString(err, &errStr);		      \
      ERR("Cuda failure %d '%s'", err, errStr);	      \
      ErrorStackTraceUtil::logErrorMessage("Cuda Error: " + std::string(errStr));			      \
      return ncclUnhandledCudaError;			      \
    }							      \
} while(false)

#define CUCALL(cmd) do {				      \
    cmd;					                \
} while(false)

#define CUCHECKGOTO(cmd, res, label) do {		      \
    CUresult err = cmd;					      \
    if( err != CUDA_SUCCESS ) {				      \
      const char *errStr;				      \
      (void) cuGetErrorString(err, &errStr);		      \
      WARN("Cuda failure %d '%s'", err, errStr);	      \
      ErrorStackTraceUtil::logErrorMessage("Cuda Error: " + std::string(errStr));			      \
      res = ncclUnhandledCudaError;			      \
      goto label;					      \
    }							      \
} while(false)

// Report failure but clear error and continue
#define CUCHECKIGNORE(cmd) do {						\
    CUresult err = cmd;							\
    if( err != CUDA_SUCCESS ) {						\
      const char *errStr;						\
      (void) cuGetErrorString(err, &errStr);				\
      INFO(NCCL_ALL,"%s:%d Cuda failure %d '%s'", __FILE__, __LINE__, err, errStr); \
    }									\
} while(false)

#define CUCHECKTHREAD(cmd, args) do {					\
    CUresult err = cmd;							\
    if (err != CUDA_SUCCESS) {						\
      INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, err); \
      args->ret = ncclUnhandledCudaError;				\
      return args;							\
    }									\
} while(0)

#define DECLARE_CUDA_PFN_EXTERN(symbol,version) // extern PFN_##symbol##_v##version pfn_##symbol

#if CUDART_VERSION >= 11030
// CUDA Driver functions - using external declarations for NEX CPU emulation
// PFN function pointer declarations are not needed; symbols resolved via nex_cuda.so
/*
DECLARE_CUDA_PFN_EXTERN(cuDeviceGet, 2000);
DECLARE_CUDA_PFN_EXTERN(cuDeviceGetAttribute, 2000);
DECLARE_CUDA_PFN_EXTERN(cuGetErrorString, 6000);
DECLARE_CUDA_PFN_EXTERN(cuGetErrorName, 6000);
DECLARE_CUDA_PFN_EXTERN(cuMemGetAddressRange, 3020);
DECLARE_CUDA_PFN_EXTERN(cuCtxCreate, 11040);
DECLARE_CUDA_PFN_EXTERN(cuCtxDestroy, 4000);
DECLARE_CUDA_PFN_EXTERN(cuCtxGetCurrent, 4000);
DECLARE_CUDA_PFN_EXTERN(cuCtxSetCurrent, 4000);
DECLARE_CUDA_PFN_EXTERN(cuCtxGetDevice, 2000);
DECLARE_CUDA_PFN_EXTERN(cuPointerGetAttribute, 4000);
DECLARE_CUDA_PFN_EXTERN(cuLaunchKernel, 4000);
#if CUDART_VERSION >= 11080
DECLARE_CUDA_PFN_EXTERN(cuLaunchKernelEx, 11060);
#endif
// cuMem API support
DECLARE_CUDA_PFN_EXTERN(cuMemAddressReserve, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemAddressFree, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemCreate, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemGetAllocationGranularity, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemExportToShareableHandle, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemImportFromShareableHandle, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemMap, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemRelease, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemRetainAllocationHandle, 11000);
DECLARE_CUDA_PFN_EXTERN(cuMemSetAccess, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemUnmap, 10020);
DECLARE_CUDA_PFN_EXTERN(cuMemGetAllocationPropertiesFromHandle, 10020);
#if CUDA_VERSION >= 11070
DECLARE_CUDA_PFN_EXTERN(cuMemGetHandleForAddressRange, 11070); // DMA-BUF support
#endif
#if CUDA_VERSION >= 12010
// NVSwitch Multicast support
DECLARE_CUDA_PFN_EXTERN(cuMulticastAddDevice, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastBindMem, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastBindAddr, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastCreate, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastGetGranularity, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastUnbind, 12010);
#endif
*/
#endif

ncclResult_t ncclCudaLibraryInit(void);

extern int ncclCudaDriverVersionCache;
extern bool ncclCudaLaunchBlocking; // initialized by ncclCudaLibraryInit()

inline ncclResult_t ncclCudaDriverVersion(int* driver) {
  int version = __atomic_load_n(&ncclCudaDriverVersionCache, __ATOMIC_RELAXED);
  if (version == -1) {
    CUDACHECK(cudaDriverGetVersion(&version));
    __atomic_store_n(&ncclCudaDriverVersionCache, version, __ATOMIC_RELAXED);
  }
  *driver = version;
  return ncclSuccess;
}

// NCCLX - API
bool ncclGetCuMemSysSupported();

#endif
