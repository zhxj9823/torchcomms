/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "common.h"
#include "primitives.h"
// #include "all_reduce.h"
// #include "all_gather.h"
// #include "reduce_scatter.h"
// #include "broadcast.h"
// #include "sendrecv.h"

thread_local shared_memory<ncclShmemData> ncclShmem(sizeof(ncclShmemData));

thread_local shared_memory<ulong2> ncclShmemPerWarp(
    ncclShmemScratchWarpSize() * (NCCL_MAX_NTHREADS/WARP_SIZE)
);

// __shared__ ncclShmemData ncclShmem[1];
// __shared__ ulong2 ncclShmemPerWarp[ncclShmemScratchWarpSize()*(NCCL_MAX_NTHREADS/WARP_SIZE)/sizeof(ulong2)];


struct RunWorkNop {
  __device__ void run() {}
};

void ncclDevKernel_Generic(ncclDevKernelArgs4K const args4K) {
  ncclKernelMain<-1, RunWorkNop>(&args4K.args);
}

__device__ void ncclDevFunc_Nop() {}
