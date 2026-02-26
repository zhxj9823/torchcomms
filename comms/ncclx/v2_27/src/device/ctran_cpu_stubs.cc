// ctran_cpu_stubs.cc - No-op CPU stub implementations of ctran GPU kernels.
//
// These stubs provide symbol definitions for ctran GPU kernel functions
// referenced at library load time via global kernel pointer arrays in files
// like AllToAllDedup.cc, SendRecv.cc, etc.
// In CPU emulation mode, these kernels are never actually called.
//
// Porting ctran .cuh device kernels to CPU is future work.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// CTRAN_DISABLE_TCPDM is defined in CXXFLAGS -> SQueues is empty struct
#include "comms/ctran/algos/AllToAll/Types.h"
#include "comms/ctran/algos/AllGather/Types.h"
#include "comms/ctran/algos/AllReduce/Types.h"
#include "comms/ctran/algos/AllReduce/AllReduceRingCommon.cuh"
#include "comms/ctran/algos/RMA/Types.h"
#include "comms/ctran/algos/SendRecv/Types.h"
#include "comms/ctran/algos/Broadcast/Types.h"
#include "comms/ctran/algos/ReduceScatter/Types.h"
#include "comms/ctran/algos/AllGatherP/Types.h"
#include "comms/ctran/algos/AllToAllvDedup/Types.h"
#include "comms/utils/commSpecs.h"

// CtranAlgoDeviceState is always passed by pointer; included via some Types.h
// but we forward-declare defensively in case include order varies.
struct CtranAlgoDeviceState;
struct KernelElem;

// ===========================================================================
// Non-template kernel stubs
// ===========================================================================

void ncclKernelSend(int* flag, CtranAlgoDeviceState* devState, ctran::sendrecv::KernelSendArgs args) {}
void ncclKernelSendNotifyOnly(int* flag, CtranAlgoDeviceState* devState, ctran::sendrecv::KernelSendArgs args) {}
void ncclKernelRecvNotifyOnly(int* flag, CtranAlgoDeviceState* devState, ctran::sendrecv::KernelRecvArgs args) {}
void ncclKernelSendRecvNotifyOnly(int* flag, CtranAlgoDeviceState* devState, ctran::sendrecv::KernelSendRecvArgs args) {}
void ncclKernelSendRecvP2p(int* flag, CtranAlgoDeviceState* devState, ctran::sendrecv::KernArgs args) {}
void ncclKernelSendRecvStaged(int* flag, CtranAlgoDeviceState* devState, ctran::sendrecv::KernArgs args) {}
void ncclKernelGet(int* flag, CtranAlgoDeviceState* devState) {}
void ncclKernelPut(int* flag, CtranAlgoDeviceState* devState) {}
void ncclKernelPutSignal(int* flag, CtranAlgoDeviceState* devState, CtranKernelPutSignalArgs args) {}
void ncclKernelWaitSignal(int* flag, CtranAlgoDeviceState* devState, CtranKernelWaitSignalArgs args) {}
void ncclKernelSignal(int* flag, CtranAlgoDeviceState* devState, CtranKernelSignalArgs args) {}
void ncclKernelNvlBarrier(int rank, int nLocalRanks, CtranAlgoDeviceState* devState) {}
void ncclKernelAllGatherCtranRing(int* flag, CtranAlgoDeviceState* devState, ctran::allgather::KernelArgs args) {}
void ncclKernelAllGatherCtranRecDbl(int* flag, CtranAlgoDeviceState* devState, ctran::allgather::KernelArgs args) {}

namespace ctran::allgatherp {
void ncclKernelAllGatherPDirect(int* flag, CtranAlgoDeviceState* devState) {}
void ncclKernelAllGatherPInit(int* flag, CtranAlgoDeviceState* devState) {}
void ncclKernelAllGatherPPipe(int* flag, CtranAlgoDeviceState* devState) {}
void ncclKernelAllGatherPPipeStart(int* flag, CtranAlgoDeviceState* devState) {}
void ncclKernelAllGatherPPipeEnd(int* flag, CtranAlgoDeviceState* devState, PipeEndKernArgs args) {}
void ncclKernelAllGatherPPipeSync(int* flag, CtranAlgoDeviceState* devState, PipeSyncKernArgs args) {}
} // namespace ctran::allgatherp

namespace ctran::alltoallvdedup {
void ncclKernelAllToAllvDedupPrepare(CtranAlgoDeviceState* devState, ExecKernArgs execArgs, PrepareConfig config, int numWorkers) {}
void ncclKernelAllToAllvDedupPrepareReset(ExecKernArgs execArgs, int a, int b) {}
} // namespace ctran::alltoallvdedup

// checksumKernel
template <int Threads>
void checksumKernel(const uint8_t* in, const uint32_t size, uint32_t* out) {}
template void checksumKernel<1024>(const uint8_t*, const uint32_t, uint32_t*);

// ===========================================================================
// Bool-template kernel stubs
// ===========================================================================

template <bool B>
void ncclKernelRecv(int* flag, CtranAlgoDeviceState* devState, ctran::sendrecv::KernelRecvArgs args) {}
template void ncclKernelRecv<false>(int*, CtranAlgoDeviceState*, ctran::sendrecv::KernelRecvArgs);
template void ncclKernelRecv<true>(int*, CtranAlgoDeviceState*, ctran::sendrecv::KernelRecvArgs);

template <bool B>
void ncclKernelSendRecv(int* flag, CtranAlgoDeviceState* devState, ctran::sendrecv::KernelSendRecvArgs args) {}
template void ncclKernelSendRecv<false>(int*, CtranAlgoDeviceState*, ctran::sendrecv::KernelSendRecvArgs);
template void ncclKernelSendRecv<true>(int*, CtranAlgoDeviceState*, ctran::sendrecv::KernelSendRecvArgs);

template <bool B>
void ncclKernelBroadcast(int* flag, CtranAlgoDeviceState* devState, ctran::broadcast::KernelArgs args) {}
template void ncclKernelBroadcast<false>(int*, CtranAlgoDeviceState*, ctran::broadcast::KernelArgs);
template void ncclKernelBroadcast<true>(int*, CtranAlgoDeviceState*, ctran::broadcast::KernelArgs);

// ===========================================================================
// Type-template kernel stubs (10 data types)
// ===========================================================================

// ncclKernelAllToAll
template <typename T>
void ncclKernelAllToAll(int* flag, CtranAlgoDeviceState* devState, ctran::alltoall::KernelArgs args) {}
template void ncclKernelAllToAll<int8_t>(int*, CtranAlgoDeviceState*, ctran::alltoall::KernelArgs);
template void ncclKernelAllToAll<uint8_t>(int*, CtranAlgoDeviceState*, ctran::alltoall::KernelArgs);
template void ncclKernelAllToAll<int32_t>(int*, CtranAlgoDeviceState*, ctran::alltoall::KernelArgs);
template void ncclKernelAllToAll<uint32_t>(int*, CtranAlgoDeviceState*, ctran::alltoall::KernelArgs);
template void ncclKernelAllToAll<int64_t>(int*, CtranAlgoDeviceState*, ctran::alltoall::KernelArgs);
template void ncclKernelAllToAll<uint64_t>(int*, CtranAlgoDeviceState*, ctran::alltoall::KernelArgs);
template void ncclKernelAllToAll<__half>(int*, CtranAlgoDeviceState*, ctran::alltoall::KernelArgs);
template void ncclKernelAllToAll<float>(int*, CtranAlgoDeviceState*, ctran::alltoall::KernelArgs);
template void ncclKernelAllToAll<double>(int*, CtranAlgoDeviceState*, ctran::alltoall::KernelArgs);
template void ncclKernelAllToAll<__nv_bfloat16>(int*, CtranAlgoDeviceState*, ctran::alltoall::KernelArgs);

// ncclKernelAllToAllv
template <typename T>
void ncclKernelAllToAllv(int* flag, CtranAlgoDeviceState* devState, ctran::alltoallv::KernelArgs args) {}
template void ncclKernelAllToAllv<int8_t>(int*, CtranAlgoDeviceState*, ctran::alltoallv::KernelArgs);
template void ncclKernelAllToAllv<uint8_t>(int*, CtranAlgoDeviceState*, ctran::alltoallv::KernelArgs);
template void ncclKernelAllToAllv<int32_t>(int*, CtranAlgoDeviceState*, ctran::alltoallv::KernelArgs);
template void ncclKernelAllToAllv<uint32_t>(int*, CtranAlgoDeviceState*, ctran::alltoallv::KernelArgs);
template void ncclKernelAllToAllv<int64_t>(int*, CtranAlgoDeviceState*, ctran::alltoallv::KernelArgs);
template void ncclKernelAllToAllv<uint64_t>(int*, CtranAlgoDeviceState*, ctran::alltoallv::KernelArgs);
template void ncclKernelAllToAllv<__half>(int*, CtranAlgoDeviceState*, ctran::alltoallv::KernelArgs);
template void ncclKernelAllToAllv<float>(int*, CtranAlgoDeviceState*, ctran::alltoallv::KernelArgs);
template void ncclKernelAllToAllv<double>(int*, CtranAlgoDeviceState*, ctran::alltoallv::KernelArgs);
template void ncclKernelAllToAllv<__nv_bfloat16>(int*, CtranAlgoDeviceState*, ctran::alltoallv::KernelArgs);

// ncclKernelAllToAllDedup
template <typename T>
void ncclKernelAllToAllDedup(int* flag, CtranAlgoDeviceState* devState, ctran::alltoalldedup::KernelArgs args) {}
template void ncclKernelAllToAllDedup<int8_t>(int*, CtranAlgoDeviceState*, ctran::alltoalldedup::KernelArgs);
template void ncclKernelAllToAllDedup<uint8_t>(int*, CtranAlgoDeviceState*, ctran::alltoalldedup::KernelArgs);
template void ncclKernelAllToAllDedup<int32_t>(int*, CtranAlgoDeviceState*, ctran::alltoalldedup::KernelArgs);
template void ncclKernelAllToAllDedup<uint32_t>(int*, CtranAlgoDeviceState*, ctran::alltoalldedup::KernelArgs);
template void ncclKernelAllToAllDedup<int64_t>(int*, CtranAlgoDeviceState*, ctran::alltoalldedup::KernelArgs);
template void ncclKernelAllToAllDedup<uint64_t>(int*, CtranAlgoDeviceState*, ctran::alltoalldedup::KernelArgs);
template void ncclKernelAllToAllDedup<__half>(int*, CtranAlgoDeviceState*, ctran::alltoalldedup::KernelArgs);
template void ncclKernelAllToAllDedup<float>(int*, CtranAlgoDeviceState*, ctran::alltoalldedup::KernelArgs);
template void ncclKernelAllToAllDedup<double>(int*, CtranAlgoDeviceState*, ctran::alltoalldedup::KernelArgs);
template void ncclKernelAllToAllDedup<__nv_bfloat16>(int*, CtranAlgoDeviceState*, ctran::alltoalldedup::KernelArgs);

// ncclKernelAllGatherCtranDirect
template <typename T>
void ncclKernelAllGatherCtranDirect(int* flag, CtranAlgoDeviceState* devState, ctran::allgather::KernelArgs args) {}
template void ncclKernelAllGatherCtranDirect<int8_t>(int*, CtranAlgoDeviceState*, ctran::allgather::KernelArgs);
template void ncclKernelAllGatherCtranDirect<uint8_t>(int*, CtranAlgoDeviceState*, ctran::allgather::KernelArgs);
template void ncclKernelAllGatherCtranDirect<int32_t>(int*, CtranAlgoDeviceState*, ctran::allgather::KernelArgs);
template void ncclKernelAllGatherCtranDirect<uint32_t>(int*, CtranAlgoDeviceState*, ctran::allgather::KernelArgs);
template void ncclKernelAllGatherCtranDirect<int64_t>(int*, CtranAlgoDeviceState*, ctran::allgather::KernelArgs);
template void ncclKernelAllGatherCtranDirect<uint64_t>(int*, CtranAlgoDeviceState*, ctran::allgather::KernelArgs);
template void ncclKernelAllGatherCtranDirect<__half>(int*, CtranAlgoDeviceState*, ctran::allgather::KernelArgs);
template void ncclKernelAllGatherCtranDirect<float>(int*, CtranAlgoDeviceState*, ctran::allgather::KernelArgs);
template void ncclKernelAllGatherCtranDirect<double>(int*, CtranAlgoDeviceState*, ctran::allgather::KernelArgs);
template void ncclKernelAllGatherCtranDirect<__nv_bfloat16>(int*, CtranAlgoDeviceState*, ctran::allgather::KernelArgs);

// ncclKernelAllToAllvDynamic
template <typename T>
void ncclKernelAllToAllvDynamic(int* flag, CtranAlgoDeviceState* devState, ctran::alltoallvdynamic::KernelArgs args) {}
template void ncclKernelAllToAllvDynamic<int8_t>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);
template void ncclKernelAllToAllvDynamic<uint8_t>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);
template void ncclKernelAllToAllvDynamic<int32_t>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);
template void ncclKernelAllToAllvDynamic<uint32_t>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);
template void ncclKernelAllToAllvDynamic<int64_t>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);
template void ncclKernelAllToAllvDynamic<uint64_t>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);
template void ncclKernelAllToAllvDynamic<__half>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);
template void ncclKernelAllToAllvDynamic<float>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);
template void ncclKernelAllToAllvDynamic<double>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);
template void ncclKernelAllToAllvDynamic<__nv_bfloat16>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);

// ncclKernelAllToAllvDynamicSplit
template <typename T>
void ncclKernelAllToAllvDynamicSplit(int* flag, CtranAlgoDeviceState* devState, ctran::alltoallvdynamic::KernelArgs args) {}
template void ncclKernelAllToAllvDynamicSplit<int8_t>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);
template void ncclKernelAllToAllvDynamicSplit<uint8_t>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);
template void ncclKernelAllToAllvDynamicSplit<int32_t>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);
template void ncclKernelAllToAllvDynamicSplit<uint32_t>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);
template void ncclKernelAllToAllvDynamicSplit<int64_t>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);
template void ncclKernelAllToAllvDynamicSplit<uint64_t>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);
template void ncclKernelAllToAllvDynamicSplit<__half>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);
template void ncclKernelAllToAllvDynamicSplit<float>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);
template void ncclKernelAllToAllvDynamicSplit<double>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);
template void ncclKernelAllToAllvDynamicSplit<__nv_bfloat16>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);

// ncclKernelAllToAllvDynamicSplitNonContig
template <typename T>
void ncclKernelAllToAllvDynamicSplitNonContig(int* flag, CtranAlgoDeviceState* devState, ctran::alltoallvdynamic::KernelArgs args) {}
template void ncclKernelAllToAllvDynamicSplitNonContig<int8_t>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);
template void ncclKernelAllToAllvDynamicSplitNonContig<uint8_t>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);
template void ncclKernelAllToAllvDynamicSplitNonContig<int32_t>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);
template void ncclKernelAllToAllvDynamicSplitNonContig<uint32_t>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);
template void ncclKernelAllToAllvDynamicSplitNonContig<int64_t>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);
template void ncclKernelAllToAllvDynamicSplitNonContig<uint64_t>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);
template void ncclKernelAllToAllvDynamicSplitNonContig<__half>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);
template void ncclKernelAllToAllvDynamicSplitNonContig<float>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);
template void ncclKernelAllToAllvDynamicSplitNonContig<double>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);
template void ncclKernelAllToAllvDynamicSplitNonContig<__nv_bfloat16>(int*, CtranAlgoDeviceState*, ctran::alltoallvdynamic::KernelArgs);

// ctran::alltoallvdedup::ncclKernelAllToAllvDedup
namespace ctran::alltoallvdedup {
template <typename T>
void ncclKernelAllToAllvDedup(int* flag, CtranAlgoDeviceState* devState, ExecKernArgs args) {}
template void ncclKernelAllToAllvDedup<int8_t>(int*, CtranAlgoDeviceState*, ExecKernArgs);
template void ncclKernelAllToAllvDedup<uint8_t>(int*, CtranAlgoDeviceState*, ExecKernArgs);
template void ncclKernelAllToAllvDedup<int32_t>(int*, CtranAlgoDeviceState*, ExecKernArgs);
template void ncclKernelAllToAllvDedup<uint32_t>(int*, CtranAlgoDeviceState*, ExecKernArgs);
template void ncclKernelAllToAllvDedup<int64_t>(int*, CtranAlgoDeviceState*, ExecKernArgs);
template void ncclKernelAllToAllvDedup<uint64_t>(int*, CtranAlgoDeviceState*, ExecKernArgs);
template void ncclKernelAllToAllvDedup<__half>(int*, CtranAlgoDeviceState*, ExecKernArgs);
template void ncclKernelAllToAllvDedup<float>(int*, CtranAlgoDeviceState*, ExecKernArgs);
template void ncclKernelAllToAllvDedup<double>(int*, CtranAlgoDeviceState*, ExecKernArgs);
template void ncclKernelAllToAllvDedup<__nv_bfloat16>(int*, CtranAlgoDeviceState*, ExecKernArgs);
} // namespace ctran::alltoallvdedup

// ===========================================================================
// Type x RedOp template kernel stubs
// ===========================================================================

// ncclKernelAllReduceCtranDirect
template <typename T, commRedOp_t op>
void ncclKernelAllReduceCtranDirect(int* flag, CtranAlgoDeviceState* devState, ctran::allreduce::KernelArgs args) {}
template void ncclKernelAllReduceCtranDirect<int8_t, commSum>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<int8_t, commProd>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<int8_t, commMax>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<int8_t, commMin>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<int8_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<uint8_t, commSum>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<uint8_t, commProd>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<uint8_t, commMax>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<uint8_t, commMin>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<uint8_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<int32_t, commSum>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<int32_t, commProd>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<int32_t, commMax>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<int32_t, commMin>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<int32_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<uint32_t, commSum>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<uint32_t, commProd>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<uint32_t, commMax>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<uint32_t, commMin>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<uint32_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<int64_t, commSum>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<int64_t, commProd>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<int64_t, commMax>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<int64_t, commMin>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<int64_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<uint64_t, commSum>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<uint64_t, commProd>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<uint64_t, commMax>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<uint64_t, commMin>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<uint64_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<__half, commSum>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<__half, commProd>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<__half, commMax>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<__half, commMin>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<__half, commAvg>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<float, commSum>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<float, commProd>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<float, commMax>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<float, commMin>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<float, commAvg>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<double, commSum>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<double, commProd>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<double, commMax>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<double, commMin>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<double, commAvg>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<__nv_bfloat16, commSum>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<__nv_bfloat16, commProd>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<__nv_bfloat16, commMax>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<__nv_bfloat16, commMin>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);
template void ncclKernelAllReduceCtranDirect<__nv_bfloat16, commAvg>(int*, CtranAlgoDeviceState*, ctran::allreduce::KernelArgs);

// ncclKernelAllReduceCtranRing
template <typename T, commRedOp_t op>
void ncclKernelAllReduceCtranRing(int* flag, CtranAlgoDeviceState* devState, ctran::allreduce::ring::KernArgs args) {}
template void ncclKernelAllReduceCtranRing<int8_t, commSum>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<int8_t, commProd>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<int8_t, commMax>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<int8_t, commMin>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<int8_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<uint8_t, commSum>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<uint8_t, commProd>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<uint8_t, commMax>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<uint8_t, commMin>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<uint8_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<int32_t, commSum>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<int32_t, commProd>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<int32_t, commMax>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<int32_t, commMin>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<int32_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<uint32_t, commSum>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<uint32_t, commProd>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<uint32_t, commMax>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<uint32_t, commMin>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<uint32_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<int64_t, commSum>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<int64_t, commProd>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<int64_t, commMax>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<int64_t, commMin>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<int64_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<uint64_t, commSum>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<uint64_t, commProd>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<uint64_t, commMax>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<uint64_t, commMin>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<uint64_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<__half, commSum>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<__half, commProd>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<__half, commMax>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<__half, commMin>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<__half, commAvg>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<float, commSum>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<float, commProd>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<float, commMax>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<float, commMin>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<float, commAvg>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<double, commSum>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<double, commProd>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<double, commMax>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<double, commMin>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<double, commAvg>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<__nv_bfloat16, commSum>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<__nv_bfloat16, commProd>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<__nv_bfloat16, commMax>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<__nv_bfloat16, commMin>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);
template void ncclKernelAllReduceCtranRing<__nv_bfloat16, commAvg>(int*, CtranAlgoDeviceState*, ctran::allreduce::ring::KernArgs);

// ncclKernelReduceScatterDirect
template <typename T, commRedOp_t op>
void ncclKernelReduceScatterDirect(int* flag, CtranAlgoDeviceState* devState, ctran::reducescatter::KernelArgs args) {}
template void ncclKernelReduceScatterDirect<int8_t, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<int8_t, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<int8_t, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<int8_t, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<int8_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<uint8_t, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<uint8_t, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<uint8_t, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<uint8_t, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<uint8_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<int32_t, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<int32_t, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<int32_t, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<int32_t, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<int32_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<uint32_t, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<uint32_t, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<uint32_t, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<uint32_t, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<uint32_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<int64_t, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<int64_t, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<int64_t, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<int64_t, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<int64_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<uint64_t, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<uint64_t, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<uint64_t, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<uint64_t, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<uint64_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<__half, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<__half, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<__half, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<__half, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<__half, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<float, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<float, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<float, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<float, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<float, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<double, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<double, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<double, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<double, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<double, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<__nv_bfloat16, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<__nv_bfloat16, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<__nv_bfloat16, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<__nv_bfloat16, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterDirect<__nv_bfloat16, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);

// ncclKernelReduceScatterRHD
template <typename T, commRedOp_t op>
void ncclKernelReduceScatterRHD(int* flag, CtranAlgoDeviceState* devState, ctran::reducescatter::KernelArgs args) {}
template void ncclKernelReduceScatterRHD<int8_t, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<int8_t, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<int8_t, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<int8_t, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<int8_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<uint8_t, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<uint8_t, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<uint8_t, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<uint8_t, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<uint8_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<int32_t, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<int32_t, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<int32_t, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<int32_t, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<int32_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<uint32_t, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<uint32_t, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<uint32_t, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<uint32_t, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<uint32_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<int64_t, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<int64_t, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<int64_t, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<int64_t, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<int64_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<uint64_t, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<uint64_t, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<uint64_t, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<uint64_t, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<uint64_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<__half, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<__half, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<__half, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<__half, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<__half, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<float, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<float, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<float, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<float, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<float, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<double, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<double, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<double, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<double, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<double, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<__nv_bfloat16, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<__nv_bfloat16, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<__nv_bfloat16, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<__nv_bfloat16, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRHD<__nv_bfloat16, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);

// ncclKernelReduceScatterRing
template <typename T, commRedOp_t op>
void ncclKernelReduceScatterRing(int* flag, CtranAlgoDeviceState* devState, ctran::reducescatter::KernelArgs args) {}
template void ncclKernelReduceScatterRing<int8_t, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<int8_t, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<int8_t, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<int8_t, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<int8_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<uint8_t, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<uint8_t, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<uint8_t, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<uint8_t, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<uint8_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<int32_t, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<int32_t, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<int32_t, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<int32_t, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<int32_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<uint32_t, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<uint32_t, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<uint32_t, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<uint32_t, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<uint32_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<int64_t, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<int64_t, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<int64_t, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<int64_t, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<int64_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<uint64_t, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<uint64_t, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<uint64_t, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<uint64_t, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<uint64_t, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<__half, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<__half, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<__half, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<__half, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<__half, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<float, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<float, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<float, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<float, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<float, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<double, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<double, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<double, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<double, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<double, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<__nv_bfloat16, commSum>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<__nv_bfloat16, commProd>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<__nv_bfloat16, commMax>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<__nv_bfloat16, commMin>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
template void ncclKernelReduceScatterRing<__nv_bfloat16, commAvg>(int*, CtranAlgoDeviceState*, ctran::reducescatter::KernelArgs);
