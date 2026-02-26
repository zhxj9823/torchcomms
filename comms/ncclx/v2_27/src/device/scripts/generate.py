#!/usr/bin/env python3
import os
import sys

# ---------------------------------------------------------------------------
# master lists (unchanged)
all_colls  = ["Broadcast","Reduce","AllGather","ReduceScatter","AllReduce","SendRecv"]
all_redops = ["Sum","Prod","MinMax","PreMulSum","SumPostDiv"]
all_tys    = ["i8","u8","i32","u32","i64","u64",
              "f16","f32","f64","bf16","f8e4m3","f8e5m2"]
all_protos = ["LL","LL128","SIMPLE"]
all_algos  = ["TREE","RING","COLLNET_DIRECT","COLLNET_CHAIN","NVLS","NVLS_TREE","PAT"]

# ---------------------------------------------------------------------------
# path of generated sources
gensrc = sys.argv[1]

if os.path.exists(gensrc):
  for name in os.listdir(gensrc):
    os.remove(os.path.join(gensrc, name))
else:
  os.mkdir(gensrc)

# ---------------------------------------------------------------------------
# optional regex filter for functions
def paste(sep, *args):
  return sep.join(x for x in args if x is not None)

func_pattern = sys.argv[2:3]
if func_pattern and func_pattern[0]:
  import re
  func_pattern = func_pattern[0].replace("*", "[^ ]*") + "$"
  def func_filter(*fn):
    return None is not re.match(func_pattern, paste(" ", *fn), flags=re.IGNORECASE)
else:
  def func_filter(*_):
    return True

# ---------------------------------------------------------------------------
# algo, coll helpers (unchanged)
algos_of_coll = {
  "AllGather":     ["RING","COLLNET_DIRECT","NVLS","PAT"],
  "AllReduce":     ["TREE","RING","COLLNET_DIRECT","COLLNET_CHAIN","NVLS","NVLS_TREE"],
  "Broadcast":     ["RING"],
  "Reduce":        ["RING"],
  "ReduceScatter": ["RING","COLLNET_DIRECT","NVLS","PAT"],
  "SendRecv":      [None]
}
coll_camel_to_lower = {
  "AllGather": "all_gather",
  "AllReduce": "all_reduce",
  "Broadcast": "broadcast",
  "Reduce": "reduce",
  "ReduceScatter": "reduce_scatter",
  "SendRecv": "sendrecv"
}
coll_lower_to_camel = {v: k for k, v in coll_camel_to_lower.items()}

# ---------------------------------------------------------------------------
# utility: pick file extension once
EXT = ".cc"            # <— generate .cc (not .cu)

# ---------------------------------------------------------------------------
# capability checks, function equivalences (unchanged)
def required_cuda(coll, redop, ty, algo, proto):
  cudart, arch = 0, 0
  if coll in ("SendRecv", "Generic", "Nop"): return (cudart, arch)
  if proto!="SIMPLE" and algo not in ("RING","TREE"): return None
  if coll in ("AllReduce","Reduce","ReduceScatter"):
    if redop=="SumPostDiv" and ty[0] not in ("i","u"): return None
    if ty=="bf16": cudart = max(cudart, 11000)
    if ty.startswith("f8"):
      cudart, arch = max(cudart, 11080), max(arch, 900)
  if "NVLS" in algo:
    if coll in ("AllReduce","Reduce","ReduceScatter"):
      ok = ((ty in ("i32","u32","i64","u64") and redop in ("Sum","MinMax"))
            or (ty in ("f32","f64") and redop=="Sum")
            or (ty in ("f16","bf16") and redop in ("Sum","MinMax")))
      if not ok: return None
    cudart, arch = max(cudart, 12010), max(arch, 900)
  return (cudart, arch)

def equivalent_primary(coll, redop, ty, algo, proto):
  if coll in ("AllReduce", "Reduce", "ReduceScatter"):
    if redop in ("Sum","Prod","PreMulSum","SumPostDiv") and ty[0]=="i":
      return (coll, redop, "u"+ty[1:], algo, proto)
    if redop=="MinMax" and ty[0]=="i" and "NVLS" not in algo:
      return (coll, redop, "u"+ty[1:], algo, proto)
  return (coll, redop, ty, algo, proto)

def best_kernel(coll, redop, ty, algo, proto):
  def best(c,r,t,a,p):
    if c=="Nop":      return ("Generic", None,None,None,None)
    if c=="SendRecv": return ("SendRecv",None,None,None,None)
    if c in ("AllGather","Broadcast"): return (c,None,None,"RING","LL")
    return (c,"Sum",t,("TREE" if a=="TREE" else "RING"),"LL")
  kfn = equivalent_primary(*best(coll, redop, ty, algo, proto))
  if not func_filter(*kfn): return ("Generic",None,None,None,None)
  return kfn

def enumerate_func_rows():
  yield ("SendRecv", None,None,None,None)
  for coll in ("AllGather","Broadcast"):
    for algo in algos_of_coll[coll]:
      for proto in all_protos:
        yield (coll,None,None,algo,proto)
  for coll in ("AllReduce","Reduce","ReduceScatter"):
    for redop in all_redops:
      for ty in all_tys:
        for algo in algos_of_coll[coll]:
          for proto in all_protos:
            yield (coll,redop,ty,algo,proto)

def is_built(c,r,t,a,p):
  built = required_cuda(c,r,t,a,p)
  return built and func_filter(c,r,t,a,p)

def validate(c,r,t,a,p):
  valid = required_cuda(c,r,t,a,p)
  built  = valid and func_filter(c,r,t,a,p)
  if built:  return (c,r,t,a,p)
  if valid:  return ("Nop",None,None,None,None)
  return None

func_rows     = [validate(*fn) for fn in enumerate_func_rows()]
primary_funcs = sorted(set(equivalent_primary(*fn) for fn in func_rows if fn))
primary_to_index = {fn:i for i,fn in enumerate(primary_funcs)}
kernel_funcs  = sorted(set(best_kernel(*fn) for fn in primary_funcs))

# ---------------------------------------------------------------------------
# helper for implementation filenames
def impl_filename(coll, redop, ty, algo, proto):
  return f"{paste('_', coll_camel_to_lower[coll], redop and redop.lower(), ty)}{EXT}"

# partition functions/kernels by file name
def partition_by_name(fns):
  ans = {}
  for fn in fns:
    name = impl_filename(*fn)
    coll = fn[0]
    ans.setdefault(name, (coll, []))[1].append(fn)
  return ans

name_to_funcs   = partition_by_name(fn for fn in primary_funcs if fn[0]!="Nop")
name_to_kernels = partition_by_name(kfn for kfn in kernel_funcs if kfn[0]!="Generic")

# ---------------------------------------------------------------------------
# generate device_table.<EXT>
with open(os.path.join(gensrc, f"device_table{EXT}"), "w") as f:
  out = f.write
  out('#include "common.h"\n')
  out('#ifdef __cplusplus\nextern "C" {\n#endif\n\n')

  for fn in primary_funcs:
    sym = paste("_","ncclDevFunc",*fn)
    cudart, arch = required_cuda(*fn)
    if (cudart, arch)!=(0,0):
      out(f"#if CUDART_VERSION >= {cudart} && __CUDA_ARCH__ >= {arch}\n")
    out(f"__device__ void {sym}();\n")
    if (cudart, arch)!=(0,0):
      out("#endif\n")
  out("\n__device__ ncclDevFuncPtr_t const ncclDevFuncTable[] = {\n")
  for idx,fn in enumerate(primary_funcs):
    sym = paste("_","ncclDevFunc",*fn)
    cudart, arch = required_cuda(*fn)
    if (cudart, arch)!=(0,0):
      out(f"#if CUDART_VERSION >= {cudart} && __CUDA_ARCH__ >= {arch}\n")
    out(f"/*{idx:4d}*/ {sym},\n")
    if (cudart, arch)!=(0,0):
      out(f"#else\n/*{idx:4d}*/ nullptr,\n#endif\n")
  out("nullptr};\n\n")
  out("// Workaround for https://reviews.llvm.org/D55580\n"
      "__device__ void ncclWorkaroundClangD55580() {}\n")
  out('#ifdef __cplusplus\n}\n#endif\n')

# ---------------------------------------------------------------------------
# generate host_table.cc (needs C linkage as well)
with open(os.path.join(gensrc, "host_table.cc"), "w") as f:
  out = f.write
  out('#include "device.h"\n')
  out('#ifdef __cplusplus\nextern "C" {\n#endif\n\n')

  out(f"extern int const ncclDevFuncIdCount = {len(primary_funcs)};\n\n")

  out("extern int const ncclDevFuncRowToId[] = {\n")
  for idx,fn in enumerate(func_rows):
    fn_id = -1; comment=""
    if fn:
      fn_id  = primary_to_index[equivalent_primary(*fn)]
      comment= " // " + paste(" ",*fn)
    out(f"/*{idx:4d}*/ {fn_id},{comment}\n")
  out("-1};\n\n")

  # forward decl of kernels
  for kfn in kernel_funcs:
    cudart,_ = required_cuda(*kfn)
    sym = paste("_","ncclDevKernel",*kfn)
    if cudart: out(f"#if CUDART_VERSION >= {cudart}\n")
    out("// coverity[declaration]\n")
    out(f"__global__ void {sym}(ncclDevKernelArgs4K const);\n")
    if cudart: out("#endif\n")
  out("\n")

  out(f"extern int const ncclDevKernelCount = {len(kernel_funcs)};\n")
  out("extern void* const ncclDevKernelList[] = {\n")
  for idx,kfn in enumerate(kernel_funcs):
    sym = paste("_","ncclDevKernel",*kfn)
    cudart,_ = required_cuda(*kfn)
    if cudart: out(f"#if CUDART_VERSION >= {cudart}\n")
    out(f"/*{idx:4d}*/ (void*){sym},\n")
    if cudart: out(f"#else\n/*{idx:4d}*/ nullptr,\n#endif\n")
  out("nullptr};\n\n")

  out("extern void* const ncclDevKernelForFunc[] = {\n")
  for idx,fn in enumerate(primary_funcs):
    kfn = best_kernel(*fn)
    sym = paste("_","ncclDevKernel",*kfn)
    cudart,_ = required_cuda(*kfn)
    if cudart: out(f"#if CUDART_VERSION >= {cudart}\n")
    out(f"/*{idx:4d}*/ (void*){sym},\n")
    if cudart: out(f"#else\n/*{idx:4d}*/ nullptr,\n#endif\n")
  out("nullptr};\n\n")

  out("extern bool const ncclDevKernelForFuncIsSpecialized[] = {\n")
  for idx,fn in enumerate(primary_funcs):
    specialized = "1" if fn==best_kernel(*fn) else "0"
    out(f"/*{idx:4d}*/ {specialized},\n")
  out("0};\n\n")
  out('#ifdef __cplusplus\n}\n#endif\n')

# ---------------------------------------------------------------------------
# Makefile rule generator
redop_to_cxx = {
  None:"FuncCopy","Sum":"FuncSum","Prod":"FuncProd","MinMax":"FuncMinMax",
  "PreMulSum":"FuncPreMulSum","SumPostDiv":"FuncSumPostDiv"
}
ty_to_cxx = {
  None:"int8_t","i8":"int8_t","u8":"uint8_t","i32":"int32_t","u32":"uint32_t",
  "i64":"int64_t","u64":"uint64_t","f16":"half","f32":"float","f64":"double",
  "bf16":"__nv_bfloat16","f8e4m3":"__nv_fp8_e4m3","f8e5m2":"__nv_fp8_e5m2"
}

with open(os.path.join(gensrc, "rules.mk"), "w") as f:
  out=f.write
  impl_names = sorted(name_to_funcs.keys())
  names = impl_names + ["host_table.cc", f"device_table{EXT}"]
  out("LIB_OBJS_GEN = $(patsubst %,$(OBJDIR)/genobj/%.o,{})\n\n"
      .format(" ".join(names)))

  for name in impl_names:
    coll = name_to_funcs[name][0]
    out("$(OBJDIR)/genobj/{n}.o: $(OBJDIR)/gensrc "
        "$(OBJDIR)/genobj/{lc}{ext}.d\n"
        "\t$(call COMPILE,$@,$(OBJDIR)/gensrc/{n})\n\n"
        .format(n=name, lc=coll_camel_to_lower[coll], ext=EXT))

# add stub .cc for each coll if needed (dependency scrape)
for coll in {c for (c,_,_,_,_) in primary_funcs if c!="Nop"}:
  name = impl_filename(coll,None,None,None,None)
  name_to_funcs.setdefault(name,(coll,[]))

# ---------------------------------------------------------------------------
# generate each per‑coll/per‑op translation unit
for name,(coll,fns) in name_to_funcs.items():
  with open(os.path.join(gensrc, name), "w") as f:
    out = f.write
    out('#include "common.h"\n')
    out(f'#include "{coll_camel_to_lower[coll]}.h"\n')
    out('#ifdef __cplusplus\nextern "C" {\n#endif\n')

    _,klist = name_to_kernels.get(name,(None,[]))
    for kfn in klist:
      c,r,t,a,p = kfn
      sym  = paste("_",c,r,t,a,p)
      fid  = primary_to_index[kfn]
      cudart,arch = required_cuda(*kfn)
      macro = ("DEFINE_ncclDevKernel({sym}, ncclFunc{c}, {rcxx}, {tcxx}, "
               "NCCL_ALGO_{algo}, NCCL_PROTO_{proto}, {id})\n")
      if (cudart,arch)!=(0,0):
        block = ("#if CUDART_VERSION >= {cudart}\n"
                 "  #if __CUDA_ARCH__ < {arch}\n"
                 "    DEFINE_ncclDevKernel_nop({sym}, ncclFunc{c}, {rcxx}, {tcxx}, "
                 "NCCL_ALGO_{algo}, NCCL_PROTO_{proto}, {id})\n"
                 "  #else\n    "+macro+"  #endif\n"
                 "#endif\n")
      else:
        block = macro
      out(block.format(sym=sym,c=c,rcxx=redop_to_cxx[r],tcxx=ty_to_cxx[t],
                       algo=(a or "RING"), proto=(p or "SIMPLE"),
                       id=fid, cudart=cudart, arch=arch))

    for fn in fns:
      c,r,t,a,p = fn
      sym = paste("_",c,r,t,a,p)
      cudart,arch=required_cuda(*fn)
      if (cudart,arch)!=(0,0):
        out(f"#if CUDART_VERSION >= {cudart} && __CUDA_ARCH__ >= {arch}\n")
      out("DEFINE_ncclDevFunc({sym}, ncclFunc{c}, {rcxx}, {tcxx}, "
          "NCCL_ALGO_{algo}, NCCL_PROTO_{proto})\n"
          .format(sym=sym,c=c,rcxx=redop_to_cxx[r],tcxx=ty_to_cxx[t],
                  algo=(a or "RING"), proto=(p or "SIMPLE")))
      if (cudart,arch)!=(0,0):
        out("#endif\n")
    out('#ifdef __cplusplus\n}\n#endif\n')
