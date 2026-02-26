#
# Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

SHELL := /usr/bin/env bash
MAKEFLAGS += -r
.SUFFIXES:
.SECONDARY:

NCCLDIR := ../..
include $(NCCLDIR)/makefiles/common.mk
include $(NCCLDIR)/makefiles/version.mk

BUILDDIR ?= $(abspath ../../build)
OBJDIR := $(BUILDDIR)/obj/device

MANIFEST := $(OBJDIR)/manifest
DEVGLUE_OBJ  := $(OBJDIR)/device_glue.o

INCFLAGS  = -I. -I.. -I$(BUILDDIR)/include -I../include
# ncclx-specific: ctran stubs need access to ctran headers and plugin includes
INCFLAGS += -I../include/plugin -I../../ -I$(BASE_DIR)
INCFLAGS  += -I$(CONDA_INCLUDE_DIR)
INCFLAGS  += -I$(CUDA_INC) -I$(CUDA_INC)/cccl

# Same as nccl port: compile with g++ instead of nvcc
# CXXFLAGS  += $(INCFLAGS) -O3 -fsanitize=undefined  -std=c++17 -Wno-unknown-pragmas -fvisibility=hidden -Wno-reorder -Wno-uninitialized -Wno-comment -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-local-typedefs -Wno-int-in-bool-context
CXXFLAGS  += $(INCFLAGS) -O3 -std=c++17 -Wno-unknown-pragmas -fvisibility=hidden -Wno-reorder -Wno-uninitialized -Wno-comment -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-local-typedefs -Wno-int-in-bool-context
# ncclx-specific defines (Meta host-code requirements, not CPU-emu)
CXXFLAGS  += -DGLOG_USE_GLOG_EXPORT -DMOCK_SCUBA_DATA -DCUDART_VERSION=12010


NVCUFLAGS_SYM := -ccbin $(CXX) $(CXXSTD) --expt-extended-lambda -Xptxas -maxrregcount=128 -Xfatbin -compress-all
NVCUFLAGS_SYM += $(INCFLAGS) --compiler-options "-fPIC -fvisibility=hidden"

ifneq ($(ENABLE_TCPDM),1)
CXXFLAGS  += -DCTRAN_DISABLE_TCPDM
endif

SAY = @bash -c 'path="$$2"; [[ "$$(realpath "$$2")" =~ ^$(subst .,\.,$(abspath $(NCCLDIR)))/(.*)$$ ]] && path="$${BASH_REMATCH[1]}"; printf "%-15s %s\n" "$$1" "$$path"' SAY

COMPILE.cu = $(NVCC) $(NVCUFLAGS) -dc $2 -o $1
COMPILE.cc = $(CXX) $(CXXFLAGS) -c $2 -o $1
define COMPILE
@$(SAY) "Compiling" $2;\
 mkdir -p $(dir $1);\
 $(call COMPILE$(or $3,$(suffix $2)),$1,$2)
endef

ifeq ($(shell echo "$$((1000*$(CUDA_MAJOR) + 10*$(CUDA_MINOR) >= 12090))"),1)
	NVCC_GENCODE_LDMC_FP8 = -gencode=arch=compute_100f,code=sm_100f
else ifeq ($(shell echo "$$((1000*$(CUDA_MAJOR) + 10*$(CUDA_MINOR) >= 12070))"),1)
  NVCC_GENCODE_LDMC_FP8 = -gencode=arch=compute_100a,code=sm_100a
else
	NVCC_GENCODE_LDMC_FP8 =
endif

define COMPILE_SYM
@$(SAY) "Compiling" $2;\
 mkdir -p $(dir $1);\
 $(call COMPILE$(or $3,$(suffix $2)),$1,$2)
endef

# @$(SAY) "Compiling" $2;\
#  mkdir -p $(dir $1);\
#  $(NVCC) $(NVCUFLAGS_SYM) $3 -dw $2 -o $1
# endef

DEPENDS.cu = $(NVCC) $(NVCUFLAGS) -M -dc $1
DEPENDS.cc = $(CXX) $(CXXFLAGS) -M -c $1
define DEPENDS
@$(SAY) "Dependencies" $2;\
 mkdir -p $(dir $1);\
 mk=$$($(call DEPENDS$(suffix $2),$2));\
 [[ $$mk =~ ^[^:]*:(.*)$$ ]];\
 files=$${BASH_REMATCH[1]};\
 files=$$(for x in $$files; do case "$$x" in '\'|$$'\t') ;; *) echo "$$x"; esac; done);\
 files=$$(for x in $$files; do [[ "$$(realpath "$$x")" == "$$(realpath "$(NCCLDIR)")"* ]] && echo "$$x"; done);\
 echo "$(patsubst %.d,%.o,$1) $1: " $$files > $1
endef

all: $(MANIFEST)

$(OBJDIR)/gensrc: generate.py
	@mkdir -p $@
	(which python3 >/dev/null || \
	  (bar='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'; \
	   printf "\n$${bar}\nERROR: Building NCCL requires a Python 3 installation invokable as 'python3'.\n$${bar}\n\n" 1>&2; \
	   exit 1)) \
	&& ./generate.py $@ "$(ONLY_FUNCS)"

$(OBJDIR)/gensrc/symmetric: $(OBJDIR)/gensrc symmetric/generate.py
	@mkdir -p $@
	./symmetric/generate.py $@

# The trailing ";" is necessary to make this an "empty recipe":
# https://www.gnu.org/software/make/manual/html_node/Empty-Recipes.html
$(OBJDIR)/gensrc/rules.mk: $(OBJDIR)/gensrc ;

$(OBJDIR)/gensrc/symmetric/rules.mk: $(OBJDIR)/gensrc/symmetric ;

-include $(OBJDIR)/gensrc/rules.mk
# "gensrc/rules.mk" populates $(LIB_OBJS_GEN)

-include $(OBJDIR)/gensrc/symmetric/rules.mk
# "gensrc/symmetric/rules.mk" populates $(LIB_OBJS_SYM_GEN)

# ncclx-specific: ctran_cpu_stubs.cc provides linker symbols for ctran kernels
# that are never called in CPU-emu mode. Ctran .cu porting is future work.
# all_reduce_sparse_block.cc is the ncclx sparse allreduce (Meta-specific).
SRCS = common.cc onerank.cc ctran_cpu_stubs.cc all_reduce_sparse_block.cc

LIB_OBJS = $(patsubst %, $(OBJDIR)/%.o, $(SRCS)) $(LIB_OBJS_GEN) $(LIB_OBJS_SYM_GEN)

$(OBJDIR)/%.o: % $(OBJDIR)/%.d
	$(call COMPILE,$@,$<)

$(OBJDIR)/genobj/%.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/%.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/$*)

$(OBJDIR)/genobj/symmetric/%.o: $(OBJDIR)/gensrc/symmetric $(OBJDIR)/genobj/symmetric/%.d
	$(call COMPILE,$@,$(OBJDIR)/gensrc/symmetric/$*)

$(OBJDIR)/%.d: %
	$(call DEPENDS,$@,$<)

$(OBJDIR)/genobj/%.d: $(OBJDIR)/gensrc/%
	$(call DEPENDS,$@,$<)

$(OBJDIR)/genobj/symmetric/%.d: $(OBJDIR)/gensrc/symmetric/%
	$(call DEPENDS,$@,$<)

$(DEVGLUE_OBJ): $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) -r $^ -o $@
#	$(NVCC) $(NVCUFLAGS) -dlink $^ -o $@

# $(MANIFEST): $(LIB_OBJS) $(DEVGLUE_OBJ)
$(MANIFEST): $(LIB_OBJS)
	@echo $^ > $@

-include $(wildcard $(OBJDIR)/*.d)
-include $(wildcard $(OBJDIR)/genobj/*.d)
-include $(wildcard $(OBJDIR)/genobj/symmetric/*.d)

.PHONY: clean
clean:
	rm -rf $(OBJDIR)
