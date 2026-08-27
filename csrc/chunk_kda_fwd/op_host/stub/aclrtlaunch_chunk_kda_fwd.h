#pragma once

#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

#define DECLARE_CHUNK_KDA_FWD_LAUNCHER(kernel_name)                          \
    extern "C" uint32_t aclrtlaunch_##kernel_name(                           \
        uint32_t, aclrtStream, void *, void *, void *, void *, void *,       \
        void *, void *, void *, void *, void *, void *, void *, void *,       \
        void *, void *, void *, void *, void *, void *, void *, void *,       \
        void *, void *)

DECLARE_CHUNK_KDA_FWD_LAUNCHER(chunk_kda_fwd_fp16_fp32);
DECLARE_CHUNK_KDA_FWD_LAUNCHER(chunk_kda_fwd_fp16_bf16);
DECLARE_CHUNK_KDA_FWD_LAUNCHER(chunk_kda_fwd_bf16_fp32);
DECLARE_CHUNK_KDA_FWD_LAUNCHER(chunk_kda_fwd_bf16_bf16);

#undef DECLARE_CHUNK_KDA_FWD_LAUNCHER
