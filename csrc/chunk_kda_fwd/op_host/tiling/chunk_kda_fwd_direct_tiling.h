/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of CANN Open Software License Agreement Version 2.0.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "chunk_kda_fwd_tiling_data.h"

namespace sglang::npu_kernel {

struct ChunkKdaFwdTilingParams {
    int64_t batch;
    int64_t seq_num;
    int64_t q_heads;
    int64_t v_heads;
    int64_t seqlen;
    int64_t k_dim;
    int64_t v_dim;
    int64_t chunk_size;
    int64_t total_chunks;
    int64_t input_rank;
    float scale;
    float lower_bound;
    bool has_initial_state;
    bool is_varlen;
    bool safe_gate;
    bool input_sequence_major;
    bool use_gate_in_kernel;
    bool has_a_log;
    bool has_dt_bias;
    bool q_is_bf16;
    int64_t gate_data_type;
    bool store_final_state;
    bool store_gk;
    bool store_w;
    bool store_u;
    bool store_qg;
    bool store_kg;
    bool store_v_new;
    bool store_h;
};

void make_chunk_kda_fwd_tiling(const ChunkKdaFwdTilingParams &params,
                               ChunkKdaFwdTilingData &tiling,
                               size_t &workspace_size,
                               uint32_t &block_dim);

}  // namespace sglang::npu_kernel
