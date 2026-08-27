/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of CANN Open Software License Agreement Version 2.0.
 */

#include "chunk_kda_fwd_direct_tiling.h"

#include <algorithm>
#include <cstdint>

#include "tiling/platform/platform_ascendc.h"

namespace sglang::npu_kernel {
namespace {

constexpr uint64_t KDA_ALIGN = 512;
constexpr uint64_t KDA_SOLVE_SCRATCH_SLOTS = 5;
constexpr uint64_t KDA_SOLVE_PIPELINE_DEPTH = 4;
constexpr uint64_t KDA_SCORE_QUEUE_SLOTS = 4;
constexpr uint64_t KDA_SCORE_SCRATCH_PLANES = 3;
constexpr uint64_t KDA_GDN_PIPELINE_DEPTH = 2;

struct Arch35Options {
    bool compute_gate_in_prepare = false;
    bool fuse_post_wu = false;
    bool fuse_post_wu_into_fwd_h = false;
    bool use_dense_fwd_h = false;
};

Arch35Options configure_arch35(const ChunkKdaFwdTilingParams &p,
                               bool is_ascend950)
{
    Arch35Options options;
    if (!is_ascend950 || p.chunk_size != 64 || p.k_dim != 128 ||
        p.v_dim != 128) {
        return options;
    }
    options.compute_gate_in_prepare =
        p.q_is_bf16 && p.gate_data_type == 2 && p.has_a_log &&
        p.use_gate_in_kernel && p.safe_gate;
    const bool dense_aligned =
        !p.is_varlen && p.seqlen % p.chunk_size == 0;
    options.use_dense_fwd_h = dense_aligned && p.q_is_bf16;
    const bool can_fuse = dense_aligned && p.q_is_bf16 && p.safe_gate &&
                          p.v_heads % 2 == 0;
    options.fuse_post_wu_into_fwd_h =
        options.use_dense_fwd_h && can_fuse &&
        options.compute_gate_in_prepare && !p.store_qg && !p.store_v_new &&
        !p.store_h;
    options.fuse_post_wu = can_fuse && !options.fuse_post_wu_into_fwd_h;
    return options;
}

uint64_t align_workspace(uint64_t bytes)
{
    return (bytes + KDA_ALIGN - 1) / KDA_ALIGN * KDA_ALIGN;
}

uint64_t allocate_workspace(uint64_t &cursor, uint64_t bytes)
{
    const uint64_t offset = align_workspace(cursor);
    cursor = offset + bytes;
    return offset;
}

}  // namespace

void make_chunk_kda_fwd_tiling(const ChunkKdaFwdTilingParams &p,
                               ChunkKdaFwdTilingData &t,
                               size_t &workspace_size,
                               uint32_t &block_dim)
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const uint32_t physical_core_num =
        std::max<uint32_t>(platform->GetCoreNumAic(), 1);
#ifdef SGL_KERNEL_ENABLE_A5_ONLY_OPS
    constexpr bool is_ascend950 = true;
#else
    constexpr bool is_ascend950 = false;
#endif
    const uint64_t fwd_h_tasks = static_cast<uint64_t>(
                                     p.is_varlen ? p.seq_num : p.batch) *
                                 p.v_heads;
    block_dim = is_ascend950
                    ? static_cast<uint32_t>(std::min<uint64_t>(
                          physical_core_num,
                          std::max<uint64_t>(fwd_h_tasks, 1)))
                    : physical_core_num;

    const auto arch35_options = configure_arch35(p, is_ascend950);

    constexpr uint64_t data_bytes = sizeof(uint16_t);
    const uint64_t token_heads = static_cast<uint64_t>(p.batch) * p.v_heads *
                                 p.seqlen;
    const uint64_t k_tensor_bytes = token_heads * p.k_dim * data_bytes;
    const uint64_t v_tensor_bytes = token_heads * p.v_dim * data_bytes;
    const uint64_t gk_bytes = token_heads * p.k_dim * sizeof(float);
    const uint64_t state_elements = static_cast<uint64_t>(p.seq_num) *
                                    p.v_heads * p.k_dim * p.v_dim;
    const uint64_t h_chunk_count =
        p.is_varlen ? static_cast<uint64_t>(p.total_chunks)
                    : static_cast<uint64_t>(p.batch) * p.total_chunks;
    const uint64_t h_bytes =
        h_chunk_count * p.v_heads * p.k_dim * p.v_dim * data_bytes;

    uint64_t cursor = 0;
    const uint64_t gk_storage_offset =
        p.store_gk ? 0 : allocate_workspace(cursor, gk_bytes);
    const uint64_t final_state_storage_offset =
        p.store_final_state
            ? 0
            : allocate_workspace(cursor, state_elements * sizeof(float));
    const uint64_t w_storage_offset =
        p.store_w ? 0 : allocate_workspace(cursor, k_tensor_bytes);
    const uint64_t u_storage_offset =
        p.store_u ? 0 : allocate_workspace(cursor, v_tensor_bytes);
    const uint64_t qg_storage_offset =
        p.store_qg ? 0 : allocate_workspace(cursor, k_tensor_bytes);
    const uint64_t kg_storage_offset =
        p.store_kg ? 0 : allocate_workspace(cursor, k_tensor_bytes);
    const uint64_t v_new_storage_bytes =
        arch35_options.use_dense_fwd_h && !p.store_v_new
            ? static_cast<uint64_t>(p.batch) * p.v_heads * p.chunk_size *
                  p.v_dim * data_bytes
            : v_tensor_bytes;
    const uint64_t v_new_storage_offset =
        p.store_v_new ? 0 : allocate_workspace(cursor, v_new_storage_bytes);
    const uint64_t h_storage_bytes =
        arch35_options.use_dense_fwd_h && !p.store_h
            ? static_cast<uint64_t>(p.batch) * p.v_heads * p.k_dim * p.v_dim *
                  data_bytes
            : h_bytes;
    const uint64_t h_storage_offset =
        p.store_h ? 0 : allocate_workspace(cursor, h_storage_bytes);
    const uint64_t qg_scaled_offset =
        allocate_workspace(cursor, k_tensor_bytes);

    const uint64_t matrix_bytes =
        token_heads * p.chunk_size * sizeof(float);
    const uint64_t prepare_aqk_fp32_offset =
        allocate_workspace(cursor, matrix_bytes);
    const uint64_t prepare_akk_fp32_offset =
        allocate_workspace(cursor, matrix_bytes);
    const uint64_t prepare_scratch_offset = align_workspace(cursor);
    const uint64_t solve_depth = p.safe_gate ? KDA_SOLVE_PIPELINE_DEPTH : 1;
    const uint64_t solve_bytes = static_cast<uint64_t>(block_dim) * solve_depth *
                                 KDA_SOLVE_SCRATCH_SLOTS * p.chunk_size *
                                 p.chunk_size * sizeof(float);
    const uint64_t score_bytes =
        static_cast<uint64_t>(block_dim) * KDA_SCORE_QUEUE_SLOTS *
        KDA_SCORE_SCRATCH_PLANES * p.chunk_size * p.k_dim * data_bytes;
    cursor = prepare_scratch_offset + align_workspace(solve_bytes) + score_bytes;

    const uint64_t post_wu_scratch_offset = align_workspace(cursor);
    if (!arch35_options.fuse_post_wu &&
        !arch35_options.fuse_post_wu_into_fwd_h) {
        cursor = post_wu_scratch_offset + token_heads * p.k_dim * sizeof(float);
    }

    const uint64_t fwd_h_workspace_base_offset = align_workspace(cursor);
    uint64_t fwd_h_cursor = 0;
    const uint64_t v_workspace_offset = allocate_workspace(
        fwd_h_cursor, static_cast<uint64_t>(block_dim) * p.chunk_size * p.v_dim *
                          sizeof(float) * KDA_GDN_PIPELINE_DEPTH);
    const uint64_t v_update_workspace_offset = allocate_workspace(
        fwd_h_cursor, static_cast<uint64_t>(block_dim) * p.chunk_size * p.v_dim *
                          sizeof(float) * KDA_GDN_PIPELINE_DEPTH);
    const uint64_t k_decay_workspace_offset = allocate_workspace(
        fwd_h_cursor, static_cast<uint64_t>(block_dim) * p.chunk_size * p.k_dim *
                          sizeof(float) * KDA_GDN_PIPELINE_DEPTH);
    const uint64_t h_workspace_offset = allocate_workspace(
        fwd_h_cursor, static_cast<uint64_t>(block_dim) * p.k_dim * p.v_dim *
                          sizeof(float) * KDA_GDN_PIPELINE_DEPTH);
    const uint64_t token_batch = p.is_varlen ? p.seq_num : 1;
    const uint64_t num_seq_workspace_offset = allocate_workspace(
        fwd_h_cursor, (token_batch + 1) * sizeof(int64_t));
    const uint64_t num_chunks_workspace_offset = allocate_workspace(
        fwd_h_cursor, (token_batch + 1) * sizeof(int64_t));
    cursor = fwd_h_workspace_base_offset + align_workspace(fwd_h_cursor);

    const uint64_t output_scratch_offset = allocate_workspace(
        cursor, 2 * token_heads * p.v_dim * sizeof(float));

    workspace_size = static_cast<size_t>(platform->GetLibApiWorkSpaceSize() +
                                         align_workspace(cursor));

    t = {};
    t.batch = p.batch;
    t.seqNum = p.seq_num;
    t.qHeadNum = p.q_heads;
    t.vHeadNum = p.v_heads;
    t.seqlen = p.seqlen;
    t.kHeadDim = p.k_dim;
    t.vHeadDim = p.v_dim;
    t.chunkSize = p.chunk_size;
    t.totalChunks = p.total_chunks;
    t.inputRank = p.input_rank;
    t.scale = p.scale;
    t.lowerBound = p.lower_bound;
    t.hasInitialState = p.has_initial_state;
    t.isVarLen = p.is_varlen;
    t.safeGate = p.safe_gate;
    t.inputSequenceMajor = p.input_sequence_major;
    t.useGateInKernel = p.use_gate_in_kernel;
    t.hasALog = p.has_a_log;
    t.hasDtBias = p.has_dt_bias;
    t.computeGateInPrepare = arch35_options.compute_gate_in_prepare;
    t.fusePostWu = arch35_options.fuse_post_wu;
    t.fusePostWuIntoFwdH = arch35_options.fuse_post_wu_into_fwd_h;
    t.useDenseFwdH = arch35_options.use_dense_fwd_h;
    t.storeFinalState = p.store_final_state;
    t.storeGk = p.store_gk;
    t.storeW = p.store_w;
    t.storeU = p.store_u;
    t.storeQG = p.store_qg;
    t.storeKg = p.store_kg;
    t.storeVNew = p.store_v_new;
    t.storeH = p.store_h;
    t.gateDataType = p.gate_data_type;
    t.gateUsedCoreNum = static_cast<int64_t>(block_dim) * 2;
    t.prepareUsedCoreNum = block_dim;
    t.postWuUsedCoreNum = block_dim;
    t.outputUsedCoreNum = block_dim;
    t.gkStorageOffset = gk_storage_offset;
    t.finalStateStorageOffset = final_state_storage_offset;
    t.wStorageOffset = w_storage_offset;
    t.uStorageOffset = u_storage_offset;
    t.qgStorageOffset = qg_storage_offset;
    t.kgStorageOffset = kg_storage_offset;
    t.vNewStorageOffset = v_new_storage_offset;
    t.hStorageOffset = h_storage_offset;
    t.qgScaledOffset = qg_scaled_offset;
    t.prepareAqkFp32Offset = prepare_aqk_fp32_offset;
    t.prepareAkkFp32Offset = prepare_akk_fp32_offset;
    t.prepareScratchOffset = prepare_scratch_offset;
    t.postWuScratchOffset = post_wu_scratch_offset;
    t.outputScratchOffset = output_scratch_offset;
    t.fwdHWorkspaceBaseOffset = fwd_h_workspace_base_offset;
    t.vWorkspaceOffset = v_workspace_offset;
    t.vUpdateWorkspaceOffset = v_update_workspace_offset;
    t.kDecayWorkspaceOffset = k_decay_workspace_offset;
    t.hWorkspaceOffset = h_workspace_offset;
    t.numSeqWorkspaceOffset = num_seq_workspace_offset;
    t.numChunksWorkspaceOffset = num_chunks_workspace_offset;
    t.tilingKey = p.chunk_size == 64 && p.k_dim == 128 && p.v_dim == 128 ? 2 : 1;
}

}  // namespace sglang::npu_kernel
