/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of CANN Open Software License Agreement Version 2.0.
 */

#include "chunk_kda_fwd.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "chunk_kda_fwd_direct_tiling.h"
#include "torch_helper.h"
#include "stub/aclrtlaunch_chunk_kda_fwd.h"

namespace sglang::npu_kernel {
namespace {

constexpr int64_t MAX_KDA_DIM = 256;
constexpr int64_t MAX_KDA_HEADS = 128;
constexpr int64_t MAX_KDA_VARLEN_SEQUENCES = 1024;

enum class Layout { BSND, BNSD, TND, NTD };

struct ShapeInfo {
    bool rank3 = false;
    int64_t batch = 0;
    int64_t seqlen = 0;
    int64_t q_heads = 0;
    int64_t v_heads = 0;
    int64_t k_dim = 0;
    int64_t v_dim = 0;
    int64_t seq_num = 0;
    int64_t total_chunks = 0;
};

Layout parse_layout(c10::string_view layout)
{
    if (layout == "BSND") {
        return Layout::BSND;
    }
    if (layout == "BNSD") {
        return Layout::BNSD;
    }
    if (layout == "TND") {
        return Layout::TND;
    }
    if (layout == "NTD") {
        return Layout::NTD;
    }
    TORCH_CHECK(false,
                "chunk_kda_fwd: layout must be uppercase and one of BSND, "
                "BNSD, TND or NTD.");
    return Layout::BSND;
}

bool has_shape(const at::Tensor &tensor,
               std::initializer_list<int64_t> expected)
{
    if (tensor.dim() != static_cast<int64_t>(expected.size())) {
        return false;
    }
    int64_t dim = 0;
    for (const int64_t size : expected) {
        if (tensor.size(dim++) != size) {
            return false;
        }
    }
    return true;
}

std::vector<int64_t> tensor_to_host_ints(const at::Tensor &tensor)
{
    at::Tensor cpu = tensor.to(
        at::TensorOptions().device(at::kCPU).dtype(at::kLong), false, true);
    cpu = cpu.contiguous().view({-1});
    const int64_t *data = cpu.data_ptr<int64_t>();
    return std::vector<int64_t>(data, data + cpu.numel());
}

std::vector<int64_t> resolve_host_ints(
    c10::OptionalIntArrayRef host_values,
    const c10::optional<at::Tensor> &device_values, const char *name)
{
    if (host_values.has_value()) {
        return std::vector<int64_t>(host_values->begin(), host_values->end());
    }
    if (device_values.has_value() && device_values->defined()) {
        return tensor_to_host_ints(*device_values);
    }
    (void)name;
    return {};
}

at::Tensor host_ints_to_device(const std::vector<int64_t> &values,
                               const at::Tensor &like)
{
    if (values.empty()) {
        return at::empty({1}, like.options().dtype(at::kLong));
    }
    at::Tensor cpu = at::empty(
        {static_cast<int64_t>(values.size())},
        at::TensorOptions().device(at::kCPU).dtype(at::kLong));
    std::memcpy(cpu.data_ptr<int64_t>(), values.data(),
                values.size() * sizeof(int64_t));
    return TorchNpuHelper::CopyTensorHostToDevice(cpu);
}

at::Tensor resolve_device_ints(const c10::optional<at::Tensor> &provided,
                               const std::vector<int64_t> &host_values,
                               const at::Tensor &like)
{
    if (provided.has_value() && provided->defined() &&
        provided->device() == like.device() &&
        provided->scalar_type() == at::kLong &&
        provided->numel() == static_cast<int64_t>(host_values.size())) {
        return provided->contiguous();
    }
    return host_ints_to_device(host_values, like);
}

int64_t ceil_div(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

void validate_cu_seqlens(const std::vector<int64_t> &cu, int64_t seqlen)
{
    if (cu.empty()) {
        return;
    }
    TORCH_CHECK(cu.size() >= 2,
                "chunk_kda_fwd: cu_seqlens must contain at least [0, T].");
    TORCH_CHECK(cu.front() == 0,
                "chunk_kda_fwd: cu_seqlens[0] must be zero.");
    TORCH_CHECK(cu.back() == seqlen,
                "chunk_kda_fwd: the last cu_seqlens value must equal T.");
    TORCH_CHECK(cu.size() - 1 <= MAX_KDA_VARLEN_SEQUENCES,
                "chunk_kda_fwd: varlen supports at most 1024 sequences.");
    for (size_t i = 0; i + 1 < cu.size(); ++i) {
        TORCH_CHECK(cu[i] <= cu[i + 1],
                    "chunk_kda_fwd: cu_seqlens must be nondecreasing.");
    }
}

int64_t count_chunks(const std::vector<int64_t> &cu, int64_t seqlen,
                     int64_t chunk_size)
{
    if (cu.empty()) {
        return ceil_div(seqlen, chunk_size);
    }
    int64_t count = 0;
    for (size_t i = 0; i + 1 < cu.size(); ++i) {
        count += ceil_div(cu[i + 1] - cu[i], chunk_size);
    }
    return count;
}

std::vector<int64_t> canonical_chunk_indices(
    const std::vector<int64_t> &cu, int64_t chunk_size)
{
    std::vector<int64_t> indices;
    for (size_t seq = 0; seq + 1 < cu.size(); ++seq) {
        const int64_t chunks = ceil_div(cu[seq + 1] - cu[seq], chunk_size);
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            indices.push_back(static_cast<int64_t>(seq));
            indices.push_back(chunk);
        }
    }
    return indices;
}

void validate_chunk_indices(const std::vector<int64_t> &indices,
                            const std::vector<int64_t> &cu,
                            int64_t total_chunks, int64_t chunk_size)
{
    if (indices.empty()) {
        return;
    }
    TORCH_CHECK(!cu.empty(),
                "chunk_kda_fwd: chunk_indices requires cu_seqlens.");
    TORCH_CHECK(indices.size() == static_cast<size_t>(total_chunks * 2),
                "chunk_kda_fwd: chunk_indices must contain one pair per chunk.");
    TORCH_CHECK(indices == canonical_chunk_indices(cu, chunk_size),
                "chunk_kda_fwd: chunk_indices must use canonical "
                "sequence-major order.");
}

ShapeInfo resolve_shape(const at::Tensor &q, const at::Tensor &k,
                        const at::Tensor &v, const at::Tensor &g,
                        const at::Tensor &beta, Layout layout,
                        const std::vector<int64_t> &cu, int64_t chunk_size)
{
    ShapeInfo info;
    info.rank3 = layout == Layout::TND || layout == Layout::NTD;
    const int64_t tensor_rank = info.rank3 ? 3 : 4;
    const int64_t beta_rank = info.rank3 ? 2 : 3;
    TORCH_CHECK(q.dim() == tensor_rank && k.dim() == tensor_rank &&
                    v.dim() == tensor_rank && g.dim() == tensor_rank &&
                    beta.dim() == beta_rank,
                "chunk_kda_fwd: input ranks do not match layout.");
    TORCH_CHECK(q.sizes() == k.sizes(),
                "chunk_kda_fwd: q and k must have identical shape.");

    if (layout == Layout::TND) {
        info.batch = 1;
        info.seqlen = q.size(0);
        info.q_heads = q.size(1);
        info.k_dim = q.size(2);
        info.v_heads = v.size(1);
        info.v_dim = v.size(2);
        TORCH_CHECK(has_shape(v, {info.seqlen, info.v_heads, info.v_dim}) &&
                        has_shape(g, {info.seqlen, info.v_heads, info.k_dim}) &&
                        has_shape(beta, {info.seqlen, info.v_heads}),
                    "chunk_kda_fwd: invalid TND v/g/beta shapes.");
    } else if (layout == Layout::NTD) {
        info.batch = 1;
        info.q_heads = q.size(0);
        info.seqlen = q.size(1);
        info.k_dim = q.size(2);
        info.v_heads = v.size(0);
        info.v_dim = v.size(2);
        TORCH_CHECK(has_shape(v, {info.v_heads, info.seqlen, info.v_dim}) &&
                        has_shape(g, {info.v_heads, info.seqlen, info.k_dim}) &&
                        has_shape(beta, {info.v_heads, info.seqlen}),
                    "chunk_kda_fwd: invalid NTD v/g/beta shapes.");
    } else if (layout == Layout::BSND) {
        info.batch = q.size(0);
        info.seqlen = q.size(1);
        info.q_heads = q.size(2);
        info.k_dim = q.size(3);
        info.v_heads = v.size(2);
        info.v_dim = v.size(3);
        TORCH_CHECK(has_shape(v, {info.batch, info.seqlen, info.v_heads,
                                  info.v_dim}) &&
                        has_shape(g, {info.batch, info.seqlen, info.v_heads,
                                      info.k_dim}) &&
                        has_shape(beta, {info.batch, info.seqlen,
                                         info.v_heads}),
                    "chunk_kda_fwd: invalid BSND v/g/beta shapes.");
    } else {
        info.batch = q.size(0);
        info.q_heads = q.size(1);
        info.seqlen = q.size(2);
        info.k_dim = q.size(3);
        info.v_heads = v.size(1);
        info.v_dim = v.size(3);
        TORCH_CHECK(has_shape(v, {info.batch, info.v_heads, info.seqlen,
                                  info.v_dim}) &&
                        has_shape(g, {info.batch, info.v_heads, info.seqlen,
                                      info.k_dim}) &&
                        has_shape(beta, {info.batch, info.v_heads,
                                         info.seqlen}),
                    "chunk_kda_fwd: invalid BNSD v/g/beta shapes.");
    }
    info.seq_num = cu.empty() ? info.batch : static_cast<int64_t>(cu.size()) - 1;
    info.total_chunks = count_chunks(cu, info.seqlen, chunk_size);
    return info;
}

void validate_common(const at::Tensor &q, const at::Tensor &k,
                     const at::Tensor &v, const at::Tensor &g,
                     const at::Tensor &beta, const ShapeInfo &info,
                     const std::vector<int64_t> &cu, int64_t chunk_size,
                     bool safe_gate, double lower_bound,
                     bool use_gate_in_kernel,
                     const c10::optional<at::Tensor> &a_log,
                     const c10::optional<at::Tensor> &dt_bias,
                     const c10::optional<at::Tensor> &initial_state,
                     bool state_v_first)
{
    TORCH_CHECK(info.batch > 0 && info.seqlen > 0,
                "chunk_kda_fwd: batch and sequence dimensions must be "
                "positive.");
    TORCH_CHECK(chunk_size == 64 || chunk_size == 128,
                "chunk_kda_fwd: chunk_size must be 64 or 128.");
    TORCH_CHECK(info.q_heads > 0 && info.v_heads >= info.q_heads &&
                    info.v_heads % info.q_heads == 0,
                "chunk_kda_fwd: HV must be >= H and divisible by H.");
    TORCH_CHECK(info.q_heads <= MAX_KDA_HEADS &&
                    info.v_heads <= MAX_KDA_HEADS,
                "chunk_kda_fwd: H and HV must be <= 128.");
    TORCH_CHECK(info.k_dim >= 16 && info.k_dim <= MAX_KDA_DIM &&
                    info.k_dim % 16 == 0 && info.v_dim >= 16 &&
                    info.v_dim <= MAX_KDA_DIM && info.v_dim % 16 == 0,
                "chunk_kda_fwd: K/V must be multiples of 16 and <= 256.");
    TORCH_CHECK(q.scalar_type() == at::kHalf ||
                    q.scalar_type() == at::kBFloat16,
                "chunk_kda_fwd: q must be float16 or bfloat16.");
    TORCH_CHECK(k.scalar_type() == q.scalar_type() &&
                    v.scalar_type() == q.scalar_type(),
                "chunk_kda_fwd: q/k/v dtypes must match.");
    TORCH_CHECK(g.scalar_type() == at::kFloat ||
                    g.scalar_type() == at::kBFloat16,
                "chunk_kda_fwd: g must be float32 or bfloat16.");
    TORCH_CHECK(beta.scalar_type() == at::kFloat ||
                    beta.scalar_type() == at::kBFloat16,
                "chunk_kda_fwd: beta must be float32 or bfloat16.");
    TORCH_CHECK(cu.empty() || info.rank3 || info.batch == 1,
                "chunk_kda_fwd: rank4 varlen requires B=1.");

    const at::Device device = q.device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1,
                "chunk_kda_fwd: inputs must be NPU tensors.");
    for (const at::Tensor *tensor : {&k, &v, &g, &beta}) {
        TORCH_CHECK(tensor->device() == device,
                    "chunk_kda_fwd: all inputs must be on the same device.");
    }
    if (a_log.has_value() && a_log->defined()) {
        TORCH_CHECK(a_log->device() == device &&
                        (a_log->scalar_type() == at::kFloat ||
                         a_log->scalar_type() == at::kBFloat16),
                    "chunk_kda_fwd: A_log must be float32 or bfloat16 on "
                    "the input device.");
    }
    if (dt_bias.has_value() && dt_bias->defined()) {
        TORCH_CHECK(dt_bias->device() == device &&
                        (dt_bias->scalar_type() == at::kFloat ||
                         dt_bias->scalar_type() == at::kBFloat16),
                    "chunk_kda_fwd: dt_bias must be float32 or bfloat16 on "
                    "the input device.");
    }
    if (use_gate_in_kernel) {
        TORCH_CHECK(a_log.has_value() && a_log->defined() &&
                        a_log->numel() == info.v_heads,
                    "chunk_kda_fwd: A_log [HV] is required when "
                    "use_gate_in_kernel=True.");
        TORCH_CHECK(!dt_bias.has_value() || !dt_bias->defined() ||
                        dt_bias->numel() == info.v_heads * info.k_dim,
                    "chunk_kda_fwd: dt_bias must have HV*K elements.");
        TORCH_CHECK(!safe_gate || (lower_bound >= -5.0 && lower_bound < 0.0),
                    "chunk_kda_fwd: lower_bound must be in [-5, 0) for "
                    "safe gate.");
    }
    if (initial_state.has_value() && initial_state->defined()) {
        TORCH_CHECK(initial_state->device() == device &&
                        initial_state->scalar_type() == at::kFloat &&
                        initial_state->dim() == 4,
                    "chunk_kda_fwd: initial_state must be a float32 rank4 "
                    "tensor on the input device.");
        const std::vector<int64_t> expected = state_v_first
                                                  ? std::vector<int64_t>{info.seq_num, info.v_heads, info.v_dim, info.k_dim}
                                                  : std::vector<int64_t>{info.seq_num, info.v_heads, info.k_dim, info.v_dim};
        TORCH_CHECK(initial_state->sizes() == at::IntArrayRef(expected),
                    "chunk_kda_fwd: initial_state shape does not match "
                    "state_v_first.");
    }
}

at::Tensor make_tiling_tensor(const ChunkKdaFwdTilingData &tiling)
{
    constexpr int64_t alignment = 32;
    const int64_t bytes =
        (static_cast<int64_t>(sizeof(tiling)) + alignment - 1) / alignment *
        alignment;
    at::Tensor cpu = at::zeros(
        {bytes}, at::TensorOptions().device(at::kCPU).dtype(at::kByte));
    std::memcpy(cpu.data_ptr(), &tiling, sizeof(tiling));
    return TorchNpuHelper::CopyTensorHostToDevice(cpu);
}

c10::optional<at::Tensor> maybe_tensor(bool present, const at::Tensor &tensor)
{
    return present ? c10::optional<at::Tensor>(tensor) : c10::nullopt;
}

}  // namespace

ChunkKdaFwdResult chunk_kda_fwd(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &g, const at::Tensor &beta, double scale,
    const c10::optional<at::Tensor> &initial_state, bool output_final_state,
    const c10::optional<at::Tensor> &cu_seqlens,
    c10::OptionalIntArrayRef cu_seqlens_cpu,
    const c10::optional<at::Tensor> &chunk_indices,
    c10::OptionalIntArrayRef chunk_indices_cpu, int64_t chunk_size,
    c10::string_view layout_string, bool safe_gate, double lower_bound,
    bool use_gate_in_kernel, bool state_v_first,
    const c10::optional<at::Tensor> &a_log,
    const c10::optional<at::Tensor> &dt_bias,
    bool return_intermediate_states, bool disable_recompute)
{
    const Layout layout = parse_layout(layout_string);
    TORCH_CHECK(chunk_size == 64 || chunk_size == 128,
                "chunk_kda_fwd: chunk_size must be 64 or 128.");
    std::vector<int64_t> cu =
        resolve_host_ints(cu_seqlens_cpu, cu_seqlens, "cu_seqlens");

    // Resolve shape first using the public layout; cu validation needs T.
    ShapeInfo info = resolve_shape(q, k, v, g, beta, layout, cu, chunk_size);
    validate_cu_seqlens(cu, info.seqlen);
    info.seq_num = cu.empty() ? info.batch : static_cast<int64_t>(cu.size()) - 1;
    info.total_chunks = count_chunks(cu, info.seqlen, chunk_size);

    std::vector<int64_t> chunk_host = resolve_host_ints(
        chunk_indices_cpu, chunk_indices, "chunk_indices");
    if (chunk_host.empty() && !cu.empty()) {
        chunk_host = canonical_chunk_indices(cu, chunk_size);
    }
    validate_chunk_indices(chunk_host, cu, info.total_chunks, chunk_size);
    validate_common(q, k, v, g, beta, info, cu, chunk_size, safe_gate,
                    lower_bound, use_gate_in_kernel, a_log, dt_bias,
                    initial_state, state_v_first);

    at::Tensor q_head = q.contiguous();
    at::Tensor k_head = k.contiguous();
    at::Tensor v_head = v.contiguous();
    at::Tensor g_head = g.contiguous();
    at::Tensor beta_head = beta.contiguous();
    bool input_sequence_major = layout == Layout::BSND;
    if (layout == Layout::BSND) {
        beta_head = beta_head.permute({0, 2, 1}).contiguous();
    } else if (layout == Layout::TND) {
        q_head = q_head.permute({1, 0, 2}).contiguous();
        k_head = k_head.permute({1, 0, 2}).contiguous();
        v_head = v_head.permute({1, 0, 2}).contiguous();
        g_head = g_head.permute({1, 0, 2}).contiguous();
        beta_head = beta_head.permute({1, 0}).contiguous();
    }
    if (info.rank3) {
        q_head = q_head.unsqueeze(0);
        k_head = k_head.unsqueeze(0);
        v_head = v_head.unsqueeze(0);
        g_head = g_head.unsqueeze(0);
        beta_head = beta_head.unsqueeze(0);
    }

    at::Tensor initial_compute = at::empty({1}, q.options().dtype(at::kFloat));
    const bool has_initial_state =
        initial_state.has_value() && initial_state->defined();
    if (has_initial_state) {
        initial_compute = initial_state->contiguous();
        if (state_v_first) {
            initial_compute = initial_compute.transpose(-1, -2).contiguous();
        }
    }
    at::Tensor a_log_compute =
        a_log.has_value() && a_log->defined()
            ? a_log->to(at::kFloat).contiguous()
            : at::empty({1}, q.options().dtype(at::kFloat));
    at::Tensor dt_bias_compute =
        dt_bias.has_value() && dt_bias->defined()
            ? dt_bias->to(at::kFloat).contiguous()
            : at::empty({1}, q.options().dtype(at::kFloat));
    at::Tensor cu_device = resolve_device_ints(cu_seqlens, cu, q);
    at::Tensor chunk_device =
        resolve_device_ints(chunk_indices, chunk_host, q);

    const bool store_gk = !use_gate_in_kernel || disable_recompute;
    const bool store_training = disable_recompute;
    const bool store_h = disable_recompute || return_intermediate_states;
    auto data_options = q.options();
    auto float_options = q.options().dtype(at::kFloat);

    at::Tensor out_internal = at::empty(
        {info.batch, info.seqlen, info.v_heads, info.v_dim}, data_options);
    at::Tensor final_internal = output_final_state
                                    ? at::empty({info.seq_num, info.v_heads,
                                                 info.k_dim, info.v_dim},
                                                float_options)
                                    : at::empty({1}, float_options);
    at::Tensor gk_internal = store_gk
                                 ? at::empty({info.batch, info.v_heads,
                                              info.seqlen, info.k_dim},
                                             float_options)
                                 : at::empty({1}, float_options);
    at::Tensor aqk = at::empty({info.batch, info.v_heads, info.seqlen,
                                chunk_size}, data_options);
    at::Tensor akk = at::empty_like(aqk);
    at::Tensor w = store_training
                       ? at::empty({info.batch, info.v_heads, info.seqlen,
                                    info.k_dim}, data_options)
                       : at::empty({1}, data_options);
    at::Tensor u = store_training
                       ? at::empty({info.batch, info.v_heads, info.seqlen,
                                    info.v_dim}, data_options)
                       : at::empty({1}, data_options);
    at::Tensor qg = store_training ? at::empty_like(w)
                                   : at::empty({1}, data_options);
    at::Tensor kg = store_training ? at::empty_like(w)
                                   : at::empty({1}, data_options);
    at::Tensor v_new = store_training ? at::empty_like(u)
                                      : at::empty({1}, data_options);
    at::Tensor h_internal =
        store_h
            ? at::empty({info.batch, info.v_heads, info.total_chunks,
                         info.k_dim, info.v_dim}, data_options)
            : at::empty({1}, data_options);

    ChunkKdaFwdTilingParams tiling_params{
        info.batch,
        info.seq_num,
        info.q_heads,
        info.v_heads,
        info.seqlen,
        info.k_dim,
        info.v_dim,
        chunk_size,
        info.total_chunks,
        4,
        static_cast<float>(scale),
        static_cast<float>(lower_bound),
        has_initial_state,
        !cu.empty(),
        safe_gate,
        input_sequence_major,
        use_gate_in_kernel,
        a_log.has_value() && a_log->defined(),
        dt_bias.has_value() && dt_bias->defined(),
        q.scalar_type() == at::kBFloat16,
        g.scalar_type() == at::kFloat ? 2 : 1,
        output_final_state,
        store_gk,
        store_training,
        store_training,
        store_training,
        store_training,
        store_training,
        store_h,
    };
    ChunkKdaFwdTilingData tiling{};
    size_t workspace_size = 0;
    uint32_t block_dim = 0;
    make_chunk_kda_fwd_tiling(tiling_params, tiling, workspace_size,
                              block_dim);
    at::Tensor tiling_device = make_tiling_tensor(tiling);
    at::Tensor workspace =
        at::empty({static_cast<int64_t>(workspace_size)},
                  q.options().dtype(at::kByte));

#define CHUNK_KDA_ARGS                                                       \
    q_head, k_head, v_head, g_head, beta_head, a_log_compute,               \
        dt_bias_compute, initial_compute, cu_device, chunk_device,           \
        out_internal, final_internal, gk_internal, aqk, akk, w, u, qg, kg,  \
        v_new, h_internal, workspace, tiling_device

    if (q.scalar_type() == at::kHalf && beta.scalar_type() == at::kFloat) {
        EXEC_KERNEL_CMD(chunk_kda_fwd_fp16_fp32, block_dim, CHUNK_KDA_ARGS);
    } else if (q.scalar_type() == at::kHalf) {
        EXEC_KERNEL_CMD(chunk_kda_fwd_fp16_bf16, block_dim, CHUNK_KDA_ARGS);
    } else if (beta.scalar_type() == at::kFloat) {
        EXEC_KERNEL_CMD(chunk_kda_fwd_bf16_fp32, block_dim, CHUNK_KDA_ARGS);
    } else {
        EXEC_KERNEL_CMD(chunk_kda_fwd_bf16_bf16, block_dim, CHUNK_KDA_ARGS);
    }
#undef CHUNK_KDA_ARGS

    at::Tensor out = info.rank3 ? out_internal.squeeze(0) : out_internal;
    at::Tensor aqk_export = info.rank3 ? aqk.squeeze(0) : aqk;
    at::Tensor akk_export = info.rank3 ? akk.squeeze(0) : akk;
    at::Tensor final_export;
    if (output_final_state) {
        final_export = state_v_first
                           ? final_internal.transpose(-1, -2).contiguous()
                           : final_internal;
    }
    at::Tensor h_export;
    if (store_h) {
        h_export = h_internal.permute({0, 2, 1, 3, 4});
        if (state_v_first) {
            h_export = h_export.transpose(-1, -2);
        }
        h_export = h_export.contiguous();
        if (info.rank3) {
            h_export = h_export.squeeze(0);
        }
    }
    auto squeeze_rank3 = [&](const at::Tensor &tensor) {
        return info.rank3 ? tensor.squeeze(0) : tensor;
    };

    return std::make_tuple(
        out, maybe_tensor(output_final_state, final_export),
        maybe_tensor(store_gk, squeeze_rank3(gk_internal)), aqk_export,
        akk_export, maybe_tensor(store_training, squeeze_rank3(w)),
        maybe_tensor(store_training, squeeze_rank3(u)),
        maybe_tensor(store_training, squeeze_rank3(qg)),
        maybe_tensor(store_training, squeeze_rank3(kg)),
        maybe_tensor(store_training, squeeze_rank3(v_new)),
        maybe_tensor(store_h, h_export), initial_state);
}

}  // namespace sglang::npu_kernel
