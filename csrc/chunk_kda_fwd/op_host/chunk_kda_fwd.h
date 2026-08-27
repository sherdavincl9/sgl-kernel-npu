/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of CANN Open Software License Agreement Version 2.0.
 */

#pragma once

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <string_view>
#include <tuple>

#include "defines.h"

namespace sglang::npu_kernel {

using ChunkKdaFwdResult = std::tuple<
    at::Tensor, c10::optional<at::Tensor>, c10::optional<at::Tensor>,
    at::Tensor, at::Tensor, c10::optional<at::Tensor>,
    c10::optional<at::Tensor>, c10::optional<at::Tensor>,
    c10::optional<at::Tensor>, c10::optional<at::Tensor>,
    c10::optional<at::Tensor>, c10::optional<at::Tensor>>;

HOST_API ChunkKdaFwdResult chunk_kda_fwd(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &g, const at::Tensor &beta, double scale,
    const c10::optional<at::Tensor> &initial_state,
    bool output_final_state,
    const c10::optional<at::Tensor> &cu_seqlens,
    c10::OptionalIntArrayRef cu_seqlens_cpu,
    const c10::optional<at::Tensor> &chunk_indices,
    c10::OptionalIntArrayRef chunk_indices_cpu, int64_t chunk_size,
    c10::string_view layout, bool safe_gate, double lower_bound,
    bool use_gate_in_kernel, bool state_v_first,
    const c10::optional<at::Tensor> &a_log,
    const c10::optional<at::Tensor> &dt_bias,
    bool return_intermediate_states, bool disable_recompute);

}  // namespace sglang::npu_kernel
