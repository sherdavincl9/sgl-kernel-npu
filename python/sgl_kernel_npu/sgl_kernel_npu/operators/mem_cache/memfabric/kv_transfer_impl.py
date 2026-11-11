"""MemFabric KV transfers on the caller's current NPU stream.

The caller owns the registered Host storage. The ACLNN adapter records AIV
inputs on the caller's stream. AICPU retains inputs and metadata until its
completion event. NPU graph capture is not supported.
"""

import os
from collections import deque
from threading import Lock

import torch

_submit_lock = Lock()
_inflight = deque()
_last_sparse_submission = {}


def _check_component(device, host, page_size, index_device):
    if device.device != index_device or host.device.type != "cpu":
        raise ValueError("KV components must be on the index NPU and CPU respectively")
    # _layer_views already checked dimensions and selected equal layer counts.
    if device.dtype != host.dtype or device.shape[0] <= 0:
        raise ValueError(
            "KV component dtypes must match and layer count must be positive"
        )
    if device.shape[2:] != host.shape[2:] or device.shape[2] != page_size:
        raise ValueError("KV component page shapes must match")
    for tensor in (device, host):
        if (
            tensor.stride(4) != 1
            or tensor.stride(3) != tensor.shape[4]
            or tensor.stride(2) != tensor.shape[3] * tensor.shape[4]
        ):
            raise ValueError("each (page,layer) payload must be contiguous")
    page_bytes = page_size * device.shape[3] * device.shape[4] * device.element_size()
    if page_bytes <= 0:
        raise ValueError("KV component page byte count must be positive")


def _layer_views(dev, host, host_layout, layers):
    if dev.ndim != 5 or host.ndim != 5:
        raise ValueError(
            "KV components must be five-dimensional (Device scales may be four-dimensional)"
        )
    if host_layout == "layer_first":
        host = host.transpose(0, 1)
    device_start, host_start, count = layers
    count = dev.shape[0] - device_start if count is None else count
    if (
        device_start < 0
        or host_start < 0
        or count < 0
        or device_start + count > dev.shape[0]
        or host_start + count > host.shape[1]
    ):
        raise ValueError("KV layer selection exceeds Device or Host storage")
    if device_start != 0 or count != dev.shape[0]:
        dev = dev.narrow(0, device_start, count)
    if host_start != 0 or count != host.shape[1]:
        host = host.narrow(1, host_start, count)
    return dev, host


def select_components(
    device_k,
    host_k,
    device_v,
    host_v,
    device_index_k,
    host_index_k,
    device_index_k_scale,
    host_index_k_scale,
    host_layout,
    kv_layers,
    index_layers,
):
    """Select the active K/V/Indexer/scale views."""
    if host_layout not in ("page_first", "layer_first"):
        raise ValueError("host_layout must be page_first or layer_first")
    if device_k is None or host_k is None or not device_k.numel() or not host_k.numel():
        raise ValueError("K buffers must be nonempty")
    if device_v is None or host_v is None:
        raise ValueError(
            "V buffers must be tensors; use an empty Device V for no V payload"
        )
    components = [_layer_views(device_k, host_k, host_layout, kv_layers)]
    if device_v.numel():
        if not host_v.numel():
            raise ValueError("nonempty Device V requires a nonempty Host V")
        components.append(_layer_views(device_v, host_v, host_layout, kv_layers))
    if index_layers[2] != 0:
        for dev, host, is_scale in (
            (device_index_k, host_index_k, False),
            (device_index_k_scale, host_index_k_scale, True),
        ):
            if dev is not None and host is not None:
                if dev.numel() or host.numel():
                    if not dev.numel() or not host.numel():
                        raise ValueError(
                            "paired Indexer components must both be nonempty"
                        )
                    if is_scale and dev.ndim == 4:
                        dev = dev.unsqueeze(-1)
                    components.append(
                        _layer_views(dev, host, host_layout, index_layers)
                    )
                    continue
    return components


def _resolve_host_dva(host):
    # Mirrors the AIV host_memory contract, device-validated there: the
    # registration table (host_register + host_get_device_pointer at
    # allocation time, interior views by byte offset) is the ONLY DVA source;
    # unregistered MF memory is an error with no HVA/get_dva fallback.
    from .host_memory import resolve_host_dva

    return resolve_host_dva(host)


def _submit_aicpu(components, di, hi, page_size, direction, device, keepalive):
    if os.environ.get("SGLANG_HICACHE_HOST_MEM_BACKEND", "").lower() == "acl":
        raise ValueError(
            "impl='aicpu' requires MF host-pool memory, not ACL host registration"
        )
    build = getattr(torch.ops.npu, "kv_sparse_metadata_aicpu", None)
    if build is None:
        raise RuntimeError("Rebuild sgl-kernel-npu with --ops kv_sparse_metadata")
    from .host_memory import get_memfabric_pool

    pool = get_memfabric_pool()
    if pool is None:
        from memfabric_hybrid import offload

        # Older MF callers initialize and allocate through the module API.
        pool = offload
    sparse_copy = getattr(pool, "sparse_copy", None)
    if not callable(sparse_copy):
        raise RuntimeError("The MemFabric pool must provide sparse_copy")

    src, dst, lengths, count = build(
        di,
        hi,
        [dev for dev, _ in components],
        [host for _, host in components],
        page_size,
        direction,
    )
    keepalive.extend((src, dst, lengths, count))
    ret = sparse_copy(src, dst, lengths, count, device)
    if ret != 0:
        raise RuntimeError(f"MF sparse_copy failed: code={ret}")


def _submit_aiv(components, di, hi, page_size, direction, host_layout):
    # Host tensors have PF-shaped metadata views even when their storage is LF.
    # Actual page/layer strides preserve the original physical layout.
    # The ACLNN adapter passes the component description to CANN tiling.
    host_dvas = [_resolve_host_dva(host) for _, host in components]
    torch.ops.npu.kv_exchange(
        di,
        hi,
        [dev for dev, _ in components],
        [host for _, host in components],
        host_dvas,
        page_size,
        direction,
        int(host_layout == "layer_first"),
    )


def transfer_kv(
    device_indices,
    host_indices,
    device_k,
    host_k,
    device_v,
    host_v,
    device_index_k=None,
    host_index_k=None,
    device_index_k_scale=None,
    host_index_k_scale=None,
    page_size=128,
    *,
    host_layout,
    direction,
    kv_layers,
    index_layers,
    backend="aiv",
):
    """Submit one selected pool/range without synchronizing the caller's stream."""
    if page_size <= 0 or device_indices.ndim != 1 or host_indices.ndim != 1:
        raise ValueError("expected positive page_size and 1-D token indices")
    if device_indices.dtype != torch.int64 or host_indices.dtype != torch.int64:
        raise ValueError("KV indices must have dtype int64")
    if (
        device_indices.numel() != host_indices.numel()
        or device_indices.numel() % page_size
    ):
        raise ValueError("indices must describe equal numbers of complete pages")
    if direction not in (1, 2) or backend not in ("aiv", "aicpu"):
        raise ValueError("expected H2D (1) or D2H (2), and aiv or aicpu")
    components = select_components(
        device_k,
        host_k,
        device_v,
        host_v,
        device_index_k,
        host_index_k,
        device_index_k_scale,
        host_index_k_scale,
        host_layout,
        kv_layers,
        index_layers,
    )
    device = device_k.device
    if device.type != "npu":
        raise ValueError("KV Device buffers must be NPU tensors")
    if device_indices.device != device or host_indices.device != device:
        raise ValueError("KV indices must be on the KV buffer's NPU")
    for dev, host in components:
        _check_component(dev, host, page_size, device)
    if not device_indices.numel():
        return
    with _submit_lock, torch.npu.device(device):
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("KV transfer currently supports eager execution only")
        di = device_indices.contiguous()
        hi = host_indices.contiguous()
        if backend == "aiv":
            _submit_aiv(components, di, hi, page_size, direction, host_layout)
            return

        while _inflight and _inflight[0][0].query():
            _inflight.popleft()
        stream = torch.npu.current_stream()
        previous = _last_sparse_submission.get(device)
        if previous is not None and stream != previous[0]:
            stream.wait_event(previous[1])
        keepalive = [tensor for pair in components for tensor in pair]
        keepalive.extend((device_indices, host_indices))
        try:
            _submit_aicpu(components, di, hi, page_size, direction, device, keepalive)
        finally:
            # Inputs and metadata survive failures too.
            done = torch.npu.Event()
            done.record(stream)
            _last_sparse_submission[device] = (stream, done)
            _inflight.append((done, keepalive))
