"""LOCAL_URMA KV transfer: select components, build metadata, submit once."""

from collections import deque
from threading import Lock

import torch

_submit_lock = Lock()
_last_event = {}
_inflight = deque()


def _device_indices(indices, device, keepalive):
    if indices.device.type == "cpu":
        pinned = indices.pin_memory()
        keepalive.append(pinned)
        indices = pinned.to(device, non_blocking=True)
    elif indices.device != device:
        raise ValueError("KV indices must be on CPU or the KV buffer's NPU")
    return indices.contiguous()


def transfer_kv_sparse_copy(
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
    direction=1,
):
    """Copy complete KV pages through caller-initialized LOCAL_URMA.

    Device buffers are [layer,page,token,head,dim], Host buffers are
    [page,layer,token,head,dim]. K is mandatory; an empty device V denotes
    no separate V payload. Optional Indexer/scales are copied only when both
    device and Host operands are present, matching transfer_kv_dim_exchange.
    Payloads are copied unchanged. This entry point is for eager execution.
    """
    direction = int(getattr(direction, "value", direction))
    if direction not in (1, 2):
        raise ValueError("direction must be H2D (1) or D2H (2)")
    if page_size <= 0 or device_indices.ndim != 1 or host_indices.ndim != 1:
        raise ValueError("expected positive page_size and 1-D token indices")
    if device_indices.dtype != torch.int64 or host_indices.dtype != torch.int64:
        raise ValueError("KV indices must have dtype int64")
    if (
        device_indices.numel() != host_indices.numel()
        or device_indices.numel() % page_size
    ):
        raise ValueError("indices must describe equal numbers of complete pages")

    if device_k is None or host_k is None or not device_k.numel() or not host_k.numel():
        raise ValueError("K buffers must be nonempty")
    if device_v is None or host_v is None:
        raise ValueError(
            "V buffers must be tensors; use an empty device V for no V payload"
        )

    components = [(device_k, host_k)]
    if device_v.numel():
        if not host_v.numel():
            raise ValueError("nonempty device V requires a nonempty host V")
        components.append((device_v, host_v))

    if device_index_k is not None and host_index_k is not None:
        if device_index_k.numel() or host_index_k.numel():
            if not device_index_k.numel() or not host_index_k.numel():
                raise ValueError("Indexer K buffers must be nonempty on both ends")
            components.append((device_index_k, host_index_k))

    if device_index_k_scale is not None and host_index_k_scale is not None:
        if device_index_k_scale.numel() or host_index_k_scale.numel():
            if not device_index_k_scale.numel() or not host_index_k_scale.numel():
                raise ValueError("Indexer scale buffers must be nonempty on both ends")
            # A metadata view only; scale values and storage stay unchanged.
            if device_index_k_scale.ndim == 4:
                device_index_k_scale = device_index_k_scale.unsqueeze(-1)
            components.append((device_index_k_scale, host_index_k_scale))
    if not device_indices.numel():
        return

    from memfabric_hybrid import offload
    from sgl_kernel_npu.implementation.triton.mem_cache.kv_sparse import (
        build_kv_sparse_metadata,
    )

    if offload._get_scene() != offload.Scene.LOCAL_URMA:
        raise RuntimeError(
            "transfer_kv_sparse_copy requires caller-initialized Memfabric Scene.LOCAL_URMA"
        )
    device = device_k.device
    if device.type != "npu":
        raise ValueError("KV device buffers must be NPU tensors")
    with _submit_lock, torch.npu.device(device):
        while _inflight and _inflight[0][0].query():
            _inflight.popleft()
        stream = torch.npu.current_stream()
        previous = _last_event.get(device)
        if previous is not None:
            stream.wait_event(previous)
        keepalive = [
            device_k,
            host_k,
            device_v,
            host_v,
            device_index_k,
            host_index_k,
            device_index_k_scale,
            host_index_k_scale,
            device_indices,
            host_indices,
        ]
        try:
            di = _device_indices(device_indices, device, keepalive)
            hi = _device_indices(host_indices, device, keepalive)
            keepalive.extend((di, hi))
            for dev, _ in components:
                dev.record_stream(stream)
            di.record_stream(stream)
            hi.record_stream(stream)

            src, dst, lengths, count = build_kv_sparse_metadata(
                components, di, hi, page_size, direction
            )
            keepalive.extend((src, dst, lengths, count))
            copy_direction = (
                offload.CopyDirection.H2D
                if direction == 1
                else offload.CopyDirection.D2H
            )
            ret = offload.sparse_copy(src, dst, lengths, count, device, copy_direction)
            if ret != 0:
                raise RuntimeError(f"LOCAL_URMA sparse_copy failed: code={ret}")

        finally:
            # Keep submitted work alive on failure too. LOCAL_URMA shares one
            # resource set, so the next stream waits through this completion.
            done = torch.npu.Event()
            done.record(stream)
            _last_event[device] = done
            _inflight.append((done, keepalive))
