"""Build LOCAL_URMA copy descriptors without reading page indices on the CPU."""

import torch
import triton
import triton.language as tl

# Current Memfabric LOCAL_URMA capacity: HCOMM_BATCH_TRANSFER_MAX_DESC_NUM
# in src/hybm/csrc/under_api/dl_hcomm_api.h; see offload_sparse_copy's contract.
MAX_SPARSE_COPY_ENTRIES = 8190
_LENGTH_DTYPE = torch.int32  # Matches the current Memfabric lengths ABI.


@triton.jit(
    do_not_specialize=[
        "device_base",
        "host_base",
        "component_offset",
        "entries",
        "total_entries",
        "device_pages",
        "host_pages",
    ]
)
def _build_kv_sparse_metadata_kernel(
    device_indices,
    host_indices,
    src,
    dst,
    lengths,
    count,
    device_base,
    host_base,
    component_offset,
    entries,
    total_entries,
    device_pages,
    host_pages,
    DEVICE_LAYER_STRIDE: tl.constexpr,
    DEVICE_PAGE_STRIDE: tl.constexpr,
    HOST_LAYER_STRIDE: tl.constexpr,
    HOST_PAGE_STRIDE: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    LAYERS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    H2D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < entries
    entry = offsets.to(tl.int64)
    output = component_offset + entry
    page, layer = entry // LAYERS, entry % LAYERS
    dt = tl.load(device_indices + page * PAGE_SIZE, mask=mask, other=0)
    ht = tl.load(host_indices + page * PAGE_SIZE, mask=mask, other=0)
    dp, hp = dt // PAGE_SIZE, ht // PAGE_SIZE
    valid = (
        (dt >= 0)
        & (ht >= 0)
        & (dt % PAGE_SIZE == 0)
        & (ht % PAGE_SIZE == 0)
        & (dp < device_pages)
        & (hp < host_pages)
    )
    daddr = device_base.to(tl.uint64) + (
        layer * DEVICE_LAYER_STRIDE + dp * DEVICE_PAGE_STRIDE
    ).to(tl.uint64)
    haddr = host_base.to(tl.uint64) + (
        layer * HOST_LAYER_STRIDE + hp * HOST_PAGE_STRIDE
    ).to(tl.uint64)
    # Invalid descriptors use pool bases and a negative length, which the
    # LOCAL_URMA consumer rejects asynchronously before dereferencing them.
    daddr = tl.where(valid, daddr, device_base.to(tl.uint64))
    haddr = tl.where(valid, haddr, host_base.to(tl.uint64))
    tl.store(src + output, haddr if H2D else daddr, mask=mask)
    tl.store(dst + output, daddr if H2D else haddr, mask=mask)
    tl.store(lengths + output, tl.where(valid, PAGE_BYTES, -1), mask=mask)
    if (component_offset == 0) & (tl.program_id(0) == 0):
        tl.store(count, total_entries)


def _check_component(device, host, page_size, index_device):
    if device.device != index_device or host.device.type != "cpu":
        raise ValueError("KV components must be on the index NPU and CPU respectively")
    if device.ndim != 5 or host.ndim != 5:
        raise ValueError(
            "expected device [layer,page,token,head,dim] and host [page,layer,token,head,dim]"
        )
    if (
        device.dtype != host.dtype
        or device.shape[0] != host.shape[1]
        or device.shape[0] <= 0
    ):
        raise ValueError("KV component dtypes and positive layer counts must match")
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
    if not 0 < page_bytes <= torch.iinfo(_LENGTH_DTYPE).max:
        raise ValueError(
            f"sparse_copy page byte count {page_bytes} must fit a positive {_LENGTH_DTYPE}"
        )


def build_kv_sparse_metadata(
    components, device_indices, host_indices, page_size, direction
):
    """Build address tables and one count for a single sparse_copy call.

    The transfer entry point has normalized indices and selected paired buffers.
    Validate payload layouts once, then launch once per active component.
    """
    for dev, host in components:
        _check_component(dev, host, page_size, device_indices.device)
    pages = device_indices.numel() // page_size
    total = pages * sum(dev.shape[0] for dev, _ in components)
    if total > MAX_SPARSE_COPY_ENTRIES:
        raise ValueError(
            f"single-batch sparse_copy supports at most {MAX_SPARSE_COPY_ENTRIES} "
            f"descriptors, got {total}"
        )
    device = device_indices.device
    src = torch.empty((total,), dtype=torch.int64, device=device)
    dst = torch.empty_like(src)
    lengths = torch.empty((total,), dtype=_LENGTH_DTYPE, device=device)
    count = torch.empty((1,), dtype=torch.int32, device=device)
    component_offset = 0
    for dev, host in components:
        entries = pages * dev.shape[0]
        if entries:
            item_size = dev.element_size()
            _build_kv_sparse_metadata_kernel[(triton.cdiv(entries, 128),)](
                device_indices,
                host_indices,
                src,
                dst,
                lengths,
                count,
                dev.data_ptr(),
                host.data_ptr(),
                component_offset,
                entries,
                total,
                dev.shape[1],
                host.shape[0],
                DEVICE_LAYER_STRIDE=dev.stride(0) * item_size,
                DEVICE_PAGE_STRIDE=dev.stride(1) * item_size,
                HOST_LAYER_STRIDE=host.stride(1) * item_size,
                HOST_PAGE_STRIDE=host.stride(0) * item_size,
                PAGE_BYTES=page_size * dev.shape[3] * dev.shape[4] * item_size,
                LAYERS=dev.shape[0],
                PAGE_SIZE=page_size,
                H2D=direction == 1,
                BLOCK=128,
            )
        component_offset += entries
    return src, dst, lengths, count
