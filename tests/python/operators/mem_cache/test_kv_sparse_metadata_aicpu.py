"""User-run NPU checks for the AICPU builder; no MF entity required."""

import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
pytest.importorskip("sgl_kernel_npu")

build = getattr(torch.ops.npu, "kv_sparse_metadata_aicpu", None)
pytestmark = pytest.mark.skipif(
    build is None, reason="build a wheel with kv_sparse_metadata"
)


def indices(starts, page_size):
    return torch.tensor(
        [start + offset for start in starts for offset in range(page_size)],
        dtype=torch.int64,
        device="npu",
    )


@pytest.mark.parametrize("layout", ["page_first", "layer_first"])
@pytest.mark.parametrize("direction", [1, 2])
@pytest.mark.parametrize("component_count", [1, 2, 3, 4])
def test_addresses_and_updated_indices(layout, direction, component_count):
    ps = 4
    devices, hosts = [], []
    for layers, width, dtype in (
        (43, 5, torch.bfloat16),
        (2, 2, torch.bfloat16),
        (65, 3, torch.uint8),
        (1, 1, torch.float32),
    )[:component_count]:
        device = torch.empty((layers + 2, 8, ps, 1, width), dtype=dtype, device="npu")
        device = device[1 : layers + 1]
        shape = (10, layers + 2, ps, 1, width)
        if layout == "layer_first":
            shape = (layers + 2, 10, ps, 1, width)
        host = torch.empty(shape, dtype=dtype)
        host = (
            host[1 : layers + 1].transpose(0, 1)
            if layout == "layer_first"
            else host[:, 1 : layers + 1]
        )
        devices.append(device)
        hosts.append(host)

    hi = indices([4, 32, 12], ps)
    di = torch.empty(2 * hi.numel(), dtype=hi.dtype, device=hi.device)[::2]
    for dstarts, hstarts in (
        ([24, 8, 16], [4, 32, 12]),
        ([0, 28, 4], [12, 0, 36]),
    ):
        # Keep addresses and shapes fixed; the kernel must read new NPU values.
        di.copy_(indices(dstarts, ps))
        hi.copy_(indices(hstarts, ps))
        src, dst, lengths, count = build(di, hi, devices, hosts, ps, direction)
        expected_src, expected_dst, expected_lengths = [], [], []
        for dev, host in zip(devices, hosts):
            item = dev.element_size()
            for dt, ht in zip(dstarts, hstarts):
                for layer in range(dev.shape[0]):
                    da = dev.data_ptr() + (
                        layer * dev.stride(0) + dt // ps * dev.stride(1)
                    ) * item
                    ha = host.data_ptr() + (
                        layer * host.stride(1) + ht // ps * host.stride(0)
                    ) * item
                    expected_src.append(ha if direction == 1 else da)
                    expected_dst.append(da if direction == 1 else ha)
                    expected_lengths.append(ps * dev.shape[3] * dev.shape[4] * item)
        # Synchronize only in the test, never in the implementation.
        assert src.cpu().tolist() == expected_src
        assert dst.cpu().tolist() == expected_dst
        assert lengths.cpu().tolist() == expected_lengths
        assert count.cpu().item() == len(expected_lengths)


def _check_invalid_page_aborts_batch():
    ps = 4
    dev = torch.empty((1, 2, ps, 1, 1), dtype=torch.uint8, device="npu")
    host = torch.empty((2, 1, ps, 1, 1), dtype=torch.uint8)
    di, hi = indices([0, 1], ps), indices([0, 4], ps)
    stream = torch.npu.current_stream()
    stream.synchronize()
    with pytest.raises(RuntimeError):
        build(di, hi, [dev], [host], ps, 2)
        stream.synchronize()


def test_invalid_page_aborts_batch():
    # An AICPU task error can leave the stream in an error state.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1])"
            "['_check_invalid_page_aborts_batch']()",
            __file__,
        ],
        check=True,
        timeout=120,
    )


def test_record_limit():
    dev = torch.empty((1, 8, 1, 1, 1), dtype=torch.uint8, device="npu")
    host = torch.empty((10, 1, 1, 1, 1), dtype=torch.uint8)
    di = torch.zeros(65537, dtype=torch.int64, device="npu")
    with pytest.raises(RuntimeError, match="65536"):
        build(di, di, [dev], [host], 1, 1)


def test_empty_tables():
    dev = torch.empty((1, 8, 4, 1, 1), dtype=torch.uint8, device="npu")
    host = torch.empty((10, 1, 4, 1, 1), dtype=torch.uint8)
    di = torch.empty(0, dtype=torch.int64, device="npu")
    src, dst, lengths, count = build(di, di, [dev], [host], 4, 1)
    assert src.numel() == dst.numel() == lengths.numel() == 0
    assert count.cpu().item() == 0
