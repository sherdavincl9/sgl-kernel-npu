"""Run explicitly on A5 with a matching LOCAL_URMA Memfabric installation."""

import os
import unittest


@unittest.skipUnless(
    os.getenv("SGL_KERNEL_NPU_TEST_LOCAL_URMA") == "1", "requires A5 LOCAL_URMA"
)
class TestTransferKVSparse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import torch_npu  # noqa: F401
        from memfabric_hybrid import offload
        from sgl_kernel_npu.implementation.triton.mem_cache.kv_sparse import (
            build_kv_sparse_metadata,
        )
        from sgl_kernel_npu.kvcacheio import TransferDirection, transfer_kv_sparse_copy

        cls.build_metadata = staticmethod(build_kv_sparse_metadata)

        cls.torch, cls.offload = torch, offload
        cls.direction, cls.transfer = TransferDirection, staticmethod(
            transfer_kv_sparse_copy
        )
        torch.npu.set_device(0)
        config = offload.OffloadConfig()
        config.device_id = 0
        config.reserve_size = config.alloc_size = 1 << 30
        config.scene = offload.Scene.LOCAL_URMA
        assert offload.initialize(config) == 0

    @classmethod
    def tearDownClass(cls):
        cls.torch.npu.synchronize()
        cls.offload.uninitialize()

    def indices(self, pages, page_size):
        t = self.torch
        return (
            t.tensor(pages, device="npu", dtype=t.int64)[:, None] * page_size
            + t.arange(page_size, device="npu")
        ).flatten()

    def test_triton_metadata_uses_strides_offsets_and_direction(self):
        t = self.torch
        dev = t.empty(3, 8, 4, 1, 5, dtype=t.uint8, device="npu")
        host = self.offload.empty([10, 5, 4, 1, 5], dtype=t.uint8)[:, 1:4]
        di, hi = self.indices([6, 2], 4), self.indices([1, 8], 4)
        for direction in (1, 2):
            src, dst, lengths, count = self.build_metadata(
                ((dev, host),), di, hi, 4, direction
            )
            expected_d = [
                dev[layer, dp].data_ptr() for dp in (6, 2) for layer in range(3)
            ]
            expected_h = [
                host[hp, layer].data_ptr() for hp in (1, 8) for layer in range(3)
            ]
            self.assertEqual(
                src.cpu().tolist(), expected_h if direction == 1 else expected_d
            )
            self.assertEqual(
                dst.cpu().tolist(), expected_d if direction == 1 else expected_h
            )
            self.assertEqual(lengths.cpu().tolist(), [20] * 6)
            self.assertEqual(count.cpu().item(), 6)

    def test_roundtrip_all_components_with_host_layer_slice(self):
        t = self.torch
        ps, layers, pages = 4, 3, 8
        devs, hosts, full_hosts, originals = [], [], [], []
        # Packed K, empty V, sparse Indexer and 4-D FP32 scale.
        for nlayer, width, dtype in (
            (layers, 7, t.uint8),
            (2, 3, t.uint8),
            (2, 1, t.float32),
        ):
            shape = (nlayer, pages, ps, 1, width)
            values = (t.arange(nlayer * pages * ps * width).reshape(shape) % 127).to(
                dtype
            )
            devs.append(values.npu())
            originals.append(values)
            full = self.offload.empty([10, nlayer + 2, ps, 1, width], dtype=dtype)
            full.fill_(255)
            full_hosts.append(full)
            hosts.append(full[:, 1 : nlayer + 1])
        empty = t.empty(0, device="npu")
        host_empty = t.empty(0)
        source = self.indices([6, 2], ps)
        saved = self.indices([1, 8], ps)
        target = self.indices([0, 5], ps)
        write, load = t.npu.Stream(), t.npu.Stream()
        ready = t.npu.Event()
        ready.record()
        kwargs = dict(
            device_k=devs[0],
            host_k=hosts[0],
            device_v=empty,
            host_v=host_empty,
            device_index_k=devs[1],
            host_index_k=hosts[1],
            device_index_k_scale=devs[2].squeeze(-1),
            host_index_k_scale=hosts[2],
            page_size=ps,
        )
        with t.npu.stream(write):
            write.wait_event(ready)
            self.transfer(source, saved, direction=self.direction.D2H, **kwargs)
        # No device synchronize between streams: wrapper must order the shared
        # URMA resource and retain all temporary pointer/index tensors.
        with t.npu.stream(load):
            self.transfer(target, saved, direction=self.direction.H2D, **kwargs)
        load.synchronize()
        for dev, host, full, original in zip(devs, hosts, full_hosts, originals):
            expected = original[:, [6, 2]]
            self.assertTrue(t.equal(dev[:, [0, 5]].cpu(), expected))
            self.assertTrue(t.equal(host[[1, 8]].transpose(0, 1), expected))
            self.assertTrue(t.all(full[:, 0] == 255))
            self.assertTrue(t.all(full[0] == 255))

    def test_service_sized_bf16_roundtrip_with_cpu_indices(self):
        """Seven 128-token pages, separate K/V and fewer Indexer layers.

        Check bytes against an independent CPU snapshot, including every
        device page that should remain untouched. No model is loaded.
        """
        t = self.torch
        ps, device_pages, host_pages = 128, 33, 17
        source_pages = [32, 1, 29, 4, 23, 7, 19]
        saved_pages = [1, 8, 15, 3, 12, 5, 16]
        target_pages = [0, 3, 11, 15, 20, 26, 31]
        generator = t.Generator(device="cpu").manual_seed(20261029)
        components = []
        for name, layers, width in (
            ("K", 78, 512),
            ("V", 78, 64),
            ("Indexer K", 21, 128),
        ):
            shape = (layers, device_pages, ps, 1, width)
            original = t.randint(
                0, 256, shape, dtype=t.uint8, generator=generator, device="cpu"
            ).to(t.bfloat16)
            dev = original.npu()
            full_host = self.offload.empty(
                [host_pages, layers + 2, ps, 1, width], dtype=t.bfloat16
            ).fill_(-77)
            components.append(
                (name, dev, full_host[:, 1 : layers + 1], full_host, original)
            )

        def cpu_indices(pages):
            return (
                t.tensor(pages, dtype=t.int64, device="cpu")[:, None] * ps
                + t.arange(ps, dtype=t.int64, device="cpu")
            ).flatten()

        source = cpu_indices(source_pages)
        saved = cpu_indices(saved_pages)
        target = cpu_indices(target_pages)
        kwargs = dict(
            device_k=components[0][1],
            host_k=components[0][2],
            device_v=components[1][1],
            host_v=components[1][2],
            device_index_k=components[2][1],
            host_index_k=components[2][2],
            page_size=ps,
        )
        write, load = t.npu.Stream(), t.npu.Stream()
        ready = t.npu.Event()
        ready.record()
        with t.npu.stream(write):
            write.wait_event(ready)
            self.transfer(source, saved, direction=self.direction.D2H, **kwargs)
        with t.npu.stream(load):
            self.transfer(target, saved, direction=self.direction.H2D, **kwargs)
        load.synchronize()

        def assert_bytes_equal(actual, expected, label):
            actual_bytes = actual.contiguous().view(t.uint8)
            expected_bytes = expected.contiguous().view(t.uint8)
            self.assertTrue(t.equal(actual_bytes, expected_bytes), label)

        for name, dev, host, full_host, original in components:
            payload = original[:, source_pages]
            assert_bytes_equal(
                host[saved_pages].transpose(0, 1),
                payload,
                f"{name}: D2H Host payload differs",
            )
            expected = original.clone()
            expected[:, target_pages] = payload
            assert_bytes_equal(
                dev.cpu(), expected, f"{name}: H2D bytes or untouched device pages differ"
            )
            # Guard Host layers and all pages outside the submitted set.
            self.assertTrue(
                t.all(full_host[:, 0] == -77), f"{name}: first guard layer"
            )
            self.assertTrue(
                t.all(full_host[:, -1] == -77), f"{name}: last guard layer"
            )
            untouched = [i for i in range(host_pages) if i not in saved_pages]
            self.assertTrue(
                t.all(host[untouched] == -77), f"{name}: untouched Host pages"
            )

    def test_large_bf16_page_and_invalid_page_metadata(self):
        t = self.torch
        ps = 128
        values = (t.arange(2 * 3 * ps * 512).reshape(2, 3, ps, 1, 512) % 127).to(
            t.bfloat16
        )
        dev = values.npu()
        host = self.offload.empty([4, 2, ps, 1, 512], dtype=t.bfloat16).zero_()
        di, hi = self.indices([2], ps), self.indices([1], ps)
        empty = t.empty(0, device="npu")
        self.transfer(
            di,
            hi,
            dev,
            host,
            empty,
            t.empty(0),
            page_size=ps,
            direction=self.direction.D2H,
        )
        t.npu.current_stream().synchronize()
        self.assertTrue(t.equal(host[1], values[:, 2]))
        # Invalid page starts must produce a rejected descriptor, not an
        # out-of-bounds pointer. Do not submit it to URMA in this test.
        invalid = self.indices([3], ps)
        _, _, lengths, _ = self.build_metadata(((dev, host),), invalid, hi, ps, 2)
        self.assertEqual(lengths.cpu().tolist(), [-1, -1])

    def test_single_submission_with_k_and_v(self):
        t = self.torch
        layers, pages, ps = 3, 8, 4
        values = (
            t.arange(layers * pages * ps).reshape(layers, pages, ps, 1, 1) % 127
        ).to(t.uint8)
        dev_k, dev_v = values.npu(), (values + 1).npu()
        host_k = self.offload.empty([10, layers, ps, 1, 1], dtype=t.uint8).zero_()
        host_v = self.offload.empty(list(host_k.shape), dtype=t.uint8).zero_()
        di, hi = self.indices([6, 2], ps), self.indices([1, 8], ps)
        self.transfer(
            di,
            hi,
            dev_k,
            host_k,
            dev_v,
            host_v,
            page_size=ps,
            direction=self.direction.D2H,
        )
        t.npu.synchronize()
        self.assertTrue(t.equal(host_k[[1, 8]].transpose(0, 1), values[:, [6, 2]]))
        self.assertTrue(
            t.equal(host_v[[1, 8]].transpose(0, 1), (values + 1)[:, [6, 2]])
        )


if __name__ == "__main__":
    unittest.main()
