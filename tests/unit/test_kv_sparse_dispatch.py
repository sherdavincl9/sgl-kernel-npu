"""CPU checks of the real wrapper's submission, lifetime and stream ordering.

Only the NPU/Memfabric boundary is mocked; no PyTorch/CANN install is needed.
"""

import importlib.util
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Device:
    type: str
    index: int = 0


NPU = Device("npu")
CPU = Device("cpu")


class Tensor:
    def __init__(self, shape, device=NPU, dtype="int64"):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.device = device
        self.dtype = dtype
        self.recorded = []
        self.pinned = None

    def numel(self):
        n = 1
        for size in self.shape:
            n *= size
        return n

    def contiguous(self):
        return self

    def pin_memory(self):
        self.pinned = Tensor(self.shape, CPU, self.dtype)
        return self.pinned

    def to(self, device, non_blocking=False):
        assert non_blocking
        return Tensor(self.shape, device, self.dtype)

    def record_stream(self, stream):
        self.recorded.append(stream)

    def unsqueeze(self, dim):
        return Tensor((*self.shape, 1), self.device, self.dtype)


class Event:
    def __init__(self):
        self.done = False
        self.stream = None

    def query(self):
        return self.done

    def record(self, stream):
        self.stream = stream


class Context:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class TestSparseDispatch(unittest.TestCase):
    def setUp(self):
        self.stream = SimpleNamespace(wait_event=Mock())

        def build(components, di, hi, page_size, direction):
            total = di.numel() // page_size * sum(d.shape[0] for d, _ in components)
            self.tables = [
                Tensor((total,)),
                Tensor((total,)),
                Tensor((total,), dtype="int32"),
                Tensor((1,), dtype="int32"),
            ]
            return self.tables

        self.builder = Mock(side_effect=build)
        self.torch = SimpleNamespace(
            Tensor=Tensor,
            int64="int64",
            npu=SimpleNamespace(
                device=lambda _: Context(),
                current_stream=lambda: self.stream,
                Event=Event,
            ),
        )
        self.offload = SimpleNamespace(
            _get_scene=lambda: 2,
            Scene=SimpleNamespace(LOCAL_URMA=2),
            CopyDirection=SimpleNamespace(H2D=1, D2H=2),
            sparse_copy=Mock(return_value=0),
        )
        self.modules = patch.dict(
            sys.modules,
            {
                "torch": self.torch,
                "sgl_kernel_npu.implementation.triton.mem_cache.kv_sparse": SimpleNamespace(
                    build_kv_sparse_metadata=self.builder
                ),
                "memfabric_hybrid": SimpleNamespace(offload=self.offload),
            },
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)
        path = ROOT / "python/sgl_kernel_npu/sgl_kernel_npu/_kv_sparse.py"
        spec = importlib.util.spec_from_file_location("_test_kv_sparse", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.modules_alias = patch.dict(
            sys.modules, {"sgl_kernel_npu._kv_sparse": self.module}
        )
        self.modules_alias.start()
        self.addCleanup(self.modules_alias.stop)

    def transfer(self, layers=3, pages=2, direction=1, cpu_indices=False):
        dev = Tensor((layers, pages + 3, 4, 1, 5), dtype="bf16")
        host = Tensor((pages + 5, layers, 4, 1, 5), CPU, "bf16")
        di = Tensor((pages * 4,), CPU if cpu_indices else NPU)
        hi = Tensor((pages * 4,), CPU if cpu_indices else NPU)
        self.module.transfer_kv_sparse_copy(
            di,
            hi,
            dev,
            host,
            Tensor((0,)),
            Tensor((0,), CPU),
            page_size=4,
            direction=direction,
        )
        return dev, host, di, hi

    def test_one_submission_uses_complete_metadata_and_retains_it(self):
        self.transfer()
        self.builder.assert_called_once()
        self.offload.sparse_copy.assert_called_once()
        args = self.offload.sparse_copy.call_args.args
        retained = self.module._inflight[0][1]
        for submitted, table in zip(args[:4], self.tables):
            self.assertIs(submitted, table)
            self.assertIn(table, retained)
        self.assertEqual(args[-1], 1)

    def test_cross_stream_dependency_covers_both_directions(self):
        self.transfer()
        first = self.module._last_event[NPU]
        second_stream = SimpleNamespace(wait_event=Mock())
        self.stream = second_stream
        self.transfer(direction=2)
        second_stream.wait_event.assert_called_once_with(first)
        self.assertEqual(self.offload.sparse_copy.call_args.args[-1], 2)
        self.assertIs(self.module._last_event[NPU].stream, second_stream)

    def test_submission_failure_keeps_metadata_and_orders_next_call(self):
        self.offload.sparse_copy.return_value = 7
        with self.assertRaisesRegex(RuntimeError, "code=7"):
            self.transfer()
        failed = self.module._last_event[NPU]
        self.assertTrue(self.module._inflight)
        self.offload.sparse_copy.return_value = 0
        self.transfer()
        self.stream.wait_event.assert_called_with(failed)

    def test_pinned_cpu_staging_is_retained_and_released_after_completion(self):
        _, _, di, hi = self.transfer(cpu_indices=True)
        retained = self.module._inflight[0][1]
        self.assertIn(di.pinned, retained)
        self.assertIn(hi.pinned, retained)
        self.module._inflight[0][0].done = True
        self.transfer()
        self.assertEqual(len(self.module._inflight), 1)

    def test_wrong_scene_and_bad_index_shape_fail_before_submission(self):
        self.offload._get_scene = lambda: 0
        with self.assertRaisesRegex(RuntimeError, "LOCAL_URMA"):
            self.transfer()
        self.builder.assert_not_called()
        with self.assertRaisesRegex(ValueError, "complete pages"):
            self.module.transfer_kv_sparse_copy(
                Tensor((3,)),
                Tensor((3,)),
                Tensor((1, 2, 4, 1, 5)),
                Tensor((2, 1, 4, 1, 5), CPU),
                Tensor((0,)),
                Tensor((0,), CPU),
                page_size=4,
            )
        self.offload.sparse_copy.assert_not_called()

    def test_empty_transfer_has_no_backend_side_effect(self):
        self.transfer(pages=0)
        self.builder.assert_not_called()
        self.offload.sparse_copy.assert_not_called()

    def public_wrapper(self):
        spec = importlib.util.spec_from_file_location(
            "_test_public_kvcacheio",
            ROOT / "python/sgl_kernel_npu/sgl_kernel_npu/kvcacheio.py",
        )
        public = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(public)
        return public.transfer_kv_sparse_copy

    def test_public_wrapper_rejects_invalid_required_or_empty_buffers(self):
        call = self.public_wrapper()
        k = Tensor((3, 8, 4, 1, 7))
        host_k = Tensor((9, 3, 4, 1, 7), CPU)
        v = Tensor((3, 8, 4, 1, 2))
        host_v = Tensor((9, 3, 4, 1, 2), CPU)
        args = dict(
            device_indices=Tensor((8,)),
            host_indices=Tensor((8,)),
            device_k=k,
            host_k=host_k,
            device_v=v,
            host_v=host_v,
            page_size=4,
        )
        invalid = (
            {"device_k": None},
            {"host_k": None},
            {"device_k": Tensor((0,))},
            {"host_k": Tensor((0,), CPU)},
            {"device_v": None},
            {"host_v": None},
            {"host_v": Tensor((0,), CPU)},
            {"device_index_k": Tensor((0,)), "host_index_k": host_k},
            {"device_index_k_scale": k, "host_index_k_scale": Tensor((0,), CPU)},
        )
        for kwargs in invalid:
            with self.subTest(overrides=tuple(kwargs)):
                with self.assertRaises(ValueError):
                    call(**dict(args, **kwargs))
        self.builder.assert_not_called()
        self.offload.sparse_copy.assert_not_called()

    def test_optional_components_skip_either_missing_end_independently(self):
        call = self.public_wrapper()
        k, h = Tensor((3, 8, 4, 1, 7)), Tensor((9, 3, 4, 1, 7), CPU)
        index, host_index = Tensor((2, 8, 4, 1, 3)), Tensor((9, 2, 4, 1, 3), CPU)
        scale, host_scale = Tensor((2, 8, 4, 1)), Tensor((9, 2, 4, 1, 1), CPU)
        for missing in (
            "device_index_k",
            "host_index_k",
            "device_index_k_scale",
            "host_index_k_scale",
        ):
            for direction in (1, 2):
                with self.subTest(missing=missing, direction=direction):
                    optional = dict(
                        device_index_k=index,
                        host_index_k=host_index,
                        device_index_k_scale=scale,
                        host_index_k_scale=host_scale,
                    )
                    optional[missing] = None
                    index.recorded.clear()
                    scale.recorded.clear()
                    call(
                        Tensor((8,)),
                        Tensor((8,)),
                        k,
                        h,
                        Tensor((0,)),
                        Tensor((0,), CPU),
                        page_size=4,
                        direction=direction,
                        **optional,
                    )
                    pairs = self.builder.call_args.args[0]
                    self.assertEqual(len(pairs), 2)
                    if missing.endswith("_scale"):
                        self.assertEqual(pairs, [(k, h), (index, host_index)])
                        self.assertEqual(scale.recorded, [])
                    else:
                        self.assertEqual(pairs[1][0].shape, (2, 8, 4, 1, 1))
                        self.assertIs(pairs[1][1], host_scale)
                        self.assertEqual(index.recorded, [])

    def test_public_wrapper_accepts_empty_device_v_with_host_placeholder(self):
        call = self.public_wrapper()
        k = Tensor((3, 8, 4, 1, 7), dtype="fp8")
        host_k = Tensor((9, 3, 4, 1, 7), CPU, "fp8")
        host_v = Tensor((9, 3, 4, 1, 2), CPU, "bf16")
        call(
            Tensor((8,)),
            Tensor((8,)),
            k,
            host_k,
            Tensor((3, 8, 4, 1, 0)),
            host_v,
            page_size=4,
        )
        self.assertEqual(self.builder.call_args.args[0], [(k, host_k)])

    def test_public_wrapper_skips_only_paired_empty_optional_buffers(self):
        call = self.public_wrapper()
        k, h = Tensor((3, 8, 4, 1, 7)), Tensor((9, 3, 4, 1, 7), CPU)
        v, hv = Tensor((3, 8, 4, 1, 2)), Tensor((9, 3, 4, 1, 2), CPU)
        call(
            Tensor((8,)),
            Tensor((8,)),
            k,
            h,
            v,
            hv,
            device_index_k=Tensor((0,)),
            host_index_k=Tensor((0,), CPU),
            page_size=4,
        )
        self.assertEqual(self.builder.call_args.args[0], [(k, h), (v, hv)])

    def test_public_wrapper_skips_empty_v_and_pads_indexer_scale(self):
        call = self.public_wrapper()
        k = Tensor((3, 8, 4, 1, 7), dtype="fp8")
        h = Tensor((9, 3, 4, 1, 7), CPU, "fp8")
        scale = Tensor((2, 8, 4, 1), dtype="fp32")
        hs = Tensor((9, 2, 4, 1, 1), CPU, "fp32")
        call(
            Tensor((8,)),
            Tensor((8,)),
            k,
            h,
            Tensor((0,)),
            Tensor((0,), CPU),
            device_index_k_scale=scale,
            host_index_k_scale=hs,
            page_size=4,
        )
        pairs = self.builder.call_args.args[0]
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[1][0].shape, (2, 8, 4, 1, 1))
        self.assertIs(pairs[1][1], hs)


if __name__ == "__main__":
    unittest.main()
