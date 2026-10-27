# LOCAL_URMA HiCache KV transfer

`sgl_kernel_npu.kvcacheio.transfer_kv_sparse_copy` accepts the same buffer/index
arguments as `transfer_kv_dim_exchange` (without the FAST2D flag). It builds
NPU address tables with Triton-Ascend inside the package and calls Memfabric `offload.sparse_copy`.
SGLang never constructs pointer tables.

Prerequisites: A5 with Triton-Ascend, a Memfabric build providing LOCAL_URMA device metadata,
the matching HybmSparseCopy AICPU/launch extension, and caller-initialized
`offload.Scene.LOCAL_URMA`. All host payload tensors must belong to that
registered offload pool. This wrapper does not initialize/change the scene.

Device payload layout: [layer, page, page_size, head, dim].
Host payload layout: [page, layer, page_size, head, dim].
K is mandatory and nonempty. V arguments must be tensors: an empty device V
explicitly means no separate V payload, even when the host pool has an unused
V placeholder. A nonempty device V requires a nonempty host V.
Optional Indexer K and scale are each copied only when both their device and
host operands are present, matching the original transfer_kv_dim_exchange
wrapper. If either operand is None, that component is skipped. When both are
tensors, both must be nonempty or both empty (no payload).
SGLang passes native buffers unchanged; draft Indexer is skipped because its
host operand is None.
Scale tensors are opaque payloads: no quantization or dequantization is done.

Layer slices may be noncontiguous across pages; actual byte strides and storage
offsets are used. Each individual (page, layer) payload must be contiguous.
Indices are int64 token slots, grouped as complete aligned pages. CPU indices
are uploaded asynchronously through retained pinned staging; NPU indices are
never copied to the CPU. The device kernel marks invalid page starts with a
negative length so LOCAL_URMA rejects them asynchronously.

The transfer entry point selects a single sequence of (device, host) buffer
pairs. The Triton builder validates each component once and builds the full
address/length tables plus a single count in one pass per component.
Each transfer calls sparse_copy once. This initial implementation supports at
most 8190 descriptors per call and rejects larger requests without splitting
them. Host loops are bounded by components (at most four), rather than by pages.
Both directions use the caller's current NPU stream.
Event dependencies serialize H2D/D2H across streams because the current
LOCAL_URMA implementation shares one resource set. Metadata, staging and
payload references are retained until the last submitted operation completes.
Use this eager wrapper outside graph capture; replay resource management is
not implemented here. Drain all transfers before uninitializing Memfabric.

In SGLang's MLA kernel_ascend / page_first_kv_split path, only the two
transfer_kv_dim_exchange call sites are replaced by transfer_kv_sparse_copy.
The existing ascendc_io_enabled() branch and _transfer_ascendc_sparse_copy
implementation remain unchanged and still take precedence when enabled.
There is no new environment-variable switch for the replacement.
The application/test supplies LOCAL_URMA initialization and registered host
memory when exercising the replaced path; allocator/scene setup is unchanged.

A5 integration test (the test initializes its own dedicated LOCAL_URMA pool):
```bash
SGL_KERNEL_NPU_TEST_LOCAL_URMA=1 python -m unittest discover \
  -s tests/python/sgl_kernel_npu -p test_transfer_kv_sparse_copy.py -v
```
CPU wrapper scheduling checks (no CANN/PyTorch required; NumPy enables the extra TL-facade arithmetic checks):
```bash
python3 -m unittest discover -s tests/unit -p 'test_kv_sparse*.py' -v
```

## Service validation

Run the A5 integration test above before testing a model server. It initializes
its own LOCAL_URMA pool and covers H2D/D2H roundtrip, K/V, Indexer K and scale,
Host layer slices, and cross-stream ordering. Install the updated kernel
package in the same environment that runs the tests and SGLang. The repository
build entry point is bash build.sh -a kernels Ascend950; use its generated
wheel in output/ so the matching target provider and native libraries are
included.

A normal server launch is not sufficient to enable this replacement in the
current framework configuration:

- npu_memfabric.ensure_memfabric_capacity still initializes Scene.LOCAL.
- SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric also makes ascendc_io_enabled()
  true, selecting the unchanged AscendC branch before the new call sites.
- Leaving that variable unset does not by itself allocate the Host tensors
  from the registered LOCAL_URMA offload pool.

The A5 service setup must therefore supply LOCAL_URMA initialization and pool
allocation while selecting the replaced kernel_ascend / page_first_kv_split
path. This change intentionally does not alter the framework scene/allocator
setup or introduce another selection switch.

Once that setup is in place, validate an actual HiCache cycle: write a short
prefix to Host (D2H), evict its L1 Device cache while retaining L2 Host cache,
then request the prefix again to trigger H2D. Repeating a prompt that remains
in L1 does not exercise H2D. Confirm the new transfer function is reached in
both directions using a breakpoint, temporary logging, or a profiler. Start
with a small transfer within the single-call descriptor limit.

## Comparison branches

The AscendC implementation is saved on npu/hicache-sparse-copy-ascendc
in both sgl-kernel-npu and SGLang. The current npu/hicache-sparse-copy
branch uses Triton for metadata generation. Compare equivalent payloads and
transfer directions; the current wrapper has since been renamed and refactored.
The Triton branch adds no C++ operator registration or AscendC build source.

Warm up each implementation before timing (Triton includes JIT on first use).
Compare the same payloads, selected pages, component counts and stream ordering.
Measure both descriptor generation and full transfer; no A5 performance result
has been collected on this development machine.
