"""A3 regressions for tail seed lifetime, padded scratch, and Aqk diagonals.

Run against a rebuilt wheel on A3:
    python -m pytest -v tests/python/sgl_kernel_npu/test_chunk_kda_fwd_a3_regression.py
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")


@pytest.fixture(scope="module", autouse=True)
def require_a3_kernel():
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    name = torch.npu.get_device_name(0)
    if "910" not in name:
        pytest.skip(f"requires the A2/A3 generic path, found {name}")
    # A missing/broken installed kernel must fail on the target device.
    __import__("sgl_kernel_npu")


def _run(q, k, v, g, beta, chunk_size, lengths=None):
    metadata = {}
    if lengths is not None:
        boundaries = [0]
        for length in lengths:
            boundaries.append(boundaries[-1] + length)
        metadata["cu_seqlens"] = torch.tensor(
            boundaries, dtype=torch.int64, device="npu"
        )
    outputs = torch.ops.npu.chunk_kda_fwd(
        *(x.to("npu") for x in (q, k, v, g, beta)),
        chunk_size=chunk_size,
        scale=1.0,
        state_v_first=True,
        output_final_state=True,
        output_w=True,
        output_u=True,
        output_v_new=True,
        **metadata,
    )
    torch.npu.synchronize()
    return tuple(None if x is None else x.cpu() for x in outputs)


@pytest.mark.parametrize("key_dim", [128, 256])
@pytest.mark.parametrize(
    "chunk_size,tokens",
    [(64, t) for t in (1, 2, 15, 16, 17, 31, 32, 33, 63)]
    + [(128, t) for t in (63, 64, 65, 127)],
)
def test_tail_w_uses_immutable_seed(chunk_size, tokens, key_dim):
    # Zero gate and dyadic beta make the prepared seed exactly beta * K.
    # Repeated keys give nonzero off-diagonals, so overwriting an earlier
    # seed row changes subsequent output rows, even within one AIV subblock.
    k = torch.zeros((1, tokens, 2, key_dim), dtype=torch.bfloat16)
    k[..., 0] = 1
    q = k.clone()
    v = torch.zeros((1, tokens, 2, 128), dtype=torch.bfloat16)
    g = torch.zeros(k.shape, dtype=torch.float32)
    beta = torch.full(k.shape[:-1], 0.5, dtype=torch.float32)
    outputs = _run(q, k, v, g, beta, chunk_size)
    akk = outputs[4][..., :tokens].float().tril()
    seed = (k.float() * beta.unsqueeze(-1)).permute(0, 2, 1, 3)
    expected = torch.zeros_like(seed)
    # Match FP32 reduction order while keeping all seed rows immutable.
    for j in range(tokens):
        expected += akk[..., j : j + 1] * seed[:, :, j : j + 1, :]
    torch.testing.assert_close(
        outputs[5], expected.to(torch.bfloat16), rtol=0, atol=0
    )


@pytest.mark.parametrize("seq_count", [127, 128, 256])
def test_padded_post_wu_scratch_does_not_overwrite_u_seed(seq_count):
    lengths = ([16, 32, 64] * ((seq_count + 2) // 3))[:seq_count]
    shape = (1, sum(lengths), 6, 128)
    generator = torch.Generator().manual_seed(819 + seq_count)
    q = torch.zeros(shape, dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = (torch.randn(shape, generator=generator) * 0.1).to(torch.bfloat16)
    g = torch.zeros(shape, dtype=torch.float32)
    beta = torch.full(shape[:-1], 0.5, dtype=torch.float32)
    outputs = _run(q, k, v, g, beta, 64, lengths)
    # K=0 makes Akk=I and W=0: both U and V_new must equal beta*V.
    # Large padded batches make the old W scratch cross into live U seeds.
    expected = (v.float() * 0.5).to(torch.bfloat16).permute(0, 2, 1, 3)
    for index in (6, 9):
        torch.testing.assert_close(outputs[index], expected, rtol=0, atol=0)
    for index in (0, 1):
        torch.testing.assert_close(
            outputs[index], torch.zeros_like(outputs[index]), rtol=0, atol=0
        )


@pytest.mark.parametrize("key_dim", [32, 128, 256])
@pytest.mark.parametrize("chunk_size,tokens", [(64, 1), (64, 63), (128, 127)])
def test_aqk_diagonal_matches_ungated_fp32_dot(chunk_size, tokens, key_dim):
    generator = torch.Generator().manual_seed(819 + tokens + key_dim)
    # Dyadic inputs make sum(q*k) exactly representable in FP32. Q=K avoids
    # cancellation; nonzero gates expose separately rounded BF16 factors.
    q = (
        torch.randint(-8, 9, (1, tokens, 2, key_dim), generator=generator).float()
        / 32
    ).to(torch.bfloat16)
    k = q.clone()
    v = torch.zeros((1, tokens, 4, 128), dtype=torch.bfloat16)
    g = -0.02 * torch.rand((1, tokens, 4, key_dim), generator=generator)
    beta = torch.zeros((1, tokens, 4), dtype=torch.float32)
    outputs = _run(q, k, v, g, beta, chunk_size)
    expected = (q.float() * k.float()).sum(-1).repeat_interleave(2, dim=2)
    actual = outputs[3].diagonal(dim1=-2, dim2=-1)
    torch.testing.assert_close(
        actual, expected.permute(0, 2, 1).to(torch.bfloat16), rtol=0, atol=0
    )
