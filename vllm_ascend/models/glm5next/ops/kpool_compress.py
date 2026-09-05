# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend replacements for the GLM-5.3-Flash kpool indexer's fused kernels.

The upstream implementation fuses the Hadamard-128 rotation with the block-128
ue8m0 FP8 quantization into a single Triton kernel that keeps the rotated tensor
in registers. That kernel is CUDA-only, so the NPU path expresses the same
numerics in torch: the rotation becomes a matmul against a cached normalized
Hadamard matrix, and the quantization stays elementwise. Every step is
device-side, so the sequence remains ACL-graph safe.

The same applies to the two pieces around it. Compressing a pool of tokens into
one cache entry and expanding selected pools back into token ids are both fused
Triton kernels upstream; here they are torch expressions chosen to reproduce the
kernels' arithmetic step for step, including where the kernels round through
bfloat16. Neither touches the KV cache or attention metadata, so both can be
compared against a reference on CPU.
"""

import torch

# The indexer query is quantized against the e4m3 maximum, and the resulting
# scale is restricted to a power of two (ue8m0), matching the cached K basis.
FP8_E4M3_MAX = 448.0

# Guards rows whose rotated vector is all but zero, so log2 stays finite.
_MIN_ABSMAX = 1e-4

_HADAMARD_CACHE: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def _normalized_hadamard(dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return a cached ``dim x dim`` Hadamard matrix scaled by ``dim ** -0.5``.

    Folding the normalization into the matrix keeps the rotation a single
    matmul. The matrix is symmetric, so it is used without a transpose.
    """
    key = (dim, device, dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        from scipy.linalg import hadamard  # type: ignore[import-untyped]
    except ImportError as err:
        raise ImportError(
            "The GLM-5.3-Flash kpool indexer requires SciPy for the Hadamard transform. Please install scipy."
        ) from err

    matrix = torch.tensor(hadamard(dim, dtype=float), dtype=dtype, device=device) * (dim**-0.5)
    _HADAMARD_CACHE[key] = matrix
    return matrix


def fwht128_quant_fp8(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate each 128-wide row by the Hadamard-128 transform, then FP8-quant.

    Args:
        q: ``[rows, 128]`` bf16 -- one head vector per row.

    Returns:
        (q_fp8 ``[rows, 128]`` float8_e4m3fn, scale ``[rows, 1]`` float32).
    """
    assert q.ndim == 2 and q.shape[1] == 128, q.shape

    rows, dim = q.shape
    if rows == 0:
        return (
            torch.empty((0, dim), dtype=torch.float8_e4m3fn, device=q.device),
            torch.empty((0, 1), dtype=torch.float32, device=q.device),
        )

    hadamard = _normalized_hadamard(dim, q.device, torch.float32)
    rotated = q.float() @ hadamard
    # The upstream kernel materializes bf16 between the rotation and the quant,
    # so the fp8 operand carries the same rounding on both backends.
    rotated = rotated.to(torch.bfloat16).to(torch.float32)

    absmax = rotated.abs().amax(dim=-1, keepdim=True).clamp_min(_MIN_ABSMAX)
    scale = torch.exp2(torch.ceil(torch.log2(absmax / FP8_E4M3_MAX)))
    q_fp8 = (rotated / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return q_fp8, scale


def _softmax_pool_slots(
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
) -> torch.Tensor:
    """Collapse each pool's ``pool_size`` token vectors down to a single vector.

    The mixing weights are a softmax over the pool's slots taken *per
    dimension*: dimension ``d`` of the result mixes the slots by
    ``softmax_s(slot_score[s, d] + ape[s, d])``. Every dimension gets its own
    mixture over the pooled tokens, which is what lets one cached key stand in
    for a whole run of them. ``ape`` is the learned per-slot position bias, so a
    slot's weight depends on where it sits inside the pool as well as on its
    gate score.

    Returns a bfloat16 ``[n_pools, head_dim]`` tensor: the fused kernel rounds
    to bfloat16 here before rotating, and the FP8 operand only matches across
    backends if that rounding happens on both.
    """
    score = slot_score.float() + ape.float()
    # Subtracting the per-dimension max keeps the exponent finite; it cancels
    # against the denominator, so the weights themselves are unchanged.
    prob = torch.exp(score - score.amax(dim=1, keepdim=True))
    pooled = (slot_k.float() * prob).sum(dim=1) / prob.sum(dim=1)
    return pooled.to(torch.bfloat16)


def kpool_compress_k(
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress whole pools of indexer keys into one FP8 vector each.

    Args:
        slot_k: ``[n_pools, pool_size, 128]`` bf16 -- raw per-token indexer K.
        slot_score: ``[n_pools, pool_size, 128]`` -- per-token gate score.
        ape: ``[pool_size, 128]`` -- per-slot position bias.

    Returns:
        (k_fp8 ``[n_pools, 128]`` float8_e4m3fn, scale ``[n_pools, 1]`` float32),
        ready to be written to the pool-granular index-K cache.
    """
    assert slot_k.ndim == 3, slot_k.shape
    assert slot_score.shape == slot_k.shape, (slot_score.shape, slot_k.shape)
    assert ape.shape == slot_k.shape[1:], (ape.shape, slot_k.shape)

    return fwht128_quant_fp8(_softmax_pool_slots(slot_k, slot_score, ape))


def expand_pools_and_append_tail(
    pool_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Turn selected pool ids into the token ids the sparse attention reads.

    Top-k runs at pool granularity, so it returns ``topk_tokens // pool_size``
    pool ids per query; each stands for the ``pool_size`` consecutive tokens it
    was compressed from. Those get expanded back out, and then the request's
    trailing pool is appended: it is still incomplete, so it was never
    compressed into the index-K cache and cannot be selected -- but the newest
    tokens must always be attended to, so they are appended unconditionally.

    Args:
        pool_ids: ``[rows, topk_tokens // pool_size]`` -- selected pools per
            query row, negative where top-k found fewer pools than the budget.
        seq_lens: ``[rows]`` -- token-granular sequence length per query row.
        pool_size: tokens per pool (the checkpoint's ``index_kpool``).

    Returns:
        ``[rows, topk_tokens + pool_size - 1]`` int32 token ids, relative to the
        start of each request and padded with ``-1``. The tail can never be
        longer than ``pool_size - 1`` tokens, which fixes the output width.
    """
    assert pool_ids.ndim == 2, pool_ids.shape
    assert seq_lens.ndim == 1, seq_lens.shape
    assert seq_lens.shape[0] == pool_ids.shape[0], (seq_lens.shape, pool_ids.shape)

    rows, num_groups = pool_ids.shape
    device = pool_ids.device

    pool_ids = pool_ids.to(torch.int64).unsqueeze(-1)
    slot_offsets = torch.arange(pool_size, device=device)
    history = pool_ids * pool_size + slot_offsets
    history = torch.where(pool_ids >= 0, history, -1).reshape(rows, num_groups * pool_size)

    if pool_size == 1:
        # Every token is its own pool, so there is no tail to append.
        return history.to(torch.int32)

    seq_lens = seq_lens.to(torch.int64).unsqueeze(-1)
    tail_start = (seq_lens // pool_size) * pool_size
    tail_offsets = torch.arange(pool_size - 1, device=device)
    tail = torch.where(tail_offsets < seq_lens - tail_start, tail_start + tail_offsets, -1)

    return torch.cat((history, tail), dim=-1).to(torch.int32)
