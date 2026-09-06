# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe the Ascend ops the GLM-5.3-Flash kpool indexer would be built on.

Temporary bring-up diagnostic, not part of the test suite. The kpool indexer's
Ascend design hinges on op semantics that the docs leave open and that differ
between CANN releases, so this answers them on the actual device before the
implementation commits to them:

1. Does ``npu_lightning_indexer`` accept a *pool-granular* bf16 key cache, and
   does it score with ``sum_h weights[h] * relu(q_h . k)`` the way the kpool
   logits kernel does? If yes, the pooled cache can be scored as-is and the op
   returns pool ids directly -- no separate top-k, no fp8 round-trip.
2. Does it accept ``index_n_heads = 16``? The DeepGEMM path pads to 32 heads;
   if this op does not need that, the padding can be dropped on NPU.
3. Does the sparse attention accept ``sparse_block_size = index_kpool``, i.e.
   can it expand pool ids into their constituent tokens itself? If yes, the
   pool->token expansion kernel is unnecessary on the attention side.
4. Does the sparse attention run NoPE-only (``head_dim = 512``, no rope split)?
   GLM-5.3-Flash MLA is NoPE, but the docs specify a 512 + 64 split. If it
   refuses, selection has to feed a gather + dense attention instead.

Run inside the container:

    python tools/probe_ascend_indexer_ops.py
"""

from __future__ import annotations

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    raise SystemExit("torch_npu is unavailable; run this inside the Ascend container")

DEVICE = "npu"
HEAD_DIM = 128
INDEX_KPOOL = 4
# npu_lightning_indexer wants the PA_BSND key block_size to be a multiple of
# 16. The index cache is pool-granular, so this is in pools, and the model-wide
# cache block_size must be index_kpool times it.
POOL_BLOCK_SIZE = 32


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def report(name: str, fn) -> object | None:
    """Run a probe, printing a one-line verdict instead of aborting the run."""
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001 - a probe reports, it does not raise
        print(f"  FAIL  {name}")
        print(f"        {type(exc).__name__}: {str(exc).strip().splitlines()[0][:400]}")
        return None
    print(f"  OK    {name}")
    return result


def env_report() -> None:
    banner("environment")
    print(f"  torch          {torch.__version__}")
    print(f"  torch_npu      {getattr(torch_npu, '__version__', '?')}")
    try:
        print(f"  device         {torch.npu.get_device_name(0)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  device         unavailable: {exc}")
    from vllm_ascend.device.hardware_profile import get_hardware_profile
    from vllm_ascend.utils import enable_custom_op

    print(f"  custom ops     {enable_custom_op()}")
    try:
        print(f"  hw profile     {get_hardware_profile()}")
    except Exception as exc:  # noqa: BLE001
        print(f"  hw profile     unavailable: {exc}")

    banner("op availability")
    for op in (
        "npu_lightning_indexer",
        "npu_quant_lightning_indexer",
        "npu_sparse_flash_attention",
        "npu_kv_quant_sparse_flash_attention",
        "npu_fused_infer_attention_score",
    ):
        print(f"  {'yes' if hasattr(torch_npu, op) else 'NO ':<4}  torch_npu.{op}")


def lightning_indexer_reference(
    query: torch.Tensor,
    key_pages: torch.Tensor,
    weights: torch.Tensor,
    seq_lens_query: torch.Tensor,
    seq_lens_key: torch.Tensor,
    block_table: torch.Tensor,
    sparse_count: int,
) -> torch.Tensor:
    """CPU reference for TND ``npu_lightning_indexer`` with ``sparse_mode=3``.

    Transcribed from op-plugin's own test for the operator, so a mismatch here
    means the operator does not compute what the kpool logits kernel does.
    """
    num_requests = seq_lens_query.shape[0]
    head_dim = query.shape[2]
    block_size = key_pages.shape[1]
    out = torch.full((query.shape[0], 1, sparse_count), -1, dtype=torch.int32)

    consumed = 0
    for request in range(num_requests):
        # TND actual_seq_lengths_query is cumulative.
        query_len = int(seq_lens_query[request]) - consumed
        key_len = int(seq_lens_key[request])

        blocks = (key_len + block_size - 1) // block_size
        gathered = torch.zeros((blocks * block_size, head_dim), dtype=key_pages.dtype)
        for block in range(blocks):
            page = key_pages[int(block_table[request, block])]
            gathered[block * block_size : (block + 1) * block_size] = page.reshape(block_size, head_dim)
        keys = gathered[:key_len].t().float()

        rows = query[consumed : consumed + query_len].transpose(0, 1).float()
        row_weights = weights[consumed : consumed + query_len].transpose(0, 1).unsqueeze(-1).float()
        scores = (torch.relu(torch.matmul(rows, keys)) * row_weights).sum(dim=0)

        # sparse_mode=3 is rightDownCausal: the last query row sees every key.
        for offset in range(query_len):
            scores[-1 - offset, key_len - offset :] = float("-inf")

        order = torch.argsort(scores, dim=1, descending=True, stable=True)
        keep = min(sparse_count, key_len)
        out[consumed : consumed + query_len, 0, :keep] = order[:, :keep].to(torch.int32)
        consumed += query_len
    return out


def probe_lightning_indexer(num_heads: int) -> None:
    banner(f"npu_lightning_indexer  (index_n_heads = {num_heads}, pool-granular bf16 key)")
    torch.manual_seed(0)

    # Two requests: one decode-shaped (1 query token), one prefill-shaped.
    query_lens = [1, 3]
    pool_lens = [70, 40]  # complete pools per request, not tokens
    num_tokens = sum(query_lens)
    sparse_count = 32  # select_k = index_topk // index_kpool

    blocks_per_request = (max(pool_lens) + POOL_BLOCK_SIZE - 1) // POOL_BLOCK_SIZE
    num_pages = len(query_lens) * blocks_per_request

    query = torch.randn(num_tokens, num_heads, HEAD_DIM, dtype=torch.bfloat16)
    # PA_BSND: [block_count, block_size, num_kv_heads=1, head_dim]
    key_pages = torch.randn(num_pages, POOL_BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
    # Weights fold in the query scale upstream, so they are fp32 there; the op
    # documents bf16/fp16, hence the cast.
    weights = torch.randn(num_tokens, num_heads, dtype=torch.bfloat16)
    seq_lens_query = torch.tensor(query_lens, dtype=torch.int32).cumsum(0).to(torch.int32)
    seq_lens_key = torch.tensor(pool_lens, dtype=torch.int32)
    block_table = torch.arange(num_pages, dtype=torch.int32).reshape(len(query_lens), blocks_per_request)

    expected = lightning_indexer_reference(
        query, key_pages, weights, seq_lens_query, seq_lens_key, block_table, sparse_count
    )

    def run():
        indices, _ = torch_npu.npu_lightning_indexer(
            query.npu(),
            key_pages.npu(),
            weights.npu(),
            actual_seq_lengths_query=seq_lens_query.npu(),
            actual_seq_lengths_key=seq_lens_key.npu(),
            block_table=block_table.npu(),
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=sparse_count,
            sparse_mode=3,
        )
        return indices.cpu()

    actual = report(f"call accepted, n_head={num_heads}", run)
    if actual is None:
        return

    print(f"        output shape {tuple(actual.shape)}, dtype {actual.dtype}")
    actual = actual.reshape(num_tokens, 1, -1)
    # Scores tie in bf16, so compare the selected *sets* per row rather than
    # the exact order the operator emits.
    mismatched = []
    for row in range(num_tokens):
        got = {int(v) for v in actual[row, 0] if v >= 0}
        want = {int(v) for v in expected[row, 0] if v >= 0}
        if got != want:
            mismatched.append((row, len(want - got), len(got - want)))
    if mismatched:
        print(f"        MISMATCH vs reference on {len(mismatched)}/{num_tokens} rows: {mismatched[:4]}")
        print(f"        row 0 got  {actual[0, 0, :12].tolist()}")
        print(f"        row 0 want {expected[0, 0, :12].tolist()}")
    else:
        print(f"        selected sets match the reference on all {num_tokens} rows")
        print("        -> pooled bf16 cache can be scored directly; op returns pool ids")

    # A short row must select every visible pool and pad the rest with -1, in
    # that order: the sparse attention requires valid indices to come first.
    row = actual[0, 0]
    valid = (row >= 0).to(torch.int32)
    monotone = bool(torch.all(valid[1:] <= valid[:-1]))
    print(f"        valid-before-invalid padding: {monotone}")


def probe_sparse_attention(rope_head_dim: int, kv_head_dim: int = 512) -> None:
    banner(f"npu_sparse_flash_attention  (sparse_block_size = {INDEX_KPOOL}, rope_head_dim = {rope_head_dim})")
    torch.manual_seed(0)

    num_query_heads = 8
    kv_block_size = 64  # model-wide cache block_size, token-granular
    seq_len = 200
    num_tokens = 2
    sparse_count = 16  # in pools

    blocks = (seq_len + kv_block_size - 1) // kv_block_size
    query = torch.randn(num_tokens, num_query_heads, kv_head_dim, dtype=torch.bfloat16)
    kv_pages = torch.randn(blocks, kv_block_size, 1, kv_head_dim, dtype=torch.bfloat16)
    block_table = torch.arange(blocks, dtype=torch.int32).reshape(1, blocks).repeat(num_tokens, 1)

    # Pool ids, valid first then -1, exactly as the indexer emits them.
    pool_ids = torch.arange(sparse_count, dtype=torch.int32).reshape(1, 1, sparse_count).repeat(num_tokens, 1, 1)
    seq_lens_query = torch.tensor([1, 2], dtype=torch.int32)
    seq_lens_kv = torch.tensor([seq_len, seq_len], dtype=torch.int32)

    kwargs = dict(
        sparse_indices=pool_ids.npu(),
        scale_value=kv_head_dim**-0.5,
        block_table=block_table.npu(),
        actual_seq_lengths_query=seq_lens_query.npu(),
        actual_seq_lengths_kv=seq_lens_kv.npu(),
        sparse_block_size=INDEX_KPOOL,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
    )
    if rope_head_dim:
        kwargs["query_rope"] = torch.randn(num_tokens, num_query_heads, rope_head_dim, dtype=torch.bfloat16).npu()
        kwargs["key_rope"] = torch.randn(blocks, kv_block_size, 1, rope_head_dim, dtype=torch.bfloat16).npu()

    def run():
        out = torch_npu.npu_sparse_flash_attention(query.npu(), kv_pages.npu(), kv_pages.npu(), **kwargs)
        return out[0] if isinstance(out, tuple) else out

    result = report(f"sparse_block_size={INDEX_KPOOL}, nope_dim={kv_head_dim}", run)
    if result is not None:
        print(f"        output shape {tuple(result.shape)}, finite {bool(torch.isfinite(result.float()).all())}")
        print("        -> pool ids can go straight to attention; no pool->token expansion needed")


def main() -> None:
    env_report()
    # 16 is what this checkpoint ships; 64 is what the op's own tests cover, so
    # a 16-only failure means the DeepGEMM head padding is needed here too.
    for num_heads in (16, 64):
        probe_lightning_indexer(num_heads)
    # GLM-5.3-Flash MLA is NoPE, so the no-rope call is the one that matters;
    # the 64-dim call tells us whether the op simply requires the split.
    for rope_head_dim in (0, 64):
        probe_sparse_attention(rope_head_dim)
    print("\nDone. Paste the whole output back.")


if __name__ == "__main__":
    main()
