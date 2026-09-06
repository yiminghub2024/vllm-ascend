# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the GLM-5.3-Flash kpool indexer's scoring and selection path."""

import sys
import types

import pytest
import torch

from vllm_ascend.models.glm5next.ops.kpool_indexer import (
    LIGHTNING_INDEXER_POOLS_ALIGNMENT,
    pa_bsnd_keys,
    row_seq_lens,
    select_token_ids,
    tail_width_for,
)

HEAD_DIM = 128
NUM_HEADS = 16
POOL_SIZE = 4
POOLS_PER_BLOCK = 32
TOPK_TOKENS = 32


@pytest.fixture
def fake_operator(monkeypatch):
    """Stand in for ``npu_lightning_indexer`` with the CPU equivalent.

    The operator itself is verified against a reference by
    ``tools/probe_ascend_indexer_ops.py`` on hardware. What cannot be checked
    there is the composition around it -- the key length it is handed, and the
    expansion of what it returns -- which is what these tests cover.
    """
    calls: list[dict] = []

    def npu_lightning_indexer(query, key, weights, **kwargs):
        calls.append(kwargs)
        assert kwargs["layout_query"] == "TND"
        assert kwargs["layout_key"] == "PA_BSND"
        # A mask would remove pools the shared boundary already made legal.
        assert kwargs["sparse_mode"] == 0

        num_tokens = query.shape[0]
        budget = kwargs["sparse_count"]
        selectable = kwargs["actual_seq_lengths_key"]
        cumulative = kwargs["actual_seq_lengths_query"]
        block_table = kwargs["block_table"]
        pools_per_block = key.shape[1]

        out = torch.full((num_tokens, 1, budget), -1, dtype=torch.int32)
        start = 0
        for request in range(cumulative.shape[0]):
            end = int(cumulative[request])
            offered = int(selectable[request])
            gathered = torch.cat(
                [key[int(block)].reshape(pools_per_block, HEAD_DIM) for block in block_table[request]]
            )[:offered]
            scores = (
                torch.relu(torch.matmul(query[start:end].float(), gathered.t().float()))
                * weights[start:end].float().unsqueeze(-1)
            ).sum(dim=1)
            order = torch.argsort(scores, dim=-1, descending=True, stable=True)
            keep = min(budget, offered)
            out[start:end, 0, :keep] = order[:, :keep].to(torch.int32)
            start = end
        return out, None

    module = types.ModuleType("torch_npu")
    module.npu_lightning_indexer = npu_lightning_indexer
    monkeypatch.setitem(sys.modules, "torch_npu", module)
    return calls


def _cache(num_blocks: int) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(num_blocks, POOLS_PER_BLOCK, 1, HEAD_DIM, dtype=torch.bfloat16)


def _run(query_lens: list[int], seq_lens: list[int], fake_operator) -> torch.Tensor:
    torch.manual_seed(1)
    num_tokens = sum(query_lens)
    num_requests = len(query_lens)
    blocks_per_request = 2
    return select_token_ids(
        torch.randn(num_tokens, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16),
        torch.randn(num_tokens, NUM_HEADS, dtype=torch.bfloat16),
        _cache(num_requests * blocks_per_request),
        block_table=torch.arange(num_requests * blocks_per_request, dtype=torch.int32).reshape(
            num_requests, blocks_per_request
        ),
        query_lens=torch.tensor(query_lens, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        pool_size=POOL_SIZE,
        topk_tokens=TOPK_TOKENS,
        max_query_len=max(query_lens),
    )


def test_selection_is_offered_the_prefix_every_row_may_see(fake_operator):
    """The key length handed over is the earliest row's complete pool count."""
    _run(query_lens=[1, 4], seq_lens=[21, 33], fake_operator=fake_operator)

    (kwargs,) = fake_operator
    # Request 0 is a plain decode row at seq_len 21 -> 5 complete pools.
    # Request 1's earliest row is at seq_len 30 -> 7, not the 8 its last row has.
    torch.testing.assert_close(kwargs["actual_seq_lengths_key"], torch.tensor([5, 7], dtype=torch.int32))
    torch.testing.assert_close(kwargs["actual_seq_lengths_query"], torch.tensor([1, 5], dtype=torch.int32))
    assert kwargs["sparse_count"] == TOPK_TOKENS // POOL_SIZE


@pytest.mark.parametrize("query_lens", [[1, 1, 1], [4], [1, 4], [3, 2, 5]])
def test_selected_ids_are_causal_and_cover_the_recent_run(query_lens, fake_operator):
    """Nothing may name a future token, and nothing recent may go missing."""
    seq_lens = [200 + 7 * i for i in range(len(query_lens))]
    selected = _run(query_lens, seq_lens, fake_operator)

    per_row = row_seq_lens(
        torch.tensor(seq_lens, dtype=torch.int32),
        torch.tensor(query_lens, dtype=torch.int32),
    )
    assert selected.shape == (
        sum(query_lens),
        TOPK_TOKENS + tail_width_for(POOL_SIZE, max(query_lens)),
    )
    assert torch.all(selected < per_row.unsqueeze(-1))

    # Every token from the request's shared boundary onwards must be named:
    # those pools were never offered to the operator, so if the expansion does
    # not list them the model simply cannot see its newest tokens.
    row = 0
    for request, query_len in enumerate(query_lens):
        boundary = ((seq_lens[request] - query_len + 1) // POOL_SIZE) * POOL_SIZE
        for _ in range(query_len):
            covered = {int(v) for v in selected[row] if v >= 0}
            assert set(range(boundary, int(per_row[row]))) <= covered
            row += 1


def test_row_sequence_lengths_count_back_from_the_step():
    per_row = row_seq_lens(torch.tensor([21, 33]), torch.tensor([1, 4]))
    torch.testing.assert_close(per_row, torch.tensor([21, 30, 31, 32, 33]))


def test_pa_bsnd_keeps_the_pool_axis():
    keys = torch.zeros(2, 160, 1, HEAD_DIM, dtype=torch.bfloat16)
    viewed = pa_bsnd_keys(keys, HEAD_DIM)
    assert viewed.shape == (2, 160, 1, HEAD_DIM)

    squeezed = keys.squeeze(2)
    assert squeezed.shape == (2, 160, HEAD_DIM)
    assert pa_bsnd_keys(squeezed, HEAD_DIM).shape == (2, 160, 1, HEAD_DIM)


def test_pa_bsnd_does_not_invent_pools_from_an_fp8_wide_page():
    """160 cells of width 132, viewed as 128, is 165 -- not a multiple of 16."""
    packed = torch.zeros(2, 160, 1, 132, dtype=torch.bfloat16)
    assert packed.numel() // (2 * HEAD_DIM) == 165
    assert 165 % LIGHTNING_INDEXER_POOLS_ALIGNMENT != 0

    with pytest.raises(RuntimeError, match="content width is 132"):
        pa_bsnd_keys(packed, HEAD_DIM)

    # The shape that actually reached npu_lightning_indexer before this guard.
    invented = packed.view(packed.shape[0], -1, 1, HEAD_DIM)
    assert invented.shape == (2, 165, 1, HEAD_DIM)
    with pytest.raises(RuntimeError, match="got 165"):
        pa_bsnd_keys(invented, HEAD_DIM)


def test_indexer_cache_is_constructed_at_the_logical_head_dim():
    """Do not pass the FP8-plus-scale width (head_dim + 4) into the cache."""
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parents[3] / "vllm_ascend" / "models" / "glm5next" / "attention.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    widened = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name != "Glm5NextIndexerCache":
            continue
        for keyword in node.keywords:
            if keyword.arg != "head_dim":
                continue
            widened = isinstance(keyword.value, ast.BinOp)
    assert not widened, (
        "Glm5NextIndexerCache must be constructed with the logical head_dim; "
        "widening it by the FP8 scale bytes made npu_lightning_indexer see "
        "165 pools on a 160-pool page."
    )
