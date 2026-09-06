# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pool scoring and selection for the GLM-5.3-Flash kpool indexer on Ascend.

``npu_lightning_indexer`` scores a paged key cache and returns the ids of the
highest-scoring entries. Because the GLM index cache holds one entry per pool of
``index_kpool`` tokens, those ids are pool ids, and the sparse attention needs
token ids -- so selection is scoring plus an expansion.

Two properties of the operator shape everything here, both established by
probing an Ascend 950:

* It takes one key length per *request*, and its ``sparse_mode=3`` mask moves
  the boundary by one key per query row. With pooled keys a key is
  ``index_kpool`` tokens, so that boundary advances four times too fast. Asking
  for a key length per query row is rejected outright (error 561002).
* ``sparse_mode=0`` applies no mask at all and matches an unmasked reference.

So selection runs unmasked over the pool prefix that is causally valid for
*every* query row of the request, and the expansion names the remaining recent
tokens explicitly. That is exact -- nothing is dropped and no future token is
named -- and it covers a draft-verify batch, whose rows sit at different
positions, for the same reason it covers a prefill chunk.
"""

import torch

from vllm_ascend.models.glm5next.ops.kpool_compress import (
    expand_pools_and_append_tail,
    shared_pool_prefix,
)


def score_and_select_pools(
    query: torch.Tensor,
    weights: torch.Tensor,
    index_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cumulative_query_lens: torch.Tensor,
    selectable_pools: torch.Tensor,
    select_pools: int,
) -> torch.Tensor:
    """Pick the highest-scoring pools for each query row.

    Args:
        query: ``[num_tokens, index_n_heads, head_dim]`` bf16, TND layout.
        weights: ``[num_tokens, index_n_heads]`` -- the per-head gate the
            scores are summed under.
        index_cache: ``[num_blocks, pools_per_block, 1, head_dim]`` bf16,
            PA_BSND layout.
        block_table: ``[num_requests, max_blocks]``.
        cumulative_query_lens: ``[num_requests]`` -- TND wants the query
            lengths cumulated.
        selectable_pools: ``[num_requests]`` -- pools offered to each request,
            from :func:`shared_pool_prefix`.
        select_pools: budget in pools, i.e. ``index_topk // index_kpool``.

    Returns:
        ``[num_tokens, 1, select_pools]`` int32 pool ids, padded with -1.
    """
    import torch_npu

    pool_ids, _ = torch_npu.npu_lightning_indexer(
        query=query,
        key=index_cache,
        weights=weights,
        actual_seq_lengths_query=cumulative_query_lens,
        actual_seq_lengths_key=selectable_pools,
        block_table=block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=select_pools,
        # Not sparse_mode=3: see the module docstring. The prefix handed over
        # in `selectable_pools` is already causally valid for every query row,
        # so a mask would only ever remove pools the request may legitimately
        # see.
        sparse_mode=0,
    )
    return pool_ids


def select_token_ids(
    query: torch.Tensor,
    weights: torch.Tensor,
    index_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    query_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_size: int,
    topk_tokens: int,
    max_query_len: int,
) -> torch.Tensor:
    """Score pools and expand the winners into token ids.

    Args:
        query: ``[num_tokens, index_n_heads, head_dim]`` bf16.
        weights: ``[num_tokens, index_n_heads]``.
        index_cache: the pooled bf16 cache in PA_BSND layout.
        block_table: ``[num_requests, max_blocks]``.
        query_lens: ``[num_requests]`` -- query rows per request this step.
        seq_lens: ``[num_requests]`` -- tokens known after this step.
        pool_size: the checkpoint's ``index_kpool``.
        topk_tokens: the checkpoint's ``index_topk``.
        max_query_len: the batch's widest request, taken from the attention
            metadata rather than from ``query_lens`` -- reading a device tensor
            to size the output would abort ACL-graph capture.

    Returns:
        ``[num_tokens, topk_tokens + tail_width]`` int32 token ids relative to
        each request's start, padded with -1.
    """
    assert topk_tokens % pool_size == 0, (topk_tokens, pool_size)

    # The MLA metadata builder keeps these two on the host, but the operator
    # rejects length arguments that do not sit with its other tensors. Both are
    # one int per request, so the copy is small and asynchronous -- unlike a
    # read in the other direction, which would stall the step.
    query_lens = query_lens.to(query.device)
    seq_lens = seq_lens.to(query.device)

    selectable = shared_pool_prefix(seq_lens, query_lens, pool_size)
    pool_ids = score_and_select_pools(
        query,
        weights,
        index_cache,
        block_table=block_table,
        cumulative_query_lens=query_lens.cumsum(0).to(torch.int32),
        selectable_pools=selectable,
        select_pools=topk_tokens // pool_size,
    )

    # The operator answers per query row, so the request-granular boundary and
    # sequence length both have to be spread out to one entry per row. The width
    # is passed in because deriving it from a device tensor is a host read.
    rows = torch.repeat_interleave(query_lens.to(torch.int64), output_size=query.shape[0])
    return expand_pools_and_append_tail(
        pool_ids.reshape(query.shape[0], -1),
        row_seq_lens(seq_lens, query_lens, rows=rows),
        pool_size,
        selectable_pools=selectable[rows],
        tail_width=tail_width_for(pool_size, max_query_len),
    )


def row_seq_lens(
    seq_lens: torch.Tensor,
    query_lens: torch.Tensor,
    *,
    rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sequence length as of each query row, not as of each request.

    ``seq_lens`` counts the tokens a request knows once the whole step has been
    applied, so a row ``i`` places before the end of its request knows ``i``
    fewer of them.

    Args:
        seq_lens: ``[num_requests]``.
        query_lens: ``[num_requests]``.
        rows: ``[num_tokens]`` request index per query row, if already built.
    """
    query_lens = query_lens.to(torch.int64)
    if rows is None:
        rows = torch.repeat_interleave(query_lens)
    starts = torch.cumsum(query_lens, 0) - query_lens
    within = torch.arange(rows.shape[0], device=seq_lens.device) - starts[rows]
    return (seq_lens[rows] - query_lens[rows] + 1 + within).to(seq_lens.dtype)


def tail_width_for(pool_size: int, max_query_len: int) -> int:
    """Output width the tail expansion needs for a shared selection boundary.

    The boundary is at most ``pool_size - 1`` tokens behind the earliest row of
    the request, and the last row is ``max_query_len - 1`` further along, so
    that many token ids can fall outside the selectable prefix.
    """
    return pool_size - 1 + max_query_len - 1
