# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm the finite tile variants used by the V4.1 indexer and ring compressor."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from vllm.triton_utils import HAS_TRITON, triton

from vllm_ascend.ops.triton.compressor.compressor_triton import (
    _cube_core_num,
    compressor_from_projected,
)
from vllm_ascend.ops.triton.prepare_indexer_indices import prepare_indexer_indices
from vllm_ascend.ops.triton.quantize_indexer_query import quantize_indexer_query
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

if TYPE_CHECKING:
    from vllm_ascend.worker.worker import NPUWorker

_V41_MODEL_TYPES = (
    "deepseek_v4.1",
    "deepseek_v41",
    "deepseek_v4.1_text",
    "deepseek_v41_text",
)


def collect_indexer_warmup_token_counts(topk: int, num_cores: int, max_tokens: int) -> list[int]:
    """One token count per reachable ``BLOCK_ROWS`` in index postprocessing."""
    # Match the 128 KiB, eight-buffer sort budget in prepare_indexer_indices.
    max_block_rows = 128 * 1024 // (triton.next_power_of_2(topk) * 4 * 8)
    token_counts = [1]
    block_rows = 1
    while block_rows < max_block_rows:
        tokens = block_rows * num_cores + 1
        if tokens > max_tokens:
            break
        token_counts.append(tokens)
        block_rows *= 2
    return token_counts


def collect_compressor_warmup_token_counts(
    max_tokens: int,
    capture_sizes: Iterable[int] = (),
) -> list[int]:
    """Decode-graph and ring-sized query lengths for the C2 Triton kernels.

    ``HEAD_DIM`` is a constexpr, so one width is enough. ``max_query_len``
    changes the pool grid and the 32-row ring write count; FULL_DECODE_ONLY
    capture sizes must be compiled before ``capture_model``.
    """
    counts = [1]
    length = 2
    while length <= min(max_tokens, 32):
        counts.append(length)
        length *= 2
    if max_tokens >= 128:
        counts.append(min(128, max_tokens))
    extra = [size for size in capture_sizes if 1 <= size <= max_tokens]
    return sorted(set(counts + extra))


def _is_v41_config(config) -> bool:
    return getattr(config, "model_type", None) in _V41_MODEL_TYPES


def _capture_sizes(worker: NPUWorker) -> tuple[int, ...]:
    compilation = getattr(getattr(worker, "vllm_config", None), "compilation_config", None)
    sizes = getattr(compilation, "cudagraph_capture_sizes", None) or ()
    captured: list[int] = []
    for size in sizes:
        try:
            count = int(size)
        except (TypeError, ValueError):
            continue
        if count >= 1:
            captured.append(count)
    return tuple(captured)


@torch.inference_mode()
def deepseek_v41_indexer_warmup(worker: NPUWorker) -> None:
    """Precompile indexer tiles before serving arbitrary eager token counts."""
    config = worker.model_config.hf_text_config
    ratios = sorted(set(config.compress_ratios[: config.num_hidden_layers]) - {0})
    if not ratios:
        return

    device = worker.device
    query = torch.zeros(1, config.index_n_heads, config.index_head_dim, dtype=worker.model_config.dtype, device=device)
    quantize_indexer_query(query)
    token_counts = collect_indexer_warmup_token_counts(
        config.index_topk, get_vectorcore_num(), worker.scheduler_config.max_num_batched_tokens
    )
    for tokens in token_counts:
        selected = torch.zeros(tokens, config.index_topk, dtype=torch.int32, device=device)
        positions = torch.zeros(tokens, dtype=torch.int64, device=device)
        for ratio in ratios:
            prepare_indexer_indices(selected, positions, ratio)


@torch.inference_mode()
def deepseek_v41_compressor_warmup(worker: NPUWorker) -> None:
    """Precompile the ratio-2 ring kernels before FULL_DECODE_ONLY capture."""
    config = worker.model_config.hf_text_config
    if 2 not in set(config.compress_ratios[: config.num_hidden_layers]):
        return
    width = int(getattr(config, "head_dim", 0))
    if width < 1 or width & (width - 1):
        return

    device = worker.device
    max_tokens = int(worker.scheduler_config.max_num_batched_tokens)
    token_counts = collect_compressor_warmup_token_counts(max_tokens, _capture_sizes(worker))
    num_cores = _cube_core_num()
    for tokens in token_counts:
        kv = torch.zeros(tokens, width, dtype=torch.float32, device=device)
        scores = torch.zeros_like(kv)
        state = torch.zeros(1, 32, 2 * width, dtype=torch.float32, device=device)
        metadata = torch.zeros(5, 1, dtype=torch.int32, device=device)
        metadata[1, 0] = tokens
        metadata[4, 0] = 1
        out = torch.zeros(tokens, width, dtype=torch.bfloat16, device=device)
        compressor_from_projected(
            kv,
            scores,
            state,
            metadata,
            out,
            max_query_len=tokens,
            num_cores=num_cores,
        )


@torch.inference_mode()
def deepseek_v41_triton_warmup(worker: NPUWorker) -> None:
    """Precompile V4.1 Triton tiles before ACL graph capture."""
    if not HAS_TRITON:
        return
    if not _is_v41_config(worker.model_config.hf_text_config):
        return
    deepseek_v41_indexer_warmup(worker)
    deepseek_v41_compressor_warmup(worker)
