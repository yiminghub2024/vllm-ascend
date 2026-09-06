# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.3-Flash KDA convolution entry points.

These wrappers exist so the KDA layer reaches ``npu_causal_conv1d_custom``,
whose per-request bookkeeping stays on device. The PyTorch fallback instead
reads ``query_start_loc`` with ``.item()`` per request, and a host sync is
rejected outright while an ACL graph is being captured, so decode-FULL capture
aborts whenever a call slips onto the fallback. What the fused operator needs is
therefore pinned here: the merged weight in its ``[width, 3C]`` kernel layout,
the mode selectors, and the state-index tensor it reads. The fallback's own
argument shapes are pinned alongside, since it takes different layouts.
"""

from __future__ import annotations

import types

import pytest
import torch

from vllm_ascend.models.glm5next.ops import causal_conv1d as conv1d_module
from vllm_ascend.models.glm5next.ops.causal_conv1d import (
    _ACTIVATION_SILU,
    _RUN_MODE_UPDATE,
    _RUN_MODE_VARLEN,
    causal_conv1d_fn,
    causal_conv1d_update,
)

NUM_TOKENS = 6
MERGED_DIM = 12
CONV_WIDTH = 4
NUM_ROWS = 2
DRAFT_SLOTS = 3


@pytest.fixture
def fused_op_calls(monkeypatch) -> list[dict]:
    """Record calls to a stand-in for the fused Ascend operator."""
    calls: list[dict] = []

    def npu_causal_conv1d_custom(output, x, weight, **kwargs):
        calls.append({"output": output, "x": x, "weight": weight, **kwargs})
        return output

    monkeypatch.setattr(
        torch.ops,
        "_C_ascend",
        types.SimpleNamespace(npu_causal_conv1d_custom=npu_causal_conv1d_custom),
        raising=False,
    )
    return calls


@pytest.fixture
def without_fused_op(monkeypatch) -> None:
    monkeypatch.setattr(conv1d_module, "_has_fused_conv1d", lambda: False)


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "x": torch.randn(NUM_TOKENS, MERGED_DIM),
        "conv_state": torch.zeros(NUM_ROWS + 1, CONV_WIDTH - 1, MERGED_DIM),
        # [width, 3C]: the layout the fused operator consumes.
        "weight": torch.randn(CONV_WIDTH, MERGED_DIM),
        "bias": None,
        "query_start_loc": torch.tensor([0, 3, NUM_TOKENS], dtype=torch.int32),
    }


def test_update_hands_the_fused_operator_its_own_layouts(fused_op_calls) -> None:
    inputs = _inputs()
    # The recurrent state index tensor carries one column per draft slot.
    cache_indices = torch.arange(NUM_ROWS * DRAFT_SLOTS, dtype=torch.int32).reshape(NUM_ROWS, DRAFT_SLOTS)
    num_accepted_tokens = torch.tensor([2, 1], dtype=torch.int32)

    out = causal_conv1d_update(
        inputs["x"],
        inputs["conv_state"],
        inputs["weight"],
        inputs["bias"],
        query_start_loc=inputs["query_start_loc"],
        cache_indices=cache_indices,
        num_accepted_tokens=num_accepted_tokens,
    )

    assert len(fused_op_calls) == 1
    call = fused_op_calls[0]
    # The operator writes through its declared output alias, so the wrapper has
    # to return that tensor rather than an independent allocation.
    assert out is call["output"]
    assert out.shape == inputs["x"].shape
    assert call["weight"].shape == (CONV_WIDTH, MERGED_DIM)
    assert call["conv_state"] is inputs["conv_state"]
    assert call["query_start_loc_opt"] is inputs["query_start_loc"]
    # Narrowing to the first column here would drop the draft slots the
    # operator slides the state across.
    assert torch.equal(call["cache_indices_opt"], cache_indices)
    assert torch.equal(call["num_accepted_tokens_opt"], num_accepted_tokens)
    assert call["initial_state_mode_opt"] is None
    assert call["run_mode"] == _RUN_MODE_UPDATE
    assert call["activation_mode"] == _ACTIVATION_SILU


def test_varlen_hands_the_fused_operator_token_major_input(fused_op_calls) -> None:
    inputs = _inputs()
    cache_indices = torch.tensor([0, 1], dtype=torch.int32)
    initial_state_mode = torch.tensor([False, True])

    out = causal_conv1d_fn(
        inputs["x"],
        inputs["conv_state"],
        inputs["weight"],
        inputs["bias"],
        query_start_loc=inputs["query_start_loc"],
        cache_indices=cache_indices,
        initial_state_mode=initial_state_mode,
    )

    assert len(fused_op_calls) == 1
    call = fused_op_calls[0]
    assert out is call["output"]
    # Token-major, unlike the fallback, which wants [dim, num_tokens].
    assert call["x"].shape == (NUM_TOKENS, MERGED_DIM)
    assert call["weight"].shape == (CONV_WIDTH, MERGED_DIM)
    assert call["initial_state_mode_opt"] is initial_state_mode
    assert call["num_accepted_tokens_opt"] is None
    assert call["run_mode"] == _RUN_MODE_VARLEN


def test_update_fallback_narrows_the_state_indices(monkeypatch, without_fused_op) -> None:
    inputs = _inputs()
    cache_indices = torch.arange(NUM_ROWS * DRAFT_SLOTS, dtype=torch.int32).reshape(NUM_ROWS, DRAFT_SLOTS)
    captured: dict = {}

    def fake_update(x, conv_state, weight, bias, **kwargs):
        captured.update(x=x, weight=weight, **kwargs)
        return torch.zeros_like(x)

    monkeypatch.setattr(conv1d_module, "_torch_causal_conv1d_update", fake_update)

    causal_conv1d_update(
        inputs["x"],
        inputs["conv_state"],
        inputs["weight"],
        inputs["bias"],
        query_start_loc=inputs["query_start_loc"],
        cache_indices=cache_indices,
    )

    # The fallback indexes conv_state per request, so only the column naming
    # the conv state survives, and it wants the [3C, width] weight.
    assert torch.equal(captured["conv_state_indices"], cache_indices[:, 0])
    assert captured["weight"].shape == (MERGED_DIM, CONV_WIDTH)
    assert captured["activation"] == "silu"


def test_varlen_fallback_uses_channel_major_tokens(monkeypatch, without_fused_op) -> None:
    inputs = _inputs()
    captured: dict = {}

    def fake_fn(x, weight, bias, **kwargs):
        captured.update(x=x, weight=weight, **kwargs)
        return torch.zeros(MERGED_DIM, NUM_TOKENS)

    monkeypatch.setattr(conv1d_module, "_torch_causal_conv1d_fn", fake_fn)

    out = causal_conv1d_fn(
        inputs["x"],
        inputs["conv_state"],
        inputs["weight"],
        inputs["bias"],
        query_start_loc=inputs["query_start_loc"],
        cache_indices=torch.tensor([0, 1], dtype=torch.int32),
        initial_state_mode=torch.tensor([False, True]),
    )

    assert captured["x"].shape == (MERGED_DIM, NUM_TOKENS)
    assert captured["weight"].shape == (MERGED_DIM, CONV_WIDTH)
    # Callers get token-major output back either way.
    assert out.shape == (NUM_TOKENS, MERGED_DIM)


def test_packed_conv_weight_keeps_qkv_channel_order() -> None:
    from vllm_ascend.models.glm5next.kda import Glm5NextLinearAttention

    channels = MERGED_DIM // 3
    # vLLM stores each conv weight checkpoint-compatible as fp32 [C, 1, width].
    weights = [torch.randn(channels, 1, CONV_WIDTH) for _ in range(3)]
    layer = object.__new__(Glm5NextLinearAttention)
    layer.q_conv1d, layer.k_conv1d, layer.v_conv1d = (types.SimpleNamespace(weight=weight) for weight in weights)
    layer.model_config = types.SimpleNamespace(dtype=torch.bfloat16)
    layer._packed_conv_weight = None

    layer._pack_conv_weight()

    packed = layer._packed_conv_weight
    assert packed.shape == (CONV_WIDTH, MERGED_DIM)
    assert packed.dtype == torch.bfloat16
    assert packed.is_contiguous()
    expected = torch.cat([weight.view(channels, CONV_WIDTH) for weight in weights], dim=0)
    assert torch.equal(packed.float(), expected.transpose(0, 1).to(torch.bfloat16).float())
