# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend causal conv1d entry points for the GLM-5.3-Flash KDA layers.

The upstream CUDA kernels reference ``tl.extra.cuda.gdc_wait``, which Ascend
Triton does not provide -- the AST visitor raises even when ``launch_pdl`` is
False. Both entry points are therefore routed to Ascend implementations.

``npu_causal_conv1d_custom`` is the same fused operator the Qwen3-Next GDN and
Kimi KDA layers use, so it keeps all per-request bookkeeping on device. That
matters beyond throughput: the PyTorch fallback walks ``query_start_loc`` with
``.item()`` per request, and a host sync is rejected outright while an ACL graph
is being captured, which used to abort decode-FULL capture. The fallback is kept
for builds without the custom operator, where capture is unavailable anyway.

Both entry points take ``x`` token-major (``[num_tokens, dim]``), ``weight`` in
the operator's ``[width, dim]`` kernel layout, and ``conv_state`` exactly as the
Mamba cache allocates it; the fallback re-derives the layouts it needs.
"""

import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.ops.causal_conv1d import (
    causal_conv1d_fn as _torch_causal_conv1d_fn,
)
from vllm_ascend.ops.causal_conv1d import (
    causal_conv1d_update as _torch_causal_conv1d_update,
)

# Mode selectors understood by npu_causal_conv1d_custom.
_ACTIVATION_SILU = 1
_RUN_MODE_VARLEN = 0
_RUN_MODE_UPDATE = 1


def _has_fused_conv1d() -> bool:
    return hasattr(torch.ops._C_ascend, "npu_causal_conv1d_custom")


def causal_conv1d_fn(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    initial_state_mode: torch.Tensor | None,
) -> torch.Tensor:
    """Run the varlen (prefill) convolution and seed ``conv_state``."""
    if not _has_fused_conv1d():
        return _torch_causal_conv1d_fn(
            x.transpose(0, 1),
            weight.transpose(0, 1),
            bias,
            activation="silu",
            conv_states=conv_state,
            has_initial_state=initial_state_mode,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        ).transpose(0, 1)

    output = torch.empty_like(x)
    # Consume the operator's declared output alias. Returning ``output``
    # independently would let graph functionalization treat the custom-op
    # result as dead and expose the uninitialized allocation instead.
    return torch.ops._C_ascend.npu_causal_conv1d_custom(
        output,
        x,
        weight,
        conv_state=conv_state,
        bias_opt=bias,
        query_start_loc_opt=query_start_loc,
        cache_indices_opt=cache_indices,
        initial_state_mode_opt=initial_state_mode,
        num_accepted_tokens_opt=None,
        activation_mode=_ACTIVATION_SILU,
        pad_slot_id=PAD_SLOT_ID,
        run_mode=_RUN_MODE_VARLEN,
    )


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Slide ``conv_state`` over the decode / draft-verify tokens.

    ``cache_indices`` is the recurrent state index tensor, which carries one
    column per draft slot on the speculative path; only its first column names
    the conv state, so the fallback narrows it.
    """
    if not _has_fused_conv1d():
        return _torch_causal_conv1d_update(
            x,
            conv_state,
            weight.transpose(0, 1),
            bias,
            activation="silu",
            conv_state_indices=cache_indices[:, 0] if cache_indices.dim() > 1 else cache_indices,
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc=query_start_loc,
        )

    output = torch.empty_like(x)
    return torch.ops._C_ascend.npu_causal_conv1d_custom(
        output,
        x,
        weight,
        conv_state=conv_state,
        bias_opt=bias,
        query_start_loc_opt=query_start_loc,
        cache_indices_opt=cache_indices,
        initial_state_mode_opt=None,
        num_accepted_tokens_opt=num_accepted_tokens,
        activation_mode=_ACTIVATION_SILU,
        pad_slot_id=PAD_SLOT_ID,
        run_mode=_RUN_MODE_UPDATE,
    )
