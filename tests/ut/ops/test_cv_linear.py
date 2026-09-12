# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch
import torch_npu

from vllm_ascend.ops.cv_linear import CVLinearWrapper
from vllm_ascend.quantization.methods import (
    AscendW8A8DynamicLinearMethod,
    AscendW8A8MXFP8DSDynamicLinearMethod,
    AscendW8A8MXFP8DynamicLinearMethod,
)


class _FakeW8A8(AscendW8A8DynamicLinearMethod):
    def __init__(self):
        return


class _FakeMXFP8(AscendW8A8MXFP8DynamicLinearMethod):
    def __init__(self, group_size=32, scale_alg=1):
        self.group_size = group_size
        self.dynamic_mx_quant_scale_alg = scale_alg


class _FakeDSMXFP8(AscendW8A8MXFP8DSDynamicLinearMethod):
    def __init__(self):
        self.group_size = 32
        self.dynamic_mx_quant_scale_alg = 1
        self.block_size = 128


def _linear(scheme, *, gather_output=False, custom_op=None):
    return SimpleNamespace(
        quant_method=scheme,
        weight=torch.zeros(4, 8),
        weight_scale=torch.zeros(4, 1, dtype=torch.uint8),
        params_dtype=torch.bfloat16,
        custom_op=custom_op,
        gather_output=gather_output,
        bias=None,
        forward=Mock(return_value=torch.ones(2, 4)),
    )


def test_detects_mxfp8_and_the_deepseek_ds_linear_subclass():
    assert CVLinearWrapper(_linear(_FakeMXFP8()))._is_mxfp8
    assert CVLinearWrapper(_linear(_FakeDSMXFP8()))._is_mxfp8
    wrapper = CVLinearWrapper(_linear(_FakeW8A8()))
    assert wrapper._is_w8a8_dynamic
    assert not wrapper._is_mxfp8


def test_mxfp8_quantize_uses_dynamic_mx_quant(monkeypatch):
    captured = {}

    def fake_mx_quant(x, dst_type, scale_alg):
        captured["dst_type"] = dst_type
        captured["scale_alg"] = scale_alg
        return torch.zeros_like(x, dtype=torch.float8_e4m3fn), torch.ones(x.shape[0], dtype=torch.uint8)

    monkeypatch.setattr(torch_npu, "npu_dynamic_mx_quant", fake_mx_quant)
    x = torch.randn(3, 8)
    quantized, scale = CVLinearWrapper(_linear(_FakeMXFP8(scale_alg=1))).quantize(x)
    assert captured == {"dst_type": torch.float8_e4m3fn, "scale_alg": 1}
    assert quantized.dtype == torch.float8_e4m3fn
    assert scale.shape == (3,)


def test_mxfp8_matmul_uses_mlapo_scale_dtypes(monkeypatch):
    captured = {}

    def fake_matmul(x, weight, weight_scale, **kwargs):
        captured.update(kwargs)
        captured["x"] = x
        return torch.zeros(x.shape[0], 4, dtype=torch.bfloat16)

    monkeypatch.setattr(torch_npu, "npu_quant_matmul", fake_matmul)
    x = torch.zeros(2, 8, dtype=torch.float8_e4m3fn)
    scale = torch.ones(2, dtype=torch.uint8)
    out = CVLinearWrapper(_linear(_FakeMXFP8(group_size=32))).matmul(x, scale)
    assert out.shape == (2, 4)
    assert captured["scale_dtype"] is torch_npu.float8_e8m0fnu
    assert captured["pertoken_scale_dtype"] is torch_npu.float8_e8m0fnu
    assert captured["group_sizes"] == [1, 1, 32]
    assert captured["output_dtype"] is torch.bfloat16


def test_wrapped_mxfp8_scheme_is_unwrapped():
    scheme = _FakeMXFP8()
    linear = _linear(SimpleNamespace(quant_method=scheme))
    wrapper = CVLinearWrapper(linear)
    assert wrapper._is_mxfp8
    assert wrapper._mxfp8_method is scheme


def test_communication_skips_mxfp8_split():
    linear = _linear(_FakeMXFP8(), gather_output=True)
    wrapper = CVLinearWrapper(linear)
    x = torch.randn(2, 8)
    quantized, scale = wrapper.quantize(x)
    assert quantized is x
    assert scale is None
    out = wrapper.matmul(x, None)
    linear.forward.assert_called_once_with(x)
    assert out is linear.forward.return_value
