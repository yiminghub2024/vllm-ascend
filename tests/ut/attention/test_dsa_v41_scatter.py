# SPDX-License-Identifier: Apache-2.0
from contextlib import ExitStack
from unittest import mock
from unittest.mock import MagicMock

import pytest
import torch
import torch_npu

from vllm_ascend.attention.dsa_v41 import builds_scatter_nd_update_sk, scatter_cache_sk
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.utils import AscendDeviceType


@pytest.fixture(autouse=True)
def _stub_scatter_ops():
    # CPU images do not register these custom ops on torch.ops._C_ascend.
    with ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(torch.ops._C_ascend, "npu_scatter_nd_update_sk", create=True, new=MagicMock())
        )
        stack.enter_context(mock.patch.object(torch_npu, "npu_scatter_nd_update_", create=True, new=MagicMock()))
        yield


def _on(device_type: AscendDeviceType):
    return mock.patch(
        "vllm_ascend.attention.dsa_v41.get_current_hardware_profile",
        return_value=get_hardware_profile(device_type),
    )


def _plane(*, page_rows: int, payload_rows: int, width: int, num_blocks: int = 2):
    """Build a cache plane the way ``reshape_cache`` does.

    ``page_rows`` is the row capacity of the containing slot page and
    ``payload_rows`` the rows this component owns, so ``page_rows >
    payload_rows`` reproduces a padded plane inside a shared slot.
    """
    itemsize = torch.bfloat16.itemsize
    block_stride = page_rows * width * itemsize
    raw = torch.zeros(num_blocks * block_stride, dtype=torch.uint8)
    return torch.as_strided(
        raw.view(torch.bfloat16),
        size=(num_blocks, payload_rows, 1, width),
        stride=(block_stride // itemsize, width, width, 1),
        storage_offset=0,
    )


@pytest.mark.parametrize(
    ("device_type", "expected"),
    [
        (AscendDeviceType.A2, True),
        (AscendDeviceType.A3, True),
        (AscendDeviceType.A5, False),
    ],
)
def test_scatter_sk_availability_follows_the_custom_op_package(device_type, expected):
    with _on(device_type):
        assert builds_scatter_nd_update_sk() is expected


def test_a3_stores_padded_planes_with_the_stride_preserving_op():
    cache = _plane(page_rows=8, payload_rows=4, width=8)
    slot_mapping = torch.tensor([[0, 1], [-1, -1]], dtype=torch.int32)
    values = torch.ones((2, 8), dtype=torch.bfloat16)

    with _on(AscendDeviceType.A3):
        scatter_cache_sk(cache, slot_mapping, values)

    torch.ops._C_ascend.npu_scatter_nd_update_sk.assert_called_once()
    assert not torch_npu.npu_scatter_nd_update_.called


def test_a5_refuses_padded_planes_instead_of_writing_to_the_wrong_page():
    # scatter_nd_update_sk has no arch35 kernel, and the dense op would address
    # this plane through its shape rather than its 8-row slot page stride.
    cache = _plane(page_rows=8, payload_rows=4, width=8)
    slot_mapping = torch.tensor([[0, 1], [-1, -1]], dtype=torch.int32)
    values = torch.ones((2, 8), dtype=torch.bfloat16)

    with _on(AscendDeviceType.A5), pytest.raises(NotImplementedError, match="arch35"):
        scatter_cache_sk(cache, slot_mapping, values)

    assert not torch_npu.npu_scatter_nd_update_.called


def test_a5_stores_unpadded_planes_with_the_dense_op():
    cache = _plane(page_rows=4, payload_rows=4, width=8)
    slot_mapping = torch.tensor([[0, 1], [-1, -1]], dtype=torch.int32)
    values = torch.ones((2, 8), dtype=torch.bfloat16)

    with _on(AscendDeviceType.A5):
        scatter_cache_sk(cache, slot_mapping, values)

    assert not torch.ops._C_ascend.npu_scatter_nd_update_sk.called
    torch_npu.npu_scatter_nd_update_.assert_called_once()
    # The dense op indexes with int64 and keeps the padded [-1, -1] rows, so
    # ACLGraph capture still sees a fixed [T, 2] shape.
    _, indices, _ = torch_npu.npu_scatter_nd_update_.call_args.args
    assert indices.dtype == torch.int64
    assert indices.tolist() == [[0, 1], [-1, -1]]


@pytest.mark.parametrize("device_type", [AscendDeviceType.A3, AscendDeviceType.A5])
def test_slot_mapping_shape_is_validated_on_every_soc(device_type):
    cache = _plane(page_rows=4, payload_rows=4, width=8)
    values = torch.ones((2, 8), dtype=torch.bfloat16)

    with _on(device_type), pytest.raises(ValueError, match=r"\[T, 2\] slot_mapping"):
        scatter_cache_sk(cache, torch.tensor([0, 1], dtype=torch.int32), values)
