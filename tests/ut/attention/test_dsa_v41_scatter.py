# SPDX-License-Identifier: Apache-2.0
from contextlib import ExitStack
from unittest import mock
from unittest.mock import MagicMock

import pytest
import torch
import torch_npu

from vllm_ascend.attention.dsa_v41 import flat_row_store, scatter_cache_sk
from vllm_ascend.core.deepseek_v41 import _flat_row_page_size, builds_scatter_nd_update_sk
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.utils import AscendDeviceType


def _dense_scatter(cache, indices, updates):
    """CPU stand-in for npu_scatter_nd_update_, which skips negative rows."""
    rows = indices.squeeze(-1)
    keep = rows >= 0
    cache.index_put_((rows[keep],), updates[keep])


@pytest.fixture(autouse=True)
def _stub_scatter_ops():
    # CPU images do not register these custom ops on torch.ops._C_ascend.
    with ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(torch.ops._C_ascend, "npu_scatter_nd_update_sk", create=True, new=MagicMock())
        )
        stack.enter_context(
            mock.patch.object(
                torch_npu, "npu_scatter_nd_update_", create=True, new=MagicMock(side_effect=_dense_scatter)
            )
        )
        yield


def _on(device_type: AscendDeviceType):
    return mock.patch(
        "vllm_ascend.core.deepseek_v41.get_current_hardware_profile",
        return_value=get_hardware_profile(device_type),
    )


def _plane(raw, *, dtype, num_pages, payload_rows, width, page_bytes, byte_offset=0):
    """Build a cache plane the way ``reshape_cache`` does.

    ``page_bytes`` is the containing slot page, so ``page_bytes >
    payload_rows * width * itemsize`` reproduces a padded plane, and
    ``byte_offset`` places a plane after the one sharing its page.
    """
    itemsize = dtype.itemsize
    return torch.as_strided(
        raw.view(dtype),
        size=(num_pages, payload_rows, 1, width),
        stride=(page_bytes // itemsize, width, width, 1),
        storage_offset=byte_offset // itemsize,
    )


# Production geometry from vllm_ascend/models/deepseek_v41/README.md, with
# slot 3 carrying the page alignment this path needs.
_C2_KV = dict(dtype=torch.bfloat16, payload_rows=64, width=512, page_bytes=131072)
_C2_INDEX_K = dict(dtype=torch.int8, payload_rows=64, width=128, page_bytes=131072, byte_offset=65536)
_C1_KV = dict(dtype=torch.bfloat16, payload_rows=128, width=512, page_bytes=148480)


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


def test_a3_stores_through_the_stride_preserving_op():
    raw = torch.zeros(2 * 131072, dtype=torch.uint8)
    cache = _plane(raw, num_pages=2, **_C2_KV)
    slot_mapping = torch.tensor([[0, 1], [-1, -1]], dtype=torch.int32)
    values = torch.ones((2, 512), dtype=torch.bfloat16)

    with _on(AscendDeviceType.A3):
        scatter_cache_sk(cache, slot_mapping, values)

    torch.ops._C_ascend.npu_scatter_nd_update_sk.assert_called_once()
    assert not torch_npu.npu_scatter_nd_update_.called


@pytest.mark.parametrize("geometry", [_C2_KV, _C2_INDEX_K, _C1_KV], ids=["c2_kv", "c2_index_k", "c1_kv"])
def test_a5_reaches_the_same_bytes_as_the_strided_page_view(geometry):
    # scatter_nd_update_sk has no arch35 kernel, so A5 remaps [block, offset]
    # onto one contiguous view. It must land on exactly the same addresses.
    num_pages = 4
    total = num_pages * geometry["page_bytes"]
    reference_raw = torch.zeros(total, dtype=torch.uint8)
    flat_raw = torch.zeros(total, dtype=torch.uint8)
    reference = _plane(reference_raw, num_pages=num_pages, **geometry)
    cache = _plane(flat_raw, num_pages=num_pages, **geometry)

    rows, width = geometry["payload_rows"], geometry["width"]
    slot_mapping = torch.tensor(
        [[0, 0], [0, rows - 1], [-1, -1], [num_pages - 1, rows - 1], [2, rows // 2], [1, 0]],
        dtype=torch.int32,
    )
    values = torch.arange(1, slot_mapping.shape[0] * width + 1, dtype=torch.float32)
    values = values.reshape(slot_mapping.shape[0], width).to(geometry["dtype"])

    squeezed = reference.squeeze(-2)
    for i in range(slot_mapping.shape[0]):
        block, offset = int(slot_mapping[i, 0]), int(slot_mapping[i, 1])
        if block >= 0:
            squeezed[block, offset] = values[i]

    with _on(AscendDeviceType.A5):
        scatter_cache_sk(cache, slot_mapping, values)

    assert reference_raw.any(), "the reference write did not touch the backing"
    assert torch.equal(flat_raw, reference_raw)
    assert not torch.ops._C_ascend.npu_scatter_nd_update_sk.called


def test_flat_row_store_rejects_a_page_that_is_not_a_row_multiple():
    # 147712 is the unaligned C1 slot page: 147712 % 1024 == 256.
    raw = torch.zeros(2 * 147712, dtype=torch.uint8)
    cache = _plane(raw, num_pages=2, dtype=torch.bfloat16, payload_rows=128, width=512, page_bytes=147712)
    indices = torch.tensor([[0, 1]], dtype=torch.int32)
    values = torch.ones((1, 512), dtype=torch.bfloat16)

    with pytest.raises(NotImplementedError, match="not a row multiple"):
        flat_row_store(cache.squeeze(-2), indices, values)


@pytest.mark.parametrize("device_type", [AscendDeviceType.A3, AscendDeviceType.A5])
def test_slot_mapping_shape_is_validated_on_every_soc(device_type):
    raw = torch.zeros(2 * 131072, dtype=torch.uint8)
    cache = _plane(raw, num_pages=2, **_C2_KV)
    values = torch.ones((2, 512), dtype=torch.bfloat16)

    with _on(device_type), pytest.raises(ValueError, match=r"\[T, 2\] slot_mapping"):
        scatter_cache_sk(cache, torch.tensor([0, 1], dtype=torch.int32), values)


class _RowSpec:
    """Minimal stand-in exposing the fields ``_cache_plane_row_bytes`` reads."""

    def __init__(self, width, itemsize):
        self.num_kv_heads = 1
        self.head_size = width
        self.dtype = torch.bfloat16 if itemsize == 2 else torch.float32


@pytest.mark.parametrize(
    ("capacity", "widths", "expected"),
    [
        # Slots 0-2: a 131072-byte page already tiles 1024- and 4096-byte rows.
        (131072, [(512, 2), (1024, 4)], 131072),
        # Slot 3: 147712 leaves 256 bytes over a 1024-byte row.
        (147712, [(512, 2)], 148480),
    ],
)
def test_slot_pages_round_up_to_a_common_row_multiple(capacity, widths, expected):
    specs = [_RowSpec(width, itemsize) for width, itemsize in widths]
    assert _flat_row_page_size(capacity, specs) == expected
