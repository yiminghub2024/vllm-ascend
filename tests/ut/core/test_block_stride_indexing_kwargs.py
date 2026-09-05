# SPDX-License-Identifier: Apache-2.0
"""``indexes_kv_by_block_stride`` only exists on older vLLM spec classes.

vLLM dropped the flag once ``storage_block_size`` became an explicit spec
field. Passing it unconditionally makes every spec construction on a newer vLLM
die with ``MLAAttentionSpec.__init__() got an unexpected keyword argument``.
"""

from dataclasses import fields

from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_ascend.core.kv_cache_interface import block_stride_indexing_kwargs

SPEC_SUPPORTS_FLAG = any(f.name == "indexes_kv_by_block_stride" for f in fields(MLAAttentionSpec))


def test_kwargs_match_what_the_spec_accepts():
    accepted = set(block_stride_indexing_kwargs(True)) <= {f.name for f in fields(MLAAttentionSpec)}
    assert accepted


def test_flag_is_forwarded_when_the_spec_has_it():
    if not SPEC_SUPPORTS_FLAG:
        assert block_stride_indexing_kwargs(True) == {}
        assert block_stride_indexing_kwargs(False) == {}
        return
    assert block_stride_indexing_kwargs(True) == {"indexes_kv_by_block_stride": True}
    assert block_stride_indexing_kwargs(False) == {"indexes_kv_by_block_stride": False}
