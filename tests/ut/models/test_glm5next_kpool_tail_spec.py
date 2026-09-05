# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.3-Flash kpool tail must use vLLM's own spec class.

vLLM's GLM-5.3-Flash KV cache grouping picks the tail layers out with
``isinstance(spec, KpoolTailSpec)`` against the class it defines itself. A
downstream duplicate of that dataclass is not the same class, so the tail
layers would stay in the attention group and the whole model would fall off
the GLM grouping path onto the generic hybrid path -- which refuses to pad MLA
pages and aborts KV cache setup.

This file guards the invariant at the source level: no ``vllm_ascend`` module
may define or import a competing ``KpoolTailSpec`` / ``KpoolTailManager``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import torch
from vllm.v1.core.single_type_kv_cache_manager import KpoolTailManager, register_all_kvcache_specs
from vllm.v1.kv_cache_interface import KpoolTailSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_ascend.core.kv_cache_interface import register_ascend_kv_cache_specs

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "vllm_ascend"

UPSTREAM_SPEC_MODULE = "vllm.v1.kv_cache_interface"
UPSTREAM_MANAGER_MODULE = "vllm.v1.core.single_type_kv_cache_manager"
GUARDED_NAMES = {"KpoolTailSpec": UPSTREAM_SPEC_MODULE, "KpoolTailManager": UPSTREAM_MANAGER_MODULE}


def _sources() -> list[tuple[Path, ast.Module]]:
    return [(path, ast.parse(path.read_text(encoding="utf-8"))) for path in PACKAGE.rglob("*.py")]


def test_no_downstream_definition_of_the_kpool_tail_classes() -> None:
    offenders = [
        f"{path.relative_to(ROOT)}: class {node.name}"
        for path, tree in _sources()
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name in GUARDED_NAMES
    ]

    assert not offenders, f"redefines vLLM's kpool tail classes: {offenders}"


def test_kpool_tail_classes_are_imported_from_vllm() -> None:
    offenders = []
    for path, tree in _sources():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            for alias in node.names:
                expected = GUARDED_NAMES.get(alias.name)
                if expected is not None and node.module != expected:
                    offenders.append(f"{path.relative_to(ROOT)}: {alias.name} from {node.module}")

    assert not offenders, f"kpool tail classes must come from vLLM: {offenders}"


def _tail_spec() -> KpoolTailSpec:
    """A tail spec shaped like ``Glm5NextTailCache.get_kv_cache_spec`` builds."""
    return KpoolTailSpec(
        block_size=4,
        num_kv_heads=2,
        head_size=128,
        head_size_v=0,
        dtype=torch.bfloat16,
        sliding_window=4,
    )


def test_upstream_tail_spec_opts_out_of_prefix_caching() -> None:
    """The tail is a per-request circular scratch block, so it is never shared.

    ``vllm_ascend.patch.platform.patch_kv_cache_coordinator`` relies on this to
    exempt the kpool-sized tail group from the hash block size divisibility
    check; upstream spells the opt-out ``prefix_cacheable``.
    """
    spec = _tail_spec()

    assert spec.prefix_cacheable is False
    assert spec.max_num_blocks_per_req(None, 8192) == 1


def test_vllm_owns_the_tail_spec_to_manager_pairing() -> None:
    """Registration belongs to vLLM, so ``register_ascend_kv_cache_specs``
    must not pair the tail spec with a manager of its own."""
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()

    assert KVCacheSpecRegistry.get_manager_class(_tail_spec()) is KpoolTailManager
