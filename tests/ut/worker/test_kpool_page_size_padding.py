# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash's KV cache pages must reach vLLM unpadded.

Both Ascend model runners pad attention pages up to the Mamba state page when a
model has Mamba layers, so a hybrid model can share one block allocation. GLM
-5.3-Flash needs the opposite: its grouping path in vLLM pads the KDA state page
up to the *MLA* page and aliases the indexer and tail pages into the attention
group, which only works on specs that arrive unpadded -- it asserts
``page_size_padded is None`` on every MLA spec and aborts KV cache setup
otherwise.

The guard is a single predicate in a long function, easy to drop in a refactor
and only observable on an 8-card GLM boot, so this checks it at the source
level instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PREDICATE = "model_uses_kpool_indexer"

RUNNERS = [
    "vllm_ascend/worker/model_runner_v1.py",
    "vllm_ascend/worker/v2/attn_utils.py",
]


def _mentions_predicate(node: ast.AST) -> bool:
    return any(isinstance(sub, ast.Name) and sub.id == PREDICATE for sub in ast.walk(node))


def _writes_page_size_padded(statements: list[ast.stmt]) -> bool:
    """Whether any statement sets ``page_size_padded``.

    Covers both spellings the runners use: the ``object.__setattr__`` string
    literal and the ``dataclasses.replace`` keyword.
    """
    for statement in statements:
        for node in ast.walk(statement):
            if isinstance(node, ast.Constant) and node.value == "page_size_padded":
                return True
            if isinstance(node, ast.keyword) and node.arg == "page_size_padded":
                return True
    return False


def _kpool_branches(tree: ast.Module) -> list[tuple[ast.If, list[ast.stmt], list[ast.stmt]]]:
    """Find the branches gated on the kpool predicate.

    Returns ``(node, taken_for_kpool, taken_otherwise)`` per branch, resolving
    ``not <predicate>`` so both spellings of the guard read the same way here.
    """
    branches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        negated = isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not)
        test = node.test.operand if negated else node.test
        if not _mentions_predicate(test):
            continue
        branches.append((node, node.orelse, node.body) if negated else (node, node.body, node.orelse))
    return branches


@pytest.mark.parametrize("runner", RUNNERS)
def test_page_size_padding_is_gated_on_the_kpool_predicate(runner: str) -> None:
    tree = ast.parse((ROOT / runner).read_text(encoding="utf-8"))
    branches = _kpool_branches(tree)

    assert branches, f"{runner}: no branch on {PREDICATE}(); page padding is unconditional"

    padding_branches = [
        (node, for_kpool) for node, for_kpool, otherwise in branches if _writes_page_size_padded(otherwise)
    ]
    assert padding_branches, f"{runner}: no {PREDICATE}() branch guards a page_size_padded write"

    for node, for_kpool in padding_branches:
        assert not _writes_page_size_padded(for_kpool), (
            f"{runner}:{node.lineno}: pads page_size_padded on the kpool path, "
            "which makes vLLM's GLM-5.3-Flash grouping abort"
        )
