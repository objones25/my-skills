"""The numbers SKILL.md and reference-configs.md quote, recomputed from the table itself."""
from __future__ import annotations

import pytest

from reference_table import dense_rows


@pytest.fixture(scope="module")
def rows(skill_root):
    return dense_rows(skill_root)


def test_table_has_32_dense_models(rows):
    assert len(rows) == 32


def test_smallest_aspect_ratio_is_smollm2_135m(rows):
    smallest = min(rows, key=lambda r: r["aspect"])
    assert (smallest["name"], round(smallest["aspect"], 1)) == ("SmolLM2-135M", 19.2)


def test_aspect_ratio_above_1b_spans_33_to_128(rows):
    big = [r["aspect"] for r in rows if r["params"] > 1e9]
    assert (round(min(big), 1), round(max(big), 1)) == (33.3, 128.0)


def test_every_model_above_21_percent_embedding_share_ties(rows):
    assert all(r["tied"] for r in rows if r["share"] > 0.21)


def test_size_rule_predicts_tying_for_28_of_32(rows):
    assert sum((r["params"] < 5e9) == r["tied"] for r in rows) == 28


def test_ten_percent_share_rule_predicts_tying_for_25_of_32(rows):
    assert sum((r["share"] > 0.10) == r["tied"] for r in rows) == 25


def test_gemma_ties_below_the_ten_percent_line(rows):
    gemma = {r["name"]: r for r in rows if r["name"] in ("Gemma2-9B", "Gemma3-12B")}
    assert all(g["tied"] and g["share"] < 0.10 for g in gemma.values())
