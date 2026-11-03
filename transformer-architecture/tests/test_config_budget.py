"""config_budget.py: exact counts, FLOP terms, KV cache, validator, and config ingestion."""
from __future__ import annotations

import json

import pytest

CONFIGS = "assets/hf-configs"


def test_every_preset_has_an_exact_published_total(budget):
    assert set(budget.PUBLISHED_TOTALS) == set(budget.PRESETS) - {"llama3.2-1b", "mistral-7b"}


@pytest.mark.parametrize("preset", [
    "gpt2-124m", "llama3-8b", "qwen2.5-0.5b", "smollm2-135m", "qwen3-8b", "olmo2-7b",
    "gemma3-1b", "mixtral-8x7b", "olmoe-1b-7b", "qwen3-30b-a3b", "gpt-oss-20b",
    "deepseek-v2-lite", "deepseek-v3"])
def test_parameter_count_matches_the_checkpoint_exactly(budget, preset):
    assert budget.param_breakdown(budget.PRESETS[preset])["total"] == budget.PUBLISHED_TOTALS[preset]


@pytest.mark.parametrize(("filename", "preset"), [
    ("qwen3-8b.json", "qwen3-8b"), ("olmo2-7b.json", "olmo2-7b"), ("gemma-3-1b.json", "gemma3-1b"),
    ("mixtral-8x7b.json", "mixtral-8x7b"), ("olmoe-1b-7b.json", "olmoe-1b-7b"),
    ("qwen3-30b-a3b.json", "qwen3-30b-a3b"), ("gpt-oss-20b.json", "gpt-oss-20b"),
    ("deepseek-v2-lite.json", "deepseek-v2-lite"), ("deepseek-v3.json", "deepseek-v3")])
def test_hf_config_ingestion_reproduces_the_checkpoint_total(budget, skill_root, filename, preset):
    c = budget.from_hf(str(skill_root / CONFIGS / filename))
    assert budget.param_breakdown(c)["total"] == budget.PUBLISHED_TOTALS[preset]


def test_olmo2_qk_norm_spans_the_full_projection_not_one_head(budget):
    c = budget.PRESETS["olmo2-7b"]
    assert budget.attention_params(c)["qk_norm"] == 2 * 4096


def test_qwen3_qk_norm_is_one_gain_vector_per_head_shape(budget):
    assert budget.attention_params(budget.PRESETS["qwen3-8b"])["qk_norm"] == 2 * 128


def test_active_parameters_drop_unused_experts_only(budget):
    c = budget.PRESETS["qwen3-30b-a3b"]
    p = budget.param_breakdown(c)
    expert = 3 * 2048 * 768
    assert p["total"] - p["active"] == 48 * (128 - 8) * expert


@pytest.mark.parametrize(("preset", "convention"), [
    ("qwen3-30b-a3b", "active"), ("gpt-oss-20b", "active_with_unembedding")])
def test_active_count_matches_the_lab_convention(budget, preset, convention):
    p = budget.param_breakdown(budget.PRESETS[preset])
    claim = budget.PUBLISHED_ACTIVE[preset]
    assert abs(p[convention] - claim) / claim < 0.02


def test_dense_model_active_equals_total(budget):
    p = budget.param_breakdown(budget.PRESETS["llama3-8b"])
    assert p["active"] == p["total"]


def test_moe_dense_flops_use_active_not_total_parameters(budget):
    c = budget.PRESETS["mixtral-8x7b"]
    p = budget.param_breakdown(c)
    assert budget.flops_per_token(c, p)["dense"] == 6 * p["active_non_embedding"]


def test_flops_include_the_output_projection(budget):
    c = budget.PRESETS["qwen2.5-0.5b"]
    flops = budget.flops_per_token(c, budget.param_breakdown(c))
    assert flops["head"] == 6 * c.vocab * c.d_model


def test_flop_total_is_the_sum_of_its_terms(budget):
    c = budget.PRESETS["llama3-8b"]
    f = budget.flops_per_token(c, budget.param_breakdown(c))
    assert f["total"] == f["dense"] + f["head"] + f["attention"]


def test_dense_attention_flops_equal_palm_formula(budget):
    c = budget.PRESETS["llama3-8b"]
    f = budget.flops_per_token(c, budget.param_breakdown(c))
    assert f["attention"] == 12 * c.n_layers * c.n_heads * c.head_dim * c.seq_len


def test_llama3_8b_kv_cache_is_128_kib_per_token(budget):
    assert budget.kv_cache_bytes_per_token(budget.PRESETS["llama3-8b"]) == 128 * 1024


def test_mla_caches_latent_plus_shared_rope_key(budget):
    c = budget.PRESETS["deepseek-v3"]
    assert budget.kv_elements_per_token_per_layer(c) == 512 + 64


def test_deepseek_v3_mla_cache_is_70272_bytes_per_token(budget):
    assert budget.kv_cache_bytes_per_token(budget.PRESETS["deepseek-v3"]) == 61 * 576 * 2


def test_sliding_layers_stop_growing_at_the_window(budget):
    c = budget.PRESETS["gpt-oss-20b"]
    per_layer = 2 * 8 * 64 * 2
    assert budget.kv_cache_bytes(c, 4096) == (12 * 4096 + 12 * 128) * per_layer


def test_sliding_layers_cost_the_same_below_the_window(budget):
    c = budget.PRESETS["gpt-oss-20b"]
    per_layer = 2 * 8 * 64 * 2
    assert budget.kv_cache_bytes(c, 100) == 24 * 100 * per_layer


@pytest.mark.parametrize("preset", [
    "llama3-8b", "llama3.2-1b", "qwen2.5-0.5b", "qwen3-8b",
    "mistral-7b", "olmo2-7b", "smollm2-135m"])
def test_validator_raises_no_warning_on_a_shipping_dense_model(budget, preset):
    c = budget.PRESETS[preset]
    severities = [s for s, _ in budget.validate(c, budget.param_breakdown(c), None)]
    assert "WARN" not in severities and "ERROR" not in severities


@pytest.mark.parametrize("preset", ["mixtral-8x7b", "qwen3-30b-a3b", "deepseek-v2-lite", "deepseek-v3"])
def test_validator_raises_no_error_on_a_shipping_moe_model(budget, preset):
    c = budget.PRESETS[preset]
    severities = [s for s, _ in budget.validate(c, budget.param_breakdown(c), None)]
    assert "ERROR" not in severities


def test_validator_warns_when_moe_is_barely_sparse(budget):
    c = budget.Config(d_model=1024, n_layers=8, n_heads=8, vocab=32000, d_ff=2816,
                      n_experts=4, top_k=3)
    msgs = [m for s, m in budget.validate(c, budget.param_breakdown(c), None) if s == "WARN"]
    assert any("barely sparse" in m for m in msgs)


def test_gpt2_only_warning_is_its_unpadded_vocab(budget):
    c = budget.PRESETS["gpt2-124m"]
    warns = [m for s, m in budget.validate(c, budget.param_breakdown(c), None) if s == "WARN"]
    assert len(warns) == 1 and warns[0].startswith("vocab=50257 is not a multiple of 64")


def test_validator_errors_on_uneven_gqa_groups(budget):
    c = budget.Config(d_model=512, n_layers=8, n_heads=8, n_kv_heads=3, vocab=32000, d_ff=1408)
    assert ("ERROR" in [s for s, _ in budget.validate(c, budget.param_breakdown(c), None)])


def test_top_k_above_expert_count_is_rejected(budget):
    with pytest.raises(ValueError, match="top_k=9 must be between 1 and n_experts=8"):
        budget.Config(d_model=512, n_layers=4, n_heads=8, vocab=32000, d_ff=1408, n_experts=8, top_k=9)


def test_gpt2_style_config_json_reproduces_published_count(budget, tmp_path):
    cfg = {"n_embd": 768, "n_layer": 12, "n_head": 12, "vocab_size": 50257,
           "n_positions": 1024, "n_inner": None}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    assert budget.param_breakdown(budget.from_hf(str(path)))["total"] == 124_439_808


def test_llama_style_config_json_reproduces_published_count(budget, tmp_path):
    cfg = {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
           "num_key_value_heads": 8, "vocab_size": 128256, "intermediate_size": 14336,
           "max_position_embeddings": 131072, "tie_word_embeddings": False,
           "hidden_act": "silu", "rms_norm_eps": 1e-5, "model_type": "llama"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    assert budget.param_breakdown(budget.from_hf(str(path)))["total"] == 8_030_261_248


def test_linear_attention_hybrid_is_refused(budget, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"hidden_size": 2048, "num_hidden_layers": 4,
                                "layer_types": ["linear_attention"] * 3 + ["full_attention"]}))
    with pytest.raises(ValueError, match="linear attention or state-space"):
        budget.from_hf(str(path))


def test_gated_shared_expert_is_refused(budget, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"hidden_size": 2048, "num_hidden_layers": 4, "num_experts": 60,
                                "shared_expert_intermediate_size": 5632}))
    with pytest.raises(ValueError, match="gated shared expert"):
        budget.from_hf(str(path))


def test_self_test_passes(budget, capsys):
    assert budget.self_test() == 0
