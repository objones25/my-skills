# Hugging Face config.json files

Copied from each model's Hub repo on 2026-11-02. `auto_map` and `quantization_config` were removed (they only affect loading); `_name_or_path` was added so reports name the model. Everything else is verbatim.

Each file's total parameter count was checked three ways: the repo's `model.safetensors.index.json` `total_size` divided by bytes per parameter, instantiating the config on PyTorch's meta device with transformers 5.22.1 (`AutoModelForCausalLM.from_config` under `torch.device("meta")`), and `config_budget.py --from-hf-config`. All three agree exactly. DeepSeek-V3 ships FP8 weights with scale tensors, so its total comes from the meta-device count only, and it excludes the multi-token-prediction module, as the Transformers implementation does.

| File | Repo | Total parameters |
|---|---|---|
| `qwen3-8b.json` | Qwen/Qwen3-8B | 8,190,735,360 |
| `olmo2-7b.json` | allenai/OLMo-2-1124-7B | 7,298,617,344 |
| `gemma-3-1b.json` | google/gemma-3-1b-pt | 999,885,952 |
| `mixtral-8x7b.json` | mistralai/Mixtral-8x7B-v0.1 | 46,702,792,704 |
| `olmoe-1b-7b.json` | allenai/OLMoE-1B-7B-0924 | 6,919,161,856 |
| `qwen3-30b-a3b.json` | Qwen/Qwen3-30B-A3B | 30,532,122,624 |
| `gpt-oss-20b.json` | openai/gpt-oss-20b | 20,914,757,184 |
| `deepseek-v2-lite.json` | deepseek-ai/DeepSeek-V2-Lite | 15,706,484,224 |
| `deepseek-v3.json` | deepseek-ai/DeepSeek-V3 | 671,026,404,352 |

Two facts surfaced while checking: DeepSeek-V3's balancing bias (`e_score_correction_bias`) is a registered buffer, not a parameter, so it is not in these totals; OLMo 2 and OLMoE normalize the full Q and K projections, so their QK-norm costs `n_heads*head_dim + n_kv_heads*head_dim` per layer, not `2*head_dim`.
