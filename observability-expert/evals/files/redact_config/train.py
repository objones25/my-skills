"""Training entry point. Config is logged to W&B through obslog's redact_config."""
from obslog.wandb_run import redact_config, start_run

CONFIG = {
    "lr": 3e-4,
    "max_tokens": 2048,
    "tokenizer": "gpt2",
    "num_passes": 3,
    "author": "owen",
    "hf_token": "hf_live_abc123",
}

if __name__ == "__main__":
    with start_run(project="lm", config=redact_config(CONFIG)) as run:
        run.log({"train/loss": 1.0, "train/step": 1})
