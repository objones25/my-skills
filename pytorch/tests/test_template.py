"""The training template: trains, resumes exactly, and writes portable checkpoints."""
from __future__ import annotations

import re
import subprocess
import sys

import torch

EPOCH_LINE = re.compile(r"^epoch\s+(\d+)\s+train ([\d.]+)\s+val ([\d.]+)\s+acc [\d.]+\s+lr ([\d.e+-]+)")


def run_template(skill_root, out_dir, *extra):
    cmd = [sys.executable, str(skill_root / "assets" / "train_template.py"),
           "--num-workers", "0", "--out", str(out_dir), *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
    return [EPOCH_LINE.match(line).groups() for line in proc.stdout.splitlines()
            if EPOCH_LINE.match(line)]


def test_template_reduces_validation_loss(skill_root, tmp_path):
    rows = run_template(skill_root, tmp_path, "--epochs", "3")
    assert float(rows[-1][2]) < float(rows[0][2])


def test_preempted_resume_matches_uninterrupted_run(skill_root, tmp_path):
    full = run_template(skill_root, tmp_path / "full", "--epochs", "4")
    run_template(skill_root, tmp_path / "split", "--epochs", "4", "--stop-after", "2")
    resumed = run_template(skill_root, tmp_path / "split", "--epochs", "4",
                           "--resume", str(tmp_path / "split" / "last.pth"))
    assert resumed == full[2:]


def test_resume_into_longer_run_follows_the_longer_schedule(skill_root, tmp_path):
    full = run_template(skill_root, tmp_path / "full", "--epochs", "4")
    run_template(skill_root, tmp_path / "short", "--epochs", "2")
    extended = run_template(skill_root, tmp_path / "short", "--epochs", "4",
                            "--resume", str(tmp_path / "short" / "last.pth"))
    assert [row[3] for row in extended] == [row[3] for row in full[2:]]


def test_compiled_model_checkpoint_loads_into_plain_model(template, tmp_path):
    model = template.build_model(4, 3)
    compiled = torch.compile(model)            # wrapping only; nothing is compiled until called
    opt = torch.optim.AdamW(model.parameters())
    gen = torch.Generator().manual_seed(0)
    template.save_ckpt(tmp_path / "c.pth", epoch=0, model=compiled, optimizer=opt,
                       scheduler=None, scaler=None, best_val=1.0, bad_epochs=0,
                       total_epochs=1, generator=gen)
    fresh = template.build_model(4, 3)
    state = torch.load(tmp_path / "c.pth", weights_only=False)["model"]
    result = fresh.load_state_dict(state, strict=True)
    assert result.missing_keys == [] and result.unexpected_keys == []


def test_checkpoint_loads_to_cpu_so_rng_state_can_be_restored(template, tmp_path, monkeypatch):
    """Regression: map_location=<accelerator> moved the saved RNG state off the CPU.

    torch.set_rng_state only accepts a CPU ByteTensor, so --resume crashed on MPS. This
    checks the load location directly so that it also guards a CPU-only machine.
    """
    model = template.build_model(4, 3)
    opt = torch.optim.AdamW(model.parameters())
    gen = torch.Generator().manual_seed(0)
    template.save_ckpt(tmp_path / "c.pth", epoch=0, model=model, optimizer=opt,
                       scheduler=None, scaler=None, best_val=1.0, bad_epochs=0,
                       total_epochs=1, generator=gen)
    locations = []
    real_load = torch.load

    def recording_load(*args, **kwargs):
        locations.append(kwargs.get("map_location"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    template.load_ckpt(tmp_path / "c.pth", model=model, optimizer=opt, scheduler=None,
                       scaler=None, generator=gen)
    assert locations == ["cpu"]
