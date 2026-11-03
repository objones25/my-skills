"""assets/ddp_template.py under torchrun with two CPU ranks: replicas agree, shards are disjoint."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch.distributed as dist

pytestmark = [
    pytest.mark.skipif(not dist.is_available(), reason="torch built without distributed"),
    pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend unavailable"),
]
RANK_LINE = re.compile(r"rank (\d) checksum ([-\d.]+) first_epoch_samples (\d+) min \d+ max \d+ sum (\d+)")


def launch(skill_root, tmp_path, *extra):
    torchrun = Path(sys.executable).with_name("torchrun")
    cmd = [str(torchrun) if torchrun.exists() else shutil.which("torchrun"),
           "--standalone", "--nproc_per_node=2",
           str(skill_root / "assets" / "ddp_template.py"), "--epochs", "2",
           "--out", str(tmp_path), *extra]
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env, check=True)
    rows = {int(m.group(1)): m.groups()[1:] for m in RANK_LINE.finditer(proc.stdout)}
    return proc.stdout, rows


@pytest.mark.parametrize("mode", [(), ("--fsdp",)], ids=["ddp", "fsdp2"])
def test_ranks_end_with_identical_weights(skill_root, tmp_path, mode):
    _, rows = launch(skill_root, tmp_path, *mode)
    assert rows[0][0] == rows[1][0]


def test_sampler_gives_each_rank_a_disjoint_half(skill_root, tmp_path):
    _, rows = launch(skill_root, tmp_path)
    assert int(rows[0][2]) + int(rows[1][2]) == sum(range(2048))


def test_loss_decreases_across_epochs(skill_root, tmp_path):
    out, _ = launch(skill_root, tmp_path)
    losses = [float(v) for v in re.findall(r"epoch \d loss ([\d.]+)", out)]
    assert losses[1] < losses[0]


def test_rank_zero_saves_unwrapped_keys(skill_root, tmp_path):
    import torch

    launch(skill_root, tmp_path)
    keys = list(torch.load(tmp_path / "model.pth", weights_only=True))
    assert keys[0] == "0.weight"
