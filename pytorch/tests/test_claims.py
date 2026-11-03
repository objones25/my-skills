"""Behavioural claims the references make, pinned so an edit cannot silently break them."""
from __future__ import annotations

import inspect
import math
import warnings

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class _Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 8)
        self.head = nn.Linear(8, 100, bias=False)
        self.head.weight = self.embed.weight


class _IndexDataset(Dataset):
    def __len__(self):
        return 32

    def __getitem__(self, i):
        return i


def _order(num_workers: int) -> list[int]:
    g = torch.Generator().manual_seed(0)
    loader = DataLoader(_IndexDataset(), batch_size=4, shuffle=True,
                        num_workers=num_workers, generator=g)
    return [int(i) for batch in loader for i in batch]


def test_parameters_counts_a_tied_weight_once():
    assert sum(p.numel() for p in _Tied().parameters()) == 800


def test_state_dict_lists_a_tied_weight_under_both_names():
    assert set(_Tied().state_dict()) == {"embed.weight", "head.weight"}


def test_linear_default_init_is_not_he_init():
    std = nn.Linear(4096, 4096).weight.std().item()
    assert std == pytest.approx(1 / math.sqrt(3 * 4096), rel=0.02)


def test_worker_count_does_not_change_sample_order():
    assert _order(0) == _order(2)


def test_functional_batch_norm_defaults_to_eval_behaviour():
    assert inspect.signature(F.batch_norm).parameters["training"].default is False


def test_linear_cross_entropy_matches_unfused_loss():
    h, w = torch.randn(8, 32), torch.randn(1000, 32) * 0.02
    t = torch.randint(0, 1000, (8,))
    fused = F.linear_cross_entropy(h, w, t)
    assert fused.item() == pytest.approx(F.cross_entropy(h @ w.T, t).item(), abs=1e-5)


def test_checkpoint_without_use_reentrant_warns():
    x = torch.randn(2, 4, requires_grad=True)
    with pytest.warns(UserWarning, match="use_reentrant"):
        torch.utils.checkpoint.checkpoint(nn.Linear(4, 4), x)


def test_fused_adamw_constructs_on_cpu():
    opt = torch.optim.AdamW([nn.Parameter(torch.zeros(2))], fused=True)
    assert opt.defaults["fused"] is True


def test_load_reads_safetensors_natively(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch", reason="writing the file needs safetensors")
    safetensors.save_file({"w": torch.ones(2)}, tmp_path / "t.safetensors")
    assert torch.equal(torch.load(tmp_path / "t.safetensors")["w"], torch.ones(2))


def test_named_tensors_are_gone():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert not hasattr(torch.randn(2), "names")


def _torch_minor() -> tuple[int, int]:
    return tuple(int(v) for v in torch.__version__.split(".")[:2])


@pytest.fixture
def one_rank_group():
    import torch.distributed as dist

    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    yield dist
    dist.destroy_process_group()


def _all_gather_warnings(dist) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dist.all_gather_into_tensor(torch.empty(2), torch.ones(2))
    return [str(w.message) for w in caught if issubclass(w.category, FutureWarning)]


@pytest.mark.skipif(_torch_minor() < (2, 15), reason="2.15 made the old collective name a silent alias")
def test_old_collective_name_is_silent_on_2_15(one_rank_group):
    assert _all_gather_warnings(one_rank_group) == []


@pytest.mark.skipif(_torch_minor() != (2, 14), reason="the FutureWarning was emitted on 2.13 and 2.14")
def test_old_collective_name_warns_on_2_14(one_rank_group):
    assert len(_all_gather_warnings(one_rank_group)) == 1


@pytest.mark.skipif(_torch_minor() < (2, 15), reason="finalize_backward is new in 2.15")
def test_ddp_finalize_backward_requires_the_opt_in_flag(one_rank_group):
    ddp = torch.nn.parallel.DistributedDataParallel(nn.Linear(2, 2))
    with pytest.raises(RuntimeError, match="require_manual_backward_finalization"):
        ddp.finalize_backward()


@pytest.mark.skipif(torch.accelerator.is_available(), reason="needs a machine with no accelerator")
def test_accelerator_rng_state_all_returns_a_list_without_an_accelerator():
    pytest.importorskip("torch.accelerator.random")
    assert torch.accelerator.random.get_rng_state_all() == []
