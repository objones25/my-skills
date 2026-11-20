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
    import torch.utils.checkpoint

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


def test_linear_cross_entropy_chunks_only_when_options_are_passed():
    import torch.nn.modules.linear_cross_entropy_options as lce

    assert inspect.signature(F.linear_cross_entropy).parameters["options"].default is None
    doc = lce.LinearCrossEntropyOptions.__doc__
    assert doc is not None
    assert "processes the batch dimension" in " ".join(doc.split())
    h, w = torch.randn(8, 32), torch.randn(1000, 32) * 0.02
    t = torch.randint(0, 1000, (8,))
    chunked = F.linear_cross_entropy(h, w, t, options=nn.LinearCrossEntropyOptions())
    assert chunked.item() == pytest.approx(F.cross_entropy(h @ w.T, t).item(), abs=1e-5)


def test_load_of_safetensors_needs_the_safetensors_package(tmp_path, monkeypatch):
    import sys

    path = tmp_path / "t.safetensors"
    path.write_bytes(b"\x00" * 16)
    monkeypatch.setitem(sys.modules, "safetensors", None)
    monkeypatch.setitem(sys.modules, "safetensors.torch", None)
    with pytest.raises(RuntimeError, match="safetensors"):
        torch.load(path)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_rmsnorm_default_eps_is_float32_eps_for_half_inputs(dtype):
    x = torch.full((1, 4), 1e-4).to(dtype)
    got = nn.RMSNorm(4, elementwise_affine=False)(x)
    expected = F.rms_norm(x, (4,), eps=torch.finfo(torch.float32).eps)
    assert torch.equal(got, expected)
    if dtype != torch.float32:
        assert not torch.equal(got, F.rms_norm(x, (4,), eps=torch.finfo(dtype).eps))


def test_densenet_classifier_is_a_linear_not_a_sequential():
    models = pytest.importorskip("torchvision.models")
    assert type(models.densenet121(weights=None).classifier) is nn.Linear
    assert type(models.vgg11(weights=None).classifier) is nn.Sequential
    assert type(models.efficientnet_b0(weights=None).classifier) is nn.Sequential


def test_named_tensors_are_gone():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert not hasattr(torch.randn(2), "names")


def _torch_minor() -> tuple[int, int]:
    major, minor = torch.__version__.split(".")[:2]
    return int(major), int(minor)


@pytest.fixture
def one_rank_group():
    import torch.distributed as dist

    store = dist.HashStore()  # pyright: ignore[reportPrivateImportUsage] # stub gap: the class is exported at runtime
    dist.init_process_group("gloo", store=store, rank=0, world_size=1)
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


@pytest.mark.skipif(_torch_minor() < (2, 15), reason="2.15 took nested tensors off the weights_only allowlist")
def test_jagged_nested_checkpoint_needs_rebuild_njt_and_nested_tensor(tmp_path):
    from torch.nested._internal.nested_tensor import NestedTensor, _rebuild_njt

    path = tmp_path / "nt.pt"
    nt = torch.nested.nested_tensor([torch.ones(2), torch.ones(3)], layout=torch.jagged)
    torch.save({"x": nt}, path)
    with torch.serialization.safe_globals([_rebuild_njt]), pytest.raises(Exception, match="NestedTensor"):
        torch.load(path, weights_only=True)
    with torch.serialization.safe_globals([_rebuild_njt, NestedTensor]):
        loaded = torch.load(path, weights_only=True)["x"]
    assert [t.numel() for t in loaded.unbind()] == [2, 3]
