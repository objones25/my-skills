"""Tests for scripts/lint_pytorch.py: a trigger and a near miss for every code, plus the CLI."""
from __future__ import annotations

import ast
import importlib.util
import json
from textwrap import dedent

import pytest


@pytest.fixture(scope="session")
def lint(skill_root):
    spec = importlib.util.spec_from_file_location("lint_pytorch", skill_root / "scripts" / "lint_pytorch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def codes(lint, tmp_path):
    """Write a snippet to tmp_path, lint it, return the sorted list of codes found."""

    def run(source: str) -> list[str]:
        path = tmp_path / "snippet.py"
        path.write_text(dedent(source))
        return sorted(f.code for f in lint.lint_file(path))

    return run


MODULE_HEAD = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"


def row(code: str, source: str, id: str):
    return pytest.param(code, MODULE_HEAD + dedent(source), id=id)


DETECTS = [
    # DISCARDED_TO
    row("DISCARDED_TO", """
        x = torch.randn(3)
        x.to("cpu")
        """, "to-on-factory-tensor"),
    row("DISCARDED_TO", """
        def run(loader, device):
            for xb, yb in loader:
                xb.to(device)
        """, "to-on-loop-unpacked-batch"),
    row("DISCARDED_TO", """
        def run(inputs):
            inputs.float()
        """, "float-on-conventional-name"),
    # SOFTMAX_BEFORE_CE
    row("SOFTMAX_BEFORE_CE", """
        def step(logits, y):
            return F.cross_entropy(F.softmax(logits, dim=1), y)
        """, "F-softmax-into-F-cross-entropy"),
    row("SOFTMAX_BEFORE_CE", """
        def step(logits, y):
            probs = logits.softmax(-1)
            crit = nn.CrossEntropyLoss()
            return crit(probs, y)
        """, "method-softmax-via-variable"),
    row("SOFTMAX_BEFORE_CE", """
        model = nn.Sequential(nn.Linear(4, 3), nn.Softmax(dim=1))
        criterion = nn.CrossEntropyLoss()
        def step(x, y):
            return criterion(model(x), y)
        """, "sequential-softmax-head-into-loss"),
    row("SOFTMAX_BEFORE_CE", """
        def step(z, y):
            return F.binary_cross_entropy_with_logits(torch.sigmoid(z), y)
        """, "sigmoid-into-bce-with-logits"),
    row("SOFTMAX_BEFORE_CE", """
        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.crit = nn.BCEWithLogitsLoss()
            def loss(self, z, y):
                p = torch.sigmoid(z)
                return self.crit(p, y)
        """, "self-bound-loss-module"),
    row("SOFTMAX_BEFORE_CE", """
        def step(z, y):
            return F.binary_cross_entropy_with_logits(F.log_softmax(z, 1), y)
        """, "log-softmax-into-bce-with-logits"),
    # SOFTMAX_HEAD
    row("SOFTMAX_HEAD", """
        head = nn.Sequential(nn.Linear(8, 2), nn.Softmax(dim=-1))
        """, "softmax-last"),
    row("SOFTMAX_HEAD", """
        head = nn.Sequential(nn.Linear(8, 1), nn.Sigmoid())
        """, "sigmoid-last"),
    # PLAIN_LIST_OF_MODULES
    row("PLAIN_LIST_OF_MODULES", """
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [nn.Linear(4, 4), nn.ReLU()]
        """, "list-literal"),
    row("PLAIN_LIST_OF_MODULES", """
        class Net(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.blocks = [nn.Linear(4, 4) for _ in range(n)]
        """, "list-comprehension"),
    row("PLAIN_LIST_OF_MODULES", """
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = {"a": nn.Linear(4, 2), "b": nn.Linear(4, 3)}
        """, "dict-literal"),
    row("PLAIN_LIST_OF_MODULES", """
        class Net(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.layers = []
                for _ in range(n):
                    self.layers.append(nn.Linear(4, 4))
        """, "append-to-empty-list"),
    # UNREGISTERED_TENSOR
    row("UNREGISTERED_TENSOR", """
        class Scale(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.ones(3)
        """, "bare-ones"),
    # ASSIGN_BEFORE_SUPER
    row("ASSIGN_BEFORE_SUPER", """
        class Net(nn.Module):
            def __init__(self):
                self.fc = nn.Linear(2, 2)
                super().__init__()
        """, "layer-before-super"),
    row("ASSIGN_BEFORE_SUPER", """
        class Net(nn.Module):
            def __init__(self):
                self.dim = 4
        """, "super-never-called"),
    # FORWARD_CALL
    row("FORWARD_CALL", """
        def predict(model, x):
            return model.forward(x)
        """, "model-forward"),
    # LOAD_WEIGHTS_ONLY_FALSE
    row("LOAD_WEIGHTS_ONLY_FALSE", """
        state = torch.load("ckpt.pt", weights_only=False)
        """, "weights-only-false"),
    # DEPRECATED_AMP
    row("DEPRECATED_AMP", """
        def step(model, x):
            with torch.cuda.amp.autocast():
                return model(x)
        """, "cuda-amp-autocast"),
    row("DEPRECATED_AMP", """
        scaler = torch.cuda.amp.GradScaler()
        """, "cuda-amp-gradscaler"),
    row("DEPRECATED_AMP", """
        from torch.cuda.amp import autocast
        """, "import-from-cuda-amp"),
    row("DEPRECATED_AMP", """
        ctx = torch.cpu.amp.autocast()
        """, "cpu-amp-autocast"),
    # PRETRAINED_TRUE
    row("PRETRAINED_TRUE", """
        from torchvision.models import resnet18
        model = resnet18(pretrained=True)
        """, "pretrained-true"),
    # TODTYPE_NO_SCALE
    row("TODTYPE_NO_SCALE", """
        from torchvision.transforms import v2
        tf = v2.ToDtype(torch.float32)
        """, "float32-no-scale"),
    # CHECKPOINT_NO_REENTRANT
    row("CHECKPOINT_NO_REENTRANT", """
        from torch.utils.checkpoint import checkpoint
        def fwd(block, x):
            return checkpoint(block, x)
        """, "imported-checkpoint"),
    row("CHECKPOINT_NO_REENTRANT", """
        def fwd(seq, x):
            return torch.utils.checkpoint.checkpoint_sequential(seq, 2, x)
        """, "checkpoint-sequential"),
    # DATAPARALLEL
    row("DATAPARALLEL", """
        model = nn.DataParallel(nn.Linear(2, 2))
        """, "nn-dataparallel"),
    # LOSS_TENSOR_ACCUMULATED
    row("LOSS_TENSOR_ACCUMULATED", """
        def train(model, loader, opt):
            total = 0.0
            for x, y in loader:
                opt.zero_grad()
                loss = F.mse_loss(model(x), y)
                loss.backward()
                opt.step()
                total += loss
            return total
        """, "augassign-loss"),
    row("LOSS_TENSOR_ACCUMULATED", """
        def train(model, loader, opt):
            losses = []
            for x, y in loader:
                opt.zero_grad()
                loss = F.mse_loss(model(x), y)
                loss.backward()
                opt.step()
                losses.append(loss)
            return losses
        """, "append-loss"),
    row("LOSS_TENSOR_ACCUMULATED", """
        def train(model, loader, opt, accelerator):
            running_loss = 0.0
            for batch in loader:
                loss = model(**batch).loss
                accelerator.backward(loss)
                opt.step()
                opt.zero_grad()
                running_loss = running_loss + loss
        """, "accelerator-backward-and-rebinding-sum"),
    # MISSING_ZERO_GRAD
    row("MISSING_ZERO_GRAD", """
        def train(model, loader, optimizer):
            for x, y in loader:
                loss = F.mse_loss(model(x), y)
                loss.backward()
                optimizer.step()
        """, "no-zero-grad"),
    # STEP_BEFORE_BACKWARD
    row("STEP_BEFORE_BACKWARD", """
        def train(model, loader, opt):
            for x, y in loader:
                opt.zero_grad()
                opt.step()
                loss = F.mse_loss(model(x), y)
                loss.backward()
        """, "step-then-backward"),
    # MHA_NO_BATCH_FIRST
    row("MHA_NO_BATCH_FIRST", """
        attn = nn.MultiheadAttention(64, 8)
        """, "mha"),
    row("MHA_NO_BATCH_FIRST", """
        rnn = nn.LSTM(10, 20, num_layers=2)
        """, "lstm"),
    row("MHA_NO_BATCH_FIRST", """
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=8)
        """, "transformer-encoder-layer"),
    # RMSNORM_NO_EPS
    row("RMSNORM_NO_EPS", """
        norm = nn.RMSNorm(64)
        """, "rmsnorm-default-eps"),
    # F_DROPOUT_NO_TRAINING
    row("F_DROPOUT_NO_TRAINING", """
        class Net(nn.Module):
            def forward(self, x):
                return F.dropout(x, 0.1)
        """, "dropout-in-forward"),
    # CUDA_HARDCODE
    row("CUDA_HARDCODE", """
        model = nn.Linear(2, 2).cuda()
        """, "dot-cuda"),
    row("CUDA_HARDCODE", """
        x = torch.zeros(3, device="cuda")
        """, "device-kwarg-literal"),
    row("CUDA_HARDCODE", """
        device = torch.device("cuda:0")
        """, "torch-device-literal"),
    # SET_OF_PARAMS
    row("SET_OF_PARAMS", """
        model = nn.Linear(2, 2)
        opt = torch.optim.SGD(set(model.parameters()), lr=0.1)
        """, "set-call"),
    row("SET_OF_PARAMS", """
        w1, w2 = nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1))
        opt = torch.optim.AdamW({w1, w2})
        """, "set-literal"),
    row("SET_OF_PARAMS", """
        def build(model):
            params = set(model.parameters())
            return torch.optim.AdamW(params)
        """, "set-via-variable"),
    # EVAL_NO_GRAD_GUARD
    row("EVAL_NO_GRAD_GUARD", """
        def evaluate(model, x):
            model.eval()
            return model(x)
        """, "eval-then-call"),
    # SYNTAX
    pytest.param("SYNTAX", "def broken(:\n", id="syntax-error"),
    # SUPPRESSION_NO_REASON
    row("SUPPRESSION_NO_REASON", """
        x = torch.randn(3)
        x.to("cpu")  # lint: ignore[DISCARDED_TO]
        """, "pragma-without-reason"),
]

NEAR_MISSES = [
    # DISCARDED_TO
    row("DISCARDED_TO", """
        model = nn.Linear(2, 2)
        model.to("cpu")
        """, "module-from-nn-call"),
    row("DISCARDED_TO", """
        class Net(nn.Module):
            pass
        m = Net()
        m.float()
        """, "local-module-subclass"),
    row("DISCARDED_TO", """
        def run(model, device):
            model.to(device)
        """, "module-like-name"),
    row("DISCARDED_TO", """
        def run(x, device):
            x = x.to(device)
            return x
        """, "reassigned"),
    row("DISCARDED_TO", """
        def run(loader, device):
            for batch in loader:
                batch.to(device)
        """, "single-loop-target-may-move-in-place"),
    row("DISCARDED_TO", """
        def run(self, device):
            self.buf.to(device)
        """, "attribute-receiver"),
    # SOFTMAX_BEFORE_CE
    row("SOFTMAX_BEFORE_CE", """
        def step(logits, y):
            probs = F.softmax(logits, dim=1)
            acc = (probs.argmax(1) == y).float().mean()
            return F.cross_entropy(logits, y), acc
        """, "softmax-only-for-metrics"),
    row("SOFTMAX_BEFORE_CE", """
        def step(z, y):
            return F.cross_entropy(F.log_softmax(z, 1), y)
        """, "log-softmax-into-ce-is-identical"),
    row("SOFTMAX_BEFORE_CE", """
        def step(z, y):
            return F.nll_loss(F.log_softmax(z, 1), y)
        """, "nll-after-log-softmax"),
    row("SOFTMAX_BEFORE_CE", """
        def step(z, y):
            probs = F.softmax(z, 1)
            probs = z
            return F.cross_entropy(probs, y)
        """, "variable-reassigned-to-logits"),
    # SOFTMAX_HEAD
    row("SOFTMAX_HEAD", """
        head = nn.Sequential(nn.Softmax(dim=-1), nn.Linear(8, 2))
        """, "softmax-not-last"),
    row("SOFTMAX_HEAD", """
        head = nn.Sequential(nn.Linear(8, 2))
        """, "linear-head"),
    # PLAIN_LIST_OF_MODULES
    row("PLAIN_LIST_OF_MODULES", """
        class Net(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(n)])
        """, "wrapped-in-modulelist"),
    row("PLAIN_LIST_OF_MODULES", """
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.acts = [nn.ReLU, nn.GELU]
                self.sizes = [4, 8]
        """, "classes-and-ints-not-instances"),
    row("PLAIN_LIST_OF_MODULES", """
        class Holder:
            def __init__(self):
                self.layers = [nn.Linear(4, 4)]
        """, "not-an-nn-module"),
    row("PLAIN_LIST_OF_MODULES", """
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList()
                self.layers += [nn.Linear(4, 4)]
        """, "augassign-onto-modulelist"),
    # UNREGISTERED_TENSOR
    row("UNREGISTERED_TENSOR", """
        class Scale(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scale", torch.ones(3))
                self.weight = nn.Parameter(torch.ones(3))
        """, "buffer-and-parameter"),
    row("UNREGISTERED_TENSOR", """
        class Scale(nn.Module):
            def forward(self, x):
                self.last = torch.zeros(1)
                return x
        """, "outside-init"),
    # ASSIGN_BEFORE_SUPER
    row("ASSIGN_BEFORE_SUPER", """
        class Net(nn.Module):
            def __init__(self, dim):
                self.dim = dim
                super().__init__()
        """, "plain-value-before-super-works"),
    row("ASSIGN_BEFORE_SUPER", """
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.fc = nn.Linear(2, 2)
        """, "old-style-super-first"),
    # FORWARD_CALL
    row("FORWARD_CALL", """
        class Child(nn.Linear):
            def forward(self, x):
                return super().forward(x) * 2
        """, "super-forward"),
    row("FORWARD_CALL", """
        class Child(nn.Linear):
            def forward(self, x):
                return nn.Linear.forward(self, x) * 2
        """, "explicit-base-forward"),
    # LOAD_WEIGHTS_ONLY_FALSE
    row("LOAD_WEIGHTS_ONLY_FALSE", """
        state = torch.load("ckpt.pt", weights_only=True)
        """, "weights-only-true"),
    # DEPRECATED_AMP
    row("DEPRECATED_AMP", """
        def step(model, x):
            scaler = torch.amp.GradScaler("cuda")
            with torch.amp.autocast("cuda"):
                return model(x), scaler
        """, "torch-amp"),
    # PRETRAINED_TRUE
    row("PRETRAINED_TRUE", """
        from torchvision.models import resnet18, ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        """, "weights-enum"),
    # TODTYPE_NO_SCALE
    row("TODTYPE_NO_SCALE", """
        from torchvision.transforms import v2
        tf = v2.ToDtype(torch.float32, scale=True)
        """, "scale-true"),
    row("TODTYPE_NO_SCALE", """
        from torchvision.transforms import v2
        tf = v2.ToDtype(torch.int64)
        """, "integer-dtype"),
    row("TODTYPE_NO_SCALE", """
        from torchvision.transforms import v2
        tf = v2.ToDtype(torch.float32, scale=False)
        """, "explicit-scale-false"),
    # CHECKPOINT_NO_REENTRANT
    row("CHECKPOINT_NO_REENTRANT", """
        from torch.utils.checkpoint import checkpoint
        def fwd(block, x):
            return checkpoint(block, x, use_reentrant=False)
        """, "use-reentrant-false"),
    row("CHECKPOINT_NO_REENTRANT", """
        def checkpoint(state, path):
            torch.save(state, path)
        checkpoint({}, "a.pt")
        """, "project-own-checkpoint-helper"),
    # DATAPARALLEL
    row("DATAPARALLEL", """
        from torch.nn.parallel import DistributedDataParallel
        model = DistributedDataParallel(nn.Linear(2, 2))
        """, "ddp"),
    # LOSS_TENSOR_ACCUMULATED
    row("LOSS_TENSOR_ACCUMULATED", """
        def train(model, loader, opt):
            total = 0.0
            for x, y in loader:
                opt.zero_grad()
                loss = F.mse_loss(model(x), y)
                loss.backward()
                opt.step()
                total += loss.item()
            return total
        """, "item"),
    row("LOSS_TENSOR_ACCUMULATED", """
        def train(model, loader, opt):
            losses = []
            for x, y in loader:
                opt.zero_grad()
                loss = F.mse_loss(model(x), y)
                loss.backward()
                opt.step()
                losses.append(loss.detach())
            return losses
        """, "detach"),
    row("LOSS_TENSOR_ACCUMULATED", """
        def train(model, parts, opt):
            opt.zero_grad()
            total_loss = 0
            for x, y in parts:
                total_loss += F.mse_loss(model(x), y)
            total_loss.backward()
            opt.step()
        """, "summing-into-the-backward-target"),
    # MISSING_ZERO_GRAD
    row("MISSING_ZERO_GRAD", """
        def train(model, loader, optimizer):
            for x, y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(model(x), y)
                loss.backward()
                optimizer.step()
        """, "optimizer-zero-grad"),
    row("MISSING_ZERO_GRAD", """
        def train(model, loader, opt):
            for x, y in loader:
                loss = F.mse_loss(model(x), y)
                loss.backward()
                opt.step()
                for p in model.parameters():
                    p.grad = None
        """, "grad-set-to-none-by-hand"),
    row("MISSING_ZERO_GRAD", """
        def train(model, loader, opt, sched):
            for x, y in loader:
                loss = F.mse_loss(model(x), y)
                loss.backward()
                sched.step()
        """, "scheduler-step-is-not-an-optimizer"),
    # regressions from the independent fact-check
    row("ASSIGN_BEFORE_SUPER", """
        import math
        class Net(nn.Module):
            def __init__(self, d):
                self.half = math.ceil(d / 2)
                super().__init__()
                self.fc = nn.Linear(d, d)
        """, "non-torch-call-before-super-is-fine"),
    row("DISCARDED_TO", """
        def hf(tokenizer, text, device):
            inputs = tokenizer(text, return_tensors="pt")
            inputs.to(device)
        """, "batchencoding-moves-in-place"),
    # STEP_BEFORE_BACKWARD
    row("STEP_BEFORE_BACKWARD", """
        def train(model, loader, opt, sched):
            for x, y in loader:
                sched.step()
                opt.zero_grad()
                loss = F.mse_loss(model(x), y)
                loss.backward()
                opt.step()
        """, "scheduler-first-optimizer-after"),
    row("STEP_BEFORE_BACKWARD", """
        def train(model, loader, opt):
            for x, y in loader:
                def closure():
                    opt.zero_grad()
                    loss = F.mse_loss(model(x), y)
                    loss.backward()
                    return loss
                opt.step(closure)
        """, "lbfgs-closure"),
    # MHA_NO_BATCH_FIRST
    row("MHA_NO_BATCH_FIRST", """
        attn = nn.MultiheadAttention(64, 8, batch_first=True)
        """, "batch-first-true"),
    row("MHA_NO_BATCH_FIRST", """
        rnn = nn.GRU(10, 20, batch_first=False)
        """, "explicit-false-is-a-choice"),
    row("MHA_NO_BATCH_FIRST", """
        rnn = nn.LSTM(10, 20, 1, True, True)
        """, "lstm-positional-batch-first"),
    # RMSNORM_NO_EPS
    row("RMSNORM_NO_EPS", """
        norm = nn.RMSNorm(64, eps=1e-6)
        """, "eps-keyword"),
    row("RMSNORM_NO_EPS", """
        norm = nn.RMSNorm(64, 1e-6)
        """, "eps-positional"),
    # F_DROPOUT_NO_TRAINING
    row("F_DROPOUT_NO_TRAINING", """
        class Net(nn.Module):
            def forward(self, x):
                return F.dropout(x, 0.1, training=self.training)
        """, "training-flag-passed"),
    row("F_DROPOUT_NO_TRAINING", """
        def augment(x):
            return F.dropout(x, 0.1)
        """, "outside-a-module"),
    row("F_DROPOUT_NO_TRAINING", """
        class Net(nn.Module):
            def forward(self, x):
                return F.alpha_dropout(x, 0.1)
        """, "alpha-dropout-defaults-to-false"),
    # CUDA_HARDCODE
    row("CUDA_HARDCODE", """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.zeros(3, device=device)
        """, "conditional-device"),
    # SET_OF_PARAMS
    row("SET_OF_PARAMS", """
        model = nn.Linear(2, 2)
        opt = torch.optim.SGD(list(model.parameters()), lr=0.1)
        """, "list-of-params"),
    # EVAL_NO_GRAD_GUARD
    row("EVAL_NO_GRAD_GUARD", """
        @torch.inference_mode()
        def evaluate(model, x):
            model.eval()
            return model(x)
        """, "inference-mode-decorator"),
    row("EVAL_NO_GRAD_GUARD", """
        def evaluate(model, x):
            model.eval()
            with torch.no_grad():
                return model(x)
        """, "no-grad-block"),
    row("EVAL_NO_GRAD_GUARD", """
        def finetune(model, x):
            model.eval()
            model.train()
            return model(x)
        """, "train-called-again"),
    # SYNTAX
    pytest.param("SYNTAX", "x = 1\n", id="valid-file"),
    # SUPPRESSION_NO_REASON
    row("SUPPRESSION_NO_REASON", """
        x = torch.randn(3)
        x.to("cpu")  # lint: ignore[DISCARDED_TO] demo of a dropped move
        """, "pragma-with-reason"),
]


@pytest.mark.parametrize(("code", "source"), DETECTS)
def test_code_fires_on_trigger(codes, code, source):
    assert code in codes(source)


@pytest.mark.parametrize(("code", "source"), NEAR_MISSES)
def test_code_stays_quiet_on_near_miss(codes, code, source):
    assert code not in codes(source)


def test_every_code_has_a_trigger_row(lint):
    assert sorted(set(lint.CODES) - {p.values[0] for p in DETECTS}) == []


def test_every_code_has_a_near_miss_row(lint):
    assert sorted(set(lint.CODES) - {p.values[0] for p in NEAR_MISSES}) == []


# --------------------------------------------------------------------------- pragmas
def test_line_pragma_with_reason_suppresses(codes):
    source = 'import torch\nx = torch.randn(3)\nx.to("cpu")  # lint: ignore[DISCARDED_TO] demo\n'
    assert codes(source) == []


def test_pragma_without_reason_is_reported_and_does_not_suppress(codes):
    source = 'import torch\nx = torch.randn(3)\nx.to("cpu")  # lint: ignore[DISCARDED_TO]\n'
    assert codes(source) == ["DISCARDED_TO", "SUPPRESSION_NO_REASON"]


def test_def_line_pragma_covers_the_whole_function(codes):
    source = dedent("""
        def evaluate(model, x):  # lint: ignore[EVAL_NO_GRAD_GUARD] caller wraps this in no_grad
            model.eval()
            return model(x)
        """)
    assert codes(source) == []


def test_pragma_on_last_line_of_multiline_statement_suppresses(codes):
    source = dedent("""
        import torch
        state = torch.load(
            "mine.pt",
            weights_only=False,
        )  # lint: ignore[LOAD_WEIGHTS_ONLY_FALSE] file written by this script
        """)
    assert codes(source) == []


def test_file_pragma_suppresses_everywhere(codes):
    source = "# lint: ignore-file[CUDA_HARDCODE] CUDA-only benchmark\nimport torch\na = torch.ones(1).cuda()\nb = torch.ones(1).cuda()\n"
    assert codes(source) == []


def test_pragma_for_another_code_does_not_suppress(codes):
    source = 'import torch\nx = torch.randn(3)\nx.to("cpu")  # lint: ignore[CUDA_HARDCODE] wrong code\n'
    assert codes(source) == ["DISCARDED_TO"]


# --------------------------------------------------------------------------- CLI
@pytest.fixture
def error_file(tmp_path):
    path = tmp_path / "bad.py"
    path.write_text('import torch\nx = torch.randn(3)\nx.to("cpu")\n')
    return path


@pytest.fixture
def warning_file(tmp_path):
    path = tmp_path / "warn.py"
    path.write_text("import torch.nn as nn\nnorm = nn.RMSNorm(64)\n")
    return path


@pytest.fixture
def clean_file(tmp_path):
    path = tmp_path / "clean.py"
    path.write_text("import torch\nx = torch.randn(3)\nx = x.to('cpu')\n")
    return path


def test_json_output_is_a_list_of_findings(lint, error_file, capsys):
    lint.main([str(error_file), "--format", "json"])
    report = json.loads(capsys.readouterr().out)
    assert report == [{
        "file": str(error_file), "line": 3, "code": "DISCARDED_TO", "severity": "error",
        "message": report[0]["message"], "reference": "references/01-devices-amp.md",
    }]


def test_json_output_for_clean_file_is_empty_list(lint, clean_file, capsys):
    lint.main([str(clean_file), "--format", "json"])
    assert json.loads(capsys.readouterr().out) == []


def test_github_output_uses_workflow_command_syntax(lint, error_file, capsys):
    lint.main([str(error_file), "--format", "github"])
    assert capsys.readouterr().out.startswith(f"::error file={error_file},line=3,title=DISCARDED_TO::")


def test_text_output_names_the_reference(lint, error_file, capsys):
    lint.main([str(error_file)])
    assert "(see references/01-devices-amp.md)" in capsys.readouterr().out


def test_exit_zero_on_clean_file(lint, clean_file):
    assert lint.main([str(clean_file)]) == 0


def test_exit_one_with_an_error_finding(lint, error_file):
    assert lint.main([str(error_file)]) == 1


def test_exit_zero_with_warnings_only(lint, warning_file):
    assert lint.main([str(warning_file)]) == 0


def test_warnings_as_errors_turns_warnings_into_failure(lint, warning_file):
    assert lint.main([str(warning_file), "--warnings-as-errors"]) == 1


def test_ignore_flag_drops_the_code(lint, error_file):
    assert lint.main([str(error_file), "--ignore", "discarded_to"]) == 0


def test_missing_path_is_a_usage_error(lint, tmp_path):
    assert lint.main([str(tmp_path / "nope.py")]) == 2


def test_no_paths_is_a_usage_error(lint):
    assert lint.main([]) == 2


def test_unknown_format_is_rejected_by_argparse(lint, clean_file):
    with pytest.raises(SystemExit, match="^2$"):
        lint.main([str(clean_file), "--format", "sarif"])


def test_directory_is_linted_recursively(lint, tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("import torch.nn as nn\nm = nn.DataParallel(nn.Linear(2, 2))\n")
    (tmp_path / "b.py").write_text("import torch.nn as nn\nn1 = nn.RMSNorm(4)\n")
    assert sorted(f.code for f in lint.lint_paths([tmp_path])) == ["DATAPARALLEL", "RMSNORM_NO_EPS"]


def test_syntax_error_is_reported_not_raised(lint, tmp_path):
    path = tmp_path / "broken.py"
    path.write_text("def f(:\n")
    assert [(f.code, f.line) for f in lint.lint_file(path)] == [("SYNTAX", 1)]


def test_list_codes_prints_every_code(lint, capsys):
    lint.main(["--list-codes"])
    printed = {line.split()[0] for line in capsys.readouterr().out.splitlines()}
    assert printed == set(lint.CODES)


def test_list_codes_shows_severity_and_reference(lint, capsys):
    lint.main(["--list-codes"])
    first = capsys.readouterr().out.splitlines()[0].split()
    assert first[:3] == ["DISCARDED_TO", "error", "references/01-devices-amp.md"]


# --------------------------------------------------------------------------- contract
def test_every_reference_file_exists(lint, skill_root):
    missing = [ref for _, ref, _ in lint.CODES.values() if ref != "-" and not (skill_root / ref).is_file()]
    assert missing == []


def test_every_severity_is_error_or_warning(lint):
    assert {sev for sev, _, _ in lint.CODES.values()} == {"error", "warning"}


def test_script_source_has_no_em_dash(skill_root):
    assert "\u2014" not in (skill_root / "scripts" / "lint_pytorch.py").read_text(encoding="utf-8")


def test_script_does_not_import_torch(skill_root):
    tree = ast.parse((skill_root / "scripts" / "lint_pytorch.py").read_text(encoding="utf-8"))
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert imported <= {"__future__", "argparse", "ast", "io", "json", "re", "sys", "tokenize", "typing", "pathlib"}


def test_train_template_has_no_error_findings(lint, skill_root):
    findings = lint.lint_file(skill_root / "assets" / "train_template.py")
    assert [f.render() for f in findings if f.severity == "error"] == []
