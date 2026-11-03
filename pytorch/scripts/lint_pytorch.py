#!/usr/bin/env python3
"""Static lint for PyTorch code against the pytorch skill's non-negotiables.

Pure stdlib (ast, tokenize, argparse, json, re, pathlib); Python 3.11+. It never imports torch or the code it
reads, so it runs anywhere, in milliseconds, and is safe on untrusted files.

Usage:
    python scripts/lint_pytorch.py train.py model.py
    python scripts/lint_pytorch.py src/                 (every *.py under src/)
    python scripts/lint_pytorch.py src/ --ignore CUDA_HARDCODE,DATAPARALLEL
    python scripts/lint_pytorch.py src/ --format json|github
    python scripts/lint_pytorch.py src/ --warnings-as-errors
    python scripts/lint_pytorch.py --list-codes

Inline suppression (a reason after the bracket is required; a pragma without one is
itself reported as SUPPRESSION_NO_REASON):
    x.to(device)  # lint: ignore[DISCARDED_TO] x is a custom batch whose .to() is in place
    def evaluate(model, x):  # lint: ignore[EVAL_NO_GRAD_GUARD] caller wraps in no_grad
    # lint: ignore-file[CUDA_HARDCODE] CUDA-only benchmark script

A pragma on a statement's line (any line of a multi-line statement) covers that
statement. A pragma on a def or class line, or on one of its decorators, covers the
whole body.

Exit status: 1 if any error-severity finding remains (or any finding at all with
--warnings-as-errors), 2 for usage errors, 0 otherwise.

How names are resolved
    Imports are followed, so `from torch import nn`, `import torch.nn.functional as F`
    and `from torch.nn import CrossEntropyLoss` all resolve to torch paths. When a file
    has no import for them, the conventional aliases `nn`, `F`, `optim` and `v2` are
    assumed to mean torch.nn, torch.nn.functional, torch.optim and
    torchvision.transforms.v2, so pasted snippets still lint.

Heuristics and their known blind spots
    The linter reads one file at a time and does no type inference, so several codes
    are guesses about what a name holds. Each guess is biased toward silence: when the
    linter cannot tell, it says nothing. That means real bugs slip through; the list
    below says where.

    DISCARDED_TO flags a bare statement `n.to(...)` (also .cuda, .cpu, .float, .half,
    .double, .bfloat16, .long, .int, .bool, .contiguous) only when `n` looks like a
    tensor and does not look like a module. A name looks like a module when its name
    ends in or contains model/net/module/network/encoder/decoder/backbone/head/layer/
    block/policy/critic/actor/teacher/student/classifier (with optional _prefix,
    _suffix or plural), when it was assigned from nn.*(...), from a class defined in
    the file that subclasses nn.Module, from a call whose name ends in
    Model/Net/Module/Network/Sequential or contains "model" (build_model), from
    torch.compile(...), torchvision.models.*, a call with a weights= keyword, or when
    it is a parameter annotated nn.Module. A name looks like a tensor when it is one of
    a short list of conventional tensor names (x, y, xb, inputs, targets, labels,
    images, logits, input_ids, attention_mask, ...), was assigned from a torch.*
    function call or a tensor method such as .view/.reshape/.float, was unpacked from
    a tuple in a for loop or from next(iter(...)), or is a parameter annotated Tensor.
    A name assigned from a call that is neither torch.* nor a tensor method is treated as
    unknown and never flagged, even if it has a conventional name: `inputs =
    tokenizer(...)` returns a BatchEncoding whose .to() moves in place.
    Blind spots: `for batch in loader: batch.to(device)` is not flagged (HF
    BatchEncoding and PyG Data objects really do move in place), nor is any tensor
    held in an attribute (self.buf.to(device)) or in an unconventional bare name.

    SOFTMAX_BEFORE_CE follows a probability through local variables, through a
    module bound in the same function, at module level or as self.<attr> in the same
    class (nn.Softmax(), nn.Sequential(..., nn.Softmax())), and into
    nn.CrossEntropyLoss / F.cross_entropy / nn.BCEWithLogitsLoss /
    F.binary_cross_entropy_with_logits. It cannot see a softmax returned from another
    function or from a custom module's forward. log_softmax into cross_entropy is NOT
    flagged: it is redundant but gives the identical loss (log_softmax is idempotent),
    which was checked numerically on torch 2.15.0. log_softmax into BCEWithLogits is
    flagged.

    PLAIN_LIST_OF_MODULES and UNREGISTERED_TENSOR look only at `self.<attr> = ...`
    inside __init__ of a class whose base resolves to nn.Module, a torch.nn class,
    a name ending in Module, or another nn.Module subclass in the same file.
    Subclasses of a module class imported from elsewhere are not recognised.

    ASSIGN_BEFORE_SUPER fires when the value assigned before super().__init__()
    calls torch.nn.* (a layer, nn.Parameter) or a module class defined in the same
    file, which raises AttributeError there (checked on torch 2.15.0). Plain values,
    including ones built by other calls (`self.half = math.ceil(d / 2)`), work at
    runtime and are left alone. Blind spot: a helper function that returns a module
    is not recognised. An __init__ that never calls super().__init__() is reported once.

    LOSS_TENSOR_ACCUMULATED tracks only names containing "loss" that have .backward()
    called on them in the same function: directly, inside the receiver as in
    scaler.scale(loss).backward(), or as an argument as in accelerator.backward(loss).
    A loss divided into a new name before backward (scaled = loss / k;
    scaled.backward()) is not tracked.

    MISSING_ZERO_GRAD and STEP_BEFORE_BACKWARD treat a receiver as an optimizer when
    its last name segment contains "optim" or "opt" (opt, optimizer, opt_g, d_optim);
    scaler.step(opt) counts as a step. MISSING_ZERO_GRAD looks at one function at a
    time, so a train_step() that leaves zero_grad to its caller is flagged (warning).
    STEP_BEFORE_BACKWARD compares textual order inside a loop body, so an accumulation
    loop that steps at the top of the next iteration is flagged; suppress it there.

    EVAL_NO_GRAD_GUARD flags a function that calls R.eval() and later calls R(...)
    (same dotted receiver) outside a no_grad / inference_mode / set_grad_enabled block,
    with no such decorator, and no R.train() in between. A guard set up by the caller
    is invisible.

    CHECKPOINT_NO_REENTRANT only fires on checkpoint names that resolve to
    torch.utils.checkpoint, so a project's own checkpoint() helper is not flagged.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import sys
import tokenize
from pathlib import Path
from typing import NamedTuple

ERROR, WARNING = "error", "warning"

# code -> (severity, reference file under the skill root, one-line message)
CODES: dict[str, tuple[str, str, str]] = {
    "DISCARDED_TO": (ERROR, "references/01-devices-amp.md",
                     "tensor .to()/.cuda()/.float() result is thrown away; tensors move by copy, so reassign: x = x.to(device)"),
    "SOFTMAX_BEFORE_CE": (ERROR, "references/04-losses-optimizers.md",
                          "softmax/sigmoid output passed to a loss that applies its own; pass raw logits"),
    "SOFTMAX_HEAD": (WARNING, "references/03-models-layers.md",
                     "nn.Sequential ends in Softmax/LogSoftmax/Sigmoid; training losses expect raw logits (fine for an inference-only head)"),
    "PLAIN_LIST_OF_MODULES": (ERROR, "references/03-models-layers.md",
                              "modules in a plain list/dict are not registered (no parameters(), state_dict() or .to()); use nn.ModuleList/ModuleDict"),
    "UNREGISTERED_TENSOR": (WARNING, "references/01-devices-amp.md",
                            "plain tensor attribute is invisible to .to(), state_dict() and the optimizer; use nn.Parameter or register_buffer"),
    "ASSIGN_BEFORE_SUPER": (ERROR, "references/03-models-layers.md",
                            "self attribute set before super().__init__(); modules and Parameters raise AttributeError there, call super().__init__() first"),
    "FORWARD_CALL": (WARNING, "references/03-models-layers.md",
                     ".forward() called directly skips module hooks; call the module: model(x)"),
    "LOAD_WEIGHTS_ONLY_FALSE": (WARNING, "references/05-data-training-loop.md",
                                "torch.load(weights_only=False) unpickles arbitrary code; only for files you wrote yourself, otherwise weights_only=True"),
    "DEPRECATED_AMP": (ERROR, "references/01-devices-amp.md",
                       "torch.cuda.amp / torch.cpu.amp are deprecated (FutureWarning on torch 2.14.1 and 2.15.0); use torch.amp.autocast(device_type) and torch.amp.GradScaler(device_type)"),
    "PRETRAINED_TRUE": (WARNING, "references/07-fine-tuning.md",
                        "pretrained=True is deprecated in torchvision (UserWarning on 0.29); use weights=<ModelWeights>.DEFAULT"),
    "TODTYPE_NO_SCALE": (WARNING, "references/08-transforms-augmentation.md",
                         "v2.ToDtype(float dtype) without scale=True casts but leaves uint8 pixels in [0, 255]"),
    "CHECKPOINT_NO_REENTRANT": (ERROR, "references/13-performance-profiling.md",
                                "checkpoint without use_reentrant warns and falls back to the legacy reentrant path (torch 2.14.1 and 2.15.0); pass use_reentrant=False"),
    "DATAPARALLEL": (WARNING, "references/12-distributed.md",
                     "nn.DataParallel is single-process multi-thread and slower than DistributedDataParallel; use DDP or FSDP2"),
    "LOSS_TENSOR_ACCUMULATED": (ERROR, "references/05-data-training-loop.md",
                                "loss tensor accumulated without .item() or .detach(); every step's autograd graph stays alive and memory grows"),
    "MISSING_ZERO_GRAD": (WARNING, "references/05-data-training-loop.md",
                          "backward() and optimizer step() with no zero_grad() in this function; gradients accumulate across steps"),
    "STEP_BEFORE_BACKWARD": (WARNING, "references/05-data-training-loop.md",
                             "optimizer step() comes before backward() in this loop body; each step applies the previous iteration's gradients"),
    "MHA_NO_BATCH_FIRST": (WARNING, "references/03-models-layers.md",
                           "batch_first defaults to False, so inputs are read as (T, B, E); pass batch_first explicitly"),
    "RMSNORM_NO_EPS": (WARNING, "references/09-attention-transformers.md",
                       "nn.RMSNorm eps defaults to None, which means torch.finfo(dtype).eps; set eps to match the checkpoint or recipe"),
    "F_DROPOUT_NO_TRAINING": (ERROR, "references/02-tensors-autograd.md",
                              "F.dropout defaults to training=True and ignores model.eval(); pass training=self.training"),
    "CUDA_HARDCODE": (WARNING, "references/01-devices-amp.md",
                      "hard-coded CUDA device breaks on CPU and MPS; thread a device variable through instead"),
    "SET_OF_PARAMS": (WARNING, "references/04-losses-optimizers.md",
                      "optimizer params given as a set, whose order changes between runs; torch 2.15.0 emits a FutureWarning (future error), pass a list"),
    "EVAL_NO_GRAD_GUARD": (WARNING, "references/05-data-training-loop.md",
                           "model called after .eval() with no torch.no_grad()/inference_mode(); eval() does not disable autograd"),
    "SUPPRESSION_NO_REASON": (ERROR, "-", "lint: ignore[...] pragma without a reason after the bracket"),
    "SYNTAX": (ERROR, "-", "file does not parse"),
}

PRAGMA = re.compile(r"#\s*lint:\s*(ignore|ignore-file)\[([A-Za-z_,\s]+)\]\s*(.*)$")

DEFAULT_ALIASES = {"nn": "torch.nn", "F": "torch.nn.functional", "optim": "torch.optim",
                   "v2": "torchvision.transforms.v2"}

MOVE_METHODS = frozenset({"to", "cuda", "cpu", "float", "double", "half", "bfloat16",
                          "long", "int", "bool", "contiguous"})
TENSOR_METHODS = MOVE_METHODS | {"view", "reshape", "unsqueeze", "squeeze", "permute",
                                 "transpose", "flatten", "detach", "clone", "expand",
                                 "repeat", "type_as", "view_as", "reshape_as"}
COMMON_TENSOR_NAMES = frozenset({
    "x", "y", "xb", "yb", "X", "Y", "inputs", "input", "targets", "target", "labels",
    "label", "images", "image", "imgs", "img", "logits", "features", "feats", "tensor",
    "input_ids", "attention_mask", "token_ids", "pixel_values",
})
MODULE_WORDS = ("model", "net", "network", "module", "encoder", "decoder", "backbone",
                "head", "layer", "block", "policy", "critic", "actor", "teacher",
                "student", "classifier", "discriminator")
MODULE_NAME_RE = re.compile(
    r"^(?:.*_)?(?:" + "|".join(MODULE_WORDS) + r")s?(?:_.*)?$|^.*(?:net|model|module)s?$",
    re.IGNORECASE,
)
MODULE_CALLEE_RE = re.compile(r"(Model|Net|Module|Network|Sequential)$")
NON_TENSOR_TORCH = frozenset({
    "compile", "device", "load", "save", "no_grad", "enable_grad", "inference_mode",
    "autocast", "manual_seed", "set_grad_enabled", "Generator", "get_default_dtype",
    "set_default_dtype", "is_tensor", "numel", "use_deterministic_algorithms",
})
TORCH_FACTORIES = frozenset({
    "ones", "zeros", "randn", "rand", "randint", "tensor", "arange", "empty", "full",
    "eye", "linspace", "logspace", "ones_like", "zeros_like", "randn_like", "rand_like",
    "full_like", "empty_like", "as_tensor", "from_numpy",
})
PROB_FUNCS = {
    "torch.nn.functional.softmax": "softmax", "torch.softmax": "softmax",
    "torch.special.softmax": "softmax",
    "torch.nn.functional.log_softmax": "log_softmax", "torch.log_softmax": "log_softmax",
    "torch.special.log_softmax": "log_softmax",
    "torch.sigmoid": "sigmoid", "torch.nn.functional.sigmoid": "sigmoid",
    "torch.special.expit": "sigmoid",
}
PROB_METHODS = {"softmax": "softmax", "log_softmax": "log_softmax", "sigmoid": "sigmoid"}
PROB_MODULES = {
    "torch.nn.Softmax": "softmax", "torch.nn.Softmax2d": "softmax",
    "torch.nn.LogSoftmax": "log_softmax", "torch.nn.Sigmoid": "sigmoid",
}
LOSS_FUNCS = {"torch.nn.functional.cross_entropy": "ce",
              "torch.nn.functional.binary_cross_entropy_with_logits": "bcel"}
LOSS_MODULES = {"torch.nn.CrossEntropyLoss": "ce", "torch.nn.BCEWithLogitsLoss": "bcel"}
BATCH_FIRST_POS = {  # class -> index of batch_first when passed positionally
    "torch.nn.MultiheadAttention": 8, "torch.nn.TransformerEncoderLayer": 6,
    "torch.nn.TransformerDecoderLayer": 6, "torch.nn.Transformer": 10,
    "torch.nn.LSTM": 4, "torch.nn.GRU": 4, "torch.nn.RNN": 5,
}
DROPOUT_FUNCS = frozenset({"torch.nn.functional.dropout", "torch.nn.functional.dropout1d",
                           "torch.nn.functional.dropout2d", "torch.nn.functional.dropout3d"})
DEPRECATED_AMP = frozenset({
    "torch.cuda.amp.autocast", "torch.cuda.amp.GradScaler", "torch.cuda.amp.custom_fwd",
    "torch.cuda.amp.custom_bwd", "torch.cpu.amp.autocast", "torch.cpu.amp.GradScaler",
})
FLOAT_DTYPES = frozenset({"torch.float", "torch.float32", "torch.float16", "torch.float64",
                          "torch.half", "torch.double", "torch.bfloat16"})
GRAD_GUARDS = frozenset({"no_grad", "inference_mode", "set_grad_enabled"})
DETACHING_METHODS = frozenset({"item", "detach", "tolist", "numpy"})
SCOPE_BARRIERS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


class Finding(NamedTuple):
    file: str
    line: int
    code: str
    severity: str
    message: str
    reference: str

    def render(self) -> str:
        ref = "" if self.reference == "-" else f" (see {self.reference})"
        return f"{self.file}:{self.line}: {self.code} [{self.severity}]: {self.message}{ref}"


# --------------------------------------------------------------------------- helpers
def dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def has_kw(call: ast.Call, *names: str) -> bool:
    return any(kw.arg in names for kw in call.keywords)


def has_star_kwargs(call: ast.Call) -> bool:
    return any(kw.arg is None for kw in call.keywords)


def kw_value(call: ast.Call, name: str) -> ast.AST | None:
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


def is_const(node: ast.AST | None, value: object) -> bool:
    return isinstance(node, ast.Constant) and node.value is value


def first_arg(call: ast.Call, kw: str) -> ast.AST | None:
    return call.args[0] if call.args else kw_value(call, kw)


def iter_scope(body: list[ast.AST]):
    """Yield every node of a scope without entering nested functions, lambdas or classes."""
    stack = list(reversed(body))
    while stack:
        node = stack.pop()
        yield node
        for child in reversed(list(ast.iter_child_nodes(node))):
            if not isinstance(child, SCOPE_BARRIERS):
                stack.append(child)


def collect_aliases(tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    aliases[a.asname] = a.name
                else:
                    root = a.name.split(".", 1)[0]
                    aliases.setdefault(root, root)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for a in node.names:
                if a.name != "*":
                    aliases[a.asname or a.name] = f"{node.module}.{a.name}"
    return aliases


class Context:
    """Per-file name resolution: import aliases and module-subclass detection."""

    def __init__(self, tree: ast.Module) -> None:
        self.aliases = collect_aliases(tree)
        self.module_classes: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and any(self._is_module_base(b) for b in node.bases):
                self.module_classes.add(node.name)
        changed = True
        while changed:  # classes that subclass another local nn.Module subclass
            changed = False
            for node in ast.walk(tree):
                if (isinstance(node, ast.ClassDef) and node.name not in self.module_classes
                        and any(dotted(b) in self.module_classes for b in node.bases)):
                    self.module_classes.add(node.name)
                    changed = True

    def qual(self, node: ast.AST) -> str:
        name = dotted(node.func) if isinstance(node, ast.Call) else dotted(node)
        if not name:
            return ""
        head, _, rest = name.partition(".")
        base = self.aliases.get(head) or DEFAULT_ALIASES.get(head) or head
        return f"{base}.{rest}" if rest else base

    def _is_module_base(self, base: ast.AST) -> bool:
        q = self.qual(base)
        return q.endswith("Module") or (q.startswith("torch.nn.") and q.rsplit(".", 1)[-1][:1].isupper())

    def is_module_ctor(self, node: ast.AST) -> bool:
        """A call that builds an nn.Module or nn.Parameter (torch.nn.X(...) or a local subclass)."""
        if not isinstance(node, ast.Call):
            return False
        q = self.qual(node)
        tail = q.rsplit(".", 1)[-1]
        if q.startswith("torch.nn.") and q.count(".") == 2 and tail[:1].isupper():
            return True
        return dotted(node.func) in self.module_classes


class Bindings:
    """Simple assignments `name = value` in one scope, looked up by name and line."""

    def __init__(self, nodes: list[ast.AST]) -> None:
        self.values: dict[str, list[tuple[int, ast.AST | None, str]]] = {}
        for node in nodes:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    self._bind(target, node.value, node.lineno, "assign")
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                self._bind(node.target, node.value, node.lineno, "assign")
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                self._bind(node.target, node.iter, node.lineno, "for")
            elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                self._bind(node.optional_vars, node.context_expr, node.context_expr.lineno, "with")
        for entries in self.values.values():
            entries.sort(key=lambda e: e[0])

    def _bind(self, target: ast.AST, value: ast.AST, line: int, how: str) -> None:
        if isinstance(target, ast.Name):
            self.values.setdefault(target.id, []).append((line, value, how))
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                inner = elt.value if isinstance(elt, ast.Starred) else elt
                if isinstance(inner, ast.Name):
                    self.values.setdefault(inner.id, []).append((line, value, how + "-unpack"))
                elif isinstance(inner, (ast.Tuple, ast.List)):
                    self._bind(inner, value, line, how)

    def latest(self, name: str, before: int | None = None) -> tuple[ast.AST | None, str] | None:
        entries = self.values.get(name, [])
        if before is not None:
            entries = [e for e in entries if e[0] <= before]
        return (entries[-1][1], entries[-1][2]) if entries else None

    def all(self, name: str) -> list[tuple[ast.AST | None, str]]:
        return [(v, h) for _, v, h in self.values.get(name, [])]


def read_pragmas(source: str) -> tuple[dict[int, set[str]], set[str], list[int]]:
    line_codes: dict[int, set[str]] = {}
    file_codes: set[str] = set()
    missing: list[int] = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return line_codes, file_codes, missing
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        match = PRAGMA.search(tok.string)
        if not match:
            continue
        kind, codes_text, reason = match.groups()
        codes = {c.strip().upper() for c in codes_text.split(",") if c.strip()}
        if not reason.strip():
            missing.append(tok.start[0])
            continue
        if kind == "ignore-file":
            file_codes |= codes
        else:
            line_codes.setdefault(tok.start[0], set()).update(codes)
    return line_codes, file_codes, missing


# --------------------------------------------------------------------------- linter
class Linter:
    def __init__(self, path: str, tree: ast.Module) -> None:
        self.path = path
        self.tree = tree
        self.ctx = Context(tree)
        self.findings: list[Finding] = []
        module_nodes = list(iter_scope(tree.body))
        self.module_bindings = Bindings(module_nodes)

    def add(self, node: ast.AST | int, code: str, message: str = "") -> None:
        severity, ref, default = CODES[code]
        line = node if isinstance(node, int) else getattr(node, "lineno", 0)
        self.findings.append(Finding(self.path, line, code, severity, message or default, ref))

    # ------------------------------------------------------------------ driver
    def run(self) -> list[Finding]:
        self._whole_file_checks()
        self._scope(self.tree.body, fn=None, cls=None)
        self._walk_defs(self.tree, cls=None)
        return self.findings

    def _walk_defs(self, node: ast.AST, cls: ast.ClassDef | None) -> None:
        """Visit every def and class; a function's `cls` is its directly enclosing class."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                self._class(child)
                self._walk_defs(child, cls=child)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._scope(child.body, fn=child, cls=cls)
                self._walk_defs(child, cls=None)
            else:
                self._walk_defs(child, cls=cls)

    def _builds_module(self, value: ast.AST) -> bool:
        """True when the value calls torch.nn.* (a layer or Parameter) or a module class from this file.

        Only these raise AttributeError before super().__init__(); a plain value built by any
        other call (math.ceil(d / 2), a config lookup) works at runtime, so it is not flagged.
        """
        local = set(self.ctx.module_classes)
        for n in ast.walk(value):
            if isinstance(n, ast.Call):
                q = self.ctx.qual(n)
                if q.startswith("torch.nn.") or q.split(".")[-1] in {"Parameter", "Buffer"} or dotted(n.func) in local:
                    return True
        return False

    def is_module_class(self, cls: ast.ClassDef | None) -> bool:
        return cls is not None and cls.name in self.ctx.module_classes

    # ------------------------------------------------------------------ file-wide
    def _whole_file_checks(self) -> None:
        q = self.ctx.qual
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom) and node.module in {"torch.cuda.amp", "torch.cpu.amp"}:
                names = ", ".join(a.name for a in node.names)
                self.add(node, "DEPRECATED_AMP", f"import of {names} from deprecated {node.module}; use torch.amp")
            elif isinstance(node, ast.Attribute) and q(node) in DEPRECATED_AMP:
                self.add(node, "DEPRECATED_AMP", f"{q(node)} is deprecated (FutureWarning on torch 2.14.1 and 2.15.0); use the torch.amp equivalent with a device_type")
            elif isinstance(node, ast.Call):
                self._call_checks(node)

    def _call_checks(self, node: ast.Call) -> None:
        q = self.ctx.qual(node)
        tail = q.rsplit(".", 1)[-1]
        func = node.func

        if is_const(kw_value(node, "pretrained"), True):
            self.add(node, "PRETRAINED_TRUE")

        if q in {"torch.load", "torch.serialization.load"} and is_const(kw_value(node, "weights_only"), False):
            self.add(node, "LOAD_WEIGHTS_ONLY_FALSE")

        if q in {"torch.nn.DataParallel", "torch.nn.parallel.DataParallel",
                 "torch.nn.parallel.data_parallel.DataParallel"} or dotted(func) == "DataParallel":
            self.add(node, "DATAPARALLEL")

        if tail == "ToDtype" and (q.startswith("torchvision.transforms") or q == "ToDtype"):
            dtype = first_arg(node, "dtype")
            scale = kw_value(node, "scale")
            scale_set = len(node.args) >= 2 or scale is not None  # an explicit scale=False is a choice
            if dtype is not None and self.ctx.qual(dtype) in FLOAT_DTYPES and not scale_set and not has_star_kwargs(node):
                self.add(node, "TODTYPE_NO_SCALE")

        if tail in {"checkpoint", "checkpoint_sequential"} and q.startswith("torch.utils.checkpoint"):
            if not has_kw(node, "use_reentrant") and not has_star_kwargs(node):
                self.add(node, "CHECKPOINT_NO_REENTRANT", f"{tail}() without use_reentrant: torch 2.14.1 and 2.15.0 warn and fall back to the legacy reentrant path; pass use_reentrant=False")

        if q in BATCH_FIRST_POS and not has_kw(node, "batch_first") and not has_star_kwargs(node):
            if len(node.args) <= BATCH_FIRST_POS[q]:
                self.add(node, "MHA_NO_BATCH_FIRST", f"{tail} without batch_first: defaults to False, so inputs are read as (T, B, E); pass batch_first explicitly")

        if q == "torch.nn.RMSNorm" and not has_kw(node, "eps") and len(node.args) < 2 and not has_star_kwargs(node):
            self.add(node, "RMSNORM_NO_EPS")

        if q == "torch.nn.Sequential" and node.args:
            last = node.args[-1]
            if isinstance(last, ast.Call) and self.ctx.qual(last) in PROB_MODULES:
                self.add(node, "SOFTMAX_HEAD", f"nn.Sequential ends in {self.ctx.qual(last).rsplit('.', 1)[-1]}; training losses expect raw logits (fine for an inference-only head)")

        if isinstance(func, ast.Attribute) and func.attr == "forward":
            receiver = func.value
            is_super = isinstance(receiver, ast.Call) and dotted(receiver.func) == "super"
            explicit_base = (dotted(receiver).rsplit(".", 1)[-1][:1].isupper() and node.args
                             and isinstance(node.args[0], ast.Name) and node.args[0].id == "self")
            if not is_super and not explicit_base:
                self.add(node, "FORWARD_CALL", f"{dotted(receiver) or 'expr'}.forward(...) skips module hooks; call {dotted(receiver) or 'the module'}(...)")

        if isinstance(func, ast.Attribute) and func.attr == "cuda" and not q.startswith("torch.cuda"):
            self.add(node, "CUDA_HARDCODE", ".cuda() hard-codes the CUDA device; use .to(device)")
        dev = kw_value(node, "device")
        if isinstance(dev, ast.Constant) and isinstance(dev.value, str) and dev.value.startswith("cuda"):
            self.add(node, "CUDA_HARDCODE", f'device="{dev.value}" hard-codes CUDA; thread a device variable through instead')
        elif (q == "torch.device" or (isinstance(func, ast.Attribute) and func.attr == "to")) and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("cuda"):
                self.add(node, "CUDA_HARDCODE", f'"{arg.value}" hard-codes CUDA; thread a device variable through instead')

    # ------------------------------------------------------------------ classes
    def _class(self, cls: ast.ClassDef) -> None:
        if not self.is_module_class(cls):
            return
        init = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
        if init is None:
            return
        nodes = list(iter_scope(init.body))
        super_line = min((n.lineno for n in nodes if self._is_super_init(n)), default=None)
        empty_containers: set[str] = set()
        reported_no_super = False
        for node in sorted(nodes, key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0))):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            for target in targets:
                if not (isinstance(target, ast.Attribute) and dotted(target.value) == "self"):
                    continue
                value = getattr(node, "value", None)
                if super_line is None and not reported_no_super:
                    reported_no_super = True
                    self.add(node, "ASSIGN_BEFORE_SUPER", f"self.{target.attr} set but __init__ never calls super().__init__()")
                elif super_line is not None and node.lineno < super_line and value is not None and self._builds_module(value):
                    self.add(node, "ASSIGN_BEFORE_SUPER", f"self.{target.attr} set before super().__init__(); modules and Parameters raise AttributeError there, call super().__init__() first")
                if value is None:
                    continue
                if isinstance(value, (ast.List, ast.Dict)) and not getattr(value, "elts", None) and not getattr(value, "keys", None):
                    empty_containers.add(target.attr)
                elif isinstance(node, ast.AugAssign) and target.attr not in empty_containers:
                    continue  # self.layers += [...] on an nn.ModuleList is fine
                elif isinstance(value, (ast.List, ast.ListComp, ast.Dict, ast.DictComp, ast.Tuple, ast.Set)):
                    if any(self.ctx.is_module_ctor(s) for s in ast.walk(value)):
                        self.add(node, "PLAIN_LIST_OF_MODULES", f"self.{target.attr} holds modules in a plain {type(value).__name__.replace('Comp', ' comprehension').lower()}; wrap it in nn.ModuleList/ModuleDict (nn.ParameterList for Parameters)")
                elif isinstance(value, ast.Call) and self.ctx.qual(value).startswith("torch.") \
                        and self.ctx.qual(value).rsplit(".", 1)[-1] in TORCH_FACTORIES and self.ctx.qual(value).count(".") == 1:
                    self.add(node, "UNREGISTERED_TENSOR", f"self.{target.attr} = {self.ctx.qual(value)}(...) is a plain attribute; use nn.Parameter or self.register_buffer")
        for node in nodes:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"append", "extend", "insert"}:
                owner = node.func.value
                if isinstance(owner, ast.Attribute) and dotted(owner.value) == "self" and owner.attr in empty_containers:
                    if any(self.ctx.is_module_ctor(s) for a in node.args for s in ast.walk(a)):
                        self.add(node, "PLAIN_LIST_OF_MODULES", f"modules appended to plain container self.{owner.attr}; make it nn.ModuleList()/nn.ModuleDict()")
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute)
                            and dotted(target.value.value) == "self" and target.value.attr in empty_containers
                            and self.ctx.is_module_ctor(node.value)):
                        self.add(node, "PLAIN_LIST_OF_MODULES", f"module stored in plain container self.{target.value.attr}; make it nn.ModuleDict()")

    @staticmethod
    def _is_super_init(node: ast.AST) -> bool:
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "__init__"):
            return False
        recv = node.func.value
        if isinstance(recv, ast.Call) and dotted(recv.func) == "super":
            return True
        return bool(dotted(recv)) and bool(node.args) and isinstance(node.args[0], ast.Name) and node.args[0].id == "self"

    def _self_bindings(self, cls: ast.ClassDef | None) -> dict[str, ast.AST]:
        out: dict[str, ast.AST] = {}
        if cls is None:
            return out
        init = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
        if init is None:
            return out
        for node in iter_scope(init.body):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and dotted(target.value) == "self":
                        out[target.attr] = node.value
        return out

    # ------------------------------------------------------------------ scopes
    def _scope(self, body: list[ast.AST], fn: ast.AST | None, cls: ast.ClassDef | None) -> None:
        nodes = list(iter_scope(body))
        local = Bindings(nodes)
        self_b = self._self_bindings(cls)
        params = self._params(fn)
        resolver = _Resolver(self.ctx, local, self.module_bindings if fn is not None else None, self_b, params)

        in_module_method = fn is not None and self.is_module_class(cls)
        for node in nodes:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                self._discarded_to(node, resolver)
            if isinstance(node, ast.Call):
                self._softmax_before_ce(node, resolver)
                if in_module_method and self.ctx.qual(node) in DROPOUT_FUNCS:
                    if not has_kw(node, "training") and len(node.args) < 3 and not has_star_kwargs(node):
                        self.add(node, "F_DROPOUT_NO_TRAINING")
                self._set_of_params(node, resolver)

        self._loss_accumulation(nodes)
        self._zero_grad(nodes, fn)
        self._step_order(nodes)
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._eval_guard(fn)

    @staticmethod
    def _params(fn: ast.AST | None) -> dict[str, ast.AST | None]:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return {}
        a = fn.args
        return {arg.arg: arg.annotation for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs]}

    # DISCARDED_TO ------------------------------------------------------------
    def _discarded_to(self, stmt: ast.Expr, r: "_Resolver") -> None:
        call = stmt.value
        if not (isinstance(call.func, ast.Attribute) and call.func.attr in MOVE_METHODS
                and isinstance(call.func.value, ast.Name)):
            return
        name = call.func.value.id
        if r.module_like(name, stmt.lineno) or not r.tensor_like(name, stmt.lineno):
            return
        self.add(stmt, "DISCARDED_TO", f"{name}.{call.func.attr}(...) result is thrown away; tensors move by copy, so reassign: {name} = {name}.{call.func.attr}(...)")

    # SOFTMAX_BEFORE_CE -------------------------------------------------------
    def _softmax_before_ce(self, call: ast.Call, r: "_Resolver") -> None:
        sink = r.loss_kind(call)
        if sink is None:
            return
        arg = first_arg(call, "input")
        if arg is None:
            return
        kind = r.prob_kind(arg, call.lineno)
        if kind is None or (kind == "log_softmax" and sink == "ce"):
            return
        loss = "cross_entropy" if sink == "ce" else "BCEWithLogits"
        self.add(call, "SOFTMAX_BEFORE_CE", f"{kind} output passed to {loss}, which applies its own; pass raw logits")

    # SET_OF_PARAMS -----------------------------------------------------------
    def _set_of_params(self, call: ast.Call, r: "_Resolver") -> None:
        q = self.ctx.qual(call)
        if not (q.startswith("torch.optim.") and q.count(".") == 2 and q.rsplit(".", 1)[-1][:1].isupper()):
            return
        params = first_arg(call, "params")
        if params is None:
            return
        if isinstance(params, ast.Name):
            bound = r.local.latest(params.id, call.lineno)
            params = bound[0] if bound and bound[1] == "assign" else params
        if isinstance(params, (ast.Set, ast.SetComp)) or (isinstance(params, ast.Call) and dotted(params.func) == "set"):
            self.add(call, "SET_OF_PARAMS")

    # LOSS_TENSOR_ACCUMULATED -------------------------------------------------
    def _loss_accumulation(self, nodes: list[ast.AST]) -> None:
        backward_names: set[str] = set()
        for node in nodes:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "backward":
                # receiver covers loss.backward() and scaler.scale(loss).backward();
                # positional args cover accelerator.backward(loss)
                for part in [node.func.value, *node.args]:
                    backward_names |= {n.id for n in ast.walk(part)
                                       if isinstance(n, ast.Name) and "loss" in n.id.lower()}
        if not backward_names:
            return
        for node in nodes:
            hits: set[str] = set()
            target_name = ""
            if isinstance(node, ast.AugAssign):
                target_name = dotted(node.target)
                hits = _bare_names(node.value) & backward_names
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target_name = node.targets[0].id
                if target_name in {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}:
                    hits = _bare_names(node.value) & backward_names
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "append" and node.args):
                target_name = dotted(node.func.value)
                hits = _bare_names(node.args[0]) & backward_names
            hits.discard(target_name)
            if hits and target_name not in backward_names:
                name = sorted(hits)[0]
                self.add(node, "LOSS_TENSOR_ACCUMULATED", f"{name} accumulated into {target_name or 'a container'} as a tensor; use {name}.item() (or .detach()) so each step's graph can be freed")

    # MISSING_ZERO_GRAD -------------------------------------------------------
    def _zero_grad(self, nodes: list[ast.AST], fn: ast.AST | None) -> None:
        backward = [n for n in nodes if _is_backward(n)]
        steps = [n for n in nodes if _is_opt_step(n)]
        if not backward or not steps:
            return
        for n in nodes:
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                if n.func.attr == "zero_grad" or (n.func.attr == "zero_" and dotted(n.func.value).endswith(".grad")):
                    return
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "grad" for t in n.targets):
                return
        self.add(steps[0], "MISSING_ZERO_GRAD")

    # STEP_BEFORE_BACKWARD ----------------------------------------------------
    def _step_order(self, nodes: list[ast.AST]) -> None:
        for loop in nodes:
            if not isinstance(loop, (ast.For, ast.AsyncFor, ast.While)):
                continue
            inner = list(iter_scope(loop.body))
            back_lines = [n.lineno for n in inner if _is_backward(n)]
            steps = [n for n in inner if _is_opt_step(n)]
            if back_lines and steps:
                first_step = min(steps, key=lambda n: n.lineno)
                if first_step.lineno < min(back_lines):
                    self.add(first_step, "STEP_BEFORE_BACKWARD")

    # EVAL_NO_GRAD_GUARD ------------------------------------------------------
    def _eval_guard(self, fn: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for dec in fn.decorator_list:
            if self.ctx.qual(dec).rsplit(".", 1)[-1] in GRAD_GUARDS:
                return
        events: list[tuple[int, int, str, str, bool]] = []  # line, col, kind, receiver, guarded

        def visit(node: ast.AST, guarded: bool) -> None:
            if isinstance(node, SCOPE_BARRIERS):
                return
            if isinstance(node, (ast.With, ast.AsyncWith)):
                if any(self.ctx.qual(item.context_expr).rsplit(".", 1)[-1] in GRAD_GUARDS for item in node.items):
                    guarded = True
            if isinstance(node, ast.Call):
                if self.ctx.qual(node) == "torch.set_grad_enabled":
                    events.append((node.lineno, node.col_offset, "global_guard", "", guarded))
                f = node.func
                if isinstance(f, ast.Attribute) and f.attr in {"eval", "train"} and dotted(f.value):
                    events.append((node.lineno, node.col_offset, f.attr, dotted(f.value), guarded))
                elif dotted(f):
                    events.append((node.lineno, node.col_offset, "call", dotted(f), guarded))
            for child in ast.iter_child_nodes(node):
                visit(child, guarded)

        for stmt in fn.body:
            visit(stmt, False)
        if any(kind == "global_guard" for _, _, kind, _, _ in events):
            return
        in_eval: set[str] = set()
        for line, _, kind, recv, guarded in sorted(events):
            if kind == "eval":
                in_eval.add(recv)
            elif kind == "train":
                in_eval.discard(recv)
            elif kind == "call" and recv in in_eval and not guarded:
                self.add(line, "EVAL_NO_GRAD_GUARD", f"{recv}(...) runs after {recv}.eval() with autograd on; wrap it in torch.inference_mode() or torch.no_grad()")
                return


class _Resolver:
    """Answers 'what does this name probably hold' for one scope."""

    def __init__(self, ctx: Context, local: Bindings, module: Bindings | None,
                 self_b: dict[str, ast.AST], params: dict[str, ast.AST | None]) -> None:
        self.ctx, self.local, self.module, self.self_b, self.params = ctx, local, module, self_b, params

    def binding(self, name: str, line: int) -> tuple[ast.AST | None, str] | None:
        found = self.local.latest(name, line)
        if found is None and self.module is not None and name not in self.params:
            found = self.module.latest(name)
        return found

    def callee_value(self, func: ast.AST, line: int) -> ast.AST | None:
        """For a callee like `crit` or `self.crit`, the expression it was bound to."""
        if isinstance(func, ast.Name):
            found = self.binding(func.id, line)
            return found[0] if found and found[1] == "assign" else None
        if isinstance(func, ast.Attribute) and dotted(func.value) == "self":
            return self.self_b.get(func.attr)
        if isinstance(func, ast.Call):
            return func
        return None

    # modules and tensors --------------------------------------------------------
    def module_expr(self, value: ast.AST | None, depth: int = 0) -> bool:
        if not isinstance(value, ast.Call) or depth > 4:
            return False
        if self.ctx.is_module_ctor(value):
            return True
        q = self.ctx.qual(value)
        name = dotted(value.func)
        tail = name.rsplit(".", 1)[-1]
        if MODULE_CALLEE_RE.search(tail) or "model" in tail.lower() or q == "torch.compile":
            return True
        if q.startswith("torchvision.models.") or has_kw(value, "weights"):
            return True
        if q == "copy.deepcopy" and value.args and isinstance(value.args[0], ast.Name):
            return bool(MODULE_NAME_RE.match(value.args[0].id))
        if isinstance(value.func, ast.Attribute) and value.func.attr in MOVE_METHODS | {"train", "eval", "requires_grad_"}:
            return self.module_expr(value.func.value, depth + 1)
        return False

    def module_like(self, name: str, line: int) -> bool:
        if MODULE_NAME_RE.match(name):
            return True
        ann = self.params.get(name)
        if ann is not None and self.ctx.qual(ann).endswith("Module"):
            return True
        found = self.binding(name, line)
        if found is None:
            return False
        value, how = found
        if how == "assign":
            return self.module_expr(value)
        if how == "for" and isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            return value.func.attr in {"modules", "children", "named_modules", "named_children"}
        if how == "for" and isinstance(value, ast.Name):
            return bool(MODULE_NAME_RE.match(value.id))
        return False

    def tensor_like(self, name: str, line: int) -> bool:
        found_first = self.binding(name, line)
        if found_first is not None:
            value0, how0 = found_first
            if how0 == "assign" and isinstance(value0, ast.Call):
                q0 = self.ctx.qual(value0)
                is_tensor_method = (isinstance(value0.func, ast.Attribute)
                                    and value0.func.attr in TENSOR_METHODS)
                if not q0.startswith("torch.") and not is_tensor_method:
                    return False   # e.g. inputs = tokenizer(...): BatchEncoding.to is in place
        if name in COMMON_TENSOR_NAMES:
            return True
        ann = self.params.get(name)
        if ann is not None and self.ctx.qual(ann) in {"torch.Tensor", "Tensor"}:
            return True
        found = self.binding(name, line)
        if found is None:
            return False
        value, how = found
        if how == "for-unpack":
            return True
        if how == "assign-unpack":
            return isinstance(value, ast.Call) and dotted(value.func) == "next"
        if how != "assign" or not isinstance(value, ast.Call):
            return False
        q = self.ctx.qual(value)
        parts = q.split(".")
        if len(parts) == 2 and parts[0] == "torch" and parts[1][:1].islower() and parts[1] not in NON_TENSOR_TORCH:
            return True
        if isinstance(value.func, ast.Attribute) and value.func.attr in TENSOR_METHODS:
            return not self.module_expr(value.func.value)
        return False

    # probabilities and losses ---------------------------------------------------
    def prob_kind(self, expr: ast.AST, line: int, depth: int = 0) -> str | None:
        if depth > 5:
            return None
        if isinstance(expr, ast.Name):
            found = self.local.latest(expr.id, line)
            if found and found[1] == "assign" and found[0] is not None:
                return self.prob_kind(found[0], getattr(found[0], "lineno", line), depth + 1)
            return None
        if not isinstance(expr, ast.Call):
            return None
        q = self.ctx.qual(expr)
        if q in PROB_FUNCS:
            return PROB_FUNCS[q]
        func = expr.func
        if isinstance(func, ast.Attribute) and func.attr in PROB_METHODS and not q.startswith("torch."):
            return PROB_METHODS[func.attr]
        bound = self.callee_value(func, line)
        if isinstance(bound, ast.Call):
            bq = self.ctx.qual(bound)
            if bq in PROB_MODULES:
                return PROB_MODULES[bq]
            if bq == "torch.nn.Sequential" and bound.args and isinstance(bound.args[-1], ast.Call):
                return PROB_MODULES.get(self.ctx.qual(bound.args[-1]))
        return None

    def loss_kind(self, call: ast.Call) -> str | None:
        q = self.ctx.qual(call)
        if q in LOSS_FUNCS:
            return LOSS_FUNCS[q]
        bound = self.callee_value(call.func, call.lineno)
        if isinstance(bound, ast.Call):
            return LOSS_MODULES.get(self.ctx.qual(bound))
        return None


PLAIN_BUILTINS = frozenset({"int", "float", "str", "bool", "len", "list", "tuple", "dict", "set",
                            "max", "min", "sum", "abs", "round", "sorted", "range", "getattr"})


def _builds_object(value: ast.AST) -> bool:
    """True when the value calls something other than a plain builtin (it may build a module)."""
    return any(isinstance(n, ast.Call) and dotted(n.func) not in PLAIN_BUILTINS for n in ast.walk(value))


def _bare_names(expr: ast.AST) -> set[str]:
    """Names used in expr, skipping anything under .item()/.detach()/.tolist()/float()/.data."""
    out: set[str] = set()
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr in DETACHING_METHODS:
                continue
            if dotted(node.func) in {"float", "int"}:
                continue
        if isinstance(node, ast.Attribute) and node.attr == "data":
            continue
        if isinstance(node, ast.Name):
            out.add(node.id)
        stack.extend(ast.iter_child_nodes(node))
    return out


def _is_backward(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "backward") or (isinstance(node, ast.Call) and dotted(node.func) == "torch.autograd.backward")


def _looks_like_optimizer(node: ast.AST) -> bool:
    name = dotted(node).rsplit(".", 1)[-1].lower()
    if not name or any(bad in name for bad in ("sched", "scaler", "adopt", "option")):
        return False
    return "optim" in name or bool(re.search(r"(^|_)opt($|_|[a-z]?\d*$)|opt$|^opt", name))


def _is_opt_step(node: ast.AST) -> bool:
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "step"):
        return False
    receiver = node.func.value
    if _looks_like_optimizer(receiver):
        return True
    return "scaler" in dotted(receiver).lower() and bool(node.args) and _looks_like_optimizer(node.args[0])


# --------------------------------------------------------------------------- driver
SKIP_DIRS = frozenset({".venv", "venv", ".tox", ".nox", "node_modules", ".git", "build", "dist",
                       "site-packages", "__pycache__"})


def collect(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in paths:
        if root.is_file():
            out.append(root)
        else:
            out.extend(p for p in sorted(root.rglob("*.py"))
                       if not SKIP_DIRS & set(p.relative_to(root).parts[:-1]))
    return out


def suppression_ranges(tree: ast.Module) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]]]:
    """Simple-statement line ranges, and (header_start, header_end, body_end) for defs/classes."""
    simple: list[tuple[int, int]] = []
    defs: list[tuple[int, int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first = min([node.lineno] + [d.lineno for d in node.decorator_list])
            header_end = node.body[0].lineno - 1 if node.body else node.lineno
            defs.append((first, max(header_end, node.lineno), node.end_lineno or node.lineno))
        elif isinstance(node, ast.stmt) and not hasattr(node, "body"):
            simple.append((node.lineno, node.end_lineno or node.lineno))
    return simple, defs


def lint_source(source: str, path: str = "<string>") -> list[Finding]:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        sev, ref, _ = CODES["SYNTAX"]
        return [Finding(path, exc.lineno or 0, "SYNTAX", sev, f"file does not parse: {exc.msg}", ref)]
    findings = Linter(path, tree).run()
    line_codes, file_codes, missing = read_pragmas(source)
    simple, defs = suppression_ranges(tree)

    def covered(code: str, lines: range) -> bool:
        return any(code in line_codes.get(n, set()) for n in lines)

    def suppressed(f: Finding) -> bool:
        if f.code in file_codes or covered(f.code, range(f.line, f.line + 1)):
            return True
        if any(covered(f.code, range(a, b + 1)) for a, b in simple if a <= f.line <= b):
            return True
        return any(covered(f.code, range(a, h + 1)) for a, h, end in defs if a <= f.line <= end)

    kept = [f for f in findings if not suppressed(f)]
    sev, ref, msg = CODES["SUPPRESSION_NO_REASON"]
    kept += [Finding(path, line, "SUPPRESSION_NO_REASON", sev, msg, ref) for line in missing]
    return sorted(set(kept), key=lambda f: (f.file, f.line, f.code))


def lint_file(path: Path) -> list[Finding]:
    return lint_source(path.read_text(encoding="utf-8", errors="replace"), str(path))


def lint_paths(paths: list[Path], ignore: frozenset[str] = frozenset()) -> list[Finding]:
    return [f for p in collect(paths) for f in lint_file(p) if f.code not in ignore]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", type=Path, nargs="*", help="Python files or directories")
    parser.add_argument("--ignore", default="", help="comma-separated codes to suppress everywhere")
    parser.add_argument("--format", choices=("text", "json", "github"), default="text")
    parser.add_argument("--warnings-as-errors", action="store_true", help="exit 1 on warning findings too")
    parser.add_argument("--list-codes", action="store_true", help="print every code and exit")
    args = parser.parse_args(argv)

    if args.list_codes:
        for code, (severity, ref, why) in CODES.items():
            print(f"{code:24} {severity:8} {ref:40} {why}")
        return 0
    if not args.paths:
        print("lint_pytorch: give at least one file or directory", file=sys.stderr)
        return 2
    missing = [p for p in args.paths if not p.exists()]
    if missing:
        print(f"lint_pytorch: no such path: {missing[0]}", file=sys.stderr)
        return 2

    ignore = frozenset(c.strip().upper() for c in args.ignore.split(",") if c.strip())
    findings = lint_paths(args.paths, ignore)

    if args.format == "json":
        print(json.dumps([f._asdict() for f in findings], indent=2))
    elif args.format == "github":
        for f in findings:
            print(f"::{f.severity} file={f.file},line={max(f.line, 1)},title={f.code}::{f.message}")
    else:
        for f in findings:
            print(f.render())
        if findings:
            counts: dict[str, int] = {}
            for f in findings:
                counts[f.code] = counts.get(f.code, 0) + 1
            print(f"\n{len(findings)} finding(s): " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        else:
            print("clean: no findings")

    failing = [f for f in findings if f.severity == ERROR or args.warnings_as_errors]
    return 1 if failing else 0


if __name__ == "__main__":
    raise SystemExit(main())
