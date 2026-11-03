# Building models

`nn.Module`, the layer catalog, shape bookkeeping, containers, initialization.

## Overview

A model is an `nn.Module`: declare layers as attributes in `__init__`, describe the
data flow in `forward`, and PyTorch handles parameter registration, device moves, and
gradient tracking. The whole skill is picking the right layers, matching their
shapes, and leaving the output as raw logits so the loss stays numerically stable.

## Recommended approach

`super().__init__()` first, layers as attributes, `forward` returns raw logits, and
you call `model(x)` - never `model.forward(x)`.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=128, out_dim=10):
        super().__init__()                   # MUST precede any layer assignment
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):                    # x: (B, 1, 28, 28)
        x = self.flatten(x)                  # (B, 784)
        x = F.relu(self.fc1(x))              # (B, 128)
        x = self.drop(x)                     # (B, 128)
        return self.fc2(x)                   # (B, 10) raw logits

model = MLP()
logits = model(torch.randn(64, 1, 28, 28))   # (64, 10)
```

**Rule 1:** activations go *between* layers, never on the output. **Rule 2:** the
final layer stays linear - the loss applies its own softmax/sigmoid.

`nn.Module.__call__` wraps `forward` to run hooks, respect `training` state, and
integrate with `torch.compile`; calling `forward` directly skips all of it (it
"works" on toys, which is what makes the habit dangerous). Raw logits matter because
`CrossEntropyLoss` and `BCEWithLogitsLoss` fuse `log_softmax`/`sigmoid` with the loss
via log-sum-exp. A softmax on the head applies the squash twice and quietly breaks
training.

## API reference

Core layers, signatures introspected from torch 2.15.0.

| Layer | Signature | Input -> Output |
|-------|-----------|-----------------|
| `nn.Linear` | `(in_features, out_features, bias=True, device=None, dtype=None)` | `(*, in)` -> `(*, out)` |
| `nn.Conv2d` | `(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')` | `(B, in, H, W)` -> `(B, out, H', W')` |
| `nn.MaxPool2d` | `(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)`; `stride=None` means `kernel_size` | halves H,W with `kernel=2` |
| `nn.BatchNorm2d` | `(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)` | shape preserved |
| `nn.LayerNorm` | `(normalized_shape, eps=1e-05, elementwise_affine=True, bias=True)` | shape preserved |
| `nn.RMSNorm` | `(normalized_shape, eps=None, elementwise_affine=True)` | shape preserved; **no bias** |
| `nn.Dropout` | `(p=0.5, inplace=False)` | shape preserved |
| `nn.Embedding` | `(num_embeddings, embedding_dim, padding_idx=None, ...)` | `(B, T)` int64 -> `(B, T, dim)` |
| `nn.LSTM` / `nn.GRU` | documented as `(input_size, hidden_size, num_layers=1, batch_first=False, ...)`; the Python signature is `(*args, **kwargs)`, so `inspect` cannot check it | `(B, T, in)` -> `(B, T, hidden)` when `batch_first=True` |
| `nn.MultiheadAttention` | `(embed_dim, num_heads, dropout=0.0, bias=True, ..., batch_first=False)` | `(B, T, dim)` -> `(B, T, dim)` when `batch_first=True` |
| `nn.Flatten` | `(start_dim=1, end_dim=-1)` | collapses all but the batch dim |
| `nn.LazyLinear` | `(out_features, bias=True)` | infers `in_features` on the first forward |

Containers and construction:

| API | Form | Use |
|-----|------|-----|
| `nn.Sequential` | `nn.Sequential(layer, layer, ...)` | straight pipe, no branching |
| `nn.ModuleList` | `nn.ModuleList([...])` | registered collection you index in `forward` |
| `nn.ModuleDict` | `nn.ModuleDict({...})` | registered, keyed |
| `nn.ParameterList` / `nn.ParameterDict` | same, for bare parameters | |
| Custom param | `self.w = nn.Parameter(torch.empty(...))` | learnable tensor, auto-registered |
| Absent param | `self.register_parameter("bias", None)` | declare a param cleanly absent |
| Init in place | `model.apply(init_fn)` with `nn.init.*` | walk the tree, fill tensors |
| Inspect | `torchinfo.summary(model, input_size=(B, C, H, W))` | per-layer shapes and memory |
| Count | `sum(p.numel() for p in model.parameters())` | parameter count |

Conv output size: `H' = floor((H + 2*padding - dilation*(kernel-1) - 1) / stride) + 1`.
With `padding="same"` (string form) the spatial size is preserved for `stride=1`.

## Walkthrough

A batch through conv -> pool -> flatten -> classifier, tracking the shape at every
line. The flatten boundary is where shape bugs hide. Output executed on torch 2.15.0.

**1. A batch of images.** The batch dim rides along untouched; feature dims must line up.

```python
import torch, torch.nn as nn
x = torch.randn(64, 3, 32, 32)        # (B=64, C=3, H=32, W=32)
print("input:", tuple(x.shape))
```
```
input: (64, 3, 32, 32)
```

**2. Conv keeps H,W with `padding=1, kernel=3`; channels go 3 -> 16.**

```python
x = nn.Conv2d(3, 16, kernel_size=3, padding=1)(x)   # H' = (32+2-3)/1 + 1 = 32
print("after conv:", tuple(x.shape))
```
```
after conv: (64, 16, 32, 32)
```

**3. MaxPool halves H,W.**

```python
x = nn.MaxPool2d(2)(x)
print("after pool:", tuple(x.shape))
```
```
after pool: (64, 16, 16, 16)
```

**4. Flatten collapses everything but the batch dim.** This number is what the next
Linear's `in_features` must equal.

```python
x = nn.Flatten()(x)                   # 16 * 16 * 16 = 4096
print("after flatten:", tuple(x.shape))
```
```
after flatten: (64, 4096)
```

**5. Classifier head - `in_features` must be 4096, output is raw logits.**

```python
print("logits:", tuple(nn.Linear(4096, 10)(x).shape))
```
```
logits: (64, 10)
```

Hard-coding step 5's `in_features` wrong is the classic
`mat1 and mat2 shapes cannot be multiplied` error. Use `nn.LazyLinear(10)` to infer
it on the first forward, or print the shape once at the flatten boundary.

## Gotchas

- **Layer assigned before `super().__init__()`** raises
  `cannot assign module before Module.__init__() call` - the constructor sets up the
  registries that attribute assignment writes into.
- **Final-layer activation plus a matching loss** double-applies the squash and
  quietly breaks training. Keep the head linear.
- **A plain `list`/`dict` of layers is not registered.** Its parameters are absent
  from `model.parameters()`, so the optimizer never steps them, and they neither move
  with `.to(device)` nor save. Use `nn.ModuleList` / `nn.ModuleDict`.
- **RNN and `nn.MultiheadAttention` default to `batch_first=False`** (expecting
  `(T, B, dim)`). Set `batch_first=True` or you silently transpose batch and time.
- **First Linear after a conv stack** needs `in_features = C*H*W` at that point.
- **`nn.RMSNorm`'s `eps` defaults to `None`**, which resolves to the dtype's epsilon -
  not LayerNorm's `1e-5`. Set it from the checkpoint's config: the original LLaMA used
  `1e-6`, Llama 3.x configs use `1e-5`.
- **Call `model(x)`, not `model.forward(x)`.**
- **Custom tensors must be `nn.Parameter` or buffers** to be tracked, moved, and saved.
- **Weight tying** (`self.head.weight = self.embed.weight`) puts one tensor under two
  names. `parameters()` dedupes shared tensors, so `sum(p.numel() for p in
  model.parameters())` counts it once; `state_dict()` lists both keys, so summing over
  `state_dict()` double-counts.

## Alternatives

**Choosing a container.** `nn.Sequential` for a straight pipe;
`nn.ModuleList`/`nn.ModuleDict` for a registered collection you loop over; a full
custom `forward` whenever the flow is not a single line - residual (`x + f(x)`),
branching, multiple inputs, or reusing an output twice.

```python
class ResMLP(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(d, d) for _ in range(n)])
    def forward(self, x):
        for blk in self.blocks:
            x = x + F.relu(blk(x))        # residual needs an explicit forward
        return x
```

**Initialization.** Built-in Linear and Conv layers call `kaiming_uniform_(a=sqrt(5))`,
which works out to U(-1/sqrt(fan_in), 1/sqrt(fan_in)): std 1/sqrt(3 x fan_in), about 0.41x
He init (0.0090 vs 0.0221 at fan-in 4096). That is not He init, so override it when depth
or activation choice makes the scale matter.

| Scheme | Function | Use with |
|--------|----------|----------|
| Kaiming / He | `nn.init.kaiming_normal_` | ReLU/SiLU nets (not the layer default; call it) |
| Xavier / Glorot | `nn.init.xavier_uniform_` | tanh/sigmoid, attention projections |
| Truncated normal | `nn.init.trunc_normal_(t, std=0.02)` | transformer weights (ViT/GPT style) |
| Zeros | `nn.init.zeros_` | biases, residual-branch final layers |

Kaiming uses roughly twice Xavier's variance because ReLU zeros about half its
inputs, halving forward-signal variance; Kaiming compensates to keep activation
variance stable across layers, while Xavier assumes a symmetric activation where no
halving occurs. Wrong scale shows up as a flat loss (vanishing) or a NaN within a few
steps (exploding).

**Inspecting.** `torchinfo.summary(model, input_size=(1, 1, 28, 28))` (replaces the
unmaintained `torchsummary`) prints per-layer input/output shapes and memory.
`torch.utils.flop_counter.FlopCounterMode` gives theoretical FLOPs - see
`13-performance-profiling.md`.

## Related

- `02-tensors-autograd.md` - why `nn.Parameter` and buffers behave as they do.
- `04-losses-optimizers.md` - the raw-logits convention is what keeps losses stable.
- `09-attention-transformers.md` - the modern transformer block built from these pieces.
- `10-debugging.md` - `mat1 and mat2` errors and the ModuleList trap.

## Sources

- `inspect.signature` against torch 2.15.0 (re-checked on 2.14.1) (`nn.Linear`, `nn.Conv2d`, `nn.RMSNorm`,
  `nn.LayerNorm`, `nn.MultiheadAttention`, `nn.LazyLinear`, `nn.Flatten`).
- docs.pytorch.org/docs/2.13 - `generated/torch.nn.Module.html`, `nn.init.html`.
- `torchinfo` (PyPI) for `summary`.
