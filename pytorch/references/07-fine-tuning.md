# Fine-tuning and transfer learning

## Overview

Fine-tuning reuses a model that already learned good features on a big dataset and
adapts it to a smaller task. Two levers: which weights you let move
(`requires_grad`) and how fast each part moves (per-group learning rates). The
spectrum runs from frozen feature extraction (new head only) through full
fine-tuning to parameter-efficient methods like LoRA that train a tiny fraction of
new weights. Pick by how much data you have and how far the task is from pretraining.

## Recommended approach

Order matters: freeze and replace the head **before** constructing the optimizer, and
use the weights' own transforms.

```python
from torchvision.models import resnet50, ResNet50_Weights
import torch, torch.nn as nn

weights = ResNet50_Weights.DEFAULT              # typed enum, NOT pretrained=True
model = resnet50(weights=weights)
preprocess = weights.transforms()               # exact resize/crop/normalize for THESE weights

for p in model.parameters():
    p.requires_grad_(False)                      # freeze backbone
model.fc = nn.Linear(model.fc.in_features, num_classes)   # new head, requires_grad=True

optimizer = torch.optim.AdamW(                   # build LAST, over trainable only
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
```

Hand-rolling `Normalize(mean=[0.5, ...])` instead of `weights.transforms()` is the
classic transfer bug - features degrade silently because the input distribution is
shifted. Do not stack your own `Normalize` on top of `weights.transforms()` either;
that double-normalizes.

## API reference

| Purpose | Form | Notes |
|---------|------|-------|
| Load pretrained (vision) | `resnet50(weights=ResNet50_Weights.DEFAULT)` | `.DEFAULT` = best current; pin a specific enum (e.g. `IMAGENET1K_V2`) for reproducibility; `weights=None` = random init |
| Matching preprocessing | `weights.transforms()` | carried on the `Weights` object |
| Class names | `weights.meta["categories"]` | plus accuracy metrics in `.meta` |
| Freeze all | `for p in model.parameters(): p.requires_grad_(False)` | autograd skips, optimizer never updates |
| Freeze a subtree | `model.backbone.requires_grad_(False)` | flips the flag on every param in one call |
| Replace head | `model.fc = nn.Linear(in_feat, n_classes)` | attribute name varies (see below) |
| Optimize trainable only | `filter(lambda p: p.requires_grad, model.parameters())` | build the optimizer over this |
| Discriminative LR | param groups with per-group `lr` | 10-100x head versus backbone |
| Add unfrozen params | `optimizer.add_param_group({"params": ..., "lr": ...})` | mid-training |
| LoRA / PEFT | `peft.get_peft_model(base, LoraConfig(...))` | freeze base, train adapters |
| Partial state load | `model.load_state_dict(sd, strict=False)` | returns missing/unexpected key lists - check them |

Head attribute name by architecture: `fc` (ResNet), `classifier` (VGG, DenseNet,
EfficientNet - a `Sequential`, so replace `classifier[-1]`), `heads.head` (ViT). Read
`in_features` off the existing layer rather than hard-coding it.

Strategy by data size and task similarity:

| Your data | Task versus pretraining | Strategy |
|-----------|-------------------------|----------|
| Small | Similar | freeze backbone, train new head only |
| Small | Different | freeze early layers, fine-tune later layers plus head |
| Large | Similar | fine-tune all, low LR |
| Large | Different | fine-tune all, or train from scratch if truly huge |

## Walkthrough

What freezing actually buys, then the LoRA parameter budget. Output executed on
torch 2.15.0 / torchvision 0.29.2.

**1. Freeze the backbone and swap the head.** Only the new head trains.

```python
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

m = resnet18(weights=None)                       # weights=None so this runs offline
total = sum(p.numel() for p in m.parameters())
for p in m.parameters():
    p.requires_grad_(False)
m.fc = nn.Linear(m.fc.in_features, 10)           # new head is trainable by default
trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"total {total:,} | trainable {trainable:,} ({100*trainable/total:.3f}%)")
print("DEFAULT weights enum:", ResNet18_Weights.DEFAULT)
```
```
total 11,689,512 | trainable 5,130 (0.044%)
DEFAULT weights enum: ResNet18_Weights.IMAGENET1K_V1
```

`5,130 = 512 * 10 + 10` - the new `Linear(512, 10)` and its bias, nothing else.

**2. LoRA freezes `W` and trains a rank-r update.** With `B` of shape `(d, r)` and
`A` of shape `(r, k)`, the forward becomes `h = Wx + (alpha/r) * (B @ A) x`.

```python
d = k = 4096
full = d * k
r = 8
lora = r * d + r * k          # A (r x k) plus B (d x r)
print(f"full {full:,} | lora r=8 {lora:,} | {100*lora/full:.2f}% | {full/lora:.0f}x fewer")
```
```
full 16,777,216 | lora r=8 65,536 | 0.39% | 256x fewer
```

You train 0.39% of the layer and still capture most of the adaptation, because the
*change* a model needs for a new task has low intrinsic rank - it lives in a
low-dimensional subspace even though the weights are high-dimensional (Hu et al.,
2021). Since the base stays frozen, one base model serves many swappable per-task
adapters. QLoRA pushes further by quantizing the frozen base to 4-bit.

In practice use Hugging Face PEFT rather than hand-rolling:

```python
from peft import LoraConfig, get_peft_model
config = LoraConfig(r=8, lora_alpha=16, target_modules="all-linear",  # MLP matters most
                    lora_dropout=0.05, task_type="CAUSAL_LM")
model = get_peft_model(base_model, config)
model.print_trainable_parameters()               # e.g. "trainable: 0.2% of all params"
```

## Gotchas

- **Use `weights=Enum.DEFAULT`, not `pretrained=True`** (deprecated since torchvision
  0.13, still accepted with a warning in 0.29), and apply
  `weights.transforms()` rather than a hand-rolled `Normalize`.
- **Freeze and replace the head before building the optimizer**, over
  `filter(lambda p: p.requires_grad, ...)`. Building first, then freezing, leaves the
  optimizer holding frozen tensors.
- **Head LR much greater than backbone LR** (10-100x) via param groups, or you either
  wreck pretrained features or starve the new head.
- **Full fine-tuning LRs are small** - `1e-5` to `5e-5` is typical. LoRA wants roughly
  10x the full fine-tuning LR ("LoRA Without Regret", Thinking Machines 2025, a vendor
  blog; the vendor's Tinker docs say 20-100x for some model sizes).
- **Catastrophic forgetting** from too-high LR or too many epochs on small data.
  Defend with freezing, low LR, gradual unfreeze, or LoRA, which forgets less than
  full fine-tuning but is not immune: LoRA runs that learn as much forget about as much
  (Biderman et al., TMLR 2024).
- **Newly unfrozen layers need `optimizer.add_param_group`** to actually be optimized.
- **BatchNorm keeps updating its running stats even when frozen**, because those are
  buffers, not parameters. If the backbone must stay fixed, put its BN layers in
  `eval()` mode as well - `requires_grad_(False)` alone does not do it.
- **`load_state_dict(..., strict=False)` returns a named tuple** of missing and
  unexpected keys. Ignoring it hides a head that never loaded.
- **`.DEFAULT` can change between torchvision releases.** Pin the explicit enum when
  a result must reproduce.

## Alternatives

**Discriminative learning rates.** Give the pretrained backbone a small LR and the
fresh head a larger one - the backbone sits near a good solution, so big steps cause
forgetting, while the random head is far from any solution and must catch up.

```python
optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-5},   # pretrained: tiny steps
    {"params": model.fc.parameters(),       "lr": 1e-3},   # new head: 100x faster
])
```

**Gradual unfreezing.** Train the head first, then unfreeze deeper blocks, adding
each to the optimizer at a low LR - this avoids large early gradients from the random
head corrupting pretrained features.

```python
model.backbone.requires_grad_(False); train(epochs=3)      # phase 1: head only
model.backbone.layer4.requires_grad_(True)                  # phase 2: last block
optimizer.add_param_group({"params": model.backbone.layer4.parameters(), "lr": 1e-5})
train(epochs=3)
```

Freezing saves compute and memory. A frozen parameter gets no gradient, so the optimizer
skips it and never allocates its Adam moments. A fully frozen prefix (the early layers,
with inputs that do not require grad) records no graph at all: its output has
`requires_grad=False` and `grad_fn=None`, so backward stops at the first trainable layer
and costs nothing in the frozen part. Autograd goes back *through* frozen layers only when
something trainable sits upstream of them, as with adapters (LoRA) in early layers.
Caching the frozen prefix's features under `inference_mode` saves its *forward* pass on
later epochs, not backward work.

For text and LLMs the pretrained-load equivalent is Hugging Face:
`AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")`. Full
fine-tuning of a large model is usually memory-bound rather than compute-bound, so
combine LoRA with gradient checkpointing (`13-performance-profiling.md`) and bf16
autocast before reaching for more GPUs.

## Related

- `05-data-training-loop.md` - fine-tuning is that loop with parameters selectively frozen.
- `02-tensors-autograd.md` - `requires_grad` and leaf tensors, the freezing mechanism.
- `04-losses-optimizers.md` - param groups.
- `08-transforms-augmentation.md` - `weights.transforms()` and eval pipelines.
- `13-performance-profiling.md` - memory reduction when full fine-tuning does not fit.

## Sources

- `inspect` and live calls against torch 2.15.0 / torchvision 0.29.2
  (`requires_grad_`, `Optimizer.add_param_group`, `ResNet18_Weights.DEFAULT`).
- torchvision models docs - `Weights` enums, `weights.transforms()`.
- Hu et al. (2021), "LoRA: Low-Rank Adaptation of Large Language Models,"
  arXiv:2106.09685; Hugging Face PEFT docs.
- Biderman et al. (2024), "LoRA Learns Less and Forgets Less," TMLR, arXiv:2405.09673.
- Schulman et al. (2025), "LoRA Without Regret," thinkingmachines.ai/blog/lora (vendor
  blog, not peer reviewed): ~10x LR, all layers, small effective batch.
