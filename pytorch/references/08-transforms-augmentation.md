# Data augmentation and transforms.v2

torchvision 0.29 (`torchvision.transforms.v2`), checked on 0.29.2 and 0.29.1.

## Overview

Augmentation expands a finite dataset by applying random, label-preserving
distortions each time a sample is drawn, so the model sees a slightly different image
every epoch and learns invariances instead of memorizing pixels. The modern tool is
`transforms.v2`: faster than v1, and able to transform images, masks, and boxes
together. Two rules prevent the common bugs: augment only the **training** set, and
order the pipeline geometric/color first, then `ToImage` plus `ToDtype(scale=True)`,
then `Normalize`.

## Recommended approach

Cheap uint8 ops first, convert with `scale=True`, `Normalize` last.

```python
from torchvision.transforms import v2
import torch

train_tf = v2.Compose([
    v2.RandomResizedCrop(224, antialias=True),     # geometric, on uint8
    v2.RandomHorizontalFlip(p=0.5),                # geometric, on uint8
    v2.RandAugment(),                              # auto policy, expects uint8
    v2.ToImage(),                                  # -> uint8 image tensor (C, H, W)
    v2.ToDtype(torch.float32, scale=True),         # -> float32 in [0, 1]   <-- scale=True
    v2.Normalize(mean=[0.485, 0.456, 0.406],       # ImageNet stats, LAST
                 std=[0.229, 0.224, 0.225]),
])
```

`v2.ToDtype(torch.float32)` **without** `scale=True` casts but does not divide by
255, leaving pixels in the hundreds. `Normalize` then produces garbage and the model
silently fails to learn - see the walkthrough. `scale` defaults to `False`, so this is
opt-in.

## API reference

| Purpose | Form | Notes |
|---------|------|-------|
| Import | `from torchvision.transforms import v2` | v1 (`torchvision.transforms`) is frozen |
| To tensor | `v2.ToImage()` | PIL/ndarray -> uint8 `(C, H, W)` |
| Cast and scale | `v2.ToDtype(dtype, scale=False)` | pass `scale=True` for uint8 `[0,255]` -> float `[0,1]` |
| Compose | `v2.Compose([...])` | runs in order |
| Normalize | `v2.Normalize(mean, std, inplace=False)` | `(x - mean) / std`, on float, last |
| Geometric | `RandomResizedCrop`, `RandomHorizontalFlip`, `RandomRotation`, `RandomAffine` | run on uint8, before conversion |
| Color | `ColorJitter`, `RandomGrayscale`, `GaussianBlur` | run on uint8 |
| Occlusion | `v2.RandomErasing(p=0.5)` | runs on uint8 or float (checked on 0.29.2); the fill `value` is in the tensor's units, so after `Normalize` the default `0` means the mean color |
| Auto policy | `RandAugment()`, `TrivialAugmentWide()`, `AutoAugment()` | uint8; before conversion |
| Batch mix | `v2.MixUp(num_classes=N)`, `v2.CutMix(num_classes=N)` | in `collate_fn`, emits soft labels |
| Match pretrained | `weights.transforms()` | correct resize/crop/normalize baked in |
| Multi-target | v2 transforms accept `(img, mask)`, `(img, boxes)` tuples | geometric ops applied consistently |

Augmentation catalog by the invariance it teaches:

| Transform | Randomizes | Teaches invariance to |
|-----------|-----------|-----------------------|
| `RandomResizedCrop(size)` | crop region and scale | object scale and position |
| `RandomHorizontalFlip(p=0.5)` | left/right mirror | horizontal orientation |
| `RandomRotation(degrees)` | rotation angle | small pose changes |
| `ColorJitter(...)` | brightness/contrast/saturation/hue | lighting and camera |
| `RandomGrayscale(p)` | drops color | color reliance |
| `RandomErasing(p)` | blanks a rectangle | occlusion |
| `GaussianBlur(kernel_size)` | blur | focus and sharpness |

## Walkthrough

One RGB pixel through the real `v2` pipeline, with and without `scale=True`, then the
train-versus-eval and MixUp behavior on a tiny image. Output executed on torch 2.15.0 /
torchvision 0.29.2 (identical on 2.14.1 / 0.29.1).

**1. A raw uint8 pixel through two pipelines that differ only in `scale`.**

```python
import torch
from torchvision.transforms import v2
img = torch.tensor([[[124]], [[116]], [[104]]], dtype=torch.uint8)   # (3, 1, 1) uint8
norm = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
good = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), norm])
bad  = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32), norm])
print("scale=True :", [round(v, 4) for v in good(img).flatten().tolist()])
print("scale=False:", [round(v, 2) for v in bad(img).flatten().tolist()])
```
```
scale=True : [0.0056, -0.0049, 0.0082]
scale=False: [539.37, 515.82, 460.42]
```

With `scale=True` the pixel is divided by 255 (124 becomes 0.4863), then centered near
zero. Without it, `Normalize` runs on values in the hundreds and lands about 500 standard
deviations out. No error is raised; training just fails to converge.

**2. `Normalize` on uint8 does raise**, which is the one ordering mistake that announces
itself.

```python
try:
    norm(v2.ToImage()(img))
except TypeError as e:
    print("Normalize on uint8:", e)
```
```
Normalize on uint8: Input tensor should be a float tensor. Got torch.uint8.
```

**3. Train transforms are random, eval transforms are not.** The same input through the
training pipeline twice gives two different crops; the eval pipeline is a pure function
of its input, which is what makes a validation metric comparable across epochs.

```python
torch.manual_seed(0)
x = torch.randint(0, 256, (3, 8, 8), dtype=torch.uint8)
train_tf = v2.Compose([v2.RandomResizedCrop(4, antialias=True), v2.RandomHorizontalFlip()])
a, b = train_tf(x), train_tf(x)
print("train: same input, same output twice?", torch.equal(a, b), "| dtype:", a.dtype, tuple(a.shape))
eval_tf = v2.Compose([v2.Resize(6, antialias=True), v2.CenterCrop(4)])
print("eval:  same input, same output twice?", torch.equal(eval_tf(x), eval_tf(x)))
```
```
train: same input, same output twice? False | dtype: torch.uint8 (3, 4, 4)
eval:  same input, same output twice? True
```

**4. MixUp works on a batch and returns soft labels.** Integer class labels come back as
a `(B, num_classes)` float matrix whose rows sum to 1, which `nn.CrossEntropyLoss`
accepts as probability targets.

```python
mix = v2.MixUp(num_classes=3)
imgs = torch.rand(4, 3, 4, 4); labels = torch.tensor([0, 1, 2, 0])
_, soft = mix(imgs, labels)
print("MixUp labels:", tuple(soft.shape), soft.dtype, "| rows sum to 1:", torch.allclose(soft.sum(1), torch.ones(4)))
```
```
MixUp labels: (4, 3) torch.float32 | rows sum to 1: True
```

## Gotchas

- **`ToDtype(torch.float32)` needs `scale=True`** or pixels stay in `[0, 255]`.
- **Order: geometric/color (uint8) -> `ToImage`/`ToDtype` -> `Normalize`.**
  Auto-policies go on uint8. `RandomErasing` is conventionally placed after `ToDtype`
  so its fill value is in float units.
- **Do not pass `antialias=False`** to resizing transforms (`True` is already the default
  in torchvision 0.29) - without antialiasing, downsampling
  aliases and training images differ subtly from inference images.
- **Augment train only.** The eval pipeline must be deterministic (resize,
  center-crop, normalize); reusing the train transform for validation makes every
  eval pass see different random crops, so the metric jitters and model selection
  runs on noise.
- **Flip and rotation must be label-preserving** for your task - they are not for
  digits, text, or left/right-dependent labels.
- **Normalize stats must match**: the dataset's own for from-scratch, the model's (or
  `weights.transforms()`) for fine-tuning. Do not double-normalize.
- **MixUp/CutMix are batch-level** (`collate_fn`) and emit soft labels - the loss must
  accept probability targets. `nn.CrossEntropyLoss` does.
- **Transforms run in DataLoader workers.** Heavy augmentation with `num_workers=0`
  turns the GPU idle - see `13-performance-profiling.md`.
- **`v2.Normalize` on a uint8 tensor raises**; it requires float input.

## Alternatives

**Auto-policies versus a hand-tuned Compose.** Hand-picking augmentation strengths is
a hyperparameter search you usually cannot afford. Drop in a searched policy instead:

```python
v2.RandAugment()          # N random ops at random magnitude; 2 knobs (num_ops, magnitude)
v2.TrivialAugmentWide()   # parameter-free; one random op per image - often the best default
v2.AutoAugment()          # policy searched on ImageNet/CIFAR; pass policy= for the dataset
```

`TrivialAugmentWide` removes even RandAugment's two knobs and empirically matches or
beats the more complex searched policies - a near-free accuracy win on standard image
tasks.

**Keep two pipelines.**

```python
eval_tf = v2.Compose([
    v2.Resize(256, antialias=True),
    v2.CenterCrop(224),                            # deterministic
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
```

**MixUp/CutMix go in `collate_fn`**, not the per-sample `Compose`:

```python
mix = v2.RandomChoice([v2.MixUp(num_classes=N), v2.CutMix(num_classes=N)])
def collate_fn(batch):
    return mix(*torch.utils.data.default_collate(batch))
# targets come back soft, (B, N) float
```

Why normalize at all, and why those numbers: optimization is better-conditioned when
inputs are centered and scaled, since features on wildly different scales elongate
the loss surface. The `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` values are
ImageNet's per-channel statistics, and any ImageNet-pretrained model expects exactly
these. For a from-scratch model on your own data, compute your dataset's channel
mean and std once. When fine-tuning torchvision weights, skip the guesswork and call
`weights.transforms()`.

## Related

- `05-data-training-loop.md` - the `Dataset` applies the per-sample `Compose`.
- `07-fine-tuning.md` - `weights.transforms()` supplies the correct preprocessing.
- `06-tuning-reproducibility.md` - augmentation strength is a regularization knob, and
  augmentation RNG lives in the workers.
- `04-losses-optimizers.md` - `CrossEntropyLoss` accepts the soft targets MixUp emits.

## Sources

- `inspect.signature` against torchvision 0.29.2 (`v2.ToDtype`, `v2.Normalize`);
  walkthrough output executed on torch 2.15.0 / torchvision 0.29.2 and re-run on 2.14.1 / 0.29.1.
- The earlier claim that `RandomErasing` needs a float tensor was tested and is false on
  torchvision 0.29.2: it erases a uint8 image without error.
- torchvision `transforms.v2` documentation - `ToImage`, `ToDtype`, `Normalize`,
  `RandAugment`, `TrivialAugmentWide`, `MixUp`, `CutMix`, v2 versus v1.
