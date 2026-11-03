# Tensors and autograd

The substrate: tensor semantics, the backward engine, and the `nn` versus
`nn.functional` split.

## Overview

PyTorch is three layers stacked: a tensor library (n-dimensional arrays with
accelerator support), an autograd engine that records every op into a graph and
replays it backward for gradients, and `torch.nn`, an object-oriented wrapper that
bundles parameters with the functions that use them. Two facts stop most mid-build
doc-hunting: which submodule owns which job (see `00-module-map.md`), and the one
rule separating `torch.nn` (stateful layer objects) from `torch.nn.functional`
(stateless functions).

## Recommended approach

Every training step is these four lines. The graph is built and freed each pass.

```python
optimizer.zero_grad()            # 1. clear old grads - they ACCUMULATE
loss = loss_fn(model(x), y)      # 2. forward: builds a fresh graph
loss.backward()                  # 3. backward: walk graph, fill .grad, free graph
optimizer.step()                 # 4. apply: per the optimizer's update rule
```

When building the model: **anything with parameters or train/eval-dependent
behavior is an `nn.*` object created in `__init__`; pure math is `F.*` called in
`forward`.**

Steps 2 and 3 build and tear down the graph every iteration ("define-by-run"), which
is why ordinary Python control flow in `forward` just works. The optimizer in step 4
only reads `.grad` - it never touches the graph, which is why one optimizer works for
any model.

## API reference

| API | Form | Effect |
|-----|------|--------|
| Track a leaf | `torch.randn(n, requires_grad=True)` | records ops on this tensor |
| Compute grads | `loss.backward(retain_graph=False)` | fills leaf `.grad`, frees graph |
| Read gradient | `tensor.grad` | accumulated `dL/d(tensor)`; `None` on non-leaves |
| Keep a non-leaf grad | `tensor.retain_grad()` | for debugging intermediates |
| Mark learnable | `nn.Parameter(tensor)` | leaf with `requires_grad=True`, auto-registered by a `Module` |
| Non-grad state | `module.register_buffer(name, tensor)` | saved and moved, not trained |
| Toggle training | `p.requires_grad_(False)` / `module.requires_grad_(False)` | freeze one param or a whole subtree |
| Cut one tensor | `tensor.detach()` | view sharing storage, severed from the graph |
| Disable tracking | `torch.no_grad()` / `torch.inference_mode()` | block-scoped |
| Grad without `.grad` | `torch.autograd.grad(outputs, inputs)` | returns grads instead of accumulating |
| Find the first NaN | `torch.autograd.set_detect_anomaly(True)` | debug only, slow |

Shape and memory ops worth knowing:

| Op | Behavior |
|----|----------|
| `.view(shape)` | zero-copy; **errors** when the new shape is incompatible with the strides (common after `transpose`/`permute`) |
| `.reshape(shape)` | copies if it must; the safe default |
| `.contiguous()` | forces a compact copy |
| `.permute` / `.transpose` | zero-copy, produces non-contiguous strides |
| `.expand(shape)` | zero-copy broadcast (stride 0); read-only in practice |
| `.repeat(shape)` | allocates a real copy |
| `.squeeze(d)` / `.unsqueeze(d)` | drop / add a size-1 dim |

## Walkthrough

One weight `w`, one multiply, backward, and what accumulation does. Output executed
on torch 2.15.0.

**1. A leaf tensor that tracks grad.** Leaves are the tensors you create; only they
accumulate `.grad`.

```python
import torch
w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)                 # no grad needed on the input
print("w.is_leaf:", w.is_leaf, "| x.requires_grad:", x.requires_grad)
```
```
w.is_leaf: True | x.requires_grad: False
```

**2. Forward records the op.** The result carries a `grad_fn` naming the backward it
will run.

```python
y = w * x
print("y:", y.item(), "| grad_fn:", y.grad_fn)
```
```
y: 6.0 | grad_fn: <MulBackward0 object at 0x7f010d2462c0>
```

**3. Backward fills `.grad`.** `dy/dw = x = 3`.

```python
y.backward()
print("w.grad after 1st:", w.grad)
```
```
w.grad after 1st: tensor(3.)
```

**4. Grads accumulate - this is why you zero them.** A second backward *adds*.

```python
(w * x).backward()
print("w.grad after 2nd:", w.grad)
w.grad = None                         # what optimizer.zero_grad() does per param
```
```
w.grad after 2nd: tensor(6.)
```

Three, then six - not three then three.

**5. Turn tracking off for eval.**

```python
with torch.inference_mode():
    z = w * x
print("inference_mode:", z.requires_grad, z.grad_fn)
```
```
inference_mode: False None
```

The chain rule autograd is running: for scalar `L` and parameter theta, it computes
`dL/dtheta` by multiplying the local derivative stored at each `grad_fn` node from
output back to the leaves, summing over paths. Step 3 is the one-edge case - a single
`MulBackward0` whose local derivative is `x`.

## Gotchas

- **Gradients accumulate.** Forgetting `optimizer.zero_grad()` sums this batch's
  gradient onto the last one.
- **`backward()` twice on one graph errors** - the graph is freed after the first
  call ("Trying to backward through the graph a second time"). The usual real cause
  is a loss tensor kept alive across iterations. `retain_graph=True` only if you
  truly need it.
- **`F.dropout` and `F.batch_norm` ignore `model.eval()`**; you pass the training flag
  by hand. `F.dropout` defaults to `training=True` and keeps dropping units during
  validation. `F.batch_norm` defaults to `training=False`, the opposite failure: it never
  updates running stats during training. Use `nn.Dropout` / `nn.BatchNorm*`.
- **Plain tensor attributes are invisible** to `.to()` and `state_dict()`. Register a
  buffer.
- **`.view()` can error after a transpose**: it needs a shape the existing strides can
  express (`torch.randn(3, 4).t().view(2, 2, 3)` works, `.view(12)` does not). `.reshape()`
  is the safe default.
- **Non-leaf tensors have `.grad is None`.** Call `.retain_grad()` to inspect one.
- **In-place ops on tensors autograd needs (`x += 1`, `relu_()`) raise a version
  counter error.** Use the out-of-place form.
- **Named tensors were removed in 2.13.** `Tensor.names` no longer exists.
- **Custom backward logic belongs in `torch.autograd.Function`**, checked with
  `gradcheck` in float64; see `15-custom-ops-autograd.md`.

## Alternatives

The three "stop the gradient" tools are not interchangeable:

| Tool | Scope | Effect |
|------|-------|--------|
| `x.detach()` | one tensor | view sharing storage, cut from the graph |
| `torch.no_grad()` | a code block | new tensors in the block do not track grad |
| `torch.inference_mode()` | a code block | like `no_grad` **plus** disables view/version tracking (faster); tensors made inside cannot re-enter autograd |

Use `detach()` to block one path while tracking others (a fixed RL target:
`td = reward + gamma * q_next.detach()`), `no_grad()` for manual tensor updates you
might later re-enable grad on, and `inference_mode()` for eval and serving - fastest,
and its one restriction never bites there.

`nn.*` versus `F.*`: module form for anything with parameters or train/eval-dependent
behavior; functional form for pure ops (`F.relu`, `F.softmax`) inside `forward`.

For per-sample gradients, jacobians, or model ensembles, `torch.func` (`vmap`,
`grad`, `jacrev`, `functional_call`) is the right tool rather than a Python loop over
`backward()`.

## Related

- `03-models-layers.md` - `nn.Module` packages parameters and `forward` on top of this.
- `04-losses-optimizers.md` - `torch.optim` consumes the `.grad` autograd produces.
- `01-devices-amp.md` - buffers versus parameters decide what `.to()` moves.
- `10-debugging.md` - NaN grads, "backward a second time", grads that are `None`.
- `15-custom-ops-autograd.md` - custom `autograd.Function`, hooks, `torch.func`.

## Sources

- `inspect` against torch 2.15.0 (re-checked on 2.14.1); walkthrough output executed on the same build.
- docs.pytorch.org/docs/2.13 - `notes/autograd.html`,
  `generated/torch.nn.Module.html`, `generated/torch.inference_mode.html`,
  `tensor_view.html`.
