# Custom ops, autograd extensions, and hooks

Hand-written backward passes, registered operators, function transforms, hooks, and
parametrizations.

## Overview

Autograd (PyTorch's automatic differentiation engine, see `02-tensors-autograd.md`)
handles almost everything on its own. This file covers the five tools for the cases it
does not, or for when you need to watch or reshape what it does:

| Tool | What it is | Reach for it when |
|------|------------|-------------------|
| `torch.autograd.Function` | a class with a `forward` and a hand-written `backward` | autograd's own derivative is unstable, slow, or memory hungry, or the forward is not built from torch ops |
| `torch.library.custom_op` | registers a Python function as a real operator in PyTorch's dispatcher (the table that routes each op call to a kernel per device and feature) | the op must work under `torch.compile` and `torch.export`, which otherwise try to trace inside it |
| `torch.func` | function transforms: each takes a function and returns a new one (`grad` returns its gradient function, `vmap` returns a batched version) | per-sample gradients, Jacobians, ensembles, anything you would otherwise write as a Python loop over `backward()` |
| hooks | callbacks a module or tensor calls during forward or backward | capturing activations or gradients, debugging shapes, editing a gradient in flight |
| `torch.nn.utils.parametrize` | replaces a stored parameter with a function of it, recomputed on access | constraining a weight (symmetric, orthogonal, normalized) while the optimizer updates an unconstrained tensor |

Two terms recur. The **Jacobian** of `y = f(x)` is the matrix of partial derivatives
`J[i, j] = dy_i / dx_j`. A **vector-Jacobian product (VJP)** is `grad_out @ J`: given
`dL/dy` (the gradient of the loss with respect to the output), it returns `dL/dx`.
A custom `backward` is exactly a VJP. It never builds `J`; it maps `dL/dy` to `dL/dx`.

## Recommended approach

**Write `autograd.Function` in the `setup_context` style**: `forward` takes only the
inputs, and a separate `setup_context(ctx, inputs, output)` saves what `backward` needs.
The older style, `forward(ctx, *inputs)`, still works eagerly but fails under every
`torch.func` transform (see Gotchas).

```python
class SiLU(torch.autograd.Function):
    generate_vmap_rule = True                 # needed for torch.func.vmap
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])      # tensors that are inputs or outputs
    @staticmethod
    def backward(ctx, grad_out):              # one return value per forward input
        (x,) = ctx.saved_tensors
        s = torch.sigmoid(x)
        return grad_out * (s + x * s * (1 - s))

y = SiLU.apply(x)                             # call through .apply
```

Then **check it with `gradcheck` on float64 inputs** before you trust it. `gradcheck`
compares your `backward` against finite differences (nudging each input by `eps` and
measuring the change in output); float32 rounding error is larger than its default
tolerance, so float32 inputs fail even for a correct backward.

**When the op must survive `torch.compile` or `torch.export`, register it with
`torch.library.custom_op`**, give it a fake kernel (`register_fake`: computes output
shape and dtype without touching data, which is how the tracers propagate shapes), a
backward (`register_autograd`), and run `torch.library.opcheck` in your tests. The
compiler then sees one opaque node instead of trying to trace into the body.

**For per-sample gradients use `vmap(grad(loss_one))` with `functional_call`**, where
`loss_one` is the loss for a single example written without a batch dimension.

## API reference

Signatures introspected from torch 2.15.0 with `inspect.signature`; every one listed
here is identical on torch 2.14.1.

**Custom autograd**

| API | Signature / form | Notes |
|-----|------------------|-------|
| `Function.forward` | `forward(*args, **kwargs)` (setup_context style) | no `ctx` argument in this style |
| `Function.setup_context` | `setup_context(ctx, inputs, output)` | `inputs` is the tuple passed to `apply` |
| `Function.backward` | `backward(ctx, *grad_outputs)` | return one value per `forward` input; `None` for non-tensors or inputs that need no grad |
| `ctx.save_for_backward(*tensors)` | read back as `ctx.saved_tensors` | lets autograd detect an in-place change to a saved tensor; non-tensors go on `ctx` as plain attributes |
| `ctx.needs_input_grad` | tuple of bools, one per input | skip computing gradients nobody will read |
| `once_differentiable` | decorator on `backward` | runs `backward` with grad mode off; double backward raises only in some cases (see Gotchas) |
| `generate_vmap_rule` | class attribute, `True` | lets `torch.func.vmap` batch the Function automatically |
| `torch.autograd.gradcheck` | `gradcheck(func, inputs, *, eps=1e-06, atol=1e-05, rtol=0.001, raise_exception=True, nondet_tol=0.0, check_undefined_grad=True, check_grad_dtypes=False, check_batched_grad=False, check_batched_forward_grad=False, check_forward_ad=False, check_backward_ad=True, fast_mode=False, masked=None)` | returns `True`, or raises `GradcheckError` (returns `False` with `raise_exception=False`) |
| `torch.autograd.gradgradcheck` | `gradgradcheck(func, inputs, grad_outputs=None, *, eps=1e-06, atol=1e-05, rtol=0.001, gen_non_contig_grad_outputs=False, raise_exception=True, nondet_tol=0.0, check_undefined_grad=True, check_grad_dtypes=False, check_batched_grad=False, check_fwd_over_rev=False, check_rev_over_rev=True, fast_mode=False, masked=False)` | checks the second derivative (the backward of your backward) |

**Registered operators (`torch.library`)**

| API | Signature | Notes |
|-----|-----------|-------|
| `custom_op` | `custom_op(name, fn=None, /, *, mutates_args, device_types=None, schema=None, tags=None)` | `name` is `"namespace::opname"`; `mutates_args` is required (`()` for a pure op); the schema is inferred from type hints |
| `register_fake` | `register_fake(op, func=None, /, *, lib=None, _stacklevel=1, allow_override=True)` | also usable as `@myop.register_fake` |
| `register_autograd` | `register_autograd(op, backward, /, *, setup_context=None, lib=None)` | also `myop.register_autograd(backward, setup_context=...)` |
| `register_vmap` | `register_vmap(op, func=None, /, *, lib=None)` | a batching rule; without one `vmap` falls back to a slow loop and warns |
| `opcheck` | `opcheck(op, args, kwargs=None, *, test_utils=('test_schema', 'test_autograd_registration', 'test_faketensor', 'test_aot_dispatch_dynamic'), raise_exception=True, atol=None, rtol=None)` | returns a dict of test name to `'SUCCESS'` or the exception |
| `triton_op` | `triton_op(name, fn=None, /, *, mutates_args, schema=None)` | the `custom_op` variant for ops written as Triton kernels |

**Function transforms (`torch.func`)**

| API | Signature | Notes |
|-----|-----------|-------|
| `grad` | `grad(func, argnums=0, has_aux=False)` | `func` must return a scalar |
| `grad_and_value` | `grad_and_value(func, argnums=0, has_aux=False)` | returns `(grad, value)` |
| `vmap` | `vmap(func, in_dims=0, out_dims=0, randomness='error', *, chunk_size=None)` | `in_dims=None` for an argument shared across the batch; `randomness='error'` raises on random ops such as dropout |
| `functional_call` | `functional_call(module, parameter_and_buffer_dicts, args=None, kwargs=None, *, tie_weights=True, strict=False)` | runs `module` with the given tensors in place of its own |
| `jacrev` | `jacrev(func, argnums=0, *, has_aux=False, chunk_size=None, _preallocate_and_copy=False)` | reverse mode; cheap when outputs are few |
| `jacfwd` | `jacfwd(func, argnums=0, has_aux=False, *, randomness='error')` | forward mode; cheap when inputs are few |
| others | `hessian`, `jvp`, `vjp`, `linearize`, `stack_module_state`, `functionalize` | listed by `dir(torch.func)` |

**Hooks** (all return a `torch.utils.hooks.RemovableHandle`; call `.remove()` on it)

| API | Signature | Hook receives |
|-----|-----------|---------------|
| `Module.register_forward_pre_hook` | `(hook, *, prepend=False, with_kwargs=False)` | `(module, args)`; return a new `args` tuple to replace the input |
| `Module.register_forward_hook` | `(hook, *, prepend=False, with_kwargs=False, always_call=False)` | `(module, args, output)`; return a value to replace the output |
| `Module.register_full_backward_pre_hook` | `(hook, prepend=False)` | `(module, grad_output)` |
| `Module.register_full_backward_hook` | `(hook, prepend=False)` | `(module, grad_input, grad_output)`; `grad_input` is with respect to the module's *inputs*, not its parameters |
| `Tensor.register_hook` | `(hook)` | the tensor's gradient; return a tensor to replace it |

**Parametrizations**

| API | Signature |
|-----|-----------|
| `parametrize.register_parametrization` | `(module, tensor_name, parametrization, *, unsafe=False)` |
| `parametrize.remove_parametrizations` | `(module, tensor_name, leave_parametrized=True)` |
| `parametrize.cached` | `()` context manager: compute each parametrized tensor once inside the block |
| `parametrizations.orthogonal` | `(module, name='weight', orthogonal_map=None, *, use_trivialization=True)` |
| `parametrizations.weight_norm` | `(module, name='weight', dim=0)` |
| `parametrizations.spectral_norm` | `(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None)` |

## Walkthrough

One function, `silu(x) = x * sigmoid(x)`, threaded through every tool. Its derivative
is `s + x * s * (1 - s)` with `s = sigmoid(x)`. Output executed on torch 2.15.0 and
identical on torch 2.14.1.

**1. A custom `autograd.Function` agrees with PyTorch's own `F.silu`.**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiLU(torch.autograd.Function):
    generate_vmap_rule = True              # lets torch.func.vmap batch it (step 6)

    @staticmethod
    def forward(x):                        # setup_context style: no ctx here
        return x * torch.sigmoid(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        (x,) = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, grad_out):           # grad_out = dL/dy, same shape as y
        (x,) = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None
        s = torch.sigmoid(x)
        return grad_out * (s + x * s * (1 - s))   # dL/dx = dL/dy * dy/dx

x = torch.tensor([-2.0, 0.0, 1.5], requires_grad=True)
y = SiLU.apply(x)
y.sum().backward()
ref = x.detach().clone().requires_grad_()
F.silu(ref).sum().backward()
print("y:     ", torch.round(y.detach(), decimals=4))
print("ours:  ", torch.round(x.grad, decimals=4))
print("F.silu:", torch.round(ref.grad, decimals=4))
```
```
y:      tensor([-0.2384,  0.0000,  1.2264])
ours:   tensor([-0.0908,  0.5000,  1.0413])
F.silu: tensor([-0.0908,  0.5000,  1.0413])
```

**2. `gradcheck` passes in float64, fails in float32, and catches a wrong backward.**
`BadSiLU` drops the `x * s * (1 - s)` term, the kind of slip that still trains, just
worse.

```python
import warnings
from torch.autograd import gradcheck, gradgradcheck

x64 = torch.randn(6, dtype=torch.float64, requires_grad=True)
print("float64 gradcheck:    ", gradcheck(SiLU.apply, (x64,)))
print("float64 gradgradcheck:", gradgradcheck(SiLU.apply, (x64,)))

x32 = x64.detach().float().requires_grad_()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    ok32 = gradcheck(SiLU.apply, (x32,), raise_exception=False)
warned = any("not a double precision" in str(w.message) for w in caught)
print("float32 gradcheck:    ", ok32, "| warned about dtype:", warned)

class BadSiLU(SiLU):
    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return grad_out * torch.sigmoid(x)
print("buggy backward:       ", gradcheck(BadSiLU.apply, (x64,), raise_exception=False))
```
```
float64 gradcheck:     True
float64 gradgradcheck: True
float32 gradcheck:     False | warned about dtype: True
buggy backward:        False
```

`gradgradcheck` passes because `backward` is itself built from differentiable torch
ops, so autograd can differentiate it a second time.

**3. `@once_differentiable` does not always raise on double backward.** It runs
`backward` with grad mode off. The documented error appears only when the incoming
`grad_out` itself requires grad; with a plain `sum()` loss the first gradient just comes
back with no graph, and a second derivative is silently unavailable.

```python
from torch.autograd.function import once_differentiable

class OnceSiLU(SiLU):
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        s = torch.sigmoid(x)
        return grad_out * (s + x * s * (1 - s))

(g,) = torch.autograd.grad(OnceSiLU.apply(x64).sum(), x64, create_graph=True)
print("loss = sum(y):     g.requires_grad =", g.requires_grad)

w = torch.randn(6, dtype=torch.float64, requires_grad=True)
(g,) = torch.autograd.grad((OnceSiLU.apply(x64) * w).sum(), x64, create_graph=True)
try:
    g.sum().backward()
except RuntimeError as e:
    print("loss = sum(y * w):", e)
```
```
loss = sum(y):     g.requires_grad = False
loss = sum(y * w): trying to differentiate twice a function that was marked with @once_differentiable
```

**4. The same math as a registered operator, with a NumPy kernel.** NumPy stands in for
any code outside PyTorch (a C extension, a vendor library). The type hints become the
op's schema, the fake kernel gives tracers the output shape, and `opcheck` runs four
consistency tests.

```python
import numpy as np

@torch.library.custom_op("pytorch_skill::silu_np", mutates_args=())
def silu_np(x: torch.Tensor) -> torch.Tensor:
    a = x.detach().cpu().numpy()
    return torch.from_numpy(a / (1 + np.exp(-a))).to(x.device)

@silu_np.register_fake
def _(x):
    return torch.empty_like(x)             # shape and dtype only, never reads values

def _setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0])

def _backward(ctx, grad_out):
    (x,) = ctx.saved_tensors
    s = torch.sigmoid(x)
    return grad_out * (s + x * s * (1 - s))

silu_np.register_autograd(_backward, setup_context=_setup)

print(torch.ops.pytorch_skill.silu_np.default._schema)
print(torch.library.opcheck(silu_np, (torch.randn(4, requires_grad=True),)))
print("gradcheck:", gradcheck(silu_np, (x64,)))
```
```
pytorch_skill::silu_np(Tensor x) -> Tensor
{'test_schema': 'SUCCESS', 'test_autograd_registration': 'SUCCESS', 'test_faketensor': 'SUCCESS', 'test_aot_dispatch_dynamic': 'SUCCESS'}
gradcheck: True
```

**5. Why register: `torch.export` keeps the op as one node, and fails on the
unregistered version.** The plain function would also silently cut the gradient at
`.detach()` in eager mode.

```python
def silu_plain(x):                         # same kernel, not registered
    a = x.detach().numpy()
    return torch.from_numpy(a / (1 + np.exp(-a)))

class Registered(nn.Module):
    def forward(self, x):
        return silu_np(x) * 2

class Plain(nn.Module):
    def forward(self, x):
        return silu_plain(x) * 2

ep = torch.export.export(Registered(), (torch.randn(3),))
print("registered:", [str(n.target) for n in ep.graph.nodes if n.op == "call_function"])
try:
    torch.export.export(Plain(), (torch.randn(3),))
except RuntimeError as e:
    print("plain:     ", str(e).splitlines()[0])
```
```
registered: ['pytorch_skill.silu_np.default', 'aten.mul.Tensor']
plain:      .numpy() is not supported for tensor subclasses.
```

`torch.compile(Plain(), fullgraph=True)` did not fail on torch 2.15.0: Dynamo (the
compiler's Python tracer) translates many NumPy calls into torch ops. It does raise on
kernels it cannot translate, for example one that loops over `x.tolist()` raised
`Unsupported Tensor.tolist() with non-integer tensor`. Registering removes the guesswork
for both tools.

**6. Per-sample gradients with `torch.func`, checked against a loop.** A per-sample
gradient is the gradient of each example's loss on its own, shape `(B, *param.shape)`,
as needed for differential privacy and gradient-noise diagnostics. `vmap` maps
`grad(loss_one)` over the batch dimension of `X` and `Y` (`in_dims=0`) while sharing
`params` (`in_dims=None`). The model uses the `SiLU` Function from step 1.

```python
from torch.func import functional_call, grad, vmap

class Act(nn.Module):
    def forward(self, h):
        return SiLU.apply(h)

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(3, 4), Act(), nn.Linear(4, 2))
X, Y = torch.randn(5, 3), torch.randint(0, 2, (5,))
params = {k: v.detach() for k, v in model.named_parameters()}

def loss_one(p, x, y):                     # one example, no batch dim
    out = functional_call(model, p, (x.unsqueeze(0),))
    return F.cross_entropy(out, y.unsqueeze(0))

per_sample = vmap(grad(loss_one), in_dims=(None, 0, 0))(params, X, Y)
print({k: tuple(v.shape) for k, v in per_sample.items()})

loop = []
for i in range(5):                         # the slow reference: one backward per example
    model.zero_grad()
    F.cross_entropy(model(X[i:i + 1]), Y[i:i + 1]).backward()
    loop.append(model[0].weight.grad.clone())
loop = torch.stack(loop)                   # (5, 4, 3)
print("matches loop:", torch.allclose(per_sample["0.weight"], loop, atol=1e-6))

model.zero_grad()
F.cross_entropy(model(X), Y).backward()    # mean-reduced batch loss
print("mean of per-sample == batch grad:",
      torch.allclose(per_sample["0.weight"].mean(0), model[0].weight.grad, atol=1e-6))
```
```
{'0.weight': (5, 4, 3), '0.bias': (5, 4), '2.weight': (5, 2, 4), '2.bias': (5, 2)}
matches loop: True
mean of per-sample == batch grad: True
```

**7. `custom_op` does not compose with `torch.func.grad`.** The error names
`autograd.Function` and asks for `setup_context`, even though `register_autograd` was
given one. Same result on torch 2.14.1. For transforms, use the `autograd.Function`.

```python
try:
    grad(lambda t: silu_np(t).sum())(torch.zeros(2))
except RuntimeError as e:
    print(str(e).split(". ")[0])
```
```
In order to use an autograd.Function with functorch transforms (vmap, grad, jvp, jacrev, ...), it must override the setup_context staticmethod
```

**8. Hooks: record activation shapes, then remove them.** A forward hook receives
`(module, args, output)`; returning `None` leaves the output alone.

```python
shapes = {}
def record(name):
    def hook(module, args, output):
        shapes[name] = tuple(output.shape)
    return hook

handles = [m.register_forward_hook(record(n)) for n, m in model.named_children()]
model(X)
print(shapes)
for h in handles:
    h.remove()
shapes.clear()
model(X)
print("after remove:", shapes)
```
```
{'0': (5, 4), '1': (5, 4), '2': (5, 2)}
after remove: {}
```

Backward-side hooks: a full backward hook on the last `Linear` sees the gradient with
respect to that layer's input `(5, 4)` and output `(5, 2)`; a tensor hook rewrites a
gradient in flight.

```python
seen = {}
h1 = model[2].register_full_backward_hook(
    lambda mod, grad_in, grad_out: seen.update(grad_in=tuple(grad_in[0].shape),
                                               grad_out=tuple(grad_out[0].shape)))
X_req = X.clone().requires_grad_()
h2 = X_req.register_hook(lambda g: g * 0)  # replace dL/dX with zeros
F.cross_entropy(model(X_req), Y).backward()
print(seen, "| dL/dX all zero:", bool((X_req.grad == 0).all()))
h1.remove()
h2.remove()
```
```
{'grad_in': (5, 4), 'grad_out': (5, 2)} | dL/dX all zero: True
```

**9. A parametrization keeps a weight symmetric.** The optimizer updates the stored
tensor `original`; every read of `lin.weight` returns `Symmetric()(original)`. Note the
renamed `state_dict` key and class.

```python
import torch.nn.utils.parametrize as P

class Symmetric(nn.Module):
    def forward(self, W):
        return W.triu() + W.triu(1).transpose(-1, -2)

lin = nn.Linear(4, 4)
P.register_parametrization(lin, "weight", Symmetric())
print("symmetric:", torch.equal(lin.weight, lin.weight.T), "| class:", type(lin).__name__)
print(sorted(lin.state_dict()))
P.remove_parametrizations(lin, "weight")   # bake in the current value
print(sorted(lin.state_dict()), "| still symmetric:", torch.equal(lin.weight, lin.weight.T))
```
```
symmetric: True | class: ParametrizedLinear
['bias', 'parametrizations.weight.original']
['bias', 'weight'] | still symmetric: True
```

## Gotchas

- **Always call `MyFunction.apply(...)`.** Calling `forward` directly skips autograd.
- **`backward` must return one value per `forward` input**, including `None` for
  non-tensor arguments. Returning too few raises `function TwoBackward returned an
  incorrect number of gradients (expected 2, got 1)` (measured on 2.15.0 and 2.14.1).
- **The old style `forward(ctx, x)` breaks under `torch.func`.** It passes `gradcheck`
  eagerly, but `torch.func.grad` raises `In order to use an autograd.Function with
  functorch transforms (vmap, grad, jvp, jacrev, ...), it must override the
  setup_context staticmethod`.
- **`vmap` over an `autograd.Function` needs `generate_vmap_rule = True`** (or a `vmap`
  staticmethod). Without it: `You tried to vmap over SiLU, but it does not have vmap
  support`.
- **`gradcheck` needs float64.** On float32 it warns (`Input #0 requires gradient and is
  not a double precision floating point or complex`) and fails a correct backward.
- **Save inputs and outputs with `ctx.save_for_backward`.** Autograd then raises `one of
  the variables needed for gradient computation has been modified by an inplace
  operation` if one was changed after forward, instead of returning a wrong gradient.
- **`@once_differentiable` can fail silently** (walkthrough step 3). If you need second
  derivatives, write a differentiable backward and verify with `gradgradcheck`. Note
  that `gradgradcheck` on a `once_differentiable` Function reports `Jacobian mismatch`,
  not the decorator's own error message.
- **`custom_op` requires `mutates_args`.** Omitting it raises `TypeError: custom_op()
  missing 1 required keyword-only argument: 'mutates_args'`. Pass `()` for a pure op,
  or the names of arguments the op writes to in place.
- **A `custom_op` may not return its input unchanged.** `return x` raises at call time;
  return `x.clone()`.
- **The fake kernel must not branch on values.** One that does `if x.sum() > 0` makes
  `opcheck` report `GuardOnDataDependentSymNode` for `test_faketensor` and
  `test_aot_dispatch_dynamic`. A missing fake kernel makes the same two tests fail with
  `RuntimeError`.
- **`custom_op` plus `torch.func.grad` fails** (step 7), and `vmap` without
  `register_vmap` runs a slow per-example fallback with `UserWarning: There is a
  performance drop because we have not yet implemented the batching rule`.
- **`vmap` refuses random ops by default** (`randomness='error'`). Put the model in
  `eval()` or pass `randomness='different'` or `'same'`.
- **Full backward hook `grad_input` is about module inputs, not parameters.** If the
  input does not require grad it is `(None,)` and PyTorch warns `Full backward hook is
  firing when gradients are computed with respect to module outputs since no inputs
  require gradients`.
- **In-place ops on the output of a module with a full backward hook raise** (`Output 0
  of BackwardHookFunction is a view and is being modified inplace`). Use out-of-place
  ops after hooked modules.
- **A forward hook that returns a value replaces the output.** A hook meant only to
  record must return `None`.
- **Hooks outlive the code that added them.** They are not in `state_dict` and keep
  firing until `.remove()` is called on the handle; remove them in a `finally` block.
- **A parametrization changes the checkpoint layout.** `weight` becomes
  `parametrizations.weight.original` and the class becomes `ParametrizedLinear`, so
  old checkpoints no longer load by name. `remove_parametrizations` bakes the value
  back into a plain `weight`.
- **`torch.nn.utils.weight_norm` is deprecated**; it warns to use
  `torch.nn.utils.parametrizations.weight_norm`.

## Alternatives

| Need | First choice | Fallback |
|------|--------------|----------|
| Different gradient for code built from torch ops | `autograd.Function` | `x + (f(x) - x).detach()` style tricks (straight-through estimators) for simple cases |
| Op that calls non-torch code and must compile/export | `torch.library.custom_op` | `torch.compiler.allow_in_graph` or `torch.compiler.disable` (a graph break) |
| Op written as a Triton kernel | `torch.library.triton_op` | `custom_op` around the launcher |
| Per-sample gradients | `vmap(grad(...))` | a loop over `backward()` (slow but obviously right, use it as the test oracle) |
| Intermediate activations | forward hooks | return them from `forward`, or `torchvision.models.feature_extraction.create_feature_extractor` (present in torchvision 0.29) |
| Weight constraints | `parametrize` | project after each `optimizer.step()` under `torch.no_grad()` |

## Related

- `02-tensors-autograd.md` - leaves, `grad_fn`, and what `backward()` does.
- `11-compile-export.md` - graph breaks and `torch.export`, the reason to register ops.
- `16-testing-pytorch-code.md` - `gradcheck` and `opcheck` as tests.
- `10-debugging.md` - gradients that are `None`, NaN, or wrong.
- `13-performance-profiling.md` - checking that a custom kernel is actually faster.

## Sources

- `inspect.signature` against torch 2.15.0 and 2.14.1 for every signature above:
  `torch.autograd.gradcheck`, `gradgradcheck`, `torch.library.custom_op`,
  `register_fake`, `register_autograd`, `register_vmap`, `opcheck`, `triton_op`,
  `torch.func.grad`, `grad_and_value`, `vmap`, `functional_call`, `jacrev`, `jacfwd`,
  the four `nn.Module.register_*_hook` methods, `Tensor.register_hook`, and
  `torch.nn.utils.parametrize` / `parametrizations`. The two versions matched exactly.
- Every error message quoted under Gotchas was produced by running the code on both
  versions; walkthrough output was executed on 2.15.0 and re-run on 2.14.1.
- Docs pages (each URL checked to resolve; the claims above come from introspection
  and runs, not from these pages):
  docs.pytorch.org/docs/2.15 - `notes/extending.html`, `notes/extending.func.html`,
  `library.html`, `generated/torch.autograd.gradcheck.gradcheck.html`, `func.html`,
  `generated/torch.nn.Module.html`, `generated/torch.nn.utils.parametrize.register_parametrization.html`.
- The `Function.backward` docstring in torch 2.15.0 (`torch/autograd/function.py`) says
  `@once_differentiable` makes double backward raise; walkthrough step 3 shows the case
  where it does not.
- docs.pytorch.org/tutorials - `advanced/custom_ops_landing_page.html`,
  `intermediate/custom_function_double_backward_tutorial.html`.
