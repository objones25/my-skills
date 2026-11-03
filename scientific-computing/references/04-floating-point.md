# Floating point, conditioning, and tolerances

The numerical-analysis layer under every other file: what float64 stores, how to compare and sum floats in numpy, how to measure conditioning and convergence order instead of asserting them, and what the `rtol`/`atol`/`xtol` knobs in scipy actually control. **Every number below is real stdout from this sandbox** (Python 3.11.15, numpy 2.4.4, scipy 1.17.1); code first, output in the fenced block after it.

## What `np.finfo` tells you

```python
import numpy as np
from decimal import Decimal
print(np.finfo(np.float64)); f32 = np.finfo(np.float32)
print(f32.eps, f32.resolution, f32.max, f32.tiny, f32.smallest_normal, f32.precision)
print(np.spacing(1.0), np.spacing(1e10), np.spacing(1e-10), np.spacing(1e10)/1e10)
print(0.1+0.2 == 0.3, repr(0.1+0.2), Decimal(0.1))
```

```
Machine parameters for float64
---------------------------------------------------------------
precision = 15   resolution = 1e-15
machep = -52   eps =        2.220446049250313e-16
negep =  -53   epsneg =     1.1102230246251565e-16
minexp = -1022   tiny =       2.2250738585072014e-308
maxexp = 1024   max =        1.7976931348623157e+308
nexp =   11   min =        -max
smallest_normal = 2.2250738585072014e-308   smallest_subnormal = 5e-324
---------------------------------------------------------------

1.1920929e-07 1e-06 3.4028235e+38 1.1754944e-38 1.1754944e-38 6
2.220446049250313e-16 1.9073486328125e-06 1.2924697071141057e-26 1.9073486328125e-16
False 0.30000000000000004 0.1000000000000000055511151231257827021181583404541015625
```

| Field | Meaning | Use it for |
|---|---|---|
| `eps` | spacing between 1.0 and the next float | the unit of *relative* error: `2.22e-16` (float64), `1.19e-07` (float32) |
| `precision`, `resolution` | decimal digits that survive a round trip, and `10**-precision` | 15 and `1e-15`; 6 and `1e-06`. Display only; `np.isclose` does not use `resolution` |
| `max` | largest finite value | past it you get `inf` and a `RuntimeWarning`, not an exception |
| `tiny` / `smallest_normal` | smallest value with a full mantissa (identical fields) | below it you are in subnormals and *losing* mantissa bits, down to `smallest_subnormal` = `5e-324` |

`eps` is relative. The absolute gap scales with magnitude: near `1e10` consecutive floats are `1.9e-06` apart, so an absolute tolerance of `1e-8` there is below the representable resolution and can never be met, while near `1e-10` the gap is `1.3e-26`. The ratio `spacing(x)/x` is always about `eps`. `Decimal(0.1)` prints the stored value exactly, which is why `0.1+0.2` lands on `0.30000000000000004`: the literal `0.1` is not 1/10, it is that 55-digit binary fraction.

## Comparing floats

```python
import inspect, math, numpy as np, numpy.testing as npt
for f in (np.isclose, np.allclose, math.isclose, npt.assert_allclose): print(f.__name__, inspect.signature(f))
a, b = 1e-12, 5e-13                                    # b is half of a
print(np.isclose(a, b), np.isclose(a, b, atol=0), math.isclose(a, b))
x, y = np.zeros(5), np.array([1e-9, -3e-9, 2e-9, 0.0, 9e-9])
print(np.allclose(x, y), np.allclose(x, y, atol=0))
print(np.isclose(1.0, 1.1, rtol=0.1, atol=0), np.isclose(1.1, 1.0, rtol=0.1, atol=0))
```

```
isclose (a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
allclose (a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
isclose (a, b, *, rel_tol=1e-09, abs_tol=0.0)
assert_allclose (actual, desired, rtol=1e-07, atol=0, equal_nan=True, err_msg='', verbose=True, *, strict=False)
True False False
True False
True False
```

numpy's predicate is `|a - b| <= atol + rtol*|b|`. Two consequences, both visible above. **It is asymmetric:** `rtol` scales `b`, the second argument, so `isclose(a,b)` and `isclose(b,a)` disagree on the last line. `math.isclose` uses `max(|a|,|b|)` and is symmetric. **The default `atol=1e-8` swallows everything smaller than 1e-8:** a 2x error at 1e-12 and an all-garbage residual against zero both report `True`, and no choice of `rtol` fixes it, because the `atol` term alone satisfies the inequality. In tests use `np.testing.assert_allclose`, whose `atol` defaults to `0`: relative-only, no silent floor, and the failure message prints the max relative mismatch with its index. Set `atol` explicitly only for components legitimately near zero.

## Catastrophic cancellation

Subtraction of nearly equal quantities does not create error, it exposes error already in the low bits. Digits lost is `log10` of the cancellation ratio `|operand| / |result|`.

```python
import numpy as np
from decimal import Decimal as D, getcontext; getcontext().prec = 50
a, b, c = 1.0, 1e8, 1.0                                       # x^2 + 1e8 x + 1
ref = (-D(b) + (D(b)*D(b) - 4*D(a)*D(c)).sqrt()) / (2*D(a))   # small root, 50 digits
s = np.sqrt(b*b - 4*a*c)
naive, stab = (-b + s)/(2*a), 2*c/(-b - s)                    # stab: times the conjugate
rel = lambda x: float(abs(D(x) - ref)/abs(ref))
print(f"ref {float(ref):.17e}  naive {naive:.17e} ({rel(naive):.1e})  stab {stab:.17e} ({rel(stab):.1e})")
print(f"ratio |b|/|-b+s| = {abs(b)/abs(-b+s):.2e} -> {np.log10(abs(b)/abs(-b+s)):.1f} digits lost")
print(f"log(1+1e-16)={np.log(1+1e-16):.3e}  log1p={np.log1p(1e-16):.17e}  expm1={np.expm1(1e-16):.17e}")
x = np.random.default_rng(0).standard_normal(100_000) + 1e9; n = x.size
print(f"one-pass var {(np.sum(x*x) - np.sum(x)**2/n)/(n-1):.8f}   np.var(ddof=1) {np.var(x, ddof=1):.8f}")
```

```
ref -1.00000000000000002e-08  naive -7.45058059692382812e-09 (2.5e-01)  stab -1.00000000000000002e-08 (7.9e-17)
ratio |b|/|-b+s| = 6.71e+15 -> 15.8 digits lost
log(1+1e-16)=0.000e+00  log1p=9.99999999999999979e-17  expm1=9.99999999999999979e-17
one-pass var 335.54767548   np.var(ddof=1) 1.00026703
```

The naive small root is wrong in the **first** digit, 25% relative error, exactly the 15.8 digits the ratio predicts. The large root `(-b-s)/2a` never cancels, so compute that one directly and get the other from `x1*x2 = c/a`. `1 + 1e-16` rounds to exactly `1.0`, so `np.log` returns 0 and every relative digit is gone, while `log1p` and `expm1` evaluate the series directly and stay exact. The textbook one-pass variance is the same failure with a large mean, wrong by 300x here; `np.var` and `np.std` are two-pass and need no defending. Same family: `np.hypot(x, y)` for `sqrt(x*x+y*y)`, `scipy.special.logsumexp` for `log(sum(exp(...)))`, `np.arctan2(y, x)` for `arctan(y/x)`.

## Summation

`np.sum` uses pairwise summation, dropping error growth from O(n) to O(log n). A Python loop does not.

```python
import math, numpy as np
from decimal import Decimal as D, getcontext; getcontext().prec = 60
rng = np.random.default_rng(0); n = 1_000_000
x = rng.standard_normal(n) * 10.0**rng.integers(-8, 9, n)     # magnitudes 1e-8 .. 1e8
exact = sum(D(float(v)) for v in x)                           # every float64 is exact in Decimal
rel = lambda v, e=None: float(abs(D(float(v))-(e or exact))/abs(e or exact))   # e: alt reference
s = 0.0
for v in x: s += v
print(f"python loop {rel(s):.2e}   np.sum {rel(np.sum(x)):.2e}   math.fsum {rel(math.fsum(x)):.2e}")
x32 = x.astype(np.float32); e32 = sum(D(float(v)) for v in x32)
print(f"np.sum(x32) {rel(np.sum(x32), e32):.2e}   np.sum(x32, dtype=float64) {rel(np.sum(x32, dtype=np.float64), e32):.2e}")
print(x32.mean(), x32.mean(dtype=np.float64))
```

```
python loop 4.79e-14   np.sum 1.87e-16   math.fsum 4.57e-17
np.sum(x32) 5.64e-08   np.sum(x32, dtype=float64) 4.48e-16
-32787.285 -32787.28416630719
```

256x from pairwise alone; `math.fsum` (Shewchuk exact accumulation, correctly rounded, 1-D, ~10x slower) buys another 4x and is the fallback when float64 pairwise is not enough. **Reduce in float64 even when you store float32.** `dtype=np.float64` on `sum`/`mean`/`var`/`dot`/`cumsum` leaves the array in float32 and costs no memory, and recovered 8 orders of magnitude here. The float32 errors are measured against the exact sum of the *stored float32 values*, so they are pure accumulation loss, not conversion loss.

## Conditioning, concretely

`cond(A) = ||A|| ||A^-1||` bounds the relative error of `solve(A, b)` at roughly `cond(A) * eps`: you lose about `log10(cond)` of your 16 digits.

```python
import numpy as np, scipy.linalg as sla
print(f"{'n':>3} {'cond2(H)':>12} {'rel err x':>11} {'rel resid':>11} {'lost digits':>11}")
for n in [4, 6, 8, 10, 12, 14, 16]:
    H = sla.hilbert(n); x_true = np.ones(n); b = H @ x_true      # exact rhs for x_true = 1
    x = sla.solve(H, b); k = np.linalg.cond(H)
    err = np.linalg.norm(x-x_true)/np.linalg.norm(x_true)
    res = np.linalg.norm(b - H@x)/(np.linalg.norm(H)*np.linalg.norm(x))
    print(f"{n:>3} {k:12.3e} {err:11.3e} {res:11.3e} {np.log10(k):11.1f}")
eps = np.finfo(float).eps                                        # A = diag(1, eps), x_true = (1,1)
A = np.diag([1.0, eps]); xt = np.ones(2); b = A @ xt; xb = np.array([1.0, 0.0])
print(np.linalg.cond(A), np.linalg.norm(xb-xt)/np.linalg.norm(xt), np.linalg.norm(b-A@xb)/np.linalg.norm(b))
```

```
  n     cond2(H)   rel err x   rel resid lost digits
  4    1.551e+04   1.055e-13   0.000e+00         4.2
  6    1.495e+07   1.910e-10   3.916e-17         7.2
  8    1.526e+10   9.030e-08   9.670e-17        10.2
 10    1.602e+13   4.367e-05   3.933e-17        13.2
 12    1.643e+16   1.866e-01   7.076e-17        16.2
 14    3.220e+17   1.657e+00   2.709e-17        17.5
 16    2.250e+18   3.442e+00   2.291e-17        18.4
4503599627370496.0 0.7071067811865475 2.220446049250313e-16
```

Read columns 3 and 4 together. **The residual sits at machine epsilon in every row while the answer goes from 13 correct digits to none.** A backward-stable solve guarantees a small residual and nothing else; `err` tracks `cond * eps` across seven orders of magnitude. The `cond2(H)` values past `1e16` are themselves noise (an SVD of a matrix that singular in float64): numpy 2.5.4 printed `2.515e+17` and `3.109e+17` for `n = 14, 16`. From n=12 `scipy.linalg.solve` also emits `LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 2.56e-17`. Treat that as "this answer is noise", not as cosmetic.

The last line is the two-element version: `x_bad = (1, 0)` against `A = diag(1, eps)` is 71% wrong with a relative residual of 2e-16. Never validate a linear solve by its residual. Check `np.linalg.cond` (2-norm by default, an SVD; `p=1` is much cheaper) and switch to `scipy.linalg.lstsq` with an explicit `cond=` cutoff when it is large.

## Measuring convergence order

Halve `h`, take `log2` of successive error ratios. Works on any scheme and is the only honest way to claim an order.

```python
import numpy as np
from scipy.integrate import simpson
def observed_order(err_of_h, hs):
    """err_of_h(h) -> |error| at step h; hs successively halved. Returns errors and p."""
    e = np.array([err_of_h(h) for h in hs])
    return e, np.log2(e[:-1]/e[1:])
fwd, cen = (lambda h: abs((np.exp(1+h)-np.exp(1))/h - np.e)), (lambda h: abs((np.exp(1+h)-np.exp(1-h))/(2*h) - np.e))
def simp(h):
    n = int(round(1/h)) + 1; n += (n % 2 == 0)                    # odd node count on [0,1]
    xs = np.linspace(0, 1, n); return abs(simpson(np.exp(xs), x=xs) - (np.e - 1))
hs = [2.0**-k for k in range(3, 9)]
for name, g in [('forward', fwd), ('central', cen), ('simpson', simp)]:
    e, p = observed_order(g, hs)
    print(name, ' '.join(f'{v:5.2f}' for v in p), f' err at h=2^-8: {e[-1]:.3e}')
hs = np.array([10.0**-k for k in range(1, 13)]); errs = np.array([cen(h) for h in hs])   # round-off floor
print('  '.join(f'{h:.0e}:{e:.3e}' for h, e in zip(hs, errs)))
eps = np.finfo(float).eps
print(f"min at h={hs[errs.argmin()]:.0e} err={errs.min():.3e};  eps**(1/3)={eps**(1/3):.3e}  eps**(2/3)={eps**(2/3):.3e}")
```

```
forward  1.03  1.02  1.01  1.00  1.00  err at h=2^-8: 5.316e-03
central  2.00  2.00  2.00  2.00  2.00  err at h=2^-8: 6.913e-06
simpson  4.00  4.00  4.00  4.00  4.00  err at h=2^-8: 2.223e-12
1e-01:4.533e-03  1e-02:4.530e-05  1e-03:4.530e-07  1e-04:4.531e-09  1e-05:5.859e-11  1e-06:1.635e-10
1e-07:5.859e-11  1e-08:6.603e-09  1e-09:6.603e-09  1e-10:6.727e-07  1e-11:1.043e-05  1e-12:2.103e-04
min at h=1e-05 err=5.859e-11;  eps**(1/3)=6.055e-06  eps**(2/3)=3.667e-11
```

(the twelve `h:err` pairs print on one line, wrapped in two above.) An observed `p` that drifts *down* as `h` shrinks means you hit the round-off floor, not that the scheme is wrong, which is what the twelve-point sweep of the central difference shows. Exactly the coursework picture: truncation `O(h^2)` down the left, round-off `O(eps/h)` up the right, minimum at `h* ~ eps**(1/3) = 6.06e-06` with floor `eps**(2/3) = 3.67e-11`. Measured minimum `5.86e-11`. The forward difference on the same sweep bottoms out at `h=1e-8` with error `6.60e-09`, against the predicted `eps**0.5 = 1.49e-08`. **A central difference cannot give more than about 11 digits of a derivative, at any step size.**

## Derivatives without the tradeoff

```python
import inspect, numpy as np, scipy.differentiate as sd
print(inspect.signature(sd.derivative))
r = sd.derivative(np.exp, 1.0)
print(r.df, r.error, r.nfev, r.status, abs(r.df-np.e)/np.e)
for h in [1e-8, 1e-16, 1e-30]:
    print(f"h={h:.0e}  complex step {np.exp(1.0 + 1j*h).imag/h:.17e}  rel err {abs(np.exp(1.0+1j*h).imag/h-np.e)/np.e:.1e}")
print((abs(1.0+1j*1e-20)*(1.0+1j*1e-20)).imag/1e-20, " true 2.0")
```

```
(f, x, *, args=(), tolerances=None, maxiter=10, order=8, initial_step=0.5, step_factor=2.0, step_direction=0, preserve_shape=False, callback=None)
2.7182818284590224 7.144507208067807e-12 11 0 8.33193580784533e-15
h=1e-08  complex step 2.71828182845904509e+00  rel err 0.0e+00
h=1e-16  complex step 2.71828182845904509e+00  rel err 0.0e+00
h=1e-30  complex step 2.71828182845904509e+00  rel err 0.0e+00
1.0  true 2.0
```

`scipy.differentiate.derivative` runs an order-8 finite difference with Richardson-style adaptive step refinement, elementwise over an array `x`, returning a result object with `df`, an `error` estimate, `nfev`, and `status` (0 = converged). Here it reaches `8.3e-15` relative in 11 evaluations, four digits past the best hand-tuned central difference above. `scipy.misc.derivative` no longer exists; this is its replacement, with a different signature and return type. **Complex step**, `Im(f(x + ih))/h`, is exact to machine precision at *any* `h`, because nothing is subtracted: the derivative arrives in the imaginary part untouched by the real part's magnitude. `h=1e-30` gives zero relative error. Requirements: `f` analytic, and written so a complex argument flows through unbranched. `abs`, `max`, `min`, sorting, `np.where` on the input, and any `if x > 0` silently return the wrong answer, as the last line shows (1.0 instead of 2.0). Use `np.` functions, not `math.` ones, which reject complex input.

## Tolerances in scipy solvers

| Entry point | Knob | Default (`inspect.signature`, 1.17.1) | Meaning |
|---|---|---|---|
| `solve_ivp` | `rtol`, `atol` | `1e-3`, `1e-6` (from `RK45`) | per-step *local* error target `atol + rtol*|y|`, not the global error; `rtol` is clamped to `100*eps = 2.22e-14` with a warning |
| `quad` | `epsabs`, `epsrel` | `1.49e-08` each | QUADPACK stops when its error estimate meets either; `1.49e-08` is `sqrt(eps)` |
| `root_scalar` | `xtol`, `rtol` | `None`, dispatched per method | `brentq`: `xtol=2e-12`, `rtol=4*eps=8.88e-16`; smaller raises `ValueError: rtol too small` |
| `least_squares` | `ftol`, `xtol`, `gtol` | `1e-8` each | relative change in cost, in `x`, and gradient orthogonality; `None` disables one |
| `minimize` | `tol` | `None`, mapped per method | `BFGS` `gtol=1e-5`; `L-BFGS-B` `ftol=2.22e-09, gtol=1e-5`; `Nelder-Mead` `xatol=fatol=1e-4` |
| `minimize` | `eps` (BFGS) | `1.49e-08` | the finite-difference step when `jac=None`; again `sqrt(eps)` |

**Never ask a solver for a tolerance tighter than `sqrt(eps) ~ 1.5e-8` on a quantity it computes by finite differences.** That is why `quad`'s default and BFGS's `eps` are both `1.49e-08`. To go tighter, supply an analytic `jac`/`fprime` or a complex-step derivative first, then lower `gtol`/`xtol`; with a supplied Jacobian, `least_squares` at `1e-14` and `brentq` at `4*eps` are attainable. `solve_ivp` controls local error per step, and global error accumulates, so the answer is typically 1 to 3 digits worse than `rtol` suggests: measured on `y' = -y` over `[0,1]`, `rtol=1e-10, atol=1e-12` gave a final error of `7.8e-12` in 170 function evaluations. Verify by halving `rtol` and watching the answer stop moving.

## Gotchas

- **`==` on computed floats.** Symptom: a branch never fires, a loop never terminates, a test fails by 1e-17. Cause: `0.1+0.2` is `0.30000000000000004`; equality is only meaningful for exactly representable values (small integers, halves, powers of two). Fix: `np.isclose`/`assert_allclose` with tolerances you chose, or `abs(a-b) <= tol` with an explicit `tol`. Exceptions: `== 0.0` is fine when you mean exactly zero, and `np.isnan`/`np.isinf` are the only correct tests for those, since `x == np.nan` is always `False`.
- **`np.allclose`'s `atol=1e-8` masking a total failure.** Symptom: a test passes on garbage. Cause: the predicate is `|a-b| <= atol + rtol*|b|`, so for data near 1e-12 the `atol` term alone always satisfies it. Measured: `np.allclose(np.zeros(5), [1e-9,-3e-9,2e-9,0,9e-9])` is `True`. Fix: `np.testing.assert_allclose`, whose `atol` defaults to `0`, setting `atol` only for components genuinely near zero. Remember `rtol` scales the second argument, so the call is asymmetric; `math.isclose` uses `max(|a|,|b|)` and is not.
- **Accumulating in float32.** Symptom: a running sum or mean drifts, and taking more, smaller steps makes it worse. Cause: float32 `eps` is `1.19e-07`, and `16777216 + 1 == 16777216` in float32 (spacing there is `2.0`). A Python loop adding `1e-3` a million times in float32 gives `991.14`, 0.9% low; float64 is off by `1.7e-11`. Fix: store float32 if you must, but pass `dtype=np.float64` to every reduction (`sum`, `mean`, `var`, `cumsum`, `dot`). Above, that cut the error from `5.6e-08` to `4.5e-16`.
- **`np.arange` with a float step.** Symptom: one element too many or too few, or the excluded endpoint appears. Cause: the length is `ceil((stop-start)/step)` computed in floating point, so representation error changes the count. Measured: `np.arange(1, 1.3, 0.1)` returns `[1. 1.1 1.2 1.3]`, four elements including `stop`. Fix: `np.linspace(start, stop, num)`, which builds each point from its index and hits the endpoint exactly (`np.linspace(0,1,11)[-1] == 1.0` is `True`).
- **Asking `solve_ivp` for `rtol` below ~1e-13.** Symptom: `UserWarning: At least one element of rtol is too small. Setting rtol = np.maximum(rtol, 2.220446049250313e-14)`, and the step count climbs for no accuracy gain. Cause: the embedded error estimate is itself computed in float64, so below `100*eps` you are controlling noise. Fix: stop at `rtol=1e-12`; for more accuracy switch method (`DOP853` for smooth non-stiff, `Radau`/`BDF` with an analytic `jac` for stiff) rather than tightening the tolerance.
- **Trusting a small residual.** Symptom: `norm(A@x - b)` is 1e-16 and the answer is still wrong. Cause: backward stability bounds the residual, not the error, which is bounded by `cond(A)*eps`. Measured: `diag(1, eps)` gives a 2e-16 relative residual on an answer 71% wrong. Fix: check `np.linalg.cond`, take `LinAlgWarning` from `scipy.linalg.solve` seriously, use `lstsq` with an explicit `cond` when near-singular.

## Quick reference

| Task | Call | Note |
|---|---|---|
| machine epsilon, gap at a value | `np.finfo(np.float64).eps`, `np.spacing(x)` | `2.220446049250313e-16`; float32 `1.19e-07`; `np.nextafter(x, y)` for the neighbour; `decimal.Decimal(x)` for the exact stored value |
| compare in a test | `np.testing.assert_allclose(actual, desired, rtol=)` | `(actual, desired, rtol=1e-07, atol=0, equal_nan=True, err_msg='', verbose=True, *, strict=False)` |
| compare in code | `np.isclose(a, b, rtol=, atol=)` | `(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)`; `|a-b| <= atol + rtol*|b|`, asymmetric; `math.isclose(a, b, *, rel_tol=1e-09, abs_tol=0.0)` is symmetric, scalars only |
| avoid cancellation | `np.log1p`, `np.expm1`, `np.hypot`, `np.arctan2`, `scipy.special.logsumexp` | mandatory for `|x| < 1e-8` in the first two |
| accurate sum | `np.sum(x)`; `math.fsum(x)`; `x32.sum(dtype=np.float64)` | pairwise O(log n) growth; `fsum` exactly rounded, 1-D, ~10x slower; `dtype=` also on `mean`, `var`, `cumsum`, `dot`, storage stays float32 |
| conditioning | `np.linalg.cond(A)` | `(x, p=None)`; 2-norm default is an SVD, `p=1` is far cheaper; then `scipy.linalg.lstsq(A, b, cond=)` truncates singular values below `cond * smax` |
| observed order, optimal step | `log2(e_h / e_{h/2})` over halved `h` | drifting down means the round-off floor; central `h* = eps**(1/3) = 6.1e-06` floor `eps**(2/3) = 3.7e-11`, forward `h* = eps**0.5 = 1.5e-08` floor `eps**0.5` |
| accurate derivative | `scipy.differentiate.derivative(f, x)` | `(f, x, *, args=(), tolerances=None, maxiter=10, order=8, initial_step=0.5, step_factor=2.0, step_direction=0, preserve_shape=False, callback=None)` |
| exact derivative, analytic `f` | `f(x + 1j*h).imag / h` with `h=1e-30` | complex step: no cancellation at all; fails on `abs`/`max`/comparisons |
| solver tolerances | `solve_ivp(rtol=, atol=)`, `quad(epsabs=, epsrel=)`, `brentq(xtol=, rtol=)` | `rtol` floors `2.22e-14` and `8.88e-16`; `quad` defaults `1.49e-08` |
