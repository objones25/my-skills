# Root finding, minimization, curve fitting: `scipy.optimize`, `scipy.differentiate`

Which entry point to call, which method to pass it, and what goes wrong silently. **Every number below is real output from this sandbox** (Python 3.11.15, numpy 2.4.4, scipy 1.17.1); code first, output in the fenced block after. Signatures come from `inspect.signature` on the installed library.

Every block in this file assumes:

```python
import numpy as np
import scipy.optimize as opt
from scipy.optimize import elementwise
from scipy import differentiate as diff
rng = np.random.default_rng(0)
```

Contents: Scalar root finding; `elementwise`; Systems: `root`; Minimization; Bounds and constraints; Curve fitting
and nonlinear least squares; Derivatives for optimizers; Global optimization; Gotchas; Quick reference.

## Scalar root finding: `root_scalar`

`root_scalar(f, args=(), method=None, bracket=None, fprime=None, fprime2=None, x0=None, x1=None, xtol=None, rtol=None, maxiter=None, options=None)`

**If you can bracket, bracket.** `brentq` is the default choice: it keeps bisection's guarantee, because it takes a bisection step whenever interpolation misbehaves, and runs superlinearly in practice. Use `newton`/`halley` only when no bracket exists, or `f` is expensive and exact derivatives are available.

| Method | Needs | Guarantee | Order |
|---|---|---|---|
| `bisect` | `bracket=[a,b]` with `f(a)f(b) < 0` | converges, `(b-a)/2^n` | 1 |
| `brentq`, `brenth` | same | converges | superlinear; inverse quadratic (`brentq`) or hyperbolic (`brenth`) interpolation with a bisection fallback |
| `ridder` | same | converges | 2, at 2 function calls per step |
| `toms748` | same | converges | ~2.7, cubic interpolation |
| `secant` | `x0`, `x1` (or `x0` alone) | none | 1.618 |
| `newton` | `x0`, `fprime` | none | 2, dropping to 1 at a multiple root |
| `halley` | `x0`, `fprime`, `fprime2` | none | 3 |
```python
f, p = lambda x: x - 2.0**-x, 0.641185744504986        # the root of x = 2**-x on [0, 1]
fp, fp2 = lambda x: 1 + np.log(2)*2.0**-x, lambda x: -np.log(2)**2 * 2.0**-x
runs = [(m, dict(bracket=[0, 1])) for m in ['bisect', 'brentq', 'brenth', 'ridder', 'toms748']]
runs += [('secant', dict(x0=0.0, x1=1.0)), ('newton', dict(x0=0.0, fprime=fp)),
         ('halley', dict(x0=0.0, fprime=fp, fprime2=fp2))]
for m, kw in runs:
    r = opt.root_scalar(f, method=m, **kw)
    print(f"{m:8s} {r.root:.15f} iter {r.iterations:3d} fcalls {r.function_calls:3d} err {abs(r.root-p):8.1e}")
try: opt.root_scalar(f, bracket=[2, 3], method='brentq')             # f(2), f(3) both > 0
except ValueError as e: print("no sign change ->", e)
for lbl, g, gp, x0 in [("x**3", lambda x: x**3, lambda x: 3*x**2, 1.0),
                       ("arctan(x)", np.arctan, lambda x: 1/(1+x**2), 2.0)]:
    r = opt.root_scalar(g, x0=x0, fprime=gp, method='newton')
    print(f"newton on {lbl:9s} root {r.root:10.3e} iter {r.iterations:3d} converged {r.converged} '{r.flag}'")
```
```
bisect   0.641185744503673 iter  39 fcalls  41 err  1.3e-12
brentq   0.641185744504986 iter   5 fcalls   6 err  0.0e+00
brenth   0.641185744504997 iter   6 fcalls   7 err  1.1e-14
ridder   0.641185744505983 iter   4 fcalls  10 err  1.0e-12
toms748  0.641185744504986 iter   3 fcalls   7 err  0.0e+00
secant   0.641185744504987 iter   5 fcalls   6 err  5.6e-16
newton   0.641185744504986 iter   4 fcalls   8 err  0.0e+00
halley   0.641185744504986 iter   3 fcalls   9 err  0.0e+00
no sign change -> f(a) and f(b) must have different signs
newton on x**3      root  2.680e-08 iter  43 converged True 'converged'
newton on arctan(x) root -7.000e+168 iter  10 converged False 'convergence error'
```
39 bisection steps against 5 Brent steps for the same accuracy on the same bracket: that is the argument for `brentq`. Compare `function_calls`, not `iterations`; `ridder` takes 4 iterations but 10 calls, `halley` 3 iterations but 9 (three each of `f`, `f'`, `f''`). `x**3` is the multiple-root degradation: `f'(0) = 0`, Newton contracts by 2/3 per step instead of squaring the error, and 43 iterations buy only `2.7e-08`, about the square root of what a simple root reaches. Brent is not immune either: on the symmetric bracket `[-1, 1]` it lands on `0.0` in 2 iterations only because the first secant step hits the root exactly, while on `[-1, 2]` `root_scalar(..., method='brentq')` returns `converged=False` after `maxiter=100` iterations at `2.5e-10` (the standalone `brentq` raises `RuntimeError: Failed to converge after 100 iterations.`) and `bisect` converges in 41. `arctan` is Newton diverging on a smooth function with a unique root: outside the basin `|x0| > 1.39` each tangent overshoots further, and `root_scalar` returns `converged=False` rather than raising (the standalone `optimize.newton` does raise `RuntimeError`, because it has `disp=True`). `RootResults` carries `root`, `iterations`, `function_calls`, `converged`, `flag`, `method`. Bracketing methods take `xtol` (absolute, `brentq` default `2e-12`) and `rtol` (default `4*eps = 8.88e-16`; smaller raises `ValueError: rtol too small`). See `04-floating-point.md`.

## `scipy.optimize.elementwise`: one root per grid point

- `find_root(f, init, /, *, args=(), tolerances=None, maxiter=None, callback=None)`
- `bracket_root(f, xl0, xr0=None, *, xmin=None, xmax=None, factor=None, args=(), maxiter=1000)`

`from scipy.optimize import elementwise` is required; the submodule is not an attribute of `scipy.optimize` until imported. `root_scalar` solves one problem; `find_root` solves an *array* of independent scalar problems at once, with `init` a 2-tuple of arrays `(xl, xr)`, `args` holding arrays broadcastable with them, and every result field an array carrying a per-element `status` and `nit`. `f` must be elementwise, meaning `f(x)[i] == f(x[i])`. This is the tool for a root per grid point, per pixel, per parameter value; looping `root_scalar` costs one Python-level call per point. The algorithm is Chandrupatla's, bracketing, so `init` must be a real bracket, and there is no guess-only mode. `bracket_root` finds one by expanding outward from `xl0`; pass `xmin`/`xmax` to keep it inside the domain, otherwise it searches both directions and can return a bracket you did not intend.
```python
c = np.array([1., 4., 9., 16.]); f2 = lambda x, c: x**2 - c          # (4,) x, (4,) c -> (4,)
res = elementwise.find_root(f2, (np.full_like(c, 0.5), np.full_like(c, 10.0)), args=(c,))
print("x", res.x, "success", res.success, "status", res.status)
print("nit", res.nit, "nfev", res.nfev, "f_x", res.f_x)
```
```
x [1. 2. 3. 4.] success [ True  True  True  True] status [0 0 0 0]
nit [10  8  7  8] nfev [12 10  9 10] f_x [0. 0. 0. 0.]
```
`nfev` counts *calls to `f`*, not scalar evaluations: 12 calls handled all four problems. `bracket_root(f2, np.full_like(c, 0.5), args=(c,), xmin=0.0)` returns `.bracket = ([0.5, 1.5, 2.5, 2.5], [1.5, 2.5, 4.5, 4.5])` in `nfev = [2 4 6 6]`, ready to hand straight to `find_root`.

## Systems: `root`

`root(fun, x0, args=(), method='hybr', jac=None, tol=None, callback=None, options=None)`

| Method | Jacobian | Use when |
|---|---|---|
| `hybr` (default) | optional, finite-differenced otherwise | square system, moderate `n`; MINPACK Powell hybrid, dense LU per step |
| `lm` | optional | square or overdetermined; Levenberg-Marquardt on `||F||^2` |
| `broyden1` / `broyden2` | never | no Jacobian and `n` too large for `hybr`; rank-1 secant updates of `J` / `J^-1` |
| `krylov` | matrix-free | large `n`; inexact Newton, `J v` by finite difference inside GMRES. Preconditioner via `options={'jac_options': {'inner_M': ...}}` |
| `df-sane` | never | derivative-free spectral residual, no linear algebra at all. `anderson`, `diagbroyden`, `linearmixing`, `excitingmixing` are the related fixed-point mixers |
```python
n = 20; h = 1.0/(n+1)                                   # Bratu: u'' + exp(u) = 0, u(0) = u(1) = 0
def G(u):
    uu = np.concatenate(([0.0], u, [0.0]))
    return (uu[:-2] - 2*uu[1:-1] + uu[2:])/h**2 + np.exp(u)          # (20,) -> (20,)
def Gjac(u):
    A = (np.diag(-2*np.ones(n)) + np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1))/h**2
    return A + np.diag(np.exp(u))                                    # (20,) -> (20, 20)
for label, m, kw in [('hybr', 'hybr', {}), ('hybr jac=', 'hybr', {'jac': Gjac}), ('lm', 'lm', {}),
                     ('lm jac=', 'lm', {'jac': Gjac}), ('broyden1', 'broyden1', {}),
                     ('krylov', 'krylov', {})]:
    r = opt.root(G, np.zeros(n), method=m, **kw)
    print(f"{label:10s} nfev {r.get('nfev', -1):5d} |G(x)| {np.linalg.norm(G(r.x)):9.2e}  success {r.success}")
```
```
hybr       nfev    29 |G(x)|  7.98e-12  success True
hybr jac=  nfev     9 |G(x)|  7.98e-12  success True
lm         nfev   129 |G(x)|  2.90e-14  success True
lm jac=    nfev     9 |G(x)|  2.74e-14  success True
broyden1   nfev  4206 |G(x)|  6.01e+05  success False
krylov     nfev    23 |G(x)|  1.72e-06  success True
```
An analytic `jac=` costs one call instead of `n` per Jacobian: 29 down to 9 for `hybr`, 129 down to 9 for `lm`. `broyden1` diverges here (`exp` overflows) and reports `success=False`; quasi-Newton without a decent initial `J` is fragile on stiff residuals. `krylov` arrives in 23 calls but stops at `1.7e-06`, because its inner tolerance is loose by design; `df-sane` needed 662 calls to reach `3.0e-08`. `fsolve` is the legacy wrapper for `root(method='hybr')`, returning a bare array (or a 4-tuple with `full_output=True`) instead of an `OptimizeResult`; new code uses `root`. For a fixed-point iteration `x = g(x)`, `fixed_point(func, x0, args=(), xtol=1e-08, maxiter=500, method='del2')` applies Aitken/Steffensen acceleration, but still needs `g` to be a contraction.

## Minimization: `minimize`

`minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)`. With `method=None` scipy picks by what you passed and nothing else: `SLSQP` if `constraints` is non-empty, else `L-BFGS-B` if `bounds is not None`, else `BFGS`.

| Method | Derivatives | Bounds | Constraints | Use when |
|---|---|---|---|---|
| `Nelder-Mead` | none | yes | no | `f` non-smooth or noisy, few variables, accuracy not critical |
| `Powell` | none | yes | no | derivative-free with conjugate directions; beats Nelder-Mead on smooth `f` |
| `CG` | grad | no | no | large `n`, minimal memory; more iterations than BFGS |
| `BFGS` | grad | no | no | the smooth unconstrained default; dense inverse Hessian, `O(n^2)` memory |
| `L-BFGS-B` | grad | yes | no | same, limited-memory `O(mn)`; the workhorse for large smooth problems, box bounds free |
| `TNC` | grad | yes | no | truncated Newton with box bounds; rarely better than L-BFGS-B |
| `SLSQP` | grad | yes | eq + ineq | small to medium constrained problems, the fast default for constraints |
| `trust-constr` | grad (+hess) | yes | eq + ineq | the general constrained solver; takes `LinearConstraint`/`NonlinearConstraint`, reports `constr_violation` |
| `COBYLA` | none | yes | ineq (eq accepted, approximate) | constrained with no usable gradient; linear models, low accuracy |
| `COBYQA` | none | yes | eq + ineq | derivative-free constrained, quadratic models; more accurate and costlier than COBYLA |
| `Newton-CG` | grad + hess/hessp | no | no | large `n` with the Hessian available only as a product |
| `trust-ncg`, `trust-krylov` | grad + hess/hessp | no | no | trust-region Newton-CG; converges where a line-search Newton stalls. `trust-krylov` solves the subproblem near-exactly, fewest iterations of the family |
| `dogleg` | grad + hess | no | no | requires a **positive definite** Hessian at every iterate; errors out otherwise |
```python
x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])            # 5-D Rosenbrock, minimum at all-ones
J, H = opt.rosen_der, opt.rosen_hess
for name, kw in [('Nelder-Mead', {}), ('BFGS (fd)', {}), ('BFGS jac=', dict(jac=J)),
                 ('Newton-CG', dict(jac=J, hess=H)), ('dogleg', dict(jac=J, hess=H))]:
    r = opt.minimize(opt.rosen, x0, method=name.split()[0], **kw); g = lambda k: r.get(k) or 0
    print(f"{name:14s} nfev {g('nfev'):4d} njev {g('njev'):4d} nhev {g('nhev'):3d} nit {g('nit'):4d} "
          f"fun {r.fun:9.2e} |x-1| {np.linalg.norm(r.x-1):8.2e}  {r.success}")
```
```
Nelder-Mead    nfev  243 njev    0 nhev   0 nit  141 fun  6.62e-05 |x-1| 1.62e-02  True
BFGS (fd)      nfev  180 njev   30 nhev   0 nit   25 fun  4.58e-11 |x-1| 1.35e-05  True
BFGS jac=      nfev   30 njev   30 nhev   0 nit   25 fun  4.01e-13 |x-1| 1.04e-06  True
Newton-CG      nfev   30 njev   30 nhev  21 nit   21 fun  1.86e-08 |x-1| 2.73e-04  True
dogleg         nfev    1 njev    1 nhev   1 nit    0 fun  8.48e+02 |x-1| 1.03e+00  False
```
`BFGS` with `jac=rosen_der` needs 30 function evaluations; the identical run with a finite-differenced gradient needs 180, exactly 6x, because each gradient costs `n = 5` extra calls. The factor is `n`, so it grows with the problem. **Supplying `jac` is the single highest-value change to a failing optimization**, and finite-differenced gradients cause most convergence failures: the gradient is accurate only to about `sqrt(eps) ~ 1.5e-08` relative, so near the minimum the reported gradient is noise and the line search gives up (`04-floating-point.md`). `dogleg` returned `success=False` at iteration 0 with `A linalg error occurred, such as a non-psd Hessian.`; the Rosenbrock Hessian at `x0` has eigenvalues `[-54.7, 72.0, 503.8, 1939.7, 4223.2]`, one negative. Use `trust-ncg` or `trust-krylov`, which handle indefinite Hessians. On the same problem `L-BFGS-B jac=` took 26 evaluations, `trust-krylov` 19 with 17 Hessians for `fun = 1.4e-15`, and derivative-free `Powell` 988.

## Bounds and constraints

- `Bounds(lb=-inf, ub=inf, keep_feasible=False)`
- `LinearConstraint(A, lb=-inf, ub=inf, keep_feasible=False)` for `lb <= A x <= ub`; `NonlinearConstraint(fun, lb, ub, jac='2-point', hess=None, keep_feasible=False, finite_diff_rel_step=None, finite_diff_jac_sparsity=None)`

Set `lb == ub` for an equality. The legacy form, accepted only by `SLSQP`, `COBYLA`, and `COBYQA`, is a list of dicts `{'type': 'eq'|'ineq', 'fun': ..., 'jac': ...}` where `ineq` means `fun(x) >= 0`. `bounds` also accepts a plain sequence of `(lo, hi)` pairs with `None` for infinite.
```python
bnds = opt.Bounds([0, -0.5], [1.0, 2.0])                             # 0 <= x <= 1, -0.5 <= y <= 2
lin  = opt.LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1.0], [1.0, 1.0])   # x+2y <= 1, 2x+y == 1
nlc  = opt.NonlinearConstraint(lambda x: x[0]**2 + x[1], -np.inf, 1.0)
r = opt.minimize(opt.rosen, [0.5, 0.0], method='trust-constr', jac=J, hess=H, bounds=bnds, constraints=[lin, nlc])
print("x", r.x, "fun %.6f nit %d nfev %d" % (r.fun, r.nit, r.nfev))
print("constr_violation", r.constr_violation, "optimality %.1e |" % r.optimality, r.message)
cons = [{'type': 'ineq', 'fun': lambda x: 1.0 - x[0] - 2*x[1]}, {'type': 'eq', 'fun': lambda x: 2*x[0] + x[1] - 1.0},
        {'type': 'ineq', 'fun': lambda x: 1.0 - x[0]**2 - x[1]}]     # legacy dict form
for m in ['SLSQP', 'COBYLA']:
    print(f"{m:7s}", opt.minimize(opt.rosen, [.5, 0.], method=m, constraints=cons).x)
```
```
x [0.4149456  0.17010881] fun 0.342718 nit 12 nfev 8
constr_violation 0.0 optimality 5.4e-09 | `gtol` termination condition is satisfied.
SLSQP   [0.41494475 0.1701105 ]
COBYLA  [0.41494981 0.17010037]
```
`trust-constr`, `SLSQP`, and `COBYLA` all accept an infeasible `x0` and work into the feasible set: from `x0=[5, 5]`, `trust-constr` still returned `[0.41494542, 0.17010916]`. What they do not tolerate is `keep_feasible=True` with an infeasible `x0`, or a constraint function that raises or returns `nan` outside the feasible region; for that, set `keep_feasible=True` on the bound guarding the domain and start feasible. `COBYLA` accepts `'eq'` dicts in 1.17.1 and satisfied the equality here to `5e-06`, but it works internally with inequalities, so use `SLSQP` or `COBYQA` when an equality must hold tightly. Read `r.constr_violation` (`trust-constr`) or evaluate the constraints at `r.x` yourself: `success=True` is the optimizer's termination test, not a feasibility claim.

## Curve fitting and nonlinear least squares

- `curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, check_finite=None, bounds=(-inf, inf), method=None, jac=None, *, full_output=False, nan_policy=None, **kwargs)`
- `least_squares(fun, x0, jac='2-point', bounds=(-inf, inf), method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=None, loss='linear', f_scale=1.0, diff_step=None, tr_solver=None, tr_options=None, jac_sparsity=None, max_nfev=None, verbose=0, args=(), kwargs=None, callback=None, workers=None)`

`curve_fit` is the convenience layer: it forms the residual `f(xdata, *p) - ydata` and returns `(popt, pcov)`. `least_squares` is the real interface, and the one to use when you need `loss=`, a sparse Jacobian, or control over scaling. `curve_fit` dispatches to `leastsq` (`method='lm'`) when unbounded and to `least_squares(method='trf')` when any bound is finite; `'lm'` supports neither bounds nor a non-linear loss (`ValueError: method='lm' supports only 'linear' loss function.`). `loss` reshapes the cost from `0.5*sum(r**2)` to `0.5*sum(f_scale**2 * rho(r**2/f_scale**2))`: `'linear'` is plain least squares, `'soft_l1'` a smooth L1 (`2(sqrt(1+z)-1)`), `'huber'` quadratic then linear, `'cauchy'` logarithmic, `'arctan'` bounded. **`f_scale` is the residual size above which a point counts as an outlier**, so set it to the scale of the ordinary noise instead of leaving it at `1.0`.
```python
rng = np.random.default_rng(0); model = lambda t, a, b, c: a*np.exp(-b*t) + c
t = np.linspace(0, 4, 40); y = model(t, 2.5, 1.3, 0.5) + rng.normal(0, 0.05, t.size); y[12] += 3.0
p, cov = opt.curve_fit(model, t, y, p0=[1.0, 1.0, 0.0])       # truth (2.5, 1.3, 0.5), y[12] a gross outlier
print("curve_fit", p, "stderr", np.sqrt(np.diag(cov)))
resid = lambda q: model(t, *q) - y
for nm, kw in [('linear', {}), ('soft_l1', dict(loss='soft_l1', f_scale=0.1)),
               ('cauchy', dict(loss='cauchy', f_scale=0.1))]:
    r = opt.least_squares(resid, [1.0, 1.0, 0.0], **kw)
    print(f"loss={nm:8s} {np.array2string(r.x, precision=4)} cost {r.cost:7.4f} nfev {r.nfev:3d}")
try: opt.curve_fit(model, t, y, p0=[1.0, -1.0, 0.0])
except RuntimeError as e: print("p0=[1,-1,0] ->", e)
```
```
curve_fit [2.45013289 0.9256577  0.40912518] stderr [0.27293927 0.27783178 0.19523028]
loss=linear   [2.4501 0.9257 0.4091] cost  3.9227 nfev   5
loss=soft_l1  [2.5099 1.309  0.5033] cost  0.3012 nfev  12
loss=cauchy   [2.5147 1.3277 0.5049] cost  0.0545 nfev  15
p0=[1,-1,0] -> Optimal parameters not found: Number of calls to function has reached maxfev = 800.
```
One bad point out of 40 pulls the decay rate `b` from 1.3 to 0.926, a 29% error, and inflates every standard error by an order of magnitude; `loss='soft_l1'` with `f_scale=0.1`, twice the true noise, recovers `1.309`. That is the practical content of "least squares is fragile to outliers": the quadratic penalty gives a residual of 3 sixty times the weight of a residual of 0.4, and an L1-flavored `rho` caps that influence. Standard errors are `np.sqrt(np.diag(pcov))`. `absolute_sigma=False`, the default, treats `sigma` as *relative* weights and rescales `pcov` by the reduced chi-square, so the errors depend on fit quality; `absolute_sigma=True` takes `sigma` at face value and reports errors in the data's own units, which is what you want when `sigma` is a real measurement uncertainty. Passing `sigma = 0.05` everywhere except `3.0` at the outlier gave stderr `[0.0328, 0.0382, 0.0144]` with `absolute_sigma=True` and `[0.0240, 0.0280, 0.0105]` with `False`. `pcov` comes back `inf` when a parameter is unidentifiable, with `Covariance of the parameters could not be estimated`: that is an over-parameterized model, not a numerical glitch. `p0` decides which minimum you land in: `p0=[1, 5, 0]` converged to the same answer as `[1, 1, 0]`, but `p0=[1, -1, 0]` sets `b < 0`, the model grows exponentially, `maxfev` runs out, and `curve_fit` raises `RuntimeError`. `least_squares` never raises for this, it returns `status <= 0`, so check `r.status`. `curve_fit` returns a plain `(popt, pcov)` tuple with no `.success`; wrap it in `try/except RuntimeError` and check `np.isfinite(pcov).all()`.

## Derivatives for optimizers

- `differentiate.derivative(f, x, *, args=(), tolerances=None, maxiter=10, order=8, initial_step=0.5, step_factor=2.0, step_direction=0, preserve_shape=False, callback=None)` -> `.df`; `differentiate.jacobian(f, x, *, tolerances=None, maxiter=10, order=8, initial_step=0.5, step_factor=2.0, step_direction=0)` -> `.df`; `differentiate.hessian(f, x, *, tolerances=None, maxiter=10, order=8, initial_step=0.5, step_factor=2.0)` -> `.ddf`, not `.df`
- `approx_fprime(xk, f, epsilon=1.4901161193847656e-08, *args)`, `check_grad(func, grad, x0, *args, epsilon=1.4901161193847656e-08, direction='all', rng=None, seed=None)`

`check_grad` returns the 2-norm of `approx_fprime(x0, func) - grad(x0)`. Small means consistent to finite-difference accuracy, roughly `1e-05` relative on a well-scaled problem. **Run it on every hand-written gradient before debugging anything else.**
```python
x = np.array([1.3, 0.7, 0.8])
def bad_der(v):
    g = opt.rosen_der(v); g[-1] = -g[-1]; return g                   # sign flipped on one component
print("check_grad correct:", opt.check_grad(opt.rosen, opt.rosen_der, x))
print("check_grad wrong  :", opt.check_grad(opt.rosen, bad_der, x))
r = opt.minimize(opt.rosen, x, method='BFGS', jac=bad_der)
print("minimize w/ wrong jac:", r.success, "|", r.message)
J2 = diff.jacobian(lambda v: np.array([v[0]**2 + v[1]**2 - 4.0, np.exp(v[0]) + v[1] - 1.0]), np.array([1., 1.]))
print("jacobian rows", J2.df[0].round(9), J2.df[1].round(9), "| max err %.1e" % J2.error.max())
Hn = diff.hessian(opt.rosen, x)                    # rosen already reduces over axis 0
print("hessian max |ddf - rosen_hess| %.1e nfev %d" % (np.abs(Hn.ddf - H(x)).max(), Hn.nfev.max()))
```
```
check_grad correct: 1.1684748312495379e-05
check_grad wrong  : 124.0000019073492
minimize w/ wrong jac: False | Desired error not necessarily achieved due to precision loss.
jacobian rows [2. 2.] [2.71828183 1.        ] | max err 7.1e-12
hessian max |ddf - rosen_hess| 1.3e-10 nfev 253
```
`1.17e-05` against `124` is not a judgement call. `direction='random'` projects onto one random direction and is far cheaper for large `n`; pass `rng=` for reproducibility. `jacobian` and `hessian` require `f` vectorized with the *variable index on axis 0* and broadcasting over trailing axes; a scalar-returning `f` raises `ValueError: When preserve_shape=False, the shape of the array returned by func...`. They reach `1e-10` to `1e-12`, far past `approx_fprime`, at 250+ evaluations, so they are for verification and one-off Jacobians, not an optimizer's inner loop. All three results also carry `.error`, `.nfev`, `.status`, `.success`.

## Global optimization

| Routine | Signature highlights | Use when |
|---|---|---|
| `differential_evolution(func, bounds, ..., rng=None, workers=1, polish=True)` | population, box bounds required, `constraints=` supported | rugged landscape, cheap `f`, parallelizable; the most reliable of the set |
| `basinhopping(func, x0, niter=100, T=1.0, stepsize=0.5, ..., rng=None)` | random restart wrapped around a local `minimize` | smooth `f` with many local minima; tune `T` to the typical barrier height |
| `shgo(func, bounds, constraints=None, n=100, ..., workers=1)` | simplicial homology, deterministic | low-dimensional and you want a guarantee; also returns all local minima in `xl`/`funl` |
| `dual_annealing(func, bounds, ..., rng=None)` | generalized simulated annealing plus local polish | many local minima, no structure to exploit |
| `direct(func, bounds, *, eps=..., ...)` | DIviding RECTangles, deterministic, derivative-free | small `n`, hard cap on evaluations; slow to finish |
| `brute(func, ranges, Ns=20, finish=fmin, workers=1)` | grid search, `Ns**n` points | `n <= 3`, or to produce a `p0` for a local method |

On 4-D Rosenbrock over `[-3, 3]^4`: `differential_evolution` reached `|x-1| = 4.4e-16` in 24965 evaluations, `shgo` `1.4e-05` in 192, `dual_annealing` `8.6e-06` in 8411, `basinhopping` `1.3e-05` in 10364, `direct` only `1.8e+00` in 491. Two orders of magnitude of cost for the same answer. `rng=` is the current keyword on `differential_evolution`, `basinhopping`, and `dual_annealing`; `seed=` is still accepted there but deprecated. `shgo` and `direct` are deterministic and take neither.

## Gotchas

- **Not reading `result.success` and `result.message`.** Symptom: a plausible `r.x` that is wrong. Cause: `minimize`, `root`, `least_squares`, `root_scalar`, and the `elementwise` functions return on failure rather than raising. Measured above: `dogleg` returned after 0 iterations at `fun = 8.5e+02`, `broyden1` returned `|G| = 6e+05`, Newton on `arctan` returned `-7e+168`. Fix: branch on `r.success` and print `r.message`. `RootResults` uses `converged`/`flag` instead; `least_squares` uses `status`, where `<= 0` is failure. The legacy wrappers differ: `curve_fit`, standalone `brentq`/`brenth`/`ridder`/`bisect`/`newton`, and `fixed_point` raise `RuntimeError`, while `fsolve` only warns and returns the last iterate.
- **`success=True` on a wrong answer, from unscaled parameters.** Symptom: the optimizer "converges" immediately and one parameter never moves. Cause: a gradient method uses one step length for all coordinates, so a parameter of size `1e6` beside one of size `1` receives a step of relative size `1e-06`. Measured: minimizing `(x0-1)**2 + 1e-12*(x1-1e6)**2` from the origin, BFGS returned `x = [1.0, 3.0e-06]`, `fun = 1.0`, `success=True`; substituting `x1 = 1e6*w1` gave `fun = 1.0e-16` in the same 9 evaluations. Fix: rescale so every variable is `O(1)`, or pass `x_scale=` to `least_squares` (`x_scale='jac'` solved it in 3 evaluations).
- **`bounds`/`constraints` silently ignored.** Symptom: the answer sits outside the box. Cause: only `Nelder-Mead`, `Powell`, `L-BFGS-B`, `TNC`, `COBYLA`, `COBYQA`, `SLSQP`, `trust-constr` honor `bounds`, and only the last four honor `constraints`. Passing them to `BFGS` emits `RuntimeWarning: Method BFGS cannot handle bounds.` and returns `success=True` at the unconstrained minimum. Fix: leave `method=None` so scipy chooses, or check the table above.
- **Tolerances tighter than finite differences can support.** Symptom: `Desired error not necessarily achieved due to precision loss.` and the evaluation count doubles for nothing. Measured on 5-D Rosenbrock with a finite-differenced gradient: `gtol=1e-5` succeeded in 180 evaluations, `gtol=1e-8` and `1e-10` both failed at 396, stuck at `fun = 4.6e-11`; with `jac=rosen_der`, `gtol=1e-10` succeeded in 34. Cause: `approx_fprime` is accurate to about `sqrt(eps) ~ 1.5e-08` relative. Fix: supply `jac`, or stop at `gtol=1e-5`. The same floor on the other side: `brentq(..., rtol=1e-16)` raises `ValueError: rtol too small (1e-16 < 8.88178e-16)`. See `04-floating-point.md`.
- **`curve_fit` with a bad `p0`.** Symptom: `RuntimeError: Optimal parameters not found: Number of calls to function has reached maxfev = 800`, or a converged fit through nothing. Cause: `p0` defaults to all ones, the wrong sign or the wrong order of magnitude for most real models. Measured: `p0=[1, -1, 0]` on `a*exp(-b*t)+c` inverts the decay and never recovers. Fix: derive `p0` from the data (peak height, half-life, offset), add `bounds=` encoding the physics, and confirm `pcov` is finite.
- **Newton without a bracket.** Symptom: overflow warnings, a root at `1e+168`, or a root outside your domain. Cause: quadratic convergence is local; outside the basin the iteration is chaotic. Fix: bracket first (`elementwise.bracket_root`, or a coarse sign scan) and use `brentq`. When `f'(p) = 0` at the root, Newton drops to linear and stalls near `sqrt(xtol)`: on `x**3` it took 43 iterations to reach `2.7e-08`. Use `bisect` or `toms748` for a known multiple root; `brentq` on the bracket `[-1, 2]` used all 100 iterations without converging.

## Quick reference

| Task | Call | Note |
|---|---|---|
| scalar root, bracket known | `root_scalar(f, bracket=[a, b])` | defaults to `brentq`, superlinear on simple roots; `xtol=2e-12`, `rtol=4*eps`; check `r.converged` |
| scalar root, only a guess | `root_scalar(f, x0=, fprime=)` | `newton`; returns `converged=False` rather than raising |
| root per grid point | `elementwise.find_root(f, (xl, xr), args=(c,))` | vectorized Chandrupatla; `bracket_root(f, xl0, xmin=, xmax=)` finds the bracket |
| system `F(x) = 0` | `root(F, x0, jac=)` | `hybr` default; `jac=` cut `nfev` 29 to 9 on a 20-D system. `fsolve` is the legacy wrapper. `method='krylov'` for large or matrix-free |
| unconstrained smooth min | `minimize(f, x0, jac=)` | `BFGS`, or `L-BFGS-B` for large `n`; supplying `jac` cut `nfev` 180 to 30 |
| bounds / constraints | `minimize(f, x0, jac=, bounds=Bounds(lb, ub), constraints=[LinearConstraint(A, lb, ub)])` | picks `L-BFGS-B` / `SLSQP`; `trust-constr` for `constr_violation` and Hessians |
| derivative-free min | `minimize(f, x0, method='Nelder-Mead')` | or `Powell` on smooth `f`; `COBYQA` if constrained; `minimize_scalar(fun, bracket=, bounds=)` in 1-D |
| curve fit | `curve_fit(model, x, y, p0=, sigma=, absolute_sigma=True)` | errors are `np.sqrt(np.diag(pcov))`; `inf` means unidentifiable |
| outlier-resistant fit | `least_squares(resid, x0, loss='soft_l1', f_scale=)` | set `f_scale` to the ordinary noise level; `method='lm'` rejects `loss` and `bounds`. Linear and bounded: `lsq_linear`, `nnls`, `linprog`, `milp` |
| verify a gradient | `check_grad(f, grad, x0)` | `~1e-5` is consistent, `1e+2` is a bug; `direction='random'` for large `n` |
| accurate Jacobian, Hessian | `differentiate.jacobian(f, x).df`, `differentiate.hessian(f, x).ddf` | variable index on axis 0, must broadcast over trailing axes; 250+ evaluations |
| global min | `differential_evolution(f, bounds, rng=)` | `shgo` when `n` is small and a guarantee matters; `rng=` replaces `seed=` |
