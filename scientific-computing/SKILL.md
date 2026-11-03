---
name: scientific-computing
description: NumPy / SciPy engineering reference (verified on numpy 2.4 / scipy 1.17, drift noted for 2.5 / 1.18) with runnable numerical patterns and a symptom-indexed debugging guide. Use when writing, reviewing, or debugging scientific Python - array shapes, broadcasting, dtypes and views, vectorization and performance, floating-point accuracy and conditioning, dense and sparse linear algebra (solve/lstsq/QR/SVD/eig/Cholesky, CG/GMRES/eigsh), root finding and optimization (root_scalar/minimize/least_squares/curve_fit), quadrature (quad/simpson/fixed_quad/tanhsinh), ODE and BVP solvers (solve_ivp/solve_bvp, stiffness, events), interpolation and splines, FFT and signal filtering, and scipy.stats distributions and resampling. Triggers on np.* or scipy.* API questions, shape/dtype/broadcast errors, singular matrices, NaN or non-convergence, "why is this slow", and questions about numerical accuracy or convergence order. Not for pandas dataframe manipulation, plotting, or deep learning.
license: MIT
compatibility: The scripts/ helpers run on Python 3.10+ with numpy and scipy installed; numpy 2.4 / scipy 1.17 themselves need Python 3.11+ (numpy 2.5 / scipy 1.18 need 3.12+). Reference docs are written against numpy 2.4.4 / scipy 1.17.1 and flag which APIs need newer versions. No network access required.
metadata:
  author: owenbeckettjones
  version: 1.0.0
  numpy-version: "2.4.4"
  scipy-version: "1.17.1"
---

# Scientific Computing

## Start here

**Step 1 - pin the version before quoting any API.** NumPy 2.0 removed a long list of
aliases and changed promotion rules; SciPy renamed `tol` to `rtol` across the Krylov
solvers, replaced `random_state=` with `rng=`, and added whole modules
(`scipy.differentiate`, `scipy.optimize.elementwise`) that do not exist before 1.15. Run:

```bash
python scripts/check_env.py
```

It prints the numpy/scipy versions, the BLAS backend and thread count, float eps, and a
warning naming which APIs in these references will be absent on an older stack. If the
versions are not numpy 2.4 / scipy 1.17, also run:

```bash
python scripts/verify_api.py
```

It introspects the installed libraries and reports every signature, removal, and behavior
in these docs that has drifted. Where the script and this skill disagree, quote the script.
Drift already measured on numpy 2.5.4 / scipy 1.18.1 is listed under "Known drift" in
`references/00-module-map.md`.

If neither library is installed, say the reference targets numpy 2.4.4 / scipy 1.17.1 and
that unverified defaults may differ.

**Step 2 - route to the reference you need.** Do not read all of `references/`. Read the
one or two files the task actually touches.

| Read this | When the task is |
|-----------|------------------|
| `references/00-module-map.md` | "which module owns this", numpy vs scipy for the same job, NumPy 2.x migration, what is new in SciPy 1.15-1.17, known drift on numpy 2.5 / scipy 1.18 |
| `references/01-arrays-dtypes.md` | ndarray semantics, dtypes, views vs copies, strides and memory layout, NEP 50 promotion, NaN handling, save/load |
| `references/02-broadcasting-indexing.md` | broadcasting rules, the five kinds of indexing, ufuncs, axis and `keepdims`, `einsum`, sorting and set ops |
| `references/03-vectorization-performance.md` | "why is this slow", replacing loops, temporaries and `out=`, memory layout, BLAS threads, profiling, when to leave numpy |
| `references/04-floating-point.md` | accuracy, `eps`, cancellation, summation, condition numbers, measuring observed convergence order, choosing tolerances |
| `references/05-linalg-dense.md` | `solve`/`assume_a`, least squares, QR, SVD, `eig` vs `eigh`, Cholesky, banded and triangular solvers, norms, batching |
| `references/06-sparse-iterative.md` | sparse formats and construction, `spsolve`/`splu`, CG/GMRES/MINRES, preconditioning, `LinearOperator`, `eigsh`/`svds` |
| `references/07-optimize-roots.md` | root finding, `minimize` method selection, bounds and constraints, `least_squares`, `curve_fit`, robust loss, gradients |
| `references/08-integrate-quadrature.md` | `quad` and its singular/infinite/oscillatory cases, sampled-data rules, Gauss-Legendre, multidimensional integration |
| `references/09-integrate-ode.md` | `solve_ivp`, stiffness and method selection, tolerances, events, Jacobian sparsity, `solve_bvp` |
| `references/10-interpolate.md` | interpolation vs smoothing, cubic/PCHIP/Akima, B-splines, `PPoly`, N-D gridded and scattered, Runge's phenomenon |
| `references/11-fft-signal.md` | spectra and frequency axes, windowing and leakage, PSD, IIR/FIR filter design and application, convolution |
| `references/12-stats-random.md` | `default_rng`, distribution objects, hypothesis tests, bootstrap and permutation tests, `scipy.special`, `scipy.spatial` |
| `references/13-debugging.md` | any error message, warning, NaN, non-convergence, or wrong-but-silent answer |

Symptom-first shortcut: any traceback, warning, `nan`, or result that looks wrong goes to
`references/13-debugging.md` before anything else. Its tables map the exact message text to
a cause and a fix.

**Step 3 - write the code.** Follow the non-negotiables below, then the patterns in the
reference file.

## Non-negotiables

These hold for every task in this skill. Violating one produces either a crash or, worse,
a number that is wrong and looks fine.

1. **Never `inv(A) @ b`.** Use `scipy.linalg.solve(A, b)`, or `solve_triangular` /
   `solve_banded` / `cho_solve` when you know the structure. Forming the inverse is slower
   and less accurate, and buys nothing: `inv` raises or warns on a singular matrix exactly
   as `solve` does.
2. **Never form `A.T @ A`.** `cond(A.T @ A) = cond(A)**2`, so the normal equations throw
   away half your correct digits. Use `lstsq` or QR.
3. **Check that the solver converged.** Most scipy solvers return a non-converged result
   that looks exactly like a converged one: check `sol.success` / `sol.status` for
   `solve_ivp`, `solve_bvp`, `minimize`, `root`, `least_squares`; `r.converged` / `r.flag`
   for `root_scalar`; `info == 0` for the Krylov solvers; `status == 0` from `tanhsinh` and
   the `elementwise`/`differentiate` result objects. A few raise instead: `curve_fit`
   (`RuntimeError`, it returns a plain `(popt, pcov)` tuple), the standalone `brentq` /
   `newton` / `fixed_point` (`RuntimeError`), and `eigsh`/`eigs` (`ArpackNoConvergence`).
   `curve_fit` also returns `inf` in `pcov` with only an `OptimizeWarning` when a parameter
   is unidentifiable, so check `np.isfinite(pcov).all()`.
4. **A reported error estimate is an estimate, not a bound.** `quad`'s `abserr`,
   `solve_ivp`'s `rtol`, and every iterative tolerance are requests the library tries to
   honor. Confirm with the accuracy protocol below.
5. **Seed explicitly with `np.random.default_rng(seed)`.** Never `np.random.seed()`, which
   mutates a global that library code also touches. Pass `rng=` to scipy functions that
   take it; the legacy distribution methods (`stats.norm.rvs`, `stats.gamma.rvs`, ...)
   still take `random_state=` and raise `TypeError` on `rng=`.
6. **`==` is never how you compare floats.** `np.isclose` / `np.allclose` in code,
   `np.testing.assert_allclose` in tests. Know that `allclose`'s `atol=1e-8` default makes
   it meaningless for values near zero.
7. **Print `.shape` and `.dtype`, not just values.** Most silent wrongness in numpy is a
   `(n,)` that broadcast against a `(n,1)` into an unintended `(n,n)`, or an integer dtype
   that overflowed without a warning.
8. **Slicing gives a view, fancy indexing gives a copy.** `a[1:3] = 0` writes through to
   the parent; `a[mask][0] = 1` writes into a temporary and does nothing.
9. **Never claim a signature or default from memory.** Every value in `references/` was
   introspected from numpy 2.4.4 / scipy 1.17.1. Outside that set, check the installed
   library with `inspect.signature` rather than guessing.

## Defaults to reach for

Use these unless the task gives a reason not to. Each links to the reference that explains
the trade-off.

| Decision | Default | Reference |
|----------|---------|-----------|
| Dense square solve | `scipy.linalg.solve(A, b, assume_a='pos')` when SPD, else `'gen'` | 05 |
| Overdetermined solve | `scipy.linalg.lstsq(A, b)` | 05 |
| Eigenproblem | `eigh` if symmetric (always), `eig` only if it is not | 05 |
| Large sparse SPD system | `cg(A, b, rtol=1e-10, M=preconditioner)` | 06 |
| Few extreme eigenvalues | `eigsh(A, k=6, which='LA')` for the largest; `eigsh(A, k=6, sigma=0, which='LM')` for the ones nearest 0 (the smallest when `A` is SPD). Singular `A` (Neumann or graph Laplacian) fails to factor at `sigma=0`: use a small negative `sigma` | 06 |
| Scalar root, bracketable | `root_scalar(f, bracket=[a,b])` (Brent) | 07 |
| Unconstrained minimization | `minimize(f, x0, jac=grad)` and verify `grad` with `check_grad` | 07 |
| Nonlinear least squares | `least_squares(resid, x0, loss='soft_l1', f_scale=noise_level)` when outliers are possible | 07 |
| Integral of a callable | `quad(f, a, b)`; `np.inf` limits directly, `points=` for kinks | 08 |
| Integral of samples | `simpson(y, x=x)` | 08 |
| ODE, non-stiff | `solve_ivp(f, span, y0, rtol=1e-8, atol=1e-10)` | 09 |
| ODE, stiff or unknown | `method='LSODA'` (real `y0` only; pass `jac` as a callable), or `'BDF'` with `jac_sparsity=` for large systems; complex stiff: split into real and imaginary parts | 09 |
| Interpolating clean data | `CubicSpline`; `PchipInterpolator` if monotonicity matters | 10 |
| Smoothing noisy data | `make_smoothing_spline(x, y)` | 10 |
| Spectrum of real data | `fft.rfft(x)` with `fft.rfftfreq(N, d=1/fs)` | 11 |
| Filtering | `butter(N, fc, fs=fs, output='sos')` + `sosfiltfilt` offline | 11 |
| Random numbers | `rng = np.random.default_rng(seed)` threaded through | 12 |
| Upper-tail probability | `dist.sf(x)`, never `1 - dist.cdf(x)` | 12 |
| Uncertainty without assumptions | `stats.bootstrap((x,), stat, rng=rng)` with `stat(x, axis)`; the default `vectorized=None` detects the `axis` parameter | 12 |

## Accuracy protocol

Before reporting any number as correct, run the two checks that catch most errors.

**Halve the tolerance and re-run.** If the answer moves, it was not converged, whatever the
solver reported. This applies to every adaptive routine: `quad`, `solve_ivp`, `minimize`,
the Krylov solvers, `solve_bvp`.

**Check the conditioning before blaming the algorithm.** `np.linalg.cond(A)` of `1e12` means
a float64 result has roughly four correct digits and no change of algorithm will fix it.
That is a modeling problem, not a numerics problem. `references/04-floating-point.md` has
the digits-lost rule and the Hilbert-matrix demonstration.

When you need to establish an order of accuracy rather than a single value, measure it:
refine `h`, compute the errors, and take `log2(e_h / e_{h/2})`. The helper and three worked
cases (forward difference, central difference, Simpson) are in
`references/04-floating-point.md`, including the round-off floor that makes very small `h`
worse rather than better.

## Writing the answer

- Show real shapes and values. When numpy/scipy are installed, run the snippet and paste
  actual output rather than describing what it would print. Code blocks in `references/`
  run in file order on numpy 2.4.4 / scipy 1.17.1; timing tables and a few summary tables
  are shown without the code that produced them.
- Compute before explaining. Walk through the actual numbers and dimensions, then say what
  they mean.
- Thread one toy example through a whole explanation: same array, same shapes, start to
  finish.
- Annotate every line that changes a shape with the shape before and after.
- Name the version when an API is version-sensitive ("`rtol` replaced `tol` in the sparse
  Krylov solvers").
- Define each non-obvious term, variable, or mechanism on first use.
- When the mathematics has a known error order or convergence rate, say what it is and,
  where it is cheap, verify it numerically rather than asserting it.
