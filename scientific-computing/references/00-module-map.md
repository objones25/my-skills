# Module map and modern API

Routing file: which submodule owns a problem, which reference file covers it, and the numpy 2.x / scipy 1.17 API
facts that make older code silently wrong. Read this before writing numpy or scipy code, then jump to the sibling
file. Ground truth throughout: numpy 2.4.4, scipy 1.17.1, OpenBLAS 0.3.31, 2 cores.

Contents: Where things live; numpy vs scipy for the same job; NumPy 2.x migration; SciPy modern API;
Known drift on numpy 2.5 / scipy 1.18; Array API and alternative backends; Gotchas; Quick reference.

## Where things live

| Submodule | Owns | File |
|---|---|---|
| `scipy.linalg` | Dense factorizations, structure-aware solves (`assume_a`, banded, triangular, Toeplitz), matrix functions | `05-linalg-dense.md` |
| `scipy.sparse` | Sparse storage formats (CSR/CSC/COO/DIA/LIL) and construction | `06-sparse-iterative.md` |
| `scipy.sparse.linalg` | Krylov solvers, `spsolve`, ARPACK/LOBPCG eigensolvers, `LinearOperator` | `06-sparse-iterative.md` |
| `scipy.optimize` | Minimization, least squares, curve fitting, root finding, global optimizers | `07-optimize-roots.md` |
| `scipy.differentiate` | Error-controlled finite-difference `derivative`/`jacobian`/`hessian` | `07-optimize-roots.md` |
| `scipy.integrate` (quadrature) | `quad`, `cubature`, `tanhsinh`, `nsum`, `quad_vec`, fixed-sample rules | `08-integrate-quadrature.md` |
| `scipy.integrate` (ODE) | `solve_ivp`, `solve_bvp`, events, dense output, stiff methods | `09-integrate-ode.md` |
| `scipy.interpolate` | Splines, `CubicSpline`, `RegularGridInterpolator`, `griddata`, rational `AAA` | `10-interpolate.md` |
| `scipy.fft` | DFT, real DFT, DCT/DST, `next_fast_len`, threading via `workers` | `11-fft-signal.md` |
| `scipy.signal` | Filter design, `lfilter`/`sosfilt`/`filtfilt`, convolution, spectrograms, resampling | `11-fft-signal.md` |
| `scipy.stats` | Distributions (legacy `rv_continuous` and the new `Normal` infra), hypothesis tests, resampling | `12-stats-random.md` |
| `scipy.special` | Gamma/Bessel/erf, orthogonal polynomials, log-domain helpers (`logsumexp`, `xlogy`) | `12-stats-random.md`; `04-floating-point.md` for why the log-domain versions exist |
| `scipy.spatial` | `KDTree`, `distance` metrics, `Delaunay`/`ConvexHull`/`Voronoi`, `Rotation` | No dedicated file; `10-interpolate.md` covers the Delaunay path behind `griddata` |
| `scipy.constants` | CODATA constants and unit conversions, e.g. `constants.value('speed of light in vacuum')` | No dedicated file, it is a lookup table |
| `scipy.ndimage` | N-D filtering, morphology, `map_coordinates`, labeling | `11-fft-signal.md`, shares the convolution material |

Numpy-side and cross-cutting: `01-arrays-dtypes.md`, `02-broadcasting-indexing.md`,
`03-vectorization-performance.md`, `04-floating-point.md`, `13-debugging.md`.

## numpy vs scipy for the same job

| Job | numpy | scipy | Prefer |
|---|---|---|---|
| Dense solve / factorization | `np.linalg` | `scipy.linalg` | scipy when structure is known or a driver choice matters; numpy is fine otherwise |
| FFT | `np.fft` | `scipy.fft` | scipy: `workers=`, `plan=`, faster real transforms, DCT/DST (numpy has none) |
| Polynomials | `np.polynomial` | `scipy.interpolate` | numpy for least-squares fits in a basis; scipy for interpolation through points |
| Random sampling | `np.random.Generator` | `scipy.stats` | numpy to generate; scipy for cdf/ppf/fit/tests |

`scipy.linalg` is always compiled against a full LAPACK, so it is complete even where numpy shipped a reduced
build. It exposes options numpy lacks:

```python
import inspect, numpy as np, scipy.linalg as sla
print("scipy eigh:", inspect.signature(sla.eigh))
print("numpy eigh:", inspect.signature(np.linalg.eigh))
print("scipy svd :", inspect.signature(sla.svd))
```

```
scipy eigh: (a, b=None, *, lower=True, eigvals_only=False, overwrite_a=False, overwrite_b=False, type=1, check_finite=True, subset_by_index=None, subset_by_value=None, driver=None)
numpy eigh: (a, UPLO='L')
scipy svd : (a, full_matrices=True, compute_uv=True, overwrite_a=False, check_finite=True, lapack_driver='gesdd')
```

`b=` gives the generalized problem `A x = lambda B x`, `subset_by_index` computes only the eigenvalues you ask
for, `driver=`/`lapack_driver=` picks the LAPACK routine. `assume_a` on `solve` selects a dedicated solver, one of
`'diagonal'`, `'tridiagonal'`, `'banded'`, `'upper triangular'`, `'lower triangular'`, `'symmetric'`/`'sym'`,
`'hermitian'`/`'her'`, `'positive definite'`/`'pos'`, `'general'`/`'gen'`; the default `None` inspects the matrix
and chooses. Structure-only solvers with no numpy counterpart: `solve_banded`, `solveh_banded`,
`solve_triangular`, `solve_toeplitz`, `solve_circulant`, `solve_sylvester`, `solve_continuous_are`,
`eigh_tridiagonal`, `eig_banded`, `cholesky_banded`, `matmul_toeplitz`.

**The batching claim, checked.** The common belief is that only `np.linalg` handles stacked `(k, n, n)` input.
That is false from scipy 1.16 on (1.15 raises `ValueError: expected matrix`). What actually differs is how `solve` reads a 2-D right-hand side.

```python
import numpy as np, scipy.linalg as sla
A = np.random.default_rng(0).random((3, 4, 4)) + 4*np.eye(4)  # stack of 3 well-conditioned 4x4
b1, bk = np.ones(4), np.ones((3, 4))       # one shared rhs; one rhs per matrix
for lib, f in [("np   ", np.linalg.solve), ("scipy", sla.solve)]:
    print(lib, "solve(A, b(4))   ->", f(A, b1).shape)
    try: f(A, bk)
    except ValueError as e: print(lib, "solve(A, b(3,4)) -> ValueError:", str(e).split(',')[0])
print("np    solve(A, bk[...,None])[...,0] ->", np.linalg.solve(A, bk[..., None])[..., 0].shape)
U, s, Vh = sla.svd(A)                       # scipy batches this on 1.17.1
print("scipy.linalg.svd(A)    ->", U.shape, s.shape, Vh.shape,
      "| reconstruct err", float(np.abs(U @ (s[..., None] * Vh) - A).max()))
print("return type 2-D vs 3-D ->", type(sla.svd(A[0])).__name__, type(sla.svd(A)).__name__)
print("norm: scipy %s  np.linalg.matrix_norm %s" % (np.shape(sla.norm(A)), np.linalg.matrix_norm(A).shape))
```

```
np    solve(A, b(4))   -> (3, 4)
np    solve(A, b(3,4)) -> ValueError: solve: Input operand 1 has a mismatch in its core dimension 0
scipy solve(A, b(4))   -> (3, 4)
scipy solve(A, b(3,4)) -> ValueError: incompatible shapes: a1.shape=(3
np    solve(A, bk[...,None])[...,0] -> (3, 4)
scipy.linalg.svd(A)    -> (3, 4, 4) (3, 4) (3, 4, 4) | reconstruct err 1.099120794378905e-14
return type 2-D vs 3-D -> tuple list
norm: scipy ()  np.linalg.matrix_norm (3,)
```

Three findings. Both libraries broadcast a 1-D `b` across the stack, and neither reads `(k, n)` as one rhs per
matrix: `b` is a stack whose core shape is `(n, nrhs)`, so write `b[..., None]` and strip the trailing axis.
`scipy.linalg` batched everything tested, verified against per-slice results: `solve`, `inv`, `det`, `cholesky`,
`lu`, `qr`, `rq`, `svd`, `eig`, `eigh`, `cho_factor`, `schur`, `polar`, `pinv`, `null_space`, `expm`, `logm`,
`sqrtm`, `funm`. But on 1.16 and 1.17 a batched factorization returns a **list** where the 2-D call returns a
**tuple** (1.18 returns a tuple from batched `svd` and `qr`, still a list from `eigh` and `lu_factor`), and
`scipy.linalg.norm` on a stack collapses to a scalar `()` instead of `(3,)`.

Timings here, 600x600 solve and a `(128, 8192)` FFT:

```
solve 600x600: scipy default 0.0059s  assume_a=pos 0.0048s  np.linalg 0.0078s
fft (128,8192): scipy workers=1 0.0062s  workers=-1 0.0039s  np.fft 0.0111s
```

Two cores, so `workers=-1` buys about 1.6x. scipy beating `np.fft` by ~1.8x single-threaded is the larger effect.

## NumPy 2.x migration

Every name below is absent on numpy 2.4.4 (`hasattr` is `False` for all fifteen). An `AttributeError` naming one
of these means the code predates numpy 2.0. Two exceptions to "removed in 2.0": `np.in1d` and `np.trapz` were only
deprecated in 2.0 (they still run, with a `DeprecationWarning`, on 2.0 through 2.3) and were removed in 2.4.
`np.row_stack` (use `np.vstack`) followed the same path and was removed in 2.5.

| Removed | Replacement | Note |
|---|---|---|
| `np.float_` | `np.float64` | `np.complex_`, `np.longfloat` went the same way |
| `np.NaN` | `np.nan` | lowercase only |
| `np.Inf`, `np.infty` | `np.inf` | `np.PINF`/`np.NINF` also gone |
| `np.alltrue` | `np.all` | |
| `np.sometrue` | `np.any` | |
| `np.product` | `np.prod` | |
| `np.cumproduct` | `np.cumprod` | |
| `np.round_` | `np.round` | |
| `np.in1d` | `np.isin` | removed in 2.4 (deprecated in 2.0); `isin` keeps the input shape, `in1d` always flattened |
| `np.trapz` | `np.trapezoid` | removed in 2.4 (deprecated in 2.0); `scipy.integrate.trapezoid` also works |
| `np.msort` | `np.sort(a, axis=0)` | |
| `np.mat` | `np.asarray`, or `np.asmatrix` to keep matrix semantics | `np.matrix` itself still exists, avoid it |
| `np.byte_bounds` | `np.lib.array_utils.byte_bounds` | moved, not deleted |
| `np.int0` | `np.intp` | same for `np.uint0` |

Also gone: `np.unicode_`/`np.string_` (use `np.str_`/`np.bytes_`), `np.bool8` (`np.bool_`), `np.object0`,
`np.find_common_type` (`np.result_type`), `np.source`, `np.lookfor`, `np.issctype`, `np.safe_eval`, `np.who`,
`np.disp`, `np.set_string_function`. Three behavior changes, one block:

```python
import numpy as np
r = np.float32(1) + 1.0
print("np.float32(1) + 1.0          ->", repr(r), "| dtype", r.dtype)
print("np.uint8(200) + np.int8(100) ->", repr(np.uint8(200) + np.int8(100)))
print("np.uint8(200) + 100          ->", repr(np.uint8(200) + 100))   # Python int adopts uint8, wraps
try: np.uint8(200) + 300
except OverflowError as e: print("np.uint8(200) + 300          -> OverflowError:", e)
print("repr/str scalar              ->", repr(np.float64(3.0)), "|", str(np.float64(3.0)),
      "|", repr([np.float64(1.5), np.int64(2)]))
arr = np.arange(3)
try: np.array([1, 2, 3], copy=False)
except ValueError as e: print("np.array(list, copy=False)   -> ValueError:", str(e).split('\n')[0])
try: np.array(arr, copy=False, dtype=np.float64)
except ValueError: print("np.array(arr, copy=False, dtype=f8) -> ValueError, dtype change forces a copy")
print("np.array(arr, copy=None) shares memory ->", np.shares_memory(np.array(arr, copy=None), arr))
```

```
np.float32(1) + 1.0          -> np.float32(2.0) | dtype float32
np.uint8(200) + np.int8(100) -> np.int16(300)
np.uint8(200) + 100          -> np.uint8(44)
np.uint8(200) + 300          -> OverflowError: Python integer 300 out of bounds for uint8
repr/str scalar              -> np.float64(3.0) | 3.0 | [np.float64(1.5), np.int64(2)]
np.array(list, copy=False)   -> ValueError: Unable to avoid copy while creating an array as requested.
np.array(arr, copy=False, dtype=f8) -> ValueError, dtype change forces a copy
np.array(arr, copy=None) shares memory -> True
```

**NEP 50 promotion**, lines 1 to 4. A Python scalar no longer participates in dtype promotion, it adopts the
array's dtype. So `float32 + 1.0` stays `float32` where numpy 1.x gave `float64`, and float32 pipelines stop
silently doubling in memory. Two numpy scalars still promote normally, so `uint8 + int8` widens to `int16` and 300
fits. But `uint8 + 100` with a *Python* int stays `uint8` and wraps to 44 (line 3 also writes
`RuntimeWarning: overflow encountered in scalar add` to stderr, which the stdout block above does not show), while
a Python int too large for the dtype raises `OverflowError` at conversion instead of wrapping.

**Scalar repr**, line 5. Zero-dimensional results print their type now. `str()` and f-strings are unchanged, so
only `repr`-based comparisons break: doctests, golden files, anything formatting a container of scalars.

**`copy=False` inverted meaning**, lines 6 to 8. In numpy 1.x it meant "copy only if you must"; in 2.x it means
"never copy" and raises when a copy is required, including when `dtype=` forces one. `copy=None` is the old
behavior and `np.asarray(x)` is the drop-in replacement.

## SciPy modern API

**`rng=` replaced `random_state=` and `seed=`.** The new name took the positional slot the old one held; the
legacy name survives as a trailing keyword-only argument, where it survives at all.

```
scipy.stats.bootstrap                 (..., method='BCa', bootstrap_result=None, rng=None, random_state=None)
scipy.optimize.differential_evolution (..., x0=None, *, integrality=None, vectorized=False, seed=None)
scipy.sparse.linalg.eigsh             (A, k=6, M=None, sigma=None, which='LM', v0=None, ncv=None, maxiter=None,
                                       tol=0, return_eigenvectors=True, Minv=None, OPinv=None, mode='normal', rng=None)
```

Observed on 1.17.1: `bootstrap` and `permutation_test` carry both `rng` and `random_state`;
`differential_evolution`, `dual_annealing`, `basinhopping` carry `rng` in the old `seed` slot plus a keyword-only
`seed`; `eigsh` carries **only** `rng`, no alias. Passing `random_state=` to `bootstrap` still works and emits no
warning, so old code fails late rather than loudly. Pass `rng=np.random.default_rng(seed)` everywhere it is
accepted (the `rng=` keyword arrived in 1.15; `eigsh` gained it only in 1.17). The legacy distribution methods are
the exception: `stats.norm.rvs(size=3, rng=...)` raises `TypeError` on 1.17 and 1.18, so pass `random_state=` there
or use the new `stats.Normal(...).sample(n, rng=...)`. `eigsh`'s
`tol=0` is unrelated: that is ARPACK's convergence tolerance, not a deprecated Krylov `tol`.

**Krylov solvers use `rtol`, keyword-only.** `tol` was removed from the `scipy.sparse.linalg` Krylov solvers; the
keyword is `rtol` and it is keyword-only.

```
cg    (A, b, x0=None, *, rtol=1e-05, atol=0.0, maxiter=None, M=None, callback=None)
gmres (A, b, x0=None, *, rtol=1e-05, atol=0.0, restart=None, maxiter=None, M=None, callback=None, callback_type=None)
cg(A, b, tol=1e-6) -> TypeError: cg() got an unexpected keyword argument 'tol'
```

Same for `bicgstab`, `minres`, `lgmres`, `cgs`, `qmr`, `gcrotmk`, `tfqmr`. `lsqr`/`lsmr` are the exception, still
positional `atol`/`btol` with no `rtol`. Convergence is `||b - Ax|| <= max(rtol*||b||, atol)`, and `atol` defaults
to `0.0`, so a near-zero `b` never converges.

**`*_array` is the way forward, `*_matrix` is legacy.** The operator semantics differ, which is the most common
silent-wrong-answer in old sparse code.

```python
import numpy as np, scipy.sparse as sp
d = np.array([[1., 2.], [0., 3.]])
Aa, Am = sp.csr_array(d), sp.csr_matrix(d)
print("dense      d * d ->", (d * d).tolist())
print("csr_array  A * A ->", (Aa * Aa).toarray().tolist(), " elementwise, matches ndarray")
print("csr_array  A @ A ->", (Aa @ Aa).toarray().tolist(), " matmul")
print("csr_matrix A * A ->", (Am * Am).toarray().tolist(), " MATMUL, not elementwise")
print("sum(axis=0): array %r  matrix %r" % (Aa.sum(axis=0), Am.sum(axis=0)))
print("eye_array %s | eye %s | isinstance sparray %s/%s"
      % (type(sp.eye_array(3, format='csr')).__name__, type(sp.eye(3)).__name__,
         isinstance(Aa, sp.sparray), isinstance(Am, sp.sparray)))
```

```
dense      d * d -> [[1.0, 4.0], [0.0, 9.0]]
csr_array  A * A -> [[1.0, 4.0], [0.0, 9.0]]  elementwise, matches ndarray
csr_array  A @ A -> [[1.0, 8.0], [0.0, 9.0]]  matmul
csr_matrix A * A -> [[1.0, 8.0], [0.0, 9.0]]  MATMUL, not elementwise
sum(axis=0): array array([1., 5.])  matrix matrix([[1., 5.]])
eye_array csr_array | eye dia_matrix | isinstance sparray True/False
```

`csr_array` also supports 1-D (`sp.coo_array(np.array([0.,1.,2.])).ndim` is `1`) and drops `getnnz`/`getH` in
favor of `.nnz` / `.conj().T`. Use `sp.sparray` for isinstance against the modern class, `sp.issparse` for either.
`.A` is gone from **both** classes on 1.17.1, use `.toarray()`. Modern constructors: `eye_array`, `diags_array`,
`random_array`, `block_array`.

**New modules and functions**, signatures from `inspect.signature` on 1.17.1.

| Call | Solves |
|---|---|
| `differentiate.derivative(f, x, *, args=(), tolerances=None, maxiter=10, order=8, initial_step=0.5, step_factor=2.0, step_direction=0, preserve_shape=False, callback=None)` | Elementwise numerical derivative with error estimate and step refinement, vectorized over `x` |
| `differentiate.jacobian(f, x, *, tolerances=None, maxiter=10, order=8, initial_step=0.5, step_factor=2.0, step_direction=0)` | Jacobian of a vector-valued `f` without hand-coded differences |
| `differentiate.hessian(f, x, *, tolerances=None, maxiter=10, order=8, initial_step=0.5, step_factor=2.0)` | Hessian, same machinery |
| `optimize.elementwise.find_root(f, init, /, *, args=(), tolerances=None, maxiter=None, callback=None)` | An independent scalar root per array element in one vectorized call, replacing a loop over `brentq` |
| `optimize.elementwise.find_minimum(f, init, /, *, args=(), tolerances=None, maxiter=100, callback=None)` | Same, for scalar minimization |
| `optimize.elementwise.bracket_root(f, xl0, xr0=None, *, xmin=None, xmax=None, factor=None, args=(), maxiter=1000)` | Finds a sign-change bracket automatically before rooting |
| `integrate.cubature(f, a, b, *, rule='gk21', rtol=1e-08, atol=0, max_subdivisions=10000, args=(), workers=1, points=None)` | Adaptive multi-dimensional quadrature over a box, the real N-D `quad` |
| `integrate.tanhsinh(f, a, b, *, args=(), log=False, maxlevel=None, minlevel=2, atol=None, rtol=None, preserve_shape=False, callback=None)` | Double-exponential quadrature: endpoint singularities and infinite intervals, vectorized over limits |
| `integrate.nsum(f, a, b, *, step=1, args=(), log=False, maxterms=1048576, tolerances=None)` | Sums a convergent series, including to infinity |
| `interpolate.AAA(x, y, *, rtol=None, max_terms=100, clean_up=True, clean_up_tol=1e-13)` | Adaptive Antoulas-Anderson rational approximation, for poles and near-singular data where polynomials fail |
| `interpolate.make_splrep(x, y, *, w=None, xb=None, xe=None, k=3, s=0, t=None, nest=None, bc_type=None)` | FITPACK smoothing spline returning a `BSpline`, replacing the tuple-returning `splrep` |
| `stats.make_distribution(dist)` | Wraps a legacy `rv_continuous` into the new distribution infrastructure |
| `stats.Normal(mu=None, sigma=None, **kwargs)` | New-infrastructure distribution object: arithmetic on distributions, better tail accuracy |

One thread through four of them, `f(x) = x**2 - 2`, whose root is `sqrt(2)`:

```python
import numpy as np
from scipy.differentiate import derivative
from scipy.optimize.elementwise import find_root
from scipy.integrate import tanhsinh, nsum
f = lambda x: x**2 - 2
print("derivative(f, [1,2,3]).df  ->", derivative(f, np.array([1., 2., 3.])).df, "| exact 2x")
print("find_root(f, bracket).x    ->", find_root(f, (np.array([1., 1.]), np.array([2., 3.]))).x)
print("tanhsinh(f, 0, 1).integral ->", tanhsinh(f, 0., 1.).integral, "| exact", 1/3 - 2)
print("nsum(1/k^2, 1, inf).sum    ->", nsum(lambda k: 1/k**2, 1, np.inf).sum, "| pi^2/6", np.pi**2/6)
```

```
derivative(f, [1,2,3]).df  -> [2. 4. 6.] | exact 2x
find_root(f, bracket).x    -> [1.41421356 1.41421356]
tanhsinh(f, 0, 1).integral -> -1.6666666666666674 | exact -1.6666666666666667
nsum(1/k^2, 1, inf).sum    -> 1.6449340668479238 | pi^2/6 1.6449340668482264
```

`derivative` and `find_root` ran three and two independent scalar problems in one vectorized call, which is the
point of both. Each returns a result *object* carrying a status field, not a bare number. Check it.

**Legacy, still present, steer away.**

| Legacy | Use instead | Why |
|---|---|---|
| `interpolate.interp1d` | `make_interp_spline`, `CubicSpline`, `PchipInterpolator`, or `np.interp` for linear | Its docstring opens "Interpolate a 1-D function (legacy)." with a `.. legacy:: class` directive |
| `interpolate.splrep`/`splev` | `make_splrep`, then call the `BSpline` | Returns a tuple of arrays instead of an object |
| `integrate.odeint` | `integrate.solve_ivp` | `odeint` uses `f(y, t)` order and has no events, dense output, or method choice |
| `optimize.leastsq` | `optimize.least_squares` | Bounds, outlier-resistant loss functions, explicit termination reasons |
| `optimize.fsolve` | `optimize.root` | Method selection, uniform `OptimizeResult` |
| `optimize.brentq` in a loop | `optimize.root_scalar`, or `elementwise.find_root` for arrays | |
| `optimize.fmin*` wrappers | `optimize.minimize(..., method=...)` | `fmin`/Nelder-Mead, `fmin_bfgs`/BFGS, `fmin_cg`/CG, `fmin_ncg`/Newton-CG, `fmin_powell`/Powell, `fmin_l_bfgs_b`/L-BFGS-B, `fmin_tnc`/TNC, `fmin_slsqp`/SLSQP, `fmin_cobyla`/COBYLA. All nine still exist |
| `np.random.RandomState`, `np.random.seed` | `np.random.default_rng(seed)` | `RandomState` is frozen for stream compatibility, gets no new distributions or speedups. Both still exist |

## Known drift on numpy 2.5 / scipy 1.18

The outputs in these references were produced on numpy 2.4.4 / scipy 1.17.1. Running every code block again on
numpy 2.5.4 / scipy 1.18.1 (the current releases on 2026-10-22) changed these results; everything else printed the
same, apart from timings and `inspect.signature` reprs.

| Area | 2.4.4 / 1.17.1 | 2.5.4 / 1.18.1 |
|---|---|---|
| Batched `scipy.linalg.svd(A)` / `qr(A)` on `(k, n, n)` | `list` | `tuple` (`eigh`, `lu_factor` still `list`) |
| `scipy.linalg.cho_factor(S)` | `lower` is a `bool`; unused triangle keeps the original entries | `lower` is a 0-d bool `ndarray`; unused triangle is zeroed |
| `scipy.linalg.cholesky` on a non-SPD matrix | `LinAlgError: 2-th leading minor of the array is not positive definite` | `LinAlgError: Internal potrf return info = [2] for slices [0].` |
| `scipy.linalg.lstsq` `residues` (overdetermined, full rank) | `np.float64` scalar | 0-d `ndarray`; `NaN` for rank-deficient slices |
| `scipy.linalg.qr(..., lwork=)` | accepted | deprecated, computed automatically |
| `scipy.optimize.nnls(A, b, *, maxiter=None, atol=...)` | `atol` accepted | `atol` removed |
| `derivative`, `tanhsinh`, `nsum`, `elementwise.find_root`/`find_minimum`/`bracket_root` | no `kwargs=` | new `kwargs=None` parameter after `args` |
| `np.row_stack` | deprecated alias | removed |
| `np.sort` | `(a, axis=-1, kind=None, order=None, *, stable=None)` | adds `descending=` |
| `arr.shape = ...` | allowed when no copy is needed | `DeprecationWarning`; use `arr.reshape(...)` |
| `np.linalg.eig` / `eigvals` | real output when all eigenvalues are real | always complex |

## Array API and alternative backends

scipy functions increasingly dispatch on array type rather than converting to numpy, so a torch or cupy array can
pass straight through. The signature-level tell is `scipy.fft.rfftfreq(n, d=1.0, *, xp=None, device=None)` against
`numpy.fft.rfftfreq(n, d=1.0, device=None)`: `xp=` names the array namespace to build the output in and `device=`
places it, so `scipy.fft.rfftfreq(8, d=0.5, xp=np)` gives `[0. 0.25 0.5 0.75 1.]`. No non-numpy backend is
installed in this sandbox, so dispatch itself is unverified here, only the parameters are confirmed present.

`np.linalg` now follows the array API spec. Verified present there: `matrix_norm`, `vector_norm`, `vecdot`,
`matmul`, `svdvals`, `cross`, `outer`, `matrix_transpose`, `tensordot`, `diagonal`, `trace`. Several live only in
`np.linalg`: `np.vecdot`, `np.cross` and `np.outer` exist at top level, but `np.matrix_norm`, `np.vector_norm` and
`np.svdvals` do not. Top-level array-API aliases that do exist: `np.concat`, `np.permute_dims`, `np.astype`,
`np.pow`, `np.unique_values`, `np.matrix_transpose`, `np.bitwise_left_shift`.

## Gotchas

- **`np.linalg.solve` with a `(k, n)` right-hand side raises instead of doing what you meant.** Symptom:
  `ValueError: solve: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (m,m),(m,n)->(m,n)`.
  Cause: any `b` with `ndim > 1` is read as a stack of `(n, nrhs)` matrices; only a strictly 1-D `b` broadcasts as
  a single vector. Fix: `np.linalg.solve(A, b[..., None])[..., 0]`. `scipy.linalg.solve` rejects it too, with
  `incompatible shapes`.
- **Batched `scipy.linalg` factorizations return a `list`, not a `tuple` (1.16 and 1.17).** `type(sla.svd(A))` is
  `tuple` for 2-D `A` and `list` for `(k, n, n)`; 1.18 changed `svd` and `qr` to `tuple` but not `eigh` or `lu_factor`. Any `isinstance(res, tuple)` branch silently takes the wrong path. Fix: unpack
  positionally (`U, s, Vh = sla.svd(A)`), which works for both.
- **`scipy.linalg.norm` on a stack collapses to a scalar.** `sla.norm(A)` for `A.shape == (3,4,4)` returns shape
  `()`, not `(3,)`. Fix: `np.linalg.matrix_norm(A)` or `np.linalg.norm(A, axis=(-2,-1))`.
- **`csr_matrix * csr_matrix` is matrix multiplication.** Symptom: a sparse "elementwise product" that is really a
  matmul, with plausible shapes whenever the matrix is square. Cause: `spmatrix` keeps `np.matrix` operator
  semantics. Fix: use `csr_array`, where `*` is elementwise and `@` is matmul.
- **`csr_matrix.sum(axis=0)` returns `np.matrix` of shape `(1, n)`**, which infects downstream broadcasting.
  `csr_array.sum(axis=0)` gives a 1-D `ndarray`. `.A` is gone from both classes on 1.17.1, use `.toarray()`.
- **`sp.eye` returns `dia_matrix` but `sp.eye_array` returns the array class.** Mixing families in one expression
  silently reintroduces matrix semantics. Same for `diags` vs `diags_array`, `random` vs `random_array`.
- **`tol=` on a Krylov solver is a `TypeError`, not a warning.** `tol` was removed from the `scipy.sparse.linalg`
  Krylov solvers; it is `rtol` and keyword-only. `lsqr`/`lsmr` are unaffected. Unrelated: `eigsh`/`svds` still take
  a positional `tol=0`, ARPACK's own tolerance.
- **`random_state=` on `scipy.stats.bootstrap` still works and warns about nothing on 1.17.1**, so old code fails
  late. `scipy.sparse.linalg.eigsh` already takes only `rng=`. Standardize on `rng=` now.
- **`np.array(x, copy=False)` raises `ValueError: Unable to avoid copy while creating an array as requested`.**
  The flag inverted meaning in numpy 2.0 and now means "never copy". Fix: `np.asarray(x)`, or `copy=None` for the
  old "copy if needed". A `dtype=` change also forces a copy and so also raises.
- **`np.uint8(200) + 100` wraps to 44 with a `RuntimeWarning`, but `np.uint8(200) + 300` raises `OverflowError`.**
  Under NEP 50 the Python int adopts the array dtype, so in-range literals wrap and out-of-range ones fail at
  conversion. Fix: cast deliberately, e.g. `np.uint8(200) + np.int16(100)`.
- **Doctests and golden files break on scalar repr.** `repr(np.float64(3.0))` is `np.float64(3.0)` on numpy 2.x;
  `str()` and f-strings are unchanged, so only `repr`-based comparisons need updating.
- **New `scipy.stats` distribution objects have no `.var()`, `.std()`, `.sf()` or `.ppf()`.**
  `stats.Normal(mu=0., sigma=1.).var()` is an `AttributeError`. Use `variance()`, `standard_deviation()`,
  `ccdf()`, `icdf()`, and `sample(shape, rng=...)` rather than `rvs(size=...)`.
- **`scipy.differentiate` and `scipy.optimize.elementwise` return result objects, not floats.** Read `.df`, `.x`,
  `.integral`, `.sum`, `.estimate`, and check `.success` / `.status`.

## Quick reference

| Task | Call | Note |
|---|---|---|
| Dense solve, known structure | `scipy.linalg.solve(A, b, assume_a='pos')` | also `'sym'`, `'her'`, `'diagonal'`, `'tridiagonal'`, `'banded'`, `'upper triangular'`, `'lower triangular'`, `'gen'` |
| Stacked `(k,n,n)` solve | `np.linalg.solve(A, b[..., None])[..., 0]` | 1-D `b` broadcasts; 2-D `b` is not one-rhs-per-matrix |
| Stacked factorization | `scipy.linalg.svd/eigh/lu/qr(A)` | batched from 1.16; unpack positionally, the container is a list on 1.16/1.17 and varies on 1.18 |
| Generalized symmetric eigenproblem | `scipy.linalg.eigh(A, B)` | numpy has no `b=` |
| Subset of eigenvalues | `scipy.linalg.eigh(A, subset_by_index=(0, 4))` | avoids a full decomposition |
| Norm of a stack | `np.linalg.matrix_norm(A)` | `scipy.linalg.norm(A)` returns a scalar |
| FFT, multithreaded | `scipy.fft.fft(x, workers=-1)` | ~1.6x on 2 cores; scipy beat `np.fft` ~1.8x single-threaded |
| DCT / DST | `scipy.fft.dct(x, type=2, norm='ortho')` | `np.fft` has none |
| Krylov solve | `scipy.sparse.linalg.cg(A, b, rtol=1e-10)` | `rtol` keyword-only, `atol` defaults to `0.0` |
| Sparse, new code | `scipy.sparse.csr_array`, `eye_array`, `diags_array` | `*` elementwise, `@` matmul, reductions give ndarray |
| Sparse isinstance check | `isinstance(x, sp.sparray)` / `sp.issparse(x)` | first excludes `spmatrix`, second accepts both |
| Numerical derivative | `scipy.differentiate.derivative(f, x)` | error-controlled, vectorized over `x`, read `.df` |
| Jacobian / Hessian | `scipy.differentiate.jacobian(f, x)`, `.hessian(f, x)` | |
| Root per array element | `scipy.optimize.elementwise.find_root(f, (lo, hi))` | replaces a loop over `brentq` |
| N-D adaptive quadrature | `scipy.integrate.cubature(f, a, b)` | read `.estimate` |
| Singular / infinite interval | `scipy.integrate.tanhsinh(f, a, b)` | double-exponential, read `.integral` |
| Infinite series | `scipy.integrate.nsum(f, 1, np.inf)` | read `.sum` |
| ODE IVP | `scipy.integrate.solve_ivp(fun, t_span, y0)` | not `odeint`; argument order is `fun(t, y)` |
| 1-D interpolation | `scipy.interpolate.make_interp_spline`, `CubicSpline` | not `interp1d` |
| Smoothing spline | `scipy.interpolate.make_splrep(x, y, s=...)` | returns a `BSpline`, not a tuple |
| Rational approximation | `scipy.interpolate.AAA(x, y)` | poles via `.poles()` |
| Nonlinear least squares | `scipy.optimize.least_squares` | not `leastsq` |
| Root of a vector function | `scipy.optimize.root` | not `fsolve` |
| Minimization | `scipy.optimize.minimize(f, x0, method='L-BFGS-B')` | not the `fmin_*` wrappers |
| Random numbers | `np.random.default_rng(seed)` | not `RandomState` or `np.random.seed` |
| Seeding any scipy routine | `rng=np.random.default_rng(0)` | not `random_state=` or `seed=` |
| Distribution, new infra | `scipy.stats.Normal(mu=0., sigma=1.)` | `variance()`, `ccdf()`, `icdf()`, `sample(n, rng=...)` |
| Wrap a legacy distribution | `scipy.stats.make_distribution(scipy.stats.lognorm)` | returns a class |
| Array without a forced copy | `np.asarray(x)` | `np.array(x, copy=False)` raises on 2.x |
| Trapezoid rule | `np.trapezoid` | `np.trapz` deprecated in numpy 2.0, removed in 2.4 |
| Membership test | `np.isin(a, b)` | `np.in1d` deprecated in numpy 2.0, removed in 2.4 |
