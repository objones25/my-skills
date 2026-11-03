# Debugging: Symptom to Fix

Start here for any traceback, warning, `nan`, or wrong-looking number. Every message below
was produced on numpy 2.4.4 / scipy 1.17.1, so the text matches what you will actually see.
The last section covers the harder case: no error at all, just a wrong answer.

## Errors, by message

| message | cause | fix |
|---|---|---|
| `ValueError: operands could not be broadcast together with shapes (3,4) (5,)` | shapes do not align right-to-left | print both `.shape`; add an axis with `[:, None]` or `[None, :]`. `02-broadcasting-indexing.md` |
| `ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?)` | `A @ B` with `A.shape[-1] != B.shape[-2]` | transpose one operand, or you meant `*` (elementwise) not `@` |
| `ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape` | ragged nested lists | pad to a rectangle, or keep a list of arrays |
| `ValueError: zero-size array to reduction operation maximum which has no identity` | `max`/`min`/`argmax` on an empty array or an empty group | guard on `.size`, or use `np.max(a, initial=-np.inf)` |
| `LinAlgError: Singular matrix` (numpy) / `A singular matrix detected: slice(s) [0] are singular.` (scipy) | `A` is exactly or numerically singular | check `np.linalg.cond(A)` and `matrix_rank(A)`; if rank-deficient use `lstsq` or `pinv`, not `solve` |
| `ValueError: array must not contain infs or NaNs` | `nan`/`inf` reached a LAPACK call | `np.isfinite(A).all()`; find where they entered, do not just mask them |
| `LinAlgError: 2-th leading minor of the array is not positive definite` (scipy 1.17), `Internal potrf return info = [2] for slices [0].` (1.18), `Matrix is not positive definite` (numpy) | `cholesky` on a matrix that is not SPD | check symmetry (`allclose(A, A.T)`) and `eigvalsh(A).min()`; add jitter `A + 1e-10*I` only if you understand why it is indefinite |
| `TypeError: cg() got an unexpected keyword argument 'tol'` | Krylov solvers renamed `tol` to `rtol`, keyword-only | `cg(A, b, rtol=1e-10)` |
| `ValueError: f(a) and f(b) must have different signs` | `brentq`/`bisect` bracket does not straddle a root | plot or scan `f` first; `optimize.bracket_root` finds one automatically |
| `ValueError: \`y0\` must be 1-dimensional.` | `solve_ivp(f, span, 1.0)` | pass `[1.0]` |
| `ValueError: Number of samples must be one plus a non-negative power of 2.` | `romb` needs exactly `2**k + 1` samples | use `simpson`, or resample |
| `AttributeError: module 'numpy' has no attribute 'float_'` (or `NaN`, `Inf`, `in1d`, `trapz`, `product`, `alltrue`, `mat`) | removed in NumPy 2.0 (`in1d`, `trapz` in 2.4) | `float64`, `nan`, `inf`, `isin`, `trapezoid`, `prod`, `all`, `asmatrix`. `00-module-map.md` |
| `ValueError: Unable to avoid copy while creating an array as requested.` | `np.array(x, copy=False)` now means "never copy" | `np.asarray(x)` |
| `TypeError: 'numpy.float64' object cannot be interpreted as an integer` | a float where a count/index is expected | `int(...)`, or use `//` instead of `/` |
| `MemoryError` / `Unable to allocate ... for an array with shape ...` | an intermediate is bigger than you think, usually from a broadcast | check the shape the error names; it is often `(n, n)` from an `x[:, None] - y[None, :]` you meant to reduce |

## Warnings, and why not to silence them

| warning | means | do |
|---|---|---|
| `RuntimeWarning: divide by zero encountered in divide` | a zero denominator produced `±inf` | mask the denominator, or `np.errstate(divide='ignore')` *only* if `inf` is the answer you want |
| `RuntimeWarning: invalid value encountered in ...` | produced a `nan` (`0/0`, `sqrt(-1)`, `log(-1)`) | find the input; a `nan` here becomes a `nan` everywhere downstream |
| `RuntimeWarning: overflow encountered in exp` | argument above ~709 for float64 | use `special.logsumexp`, `special.expit`, `special.softmax`, or work in logs |
| `IntegrationWarning: The maximum number of subdivisions (50) has been achieved.` | `quad` could not resolve the integrand | raise `limit=`, pass `points=` for kinks, split the interval, or use `tanhsinh`. `08-integrate-quadrature.md` |
| `UserWarning: At least one element of \`rtol\` is too small. Setting rtol = 2.22e-14` | asked `solve_ivp` for tighter than `100*eps` | that is the floor; if you need more accuracy the problem needs reformulating |
| `SparseEfficiencyWarning: Changing the sparsity structure of a csr_array is expensive.` | assigning into CSR/CSC | build in COO or LIL, convert once. `06-sparse-iterative.md` |
| `LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = ...` | `cond(A)` is large | the answer has roughly `log10(cond)` fewer correct digits; regularize or reformulate. `04-floating-point.md` |
| `OptimizeWarning: Covariance of the parameters could not be estimated` | `curve_fit`'s Jacobian is rank-deficient at the solution | the model is over-parameterized for this data, or `p0` is far off |
| `RuntimeWarning: invalid value encountered in ...` (e.g. `scalar divide`, `sqrt`, `log`) inside an optimizer | the objective returned `nan` for some trial point | clip or guard the objective's domain; return a large finite value instead of `nan` |

To find where a warning comes from, promote it to an exception and read the traceback:

```python
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)
```

For numpy specifically, `np.seterr(all="raise")` or `with np.errstate(all="raise"):` does the
same for floating-point flags.

## No error, wrong answer

This is the harder half. Work down the list.

**1. Is it silent integer overflow?** NumPy 2 does not warn on integer overflow in array
arithmetic:

```
np.array([100, 100], dtype=np.int8) + np.array([100, 100], dtype=np.int8)  ->  -56
```

Check `.dtype` on every array in the chain. Cast to `int64` or `float64` early.

**2. Is it a view aliasing a parent?** `b = a[1:3]` shares memory with `a`; writing to `b`
changes `a`. `np.shares_memory(a, b)` and `b.base` tell you. The reverse trap is
`a[mask][0] = 1`, which writes into a temporary copy and does nothing to `a`.
`01-arrays-dtypes.md`.

**3. Is it a shape that broadcast when you wanted an error?** `(n,)` against `(n, 1)` gives
`(n, n)`. If a result is suddenly quadratic in size, or an intermediate blew up memory, this
is why. Print shapes at every step; `keepdims=True` on the reduction that produced the
`(n,)` usually fixes it.

**4. Did the solver actually converge?** Most scipy solvers return a failed result without raising:

```python
sol.success, sol.status, sol.message      # solve_ivp, solve_bvp, minimize, root, least_squares
r.converged, r.flag                       # root_scalar
x, info = spla.cg(...); assert info == 0  # Krylov solvers: info != 0 means NOT converged
value, abserr = quad(...)                 # abserr is an estimate, not a bound; failures only warn
```

The exceptions raise instead: `curve_fit` (`RuntimeError`; it returns a plain `(popt, pcov)` with no
`.success`, so also check `np.isfinite(pcov).all()`), standalone `brentq`/`bisect`/`newton`/`fixed_point`
(`RuntimeError`), and `eigsh`/`eigs` (`ArpackNoConvergence`).

A nonzero `info` from a Krylov solver returns the last iterate, which looks like a solution
and is not one.

**5. Is the tolerance a lie?** A tolerance is a request. `quad`'s `abserr` is an estimate
that a discontinuous or oscillatory integrand can make arbitrarily optimistic. `solve_ivp`'s
`rtol` controls per-step local error, not the accumulated global error. Confirm by halving
the tolerance and checking the answer stops moving. That two-run check is the single most
useful habit in this whole file.

**6. Is it conditioning?** Compute `np.linalg.cond(A)`. If it is `1e12`, a float64 solve has
about four correct digits and there is no algorithm that fixes that, only a reformulation.
Forming `A.T @ A` squares it, which is why the normal equations lose half your digits and QR
does not. `05-linalg-dense.md`.

**7. Is it a convention you assumed?** The recurring ones:

| you probably assumed | it is actually |
|---|---|
| `svd` returns `U, s, V` | `U, s, Vh`, already transposed |
| `eig` values come out sorted | unsorted, and complex even for real input |
| `scipy.linalg.cholesky` returns lower | upper by default (`np.linalg.cholesky` returns lower) |
| `solve_ivp`'s `sol.y` is `(n_times, n_states)` | `(n_states, n_times)` |
| `rfftfreq` takes the sample rate | takes the sample *spacing*, `d=1/fs` |
| `dblquad`'s integrand is `f(x, y)` | `f(y, x)` |
| `1 - cdf(x)` works in the tail | underflows to 0; use `sf(x)` |
| `csr_array * B` is matmul | elementwise; `@` is matmul (opposite for `csr_matrix`) |
| `stats.gamma.fit` returns `(loc, scale, a)` | shape params first: `(a, loc, scale)` |

**8. Is it the objective, not the optimizer?** Before blaming `minimize`, check the gradient
you supplied with `optimize.check_grad`, and rescale parameters that differ by orders of
magnitude. Unscaled parameters break every gradient-based method.
`07-optimize-roots.md`.

## A minimal reproduction protocol

When none of the above lands, shrink the problem until the bug has nowhere to hide.

1. **Replace the data with something whose answer you know.** A `y' = -y` ODE, an identity or
   diagonal matrix, a polynomial the quadrature rule integrates exactly. If the known case is
   wrong, the bug is in the call. If it is right, the bug is in the data or the model.
2. **Shrink `n`** until you can print the whole array and read it.
3. **Print shape and dtype at every step**, not just values. Most silent wrongness is one or
   the other.
4. **Check the two-run convergence test**: halve the tolerance or double the resolution. An
   answer that moves is not converged, whatever the solver reported.
5. **Compare against a second method.** `solve_ivp` with `RK45` versus `Radau`, `quad` versus
   `tanhsinh`, a dense solve versus an iterative one. Two independent implementations
   agreeing is much stronger evidence than one reporting success.
6. **Seed everything.** `rng = np.random.default_rng(0)` threaded through, so the failure is
   the same failure every run.

## Quick reference

| task | call |
|---|---|
| turn a warning into a traceback | `warnings.filterwarnings("error", category=RuntimeWarning)` |
| same for floating-point flags | `np.seterr(all="raise")` or `with np.errstate(all="raise"):` |
| locate `nan`/`inf` | `np.argwhere(~np.isfinite(a))` |
| check aliasing | `np.shares_memory(a, b)`, `b.base is a` |
| check conditioning | `np.linalg.cond(A)`, `np.linalg.matrix_rank(A)` |
| check symmetry / definiteness | `np.allclose(A, A.T)`, `np.linalg.eigvalsh(A).min()` |
| verify a hand-written gradient | `scipy.optimize.check_grad(f, grad, x0)` |
| verify a signature or default | `inspect.signature(scipy.x.f)` |
| compare with tolerance | `np.testing.assert_allclose(a, b, rtol=1e-10)` (`atol` defaults to 0) |
| find the slow line | `cProfile.run("main()", sort="cumtime")` |
| peak memory of an array | `a.nbytes`, `tracemalloc` |
