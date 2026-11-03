# Interpolation and Smoothing

Choosing an interpolator in `scipy.interpolate` and knowing what it will do between the
knots. Open this for 1-D interpolators, splines, smoothing versus interpolating fits, and
scattered or gridded N-D data. The theory (Lagrange form, uniqueness of the degree-`n`
interpolant through `n+1` distinct nodes, the `f^(n+1)(xi)/(n+1)! * prod(x - x_j)` error
term, Runge's phenomenon) is assumed known.

## The decision

Two questions settle almost every case.

**Must the curve pass through the data?** Yes means interpolation, no means smoothing. Noisy
measurements almost always want smoothing, because an interpolant is obliged to reproduce
every noise spike.

**Do you need a shape guarantee?** Monotone data that must stay monotone, or positive data
that must stay positive, rules out the cubic spline and points at `PchipInterpolator`.

| you have | you want | use |
|---|---|---|
| clean samples, smooth `f` | smooth curve, `C2` | `CubicSpline` |
| clean samples, monotone or positive | no overshoot | `PchipInterpolator` |
| clean samples, sharp local features | less ringing than cubic | `Akima1DInterpolator` |
| noisy samples | a fit, not an interpolant | `make_smoothing_spline` or `make_splrep(s=...)` |
| arbitrary spline degree, custom knots | full B-spline control | `make_interp_spline` |
| a function you can evaluate anywhere | near-optimal rational fit | `AAA` |
| values on a regular grid, N-D | gridded lookup | `RegularGridInterpolator` |
| scattered points, N-D | scattered fit | `RBFInterpolator` |

`interp1d` still exists and still works. It is documented as legacy. `make_interp_spline`
replaces `kind='cubic'`/`'quadratic'`, and `np.interp` replaces `kind='linear'` and is faster.
Do not reach for `interp1d` in new code.

## Cubic splines and the `PPoly` interface

```python
import numpy as np
from scipy import interpolate as ip

x = np.array([0., 1., 2., 3., 4.])
y = np.array([0., 0.8, 0.9, 0.1, -0.8])
xs = np.array([0.5, 2.5, 3.5])

cs = ip.CubicSpline(x, y)
print(cs(xs))
print(cs.c.shape, cs.x)
print(cs(2.5, 1), cs.derivative()(2.5))
print(cs.integrate(0, 4), cs.antiderivative()(4.0))
print(cs.roots(extrapolate=False))
```

```
[ 0.459375  0.578125 -0.403125]
(4, 4) [0. 1. 2. 3. 4.]
-0.84375 -0.84375
1.5333333333333332 1.5333333333333334
[0.         3.09725663]
```

`CubicSpline` returns a `PPoly`: a piecewise polynomial stored as coefficients `c` of shape
`(k+1, n_intervals)` over breakpoints `x`. Column `j` holds the coefficients on
`[x[j], x[j+1]]`, highest power first, in the local variable `(t - x[j])`. Everything a
`PPoly` gives you is exact for the piecewise polynomial, not a numerical approximation:
`derivative()` and `antiderivative()` return new `PPoly` objects, `integrate(a, b)` is the
exact definite integral, and `roots()` solves each cubic in closed form. This makes a spline
a good way to differentiate or integrate noisy sampled data once you have smoothed it.

`bc_type` sets the end conditions: `'not-a-knot'` (default), `'natural'` (zero second
derivative at the ends), `'clamped'` (zero first derivative), `'periodic'`, or an explicit
pair of `(order, value)` tuples. Natural splines are the textbook default and are *less*
accurate near the ends than not-a-knot.

`CubicSpline` extrapolates by continuing the end cubic, which diverges fast:

```
cs(5.0)                           -> -0.75
CubicSpline(..., extrapolate=False)(5.0) -> nan
```

Pass `extrapolate=False` whenever silently-wrong extrapolated values would be worse than a
`nan` you can see.

## Shape preservation

A cubic spline is `C2` and therefore free to overshoot. On data with a near-flat stretch
followed by a jump:

```python
xm = np.array([0., 1, 2, 3, 4, 5])
ym = np.array([0., 1, 1.02, 1.03, 4, 5])
```

```
                on [1, 3]      data range there is 1.00 to 1.03
CubicSpline     min 0.7531  max 1.1384    monotone: False
Pchip           min 1.0000  max 1.0300    monotone: True
```

The cubic dips 25% below the smallest data value in a region where the data barely moves,
because `C2` continuity propagates the curvature needed for the later jump backwards.
`PchipInterpolator` gives up `C2` (it is only `C1`) and in exchange never overshoots and
preserves monotonicity on monotone data. `Akima1DInterpolator` sits between them: local,
less ringing than cubic, no monotonicity guarantee. `method='makima'` is the modified
Akima variant that behaves better on flat stretches.

## Global polynomials and Runge's phenomenon

`BarycentricInterpolator` and `KroghInterpolator` build the unique degree-`n` polynomial
through `n+1` points. On equispaced nodes this is the textbook failure:

```
f(t) = 1/(1 + 25 t^2) on [-1, 1], equispaced nodes

n=11   barycentric max err  1.9156    cubic spline max err  0.0220
n=21   barycentric max err 59.7683    cubic spline max err  0.0032
```

Doubling the nodes makes the polynomial 30x worse and the spline 7x better. The error
formula explains it: `prod(x - x_j)` is enormous near the interval ends for equispaced
nodes, and `f^(n+1)` grows faster than `(n+1)!` for this `f`. Global polynomial
interpolation is only safe on Chebyshev-clustered nodes. If you control the node placement,
cluster them; if you do not, use a spline.

`AAA` is the modern alternative when you can evaluate `f` anywhere: an adaptive rational
(barycentric) approximation that handles poles, which no polynomial or spline can.

```python
zz = np.linspace(-1.4, 1.4, 200)
a = ip.AAA(zz, np.tan(zz))
print(np.max(np.abs(a(zz) - np.tan(zz))))
print(np.sort(a.poles()[np.abs(a.poles().imag) < 1e-6].real)[:4])
```

```
6.987743716990735e-13
[-9.30592098 -4.71954979 -1.57079633  1.57079633]
```

Machine precision on `tan`, and `a.poles()` recovers `±pi/2 = ±1.570796` from samples that
never went near the singularity.

## Smoothing noisy data

`s` is the smoothing parameter: the fit satisfies `sum(w_i * (y_i - f(x_i))^2) <= s`. So
`s=0` interpolates and larger `s` is smoother. The useful range scales with the noise
variance times the number of points, which is why picking `s` by hand is guesswork.

60 equispaced samples of `sin(x)` on `[0, 10]` with `sigma = 0.15` noise
(`np.random.default_rng(0)`), RMS error against the true `sin`:

```
make_splrep(x, y, s=0)            0.13129     interpolates the noise
make_splrep(x, y, s=len(x))       0.57714     over-smoothed, flattens the signal
make_smoothing_spline(x, y)       0.07187     lambda chosen by GCV
```

`make_smoothing_spline` picks its own penalty by generalized cross-validation and beats both
hand-picked values here. Reach for it first. Use `make_splrep(s=...)` when you need to control
the fit explicitly or need a specific knot vector.

`make_splrep` and `make_splprep` are the modern replacements for `splrep`/`splprep`; they
return a `BSpline` object rather than a raw `(t, c, k)` tuple you then have to feed to
`splev`. `UnivariateSpline` is the older class-based interface for the same job.

Two other smoothers worth knowing: `scipy.signal.savgol_filter` (local polynomial least
squares on a sliding window, cheap, gives derivatives directly, see `11-fft-signal.md`) and
`scipy.ndimage.gaussian_filter1d`.

## B-splines directly

`make_interp_spline(x, y, k=3)` returns a `BSpline` and is the general interpolating call:
any degree `k` (even degrees work too: `k=2` and `k=4` reproduce `x**2` exactly), optional
custom knot vector `t`, optional `bc_type`. For `k=3` with default
boundary conditions it agrees with `CubicSpline` exactly:

```
CubicSpline          [ 0.459375  0.578125 -0.403125]
make_interp_spline   [ 0.459375  0.578125 -0.403125]
```

`BSpline(t, c, k)` constructs one from knots and coefficients if you have them from
elsewhere. `make_lsq_spline` fits given knots by least squares. `generate_knots` produces a
knot sequence for a target smoothness. `BSpline.design_matrix` gives the sparse collocation
matrix, which is what you want if you are building your own regularized spline fit.

## N-D interpolation

**On a regular (rectilinear) grid**, `RegularGridInterpolator`. The grid axes do not have to
be uniformly spaced, only sorted and separable.

```python
RegularGridInterpolator(points, values, method='linear', bounds_error=True,
                        fill_value=nan, *, solver=None, solver_args=None)
```

`points` is a tuple of 1-D coordinate arrays, one per axis; `values` has shape
`(len(points[0]), len(points[1]), ...)`. Query points are `(m, ndim)`: a bare `[0.3, 0.7]`
is accepted as one point, but a transposed `(ndim, m)` array is read as `ndim` points and,
when `m == ndim`, returns wrong values without raising. `method` accepts `'linear'`, `'nearest'`, `'slinear'`,
`'cubic'`, `'quintic'`, `'pchip'`.

```
grid 5x7 of sin(x)cos(y) on [0,1]x[0,2], method='cubic'
query [[0.3, 0.7], [0.8, 1.5]]  ->  [0.226028 0.050751]
exact                           ->  [0.226026 0.050744]
```

`interpn` is the one-shot functional form of the same thing.

**Scattered points**, `RBFInterpolator`. Radial basis functions: the value at a query point
is a weighted sum of a kernel applied to the distances to the data points. Default kernel is
`'thin_plate_spline'`; `'linear'`, `'cubic'`, `'multiquadric'`, and `'gaussian'` (needs
`epsilon`) are the others. `smoothing > 0` turns it from interpolation into regression.
`neighbors=k` restricts each evaluation to the `k` nearest data points, which is the
difference between `O(n^3)` and tractable above a few thousand points.

`griddata` is the convenience wrapper over Delaunay-triangulation-based interpolation
(`LinearNDInterpolator`, `CloughTocher2DInterpolator`, `NearestNDInterpolator`). It returns
`nan` outside the convex hull of the data, and it rebuilds the triangulation on every call,
so use the class directly if you query more than once. On the same test:

```
RBFInterpolator     [0.222671 0.050456]
griddata(cubic)     [0.224884 0.051077]
exact               [0.226026 0.050744]
```

`rescale=True` matters when the coordinate axes have wildly different units, since the
distance metric is otherwise dominated by whichever axis has the larger numbers.

## Gotchas

- **Extrapolation is on by default** for `CubicSpline`, `make_interp_spline`, `BSpline`, and
  `PPoly`. The end polynomial is continued and diverges. Pass `extrapolate=False` to get
  `nan` instead of a plausible wrong number.
- **`RegularGridInterpolator` query points are `(m, ndim)`.** Passing the transpose, `(ndim, m)`,
  raises when `m != ndim` and silently returns wrong values when `m == ndim`: on the 2-D test
  grid, `rgi(pts.T)` gave `[0.2059 0.0456]` against the correct `[0.2260 0.0508]`.
- **Cubic splines overshoot.** On monotone or non-negative data that matters, use `Pchip`.
- **Global polynomial interpolation on equispaced nodes diverges as `n` grows.** More points
  makes it worse, not better.
- **`s` in `make_splrep` is not scale-free.** It is a residual sum of squares, so it changes
  meaning if you rescale `y`. Prefer `make_smoothing_spline`.
- **`griddata` returns `nan` outside the convex hull** and rebuilds the triangulation each
  call. `RBFInterpolator` extrapolates instead, which is a different failure mode, not a
  better one.
- **`x` must be strictly increasing** for the 1-D interpolators. Duplicated or unsorted `x`
  raises, and sorting `x` without sorting `y` alongside it silently fits garbage.
- **Differentiating an interpolant amplifies noise.** `cs.derivative()` on a spline fitted
  through noisy data is far worse than the spline itself. Smooth first, differentiate second.
- **`interp1d` is legacy.** `np.interp` for linear, `make_interp_spline` or `CubicSpline`
  otherwise.

## Quick reference

| task | call | note |
|---|---|---|
| linear, 1-D, fast | `np.interp(xq, x, y)` | no extrapolation, clamps to ends |
| smooth `C2` interpolant | `CubicSpline(x, y, bc_type=...)` | returns `PPoly` |
| no overshoot | `PchipInterpolator(x, y)` | `C1`, monotonicity preserving |
| fewer wiggles, local | `Akima1DInterpolator(x, y, method='makima')` | |
| any spline degree | `make_interp_spline(x, y, k=5)` | returns `BSpline` |
| smooth noisy data | `make_smoothing_spline(x, y)` | GCV picks the penalty |
| smooth with explicit control | `make_splrep(x, y, s=...)` | `s=0` interpolates |
| parametric curve fit | `make_splprep([x, y], s=...)` | closed or open curves |
| derivative of a fit | `f.derivative()(xq)` or `f(xq, 1)` | exact for the piecewise poly |
| integral of a fit | `f.integrate(a, b)` | exact |
| roots of a fit | `f.roots(extrapolate=False)` | per-interval closed form |
| function with poles | `AAA(x, y)`, then `.poles()` | rational, machine precision |
| regular grid, N-D | `RegularGridInterpolator(pts, V, method='cubic')` | query shape `(m, ndim)` |
| one-shot grid lookup | `interpn(pts, V, xq)` | |
| scattered, N-D | `RBFInterpolator(pts, vals, neighbors=50)` | `smoothing>0` to regress |
| scattered, one shot | `griddata(pts, vals, xq, method='cubic')` | `nan` outside the hull |
| suppress extrapolation | `extrapolate=False` / `bounds_error=True` | |
