# Quadrature: `scipy.integrate` for functions and for samples

Which call implements which rule, and what to do in the cases the error analysis assumes away: singularities, infinite limits, oscillation, high dimension. **Every number below is real output from this sandbox** (Python 3.11.15, numpy 2.4.4, scipy 1.17.1); code first, output in the fenced block after. Signatures come from `inspect.signature` on the installed library.

Every block in this file assumes:

```python
import numpy as np
from scipy.integrate import (quad, quad_vec, fixed_quad, tanhsinh, cubature, qmc_quad,
                             nsum, dblquad, tplquad, nquad, newton_cotes, romb,
                             trapezoid, simpson, cumulative_trapezoid, cumulative_simpson)
from scipy.special import roots_legendre, eval_legendre
```

Contents: The first fork; Sampled data (orders, even-count `simpson`, cumulative); `quad` (infinite limits, `points=`,
endpoint singularities, oscillatory weights); `fixed_quad`; Multidimensional; Series and rule generation; Gotchas;
Quick reference.

## The first fork: a callable, or an array?

Decide this before anything else. It selects a different half of the module and mixing the two up is the most common mistake.

| You have | Call | It can | It cannot |
|---|---|---|---|
| `f` you may evaluate at any `x` | `quad`, `tanhsinh`, `fixed_quad`, `cubature`, `quad_vec` | choose its own nodes, subdivide where the integrand is hard, return an error estimate | be used on data you already collected |
| `y` sampled at fixed `x` | `trapezoid`, `simpson`, `romb`, `cumulative_*` | apply one fixed rule to the nodes you supply | choose nodes, refine, or estimate its own error |

There is no error estimate anywhere in the sampled-data half. `simpson(y, x=x)` returns a bare float. If you want an error bar on sampled data you compute it yourself from two grids, or you go back and get a callable.

## Sampled data: the fixed rules

`np.trapz` was deprecated in NumPy 2.0 and removed in 2.4; the name is `np.trapezoid`, and `scipy.integrate.trapezoid` is the same rule with the same signature `(y, x=None, dx=1.0, axis=-1)`. Pass `x` for a non-uniform grid, `dx` for a uniform one. The rest:

| Call | Signature | Requires |
|---|---|---|
| `trapezoid` | `(y, x=None, dx=1.0, axis=-1)` | nothing; any spacing |
| `simpson` | `(y, x=None, *, dx=1.0, axis=-1)` | odd point count for full accuracy; any spacing |
| `romb` | `(y, dx=1.0, axis=-1, show=False)` | exactly `2**k + 1` points, **equally spaced** |
| `cumulative_trapezoid` | `(y, x=None, dx=1.0, axis=-1, initial=None)` | nothing |
| `cumulative_simpson` | `(y, *, x=None, dx=1.0, axis=-1, initial=None)` | nothing |

### The convergence orders, measured

One toy problem for the whole section: `int_0^1 e^x dx = e - 1`, sampled on grids of `n = 2**k + 1` points so that all three rules are legal on the same array.

```python
f, exact = np.exp, np.e - 1.0
prev_t = prev_s = None
for k in range(2, 9):
    n = 2**k + 1; x = np.linspace(0, 1, n); y = f(x); h = x[1] - x[0]
    et = abs(trapezoid(y, x) - exact)                       # (n,) y -> scalar
    es = abs(simpson(y, x=x) - exact)                       # (n,) y -> scalar
    er = abs(romb(y, dx=h) - exact)                         # (n,) y, uniform dx only
    ot = np.log2(prev_t/et) if prev_t else np.nan           # observed order = log2(err ratio)
    os_ = np.log2(prev_s/es) if prev_s else np.nan
    print(f"{n:5d} {h:9.2e} {et:10.3e} {ot:5.2f} {es:10.3e} {os_:5.2f} {er:10.3e}")
    prev_t, prev_s = et, es
```
```
    n         h   trap err   ord   simp err   ord   romb err      <- header added for reading; the loop prints no header
    5  2.50e-01  8.940e-03   nan  3.701e-05   nan  8.595e-07
    9  1.25e-01  2.237e-03  2.00  2.326e-06  3.99  3.355e-10
   17  6.25e-02  5.593e-04  2.00  1.456e-07  4.00  3.331e-14
   33  3.12e-02  1.398e-04  2.00  9.103e-09  4.00  2.220e-16
   65  1.56e-02  3.496e-05  2.00  5.690e-10  4.00  2.220e-16
  129  7.81e-03  8.740e-06  2.00  3.556e-11  4.00  0.000e+00
  257  3.91e-03  2.185e-06  2.00  2.223e-12  4.00  0.000e+00
```
`2.00` and `4.00` to two digits: composite trapezoid is `O(h^2)`, composite Simpson `O(h^4)` on this smooth integrand, exactly as derived. `romb` is Romberg, Richardson extrapolation up the trapezoid column, and it reaches roundoff at `n = 33` where Simpson still has `9e-09` to go. `romb(y, dx=h, show=True)` prints the extrapolation table itself, which is the fastest way to see whether the columns have stopped improving.

`romb` is fussy about the count and says so: `romb(np.exp(np.linspace(0,1,20)), dx=1/19)` raises `ValueError: Number of samples must be one plus a non-negative power of 2.`

### `simpson` with an even number of points

Simpson needs an odd point count (an even number of intervals). Given an even count, scipy 1.17.1 applies composite Simpson to the first `N-3` points and closes the last interval with a Cartwright correction: a quadratic fitted through the final three samples, integrated over the final interval only. There is no keyword to change this; the old `even='avg'/'first'/'last'` option is gone from the signature.

```python
prev = None
for n in [8, 16, 32, 64, 128]:                      # even counts -> odd interval counts
    x = np.linspace(0, 1, n); e = abs(simpson(np.exp(x), x=x) - (np.e - 1))
    print(f"n={n:4d} err {e:9.3e} order {np.log2(prev/e) if prev else np.nan:5.2f}"); prev = e
x9, x8 = np.linspace(0, 1, 9), np.linspace(0, 1, 8)
print("x**3, 9 pts:", simpson(x9**3, x=x9) - 0.25, " 8 pts:", simpson(x8**3, x=x8) - 0.25)
```
```
n=   8 err 4.486e-05 order   nan
n=  16 err 2.281e-06 order  4.30
n=  32 err 1.291e-07 order  4.14
n=  64 err 7.683e-09 order  4.07
n= 128 err 4.687e-10 order  4.03
x**3, 9 pts: 0.0  8 pts: 0.00010412328196585108
```
The asymptotic order survives, but the degree of precision does not: the parity bonus that makes Simpson exact on cubics is a property of the whole composite rule, and the odd-interval patch breaks it. `1.0e-04` on `x**3` where the odd grid gives exactly `0.0`. If your data has an even count, drop a point or resample rather than accept this.

### Cumulative integration

Same `e^x` data on `n = 9`. `cumulative_*` returns the running integral, one value per interval. `initial=` prepends that value so the output length matches the input, which is what you want whenever the result has to line up with `x`.

```python
x = np.linspace(0, 1, 9); y = np.exp(x)
print(cumulative_trapezoid(y, x=x).shape, cumulative_trapezoid(y, x=x, initial=0).shape)
c1 = cumulative_trapezoid(y, x=x, initial=0); cs = cumulative_simpson(y, x=x, initial=0)
print("max err trap", abs(c1 - (np.exp(x)-1)).max(), " simp", abs(cs - (np.exp(x)-1)).max())
```
```
(8,) (9,)
max err trap 0.0022367637052567170  simp 2.2508823652511722e-05
```

## Adaptive quadrature of a callable: `quad`

```
quad(func, a, b, args=(), full_output=0, epsabs=1.49e-08, epsrel=1.49e-08, limit=50,
     points=None, weight=None, wvar=None, wopts=None, maxp1=50, limlst=50,
     complex_func=False)
```

QUADPACK underneath: a 21-point Gauss-Kronrod rule on each subinterval, plus epsilon-algorithm extrapolation, bisecting whichever subinterval currently has the largest local error until the tolerance is met or `limit` subdivisions are used. It returns `(value, abserr)`. **`abserr` is an estimate, not a bound**: it is the difference between the Gauss and Kronrod sums, aggregated over subintervals, so it is only as honest as the sampling. The tolerance it is trying to meet is `|result - true| <= max(epsabs, epsrel*|result|)`; the defaults `1.49e-08` are roughly `sqrt(eps)`, and asking for less than that (`epsabs=1e-3`) does not make it stop early on an easy integrand.

```python
v, e, info = quad(np.exp, 0, 1, full_output=1)
print(v, e, "true err", abs(v - (np.e - 1)), "| neval", info['neval'], "subintervals", info['last'])
g = lambda x, a, b: np.exp(-a*x)*np.sin(b*x)                      # extra params via args=
print(quad(g, 0, np.inf, args=(2., 3.)), "exact 3/13 =", 3/13)
```
```
1.718281828459045 1.9076760487502454e-14 true err 0.0 | neval 21 subintervals 1
(0.2307692307692301, 6.740598871094127e-09) exact 3/13 = 0.23076923076923078
```
`full_output=1` returns a third dict with `neval`, `last` (subintervals used), and the arrays `alist`/`blist`/`rlist`/`elist`: the endpoints, result, and error estimate of every subinterval, ordered by `iord`. Read `elist` when you want to see *where* the difficulty is. Warnings are `scipy.integrate.IntegrationWarning`, a `UserWarning` subclass, so `warnings.simplefilter("error", IntegrationWarning)` turns silent failures into exceptions.

### Infinite limits: pass `np.inf`, never a big number

```python
print(quad(lambda x: np.exp(-x**2), -np.inf, np.inf), " sqrt(pi) =", np.sqrt(np.pi))
for L in [5, 500, 5000]:
    print(f"truncated at +-{L}:", quad(lambda x: np.exp(-x**2), -L, L))
```
```
(1.7724538509055159, 1.420263678094492e-08)  sqrt(pi) = 1.7724538509055159
truncated at +-5: (1.7724538509027912, 4.6261378229003154e-14)
truncated at +-500: (1.7724538509055159, 7.768259829669943e-09)
truncated at +-5000: (3.748541870780465e-50, 0.0)
```
`3.7e-50` with a reported error of exactly `0.0`. On an infinite limit `quad` applies a `x = (1-t)/t` transformation that concentrates nodes near the origin; truncate instead and the first Kronrod pass over `[-5000, 5000]` samples only points where the integrand has underflowed, sees a flat zero, and reports full confidence in it.

### An interior kink or spike: `points=`

`points` is a list of interior abscissae to force as subdivision boundaries. It is accepted only on finite intervals and only with `weight=None`.

```python
k = lambda x: abs(x - 0.3)                                        # kink at 0.3, exact 0.29
sp = lambda x: 1.0/(1.0 + ((x-0.3)/1e-5)**2)                      # width-1e-5 spike at 0.3
exs = 1e-5*(np.arctan(0.7/1e-5) + np.arctan(0.3/1e-5))            # its exact integral on [0,1]
for lbl, f, ex, kw in [("kink  plain", k, 0.29, {}), ("kink  points", k, 0.29, dict(points=[0.3])),
                       ("spike plain", sp, exs, {}), ("spike points", sp, exs, dict(points=[0.3]))]:
    out = quad(f, 0, 1, full_output=1, **kw)          # points= adds a 4th return
    v, e, info = out[0], out[1], out[2]
    print(f"{lbl}: {v!r:24s} abserr {e:9.2e} true err {abs(v-ex):9.2e} neval {info['neval']}")
```
```
kink  plain: 0.2899999999999998       abserr  3.22e-16 true err  1.67e-16 neval 357
kink  points: 0.29                    abserr  3.22e-15 true err  0.00e+00 neval 42
spike plain: -4.759602804818851e-10    abserr  4.14e-10 true err  3.14e-05 neval 399
spike points: 3.141545034539956e-05    abserr  8.92e-11 true err  2.23e-17 neval 1176
```
Two different lessons. On the kink, QUADPACK grinds to the right answer anyway, and `points` only buys cost: 42 evaluations instead of 357. On the spike it is the difference between right and wrong. Plain `quad` never places a node inside the `1e-5`-wide peak, returns `-4.8e-10`, and reports `abserr = 4.1e-10`, an error estimate smaller than the answer it missed by five orders of magnitude. Any feature narrower than the initial 21-node spacing is invisible to the error estimator, because the estimator only knows what it sampled.

### An endpoint singularity: `tanhsinh`, or `weight='alg'`

```
tanhsinh(f, a, b, *, args=(), log=False, maxlevel=None, minlevel=2, atol=None,
         rtol=None, preserve_shape=False, callback=None)
```

Double-exponential quadrature: the substitution `x = tanh(pi/2 sinh(t))` pushes the endpoints to `t = +-inf` and drives the transformed integrand to zero so fast that endpoint singularities stop mattering. It is the modern tool for this case and it is vectorized: `f` receives an array, and `args` may hold arrays, so one call solves a batch of problems.

```python
for lbl, f, ex in [("1/sqrt(x)", lambda x: 1/np.sqrt(x), 2.0),
                   ("log(x)/sqrt(x)", lambda x: np.log(x)/np.sqrt(x), -4.0)]:
    v, e, info = quad(f, 0, 1, full_output=1)
    r = tanhsinh(f, 0, 1)
    print(f"{lbl:15s} quad {v!r:20s} err {abs(v-ex):8.1e} neval {info['neval']:4d}"
          f" | tanhsinh {r.integral!r:20s} err {abs(r.integral-ex):8.1e} nfev {r.nfev} status {r.status}")
print("weight='alg', wvar=(-0.5, 0.0) on f(x)=1:", quad(lambda x: 1.0, 0, 1, weight='alg', wvar=(-0.5, 0.0)))
print("batched:", tanhsinh(lambda x, c: np.exp(-c*x), 0, np.inf, args=(np.array([1., 2., 4.]),)).integral)
```
```
1/sqrt(x)       quad 1.9999999999999984   err  1.6e-15 neval  231 | tanhsinh 1.9999999999999998   err  2.2e-16 nfev 67 status 0
log(x)/sqrt(x)  quad -4.000000000000085   err  8.5e-14 neval  315 | tanhsinh -3.9999999999999996  err  4.4e-16 nfev 67 status 0
weight='alg' wvar=(-0.5, 0.0) on f(x)=1: (2.0, 5.005088678075919e-14)
batched: [1.   0.5  0.25]
```
`quad`'s extrapolation copes with a mild algebraic singularity without warning, at three to five times the evaluations. `weight='alg'` with `wvar=(alpha, beta)` integrates `f(x) (x-a)**alpha (b-x)**beta` with the singular factor handled analytically, which is exact here. Strong singularities defeat `tanhsinh` at the default level: `tanhsinh(lambda x: x**-0.99, 0, 1)` returns `99.9158` against the true `100`, `status = -2` (maxlevel reached) with `nfev = 16387`, while `quad` gets `99.99999999999804`. Check `status` and `success`, not just `integral`.

### Oscillatory integrands: `weight='cos'`/`'sin'`

For `f(x) cos(w x)` or `f(x) sin(w x)`, pass the *non-oscillatory* factor as `func` and put `w` in `wvar`. QUADPACK then uses a modified Clenshaw-Curtis rule that integrates the trigonometric factor exactly instead of trying to resolve it with nodes.

```python
import warnings
def run(label, call):                       # print the result and any IntegrationWarning text
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        print(f"{label}:", call(), *(str(m.message) for m in w))

run("plain quad, w=50 ", lambda: quad(lambda x: np.exp(-x)*np.cos(50*x), 0, np.inf))
run("weight='cos', w=50", lambda: quad(lambda x: np.exp(-x), 0, np.inf, weight='cos', wvar=50.0))
run("sin/(1+x^2), limlst=3", lambda: quad(lambda x: 1/(1+x**2), 0, np.inf, weight='sin', wvar=1.0, limlst=3))
```
```
plain quad, w=50 : (0.00040454111362121694, 0.0004026568448006002)  The maximum number of
  subdivisions (50) has been achieved.   If increasing the limit yields no improvement it is
  advised to analyze the integrand ...
weight='cos', w=50: (0.00039984006397441257, 6.810342352492785e-10)      exact 1/2501 = 0.00039984006397441024
sin/(1+x^2), limlst=3: (0.648001311330067, 0.20010236992656616)  The maximum number of cycles
  allowed has been achieved., e.e.   of subintervals (a+(k-1)c, a+kc) where c =
  (2*int(abs(omega)+1))*pi/abs(omega), for k = 1, 2, ..., lst. ...
```
Plain `quad` returns `4.05e-04` with `abserr = 4.03e-04`, an error estimate the size of the answer, and warns. The weighted call is right to `7e-10`. On an infinite range the weighted rule integrates cycle by cycle and extrapolates; `limlst` caps the number of cycles (default 50), and `limlst=3` produces the second warning above with `ierlst` in `full_output` naming the cycle that failed. Raise `limlst`, do not truncate the range.

## Fixed-order Gauss-Legendre: `fixed_quad`

`fixed_quad(func, a, b, args=(), n=5)` is the reader's `n`-point Gauss-Legendre rule, mapped from `[-1, 1]` to `[a, b]`, degree of accuracy `2n-1`. No adaptivity, no error estimate; the second return value is always `None`.

```python
for d in [9, 10]:
    v, _ = fixed_quad(lambda x: x**d, 0, 1, n=5)                  # n=5 -> exact through degree 9
    print(f"n=5 on x**{d:2d}: {v!r} exact {1/(d+1)!r} err {abs(v - 1/(d+1)):.3e}")
x3, w3 = roots_legendre(3)                                        # == np.polynomial.legendre.leggauss(3)
print("nodes", x3, "weights", w3, "sum(w)", w3.sum())
print("P_3 at nodes:", eval_legendre(3, x3), " +-sqrt(3/5) =", np.sqrt(0.6))
```
```
n=5 on x** 9: 0.0999999999999999 exact 0.1 err 1.110e-16
n=5 on x**10: 0.09090765936004021 exact 0.09090909090909091 err 1.432e-06
nodes [-0.77459667  0.          0.77459667] weights [0.55555556 0.88888889 0.55555556] sum(w) 2.0
P_3 at nodes: [0.00000000e+00 0.00000000e+00 1.11022302e-16] +-sqrt(3/5) = 0.7745966692414834
```
Degree 9 to roundoff, degree 10 off by `1.4e-06`: the `2n-1` boundary, visible. `scipy.special.roots_legendre(n, mu=False)` and `np.polynomial.legendre.leggauss(deg)` return the identical `(nodes, weights)` on `[-1, 1]` when you need the rule rather than the integral; the nodes are the roots of `P_3` and the weights sum to the length of `[-1, 1]`. Map yourself with `xm = 0.5*(b-a)*x + 0.5*(b+a)`, `I = 0.5*(b-a) * (w*f(xm)).sum()`.

## Multidimensional

`dblquad(func, a, b, gfun, hfun, args=(), epsabs=1.49e-08, epsrel=1.49e-08)` and `tplquad(func, a, b, gfun, hfun, qfun, rfun, ...)` nest `quad` inside `quad`. **The integrand takes its arguments innermost-first**: `dblquad` calls `func(y, x)` where `x` is the outer variable ranging over `[a, b]` and `y` the inner, ranging over `[gfun(x), hfun(x)]`. Get it backwards and nothing raises.

```python
print("f(y, x) = x**2 * y :", dblquad(lambda y, x: x**2 * y, 0, 2, 0, 1)[0], "exact", (8/3)*0.5)
print("f(x, y) = x**2 * y :", dblquad(lambda x, y: x**2 * y, 0, 2, 0, 1)[0], "  <- integrates y**2 * x")
print("quarter disc:", dblquad(lambda y, x: 1.0, 0, 1, 0, lambda x: np.sqrt(1-x**2))[0], "pi/4 =", np.pi/4)
print("nquad:", nquad(lambda x0, x1, x2: x0*x1*x2, [[0,1]]*3)[0])
```
```
f(y, x) = x**2 * y : 1.3333333333333335 exact 1.3333333333333333
f(x, y) = x**2 * y : 0.6666666666666667   <- integrates y**2 * x
quarter disc: 0.7853981633974481 pi/4 = 0.7853981633974483
nquad: 0.12499999999999999
```
`gfun`/`hfun` may be constants or callables of the outer variable, which is how you get non-rectangular domains. `nquad(func, ranges, args=None, opts=None, full_output=False)` generalizes to `n` dimensions with `func(x0, x1, ..., xn)` outermost-first and per-axis `opts`.

`cubature(f, a, b, *, rule='gk21', rtol=1e-08, atol=0, max_subdivisions=10000, args=(), workers=1, points=None)` is the modern replacement: `a` and `b` are corner vectors, `f` is **vectorized** and receives shape `(npoints, ndim)` returning `(npoints,)`, and the result carries `.estimate`, `.error`, `.status`, `.subdivisions`. `qmc_quad(func, a, b, *, n_estimates=8, n_points=1024, qrng=None, log=False)` is quasi-Monte Carlo: `func` receives `(ndim, npoints)`, note the transposed convention, and the result gives `.integral` and `.standard_error`.

Timings on `int_{[0,1]^d} exp(-|x|^2)`:

```
d=2  nquad err  2.2e-16      0.5 ms | cubature err  1.1e-16     3.5 ms | qmc err  6.1e-07   29.3 ms sd 2.0e-06
d=3  nquad err  2.8e-16      7.1 ms | cubature err  2.2e-16     2.2 ms | qmc err  1.1e-05    3.8 ms sd 4.9e-06
d=4  nquad err  2.2e-16    153.7 ms | cubature err  1.7e-16   125.1 ms | qmc err  6.9e-06    4.4 ms sd 9.4e-06
d=5  nquad err  2.2e-16   3334.6 ms | cubature err  2.2e-16  4736.7 ms | qmc err  2.6e-05    5.2 ms sd 2.2e-05
```
`0.5 -> 7 -> 154 -> 3335` ms is a factor of about 21 per dimension, which is the 21-node Gauss-Kronrod rule tensored: cost is `n^d`, and the accuracy is machine precision you did not need. QMC stays flat at ~5 ms and converges as roughly `1/N` regardless of `d`. **Nested adaptive quadrature to `d = 3`, `d = 4` if the integrand is cheap; `qmc_quad` or plain Monte Carlo above that.** The crossover is where `n^d` evaluations cost more than the `1e-5` accuracy is worth.

## Series and rule generation

`nsum(f, a, b, *, step=1, args=(), log=False, maxterms=1048576, tolerances=None)` sums `f(k)` over an arithmetic index range, including `b=np.inf`, by direct summation plus Euler-Maclaurin integral correction. `f` must be non-negative and eventually monotonically decreasing for the tail bound to hold. `nsum(lambda k: 1/k**2, 1, np.inf).sum` gives `1.6449340668479238` against `pi**2/6 = 1.6449340668482264`, at `nfev = 8561`.

`newton_cotes(rn, equal=0)` returns `(an, B)` for the closed rule on `rn+1` equally spaced points, defined by `int = dx * sum(an*f(xi)) + B * dx**(N+2) * f^(N+1)(xi)`, and for even `N` the error term is `B * dx**(N+3) * f^(N+2)(xi)` instead. That parity switch is the even-`n` bonus, stated in the library's own convention.

```python
for n in [1, 2, 3, 4]:
    print(f"newton_cotes({n}, 1) -> {newton_cotes(n, 1)}")
```
```
newton_cotes(1, 1) -> (array([0.5, 0.5]), -0.08333333333333333)
newton_cotes(2, 1) -> (array([0.33333333, 1.33333333, 0.33333333]), -0.011111111111111112)
newton_cotes(3, 1) -> (array([0.375, 1.125, 1.125, 0.375]), -0.0375)
newton_cotes(4, 1) -> (array([0.31111111, 1.42222222, 0.53333333, 1.42222222, 0.31111111]), -0.008465608465608466)
```
Read them off: `N=1` is `[1,1]/2` with `B = -1/12`, so error `-(h^3/12) f''`. `N=2` is `[1,4,1]/3` with `B = -1/90`, and because `N` is even the error is `-(h^5/90) f^(4)`. `N=3` is Simpson 3/8, `[1,3,3,1]*3/8`. `rn` may also be an array of relative node positions for an unequally spaced rule.

`lebedev_rule(n)` returns `(x, w)` for integration over the unit sphere: `lebedev_rule(11)` gives `x.shape (3, 50)`, `w.shape (50,)`, `w.sum() == 4*pi` exactly.

## Gotchas

- **Treating `abserr` as a bound.** Symptom: a confident wrong answer. Cause: `abserr` is a Gauss-versus-Kronrod difference over the subintervals actually sampled; a feature narrower than the node spacing contributes to neither. Measured: a width-`1e-5` spike on `[0,1]` returned `-4.76e-10` with `abserr = 4.14e-10`, true error `3.14e-05`. Fix: know where your integrand is hard and pass `points=`; sanity-check against a second method or a plot.
- **`quad` on a discontinuous or oscillatory integrand.** Symptom: either a small `abserr` on garbage, or `abserr` the size of the result. Measured: `exp(-x)cos(50x)` on `[0, inf)` gave `4.05e-04 +- 4.03e-04` and warned `The maximum number of subdivisions (50) has been achieved.`; `weight='cos', wvar=50.0` gave `3.9984e-04 +- 6.8e-10`. Fix: split at the discontinuities with `points=`, or move the oscillation into `weight='cos'`/`'sin'` with `wvar`. Raise `limit` only after that.
- **Truncating an infinite limit.** Symptom: an answer of `0.0` or a garbage magnitude, with `abserr` exactly `0.0`. Measured: `exp(-x**2)` over `[-5000, 5000]` returned `3.75e-50`. Cause: the first Kronrod pass samples only where the integrand has underflowed. Fix: pass `np.inf`/`-np.inf` and let QUADPACK apply its transformation.
- **`simpson` with an even number of points.** Symptom: an error floor you cannot explain, or a cubic that is not integrated exactly. Cause: Simpson needs an odd count; scipy 1.17.1 patches the odd interval out with a quadratic through the last three samples. Measured: `x**3` on 8 points is off by `1.04e-04`, on 9 points by exactly `0.0`. The `even=` keyword no longer exists. Fix: sample an odd count.
- **`romb` on the wrong count.** `ValueError: Number of samples must be one plus a non-negative power of 2.` It also assumes uniform spacing and takes `dx`, not `x`, so a non-uniform grid is silently wrong. Fix: `simpson(y, x=x)` for anything that is not `2**k + 1` uniform samples.
- **`dblquad`'s reversed argument order.** Symptom: a plausible number that is wrong, no error. Cause: the integrand is called `func(y, x)`, innermost variable first, while the limits are given outermost first. Measured: `lambda y, x: x**2*y` over `x in [0,2], y in [0,1]` gives `1.3333`; the same body written `lambda x, y:` gives `0.6667`. Fix: name the first parameter after the innermost variable and check against a separable case you can do by hand.
- **`cumulative_*` output one element short.** Symptom: a broadcast error against `x`. Cause: `initial=None` is the default and returns `n-1` values. Fix: `initial=0`.
- **Nested `quad` above four dimensions.** Symptom: minutes per integral. Cause: `n^d`; measured `0.5 ms` at `d=2` rising to `3.3 s` at `d=5`, ~21x per dimension. Fix: `qmc_quad`, whose cost is flat in `d`, at `1e-5`-ish accuracy.

## Quick reference

| Task | Call | Note |
|---|---|---|
| samples, any spacing | `trapezoid(y, x=x)` | `np.trapezoid` is identical; `np.trapz` was removed in NumPy 2.4 (deprecated in 2.0) |
| samples, `O(h^4)` | `simpson(y, x=x)` | odd point count only; even counts lose exactness on cubics |
| samples, `2**k+1` uniform | `romb(y, dx=h, show=True)` | Romberg; hit roundoff at `n=33` where Simpson was at `9e-09` |
| running integral | `cumulative_trapezoid(y, x=x, initial=0)` | `initial=` keeps the length; `cumulative_simpson` is keyword-only past `y` |
| callable, general | `quad(f, a, b)` | `(value, abserr)`, `abserr` an estimate; `|err| <= max(epsabs, epsrel*|result|)` |
| infinite range | `quad(f, -np.inf, np.inf)` | never truncate; truncation returned `3.7e-50` with `abserr=0.0` |
| known interior trouble | `quad(f, a, b, points=[c, ...])` | finite interval, `weight=None` only; 42 evals vs 357 on a kink |
| endpoint singularity | `tanhsinh(f, a, b)` | vectorized `f`, batches over `args`; check `.status`. `quad(..., weight='alg', wvar=(al, be))` for `(x-a)**al (b-x)**be` |
| `f(x)cos(wx)`, `f(x)sin(wx)` | `quad(f, a, b, weight='cos', wvar=w)` | pass only the smooth factor; `limlst` caps cycles on an infinite range |
| Gauss-Legendre, fixed `n` | `fixed_quad(f, a, b, n=5)` | degree `2n-1`; second return is `None`, no error estimate |
| the nodes and weights | `roots_legendre(n)` / `np.polynomial.legendre.leggauss(n)` | identical, on `[-1, 1]`; map with `0.5*(b-a)*x + 0.5*(b+a)` |
| any closed rule's weights | `newton_cotes(n, 1)` | `n=2` -> `[1,4,1]/3`, `B=-1/90`; `n=3` -> Simpson 3/8 |
| 2-D, 3-D | `dblquad(f, a, b, gfun, hfun)`, `tplquad`, `nquad` | integrand is innermost-first: `f(y, x)`. Limits may be callables |
| N-D, vectorized | `cubature(f, a, b, rule='gk21', workers=)` | `f` takes `(npoints, ndim)`; `.estimate`, `.error`, `.status` |
| `d >= 5` | `qmc_quad(f, a, b, qrng=qmc.Sobol(d, seed=0))` | `f` takes `(ndim, npoints)`; `.standard_error`; flat cost in `d` |
| infinite series | `nsum(f, 1, np.inf)` | needs eventually monotone decreasing `f` |
| sphere surface | `lebedev_rule(n)` | `x` is `(3, npoints)`, `w.sum() == 4*pi` |
| vector-valued `f` | `quad_vec(f, a, b, workers=)` | one adaptive subdivision shared across all components |
