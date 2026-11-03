# Initial and Boundary Value Problems

Marching an ODE forward in time with `scipy.integrate.solve_ivp`, and solving a two-point
BVP with `solve_bvp`. Open this for method selection, stiffness, events, and the tolerance
knobs. The theory (LTE vs global error, order `p` from `O(h^{p+1})` per step, explicit vs
implicit, absolute stability regions, Dahlquist) is assumed known; this file is about which
call implements it and what goes wrong.

> Timings and speed ratios in this file: date and machine not recorded. They vary by machine.

## The interface

```python
solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False,
          events=None, vectorized=False, args=None, **options)
```

`fun(t, y)` returns `dy/dt` as an array of the same shape as `y`. `y0` must be
array-like even for a scalar problem: pass `[1.0]`, not `1.0`. `t_span` is a
`(t0, tf)` pair. Everything else is optional.

```python
import numpy as np
from scipy.integrate import solve_ivp, solve_bvp

def f(t, y): return -0.5 * y                 # y' = -y/2, y(0) = 1
sol = solve_ivp(f, (0, 10), [1.0], dense_output=True)

print(np.round(sol.t, 3))
print(sol.y.shape, np.round(sol.y[0][:4], 6))
print(sol.nfev, sol.njev, sol.nlu, sol.status, sol.success)
print(sol.sol(2.5), np.exp(-0.5 * 2.5))
```

```
[ 0.     0.115  1.264  3.061  4.817  6.576  8.338 10.   ]
(1, 8) [1.       0.944172 0.531588 0.216549]
44 0 0 0 True
[0.28642722] 0.2865047968601901
```

Read the shapes. `sol.y` is `(n_states, n_times)`, so state `i` over all time is
`sol.y[i]` and the state at time index `k` is `sol.y[:, k]`. The transpose of what
most people expect.

`sol.t` is where the adaptive controller chose to step, not a uniform grid. Eight
points over ten time units is not eight steps of accuracy; it is eight *accepted*
steps, and `nfev=44` is the real work. To get output on a grid of your choosing,
pass `t_eval=np.linspace(0, 10, 101)`, which interpolates onto that grid without
forcing the solver to step there. `dense_output=True` gives `sol.sol(t)`, a
continuous interpolant callable at any `t` in `t_span`, accurate to the method's
order.

`status` is `0` for reached the end, `1` for a terminal event fired, `-1` for
failure. Check `sol.success` before using `sol.y`. A failed solve returns whatever
it had, it does not raise.

## Method selection

The whole decision is one question: is the problem stiff?

| method | type | order | stiff | needs Jacobian | use when |
|---|---|---|---|---|---|
| `RK45` | explicit RK (Dormand-Prince) | 5(4) | no | no | default; smooth non-stiff problems |
| `RK23` | explicit RK (Bogacki-Shampine) | 3(2) | no | no | loose tolerances, cheap `f` |
| `DOP853` | explicit RK | 8(5,3) | no | no | `rtol < 1e-8`; smooth `f` |
| `Radau` | implicit Runge-Kutta (Radau IIA) | 5 | yes | helps | stiff, need high accuracy, L-stable |
| `BDF` | implicit multistep (backward differentiation) | 1-5, variable | yes | helps | stiff, large systems, cheap per step |
| `LSODA` | switches Adams <-> BDF automatically | variable | yes | helps | you do not know if it is stiff |

`BDF` is the linear multistep family from the coursework: variable order 1 through 5,
implicit. BDF methods are zero-stable through order 6 and unstable from order 7 (the
characteristic polynomial of BDF7 has a root of modulus 1.02); scipy's implementation caps
the order at 5. `Radau` is
a 3-stage implicit RK, so its Butcher `A` is full rather than strictly lower triangular
and every step solves a nonlinear system.

Stiffness is not a property you can read off `f`. It is the ratio of the fastest to the
slowest timescale, and the symptom is an explicit method taking absurdly many steps for
a solution that looks smooth. Van der Pol at `mu = 1000`:

```python
def vdp(t, y, mu):     return [y[1], mu*(1 - y[0]**2)*y[1] - y[0]]
def vdp_jac(t, y, mu): return [[0, 1], [-2*mu*y[0]*y[1] - 1, mu*(1 - y[0]**2)]]

for m in ("RK45", "LSODA", "BDF", "Radau"):
    kw = dict(args=(1000.0,), rtol=1e-6, atol=1e-8)
    if m != "RK45":
        kw["jac"] = vdp_jac
    s = solve_ivp(vdp, (0, 3000), [2.0, 0.0], method=m, **kw)
    print(m, len(s.t), s.nfev, s.njev, s.nlu)     # table below adds a header and wall time
```

```
method  steps      nfev   njev   nlu    wall time
RK45    1689291    11824598   0      0     66.778 s
LSODA      1818        3129  230    230     0.050 s
BDF        1648        5107  109    374     0.175 s
Radau      1188        9920  273    792     0.234 s
```

Same answer, 1300x the wall time. `RK45` is not converging slowly, it is stability-limited:
its absolute stability region is a bounded blob near the origin, so `h` is pinned to
`O(1/mu)` no matter how smooth the solution is. The implicit methods have unbounded stability
regions covering the left half plane, so `h` is set by accuracy alone. If a run is taking
millions of steps, that is the diagnosis, and the fix is a stiff method, not a looser `rtol`.

`args=` passes extra parameters to `fun` (and to `jac`, and to every event function).
Do not close over parameters in a lambda when you can pass them.

## Tolerances

`rtol` (default `1e-3`) and `atol` (default `1e-6`) control the local error estimate per
step, not the global error. The controller accepts a step when the estimated local error is
below `atol + rtol * |y|`, componentwise. Two consequences:

- The default `rtol=1e-3` is far too loose for anything quantitative. Three digits per step,
  accumulated over hundreds of steps, is not three digits at the end. Start at `rtol=1e-8,
  atol=1e-10` for scientific work and check that the answer stops moving.
- `atol` sets the scale below which a component is treated as zero. If one state is `O(1)`
  and another is `O(1e-9)`, a scalar `atol` ignores the small one. Pass `atol` as an array,
  one entry per state.

`rtol` is clamped from below at `100 * eps` (about `2.2e-14`) with a warning. Asking for
more than that is asking the controller to resolve round-off.

Error against `rtol` on `y' = -y/2` over `[0, 10]`, with `atol = rtol * 1e-3` so `atol` never binds. With the
default `atol=1e-6` the error stalls near `1.5e-7` (RK45) and `7.5e-9` (DOP853) from `rtol=1e-6` down, because
`y(10) = 6.7e-3` is small enough that `atol` controls the step:

```
rtol      RK45 err      DOP853 err
1e-03   1.597e-05    5.296e-07
1e-06   8.817e-09    1.519e-09
1e-09   7.468e-12    1.901e-12
1e-12   7.144e-15    2.056e-15
```

The achieved error tracks `rtol` closely and beats it here, because this problem is benign.
Do not rely on that. `rtol` is a request, not a bound.

## Events

An event is a scalar function `g(t, y)` whose sign change the solver locates by root-finding
on the dense interpolant, so the reported time is accurate to the interpolant, not to the step
grid. Two attributes control it:

- `g.terminal = True` stops the integration (or set it to an integer `k` to stop after `k`
  crossings).
- `g.direction = -1` counts only downward crossings, `+1` only upward, `0` (default) both.

```python
def ball(t, y): return [y[1], -9.81]
def hit_ground(t, y): return y[0]
hit_ground.terminal = True
hit_ground.direction = -1

s = solve_ivp(ball, (0, 100), [10.0, 0.0], events=hit_ground, rtol=1e-10, atol=1e-12)
print(s.t_events, s.y_events, s.status)
```

```
[array([1.42784312])] [array([[ 1.77635684e-15, -1.40071410e+01]])] 1
```

Exact answer `sqrt(2*10/9.81) = 1.4278431229270645`: eleven correct digits on the crossing time from a `t_span`
that ran to 100. `t_events` and
`y_events` are lists with one entry per event function, each entry an array of all crossings.

## Speeding up implicit solves

For an implicit method the cost is dominated by forming and factoring the Jacobian, not by
evaluating `f`. Two levers:

**`jac=`** supplies the analytic Jacobian `df/dy`. Without it the solver finite-differences,
costing `n` extra `f` evaluations per Jacobian and introducing the `sqrt(eps)` accuracy floor
from `04-floating-point.md`.

**`jac_sparsity=`** supplies a boolean/sparse matrix marking which entries can be nonzero.
The solver then colors the columns and gets the whole Jacobian in a handful of `f`
evaluations instead of `n`. For a method-of-lines discretization the sparsity is the stencil,
which you already have as the finite-difference matrix. Fisher-KPP, `u_t = u_xx + u(1-u)`,
`n = 600` grid points:

```python
from scipy.sparse import diags_array
n = 600; h = 1.0/(n+1)
T = diags_array([np.ones(n-1), -2*np.ones(n), np.ones(n-1)],
                offsets=[-1, 0, 1], format="csr") / h**2
def fk(t, u): return T @ u + u*(1 - u)
u0 = np.exp(-100*(np.linspace(h, 1-h, n) - 0.5)**2)

solve_ivp(fk, (0, 0.5), u0, method="BDF")                     # dense FD Jacobian
solve_ivp(fk, (0, 0.5), u0, method="BDF", jac_sparsity=T)     # banded structure known
```

Timed with `time.perf_counter` around each call, printing `nfev`, `njev`, `nlu`:

```
dense FD jac : nfev=126 njev=1 nlu=17 time=0.198s
jac_sparsity : nfev=126 njev=1 nlu=17 time=0.017s
```

12x, same step sequence, same accuracy. On a 2-D or 3-D PDE the gap is much larger.

`vectorized=True` tells the solver `fun` accepts a `(n, k)` block of states and returns
`(n, k)`, which lets it batch the finite-difference Jacobian columns. It only helps when
`jac` is absent, and it is a promise the solver does not check: a `fun` that assumes 1-D `y`
will produce wrong Jacobians silently.

## Systems, complex values, and the old interface

A system is the same call with a longer `y0`. Order the states however you like; `fun` just
has to be consistent. Second-order equations get reduced to first-order by hand, exactly as
in the coursework: `y'' = g(t, y, y')` becomes `u = [y, y']`, `u' = [u[1], g(t, u[0], u[1])]`.

Complex-valued problems work with `RK23`, `RK45`, `DOP853`, and `BDF`: pass a complex `y0`. `Radau` and
`LSODA` raise `ValueError: \`y0\` is complex, but the chosen solver does not support integration in a complex
domain.` For a stiff complex problem, split into real and imaginary parts (`z = [Re y, Im y]`, Jacobian
`[[Re A, -Im A], [Im A, Re A]]`) and use `LSODA` or `Radau` on the real system. `BDF` accepts complex `y0`, but its
orders 3 to 5 are not A-stable near the imaginary axis, which is where oscillatory (Schrodinger-type) eigenvalues
sit: on `y' = (-0.001 + 10i) y` over `[0, 10]` at `rtol=1e-8` it took 1996 steps with error 5.5e-6, against 180 steps
and 6.3e-8 for `DOP853` (scipy 1.17.1). On scipy 1.17.1, `LSODA` with `jac=` given as an ndarray raises
`ValueError: The truth value of an array ... is ambiguous`; pass `jac=lambda t, y: J` instead. `solve_bvp` also
accepts complex problems: pass a complex initial guess `y`.

`odeint` is the older LSODA wrapper. Its `fun` signature is `f(y, t)`, arguments reversed
(unless you pass `tfirst=True`), it has no events and no method selection, and it returns a
bare array rather than a result object. It still works. New code should use `solve_ivp`.

## Boundary value problems

```python
solve_bvp(fun, bc, x, y, p=None, S=None, fun_jac=None, bc_jac=None,
          tol=0.001, max_nodes=1000, verbose=0, bc_tol=None)
```

A collocation solver on an adaptive mesh, not a shooting method. Four things it needs:

- `fun(x, y)` must be **vectorized**: `x` is `(m,)`, `y` is `(n, m)`, and the return is
  `(n, m)`. This is not optional, unlike `solve_ivp`'s `vectorized=`.
- `bc(ya, yb)` returns `n` residuals that should be zero at the solution, where `ya = y(x[0])`
  and `yb = y(x[-1])`. Dirichlet `u(0)=0, u(1)=0` is `[ya[0], yb[0]]`.
- `x` is an initial mesh, `y` an initial guess of shape `(n, x.size)`. Both get refined.
- The result carries `sol.sol(x)`, a continuous cubic-spline interpolant.

The model problem `u'' = -pi^2 sin(pi x)` with `u(0) = u(1) = 0`, exact solution `sin(pi x)`:

```python
def rhs(x, y): return np.vstack([y[1], -np.pi**2 * np.sin(np.pi*x)])
def bc(ya, yb): return np.array([ya[0], yb[0]])

x = np.linspace(0, 1, 5)
y0 = np.zeros((2, x.size))
b = solve_bvp(rhs, bc, x, y0)
xx = np.linspace(0, 1, 1001)
print("status", b.status, "| nodes used", b.x.size, "| max error %.2e" % np.abs(b.sol(xx)[0] - np.sin(np.pi*xx)).max())
```

```
status 0 | nodes used 11 | max error 7.27e-05
```

Five initial nodes refined to eleven, `tol=1e-3` requested, `7e-5` delivered. `status` is
`0` for success, `1` for `max_nodes` exceeded, `2` for a singular Jacobian. A nonlinear BVP
needs an initial guess in the right basin; a guess of zeros converges to the trivial solution
when one exists.

For a linear BVP on a fixed grid, assembling the tridiagonal finite-difference matrix and
calling `scipy.linalg.solve_banded` is faster and gives you the `O(h^2)` error you can
analyze directly. See `06-sparse-iterative.md` for the sparse version and `05-linalg-dense.md`
for the banded storage layout.

## Gotchas

- **`sol.y` is `(n_states, n_times)`.** Plotting `sol.y` against `sol.t` without a transpose
  or an index silently plots the wrong thing when `n_states > 1`.
- **`y0` must be 1-D.** `solve_ivp(f, (0,1), 1.0)` and a 2-D `y0` both raise `` ValueError: `y0` must be
  1-dimensional. ``; pass `[1.0]`.
- **Default `rtol=1e-3` is a demo tolerance.** Any result you intend to trust needs it
  tightened and re-run to confirm the answer stopped moving.
- **A failed solve does not raise.** `status=-1` returns a partial trajectory. Check
  `sol.success`.
- **`t_eval` does not improve accuracy.** It only chooses output points; the step sequence is
  unchanged. Passing a dense `t_eval` to "make it more accurate" does nothing but interpolate.
- **`t_eval` values must lie inside `t_span`** and be sorted in the direction of integration.
  Backwards integration (`t_span=(10, 0)`) needs a descending `t_eval`.
- **Millions of steps means stiffness, not a bad tolerance.** Switch to `BDF`/`Radau`/`LSODA`
  before touching `rtol`.
- **`vectorized=True` is an unchecked promise.** If `fun` cannot handle a `(n, k)` argument
  the finite-differenced Jacobian is wrong and the solver converges to the wrong answer or
  stalls.
- **Event functions must be smooth and change sign.** An event that only touches zero
  (`g = y**2`) is never detected; the root-finder needs a bracket.
- **`solve_bvp`'s `fun` must be vectorized** and must return shape `(n, m)`. Returning a list
  of scalars works for `solve_ivp` and fails here.
- **`odeint` reverses the argument order** to `f(y, t)`. Copying an `odeint` right-hand side
  into `solve_ivp` without swapping produces a plausible-looking wrong answer.

## Quick reference

| task | call | note |
|---|---|---|
| non-stiff IVP | `solve_ivp(f, (t0,tf), y0, rtol=1e-8, atol=1e-10)` | `RK45` default |
| very high accuracy | `method="DOP853"` | worth it below `rtol=1e-8` |
| stiff, unknown | `method="LSODA"` | auto-switches Adams/BDF; real `y0` only |
| stiff, large system | `method="BDF", jac_sparsity=S` | `S` is the stencil matrix |
| stiff, high accuracy | `method="Radau"` | L-stable implicit RK |
| output on a grid | `t_eval=np.linspace(...)` | interpolated, not enforced |
| continuous solution | `dense_output=True` then `sol.sol(t)` | order-accurate interpolant |
| stop at a condition | `g.terminal = True; events=g` | root-found on the interpolant |
| one-way crossings | `g.direction = ±1` | |
| pass parameters | `args=(mu,)` | reaches `fun`, `jac`, and events |
| per-state tolerance | `atol=np.array([...])` | for states of different magnitude |
| analytic Jacobian | `jac=jfun` | implicit methods only |
| two-point BVP | `solve_bvp(fun, bc, x, y0)` | `fun` vectorized, returns `(n, m)` |
| BVP solution | `b.sol(x)` | cubic spline, `(n, len(x))` |
| check it worked | `sol.success`, `sol.status`, `sol.message` | never assume |
