# Vectorization and performance

How to make numpy/scipy code fast and how to tell when it already is: the interpreter-overhead cost model, loop-to-array rewrites, the price of temporaries, memory layout and BLAS, and where to profile instead of guess.

**Every timing below came from a `timeit` run in this sandbox** (Python 3.11.15, numpy 2.4.4, scipy 1.17.1, scipy-openblas 0.3.31.188.0 `USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell`, 2 vCPU, L2 4 MiB, L3 260 MiB). Absolute milliseconds are machine-specific and will not reproduce elsewhere. **Only the ratios transfer**, and even those shift with core count and cache size. Output is pasted verbatim in a fenced block; where a timing table stands alone, the code was `timeit.repeat` around the expressions named in it.

Every block in this file assumes:

```python
import numpy as np
import timeit
rng = np.random.default_rng(0)
```

## The cost model

Two numbers explain almost every numpy performance question. A Python-level loop over array elements costs **tens to hundreds of nanoseconds per element** in interpreter overhead alone: bytecode dispatch, a boxed `float` per value, refcount traffic. A vectorized ufunc costs roughly the **memory bandwidth of the arrays it touches**, order 1 ns per element, because the loop body is compiled C over a contiguous buffer.
So the first question is never "which numpy function is fastest". It is **how many times does the Python interpreter touch an individual element**. Zero means you are near the floor and further tuning buys percent, not multiples. Once per element means you are paying 20x to 200x and no choice of numpy function fixes it.

```python
import numpy as np, timeit
x = np.random.default_rng(0).random(1_000_000); xl = x.tolist()
def loop():
    s = 0.0
    for v in xl: s += v*v          # the only Python-level touch of an element in this file
    return s
for name, f in [('python for-loop', loop), ('generator sum', lambda: sum(v*v for v in xl)),
                ('(x**2).sum()', lambda: (x**2).sum()), ('np.dot(x, x)', lambda: np.dot(x, x))]:
    print(f'{name:20s} {min(timeit.repeat(f, number=1, repeat=5))*1e3:9.3f} ms')
```

```
python for-loop         22.571 ms      22.6 ns/elt
generator sum           28.052 ms      28.1 ns/elt
(x**2).sum()             1.106 ms       1.1 ns/elt
np.dot(x, x)             0.149 ms       0.15 ns/elt
```

**20x from vectorizing, another 7x from picking the call that fuses the multiply into the reduction** instead of materializing `x**2` first. 22.6 ns is the *floor* for a Python loop, with the simplest possible body; a body with a branch and a function call measured 183 ns per element (see `np.vectorize` below).

## Vectorization patterns

Each row is a loop shape and the array expression that replaces it. Valid only when the loop body does not depend on an earlier iteration's result (see the last section for when it does).

| Loop shape | Vectorized replacement | Note |
|---|---|---|
| `out = []` then `out.append(f(v))` | `np.array([...])` once, or `np.fromiter(gen, dtype, count=n)` | never grow an array in the loop |
| `if cond: y=a else: y=b` per element | `np.where(cond_arr, a, b)` | both branches are evaluated on all elements |
| `if cond: total += v` | `arr[mask].sum()` or `(arr*mask).sum()` | `np.count_nonzero` is faster for counting alone |
| nested loop over point pairs | `x[:, None, :] - y[None, :, :]`, or `scipy.spatial.distance.cdist` | broadcasting builds an `(n, m, d)` temporary; `cdist` does not |
| `totals[g[i]] += w[i]` (group-by) | `np.bincount(g, weights=w, minlength=k)` | `np.add.at(out, idx, w)` for non-1-D or non-int indices |
| overlapping windows of length `W` | `np.lib.stride_tricks.sliding_window_view(x, W)` | a strided view, not a copy |
| `arr = np.append(arr, v)` in a loop | collect in a `list`, `np.concatenate` once | the loop form is O(n^2), see Gotchas |

`sliding_window_view` is the least known: it fabricates a 2-D array out of a 1-D buffer without copying.
```python
swv = np.lib.stride_tricks.sliding_window_view    # (x, window_shape, axis=None, *, subok=False, writeable=False)
x = np.arange(10.); w = swv(x, 4)
print(w.shape, w.strides, x.strides, w.flags['OWNDATA'], w.flags['WRITEABLE']); print(w[:2]); print(w.mean(axis=-1)[:3])
```

```
(7, 4) (8, 8) (8,) False False
[[0. 1. 2. 3.]
 [1. 2. 3. 4.]]
[1.5 2.5 3.5]
```

Both strides are 8 bytes, one float64: advancing a row and advancing a column move by the same single element, which is exactly what an overlapping window is. It is read-only because one buffer element backs `W` positions. `nbytes` reports the *virtual* size and lies about memory: for `x` of 2e6 floats with `W=50`, `w.nbytes` is 762.9 MiB while the real allocation stays at `x.nbytes` = 15.3 MiB (`tracemalloc` peak during `w.mean(axis=-1)` was 15.3 MiB). The reduction still reads every window element, so it does `W` times the memory traffic. When the window statistic telescopes, use the closed form:

```
sliding_window_view(x, 50).mean(-1)     62.05 ms
(cumsum[W:] - cumsum[:-W]) / W          10.79 ms   6x faster
```

## What vectorization costs: temporaries

`a**2 + b**2 + c**2` allocates three squared arrays and two partial sums, then frees them. At n=5e6 float64 that is five 38 MiB allocations that never fit in cache. The fixes are `out=` on the ufunc, in-place `+=`, and `np.einsum`, which fuses a multiply-and-reduce without building the product array.

```python
n = 1_000_000
a, b, c = rng.random(n), rng.random(n), rng.random(n); out, tmp = np.empty(n), np.empty(n)
def naive(): return a**2 + b**2 + c**2
def ip():                                                        # np.allclose(naive(), ip()) -> True
    np.multiply(a, a, out=out)
    np.multiply(b, b, out=tmp); out += tmp
    np.multiply(c, c, out=tmp); out += tmp; return out
```

```
a**2 + b**2 + c**2            31.18 ms
out=/+= preallocated          23.34 ms      1.34x
np.einsum('i,i->', ...) x3     9.94 ms      scalar total, no array built
a@a + b@b + c@c                2.27 ms      scalar total, BLAS ddot
```

**1.34x, for five lines and two scratch buffers instead of one expression.** That is the honest trade: take it in an inner loop that runs thousands of times, not in setup code. The last two rows are a different computation (scalar, not array) and make the point that if you only need the reduction, never materialize the elementwise result. The win is also not stable across sizes, and it disappears when the arrays fit in cache:

```
n=      1,000  bytes=   0.0 MiB  naive 3.31 ns/elt  out= 2.93 ns/elt  speedup 1.13x
n=    100,000  bytes=   2.3 MiB  naive 2.25 ns/elt  out= 3.03 ns/elt  speedup 0.74x
n=  2,000,000  bytes=  45.8 MiB  naive 4.26 ns/elt  out= 4.11 ns/elt  speedup 1.04x
n= 30,000,000  bytes= 686.6 MiB  naive 8.97 ns/elt  out= 7.63 ns/elt  speedup 1.17x
```

At 100k the in-place version is *slower*: the working set fits in L2, the temporaries are cheap, and the extra call overhead dominates. Per-element cost quadruples from 2.25 ns to 8.97 ns as the working set outgrows cache. Once on that plateau you are bandwidth-bound, eliminating temporaries is the only lever left, and its ceiling is the ratio of bytes moved, not a large factor.

## Memory layout and BLAS

**C-order** (row-major, numpy's default) stores the last axis contiguously: for `(4000, 4000)` float64 the strides are `(32000, 8)`, so `M[i, j]` and `M[i, j+1]` are adjacent bytes. **F-order** (column-major, the Fortran and LAPACK convention) stores the first axis contiguously. `M.T` is a view of the same bytes with strides reversed, so a transposed C-order array is F-contiguous and costs nothing to produce.
The folklore is that row sums beat column sums on a C-order array. **On numpy 2.4.4 that is backwards for reductions:**

```
M.sum(axis=1)  contiguous last axis     14.58 ms
M.sum(axis=0)  contiguous inner rows     6.57 ms
T.sum(axis=1)  strided last axis         6.51 ms   (T = M.T, same bytes, same values)
```

Reducing over axis 0 walks whole contiguous rows and adds each into a length-4000 accumulator: pure SIMD, sequential reads, no horizontal add. Reducing over the last axis does a pairwise summation per row ending in a horizontal reduction of a vector register, which is the slower shape. Both stream memory in address order, so layout barely enters. Layout does bite on elementwise work over a strided view: `M[:, ::2]*2` took 25.80 ms against 31.51 ms for `M*2`, so **1.6x per element** for reading every other value.

`@` and `np.dot` on 2-D float64 arrays run no numpy loop at all. They dispatch to the BLAS routine `dgemm` (**d**ouble, **ge**neral **m**atrix **m**ultiply), a hand-tuned cache-blocked kernel that is already multithreaded. It takes transpose flags, so an F-order operand is free:

```
A @ B          (BLAS dgemm)      14.87 ms   134.46 GFLOP/s      N=1000
A @ B.T (F-order rhs)            14.74 ms
float32 A @ B                     7.10 ms   sgemm, half the bytes, ~2x
```

134 GFLOP/s on 2 vCPU means the kernel is at hardware peak. **Code already inside `dgemm` cannot be optimized by you.** Read the BLAS in use with `np.show_config(mode='dicts')['Build Dependencies']['blas']`, or with `threadpoolctl`, **installed here at 3.6.0**:

```python
import threadpoolctl
[{k: p[k] for k in ('user_api','internal_api','version','num_threads')} for p in threadpoolctl.threadpool_info()]
# -> [{'user_api': 'blas', 'internal_api': 'openblas', 'version': '0.3.31.188.0', 'num_threads': 2}]
with threadpoolctl.threadpool_limits(limits=1, user_api='blas'):    # scoped, restores on exit
    ...                                                             # BLAS threads=1: 30.31 ms, =2: 15.70 ms (1.93x)
```

`OMP_NUM_THREADS` (also `OPENBLAS_NUM_THREADS`) does the same thing but **must be set before numpy is imported**; `threadpool_limits` works at runtime and is scoped. **BLAS threads and process pools multiply.** A `multiprocessing.Pool(8)` whose workers each call `@` with 8 BLAS threads asks for 64 threads on 8 cores; the context switching makes it slower than serial. Set BLAS to 1 thread inside pool workers.

## `np.vectorize` is not vectorization

It is a convenience wrapper that calls your Python function once per element. The numpy docstring says so: "The implementation is essentially a for loop."

```python
def scalar_f(v): return v*v + 3.0 if v > 0.5 else 0.0
vf = np.vectorize(scalar_f)      # (pyfunc, otypes=None, doc=None, excluded=None, cache=False, signature=None)
```

```
np.vectorize(f)(x)                   149.43 ms     n = 1e6
np.where(x > 0.5, x*x + 3.0, 0.0)      6.56 ms     np.vectorize is 23x slower
list comp + np.array                 183.12 ms     np.vectorize is 0.82x this, i.e. the same thing
```

It lands within 20% of a plain list comprehension because it *is* one. What it is for: giving a scalar-only function (a special function you wrote, a `scipy.optimize.brentq` call per element) numpy's broadcasting and shape semantics so it accepts arrays and returns the right shape. Convenience and correctness, never speed. One trap: without `otypes=` it infers the output dtype from the first element's result, so `np.vectorize(lambda v: 0 if v <= 0.5 else v*v + 3.0)(np.array([0.2, 0.7]))` returns `int64` `[0 3]`, silently truncating `3.49`. Pass `otypes=[float]` to pin it.

## Choosing the right call

Same result, different cost. Every ratio measured here.

| Task | Slow | Fast | Measured |
|---|---|---|---|
| matmul `(1000,1000)` | `np.einsum('ij,jk->ik', A, B)` | `A @ B` | 287.61 vs 14.87 ms, **19.3x** |
| matmul `(300,300)` | `(A[:,:,None]*B[None,:,:]).sum(1)` | `A @ B` | 129.27 vs 0.50 ms, **260x**, plus a 206 MiB temporary |
| solve `Ax=b`, N=1200 | `sla.inv(A) @ b` | `sla.solve(A, b)` | 114.41 vs 27.48 ms, **4.2x** |
| membership, 2000 probes | `p in arr` in a loop | `np.isin(probe, arr)` | 9.60 ms for **200** probes vs 0.82 ms for 2000 |
| top-10 of 5e6 | `np.sort(y)[-10:]` | `y[np.argpartition(y, -10)[-10:]]` | 46.01 vs 17.15 ms, **2.7x** (O(n) vs O(n log n)) |
| pairwise dist 2000x2000 | broadcast then `sqrt(sum)` | `scipy.spatial.distance.cdist(P, Q)` | 136.57 vs 8.87 ms, **15.4x** |
| build an array of n values | `np.append` in a loop | `np.array(list)` | n=80000: 902.68 vs 3.69 ms, **245x** |
| group sums, 2e6 into 500 | `np.add.at(out, g, w)` | `np.bincount(g, weights=w)` | 2.83 vs 2.95 ms, **equal on 2.4.4** |

`np.einsum` without `optimize=True` runs its own C loop and never calls BLAS, which is the 19x; with `optimize=True` it plans the contraction and hands pairwise products to BLAS, so it matches `@` and is the right tool only for contractions `@` cannot express. `inv(A) @ b` is slower and, in general, less accurate; here both residuals landed near 1.5e-14 because the test matrix was diagonally dominant, so only the speed gap showed. Prefer `solve` regardless. `np.add.at` was historically far slower than `bincount`; on 2.4.4 they measured the same for float weights into 500 bins.

## Profiling

Do not guess the hot line. Measure in this order.

| Tool | Use for | Call |
|---|---|---|
| `timeit.repeat` | one expression | `min(timeit.repeat(f, number=n, repeat=5))/n`, take the **min**, not the mean |
| `cProfile` + `pstats` | which *function* in a script | `cProfile.run('main()', 'out.prof')`, then `pstats.Stats('out.prof').sort_stats('tottime')` |
| `line_profiler` | which *line* in a function | `@profile` + `kernprof -l -v`. **Not installed here** (`pip install line_profiler`) |
| `tracemalloc` | peak Python-side allocation | `start()`, then `get_traced_memory()` returns `(current, peak)` |

Sort by `tottime` (time in the function itself), not `cumtime` (which includes callees and puts `main` on top). Numpy calls appear as C functions with no line detail, which is itself the signal that time is inside a kernel and not in your code. Real output of `cProfile.Profile()` around a function that calls a Python-loop norm and a `np.sqrt(a @ a)` norm three times each, then `pstats.Stats(pr).sort_stats('tottime').print_stats(6)`:

```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        3    0.158    0.053    0.158    0.053 b9.py:4(slow_norm)
        3    0.000    0.000    0.000    0.000 b9.py:8(fast_norm)
        1    0.000    0.000    0.159    0.159 b9.py:9(work)
```

`slow_norm` (a Python loop) is 100% of `tottime`; `fast_norm` (`np.sqrt(a @ a)`) rounds to zero on the same 400k input. cProfile adds per-call overhead, so it distorts many-small-calls code and is useless for timing a single ufunc. Use `timeit` for that.

## When to leave numpy

If the algorithm is genuinely sequential, vectorization cannot help. An ODE time step, a Newton iteration, a Gauss-Seidel sweep, and an adaptive stepsize controller all need iteration `k` before they can start `k+1`. No array expression exists. In order:

1. **Restructure so the inner dimension is vectorized and only the sequential dimension loops.** Try first: usually available, no new dependency.
2. **`numba.njit`**, which JIT-compiles the loop to machine code. **Not installed here** (`pip install numba`).
3. **Cython**, ahead-of-time C from annotated Python, worth it for a stable kernel you ship. **Not installed here.**
4. **C or Fortran behind a small extension.** Last resort.

The restructuring that covers most ODE and iterative-solver code is **vectorize across the ensemble, loop over time**. State becomes an array over trajectories (or grid points, or parameter samples), the time loop stays Python, and each step is one array expression. Python overhead is then paid once per *step*, not once per step per trajectory.

```python
M, T, dt = 20_000, 500, 1e-3
k = rng.random(M) + 0.5            # (M,)  one rate per trajectory
y = np.ones(M)                     # (M,)  state of the whole ensemble
for _ in range(T):                 #       sequential axis stays a Python loop
    y -= dt * k * y                # (M,)  one vectorized Euler step, in place
```

```
M=400   trajectories, nested python loops    39.1 ms  ->  97.700 us/trajectory
M=20000 trajectories, vectorized ensemble    11.0 ms  ->   0.549 us/trajectory
per-trajectory speedup: 178x     (values agree: True)
```

500 Python iterations total instead of 500 x 20000. The same shape covers a stencil sweep (loop sweeps, vectorize the grid) and `scipy.integrate.solve_ivp` with a vector-valued `fun` (pack the ensemble into `y`, return the stacked derivative). It does not apply when trajectories need different step counts, which is why an adaptive solver over an ensemble usually still has to loop.

## Gotchas

- **`np.append` or `np.concatenate` inside a loop is O(n^2).** Symptom: runtime grows far faster than input size. `np.append` always allocates a new array and copies everything, so building n elements copies n(n+1)/2. Measured: n=5000 -> 11.79 ms, n=20000 -> 79.73 ms, n=80000 -> 902.68 ms, an 11x cost for a 4x input. Fix: append to a Python `list`, then `np.array(list)` or `np.concatenate(list_of_arrays)` once. If the final length is known, preallocate `np.empty(n)` and assign into slices. `np.fromiter(gen, dtype, count=n)` handles generators without the intermediate list.
- **Optimizing code that is already BLAS-bound.** Symptom: hours of tuning buy 2%. If `cProfile` shows the time inside `@`, `np.dot`, `scipy.linalg.*`, or an FFT, you are in a hand-tuned kernel at hardware peak (134 GFLOP/s here). The only real levers are a smaller problem, float32 instead of float64 (~2x, 7.10 vs 14.87 ms), exploiting structure (`assume_a='pos'` on `scipy.linalg.solve`, a sparse format, a factorization reused across right-hand sides), or more cores.
- **Benchmarking with a cold cache, or on an array too small to matter.** Symptom: the ratio flips when you change n. The first call includes page faults and lazy allocation, so use `timeit.repeat(..., repeat=5)` and take the **min**, the warm run. Below roughly 10k elements the ~1 us of numpy call overhead dominates and every option looks the same. Above cache size the answer changes again: the temporaries benchmark above gave 0.74x at n=100k and 1.17x at n=30M for identical code. **Benchmark at the size you will actually run.**
- **BLAS threads multiply with process pools.** Symptom: a `multiprocessing` or `joblib` version is slower than serial with every core at 100% system time. Each worker inherits the full BLAS thread count, so P workers x T threads oversubscribe the machine. Fix: `with threadpoolctl.threadpool_limits(limits=1, user_api='blas'):` around the worker body, or `OMP_NUM_THREADS=1` before import in the workers. Parallelize at one level only.
- **A strided view is cheap to build and expensive to reduce.** `sliding_window_view` allocates nothing, but its `nbytes` (762.9 MiB above, backed by 15.3 MiB) is virtual and any reduction over it does W times the memory traffic. Prefer a `cumsum` identity when the statistic has one: 6x here.
- **`np.trapz` was deprecated in NumPy 2.0 and removed in 2.4.** Use `np.trapezoid`.

## Quick reference

| Task | Call | Note |
|---|---|---|
| elementwise if/else | `np.where(cond, a, b)` | `(condition, x=None, y=None, /)`; both branches computed on all elements |
| group-by sum | `np.bincount(g, weights=w, minlength=k)` | `(x, /, weights=None, minlength=0)`; `g` non-negative int |
| overlapping windows | `np.lib.stride_tricks.sliding_window_view(x, W)` | view, read-only, `nbytes` is virtual |
| pairwise distances | `scipy.spatial.distance.cdist(XA, XB)` | `(XA, XB, metric='euclidean', *, out=None, **kwargs)`; no O(nmd) temporary |
| matmul | `A @ B` | BLAS `dgemm` on 2-D float64, already multithreaded, transpose free |
| general contraction | `np.einsum(spec, *ops, optimize=True)` | without `optimize=True` it never calls BLAS |
| linear solve | `scipy.linalg.solve(A, b)` | `(a, b, lower=False, overwrite_a=False, overwrite_b=False, check_finite=True, assume_a=None, transposed=False)`; never `inv(A) @ b` |
| top-k | `x[np.argpartition(x, -k)[-k:]]` | `(a, kth, axis=-1, kind='introselect', order=None)`; O(n), result unsorted |
| membership test | `np.isin(elem, test_elem)` | `(element, test_elements, assume_unique=False, invert=False, *, kind=None)` |
| build an array in a loop | list, then `np.array(list)` / `np.concatenate(list)`, or `np.fromiter(gen, dtype, count=n)` | never `np.append` in a loop |
| avoid a temporary | `np.multiply(a, b, out=buf)`, `buf += c` | ~1.3x at 5e6, can lose in cache; measure before adopting |
| control BLAS threads | `threadpoolctl.threadpool_limits(limits=1, user_api='blas')` | installed here (3.6.0), scoped; `OMP_NUM_THREADS` must precede import |
| sequential inner loop | vectorize the ensemble, loop the time axis | 178x here; `numba` and `cython` are **not installed** in this sandbox |
