# Broadcasting, indexing, and array combination

How numpy decides which elements line up with which: broadcasting, the five indexing forms and which alias, ufunc mechanics, axis semantics,
einsum, and the sort/search/set family. Open this for a shape error, an assignment that silently did nothing, or an axis that collapsed the
wrong way. numpy 2.4.4; every `# ->` is a value the line actually produced, with multi-row arrays compacted onto one line.

```python
import numpy as np
A = np.arange(12.).reshape(3, 4)   # -> [[ 0.  1.  2.  3.], [ 4.  5.  6.  7.], [ 8.  9. 10. 11.]]
v = np.array([10., 20., 30., 40.]); x = np.array([1., 2., 3.]); w = 100. * x   # (4,), (3,), (3,)
S = A[:, :3].copy(); B = np.arange(24.).reshape(2, 3, 4)   # S is A's 3x3 left block; B a batch of two 3x4
```

## Broadcasting

The rule, before any example. Align the two shape tuples at their **right** ends, prepending 1s to the shorter. Each aligned pair must then
be equal or contain a 1; a 1 stretches to the other's length, any other pair is an error, and the result shape is the elementwise maximum.

```python
A.shape, v.shape                     # -> (3, 4) (4,)   v pads to (1,4), the 1 stretches to 3
A + v                                # -> [[10. 21. 32. 43.], [14. 25. 36. 47.], [18. 29. 40. 51.]]
np.broadcast_shapes((3, 4), (4,)), np.broadcast_shapes((3, 1), (1, 4)), np.broadcast_shapes((2, 3, 4), (4,))
                                     # -> (3, 4) (3, 4) (2, 3, 4)   leading axes are free
np.broadcast_shapes((3, 4), (3,))    # -> ValueError: shape mismatch: objects cannot be broadcast to a single
                                     #    shape. Mismatch is between arg 0 with shape (3, 4) and arg 1 with shape (3,)
w[:, None].shape                     # -> (3, 1)   the fix: an explicit trailing axis, now (3,1) vs (3,4)
A + w[:, None]                       # -> [[100. 101. 102. 103.], [204. 205. 206. 207.], [308. 309. 310. 311.]]
x[:, None] * v[None, :]              # the outer-product idiom: (3,1) against (1,4), each 1 stretches
                                     # -> shape (3, 4): [[10. 20. 30. 40.], [20. 40. 60. 80.], [30. 60. 90. 120.]]
```

The `(3,4)` against `(3,)` failure is the common one: alignment is from the right, so the length-3 vector meets the 4. **Broadcasting never
allocates the expanded array.** A stretched dimension is a stride of 0, so the same bytes are reread, and the result is read-only because
one element would otherwise back many index positions. `np.broadcast_arrays(*xs)` puts every input at the common shape by the same trick;
`np.broadcast(*xs).shape` gives the shape alone.

```python
b = np.broadcast_to(v, (3, 4)); b.shape, b.strides, b.flags['WRITEABLE'], b.base.nbytes
                        # -> (3, 4) (0, 8) False 32   axis 0 costs 0 bytes; still the original four float64s
b[0, 0] = 5.            # -> ValueError: assignment destination is read-only
```

## The five kinds of indexing

Same rule as `01-arrays-dtypes.md`: **basic indexing is always a view, fancy and boolean indexing always copy**, because a view is only an
offset plus a shape and strides, and gathered elements fit no stride pattern.

| Form | Example | Result | View? |
|---|---|---|---|
| Basic slicing | `A[1:, :2]` -> shape `(2, 2)` | subrectangle | view |
| Integer | `A[1, 2]` -> `6.0`; `A[1]` -> `[4. 5. 6. 7.]` | drops that axis | view (scalar copies) |
| Boolean mask | `A[A > 6]` -> `[ 7.  8.  9. 10. 11.]`, `(5,)` | 1-D, row-major order | copy |
| Fancy (integer array) | `A[[0, 2]]` -> shape `(2, 4)` | gathered rows | copy |
| `np.newaxis` / `Ellipsis` | `A[..., 1]` -> `[1. 5. 9.]`; `A[None].shape` -> `(1, 3, 4)` | reshape only | view |

`np.newaxis is None` is `True`; `...` expands to as many full slices as needed, so `B[..., 0]` on `(2,3,4)` gives `(2,3)` at any rank.
**Fancy-index shape rule:** with one index array per axis, the result shape is the *broadcast* shape of the index arrays, replacing the
indexed axes: two `(2,2)` index arrays give a `(2,2)` result. The arrays are zipped, not crossed. To cross them into a submatrix they must
instead broadcast against each other, which is what `np.ix_` builds (it also accepts boolean masks).

```python
A[[0, 2], [1, 3]]           # -> [ 1. 11.]  shape (2,): elements (0,1) and (2,3), not a 2x2 block
np.ix_([0, 2], [1, 3])      # -> (array([[0], [2]]), array([[1, 3]]))   shapes (2,1) and (1,2)
A[np.ix_([0, 2], [1, 3])]   # -> [[ 1.  3.], [ 9. 11.]]   shape (2,2), the submatrix
```

## Assignment and in-place

Fancy and boolean indexing copy on the right of `=` but write back on the **left**. `a[mask] = 0` works; `a[mask][0] = 0` does not, because
the inner `a[mask]` already produced a throwaway copy.

```python
a = A.copy(); a[a > 6] = 0.; a   # -> [[0. 1. 2. 3.], [4. 5. 6. 0.], [0. 0. 0. 0.]]
np.where(A > 6, A, -1.)          # -> row 0 [-1. -1. -1. -1.], row 2 [ 8.  9. 10. 11.]
np.clip(A, 2., 8.)[0]            # -> [2. 2. 2. 3.]   positional a_min/a_max, or min=/max= keywords
a = A.copy(); np.putmask(a, A % 2 == 0, -1.); a[0]         # -> [-1.  1. -1.  3.]   in place
```

`np.where(cond, x, y)` broadcasts all three and returns a new array; with only `cond` it is `np.nonzero`. `np.clip` on 2.4.4 takes
positional `a_min, a_max` or keyword-only `min=`/`max=`. Masked assignment accepts a vector matching the count of selected elements:
`a[A > 8] = np.array([-1., -2., -3.])` fills row-major. **The duplicate-index trap:** `a[idx] += 1` expands to `a[idx] = a[idx] + 1`, so the
right side is computed once against the copy and a repeated index is written, not accumulated.

```python
z = np.zeros(3); z[[0, 0, 0]] += 1; z                      # -> [1. 0. 0.]   incremented once
z = np.zeros(3); np.add.at(z, [0, 0, 0], 1); z             # -> [3. 0. 0.]   unbuffered, accumulates
np.bincount([0, 0, 2], weights=[1., 2., 5.], minlength=3)  # -> [3. 0. 5.]
```

`np.add.at` is correct but slow (no buffering, no vectorized loop); `np.bincount(idx, weights=w, minlength=n)` is far faster when the
indices are 1-D integers into an accumulator.

## ufuncs

A **ufunc** (universal function) is a C-level elementwise kernel with three properties: it broadcasts its inputs, it dispatches to a typed
inner loop from a fixed list, and it exposes a common keyword and method interface. `np.add`, `np.sqrt`, `np.maximum` are ufuncs; `np.sum`
wraps one. Keywords on all of them: `out=` writes into an existing array of the broadcast shape with no allocation; `where=` is a boolean
mask leaving unmasked positions of `out` untouched; `dtype=` forces the loop or accumulation type; `casting=` sets how strictly inputs may
be cast (default `'same_kind'`). Four methods serve binary ufuncs. `type(np.add), np.add.nin, np.add.ntypes, np.add.identity` ->
`<class 'numpy.ufunc'> 2 22 0`: two inputs, 22 typed loops, identity `0`.

```python
out = np.empty((3, 4)); np.add(A, v, out=out); out[0]      # -> [10. 21. 32. 43.]   no allocation
r = np.full((3, 4), -1.); np.sqrt(A, out=r, where=A > 4); r[0], r[2]
                              # -> [-1. -1. -1. -1.]  [2.82842712 3.  3.16227766 3.31662479]
np.add.reduce(A, axis=0), np.add.reduce(A, dtype=np.float32).dtype   # -> [12. 15. 18. 21.] float32
np.add.accumulate(A, axis=1)[0]  # -> [0. 1. 3. 6.]   running reduction, input shape kept
np.multiply.outer(x, v).shape    # -> (3, 4)  all pairs; == x[:, None] * v[None, :]
```

`np.add.reduce(A, axis=0)` is exactly `A.sum(axis=0)`. `identity` is what an empty reduction returns: `0` for `add`, `1` for `multiply`,
`None` for `maximum`, so `np.maximum.reduce([])` raises `ValueError: zero-size array to reduction operation maximum which has no identity`.
The fourth method is the unbuffered `np.add.at`; `np.add.reduceat(np.arange(8.), [0,3,5])` -> `[ 3.  7. 18.]` is segmented.

## Reductions and axis semantics

`axis=k` **collapses axis k**: that dimension is removed and each entry is the reduction over it. `axis=None` reduces to a scalar and is the
default for `np.sum` but not `np.sort`. `keepdims=True` keeps it at length 1, which is what broadcasting needs to realign the result.

```python
A.sum(axis=0)              # -> [12. 15. 18. 21.]   shape (4,), one per column
A.sum(axis=1)              # -> [ 6. 22. 38.]       shape (3,), one per row
A.sum(axis=(0, 1))         # -> 66.0   a tuple collapses several at once; A.sum() is the same
A.sum(axis=0, keepdims=True).shape, A.sum(axis=1, keepdims=True).shape   # -> (1, 4) (3, 1)
A - A.mean(axis=1, keepdims=True)   # (3,4)-(3,1) -> mean-centered rows, each [-1.5 -0.5  0.5  1.5]
A - A.mean(axis=1)                  # -> ValueError: operands could not be broadcast together
                                    #    with shapes (3,4) (3,)
A - A.mean(axis=0)                  # -> no error, but centers COLUMNS: row 0 is [-4. -4. -4. -4.]
```

The `axis=1` version fails loudly. The `axis=0` version is the dangerous one: `(3,4) - (4,)` broadcasts fine and silently answers a
different question. `argmin`/`argmax` return a **flat** index when `axis=None`, unusable for indexing a 2-D array directly.

```python
A.argmax(), np.unravel_index(A.argmax(), A.shape)   # -> 11 (np.int64(2), np.int64(3))   flat, then subscripts
np.take_along_axis(A, A.argmax(axis=1)[:, None], axis=1).shape   # -> (3, 1)   per-row maxima
```

`np.take_along_axis(a, idx, axis)` is the general consumer of per-axis `argsort`/`argmax` output. `np.nonzero(m)` gives a tuple of index
arrays usable as a fancy index; `np.argwhere(m)` gives the same as an `(n_hits, ndim)` array, for reading rather than indexing.

## einsum, matmul, and friends

`np.einsum` writes a contraction in index notation. Each operand gets one letter per axis; a letter repeated across operands means those
axes align and multiply elementwise. Letters on the left of `->` but not the right are **summed over**; letters kept on the right become
output axes in the order written. That is the usual summation convention, with the sum marked by omission.

| Operation | einsum | Plain numpy | Result here |
|---|---|---|---|
| Trace | `np.einsum('ii->', S)` | `S.trace()` | `15.0` |
| Diagonal | `np.einsum('ii->i', S)` | `np.diagonal(S)` | `[ 0.  5. 10.]` |
| Transpose | `np.einsum('ij->ji', A)` | `A.T` | shape `(4, 3)` |
| Matmul | `np.einsum('ij,jk->ik', A, A.T)` | `A @ A.T` | shape `(3, 3)` |
| Batched matmul | `np.einsum('bij,bjk->bik', B, Bt)` | `B @ Bt` | `(2, 3, 3)` for `Bt = B.transpose(0,2,1)` |
| Outer | `np.einsum('i,j->ij', x, v)` | `np.outer(x, v)` | shape `(3, 4)` |
| Inner | `np.einsum('i,i->', v, v)` | `v @ v` | `3000.0` |
| Weighted total | `np.einsum('ij,j->', A, v)` | `(A * v).sum()` | `1800.0` |

`...` stands for any number of leading batch axes: `np.einsum('...ij,...jk->...ik', B, B.transpose(0,2,1))` on `(2,3,4)` gives `(2,3,3)`.
`optimize` defaults to `False`, a naive left-to-right contraction often much worse for 3+ operands, so pass `optimize=True`.

```python
x2, y2, w2 = (np.random.default_rng(0).random(s) for s in [(30, 40), (40, 50), (50, 60)])
p, s = np.einsum_path('ij,jk,kl->il', x2, y2, w2, optimize='greedy')
p          # -> ['einsum_path', (1, 2), (0, 1)]   contract operands 1,2 first, then 0 with that
print(s)   # ->  Naive scaling 4 / optimized 3.  Naive FLOP count 1.080e+07 vs optimized 3.840e+05.  Theoretical
           #     speedup: 28.125.  Largest intermediate: 2.400e+03 elements.  Steps: 3 kl,jk->jl then 3 jl,ij->il
```

`np.matmul` (`@`) is a generalized ufunc: all axes before the last two are **batch** axes and broadcast normally, so
`np.ones((5,3,4)) @ np.ones((4,2))` is `(5,3,2)`. It never does scalar multiplication and never contracts more than one axis; for arbitrary
axis pairs use `np.tensordot(a, b, axes=([1],[1]))` (`axes=2` contracts the last two against the first two). NumPy 2 added three gufuncs so
matrix-vector work needs no `[:, None]` dance; all four exist on 2.4.4 with these `.signature` values:

| Call | `.signature` | On the toy data |
|---|---|---|
| `np.vecdot(x1, x2)` | `(n),(n)->()` | `np.vecdot(A, v)` -> `[ 200.  600. 1000.]`, row-wise dot |
| `np.matvec(x1, x2)`, `np.vecmat(x1, x2)` | `(m,n),(n)->(m)`, `(n),(n,m)->(m)` | `np.matvec(A, v)` is `A @ v` but batches vector stacks; `np.vecmat(v, A.T)` -> `[ 200.  600. 1000.]` |
| `np.matmul(x1, x2)` | `(n?,k),(k,m?)->(n?,m?)` | `?` marks the 1-D promotion rule |

A `?` dimension is optional: a 1-D operand is promoted (prepended on the left, appended on the right) and the axis stripped from the result.
What matters: with `B` of shape `(2,3,4)`, `np.matvec(B, np.ones((2,4)))` gives `(2,3)`, one product per batch element, while
`B @ np.ones((2,4))` raises a core-dimension mismatch. `np.vecdot` conjugates arg 1: for complex `c`, `np.vecdot(c, c)` is
`(c.conj()*c).sum()`, not `(c*c).sum()`.

## Sorting, searching, set operations

| Task | Call | Note |
|---|---|---|
| Sort values / indices | `np.sort(a, axis=-1, kind=None, order=None, *, stable=None)`, `np.argsort(a, ...)` | default axis `-1`, not `None`; `kind='stable'` (or `stable=True`) keeps ties in input order |
| Multi-key sort | `np.lexsort((secondary, primary))` | the **last** key is primary |
| Top-k | `np.argpartition(a, -k)[-k:]`, `np.partition(a, -k)` | O(n); all before `kth` are <= it, otherwise unordered |
| Binned lookup | `np.searchsorted(a, v, side='left', sorter=None)` | `a` must be sorted; gives insertion positions |
| Unique | `np.unique(ar, return_index, return_inverse, return_counts, axis=None, *, equal_nan=True, sorted=True)` | output sorted |

```python
q = np.array([5., 1., 4., 1., 9., 2., 6.])   # note the repeated 1.
np.argsort(q, kind='stable')  # -> [1 3 5 2 0 6 4]   ties keep position 1 before 3
t = np.argpartition(q, -3)[-3:]; q[t]   # -> [0 6 4]  [5. 6. 9.]   the top 3, in no order
np.searchsorted(np.array([0., 10., 20., 30.]), np.array([-1., 0., 5., 20., 99.]))   # -> [0 0 1 2 4]
u, idx, inv, cnt = np.unique(q, return_index=True, return_inverse=True, return_counts=True)
u, cnt, np.isin(q, [1., 9.])  # -> [1. 2. 4. 5. 6. 9.] [2 1 1 1 1 1] [False True False True True False False]
idx, inv                      # -> [1 5 2 0 6 4] so q[idx]==u; [3 0 2 0 5 1 4] so u[inv]==q
```

On 2.4.4, `return_inverse` with `axis=None` gives an inverse with the **input's** shape, not a flat one, so
`np.unique(A, return_inverse=True)[1].shape` is `(3, 4)`; ravel it if downstream code expects 1-D.

## Gotchas

- **`a[idx] += 1` with duplicate indices increments once.** `z = np.zeros(3); z[[0,0,0]] += 1` gives `[1. 0. 0.]`, because it expands to
  `z[idx] = z[idx] + 1` and the right side reads a copy. Symptom: histograms and scatter-adds that undercount. Fix `np.add.at(z, idx, 1)`,
  or `np.bincount(idx, weights=w, minlength=n)` for 1-D integer indices (much faster).
- **Fancy and boolean indexing return copies, so chained assignment silently does nothing.** `a[a > 6][0] = -99.` runs without error and
  leaves `a` unchanged; fix with one `=` and the mask on the left, `a[a > 6] = -99.`. Basic slices are views, so `a[0][0] = -1.` does work,
  which is why the bug appears only after a slice becomes a mask. Two neighbours: `A[[0,2],[1,3]]` gives 2 elements and not a 2x2 block
  (index arrays zip, so use `A[np.ix_([0,2],[1,3])]`), and `np.broadcast_to` output is read-only (stride 0).
- **Forgetting `keepdims` either raises or, worse, does not.** `A - A.mean(axis=1)` raises
  `ValueError: operands could not be broadcast together with shapes (3,4) (3,)`, but `A - A.mean(axis=0)` succeeds and centers columns
  instead of rows. Fix: `keepdims=True` whenever a reduction feeds back into an expression with the original array.
- **`(n,)` versus `(n,1)` produces an `(n,n)` instead of an error.** With `a` of shape `(3,)` and `b` of shape `(3,1)`, `a - b` is `(3, 3)`
  and `np.linalg.norm(a - b)` returns `59.35` where the intended residual is `33.67`. Nothing warns. Symptom: a plausible but wrong residual
  or loss, and quadratic memory growth at scale. Fix `b.ravel()` or `b[:, 0]`, and assert shapes at function boundaries.
- **`A.argmax()` without `axis` returns a flat index**, so `A[A.argmax()]` indexes the wrong axis or raises; fix with
  `A[np.unravel_index(A.argmax(), A.shape)]`. Relatedly `np.sort` defaults to `axis=-1` while `np.sum` defaults to `axis=None`, so
  `np.sort(A)` sorts each row and keeps the `(3,4)` shape; pass `axis=None` for a flat sort.
- **Two quieter defaults.** `out=` with `where=` leaves masked positions untouched, so an `out=np.empty(...)` holds uninitialized memory
  there: pass `np.full` or a pre-filled `out`. And `np.einsum` does not optimize by default, which for 3+ operands can cost an order of
  magnitude in FLOPs: check `np.einsum_path`, pass `optimize=True`.

## Quick reference

| Task | Call | Note |
|---|---|---|
| Predict / force a broadcast | `np.broadcast_shapes(s1, s2)`; `np.broadcast_to(a, shape)`, `np.broadcast_arrays(a, b)` | raises on mismatch; expanded views are `WRITEABLE False`, stride 0 |
| Add an axis, outer, submatrix | `a[:, None]`, `a[None, :]`; `x[:, None] * y[None, :]`; `A[np.ix_(rows, cols)]` | `np.newaxis is None`; `A[rows, cols]` zips instead of crossing |
| Mask to indices, select, bound | `np.nonzero(m)`, `np.argwhere(m)`; `np.where(c, x, y)`, `np.clip(a, lo, hi)`, `np.putmask(a, m, vals)` | `nonzero` is indexable, `argwhere` is `(k, ndim)`; `where` broadcasts all three; `np.sqrt(a, out=r, where=m)` leaves the rest of `out` untouched |
| Scatter-add, reduce, keep axis | `np.add.at(a, idx, v)`, `np.bincount(idx, weights=w, minlength=n)`; `a.sum(axis=k)`, `a.mean(axis=k, keepdims=True)`, `np.add.accumulate(a, axis=k)` | `a[idx] += v` is wrong on duplicates; `axis=k` collapses axis k; `np.add.reduceat` for segments |
| Flat argmax to subscripts, gather | `np.unravel_index(a.argmax(), a.shape)`; `np.take_along_axis(a, idx, axis=k)` | inverse `np.ravel_multi_index`; `take_along_axis` consumes `argsort`/`argmax` output |
| Contraction and products | `np.einsum('ij,jk->ik', A, B, optimize=True)`, `np.einsum_path`; `A @ B`, `np.tensordot(a, b, axes=([1],[1]))`, `np.vecdot`, `np.matvec`, `np.vecmat` | omitted einsum letters are summed; `(n?,k),(k,m?)->(n?,m?)`; `vecdot` conjugates arg 1 |
| Sort, top-k, lookup, unique, sets | `np.argsort(a, kind='stable')`, `np.lexsort((k2, k1))`, `np.argpartition(a, -k)[-k:]`, `np.searchsorted(a, v, side='left')`, `np.unique(a, return_index=True, return_inverse=True, return_counts=True)`, `np.isin(a, t)`, `np.intersect1d`/`setdiff1d`/`union1d` | sort default axis `-1`; last lexsort key is primary; `argpartition` is O(n); `searchsorted` needs `a` sorted; `u[inv] == a` and `inv` keeps the input shape; set ops flatten |
