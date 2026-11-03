# Arrays, dtypes, and memory

What an ndarray physically is, when an operation returns a view onto the same bytes instead of a copy, how dtypes
promote under NumPy 2's NEP 50 rules, and how to create, reshape, and persist arrays. Open this for aliasing
bugs, dtype surprises, integer overflow, or layout questions. Ground truth: numpy 2.4.4. Throughout, `# ->` shows
what `print()` of that expression displayed when the line was run.

One toy array runs through the whole file: `a = np.arange(12, dtype=np.float64).reshape(3, 4)`.

Contents: Anatomy of an ndarray; Views vs copies; dtypes; Promotion: NEP 50; Creation and filling; Shape
manipulation; NaN, inf, and floating-point error state; Structured arrays; Saving and loading; Gotchas; Quick reference.

## Anatomy of an ndarray

An ndarray is four things: a flat block of bytes (the data buffer), a `dtype` saying how to read each element, a
`shape`, and a `strides` tuple. Everything else is derived.

```python
import numpy as np
a = np.arange(12, dtype=np.float64).reshape(3, 4)
a            # -> [[ 0.  1.  2.  3.]
             #     [ 4.  5.  6.  7.]
             #     [ 8.  9. 10. 11.]]
a.shape      # -> (3, 4)      3 along axis 0, 4 along axis 1
a.dtype      # -> float64     how to interpret each element
a.itemsize   # -> 8           bytes per element, same as a.dtype.itemsize
a.nbytes     # -> 96          = size * itemsize, the buffer, not the Python object
a.strides    # -> (32, 8)     bytes to step to advance one index along that axis
a.flags      # -> C_CONTIGUOUS True, F_CONTIGUOUS False, OWNDATA False, WRITEABLE True, ALIGNED True
a[2, 1]                          # -> 9.0
2*a.strides[0] + 1*a.strides[1]  # -> 72   its byte offset: base + i*32 + j*8
np.asfortranarray(a).strides     # -> (8, 24)          C False, F True
a.T.strides, a.T.shape           # -> (8, 32) (4, 3)   C False, F True
a[:, ::2].strides                # -> (32, 16)         C False, F False
```

A **stride** is a byte offset, not an element count, so indexing is arithmetic and never a search.
**C-contiguous** (row-major) means the last axis has stride `itemsize` and each earlier stride is the next axis's
stride times its length, so a row is a solid run of memory. **F-contiguous** (column-major, Fortran order) is the
mirror: axis 0 has stride `itemsize` and a column is solid. An array can be both (1-D, or any axis of length 1)
or neither. Transpose only permutes the shape and strides tuples, so `.T` is free and the transpose of a
C-contiguous array is F-contiguous; a step-2 slice doubles a stride, giving an array contiguous in neither order.
`OWNDATA: False` says `a` does not own its buffer: `a.base` is the flat `(12,)` array `arange` produced.
Contiguity decides whether `reshape` can avoid a copy and drives speed (see `03-vectorization-performance.md`).

## Views vs copies

The most consequential distinction in numpy. A **view** shares the data buffer with its parent, so writing
through one changes the other. A **copy** does not. `np.shares_memory(x, y)` is the exact test (slow on large
strided arrays); `np.may_share_memory` is the cheap conservative one and says `True` for disjoint columns of one
buffer. `.base` is the array a view derives from, `None` for an owner.

```python
np.shares_memory(a, a[1:, :2])           # -> True    basic slice
np.shares_memory(a, a[[0, 2]])           # -> False   fancy (integer-array) index
np.shares_memory(a, a[a > 5])            # -> False   boolean index
np.shares_memory(a, a.reshape(4, 3))     # -> True    reshape when strides allow
np.shares_memory(a, a.T)                 # -> True    transpose
np.shares_memory(a, a.ravel())           # -> True    ravel when strides allow
np.shares_memory(a, a.flatten())         # -> False   flatten always copies
np.shares_memory(a, a.T.ravel())         # -> False   reshape forced to copy
```

The rule behind the table: basic indexing (slices, integers, `None`, `Ellipsis`) is always a view, because the
result is an offset plus a new shape and strides. Fancy and boolean indexing gather scattered elements, so they
must copy. `reshape` returns a view when the new shape can be walked with some stride tuple over the existing
buffer and copies otherwise; `a.T.ravel()` copies because reading a transpose in row-major order visits the
buffer in a pattern no single stride tuple describes.

The aliasing bug, and its mirror image:

```python
a = np.arange(12, dtype=np.float64).reshape(3, 4)
row = a[0]; row *= 100              # row is a VIEW
a[0]                        # -> [  0. 100. 200. 300.]   parent changed

a = np.arange(12, dtype=np.float64).reshape(3, 4)
picked = a[[0, 1]]; picked *= 100   # picked is a COPY
a[0]                        # -> [0. 1. 2. 3.]           parent unchanged
picked[0]                   # -> [  0. 100. 200. 300.]

a = np.arange(12, dtype=np.float64).reshape(3, 4)
a[[0, 1]] *= 100                    # fancy index on the LEFT does write back
a[1]                        # -> [400. 500. 600. 700.]

b = np.arange(6, dtype=np.float64)
np.shares_memory(b, b.astype(np.float64, copy=False))   # -> True
np.shares_memory(b, b.astype(np.float32, copy=False))   # -> False   dtype differs, must copy
np.shares_memory(b, np.asarray(b))                      # -> True
np.array([1.7, -1.7, 2.99]).astype(np.int64)            # -> [ 1 -1  2]   truncates toward zero
```

An in-place update on a slice silently mutates the parent; the same update on a fancy-indexed result silently
does not. Assignment differs again, because `a[idx] op= v` expands to
`a.__setitem__(idx, a.__getitem__(idx) op v)` and reaches the parent. Copy when you are about to mutate something
you did not allocate, and when a small slice is keeping a huge buffer alive: `arr.copy()` (order `'C'`) or
`np.copy(a, order='K', subok=False)`; `np.ascontiguousarray` / `np.asfortranarray` copy only if the layout is
wrong. `astype` allocates a new array, and `copy=False` there means "reuse only if nothing at all needs to
change"; use `np.rint(x).astype(int)` if you meant rounding rather than truncation.
`np.array(x, copy=False)` raises in numpy 2.x
(`ValueError: Unable to avoid copy while creating an array as requested`): `copy=False` now means "must not
copy", and the never-copy request is `np.asarray(x)`.

## dtypes

| dtype | itemsize | char | kind | Notes |
|---|---|---|---|---|
| `bool_` | 1 | `?` | `b` | one byte per element, not one bit |
| `int8` / `int16` / `int32` / `int64` | 1 / 2 / 4 / 8 | `b h i l` | `i` | two's complement, wraps on overflow |
| `uint8` / `uint16` / `uint32` / `uint64` | 1 / 2 / 4 / 8 | `B H I L` | `u` | wraps to the top on subtraction below 0 |
| `float16` / `float32` / `float64` | 2 / 4 / 8 | `e f d` | `f` | IEEE half / single / double |
| `longdouble` (reports `float128`) | 16 | `g` | `f` | x86 80-bit extended padded to 16 bytes, 64-bit mantissa, not 113 |
| `complex64` / `complex128` | 8 / 16 | `F D` | `c` | a pair of `float32` / `float64` |

`kind` is the one-letter family code (`b` bool, `i` signed, `u` unsigned, `f` float, `c` complex, `U` unicode,
`V` void/structured). Limits come from `np.finfo` (floats) and `np.iinfo` (integers):

```python
np.finfo(np.float32).eps, np.finfo(np.float32).max    # -> 1.1920929e-07 3.4028235e+38
np.finfo(np.float32).tiny                             # -> 1.1754944e-38    smallest normal
np.finfo(np.float64).eps, np.finfo(np.float64).max    # -> 2.220446049250313e-16 1.7976931348623157e+308
np.finfo(np.float64).tiny                             # -> 2.2250738585072014e-308
np.iinfo(np.int8).min, np.iinfo(np.int8).max          # -> -128 127
np.iinfo(np.int64).max                                # -> 9223372036854775807
np.array([1]).dtype, np.array([1.0]).dtype            # -> int64 float64
```

`eps` is the gap between 1.0 and the next representable float; see `04-floating-point.md`. The default integer
dtype is `int64` here. It was `int32` on Windows under numpy 1.x; numpy 2.0 changed the Windows default to
`int64` to match every other platform. Integer arithmetic wraps modulo 2**bits and never promotes to a wider
type. numpy scalars warn, arrays do not:

```python
np.int8(127) + np.int8(1)                    # -> -128, and RuntimeWarning: overflow encountered in scalar add
np.array([127], dtype=np.int8) + 1           # -> [-128]   silent, no warning
(np.array([2**62], dtype=np.int64) * 4)[0]   # -> 0        silent int64 wraparound
```

The array path is silent by design: checking every element would cost more than the arithmetic. Sums of int32 and
int64 arrays are the usual victims. `arr.sum(dtype=np.int64)` or `dtype=float` forces a wider accumulator.

## Promotion: NEP 50 (numpy 2.x)

One rule, three clauses.

1. **Python scalars are weak.** A literal `int`, `float`, `complex`, or `bool` adopts the other operand's dtype
   and never widens it. The result dtype comes from the array alone.
2. **Array with array** follows the kind/width lattice: same kind takes the wider width; mixed kinds go to the
   smallest type holding both, with signed plus unsigned climbing a width, and `int64 + uint64 -> float64`
   because no integer type covers both ranges.
3. **numpy scalars are not weak.** `np.float32(1)` and `np.int8(1)` carry their dtype into the lattice exactly as
   0-d arrays do.

```python
f32 = np.float32([1.0, 2.0])
(f32 + 1.0).dtype                       # -> float32    weak Python float, no upcast
(f32 * 2).dtype                         # -> float32    weak Python int
(f32 + np.float64(1)).dtype             # -> float64    numpy scalar is strong
(np.float32(1) + np.float64(1)).dtype   # -> float64
np.uint8(200) + np.int8(100)            # -> 300, dtype int16   lattice climbs a width
np.result_type(np.int64, np.uint64)     # -> float64
f32 + 1e300                             # -> [inf inf], RuntimeWarning: overflow encountered in cast
```

Under numpy 1.x `np.float32([1.]) + 1.0` returned float64: the Python float's value decided the kind and the
result upcast. Under NEP 50 it stays float32, so `f32 + 1e300` overflows to `inf` instead of quietly widening.
The same rule turns an out-of-range Python int into an error:

```python
np.array([1], dtype=np.uint8) + 300     # -> OverflowError: Python integer 300 out of bounds for uint8
np.uint8(1) - 2                         # -> 255, uint8, RuntimeWarning: overflow encountered in scalar subtract
```

`np.result_type(*args)` answers any promotion question without running the operation; `np.promote_types(a, b)`
takes dtypes only and ignores the weak-scalar rule.

## Creation and filling

| Call | Result |
|---|---|
| `np.zeros(shape, dtype=float)`, `np.ones` | filled, float64 default |
| `np.empty(shape)` | allocated, **uninitialized**, holds whatever was in that memory |
| `np.full(shape, fill_value, dtype=None, order='C')` | dtype inferred from `fill_value` when `dtype=None` |
| `np.arange(start, stop, step)` | half-open `[start, stop)`, integer semantics |
| `np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)` | endpoint included |
| `np.logspace(start, stop, num=50, endpoint=True, base=10.0, ...)` | `base**start` through `base**stop` |
| `np.eye(N, M=None, k=0, dtype=float)`, `np.identity(n)` | `k` shifts the diagonal; `identity` is square only |
| `np.zeros_like(a)`, `ones_like`, `empty_like`, `full_like` | copies shape, dtype, and memory order |
| `np.fromiter(iter, dtype, count=-1)` | consumes any iterable, `count` preallocates |
| `np.frombuffer(buffer, dtype=None, count=-1, offset=0)` | zero-copy over bytes, result is read-only |

```python
np.linspace(0, 1, 5)                  # -> [0.   0.25 0.5  0.75 1.  ]     endpoint included
np.linspace(0, 1, 5, endpoint=False)  # -> [0.  0.2 0.4 0.6 0.8]
len(np.arange(0, 1, 0.1))             # -> 10
np.arange(1, 1.3, 0.1)                # -> [1.  1.1 1.2 1.3]   the excluded endpoint appeared
np.empty(3, dtype=np.int64)           # -> [ -36028799174836225 3746994891045994496          4303355904]
```

`arange` computes its length as `ceil((stop - start) / step)` in floating point. `(1.3 - 1) / 0.1` rounds to
slightly more than 3, so the length is 4 and `1.3` shows up inside a half-open range. Use `linspace` whenever the
endpoints matter. `np.empty` beats `zeros` only because it skips the write; anything read before being written is
garbage, and in float dtypes that can be a NaN or denormal that poisons everything downstream.

## Shape manipulation

```python
a = np.arange(12, dtype=np.float64).reshape(3, 4)
c = a.reshape(2, 3, 2)
a.reshape(-1).shape                       # -> (12,)      -1 is solved from the element count
a.reshape(2, -1).shape                    # -> (2, 6)     at most one -1 allowed
a[:, None].shape                          # -> (3, 1, 4)  np.newaxis IS None
a[np.newaxis].shape                       # -> (1, 3, 4)
np.expand_dims(a, 2).shape                # -> (3, 4, 1)
np.squeeze(a.reshape(1, 3, 1, 4)).shape   # -> (3, 4)     drops EVERY length-1 axis
a.reshape(1, 3, 1, 4).squeeze(axis=2).shape  # -> (1, 3, 4)   drops only the one named
a.T.shape                                 # -> (4, 3)
c.transpose(2, 0, 1).shape                # -> (2, 2, 3)  full permutation
np.moveaxis(c, 0, -1).shape               # -> (3, 2, 2)  relocate one axis, slide the rest
np.swapaxes(c, 0, 2).shape                # -> (2, 3, 2)  exchange two

b = a + 100
np.stack([a, b]).shape                    # -> (2, 3, 4)  new axis 0
np.stack([a, b], axis=1).shape            # -> (3, 2, 4)  new axis 1
np.concatenate([a, b]).shape              # -> (6, 4)     axis 0 grows
np.concatenate([a, b], axis=1).shape      # -> (3, 8)     axis 1 grows
np.vstack([a, b]).shape                   # -> (6, 4)
np.hstack([a, b]).shape                   # -> (3, 8)
r = np.arange(3.)                         # shape (3,)
np.vstack([r, r]).shape                   # -> (2, 3)
np.hstack([r, r]).shape                   # -> (6,)   concatenates 1-D, does NOT make columns
np.column_stack([r, r]).shape             # -> (3, 2)
[p.shape for p in np.split(a, 2, axis=1)] # -> [(3, 2), (3, 2)]   and these are views
[p.shape for p in np.array_split(a, 2)]   # -> [(2, 4), (1, 4)]   ragged last piece allowed
np.split(a, 2)                            # -> ValueError: array split does not result in an equal division
```

**`stack` adds a new axis, `concatenate` does not.** `stack` needs identical input shapes; `concatenate` needs
agreement on every axis except the join axis. `vstack` and `hstack` are `concatenate` with 1-D promotion built
in, which is why `hstack` on 1-D concatenates and only `column_stack` builds columns. `squeeze()` with no
argument is a bug magnet when a dimension happens to be 1 at runtime; pass `axis=`.

## NaN, inf, and floating-point error state

```python
a = np.arange(12, dtype=np.float64).reshape(3, 4)
a[1, 2] = np.nan; a[2, 3] = np.inf
np.nan == np.nan                          # -> False    NaN is unordered, use np.isnan
np.isinf(a).sum(), np.isfinite(a).sum()   # -> 1 10
a.sum()                                   # -> nan      every plain reduction propagates
np.nansum(a)                              # -> inf      skips NaN, does NOT skip inf
np.nanmean(a)                             # -> inf
np.nanargmax(a)                           # -> 11       a FLAT index
np.unravel_index(np.nanargmax(a), a.shape)            # -> (2, 3)
np.nanpercentile(np.where(np.isfinite(a), a, np.nan), 50)   # -> 4.5

x = np.array([1.0, 0.0, -1.0])
1.0 / x                                   # -> [ 1. inf -1.], RuntimeWarning: divide by zero encountered in divide
with np.errstate(divide='ignore', invalid='ignore'):
    1.0 / x                               # -> [ 1. inf -1.]    silent
    np.array([0.0]) / np.array([0.0])     # -> [nan]            silent
old = np.seterr(divide='raise')
1.0 / x                                   # -> FloatingPointError: divide by zero encountered in divide
np.seterr(**old)                          # restore what seterr returned
np.geterr()   # -> {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
```

The `nan*` family is `nansum`, `nanmean`, `nanstd`, `nanvar`, `nanmin`, `nanmax`, `nanargmin`, `nanargmax`,
`nanmedian`, `nanpercentile`, `nanquantile`, `nancumsum`, `nancumprod`. Mask with `np.isfinite` when infinities
are also invalid. An all-NaN slice makes `nanmean` warn (`RuntimeWarning: Mean of empty slice`) and return NaN,
and makes `nanargmax` raise `ValueError`. Division by zero and invalid operations warn rather than raise, and the
policy is per thread.
`np.errstate(*, call=..., all=None, divide=None, over=None, under=None, invalid=None)` is a context manager and a
decorator; each category takes `'ignore'`, `'warn'`, `'raise'`, `'call'`, or `'print'`. `np.seterr` sets the same
globally and returns the previous dict, so keep it and restore it. Wrapping a numerical kernel in
`np.errstate(all='raise')` is the fastest way to find where NaN first appears; see `13-debugging.md`.

## Structured arrays

A compound dtype gives each element named fields. Use it for binary file layouts and C interop; pandas is the
better tool once you want joins, grouping, or missing values.

```python
dt = np.dtype([('id', 'i4'), ('pos', 'f8', (3,)), ('label', 'U8')])
r = np.zeros(2, dtype=dt)
r['id'] = [1, 2]; r['pos'][0] = [0.5, 1.0, 1.5]; r['label'] = ['left', 'right']
r                                    # -> [(1, [0.5, 1. , 1.5], 'left') (2, [0. , 0. , 0. ], 'right')]
dt.itemsize, dt.names                # -> 60 ('id', 'pos', 'label')
[dt.fields[n][1] for n in dt.names]  # -> [0, 4, 28]     byte offsets of each field
r['pos'].shape                       # -> (2, 3)         subarray shape appends on the right
np.shares_memory(r, r['pos'])        # -> True           field access is a VIEW
r.view(np.recarray).id               # -> [1 2]          attribute access instead of r['id']
```

## Saving and loading

| Call | Format | Note |
|---|---|---|
| `np.save(file, arr, allow_pickle=True)` | `.npy` binary | one array, dtype and shape preserved exactly |
| `np.load(file, mmap_mode=None, allow_pickle=False, ...)` | | `allow_pickle` defaults to **False** on load |
| `np.savez(file, **arrays)`, `np.savez_compressed` | `.npz`, a zip of `.npy` | lazy: index by key, then close it |
| `np.savetxt(fname, X, fmt='%.18e', delimiter=' ', header='', ...)` | text | 2-D max, slow, lossy unless `fmt` is wide |
| `np.loadtxt(fname, dtype=float, delimiter=None, skiprows=0, usecols=None, ...)` | text | slow; use `pandas.read_csv` for big files |
| `np.memmap(filename, dtype=uint8, mode='r+', offset=0, shape=None, order='C')` | raw bytes | no header, you supply dtype and shape |

```python
import os; d = "/tmp/npdemo"; os.makedirs(d, exist_ok=True)
np.savez(f"{d}/two.npz", a=a, b=a.T)
with np.load(f"{d}/two.npz") as z:
    list(z.keys()), z['b'].shape     # -> ['a', 'b'] (4, 3)
big = np.random.default_rng(0).random((2000, 500))
np.save(f"{d}/big.npy", big); np.savetxt(f"{d}/big.csv", big)
os.path.getsize(f"{d}/big.npy"), os.path.getsize(f"{d}/big.csv")   # -> 8000128 25000000

mm = np.memmap(f"{d}/m.dat", dtype=np.float64, mode="w+", shape=(3, 4))
mm[:] = a; mm.flush(); del mm
mm2 = np.memmap(f"{d}/m.dat", dtype=np.float64, mode="r", shape=(3, 4))
mm2[2]                               # -> [ 8.  9. 10. inf]   (a still carries the inf set above)
os.path.getsize(f"{d}/m.dat")        # -> 96    raw bytes, no header
np.save(f"{d}/o.npy", np.array([{'k': 1}, [1, 2]], dtype=object)); np.load(f"{d}/o.npy")
# -> ValueError: Object arrays cannot be loaded when allow_pickle=False
```

Warm-cache load of that 2000x500 array: about 2 ms from `.npy` against about 265 ms from CSV, on top of the 3x
size gap, because text I/O parses every number. `allow_pickle=False` is the load default because a pickled `.npy`
executes arbitrary code on load; keep numeric dtypes and the question never arises. `np.memmap` maps the file
into the address space and pages in on demand, so an array larger than RAM works when access is local. Write mode
needs `.flush()`, slices of a memmap are still memmaps, and `np.asarray(mm)` forces a real in-memory array.
`np.load(path, mmap_mode='r')` gives the same laziness for a proper `.npy` file, header and all.

## Gotchas

- **Silent integer overflow.** `np.array([127], np.int8) + 1` gives `-128` with no warning; only numpy *scalar*
  ops emit `RuntimeWarning: overflow encountered in scalar add`. Sums of int32/int64 arrays wrap the same way.
  Fix: `arr.sum(dtype=np.int64)`, accumulate in float64, or bound-check against `np.iinfo(arr.dtype).max`.
- **`np.empty` holds garbage.** One run of `np.empty(3, dtype=np.int64)` printed
  `[-36028799174836225 3746994891045994496 4303355904]`; the next differs, and in float dtypes the leftover bytes
  can be NaN or a denormal. Use `np.zeros` unless you provably overwrite every element.
- **Aliasing through views.** `row = a[0]; row *= 100` mutates `a`. Every basic slice, `.T`, a successful
  `reshape`, `.ravel()`, and structured-field access is a view. Fix: `a[0].copy()` when you intend to own the
  data, and test with `np.shares_memory`, not `is`.
- **The reverse trap.** `picked = a[[0, 1]]; picked *= 100` leaves `a` untouched, because fancy and boolean
  indexing copy. Code that works on a slice silently stops working when the index becomes a list or a mask.
- **`astype` truncates toward zero.** `np.array([1.7, -1.7]).astype(np.int64)` is `[1, -1]`, not `[2, -2]`.
  Fix: `np.rint(x).astype(np.int64)` (round-half-to-even), or `np.floor`/`np.ceil` when you mean those.
- **`arange` with a float step drifts.** `np.arange(1, 1.3, 0.1)` returns `[1. 1.1 1.2 1.3]`: the excluded
  endpoint appears, because the length is `ceil((stop-start)/step)` computed in floating point. Use `linspace`.
- **`==` on floats.** `0.1 + 0.2 == 0.3` is `False`, and float32 accumulation drifts fast:
  `np.arange(1e6, dtype=np.float32).sum()` gives `4.999995e+11` against an exact `499999500000.0`. Use
  `np.isclose` / `np.allclose` and read `04-floating-point.md`.
- **NEP 50 changed float32 pipelines (numpy 2.x).** `np.float32([1.]) + 1.0` stays float32 where numpy 1.x gave
  float64, so `f32 + 1e300` now overflows to `inf`, and `np.array([1], np.uint8) + 300` raises `OverflowError`
  instead of upcasting. Fix: write `np.float64(1.0)` when you actually want the wider type.
- **`np.array(x, copy=False)` raises in numpy 2.x** ("Unable to avoid copy"); `copy=False` now means "never
  copy". Use `np.asarray(x)`. Relatedly, `a.T.shape = (12,)` raises `AttributeError: Incompatible shape for
  in-place modification` where `a.T.reshape(12)` succeeds, because `.shape =` refuses to copy. NumPy 2.5 deprecates
  assigning `.shape` altogether, so write `reshape`.
- **`squeeze()` with no axis** drops every length-1 dimension, so a `(1, N)` result silently becomes `(N,)`.
  Always pass `axis=`. Similarly `hstack` on two `(3,)` inputs gives `(6,)`, not `(3, 2)`; use `column_stack`.
- **`nansum` does not skip `inf`.** Mask with `np.isfinite` when infinities are also invalid. And `np.load`
  refuses object arrays by default: fix the dtype rather than flipping `allow_pickle` on an untrusted file.

## Quick reference

| Task | Call | Note |
|---|---|---|
| Bytes per element / total | `a.itemsize`, `a.nbytes` | `nbytes = size * itemsize` |
| Byte step per axis, layout | `a.strides`, `a.flags['C_CONTIGUOUS']` | `.T` of C-contiguous is F-contiguous |
| Force layout | `np.ascontiguousarray(a)`, `np.asfortranarray(a)` | copy only if needed |
| Do two arrays alias? | `np.shares_memory(x, y)` | `may_share_memory` is the cheap conservative test |
| Where a view came from | `a.base` | `None` for an owner |
| Guaranteed copy | `a.copy()`, `np.copy(a, order='K')` | method defaults to `order='C'` |
| No-copy conversion | `np.asarray(x)` | `np.array(x, copy=False)` raises on 2.x |
| Cast dtype | `a.astype(np.float32)` | always copies; float to int truncates toward zero |
| Float limits | `np.finfo(np.float64).eps / .max / .tiny` | eps `2.22e-16` float64, `1.19e-07` float32 |
| Integer limits | `np.iinfo(np.int8).min / .max` | `-128 / 127` |
| Predict a result dtype | `np.result_type(a, b, 1.0)` | applies the weak-scalar rule |
| Promote dtypes only | `np.promote_types(np.uint8, np.int8)` | `int16`; ignores weak scalars |
| Grid with endpoints | `np.linspace(a, b, n)` | endpoint included; `retstep=True` gives the spacing |
| Integer range | `np.arange(n)` | never use a float step |
| Uninitialized buffer | `np.empty(shape)` | garbage until written |
| Same shape and dtype | `np.zeros_like(a)`, `np.empty_like(a)` | preserves memory order |
| From an iterable / bytes | `np.fromiter(it, dtype, count=-1)`, `np.frombuffer(buf, dtype)` | `frombuffer` is read-only |
| Add / drop an axis | `a[:, None]`, `np.expand_dims(a, k)`, `a.squeeze(axis=k)` | `np.newaxis is None` |
| Permute axes | `a.transpose(2, 0, 1)`, `np.moveaxis`, `np.swapaxes` | all views |
| Flatten | `a.ravel()` view when possible, `a.flatten()` always a copy | `a.T.ravel()` must copy |
| Join, new axis | `np.stack(arrs, axis=0)` | input shapes must match exactly |
| Join, existing axis | `np.concatenate(arrs, axis=0)` | all axes but the join axis must match |
| Rows / columns from 1-D | `np.vstack`, `np.column_stack` | `hstack` on 1-D concatenates instead |
| Split | `np.split(a, k, axis)`, `np.array_split` | `split` needs an exact division; both return views |
| NaN test | `np.isnan(a)`, `np.isfinite(a)` | `a == np.nan` is always False |
| NaN-skipping reductions | `np.nansum`, `nanmean`, `nanmax`, `nanargmax`, `nanpercentile` | none of them skip `inf` |
| Flat index to subscripts | `np.unravel_index(np.nanargmax(a), a.shape)` | the `argmax` family returns flat indices |
| Control fp warnings | `with np.errstate(divide='ignore'):`, `np.seterr(all='raise')` | `seterr` returns the old dict |
| Compound dtype | `np.dtype([('id','i4'), ('pos','f8',(3,))])` | field access is a view; pandas for analysis |
| Save / load one array | `np.save(f, a)`, `np.load(f)` | `allow_pickle=False` on load |
| Save many | `np.savez(f, a=a, b=b)`, `np.savez_compressed` | close the returned `NpzFile` |
| Text I/O | `np.savetxt`, `np.loadtxt` | ~100x slower and ~3x larger than `.npy` |
| Bigger than RAM | `np.memmap(path, dtype, mode='r', shape=...)` | or `np.load(path, mmap_mode='r')` for `.npy` |
