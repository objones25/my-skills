# Dense linear algebra: `scipy.linalg` and `np.linalg`

Which call implements the factorization you already know, what it hands back, and where the return convention or an unchecked assumption wrecks the answer silently. **Every number below is real output from this sandbox** (Python 3.11.15, numpy 2.4.4, scipy 1.17.1, OpenBLAS 0.3.31); code first, output in the fenced block after. `t(f)` is a min-of-repeats timer. Prefer `scipy.linalg`: it exposes structure keywords, more drivers, more factorizations. Reach for `np.linalg` when you want array-API-standard batching.

Every block in this file assumes:

```python
import numpy as np
import scipy.linalg as sla
import timeit
rng = np.random.default_rng(0)
t = lambda f: min(timeit.repeat(f, number=10, repeat=5)) / 10   # min-of-repeats timer, seconds
```

Contents: Solving `Ax = b` (and the `ab` banded layout); Least squares; Factorizations (`qr`, `svd`, `eigh`/`eig`,
`cholesky`); Norms and conditioning; Batching; Gotchas; Quick reference.

## Solving `Ax = b`

Never `inv(A) @ b`: an inverse is a factorization plus n solves, and multiplying by it adds a second rounding stage. Never `np.linalg.solve` when you know the structure: it cannot be told, and always runs the general LU. `solve(a, b, lower=False, overwrite_a=False, overwrite_b=False, check_finite=True, assume_a=None, transposed=False)` is the entry point. `assume_a=None` is not "general": scipy scans `a`, infers the structure, and dispatches. `lower` is read only for `'sym'`, `'her'`, `'pos'`.

| `assume_a` | Dispatches to | Asserts |
|---|---|---|
| `'general'`, `'gen'` | `?getrf` + `?getrs`, `?gecon` for `rcond` | nothing |
| `'symmetric'`, `'sym'` | `?sysv` (Bunch-Kaufman) | `a == a.T`; only the `lower`-selected triangle is read |
| `'hermitian'`, `'her'` | `?hesv`; silently becomes `'symmetric'` for real input | `a == a.conj().T` |
| `'positive definite'`, `'pos'` | `?posv` (Cholesky) | symmetric **and** positive definite |
| `'diagonal'`, `'tridiagonal'`, `'banded'` | numpy division by `np.diag(a)`; `?gttrf` + `?gttrs`; `?gbsv` with the band measured by `scipy.linalg.bandwidth` | zero off the diagonal; bandwidth `(1, 1)`; nothing beyond squareness |
| `'upper triangular'`, `'lower triangular'` | `?trtrs` | the other triangle is zero |

```python
rng = np.random.default_rng(0); n = 600
M = rng.standard_normal((n, n)); A = M @ M.T + n*np.eye(n); b = rng.standard_normal(n)   # (600,600) SPD
print({a: round(t(lambda: sla.solve(A, b, assume_a=a)), 4) for a in ('gen','sym','pos',None)}, "inv@b:", round(t(lambda: sla.inv(A) @ b), 4))
B = np.array([[4.,1.,2.], [1.,-3.,0.], [2.,0.,5.]]); D = np.array([[4.,1.,2.], [9.,3.,.5], [7.,-2.,5.]]); r3 = np.array([1., 2., 3.])
try: sla.solve(B, r3, assume_a='pos')
except Exception as e: print("pos on B ->", type(e).__name__, e)
for a in ('gen', 'pos'): x = sla.solve(D, r3, assume_a=a); print(f"{a:4} on D:", x.round(4), "resid %.1e" % np.linalg.norm(D@x - r3))
```

```
{'gen': 0.007, 'sym': 0.006, 'pos': 0.005, None: 0.0054} inv@b: 0.0082
pos on B -> LinAlgError A singular matrix detected: slice(s) [0] are singular.
gen  on D: [ 0.3333 -0.3333  0.    ] resid 2.2e-16
pos  on D: [-0.2216  0.6364  0.625 ] resid 3.2e+00
```

Cholesky is 1.4x the general LU, and auto-detection finds it. The failure modes split. An **indefinite** symmetric matrix fails `?posv`'s pivot test and raises `LinAlgError`. A matrix that is simply not symmetric, whose upper triangle happens to be positive definite, returns a confident answer with residual 3.2. `assume_a` is a promise, not a request. Structure pays far more on a band: same call, one keyword, on a 1000x1000 tridiagonal `T`.

```python
m = 1000                                              # 1000x1000 tridiagonal
T = np.diag(4.0*np.ones(m)) + np.diag(-np.ones(m-1), 1) + np.diag(-np.ones(m-1), -1)
b = rng.standard_normal(m)
for a in ('gen', 'tridiagonal', 'banded', 'diagonal'):
    x = sla.solve(T, b, assume_a=a)
    print("%-12s %.4f s  resid %.2e" % (a, t(lambda: sla.solve(T, b, assume_a=a)), np.linalg.norm(T@x - b)))
```

```
gen          0.0195 s  resid 3.98e-15
tridiagonal  0.0025 s  resid 3.95e-15
banded       0.0016 s  resid 3.99e-15
diagonal     0.0022 s  resid 4.72e+00
```

12x for one string, and `'diagonal'` discards the off-diagonals without a word.

### The `ab` banded layout

`solve_banded(l_and_u, ab, b, ...)` and `solveh_banded(ab, b, ..., lower=False)` want diagonal-ordered storage. The invariant is one line: `ab[u + i - j, j] == A[i, j]`. Row index is the diagonal offset counted down from the top superdiagonal, column index is the matrix column, so each diagonal occupies a row, right-justified for superdiagonals and left-justified for subdiagonals, corners unused.

```python
A = np.array([[5.,2.,1.,0.,0.], [1.,6.,2.,1.,0.], [0.,1.,7.,2.,1.], [0.,0.,1.,8.,2.], [0.,0.,0.,1.,9.]])
l, u, n = 1, 2, 5                                  # 1 subdiagonal, 2 superdiagonals
ab = np.zeros((l+u+1, n))
for k in range(-l, u+1): ab[u-k, max(k,0):n+min(k,0)] = np.diag(A, k)   # k > 0 is a superdiagonal
b = np.array([1., 2., 3., 4., 5.]); x = sla.solve_banded((l, u), ab, b)
print(ab, "\n", x.round(4), np.allclose(A @ x, b),
      all(ab[u+i-j, j] == A[i, j] for i in range(n) for j in range(n) if -u <= i-j <= l))
```

```
[[0. 0. 1. 1. 1.]
 [0. 2. 2. 2. 2.]
 [5. 6. 7. 8. 9.]
 [1. 1. 1. 1. 0.]] 
 [0.0794 0.1863 0.2303 0.3418 0.5176] True True
```

`ab` is `(l+u+1, n)` and the main diagonal is row `u`. `solveh_banded` takes SPD input in half of that: with `lower=False` (default) the **last** row is the diagonal and superdiagonals stack above; with `lower=True` the **first** row is the diagonal. For the 4x4 SPD matrix with diagonal `[4,5,6,7]` and unit off-diagonals, upper form `[[0,1,1,1],[4,5,6,7]]` gives `[0.177 0.292 0.365 0.519]`, matching `solve`. `solve_toeplitz(c_or_cr, b)` takes the first column, or a `(c, r)` tuple of first column and first row, not a matrix, and runs in O(n^2). `solve_circulant(c, b, singular='raise', ...)` takes only the first column and solves by FFT in O(n log n); a singular circulant raises `LinAlgError: near singular circulant matrix.` unless you pass `singular='lstsq'`.

`lu_factor` returns `(lu, piv)`: `lu` packs L (unit diagonal, implicit) and U into one array, `piv` is the int32 row-swap record, and feeding that tuple to `lu_solve(lu_and_piv, b, trans=0, ...)` reuses the factorization for every new right-hand side. Measured on an 800x800 general matrix with 200 columns of RHS: `piv` comes back as `(800,) int32 [443 322 9 142 ...]`, 200 fresh `solve` calls take `2.867 s`, one `lu_solve` on the cached factor takes `0.0050 s`, a 575x win, and `A @ lu_solve((lu, piv), B)` is `allclose` to `B`. One `solve(A, B)` on all 200 columns also factors once; `lu_solve` is for right-hand sides that arrive over time (time stepping, Newton with a frozen Jacobian). `sla.lu(a, permute_l=False, overwrite_a=False, check_finite=True, p_indices=False)` returns readable `(P, L, U)` with `A == P@L@U`, for inspection rather than solving.

## Least squares

`scipy.linalg.lstsq(a, b, cond=None, overwrite_a=False, overwrite_b=False, check_finite=True, lapack_driver=None)` returns `x, residues, rank, s`; `numpy.linalg.lstsq(a, b, rcond=None)` returns `x, residuals, rank, s`. Same four names, different types. scipy's `residues` is a `np.float64` scalar when the system is overdetermined and full rank; numpy's `residuals` is an ndarray of shape `(1,)`, and is **empty** when `rank < N` or `M <= N`. numpy's `rcond=None` means `eps * max(M, N)`; it was `-1` before **NumPy 2.0** and the docstring still carries a `versionchanged:: 2.0` note. scipy's equivalent knob is `cond`. scipy's `lapack_driver=None` picks `gelsd` (SVD, divide and conquer); `'gelsy'` (complete orthogonal factorization) is faster. Three routes to one overdetermined system: a degree-11 Vandermonde design over `[0,1]`, 40 rows, true coefficients `1..12`, `b` exactly in the column space.

```python
tt = np.linspace(0, 1, 40); A = np.vander(tt, 12, increasing=True)   # (40,12) Vandermonde
c_true = np.arange(1., 13.); b = A @ c_true                          # b exactly in range(A)
print("cond(A) = %.3e   cond(A.T@A) = %.3e" % (np.linalg.cond(A), np.linalg.cond(A.T@A)))
c_ne = sla.solve(A.T @ A, A.T @ b, assume_a='pos')                   # normal equations
Q, R = sla.qr(A, mode='economic'); c_qr = sla.solve_triangular(R, Q.T @ b)   # Q (40,12) R (12,12)
c_ls, res, rank, sv = sla.lstsq(A, b)                                # SVD via gelsd
err = lambda c: np.linalg.norm(c - c_true)/np.linalg.norm(c_true)     # relative error against c_true
for nm, c in (("normal eq", c_ne), ("QR       ", c_qr), ("lstsq    ", c_ls)):
    print("%s rel err %.3e  resid %.3e" % (nm, err(c), np.linalg.norm(A@c - b)))
print("rank", rank, "| s[0]/s[-1] = %.3e" % (sv[0]/sv[-1]), "| pinv rel err %.3e" % err(sla.pinv(A) @ b))
```

```
cond(A) = 1.179e+08   cond(A.T@A) = 1.343e+16
LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 3.4736835068675834e-17.
normal eq rel err 1.409e-02  resid 2.737e-08
QR        rel err 2.779e-09  resid 2.107e-14
lstsq     rel err 6.385e-10  resid 6.338e-14
rank 12 | s[0]/s[-1] = 1.179e+08 | pinv rel err 4.534e-09
```

`cond(A.T@A)` matches `cond(A)**2` to three digits, and `1.34e16` is past `1/eps = 4.5e15`, so `A.T@A` is numerically singular while `A` has full rank 12. Normal equations keep **1 correct digit**; QR and `lstsq` keep 9 to 10. The residual column shows why a residual is not a diagnostic: `2.7e-08`, on an answer 1.4% wrong (see `04-floating-point.md` for residual versus error and the digits-lost rule).

`pinv(a, *, atol=None, rtol=None, return_rank=False, check_finite=True)` builds the pseudoinverse by SVD, the same accuracy class as `lstsq` at the cost of an `(N, M)` array you then discard. `rtol` truncates singular values below `rtol * s[0]`: `pinv(hilbert(14), rtol=1e-10, return_rank=True)[1]` is 9 against `matrix_rank`'s default 11. Constrained variants live in `scipy.optimize`: `lsq_linear(A, b, bounds=(-inf, inf), method='trf', tol=1e-10, ...)` returns an `OptimizeResult` (`.x`, `.cost`, `.active_mask`), `nnls(A, b, *, maxiter=None, atol=...)` returns `(x, rnorm)`. On a 20x4 problem with one true coefficient at `-1.0`, `lsq_linear(..., bounds=(0, np.inf)).x` and `nnls(A, b)[0]` agree to all printed digits at `[2.3571 0. 0.4977 3.3988]`.

## Factorizations

### `qr`

One matrix threads this subsection: a `(6, 4)` normal matrix with column 3 overwritten by column 0 plus column 1, so `matrix_rank(A) == 3`. The seed is chosen so the pivot order is non-trivial.

```python
A = np.random.default_rng(4).standard_normal((6, 4)); A[:, 3] = A[:, 0] + A[:, 1]
shp = lambda o: [shp(e) for e in o] if isinstance(o, tuple) else np.shape(o)
print({m: shp(sla.qr(A, mode=m)) for m in ('full','economic','r','raw')}, "| recon:", np.allclose(A, np.matmul(*sla.qr(A))), np.allclose(A, np.matmul(*sla.qr(A, mode='economic'))))
Qp, Rp, P = sla.qr(A, pivoting=True); iP = np.empty_like(P); iP[P] = np.arange(len(P))   # invert P
print("P =", P, P.dtype, "| A[:,P] == Qp@Rp:", np.allclose(A[:, P], Qp@Rp), "| A == (Qp@Rp)[:,iP]:", np.allclose(A, (Qp@Rp)[:, iP]), "| |diag Rp| =", np.abs(np.diag(Rp)).round(4))
```

```
{'full': [(6, 6), (6, 4)], 'economic': [(6, 4), (4, 4)], 'r': [(6, 4)], 'raw': [[(6, 4), (4,)], (4, 4)]} | recon: True True
P = [0 2 1 3] int32 | A[:,P] == Qp@Rp: True | A == (Qp@Rp)[:,iP]: True | |diag Rp| = [3.1474 2.3804 1.0878 0.    ]
```

`'full'` gives square `(M, M)` Q and `(M, N)` R; `'economic'` gives `(M, N)` and `(N, N)`, the one for least squares. `'r'` returns a **1-tuple**, so write `R, = sla.qr(A, mode='r')` with the trailing comma. `'raw'` returns `((qr, tau), R)`, the packed Householder reflectors for `?ormqr`. With `pivoting=True`, `P` is an **index array**, not a permutation matrix: it factors `A[:, P]`, so you apply it by fancy-indexing columns and undo it with the scatter-inverse above. `diag(R)` comes out in decreasing magnitude, and the trailing zero reads off rank 3. `qr_update(Q, R, u, v, overwrite_qruv=False, check_finite=True)` refactors `A + np.outer(u, v)` in O(mn); siblings `qr_insert`, `qr_delete`.

### `svd`

```python
rng = np.random.default_rng(0)                    # the svd, eigh, and cholesky blocks below share this stream
B = rng.standard_normal((6, 3))                   # 6x4 of rank 3: 4th column is the sum
A = np.column_stack([B, B.sum(axis=1)])           # of the first three
U, s, Vh = sla.svd(A); Ue, se, Vhe = sla.svd(A, full_matrices=False)
print("U", U.shape, "s", s.shape, s.round(4), "Vh", Vh.shape, "| economy:", Ue.shape, Vhe.shape)
k = len(s)                                        # reconstruct with the first k columns of U
print("U[:,:k]@diag(s)@Vh:", np.allclose(A, U[:, :k] @ np.diag(s) @ Vh), "| ...@Vh.T:", np.allclose(A, U[:, :k] @ np.diag(s) @ Vh.T), "| economy:", np.allclose(A, Ue @ np.diag(se) @ Vhe))
print("rank", np.linalg.matrix_rank(A), "| null_space", sla.null_space(A).shape, "| orth", sla.orth(A).shape, "| |A@ns| = %.0e" % np.abs(A @ sla.null_space(A)).max(),
      "| ||A-A2||_2 = %.4f = s[2] = %.4f" % (np.linalg.norm(A - Ue[:, :2] @ np.diag(se[:2]) @ Vhe[:2], 2), se[2]))
```

```
U (6, 6) s (4,) [5.8112 1.7219 0.522  0.    ] Vh (4, 4) | economy: (6, 4) (4, 4)
U[:,:k]@diag(s)@Vh: True | ...@Vh.T: False | economy: True
rank 3 | null_space (4, 1) | orth (6, 3) | |A@ns| = 2e-16 | ||A-A2||_2 = 0.5220 = s[2] = 0.5220
```

Three conventions. `s` is a **1-D array** in descending order, not a diagonal matrix, so reconstruction needs `np.diag(s)`. The third return is **V-transpose already**: right singular vectors are the **rows** of `Vh`, and `V = Vh.conj().T`. `full_matrices=True` (the default) makes `U` `(6, 6)` while `s` has 4 entries, so the product needs `U[:, :len(s)]`; `full_matrices=False` is what you almost always want. Eckart-Young is the last field: the 2-norm error of the rank-2 truncation equals `s[2]` exactly. `svdvals(a)` skips the vectors. `lapack_driver='gesvd'` is the slower, more reliable fallback when the default `gesdd` raises `LinAlgError: SVD did not converge`. `null_space(A, rcond=None, *, overwrite_a=False, check_finite=True, lapack_driver='gesdd')`, `orth(A, rcond=None)` and `np.linalg.matrix_rank(A, tol=None, hermitian=False, *, rtol=None)` all read the same SVD.

### `eigh` and `eig`

If the matrix is symmetric or Hermitian, use `eigh`, always: half the work, real **ascending** eigenvalues, an orthonormal basis, and partial spectra. Signatures are `eigh(a, b=None, *, lower=True, eigvals_only=False, overwrite_a=False, overwrite_b=False, type=1, check_finite=True, subset_by_index=None, subset_by_value=None, driver=None)` and `eig(a, b=None, left=False, right=True, overwrite_a=False, overwrite_b=False, check_finite=True, homogeneous_eigvals=False)`.

```python
M = rng.standard_normal((4, 4)); S = M @ M.T + 0.25*np.eye(4)   # symmetric positive definite
w, V = sla.eigh(S); we, _ = sla.eig(S)
print("eigh w:", w.round(4), "| recon:", np.allclose(S, V @ np.diag(w) @ V.T), "| V.T@V == I:", np.allclose(V.T@V, np.eye(4)))
print("eig  w:", we.round(4), we.dtype, "| eig on a 2x2 rotation block plus [2]:", sla.eig(np.diag([0.,0.,2.]) + np.diag([-1.,0.], 1) + np.diag([1.,0.], -1))[0])
print("subset_by_index=[2,3]:", sla.eigh(S, subset_by_index=[2, 3])[0].round(4), "| subset_by_value=(2,inf):", sla.eigh(S, subset_by_value=(2, np.inf))[0].round(4))
B = np.diag([1., 2., 3., 4.]); wg, Vg = sla.eigh(S, B)
print("eigh(S,B):", wg.round(4), np.allclose(S@Vg, B@Vg@np.diag(wg)), np.allclose(Vg.T@B@Vg, np.eye(4)))
```

```
eigh w: [0.2833 1.7195 2.6665 3.9033] | recon: True | V.T@V == I: True
eig  w: [0.2833+0.j 3.9033+0.j 1.7195+0.j 2.6665+0.j] complex128 | eig on a 2x2 rotation block plus [2]: [0.+1.j 0.-1.j 2.+0.j]
subset_by_index=[2,3]: [2.6665 3.9033] | subset_by_value=(2,inf): [2.6665 3.9033]
eigh(S,B): [0.0863 0.7812 0.8983 3.4893] True True
```

`eig` on the *same symmetric matrix* returns `complex128` ordered `[0.28, 3.90, 1.72, 2.67]`, neither ascending nor descending: `?geev` emits eigenvalues in the order the QR iteration deflates them. The dtype is complex regardless of input because a real matrix can have complex eigenvalues, as the rotation block shows. Sort with `np.argsort(w.real)` and take `.real` only after checking `np.abs(w.imag).max()`. `eig` returns **right** eigenvectors as columns of the second array, `A @ V[:, i] == w[i] * V[:, i]`, neither orthogonal nor scaled to anything but unit 2-norm; `left=True` adds the left set. `eigh(S, B)` solves `S v = w B v` for SPD `B` and returns a `B`-orthonormal basis; `type=2` and `type=3` switch to `S B v = w v` and `B S v = w v`. `subset_by_index` is an inclusive 0-based `[lo, hi]` rank pair, `subset_by_value` a half-open interval; either switches the driver to `?syevr`/`?heevr` and skips the rest of the spectrum. `eigvals_only=True`, or the shorthands `eigvalsh`/`eigvals`, skip the vectors.

### `cholesky`

scipy returns **upper** by default, numpy returns **lower**. Mixing them reconstructs `A.T`.

```python
Us = sla.cholesky(S); Ln = np.linalg.cholesky(S); c, low = sla.cho_factor(S)  # scipy lower=False, numpy upper=False
print("scipy row 0:", Us[0].round(4), "| numpy row 0:", Ln[0].round(4))
print("Us.T@Us == S:", np.allclose(Us.T@Us, S), "| Ln@Ln.T == S:", np.allclose(Ln@Ln.T, S), "| Us == Ln.T:", np.allclose(Us, Ln.T), "| cho_factor low =", low, "| max|tril(c,-1)| = %.4f" % np.abs(np.tril(c, -1)).max())
x = sla.cho_solve((c, low), np.ones(4)); print("cho_solve:", x.round(4), np.allclose(S @ x, np.ones(4)))
```

```
scipy row 0: [ 1.8412  0.057  -0.4927  0.0684] | numpy row 0: [1.8412 0.     0.     0.    ]
Us.T@Us == S: True | Ln@Ln.T == S: True | Us == Ln.T: True | cho_factor low = False | max|tril(c,-1)| = 1.1354
cho_solve: [ 0.6469  0.9517  1.3336 -0.6599] True
```

`np.linalg.cholesky(a, /, *, upper=False)` takes `upper=True` to match scipy. `cho_factor` returns `(c, lower)`; on 1.17 the *unused* triangle still holds original matrix entries (1.14, not 0), while 1.18 zeroes it and returns `lower` as a 0-d bool array instead of a `bool`. Either way, pass the tuple to `cho_solve` rather than slicing `c` as a triangular factor. `cholesky` zeroes the other triangle.

`schur(a, output='real', lwork=None, overwrite_a=False, sort=None, check_finite=True)` gives `(T, Z)` with `A == Z@T@Z.T` and `Z` orthogonal; `output='complex'` makes `T` strictly upper triangular, the real form leaves 2x2 blocks for complex pairs. `polar(a, side='right')` gives `(U, P)` with `A == U@P`, `U` unitary, `P` SPD. `expm(A)` (scaling and squaring) matches `funm(A, np.exp)` to `allclose`; `sqrtm(A)` and `funm(A, func, disp=True)` go complex when the spectrum demands it. Constructors: `block_diag`, `toeplitz(c, r=None)` (symmetric when `r` is omitted), `circulant`, `hankel`, `hilbert(n)` (`cond(hilbert(5)) = 4.77e+05`, a ready-made conditioning test bed), `invhilbert(n)` (exact).

## Norms and conditioning

`np.linalg.norm(x, ord=None, axis=None, keepdims=False)` and `scipy.linalg.norm(a, ord=None, axis=None, keepdims=False, check_finite=True)` agree numerically; scipy adds the finiteness check. The array-API pair removes the `ndim`-dependent meaning: `np.linalg.matrix_norm(x, /, *, keepdims=False, ord='fro')` always treats the last two axes as a matrix and batches over the rest, `np.linalg.vector_norm(x, /, *, axis=None, keepdims=False, ord=2)` always flattens.

| `ord` | 1-D vector | 2-D matrix |
|---|---|---|
| `None` | 2-norm | Frobenius |
| `1` | `sum(abs(x))` | max absolute **column** sum |
| `2` | Euclidean | largest singular value (an SVD) |
| `inf` / `-inf` | `max` / `min` of `abs(x)` | max / min absolute **row** sum |
| `'fro'`, `'nuc'` | not defined | `sqrt(sum(s**2))` entrywise 2-norm; `sum(s)`, the convex relaxation of rank |

`ord=0` counts nonzeros in a vector and is undefined for a matrix; `ord=-2` is the smallest singular value of a matrix and undefined for a vector. On `A = [[1,2,3],[4,5,6],[7,8,10]]` with `s = [17.4125, 0.8752, 0.1969]`: `ord=2` gives `17.412505` (= `s[0]`), `'fro'` `17.435596`, `'nuc'` `18.484533` (= `s.sum()`), `ord=-2` `0.196867` (= `s[-1]`). `np.linalg.cond(x, p=None)` defaults to `p=2` and returns `s[0]/s[-1]` by SVD, here `88.448280`, matching the ratio exactly; `cond(A, 1)` is `norm(A,1)*norm(inv(A),1) = 133.0` and costs an LU, so it is the cheap screen. `scipy.linalg.solve` computes `rcond` internally and emits `LinAlgWarning: An ill-conditioned matrix detected: slice 0 has rcond = 1.33e-18` on `hilbert(14)`. `04-floating-point.md` converts `cond` into digits lost.

## Batching: stacked `(k, n, n)` inputs

`np.linalg.*` is batched per the array API: the last two axes are the matrix, everything before is batch. `scipy.linalg` accepts stacked input from 1.16 on, but on 1.16 and 1.17 **batched factorizations return a `list`, not a `tuple`**. On 1.18.1 batched `svd` and `qr` return a `tuple` while `eigh` and `lu_factor` still return a `list`, and the `shp` helper below then fails on the tuple results.

```python
rng = np.random.default_rng(8); k, n = 3, 4
mk = lambda k: (lambda M: M @ M.transpose(0, 2, 1) + n*np.eye(n))(rng.standard_normal((k, n, n)))
As, Aq = mk(k), mk(n)                        # As is (3,4,4); Aq is (4,4,4), so k == n
shp = lambda r: (type(r).__name__, [np.shape(e) for e in r] if isinstance(r, list) else np.shape(r)); bq = rng.standard_normal((n, n))
for nm, f in (("svd", sla.svd), ("eigh", sla.eigh), ("qr", sla.qr), ("lu_factor", sla.lu_factor), ("cholesky", sla.cholesky)): print(nm, "->", *shp(f(As)))
print("unbatched svd ->", type(sla.svd(As[0])).__name__, "| solve(As, b=(4,)) ->", sla.solve(As, rng.standard_normal(n)).shape)
try: sla.solve(As, rng.standard_normal((k, n)))
except Exception as e: print("solve(As, b=(3,4)) ->", type(e).__name__, e)
print("solve(Aq, b=(4,4)) ->", sla.solve(Aq, bq).shape, np.linalg.solve(Aq, bq).shape, "| read as a broadcast matrix:", np.allclose(Aq @ sla.solve(Aq, bq), np.broadcast_to(bq, (n, n, n))),
      "| b[...,None] ->", sla.solve(Aq, bq[..., None]).shape, np.allclose((Aq @ sla.solve(Aq, bq[..., None]))[..., 0], bq))
```

```
svd -> list [(3, 4, 4), (3, 4), (3, 4, 4)]
eigh -> list [(3, 4), (3, 4, 4)]
qr -> list [(3, 4, 4), (3, 4, 4)]
lu_factor -> list [(3, 4, 4), (3, 4)]
cholesky -> ndarray (3, 4, 4)
unbatched svd -> tuple | solve(As, b=(4,)) -> (3, 4)
solve(As, b=(3,4)) -> ValueError incompatible shapes: a1.shape=(3, 4, 4) and b1.shape=(3, 4)
solve(Aq, b=(4,4)) -> (4, 4, 4) (4, 4, 4) | read as a broadcast matrix: True | b[...,None] -> (4, 4, 1) True
```

Same call, different container depending on whether the input was stacked (and, on 1.18, on which factorization). Unpacking (`U, s, Vh = sla.svd(As)`) works on either, so this only bites code that tests `isinstance(out, tuple)`, does `out + (extra,)`, or caches the return. The right-hand side is the real trap: `solve` documents `b` as `(..., N, NRHS)`, so a trailing axis is always the **column index of a multi-RHS matrix**, never a batch index. A `(k, n)` array meaning one vector per batch element raises `ValueError` when `k != n`. When `k == n` the shapes line up and nothing complains: `(4, 4, 4)` against `(4, 4)` returns `(4, 4, 4)` from both libraries, reading `b` as one 4-column right-hand side broadcast across all 4 matrices. Write `b[..., None]` whenever you mean one vector per batch element.

## Gotchas

- **`svd` returns `Vh`, not `V`.** Symptom: reconstruction or a projection is wrong, but shapes match and nothing raises. Cause: the third return is already `V.conj().T`; right singular vectors are its **rows**. Fix: `A == U[:, :k] @ np.diag(s) @ Vh` with `k = len(s)`, or `full_matrices=False` to drop the slice. `s` is 1-D, so `np.diag(s)` is mandatory. `np.linalg.svd` is identical, returning an `SVDResult` named tuple with fields `U, S, Vh`.
- **`eig` output is unsorted and complex.** Symptom: `w[0]` is not the dominant eigenvalue, or downstream code chokes on `complex128`. Cause: `?geev` emits in deflation order and cannot promise real output for real input. Measured on symmetric `S`: `[0.2833+0.j, 3.9033+0.j, 1.7195+0.j, 2.6665+0.j]`. Fix: `eigh` for symmetric or Hermitian input (real, ascending). Otherwise sort by `np.argsort(w.real)` and take `.real` only after checking `np.abs(w.imag).max()`.
- **scipy `cholesky` is upper, numpy `cholesky` is lower.** Symptom: `L @ L.T` reconstructs `A.T`, or a triangular solve returns nonsense. Cause: `scipy.linalg.cholesky(a, lower=False)` against `np.linalg.cholesky(a, /, *, upper=False)`. Fix: name the keyword every time. `cho_factor` on 1.17 additionally leaves the unused triangle full of original entries, so pass its `(c, lower)` tuple to `cho_solve` rather than slicing `c`.
- **`inv(A) @ b`.** Symptom: slower code, lost digits, no error. Cause: an inverse is a factorization plus n solves, and multiplying by it adds a second rounding stage. Measured: `inv` `0.0082 s` against `solve(..., assume_a='pos')` `0.0050 s` on a 600x600 SPD matrix. Fix: `solve`, or `lu_factor`/`cho_factor` plus repeated `*_solve`. Genuinely needing `A^{-1}` as an object is rare.
- **Forming `A.T @ A`.** Symptom: the fit degrades sharply as the model gets richer while the residual stays small. Cause: `cond(A.T@A) == cond(A)**2`, so the available digits halve. Measured on a degree-11 Vandermonde: `cond(A) = 1.18e+08`, `cond(A.T@A) = 1.34e+16`, normal-equation relative error `1.4e-02` against `lstsq`'s `6.4e-10`. Fix: `lstsq`, or economy `qr` plus `solve_triangular`. Never build normal equations for a fit you care about.
- **`assume_a` is an unchecked promise.** Symptom: a plausible answer with a residual of order 1. Cause: `solve` reads only the triangle or band you claimed. Measured: `assume_a='pos'` on a nonsymmetric 3x3 whose upper triangle is SPD gives residual `3.2`; `assume_a='diagonal'` on a tridiagonal matrix gives residual `4.7`. Only an indefinite *symmetric* matrix reliably raises. Fix: check `np.allclose(A, A.T)` and `scipy.linalg.bandwidth(A)` once outside the loop, or leave `assume_a=None` and let scipy detect the structure.

## Quick reference

| Task | Call | Note |
|---|---|---|
| square solve | `scipy.linalg.solve(a, b, assume_a=)` | `(a, b, lower=False, overwrite_a=False, overwrite_b=False, check_finite=True, assume_a=None, transposed=False)`; `None` auto-detects |
| structure keywords | `'gen'`, `'sym'`, `'her'`, `'pos'`, `'diagonal'`, `'tridiagonal'`, `'banded'`, `'upper/lower triangular'` | `?getrf`, `?sysv`, `?hesv`, `?posv`, numpy divide, `?gttrf`, `?gbsv`, `?trtrs`; never verified |
| triangular, banded | `solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False, ...)`; `solve_banded((l, u), ab, b)`, `solveh_banded(ab, b, lower=False)` | `trans` in `0, 1, 2` = `A`, `A.T`, `A.H`; `ab[u+i-j, j] == A[i, j]`, shape `(l+u+1, n)`, upper `solveh_banded` puts the diagonal in the **last** row |
| Toeplitz, circulant | `solve_toeplitz(c_or_cr, b)`, `solve_circulant(c, b, singular='raise')` | first column (or `(c, r)`), not the matrix; `singular='lstsq'` for min-norm |
| many right-hand sides | `lu, piv = lu_factor(A)`; `lu_solve((lu, piv), b)` | 575x over refactoring per RHS; `lu(a, ..., p_indices=False)` gives readable `(P, L, U)` |
| least squares | `scipy.linalg.lstsq(a, b, cond=None, ..., lapack_driver=None)`, or `Q, R = qr(A, mode='economic')` then `solve_triangular(R, Q.T @ b)` | `-> x, residues, rank, s`, driver `gelsd`; both are 7 digits better than `solve(A.T@A, A.T@b)` at `cond(A) = 1.2e8`; numpy's is `(a, b, rcond=None)`, `rcond=None` meaning `eps*max(M,N)` since **NumPy 2.0** (was `-1`) |
| pseudoinverse, constrained LS | `pinv(a, *, atol=None, rtol=None, return_rank=False)`; `scipy.optimize.lsq_linear(A, b, bounds=)`, `scipy.optimize.nnls(A, b)` | `pinv` truncates `s < rtol*s[0]`; `lsq_linear` returns an `OptimizeResult`, `nnls` the plain tuple `(x, rnorm)` |
| QR | `qr(a, overwrite_a=False, lwork=None, mode='full', pivoting=False, check_finite=True)` | `'economic'`, `'r'` (a 1-tuple), `'raw'`; `pivoting=True` gives `(Q, R, P)` with `A[:, P] == Q@R`, `P` an int32 index array; `qr_update` for rank-1 |
| SVD | `svd(a, full_matrices=True, compute_uv=True, ..., lapack_driver='gesdd')` | `-> U, s, Vh`; `s` 1-D descending, third return is **V-transpose**; `'gesvd'` when `gesdd` will not converge; `svdvals` for values only; `matrix_rank(A, ..., rtol=)`, `null_space(A, rcond=)`, `orth(A, rcond=)` read the same SVD, so tune the cutoff before trusting a rank |
| symmetric eigen | `eigh(a, b=None, *, lower=True, eigvals_only=False, type=1, subset_by_index=None, subset_by_value=None, driver=None)` | real, **ascending**, orthonormal `V`; `eigh(A, B)` solves `Av = wBv` with `V.T@B@V == I`; `eig(a, b=None, left=False, right=True, ...)` for general input is `complex128` always and **unsorted**, right eigenvectors in the columns of the second return |
| Cholesky | `scipy.linalg.cholesky(a, lower=False)`, `np.linalg.cholesky(a, *, upper=False)` | opposite defaults; `cho_factor`/`cho_solve` for repeated RHS |
| other factorizations, matrix functions | `schur`, `polar`, `qz`, `rq`, `ldl`, `hessenberg`; `expm`, `sqrtm`, `funm`, `logm`, `expm_multiply` | `schur(output='complex')` for strictly triangular `T`; `expm_multiply(A, B)` when you only need `expm(A) @ B` |
| constructors | `block_diag`, `toeplitz`, `circulant`, `hankel`, `hilbert`, `invhilbert`, `pascal` | `cond(hilbert(5)) = 4.77e+05`, a ready-made ill-conditioned test bed |
| norms, conditioning | `np.linalg.norm(x, ord=)`, `matrix_norm`, `vector_norm`; `np.linalg.cond(x, p=None)` | the array-API pair fixes the meaning and batches; `p=2` is `s[0]/s[-1]` by SVD, `p=1` is a cheap LU estimate |
| batched | `np.linalg.*` per the array API; `scipy.linalg.*` accepts `(k, n, n)` from 1.16 | scipy factorizations return a **list** when batched on 1.16/1.17 (1.18: `svd`/`qr` tuple, `eigh`/`lu_factor` list); `b` is `(..., N, NRHS)`, so use `b[..., None]` for one vector per batch element |
