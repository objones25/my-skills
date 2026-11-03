# Sparse matrices and iterative solvers: `scipy.sparse`, `scipy.sparse.linalg`

Which container to build, which solver to call, and the API conventions that return a wrong answer without raising. **Every number below is real output from this sandbox** (Python 3.11.15, numpy 2.4.4, scipy 1.17.1); code first, output in the fenced block after. One example runs through the whole file: the 1-D Dirichlet Poisson operator `u'' = f`, the tridiagonal `(1, -2, 1)/h^2` on an interior grid of `n` points with `h = 1/(n+1)`, and its 2-D Kronecker extension.

> Timings and speed ratios in this file: date and machine not recorded. They vary by machine.

Every block in this file assumes:

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
rng = np.random.default_rng(0)
def poisson1d(n, fmt='csr'):                     # (1, -2, 1)/h^2 on n interior points, h = 1/(n+1)
    h = 1.0/(n+1)
    return sp.diags_array([np.ones(n-1), -2*np.ones(n), np.ones(n-1)], offsets=[-1, 0, 1], format=fmt)/h**2
```

Contents: `*_array` vs `*_matrix`; Formats; Inspection and conversion; Direct solves and fill-in; Krylov solvers;
Preconditioning; `LinearOperator`; Sparse eigenvalues and SVD; Gotchas; Quick reference.

## `*_array` is the supported path, `*_matrix` is legacy

The `*_matrix` classes inherit `np.matrix` semantics: always 2-D, and `*` is matrix multiplication. The `*_array` classes follow ndarray semantics: `*` is elementwise, `@` is matrix multiplication, and a row can come back 1-D.

```python
n = 6; h = 1.0/(n+1)                                        # T has -98 on the diagonal, 49 off it
T = sp.diags_array([np.ones(n-1), -2*np.ones(n), np.ones(n-1)], offsets=[-1, 0, 1], format='csr')/h**2
Tm = sp.csr_matrix(T)                                       # legacy twin, same entries
print(type(T).__name__, T.shape, "nnz", T.nnz, T.dtype, T.format)
print("csr_array  T*T  nnz", (T*T).nnz,  "elementwise:", np.allclose((T*T).toarray(), T.toarray()**2))
print("csr_matrix Tm*Tm nnz", (Tm*Tm).nnz, "matmul:    ", np.allclose((Tm*Tm).toarray(), (T@T).toarray()))
print("T[0] ->", type(T[0]).__name__, T[0].shape, "| Tm[0] ->", type(Tm[0]).__name__, Tm[0].shape)
```
```
csr_array (6, 6) nnz 16 float64 csr
csr_array  T*T  nnz 16 elementwise: True
csr_matrix Tm*Tm nnz 24 matmul:     True
T[0] -> coo_array (6,) | Tm[0] -> csr_matrix (1, 6)
```
Same `*`, two different operations, both silent. The rule: build `csr_array` / `csc_array` / `coo_array`, and write `@` for every product. `sp.issparse(x)` is true for both families; `isinstance(x, sp.sparray)` is the test that separates them (`True` for `csr_array`, `False` for `csr_matrix`).

## Formats: build in one, compute in another

| Format | Built by | Good at | Cannot |
|---|---|---|---|
| COO | `coo_array((vals, (rows, cols)), shape=)` | assembling from triplets, cheap conversion; duplicate `(i, j)` entries are **summed** on conversion | indexing, arithmetic, slicing |
| CSR | `.tocsr()`, `csr_array` | row slicing, `A @ v`, what nearly every solver wants | cheap structural change |
| CSC | `.tocsc()`, `csc_array` | column slicing; `splu`, `spsolve`, `spilu` all convert to it | cheap structural change |
| LIL, DOK | `lil_array`, `dok_array` | incremental `A[i, j] = v` assignment | arithmetic (convert first) |
| DIA, BSR | `diags_array`; `bsr_array(..., blocksize=)` | banded operators stored as offset diagonals; dense small blocks (several PDE unknowns per node) | any nonzero off a stored diagonal; scalar-level indexing |

The canonical assembly is triplets to COO to CSR. Building directly into CSR inside a loop reallocates the whole `indices`/`data` array on every structural insertion, which is why it warns. Structured operators skip triplets: `diags_array(diagonals, /, *, offsets=0, shape=None, format=None, dtype=)` takes diagonals and their offsets, `eye_array(m, n=None, *, k=0, dtype=float, format=None)` gives a sparse identity, and `kron` lifts the 1-D operator to a tensor-product grid, so the 5-point 2-D Laplacian is `kron(I, T) + kron(T, I)`.

```python
rows, cols, vals = [], [], []
for i in range(n):                                          # one triplet per contribution
    for j, v in ((i, -2.0/h**2), (i-1, 1.0/h**2), (i+1, 1.0/h**2)):
        if 0 <= j < n: rows.append(i); cols.append(j); vals.append(v)
C = sp.coo_array((vals, (rows, cols)), shape=(n, n))
print("coo nnz", C.nnz, "-> tocsr nnz", C.tocsr().nnz, "| equals diags_array:", np.allclose(C.toarray(), T.toarray()))
D = sp.coo_array(([1.0, 2.0], ([0, 0], [0, 0])), shape=(2, 2))          # two entries at (0,0)
print("duplicates: coo nnz", D.nnz, "-> csr nnz", D.tocsr().nnz, "value", D.tocsr()[0, 0])
m = 100; hh = 1.0/(m+1); I = sp.eye_array(m, format='csr')
T1 = sp.diags_array([np.ones(m-1), -2*np.ones(m), np.ones(m-1)], offsets=[-1, 0, 1], format='csr')/hh**2
A2 = sp.csr_array(sp.kron(I, T1) + sp.kron(T1, I))
print(type(A2).__name__, A2.shape, "nnz", A2.nnz, "density %.2e" % (A2.nnz/np.prod(A2.shape)),
      "| dense float64 would be %.1f GB" % (A2.shape[0]*A2.shape[1]*8/1e9))
B = sp.csr_array((3, 3)); B[0, 1] = 5.0                                 # structural insertion into CSR
```
```
coo nnz 16 -> tocsr nnz 16 | equals diags_array: True
duplicates: coo nnz 2 -> csr nnz 1 value 3.0
csr_array (10000, 10000) nnz 49600 density 4.96e-04 | dense float64 would be 0.8 GB
SparseEfficiencyWarning: Changing the sparsity structure of a csr_array is expensive. lil and dok are more efficient.
```
Duplicate summing is the feature that makes finite-element assembly work: emit one triplet per element contribution and let the conversion accumulate them. `kron` returns COO, so wrap it in `csr_array` before solving. `sp.block_array`, `sp.hstack`, `sp.vstack` assemble block systems from sub-blocks.

## Inspection, conversion, and numpy's silent failures

```python
print("csr_array : toarray ->", type(T.toarray()).__name__,  " todense ->", type(T.todense()).__name__)
print("csr_matrix: toarray ->", type(Tm.toarray()).__name__, " todense ->", type(Tm.todense()).__name__)
print("T.sum() ->", type(T.sum()).__name__, "| T.sum(axis=0) ->", type(T.sum(axis=0)).__name__, T.sum(axis=0).shape,
      "| Tm.sum(axis=0) ->", type(Tm.sum(axis=0)).__name__, Tm.sum(axis=0).shape)
print("np.sum(T)        ->", repr(np.sum(T)))
print("np.dot(T, ones)  ->", type(np.dot(T, np.ones(n))).__name__, np.dot(T, np.ones(n)).shape, np.dot(T, np.ones(n)).dtype)
print("np.asarray(T)    ->", np.asarray(T).shape, np.asarray(T).dtype)
try: T + 1
except Exception as e: print("T + 1            ->", type(e).__name__ + ":", e)
```
```
csr_array : toarray -> ndarray  todense -> ndarray
csr_matrix: toarray -> ndarray  todense -> matrix
T.sum() -> float64 | T.sum(axis=0) -> ndarray (6,) | Tm.sum(axis=0) -> matrix (1, 6)
np.sum(T)        -> np.float64(-98.00000000000001)
np.dot(T, ones)  -> ndarray (6,) object
np.asarray(T)    -> () object
T + 1            -> NotImplementedError: adding a nonzero scalar to a sparse array is not supported
```
Read those four numpy lines carefully. `np.sum` happens to work because it defers to `T.sum()`. `np.dot` does **not** raise: it treats `T` as a scalar object and returns a `(6,) object` array whose entries are the sparse array itself, so any downstream `.shape` check passes and the arithmetic is garbage. `np.asarray(T)` gives a 0-d object array, not the dense matrix; `toarray()` is the conversion. Adding a scalar raises because it would destroy sparsity. On a `csr_array`, `todense()` and `toarray()` both return `ndarray` in 1.17.1; only on the legacy `*_matrix` does `todense()` return `np.matrix`, and only there does `sum(axis=)` return a `(1, n)` matrix instead of an `(n,)` ndarray.

`.eliminate_zeros()` drops explicitly stored zeros (assigning `0.0` stores a structural nonzero: measured `nnz` went `2 -> 3 -> 2` across assignment and cleanup). `.sum_duplicates()` merges repeated `(i, j)` entries in place. `.diagonal()` returns a dense 1-D ndarray. `save_npz(file, matrix)` / `load_npz(file)` round-trip a `csr_array` back to `csr_array`. `scipy.sparse.linalg.norm(x, ord=None, axis=None)` is the sparse-aware norm; `np.linalg.norm` is not.

## Direct solves and fill-in

`spsolve(A, b, permc_spec=None, use_umfpack=True)` factors and solves once. `use_umfpack=True` only matters if scikit-umfpack is installed; it is **not** present here, so SuperLU handles everything and the flag is inert. Dense `b` gives a dense answer; a sparse `b` with several columns gives a sparse answer, but a sparse single-column `b` comes back as a dense 1-D `ndarray`. `splu(A, permc_spec=None, diag_pivot_thresh=None, relax=None, panel_size=None, options=None)` returns an object with `.solve`, `.L`, `.U`, `.perm_r`, `.perm_c`; `factorized(A)` wraps the same thing as a plain callable.

```python
n = 2000; h = 1/(n+1); A = poisson1d(n, 'csc'); x = np.linspace(h, 1-h, n)
b = -np.pi**2*np.sin(np.pi*x); u = spla.spsolve(A, b)                 # u'' = f, exact u = sin(pi x)
print("spsolve ->", type(u).__name__, u.shape, "resid %.1e" % np.linalg.norm(A@u-b), "max err vs exact %.1e" % np.max(np.abs(u-np.sin(np.pi*x))))
B = np.random.default_rng(0).standard_normal((n, 100))
t0 = time.perf_counter(); [spla.spsolve(A, B[:, j]) for j in range(100)]; t1 = time.perf_counter()
lu = spla.splu(A); t2 = time.perf_counter(); [lu.solve(B[:, j]) for j in range(100)]; t3 = time.perf_counter()
print("100 spsolve %.3f s | splu factor %.4f s + 100 lu.solve %.4f s -> %.0fx" % (t1-t0, t2-t1, t3-t2, (t1-t0)/(t3-t1)))
A2 = sp.csc_array(sp.kron(I, T1) + sp.kron(T1, I)); lu2 = spla.splu(A2)          # 2-D, m = 100
print("A2", A2.shape, "nnz", A2.nnz, "-> L nnz", lu2.L.nnz, "U nnz", lu2.U.nnz, "= %.1fx fill-in" % ((lu2.L.nnz+lu2.U.nnz)/A2.nnz))
spla.splu(sp.csr_array(A2))
```
```
spsolve -> ndarray (2000,) resid 1.6e-08 max err vs exact 2.1e-07
100 spsolve 0.085 s | splu factor 0.0007 s + 100 lu.solve 0.0054 s -> 14x
A2 (10000, 10000) nnz 49600 -> L nnz 322875 U nnz 322875 = 13.0x fill-in
SparseEfficiencyWarning: splu converted its input to CSC format
```
That 13x is the whole argument for iterative methods. The 2-D Poisson matrix has 5 nonzeros per row; its exact LU factors have about 65, and the ratio grows as the grid refines until the factors no longer fit in memory. The 1-D case never fills in (a tridiagonal LU is bidiagonal), which is why `splu` there is a pure win. `spsolve_triangular(A, b, lower=True, overwrite_A=False, overwrite_b=False, unit_diagonal=False)` handles an already-triangular sparse system.

## Krylov solvers

All take `(A, b, x0=None, *, rtol=1e-05, atol=0.0, maxiter=None, M=None, callback=None)` plus solver-specific extras, and return `(x, info)`. `A` may be a sparse array, a dense array, or a `LinearOperator`.

| Solver | Requires | Notes |
|---|---|---|
| `cg` | symmetric positive definite | cheapest per iteration, three vectors of storage |
| `minres` | symmetric, indefinite allowed | extra `shift=0.0`; no `atol` |
| `bicgstab`, `cgs`, `qmr`, `tfqmr` | general square | short recurrences, can break down; `qmr` takes `M1`/`M2`, not `M` |
| `gmres`, `lgmres`, `gcrotmk` | general square | `gmres` adds `restart`, `callback_type` in `'x'`/`'pr_norm'`; the other two recycle a subspace across restarts and default `maxiter=1000`, not `None` |

Convergence is `||b - A x|| <= max(rtol*||b||, atol)`, with `atol=0.0` by default, so the test is purely relative. `rtol` defaults to `1e-5`, which is loose: it bounds the residual, and the solution error it permits is `rtol` amplified by `cond(A)`.

```python
A2c = sp.csr_array(-A2); rhs = rng.standard_normal(10000)             # SPD: minus the 2-D Laplacian
xstar = spla.spsolve(sp.csc_array(A2c), rhs); hist = []
u, info = spla.cg(A2c, rhs, rtol=1e-8, callback=lambda xk: hist.append(np.linalg.norm(rhs - A2c@xk)))
print("info", info, "iters", len(hist), "||b|| %.3f" % np.linalg.norm(rhs))
print("residuals[:6]", np.array(hist[:6]).round(3), "... final %.2e" % hist[-1])
for rt in (1e-5, 1e-12):
    c = [0]; uu, i2 = spla.cg(A2c, rhs, rtol=rt, callback=lambda xk: c.__setitem__(0, c[0]+1))
    print("rtol=%-6.0e info=%d iters=%3d ||r||=%.2e rel solution err=%.2e" % (rt, i2, c[0],
          np.linalg.norm(rhs - A2c@uu), np.max(np.abs(uu-xstar))/np.max(np.abs(xstar))))
u3, i3 = spla.cg(A2c, rhs, rtol=1e-12, maxiter=50)
print("maxiter=50 -> info", i3, "||r||=%.2e" % np.linalg.norm(rhs - A2c@u3), "rel err=%.2f" %
      (np.max(np.abs(u3-xstar))/np.max(np.abs(xstar))), "| returned", type(u3).__name__, "no exception")
spla.cg(A2c, rhs, tol=1e-8); spla.cg(A2c, rhs, 1e-8)
```
```
info 0 iters 304 ||b|| 99.810
residuals[:6] [49.148 36.078 30.235 26.365 23.619 21.452] ... final 9.47e-07
rtol=1e-05  info=0 iters=230 ||r||=9.72e-04 rel solution err=1.82e-05
rtol=1e-12  info=0 iters=382 ||r||=9.70e-11 rel solution err=9.40e-13
maxiter=50 -> info 50 ||r||=5.14e+00 rel err=0.34 | returned ndarray no exception
tol=1e-8   -> TypeError: cg() got an unexpected keyword argument 'tol'
positional -> ValueError: shapes of A (10000 (third positional is x0)
```
The callback receives the current iterate `xk`; with `gmres` and `callback_type='pr_norm'` it receives a scalar preconditioned residual norm instead. The `info` convention: `0` converged, `> 0` hit the iteration limit and the value **is** `maxiter`, `< 0` breakdown. A nonzero `info` returns the last iterate and raises nothing, so `u, _ = cg(...)` at `maxiter=50` hands back a vector that is 34% wrong and looks like a solution. Always branch on `info`.

- `tol` was removed from the Krylov solvers in SciPy 1.14; the keyword is `rtol` and it is keyword-only. Passing `tol=` raises `TypeError`. Passing a tolerance positionally is worse: the third positional parameter is `x0`, so the scalar is read as an initial guess and the shape error names `x0`, not the tolerance.

## Preconditioning

Solve `M^-1 A x = M^-1 b` instead, with `M` cheap to invert and `M^-1 A` better conditioned than `A`. Pass `M` as anything supporting `@`: a sparse array, or a `LinearOperator` wrapping a factorization. Jacobi preconditioning is `M = D`, the same `D^-1` splitting as the Jacobi iteration, used to accelerate a Krylov method rather than as a standalone fixed-point scheme. `count` below runs `cg` at `rtol=1e-8` and reports `info`, iterations, wall time, residual.

```python
X, Y = np.meshgrid(np.linspace(hh, 1-hh, m), np.linspace(hh, 1-hh, m), indexing='ij')   # the m x m interior grid
def count(A, M=None, tag="", maxiter=None):                 # cg at rtol=1e-8 on rhs, report the cost
    c = [0]; t0 = time.perf_counter()
    u, info = spla.cg(A, rhs, rtol=1e-8, M=M, maxiter=maxiter, callback=lambda xk: c.__setitem__(0, c[0]+1))
    print(f"{tag:22s} info={info} iters={c[0]:5d} {time.perf_counter()-t0:.3f} s ||r||={np.linalg.norm(rhs - A@u):.1e}")
q  = 1e6*np.exp(-10*((X-0.3)**2 + (Y-0.7)**2)).ravel()      # variable reaction coefficient on the grid
Aq = sp.csr_array(A2c + sp.diags_array(q, format='csr'))    # -u'' + q u = f
Dinv = lambda M: sp.diags_array(1/M.diagonal(), format='csr')
print("A2c.diagonal() unique:", np.unique(A2c.diagonal()), "= 4/h^2")
count(A2c, tag="A, none");         count(A2c, M=Dinv(A2c), tag="A, Jacobi D^-1")
count(Aq,  tag="A+diag(q), none"); count(Aq,  M=Dinv(Aq),  tag="A+diag(q), Jacobi")
t0 = time.perf_counter(); ilu = spla.spilu(sp.csc_array(A2c), drop_tol=1e-2, fill_factor=10); tf = time.perf_counter() - t0
print("spilu drop_tol=1e-2: %.3f s, L+U nnz %d = %.1fx A" % (tf, ilu.L.nnz+ilu.U.nnz, (ilu.L.nnz+ilu.U.nnz)/A2c.nnz))
count(A2c, M=spla.LinearOperator(A2c.shape, matvec=ilu.solve), tag="A, ILU")
ilu3 = spla.spilu(sp.csc_array(A2c), drop_tol=1e-3, fill_factor=10)
count(A2c, M=spla.LinearOperator(A2c.shape, matvec=ilu3.solve), tag="A, ILU drop_tol=1e-3", maxiter=3000)
```
```
A2c.diagonal() unique: [40804.] = 4/h^2
A, none                info=0 iters= 304 0.021 s ||r||=9.5e-07
A, Jacobi D^-1         info=0 iters= 304 0.027 s ||r||=9.5e-07
A+diag(q), none        info=0 iters= 223 0.016 s ||r||=9.7e-07
A+diag(q), Jacobi      info=0 iters=  61 0.006 s ||r||=7.4e-07
spilu drop_tol=1e-2: 0.031 s, L+U nnz 224858 = 4.5x A
A, ILU                 info=0 iters=  46 0.033 s ||r||=6.9e-07
A, ILU drop_tol=1e-3   info=3000 iters=3000 2.359 s ||r||=1.7e+01
```
Jacobi does nothing on the constant-coefficient Poisson matrix, and that is not a bug: every diagonal entry is `4/h^2`, so `D^-1 A` is `A` rescaled and the ratio of extreme eigenvalues is untouched. Add a variable coefficient and the same preconditioner cuts 223 iterations to 61. ILU is the general-purpose choice: `spilu(A, drop_tol=None, fill_factor=None, drop_rule=None, permc_spec=None, ...)` computes an approximate LU, dropping entries below `drop_tol` and capping fill at `fill_factor` times `nnz(A)`, and `.solve` is the `matvec` of `M^-1`. 304 iterations become 46 for 4.5x the storage. The last row is the trap: an ILU preconditioner is not symmetric, CG assumes a symmetric positive definite `M`, and at `drop_tol=1e-3` CG stalls at `maxiter` with `info=3000`. The same `ilu3` preconditioner with `bicgstab` converged in 48 iterations and with `gmres` in 127, so pair ILU with `bicgstab` or `gmres`.

## `LinearOperator` and matrix-free solves

`LinearOperator(shape, matvec, rmatvec=None, matmat=None, dtype=None, rmatmat=None)`. Every Krylov solver, `eigsh`, `eigs`, and `svds` accept one wherever they accept a matrix, so `A` need never be assembled.

```python
n = 2000; h = 1.0/(n+1); b = rng.standard_normal(n); A = -poisson1d(n)   # SPD
def lap(v):
    w = np.concatenate(([0.0], v, [0.0]))                   # Dirichlet ghost nodes, (n+2,)
    return -np.diff(w, 2)/h**2                              # second difference -> (n,)
L = spla.LinearOperator((n, n), matvec=lap, dtype=np.float64)
print("L", L.shape, L.dtype, "| L@b == A@b:", np.allclose(L @ b, A @ b))
u1, i1 = spla.cg(A, b, rtol=1e-10); u2, i2 = spla.cg(L, b, rtol=1e-10)
print("assembled info", i1, "| matrix-free info", i2, "| max|u1-u2| %.1e" % np.max(np.abs(u1-u2)))
L.T @ b
```
```
L (2000, 2000) float64 | L@b == A@b: True
assembled info 0 | matrix-free info 0 | max|u1-u2| 3.2e-17
L.T @ b -> NotImplementedError: rmatvec is not defined
```
Bit-identical iterates from a function and a matrix. Set `dtype` explicitly: without it scipy probes the operator with a test vector. Supplying only `matvec` is enough for `cg`, `gmres`, `bicgstab`; `qmr`, `lsqr`, `lsmr`, and any transpose use raise `NotImplementedError` at call time, not at construction. `aslinearoperator(A)` wraps an existing matrix.

## Sparse eigenvalues and SVD

`eigsh(A, k=6, M=None, sigma=None, which='LM', v0=None, ncv=None, maxiter=None, tol=0, return_eigenvectors=True, Minv=None, OPinv=None, mode='normal', rng=None)` for symmetric or Hermitian input; `eigs` is the same minus `mode` plus `OPpart`, and always returns `complex128`; `svds(A, k=6, ncv=None, tol=0, which='LM', v0=None, maxiter=None, return_singular_vectors=True, solver='arpack', rng=None, options=None, *, random_state=None)`. All are ARPACK: Lanczos or Arnoldi on `span{v, Av, A^2 v, ...}`, so they converge fast to **exterior** eigenvalues and slowly to interior ones. `k < n` is required; `k >= n` raises, telling you to densify.

`which` selects the end: `'LM'`/`'SM'` largest/smallest magnitude, `'LA'`/`'SA'` largest/smallest algebraic (`eigsh` only), `'BE'` half from each end. `'SM'` is the one to avoid, because the smallest eigenvalues are exactly what a Krylov space reaches last. Shift-invert instead: `sigma=s` makes ARPACK work with `(A - sI)^-1`, whose largest-magnitude eigenvalues are the eigenvalues of `A` nearest `s`, and it factors `A - sI` once with SuperLU. Two consequences. `sigma=0` finds the eigenvalues nearest zero, which are the smallest only for a definite matrix: on `diag(-4.5, ..., 4.5)` it returned `[-0.5, 0.5]`. And it fails outright on a singular `A`: a Neumann or graph Laplacian raises `RuntimeError: Factor is exactly singular` at `sigma=0`, while `sigma=-1e-3` returns the zero eigenvalue and its neighbours.

```python
n = 2000; h = 1.0/(n+1); T = poisson1d(n); k = np.arange(1, 4)
exact = -4/h**2*np.sin(k*np.pi*h/2)**2; rel = lambda w: np.max(np.abs(np.sort(w)[::-1] - exact)/np.abs(exact))
t0 = time.perf_counter(); wSM = spla.eigsh(T, k=3, which='SM', return_eigenvectors=False); tSM = time.perf_counter()-t0
t0 = time.perf_counter(); wSI = spla.eigsh(T, k=3, sigma=0.0, which='LM', return_eigenvectors=False); tSI = time.perf_counter()-t0
print("exact -4/h^2 sin^2(k pi h/2):", exact)
print("which='SM'    %7.3f s" % tSM, np.sort(wSM)[::-1], "rel err %.1e" % rel(wSM))
print("sigma=0,'LM'  %7.3f s" % tSI, np.sort(wSI)[::-1], "rel err %.1e" % rel(wSI), "-> %.0fx faster" % (tSM/tSI))
U, s, Vh = spla.svds(T, k=3); print("svds k=3 -> s ascending", s.round(3), U.shape, Vh.shape)
```
```
exact -4/h^2 sin^2(k pi h/2): [ -9.86960237 -39.47838517 -88.8262754 ]
which='SM'      6.628 s [ -9.86960237 -39.47838517 -88.8262754 ] rel err 4.5e-12
sigma=0,'LM'    0.003 s [ -9.86960237 -39.47838517 -88.8262754 ] rel err 1.7e-13 -> 2070x faster
svds k=3 -> s ascending [16015915.174 16015964.522 16015994.13 ] (2000, 3) (3, 2000)
```
2070x for one keyword, and the shift-invert answer is 26x more accurate. Both match the analytic eigenvalues to near machine precision. `eigsh(T, k=2, which='LM')` returns `[-16015994.13039051, -16015964.52160939]`, the extreme end, against the exact `-4/h^2 sin^2(n pi h / 2) = -16015994.1304`. With eigenvectors requested (the default), eigenvalues come back in **ascending algebraic** order with eigenvectors in the columns of `V`; with `return_eigenvectors=False` the order is not guaranteed (`which='LM'` returned the two values above in descending order), so sort before indexing; `svds` returns singular values **ascending**, opposite to dense `svd`. The random starting vector is controlled by `rng=`: `eigsh` and `eigs` reject `random_state` with `TypeError`, `svds` still accepts it as a keyword-only alias. Pass `v0=` for a reproducible start either way.

## Gotchas

- **`*` means two different things.** Symptom: a product with the right shape and wrong values, or an `nnz` that is too small. Cause: `csr_matrix.__mul__` is matmul (legacy `np.matrix` behavior), `csr_array.__mul__` is elementwise. Measured: `T*T` has `nnz` 16 on the array, `Tm*Tm` has 24 on the matrix. Fix: build `*_array` types and write `@`. Audit with `isinstance(x, sp.sparray)`.
- **`tol` was removed from the Krylov solvers in SciPy 1.14; the keyword is `rtol` and it is keyword-only.** Symptom: `TypeError: cg() got an unexpected keyword argument 'tol'`, or a confusing `ValueError` about `x0` shapes when the tolerance was passed positionally. Cause: the third positional parameter is `x0`. Fix: `rtol=`, by keyword. The default `rtol=1e-5` with `atol=0.0` is loose for scientific work: the solution error it allows is `rtol` times `cond(A)`.
- **Unchecked `info`.** Symptom: a plausible-looking vector, a residual of order `||b||`, no traceback. Cause: a nonzero `info` returns the last iterate rather than raising. Measured: `cg` at `maxiter=50` on the 2-D Poisson system returns `info=50` and a solution 34% wrong. Fix: `if info != 0: raise`. `info > 0` is the iteration limit and equals `maxiter`; `info < 0` is breakdown.
- **`which='SM'`.** Symptom: `eigsh`/`eigs` runs for minutes, or returns values that do not match a known spectrum. Cause: ARPACK converges to the exterior of the spectrum; the smallest eigenvalues are the hardest. Measured: 6.628 s against 0.003 s for shift-invert on the same 2000x2000 matrix. Fix: `sigma=<target>, which='LM'`, which factors `A - sigma*I` once and finds the eigenvalues nearest `sigma`.
- **`SparseEfficiencyWarning` from assigning into CSR/CSC.** Symptom: a build loop that crawls and prints the warning once. Cause: inserting a new nonzero rebuilds `indices` and `data`. Fix: accumulate `(row, col, val)` lists, build one `coo_array`, call `.tocsr()`; use `lil_array`/`dok_array` if the pattern genuinely must change incrementally. The same warning fires when `splu`/`spilu` silently converts CSR to CSC, so pass `csc_array` there in the first place.
- **numpy functions on sparse arrays.** Symptom: object-dtype arrays, 0-d arrays, or nonsense that never raises. Measured on 1.17.1: `np.dot(T, np.ones(6))` returns a `(6,) object` ndarray of copies of `T`; `np.asarray(T)` returns a 0-d object array; `np.sum(T)` works only by deferring to `T.sum()`; `T + 1` raises `NotImplementedError`. Fix: use methods and operators (`A @ v`, `A.sum(axis=)`, `A.max()`) and `scipy.sparse.linalg.norm`, never `np.linalg.norm`. Convert with `.toarray()` when you truly need dense.
- **`.toarray()` on something too large.** Symptom: `MemoryError`, or the process is killed by the OS. Cause: a `(100000, 100000)` matrix with 5 nonzeros per row is about 6 MB sparse and 80 GB dense; the 2-D Poisson matrix at `m=100` is already 0.8 GB dense against 49600 stored entries. Fix: check `A.shape[0]*A.shape[1]*A.dtype.itemsize` first, inspect with `A[:k, :k].toarray()`, and use a `LinearOperator` when the operator only ever needs to be applied.

## Quick reference

| Task | Call | Note |
|---|---|---|
| build from triplets | `sp.coo_array((vals, (rows, cols)), shape=).tocsr()`; `sp.lil_array`/`sp.dok_array` then `.tocsr()` for incremental assignment | duplicate `(i, j)` entries summed on conversion; assigning into CSR/CSC raises `SparseEfficiencyWarning` and reallocates |
| structured build | `sp.diags_array(diagonals, /, *, offsets=0, shape=None, format=None)`, `sp.eye_array(m, n=None, *, k=0, dtype=float, format=None)`, `sp.kron(A, B)`, `sp.block_array`, `sp.hstack`, `sp.vstack` | `kron(I, T) + kron(T, I)` is the 5-point 2-D Laplacian; `kron` returns COO, rewrap in `csr_array` |
| inspect, clean, persist | `.nnz`, `.shape`, `.dtype`, `.format`, `.diagonal()`, `.toarray()`; `.eliminate_zeros()`, `.sum_duplicates()`, `.sort_indices()`; `sp.save_npz`/`sp.load_npz` | `.todense()` returns `ndarray` on `*_array`, `np.matrix` on `*_matrix`; assigning `0.0` stores a structural nonzero until eliminated |
| direct solve | `spla.spsolve(A, b, permc_spec=None, use_umfpack=True)` | wants CSC; `use_umfpack` inert without scikit-umfpack (absent here); sparse `b` gives sparse output |
| factor once, solve many | `lu = spla.splu(A)`; `lu.solve(b)`, or `f = spla.factorized(A)` | 14x over repeated `spsolve` on a 2000-point 1-D problem; `.L`, `.U`, `.perm_r`, `.perm_c` expose the factors |
| triangular, incomplete LU | `spla.spsolve_triangular(A, b, lower=True, ...)`, `spla.spilu(A, drop_tol=, fill_factor=)` | exact 2-D Poisson LU is 13x the nnz of `A`; that fill-in is why iterative methods win as the grid refines |
| SPD Krylov | `spla.cg(A, b, x0=None, *, rtol=1e-05, atol=0.0, maxiter=None, M=None, callback=None)` | `-> (x, info)`; stop test `||r|| <= max(rtol*||b||, atol)` |
| general Krylov | `gmres` (adds `restart`, `callback_type`), `bicgstab`, `lgmres`, `gcrotmk`, `qmr` (`M1`/`M2`), `tfqmr`, `cgs`; `minres` for symmetric indefinite | `lgmres`/`gcrotmk` default `maxiter=1000`, the rest `None`; least squares and rectangular systems go to `spla.lsqr(A, b, damp=0.0, atol=1e-06, btol=1e-06, conlim=1e8, ...)` or `spla.lsmr`, which take `atol`/`btol` instead of `rtol` and return a long tuple |
| preconditioner, matrix-free | `M=` a sparse array or a `LinearOperator`; `spla.LinearOperator(shape, matvec, rmatvec=None, matmat=None, dtype=None, rmatmat=None)`, `spla.aslinearoperator(A)` | Jacobi `sp.diags_array(1/A.diagonal())`, ILU `LinearOperator(A.shape, matvec=spilu(A).solve)`, measured 304 -> 46 iterations; set `dtype`, and a missing `rmatvec` raises `NotImplementedError` at call time |
| sparse eigen | `spla.eigsh(A, k=6, M=None, sigma=None, which='LM', ..., rng=None)`, `spla.eigs(...)`; smallest via `sigma=0.0, which='LM'` | `eigsh` ascending algebraic and real, `eigs` always `complex128`; `k < n`; shift-invert is 2070x faster than `which='SM'` |
| sparse SVD, norms | `spla.svds(A, k=6, ..., solver='arpack'/'lobpcg'/'propack', rng=None)`; `spla.norm(x, ord=None, axis=None)`, `spla.onenormest`, `spla.expm`, `spla.expm_multiply` | `svds` singular values come back **ascending**, opposite to dense `svd`; `np.linalg.norm` does not work on sparse input |
