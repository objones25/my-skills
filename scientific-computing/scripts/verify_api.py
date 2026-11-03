#!/usr/bin/env python3
"""Check the signatures and behaviors this skill's references assert.

Every entry below was introspected from numpy 2.4.4 / scipy 1.17.1. Run this
when check_env.py reports different versions. Anything printed as DRIFT or
MISSING means the references are wrong for the installed library and the
script's output wins. Exit status is 0 when nothing drifted, 1 otherwise.

    python scripts/verify_api.py           # summary
    python scripts/verify_api.py -v        # also list every OK entry
"""
from __future__ import annotations

import importlib
import inspect
import re
import sys

_SENTINEL = re.compile(r"<object object at 0x[0-9a-fA-F]+>")


def _norm(sig: str) -> str:
    """Replace private sentinel defaults, whose repr embeds a memory address."""
    return _SENTINEL.sub("<sentinel>", sig)

# (dotted path, expected signature string as of numpy 2.4.4 / scipy 1.17.1)
SIGNATURES: list[tuple[str, str]] = [
    # linear algebra
    ("scipy.linalg.solve",
     "(a, b, lower=False, overwrite_a=False, overwrite_b=False, "
     "check_finite=True, assume_a=None, transposed=False)"),
    ("scipy.linalg.lstsq",
     "(a, b, cond=None, overwrite_a=False, overwrite_b=False, "
     "check_finite=True, lapack_driver=None)"),
    ("scipy.linalg.qr",
     "(a, overwrite_a=False, lwork=None, mode='full', pivoting=False, "
     "check_finite=True)"),
    ("scipy.linalg.svd",
     "(a, full_matrices=True, compute_uv=True, overwrite_a=False, "
     "check_finite=True, lapack_driver='gesdd')"),
    ("scipy.linalg.cholesky",
     "(a, lower=False, overwrite_a=False, check_finite=True)"),
    ("scipy.linalg.solve_banded",
     "(l_and_u, ab, b, overwrite_ab=False, overwrite_b=False, "
     "check_finite=True)"),
    ("numpy.linalg.lstsq", "(a, b, rcond=None)"),
    ("numpy.linalg.qr", "(a, mode='reduced')"),
    ("numpy.sort", "(a, axis=-1, kind=None, order=None, *, stable=None)"),
    ("scipy.optimize.nnls", "(A, b, *, maxiter=None, atol=<sentinel>)"),

    # sparse
    ("scipy.sparse.linalg.cg",
     "(A, b, x0=None, *, rtol=1e-05, atol=0.0, maxiter=None, M=None, "
     "callback=None)"),
    ("scipy.sparse.linalg.gmres",
     "(A, b, x0=None, *, rtol=1e-05, atol=0.0, restart=None, maxiter=None, "
     "M=None, callback=None, callback_type=None)"),
    ("scipy.sparse.linalg.spsolve",
     "(A, b, permc_spec=None, use_umfpack=True)"),
    ("scipy.sparse.linalg.eigsh",
     "(A, k=6, M=None, sigma=None, which='LM', v0=None, ncv=None, "
     "maxiter=None, tol=0, return_eigenvectors=True, Minv=None, OPinv=None, "
     "mode='normal', rng=None)"),
    ("scipy.sparse.diags_array", None),   # sentinel default in signature
    ("scipy.sparse.eye_array",
     "(m, n=None, *, k=0, dtype=<class 'float'>, format=None)"),

    # optimize
    ("scipy.optimize.minimize",
     "(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, "
     "bounds=None, constraints=(), tol=None, callback=None, options=None)"),
    ("scipy.optimize.root_scalar",
     "(f, args=(), method=None, bracket=None, fprime=None, fprime2=None, "
     "x0=None, x1=None, xtol=None, rtol=None, maxiter=None, options=None)"),
    ("scipy.optimize.root",
     "(fun, x0, args=(), method='hybr', jac=None, tol=None, callback=None, "
     "options=None)"),
    ("scipy.optimize.curve_fit",
     "(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, "
     "check_finite=None, bounds=(-inf, inf), method=None, jac=None, *, "
     "full_output=False, nan_policy=None, **kwargs)"),
    ("scipy.optimize.elementwise.find_root",
     "(f, init, /, *, args=(), tolerances=None, maxiter=None, callback=None)"),

    # integrate
    ("scipy.integrate.solve_ivp",
     "(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False, "
     "events=None, vectorized=False, args=None, **options)"),
    ("scipy.integrate.solve_bvp",
     "(fun, bc, x, y, p=None, S=None, fun_jac=None, bc_jac=None, tol=0.001, "
     "max_nodes=1000, verbose=0, bc_tol=None)"),
    ("scipy.integrate.quad",
     "(func, a, b, args=(), full_output=0, epsabs=1.49e-08, epsrel=1.49e-08, "
     "limit=50, points=None, weight=None, wvar=None, wopts=None, maxp1=50, "
     "limlst=50, complex_func=False)"),
    ("scipy.integrate.simpson", "(y, x=None, *, dx=1.0, axis=-1)"),
    ("scipy.integrate.fixed_quad", "(func, a, b, args=(), n=5)"),
    ("scipy.integrate.tanhsinh",
     "(f, a, b, *, args=(), log=False, maxlevel=None, minlevel=2, atol=None, "
     "rtol=None, preserve_shape=False, callback=None)"),

    # interpolate
    ("scipy.interpolate.CubicSpline",
     "(x, y, axis=0, bc_type='not-a-knot', extrapolate=None)"),
    ("scipy.interpolate.make_interp_spline",
     "(x, y, k=3, t=None, bc_type=None, axis=0, check_finite=True)"),
    ("scipy.interpolate.RegularGridInterpolator",
     "(points, values, method='linear', bounds_error=True, fill_value=nan, *, "
     "solver=None, solver_args=None)"),
    ("scipy.interpolate.RBFInterpolator",
     "(y, d, neighbors=None, smoothing=0.0, kernel='thin_plate_spline', "
     "epsilon=None, degree=None)"),
    ("scipy.interpolate.make_smoothing_spline",
     "(x, y, w=None, lam=None, *, axis=0)"),

    # differentiate
    ("scipy.differentiate.derivative",
     "(f, x, *, args=(), tolerances=None, maxiter=10, order=8, "
     "initial_step=0.5, step_factor=2.0, step_direction=0, "
     "preserve_shape=False, callback=None)"),

    # fft / signal
    ("scipy.fft.rfftfreq", "(n, d=1.0, *, xp=None, device=None)"),
    ("scipy.signal.butter",
     "(N, Wn, btype='low', analog=False, output='ba', fs=None)"),
    ("scipy.signal.savgol_filter",
     "(x, window_length, polyorder, deriv=0, delta=1.0, axis=-1, "
     "mode='interp', cval=0.0)"),

    # stats
    ("scipy.stats.bootstrap",
     "(data, statistic, *, n_resamples=9999, batch=None, vectorized=None, "
     "paired=False, axis=0, confidence_level=0.95, alternative='two-sided', "
     "method='BCa', bootstrap_result=None, rng=None, random_state=None)"),
    ("scipy.stats.linregress",
     "(x, y, alternative='two-sided', *, axis=0, nan_policy='propagate', "
     "keepdims=False)"),
]

# Names the references assert are GONE from numpy 2.x.
NUMPY_REMOVED = [
    "float_", "int0", "NaN", "Inf", "NAN", "infty", "alltrue", "sometrue",
    "cumproduct", "product", "round_", "in1d", "trapz", "msort", "mat",
    "byte_bounds", "issubsctype", "set_string_function", "safe_eval",
    "deprecate", "source", "who", "disp",
]

# Names the references assert EXIST.
MUST_EXIST = [
    "numpy.trapezoid", "numpy.isin", "numpy.vecdot", "numpy.matvec",
    "numpy.vecmat", "numpy.linalg.matrix_norm", "numpy.linalg.vector_norm",
    "numpy.linalg.svdvals",
    "scipy.differentiate.jacobian", "scipy.differentiate.hessian",
    "scipy.optimize.elementwise.bracket_root",
    "scipy.optimize.elementwise.find_minimum",
    "scipy.integrate.cubature", "scipy.integrate.nsum",
    "scipy.integrate.cumulative_simpson", "scipy.integrate.qmc_quad",
    "scipy.interpolate.AAA", "scipy.interpolate.make_splrep",
    "scipy.interpolate.FloaterHormannInterpolator",
    "scipy.sparse.csr_array", "scipy.sparse.random_array",
    "scipy.stats.make_distribution", "scipy.stats.Normal",
    "scipy.stats.permutation_test", "scipy.stats.false_discovery_control",
    "scipy.signal.ShortTimeFFT", "scipy.signal.oaconvolve",
]


def resolve(path: str):
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        try:
            mod = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        obj = mod
        for p in parts[i:]:
            obj = getattr(obj, p)
        return obj
    raise ImportError(path)


def main(verbose: bool = False) -> int:
    problems = 0
    ok = 0

    print("=== signatures ===")
    for path, expected in SIGNATURES:
        try:
            obj = resolve(path)
        except (ImportError, AttributeError):
            print(f"MISSING  {path}")
            problems += 1
            continue
        try:
            actual = _norm(str(inspect.signature(obj)))
        except (ValueError, TypeError):
            if verbose:
                print(f"SKIP     {path}  (no introspectable signature)")
            continue
        if expected is None:
            if verbose:
                print(f"INFO     {path}{actual}")
            ok += 1
        elif actual == expected:
            ok += 1
            if verbose:
                print(f"OK       {path}{actual}")
        else:
            print(f"DRIFT    {path}")
            print(f"  expected {expected}")
            print(f"  actual   {actual}")
            problems += 1

    print("\n=== numpy 2.x removals ===")
    try:
        import numpy as np
        still_there = [n for n in NUMPY_REMOVED if hasattr(np, n)]
        if still_there:
            print(f"STILL PRESENT: {', '.join('np.' + n for n in still_there)}")
            if set(still_there) <= {"in1d", "trapz"}:
                print("  numpy 2.0-2.3: these are deprecated, not yet removed "
                      "(removal came in 2.4).")
            else:
                print("  This is numpy 1.x; the references assume numpy 2.x.")
            problems += 1
        else:
            print(f"all {len(NUMPY_REMOVED)} removed aliases absent, as documented")
            ok += 1
    except ImportError:
        print("numpy not importable")
        return 1

    print("\n=== required APIs ===")
    missing = []
    for path in MUST_EXIST:
        try:
            resolve(path)
        except (ImportError, AttributeError):
            missing.append(path)
    if missing:
        for m in missing:
            print(f"MISSING  {m}")
        problems += len(missing)
    else:
        print(f"all {len(MUST_EXIST)} present, as documented")
        ok += 1

    print("\n=== behaviors ===")
    checks = []

    import numpy as np
    checks.append((
        "NEP 50: np.float32(1) + 1.0 stays float32",
        (np.float32(1) + 1.0).dtype == np.float32))
    checks.append((
        "np.array(x, copy=False) raises when a copy is required",
        _raises(lambda: np.array(np.arange(3.0), dtype=np.int64, copy=False),
                ValueError)))

    try:
        from scipy import sparse
        A = sparse.csr_array(np.array([[1.0, 2.0], [3.0, 4.0]]))
        checks.append((
            "csr_array `*` is elementwise, not matmul",
            np.allclose((A * A).toarray(), np.array([[1.0, 4.0], [9.0, 16.0]]))))
        import scipy.sparse.linalg as spla
        checks.append((
            "Krylov solvers reject `tol` (renamed to rtol)",
            _raises(lambda: spla.cg(sparse.eye_array(3).tocsr(),
                                    np.ones(3), tol=1e-8), TypeError)))
    except ImportError:
        pass

    try:
        from scipy import linalg
        L = np.linalg.cholesky(np.eye(2) * 4.0)
        U = linalg.cholesky(np.eye(2) * 4.0)
        checks.append((
            "np.linalg.cholesky lower, scipy.linalg.cholesky upper",
            np.allclose(L, np.tril(L)) and np.allclose(U, np.triu(U))))
        stack = np.eye(3) * 4.0 + np.zeros((2, 3, 3))
        batched = _call(lambda: linalg.svd(stack))
        checks.append((
            "scipy.linalg accepts stacked (k, n, n) input",
            batched is not _FAILED))
        checks.append((
            "batched scipy.linalg.svd returns a list (2-D input: a tuple)",
            isinstance(batched, list)))
        S = np.array([[4.0, 2.0], [2.0, 3.0]])
        c, low = linalg.cho_factor(S)
        checks.append((
            "cho_factor returns a bool `lower` and leaves the unused triangle "
            "unzeroed",
            isinstance(low, bool) and c[1, 0] == S[1, 0]))
        A = np.vander(np.linspace(0.0, 1.0, 5), 2)
        res = linalg.lstsq(A, np.arange(5.0) ** 2)[1]
        checks.append((
            "scipy.linalg.lstsq residues is a np.float64 scalar (overdetermined)",
            type(res) is np.float64))
    except ImportError:
        pass

    try:
        from scipy import stats
        checks.append((
            "legacy stats.norm.rvs rejects rng= (takes random_state=)",
            _raises(lambda: stats.norm.rvs(size=2,
                                           rng=np.random.default_rng(0)),
                    TypeError)))
    except ImportError:
        pass

    for name, passed in checks:
        print(f"{'OK      ' if passed else 'CHANGED '} {name}")
        if passed:
            ok += 1
        else:
            problems += 1

    print(f"\n{ok} checks passed, {problems} problems.")
    if problems:
        print("Where this script and the references disagree, quote this "
              "script's output.")
    return 1 if problems else 0


_FAILED = object()


def _call(fn):
    try:
        return fn()
    except Exception:
        return _FAILED


def _raises(fn, exc) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


if __name__ == "__main__":
    raise SystemExit(main(verbose="-v" in sys.argv))
