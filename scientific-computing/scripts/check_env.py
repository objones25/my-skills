#!/usr/bin/env python3
"""Report the installed scientific-Python stack before quoting any API.

Run this first. If numpy is not 2.4.x or scipy is not 1.17.x, also run
verify_api.py, which introspects the installed library and reports every
signature in references/ that has drifted.
"""
from __future__ import annotations

import importlib
import os
import platform
import sys


def _ver(name: str) -> str | None:
    try:
        return importlib.import_module(name).__version__
    except Exception:
        return None


def main() -> int:
    print(f"python           {platform.python_version()}  ({sys.executable})")
    print(f"platform         {platform.system()} {platform.machine()}")

    np_v = _ver("numpy")
    sp_v = _ver("scipy")
    print(f"numpy            {np_v or 'NOT INSTALLED'}")
    print(f"scipy            {sp_v or 'NOT INSTALLED'}")

    for pkg in ("pandas", "matplotlib", "numba", "cython", "threadpoolctl",
                "sympy", "sparse", "dask", "torch", "jax"):
        v = _ver(pkg)
        if v:
            print(f"{pkg:<16} {v}")

    if np_v is None:
        print("\nnumpy is required. Install it before using this skill.")
        return 1

    import numpy as np

    print(f"\ndefault int      {np.array([1]).dtype}")
    print(f"float64 eps      {np.finfo(np.float64).eps:.6e}")
    print(f"float32 eps      {np.finfo(np.float32).eps:.6e}")

    # BLAS / LAPACK backend and threading
    blas = "unknown"
    try:
        cfg = np.__config__.CONFIG  # numpy >= 1.25
        blas = cfg["Build Dependencies"]["blas"].get("name", "unknown")
        blas_ver = cfg["Build Dependencies"]["blas"].get("version", "")
        blas = f"{blas} {blas_ver}".strip()
    except Exception:
        pass
    print(f"BLAS             {blas}")

    try:
        import threadpoolctl
        for p in threadpoolctl.threadpool_info():
            print(f"  thread pool    {p['internal_api']} "
                  f"{p.get('version', '')} num_threads={p['num_threads']}")
    except ImportError:
        print("  thread pool    threadpoolctl not installed "
              "(pip install threadpoolctl to inspect/control BLAS threads)")

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if var in os.environ:
            print(f"  {var}={os.environ[var]}")
    print(f"cpu count        {os.cpu_count()}")

    # Version drift warnings
    print()
    if np_v and not np_v.startswith("2."):
        print(f"WARNING: references target numpy 2.4.x, found {np_v}. "
              "Removed aliases (np.float_, np.NaN, np.trapz, np.in1d) may still "
              "exist and NEP 50 promotion rules differ. Run verify_api.py.")
    elif np_v and not np_v.startswith("2.4"):
        print(f"note: references target numpy 2.4.x, found {np_v}. "
              "Run verify_api.py to check for signature drift.")

    if sp_v and not sp_v.startswith("1.17"):
        major_minor = tuple(int(p) for p in sp_v.split(".")[:2])
        print(f"note: references target scipy 1.17.x, found {sp_v}. "
              "Run verify_api.py to check for signature drift.")
        if major_minor < (1, 15):
            print("  scipy < 1.15: scipy.differentiate, "
                  "scipy.optimize.elementwise, integrate.cubature, "
                  "integrate.tanhsinh, integrate.nsum, interpolate.AAA, "
                  "and stats.make_distribution are absent.")
        if major_minor < (1, 16):
            print("  scipy < 1.16: scipy.linalg functions do not accept stacked "
                  "(k, n, n) input.")
        if major_minor < (1, 17):
            print("  scipy < 1.17: sparse.linalg.eigsh has no `rng=` keyword.")
        if major_minor < (1, 12):
            print("  scipy < 1.12: sparse.linalg Krylov solvers take `tol` and "
                  "have no `rtol`.")
        elif major_minor < (1, 14):
            print("  scipy 1.12-1.13: Krylov solvers accept both `rtol` and the "
                  "deprecated `tol`.")
        if major_minor >= (1, 18):
            print("  scipy >= 1.18: see 'Known drift' in "
                  "references/00-module-map.md.")

    if np_v and np_v.startswith("2.4") and sp_v and sp_v.startswith("1.17"):
        print("Versions match the references. No drift check needed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
