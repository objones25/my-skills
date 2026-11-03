"""ML/PyTorch addendum to conftest.py.

Merge these into tests/conftest.py (or keep as tests/ml/conftest.py so the
torch import cost is only paid by tests that need it).

Determinism settings follow PyTorch's own Reproducibility note:
https://docs.pytorch.org/docs/stable/notes/randomness.html

Verified against PyTorch 2.13.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable, Iterator

import pytest

SEED = 20260826


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _torch_determinism() -> Iterator[None]:
    """Make PyTorch ops reproducible for the duration of each test.

    Four separate sources of nondeterminism, each needing its own switch:

      torch.manual_seed          seeds CPU and all CUDA devices
      cudnn.benchmark = False    stops cuDNN re-picking a conv algorithm per run
      cudnn.deterministic = True forces the deterministic conv algorithm
      use_deterministic_algorithms(True)
                                 raises on any op with no deterministic
                                 implementation, so nondeterminism becomes a
                                 loud error rather than a slow drift

    warn_only=True is the pragmatic default: ops without a deterministic
    kernel warn instead of erroring. Flip to warn_only=False when chasing an
    actual reproducibility bug -- note that under filterwarnings=error a warn
    will itself fail the test, which is usually what you want.

    Determinism is slower. That is the trade you are making on purpose.
    """
    torch = pytest.importorskip("torch")

    # CUBLAS needs this set before the first CUDA context to be deterministic.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(SEED)
    torch.manual_seed(SEED)
    try:
        import numpy as np

        np.random.seed(SEED)
    except ImportError:
        pass

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(False)


# --------------------------------------------------------------------------
# Devices
# --------------------------------------------------------------------------
@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def device(request: pytest.FixtureRequest):
    """Run the test on CPU, and again on CUDA when the gpu marker is selected.

    A parametrized fixture: every test that requests `device` runs once per
    param. The cuda case carries the gpu marker, so the default
    `-m "not gpu"` selection skips it on a laptop and CI picks it up on a
    GPU runner with `-m gpu`.
    """
    torch = pytest.importorskip("torch")
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available on this machine")
    return torch.device(request.param)


@pytest.fixture
def seeded_generator():
    """A torch.Generator with fixed state, for reproducible DataLoader shuffling.

    Pass to DataLoader(generator=...) alongside worker_init_fn so multi-worker
    loading is reproducible too -- workers otherwise get their own RNG state.
    """
    torch = pytest.importorskip("torch")
    g = torch.Generator()
    g.manual_seed(SEED)
    return g


@pytest.fixture
def seed_worker() -> Callable[[int], None]:
    """worker_init_fn for DataLoader. Straight from the PyTorch docs.

        DataLoader(ds, num_workers=2, worker_init_fn=seed_worker, generator=g)
    """
    torch = pytest.importorskip("torch")

    def _seed_worker(_worker_id: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        try:
            import numpy as np

            np.random.seed(worker_seed)
        except ImportError:
            pass

    return _seed_worker


# --------------------------------------------------------------------------
# Tiny fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def tiny_batch(device):
    """A batch small enough that a test costs milliseconds.

    Shape (4, 3, 8, 8): batch 4, 3 channels, 8x8 spatial. Big enough to catch
    a broadcasting bug, small enough that a full forward+backward is instant.
    Never load a real dataset in a unit test -- that is an integration test.
    """
    torch = pytest.importorskip("torch")
    x = torch.randn(4, 3, 8, 8, device=device)
    y = torch.randint(0, 2, (4,), device=device)
    return x, y


@pytest.fixture
def tiny_model(device):
    """A 2-layer model with a handful of parameters.

    Use this to test the *plumbing* -- shapes, dtypes, gradient flow, the
    training step -- not model quality. Testing a real architecture's accuracy
    is a benchmark, not a unit test, and belongs behind @pytest.mark.slow.
    """
    pytest.importorskip("torch")
    from torch import nn

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 8 * 8, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    ).to(device)
    return model


@pytest.fixture
def assert_close():
    """torch.testing.assert_close with the tolerances stated explicitly.

    Never compare float tensors with ==. assert_close checks shape, dtype and
    device as well as values, and its failure message shows the largest
    absolute and relative deviation and where it occurred.

    rtol/atol defaults here are the float32 defaults; tighten for float64,
    loosen for float16/bfloat16 (roughly rtol=1e-2 for half precision).
    """
    torch = pytest.importorskip("torch")

    def _assert_close(actual, expected, *, rtol: float = 1.3e-6, atol: float = 1e-5) -> None:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    return _assert_close
