"""Integration conftest: put at tests/integration/conftest.py.

Provides a real PostgreSQL per test session via testcontainers, and a per-test
transaction that rolls back, so tests share the expensive container but never
see each other's rows. Skips cleanly (with a reason) when Docker is unavailable.

Requires: pip install "testcontainers[postgres]" "psycopg[binary]"
Verified here: the skip path. The live-container path needs Docker, which was not
available when this template was written; run it once before relying on it.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from collections.abc import Iterator

import pytest

POSTGRES_IMAGE = "postgres:17-alpine"  # pin by digest in real projects: postgres:17-alpine@sha256:...


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0  # noqa: S607 docker from PATH by design


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    if importlib.util.find_spec("testcontainers") is None:
        pytest.skip("testcontainers not installed: pip install 'testcontainers[postgres]'")
    if not _docker_available():
        pytest.skip("Docker is not available; integration tests need a container runtime")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer(POSTGRES_IMAGE, driver=None) as container:
        yield container.get_connection_url()


@pytest.fixture
def pg(postgres_url: str):
    """A connection inside a transaction that is always rolled back."""
    psycopg = pytest.importorskip("psycopg", reason="pip install 'psycopg[binary]'")
    with psycopg.connect(postgres_url) as conn:
        conn.autocommit = False
        try:
            yield conn
        finally:
            conn.rollback()
