"""The suite's own safety net (tests/conftest.py): each autouse fixture must really act.

Disabling any of them would leave every other wallet test green, so each gets a test
that fails when it is switched off. No test here sends a packet: blocked calls are made
on closed sockets or with numeric hosts, so a disabled guard fails with an OSError
instead of reaching a network.
"""

import errno
import os
import random
import socket
import subprocess
import sys
from pathlib import Path

import pytest

WALLET_ROOT = Path(__file__).resolve().parents[2]
NON_LOOPBACK = "192.0.2.1"  # TEST-NET-1 (RFC 5737): reserved for documentation


@pytest.fixture
def closed_tcp_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.close()
    return sock


@pytest.fixture
def closed_udp_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.close()
    return sock


def test_network_guard_blocks_connect(closed_tcp_socket):
    with pytest.raises(RuntimeError, match=r"Test tried to reach \('192\.0\.2\.1', 80\)"):
        closed_tcp_socket.connect((NON_LOOPBACK, 80))


def test_network_guard_blocks_connect_ex(closed_tcp_socket):
    with pytest.raises(RuntimeError, match=r"Test tried to reach \('192\.0\.2\.1', 80\)"):
        closed_tcp_socket.connect_ex((NON_LOOPBACK, 80))


def test_network_guard_blocks_udp_sendto(closed_udp_socket):
    with pytest.raises(RuntimeError, match=r"Test tried to reach \('192\.0\.2\.1', 9\)"):
        closed_udp_socket.sendto(b"x", (NON_LOOPBACK, 9))


def test_network_guard_blocks_udp_sendto_with_flags(closed_udp_socket):
    with pytest.raises(RuntimeError, match=r"Test tried to reach \('192\.0\.2\.1', 9\)"):
        closed_udp_socket.sendto(b"x", 0, (NON_LOOPBACK, 9))


def test_network_guard_blocks_name_lookup_of_a_remote_host():
    with pytest.raises(RuntimeError, match="Test tried to reach '192.0.2.1'"):
        socket.getaddrinfo(NON_LOOPBACK, 80, flags=socket.AI_NUMERICHOST)  # audit: ignore[NETWORK] the guard under test blocks this numeric-host lookup


def test_network_guard_lets_loopback_connections_through():
    with socket.socket() as server:  # audit: ignore[NETWORK] loopback only: the guard allows it
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.socket() as client:  # audit: ignore[NETWORK] loopback only: the guard allows it
            assert client.connect_ex(server.getsockname()) == 0


def test_network_guard_lets_unix_sockets_through():
    # Relative path: the cwd is this test's own directory, and absolute temp paths can
    # exceed the AF_UNIX length limit on macOS.
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as unix:  # audit: ignore[NETWORK] Unix socket: local, the guard allows it
        assert unix.connect_ex("absent.sock") == errno.ENOENT


@pytest.fixture
def drawn_during_setup() -> float:
    return random.random()


def test_global_random_is_reseeded_before_fixtures_run(drawn_during_setup, rng):
    # Checked during fixture setup: pytest-randomly, when installed, reseeds again just
    # before the test body, so only setup time shows the conftest fixture's own seed.
    assert drawn_during_setup == rng.random()


def test_each_test_runs_in_its_own_empty_directory(tmp_path):
    assert Path.cwd().resolve() == tmp_path.resolve()
    assert list(Path.cwd().iterdir()) == []


@pytest.fixture(scope="module", autouse=True)
def _environment_a_leaking_test_would_have_left_behind():
    # Module scope: set before the function-scoped _clean_env fixture runs for each test.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("WALLET_API_KEY", "leaked")
        patch.setenv("UNRELATED_SETTING", "kept")
        yield


def test_app_environment_variables_are_removed_and_others_kept():
    assert "WALLET_API_KEY" not in os.environ
    assert os.environ["UNRELATED_SETTING"] == "kept"


def test_directory_names_mark_integration_tests():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "--no-cov", "-p", "no:cacheprovider",
         "-p", "no:randomly", "-m", "integration", str(WALLET_ROOT / "tests" / "integration")],
        cwd=WALLET_ROOT, capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 tests collected" in result.stdout
