"""hookspec.py is a dated copy of the hooks reference, so check it against the live page."""
from __future__ import annotations

import pytest

import hookspec as hs


def test_every_event_in_the_table_is_in_the_live_reference(project, run_script):
    missing = []
    for event in hs.EVENTS:
        result = run_script("fetch_reference.py", "--grep", event, cwd=project)
        if result.returncode != 0 and "No matches" not in result.stdout:
            pytest.skip(f"could not fetch the reference: {result.stderr.strip()[:120]}")
        if event not in result.stdout:
            missing.append(event)
    assert not missing, f"not in the live reference, update hookspec.py: {missing}"
