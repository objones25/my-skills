from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_python_claims.py"


def test_every_documented_python_claim_holds_on_this_interpreter() -> None:
    out = subprocess.run([sys.executable, str(SCRIPT), "--skip-timing"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout
