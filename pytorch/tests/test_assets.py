"""assets/test_model_template.py must pass on the placeholder model and catch planted bugs.

The template runs in a subprocess so its fixtures (from assets/conftest_torch.py)
and its pytest options stay separate from this suite's pytest.ini. Each planted
bug redefines make_model() at the end of a copy of the template, which is the
same edit a user makes when they swap in their own model.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parent.parent
ASSETS = SKILL_ROOT / "assets"
TEMPLATE = ASSETS / "test_model_template.py"
CONFTEST = ASSETS / "conftest_torch.py"
CONTROL = "test_output_shape_is_batch_by_classes[cpu-full-batch]"


def _outcomes(report: Path) -> dict[str, str]:
    outcomes = {}
    for case in ET.parse(report).getroot().iter("testcase"):
        kinds = {child.tag for child in case} & {"failure", "error", "skipped"}
        outcomes[case.get("name")] = kinds.pop() if kinds else "passed"
    return outcomes


def _pytest(target: str, cwd: Path, report: Path, *extra: str, env=None) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, "-m", "pytest", target,
        "-p", "no:cacheprovider",
        "-c", os.devnull,            # ignore the skill's pytest.ini (testpaths, addopts)
        "--rootdir", str(cwd),
        "--strict-markers", "-W", "error", "-q",
        f"--junitxml={report}",
        *extra,
    ]
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def template_run(tmp_path_factory) -> tuple[subprocess.CompletedProcess, dict[str, str]]:
    """Run the template in place, loading conftest_torch.py as a plugin with -p."""
    report = tmp_path_factory.mktemp("template") / "report.xml"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [str(ASSETS), os.environ.get("PYTHONPATH")])),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    proc = _pytest("assets/test_model_template.py", SKILL_ROOT, report,
                   "--confcutdir", str(ASSETS), "-p", "conftest_torch", env=env)
    return proc, _outcomes(report)


def test_template_passes_on_the_placeholder_model(template_run):
    proc, _ = template_run
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]


def test_template_runs_every_cpu_case_except_an_unavailable_compiler(template_run):
    _, outcomes = template_run
    not_passed = {
        name: outcome for name, outcome in outcomes.items()
        if "cpu" in name and outcome != "passed" and not name.startswith("test_compiled_model")
    }
    assert not_passed == {}


def test_template_skips_absent_accelerators_instead_of_failing(template_run):
    _, outcomes = template_run
    accelerator_failures = {
        name: outcome for name, outcome in outcomes.items()
        if ("cuda" in name or "mps" in name) and outcome in {"failure", "error"}
    }
    assert accelerator_failures == {}


PLANTS = {
    "softmax-head": (
        """
        class Planted(TinyClassifier):
            def forward(self, x):
                return super().forward(x).softmax(dim=1)
        """,
        "test_head_emits_raw_logits[cpu-softmax]",
    ),
    "detached-layer": (
        """
        class Planted(TinyClassifier):
            def forward(self, x):
                h = self.net[0](x).detach()          # first Linear is cut off from the loss
                return self.net[1:](h)
        """,
        "test_every_parameter_gets_a_finite_gradient[cpu]",
    ),
    "unregistered-tensor": (
        """
        class Planted(TinyClassifier):
            def __init__(self):
                super().__init__()
                self.scale = torch.rand(NUM_CLASSES) + 0.5   # plain attribute, not a buffer
            def forward(self, x):
                return super().forward(x) * self.scale.to(x.device)
        """,
        "test_state_dict_round_trip_reproduces_outputs[cpu]",
    ),
    "functional-dropout": (
        """
        class Planted(TinyClassifier):
            def forward(self, x):
                h = self.net[:3](x)
                h = F.dropout(h, p=0.2)              # training defaults to True, ignores eval()
                return self.net[4](h)
        """,
        "test_eval_mode_is_deterministic[cpu]",
    ),
    "batch-statistics-in-eval": (
        """
        class Planted(TinyClassifier):
            def forward(self, x):
                return super().forward(x - x.mean(dim=0, keepdim=True))
        """,
        "test_eval_output_for_a_row_ignores_the_other_rows[cpu]",
    ),
    "unseeded-init": (
        """
        class Planted(TinyClassifier):
            def __init__(self):
                super().__init__()
                g = torch.Generator()
                g.seed()                             # fresh OS entropy, ignores manual_seed
                with torch.no_grad():
                    self.net[0].weight.normal_(0, 0.3, generator=g)
        """,
        "test_training_is_reproducible_under_the_same_seed[cpu]",
    ),
}


@pytest.fixture(scope="module")
def run_planted(tmp_path_factory):
    """Copy the template plus conftest into a fresh folder, plant the bug, run once per plant.

    This is the layout a user ends up with: the template and conftest_torch.py
    (renamed conftest.py) side by side.
    """
    cache: dict[str, dict[str, str]] = {}

    def _run(plant: str) -> dict[str, str]:
        if plant not in cache:
            folder = tmp_path_factory.mktemp(plant)
            code = textwrap.dedent(PLANTS[plant][0]) + "\n\ndef make_model():\n    return Planted()\n"
            (folder / "test_model_template.py").write_text(TEMPLATE.read_text() + "\n" + code)
            (folder / "conftest.py").write_text(CONFTEST.read_text())
            report = folder / "report.xml"
            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            _pytest("test_model_template.py", folder, report, "-m", "not slow", env=env)
            cache[plant] = _outcomes(report)
        return cache[plant]

    return _run


@pytest.mark.parametrize("plant", list(PLANTS))
def test_planted_bug_fails_its_matching_test(run_planted, plant):
    outcomes = run_planted(plant)
    assert outcomes.get(PLANTS[plant][1]) == "failure", outcomes


@pytest.mark.parametrize("plant", list(PLANTS))
def test_planted_bug_leaves_the_shape_test_passing(run_planted, plant):
    """Guards against a plant that breaks import, which would fail every test at once."""
    outcomes = run_planted(plant)
    assert outcomes.get(CONTROL) == "passed", outcomes
