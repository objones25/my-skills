"""Structural checks on the skill itself: links resolve, citations are registered, evals are well formed.

These catch the drift that creeps in when references are edited separately:
a file renamed but still linked, an arXiv id quoted in a reference but missing
from sources.md, a duplicate eval id.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = (ROOT / "SKILL.md").read_text()
REFS = {p.name: p.read_text() for p in (ROOT / "references").glob("*.md")}
SOURCES = REFS["sources.md"]


def test_frontmatter():
    m = re.match(r"^---\nname: \"?([a-z0-9-]+)\"?\ndescription: (.+?)\n(?:license: .+\n)?(?:metadata:\n(?:  .+\n)+)?---\n", SKILL, re.S)
    assert m, "frontmatter must open with name and description"
    assert m.group(1) == "building-effective-agents"
    assert len(m.group(2)) <= 1024
    assert "<" not in m.group(2) and ">" not in m.group(2)


def test_every_linked_path_exists():
    linked = set(re.findall(r"`((?:references|scripts|examples|tests|evals)/[\w./-]+)`", SKILL))
    for text in list(REFS.values()) + [p.read_text() for p in (ROOT / "examples").glob("*.md")]:
        linked |= set(re.findall(r"`((?:references|scripts|examples|tests|evals)/[\w./-]+)`", text))
    missing = [p for p in sorted(linked) if not (ROOT / p.split()[0]).exists()]
    assert not missing, missing


def test_bare_reference_names_exist():
    names = set(re.findall(r"`(\w[\w-]*\.md)`", SKILL + "".join(REFS.values())))
    known = set(REFS) | {p.name for p in ROOT.glob("*.md")} | {"README.md", "NOTES.md", "CLAUDE.md"}
    unknown = sorted(n for n in names if n not in known and not n[0].isdigit())
    assert not unknown, unknown


def test_every_reference_file_is_listed_in_skill_md():
    for name in REFS:
        if name != "sources.md":
            assert f"references/{name}" in SKILL, name


def test_every_arxiv_id_cited_is_in_sources():
    cited = set()
    for name, text in REFS.items():
        if name != "sources.md":
            cited |= set(re.findall(r"arXiv (\d{4}\.\d{4,5})", text))
    registered = set(re.findall(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", SOURCES))
    missing = sorted(cited - registered)
    assert not missing, f"cited but not in sources.md: {missing}"


def test_evals_well_formed():
    d = json.loads((ROOT / "evals" / "evals.json").read_text())
    ids = [e["id"] for e in d["evals"]]
    assert len(ids) == len(set(ids))
    for e in d["evals"]:
        assert e["prompt"].strip() and len(e["expectations"]) >= 3, e["name"]


def test_examples_are_marked_illustrative():
    for p in (ROOT / "examples").glob("0*.md"):
        assert "*Illustrative" in p.read_text(), p.name


def test_scripts_do_not_hardcode_a_tmp_path():
    offenders = [p.name for p in (ROOT / "scripts").glob("*.py")
                 if re.search(r"""["']/tmp/""", p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_every_script_compiles():
    # Some scripts import optional packages and their tests skip, so check syntax directly.
    for path in (ROOT / "scripts").glob("*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


# Third-party imports each script is allowed, and which SKILL.md must name.
OPTIONAL_IMPORTS = {"inspect_eval_template.py": {"inspect_ai"}, "agent_loop.py": {"anthropic"}}


def test_scripts_use_only_the_stdlib_apart_from_the_named_optional_packages():
    import ast
    import sys

    siblings = {p.stem for p in (ROOT / "scripts").glob("*.py")}
    for path in (ROOT / "scripts").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
        third_party = roots - set(sys.stdlib_module_names) - siblings - {"__future__"}
        assert third_party == OPTIONAL_IMPORTS.get(path.name, set()), (path.name, third_party)
    for package in set().union(*OPTIONAL_IMPORTS.values()):
        assert f"`{package}`" in SKILL, f"SKILL.md does not name the optional package {package}"


def _example_runs():
    for path in sorted((ROOT / "examples").glob("0*.md")):
        for block in re.findall(r"```\n(.*?)```", path.read_text(), re.S):
            lines = block.rstrip("\n").split("\n")
            if lines[0].startswith("$ python "):
                yield path.name, lines[0][2:], "\n".join(lines[1:])


def test_example_script_output_matches_a_real_run():
    import shlex
    import subprocess
    import sys

    runs = list(_example_runs())
    assert len({name for name, _, _ in runs}) == 4, "examples 01-04 show script output"
    for name, command, shown in runs:
        argv = [sys.executable, *shlex.split(command)[1:]]
        real = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=60).stdout
        assert _matches(shown.split("\n"), real.rstrip("\n").split("\n")), (name, command)


def _matches(shown: list[str], real: list[str]) -> bool:
    """Line-for-line match where a "..." line stands for any number of omitted lines
    and a line ending in "; ..." stands for a real line it is a prefix of. An omission
    without one of these marks fails."""
    if not shown:
        return not real
    head, rest = shown[0], shown[1:]
    if head.strip() == "...":
        return any(_matches(rest, real[i:]) for i in range(len(real) + 1))
    if not real:
        return False
    if head.endswith(" ..."):
        ok = real[0].startswith(head[:-3])
    else:
        ok = real[0] == head
    return ok and _matches(rest, real[1:])
