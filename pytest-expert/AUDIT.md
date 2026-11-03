# Audit, 2026-11-02

Previous version: 2026-10-04 (verified on Python 3.12 with pytest 9.1.1; 60 selftests,
6 evals, no paper digests, no async or agent-gaming material).

## Gaps found

- Nothing caught a suite made green by weakening it. The Stop hook checked that tests
  passed, so deleting the failing test, dropping a parametrize row, adding a skip,
  deselecting in config, or lowering the coverage floor all got through. ImpossibleBench
  shows this is the main way coding agents cheat (`references/papers/impossiblebench.md`).
- No escalation path: an agent facing a wrong test had only "keep working" or "game it".
- No async testing material, although pytest-asyncio 1.x changed its configuration and
  breaks the shipped strict config.
- No snapshot testing, and no warning that update flags bless regressions.
- `mutate_check.py` and `flake_hunt.py` had no selftests.
- No research digests; claims about mutation testing, flaky tests, and LLM-written tests
  rested on three citations.
- pytest 9.1 and 9.2 features missing: `max_warnings`, `strict_markers` checking `-m`,
  `yield_fixture` removal schedule, `RaisesGroup`, `register_fixture`.

## Corrections to existing text and code

| Problem | Evidence | Fix |
|---|---|---|
| pytest-asyncio 1.4.1 plus `filterwarnings = ["error"]` gives `INTERNALERROR` before any test | reproduced; the plugin warns from `pytest_configure` when `asyncio_default_fixture_loop_scope` is unset | commented asyncio block in `assets/pytest-config.toml`; `references/async-testing.md` |
| Demo and security template leak `sqlite3` connections; on Python 3.13 that is a `ResourceWarning`, so the strict filter fails an unrelated later test or crashes at exit | `-X dev` shows the warning on 3.13 and not on 3.12; templates failed `1 failed, 67 passed` under the strict config | closing fixtures in `demo/test_v0_findings.py` and `assets/test_security_template.py` |
| `audit_tests.py` reported `NO_ASSERT` for a test whose only check was `with pytest.RaisesGroup(...)` | found writing the async template | `RaisesGroup` and `RaisesExc` count as assertions; two selftests |
| Flaky-test reference said "about 170 reruns" without the order-dependent figure | Gruber et al. §IV-C: 170 for non-order-dependent, 31 for order-dependent; 1 or 3 reruns to classify one failure | table in `flaky-tests.md`; `flake_hunt.py` docstring |
| Just et al. 2014 PDF link | the umass.edu URL returned 404 | cs.ubc.ca copy |
| Inozemtseva and Holmes cited without the reply to their size control | Chen et al. ASE 2020 | added next to it in `mutation-testing.md` |
| Workflow step 5 allowed only killed or equivalent survivors | Google treats most raw mutants as unproductive | third outcome, unproductive with a reason |
| Solver timing "about 1.7 s" | 3 runs on CrossHair 0.0.112: about 0.4 s | updated, old figure kept with its version |
| ruff 0.17 default output format changed | demo README showed the old one-line format | `--output-format concise` in the documented command |
| Em dashes in four references (house style) | 92 occurrences | replaced; structure test now fails on any |

## Added

- `scripts/diff_guard.py` with 16 codes (12 errors, 4 warnings, including `CASES_REDUCED`
  for dropped parametrize rows, `SYNTAX` so a broken file is not reported as mass
  deletion, and source-side `CONSTANT_EQ` and `SPECIAL_CASE`); 45 selftests on throwaway
  git repos.
- Hooks: `pytest_expert_session_start.py` records the starting commit so committing a
  weakened test does not hide it; the Stop gate runs the guard and also serves
  `SubagentStop`; an escalation exit (`.claude/pytest-expert-escalation.md`, five
  required fields, archived after use); a block message worded after ImpossibleBench's
  strictest prompt. Hook selftests grew from 8 to 17.
- Audit code `YIELD_FIXTURE`; rules 32 to 34.
- References: `async-testing.md`, `llm-written-tests.md`, `sources.md`, 16 digests in
  `references/papers/`; snapshot testing in `parametrize-and-doubles.md`; generator
  checks in `property-fuzz-symbolic.md`; `register_fixture` in `fixtures.md`; zizmor's
  input-collection behavior in `ci-cd.md`; plugin entry-point fixtures.
- `assets/test_async_template.py` (9 tests, audit-clean).
- `examples/diagnoses/` with five worked diagnoses from real runs.
- Selftests for `mutate_check.py` and `flake_hunt.py`, and `test_skill_structure.py`.
- Evals rewritten in the expectations format, 6 to 17 cases.

## How sources were checked

- Library APIs: signatures read with Python's `inspect` on the installed versions
  (pytest 9.2.0, Hypothesis 6.170.14, pytest-asyncio 1.4.1, anyio 4.16.0), options from
  `pytest --help`, and changelogs and docs through Context7 (pytest 9.2.0,
  pytest-asyncio v1.4.0). Every behavior quoted in a reference was reproduced in a
  scratch project.
- One disagreement: Context7's pytest reference shows `pytest.register_fixture(fixture_func,
  name=None)`. The installed 9.2.0 signature is keyword-only, `(*, name, func, node,
  scope="function", params=None, ids=None, autouse=False)`; a positional call raises
  `TypeError`. The skill follows the installed code.
- Claude Code hooks: the hooks reference page fetched 2026-11-02 (`args`,
  `SubagentStop`, `SessionStart` input and matchers, 10,000-character cap).
- anyio testing docs fetched 2026-11-02.
- Action and pre-commit pins: `git ls-remote` (the GitHub API was not reachable from the
  verification machine).
- Papers: read from arXiv, author PDFs, or open-access proceedings by a research pass;
  each digest header says how much was read. Industrial papers are self-reports.

## Not verified or not done

- Not exercised: `permissions.deny` rules that make tests read-only (described in
  `hooks.md` as optional), and the anyio trio backend (trio not installed).
- The guard does not detect recorded extra state (behavior that changes on the second
  call), weaker assertions that keep the same count, or special cases on literals no
  test contains.
- Suggested by the digests and left for later: statement-deletion and arid-line handling
  in `mutate_check.py`, a `--changed-since` option, survivors printed as test prompts,
  grouping co-failing tests in `flake_hunt.py`, `DUPLICATE_ASSERT` and suboptimal-assert
  audit codes from PyNose, repeating newly written tests in the post-edit hook.
- `security-testing.md` regex traps were verified on Python 3.12 in October and not
  re-run on 3.13.
