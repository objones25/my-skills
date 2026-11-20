# Hooks: git and Claude Code

A **hook** is a command that runs automatically on an event, so a check happens
without anyone remembering to run it. Two kinds matter here:

| Kind | Fires on | Enforces | Config |
|---|---|---|---|
| Git hooks, via the pre-commit framework | `git commit`, `git push` | lint, audit, secrets, workflow checks, fast tests | `assets/.pre-commit-config.yaml` |
| Claude Code hooks | session start, Claude editing a file, Claude or a subagent ending its turn | audit on every test edit; no "done" while the suite is red or the tests were weakened to get green | `assets/claude/settings.json` + `assets/claude/hooks/` |

Both can be bypassed (`git commit --no-verify`, disabling hooks), so CI re-runs the
same checks. Hooks give fast feedback; CI is the gate.

## Contents

- [Git hooks with pre-commit](#git-hooks-with-pre-commit)
- [Claude Code hooks](#claude-code-hooks)
- [Installing the Claude Code hooks](#installing-the-claude-code-hooks)
- [How the hooks behave](#how-the-hooks-behave)
- [Exit codes and output, the rules that bite](#exit-codes-and-output-the-rules-that-bite)
- [Testing hooks](#testing-hooks)

## Git hooks with pre-commit

| Stage | Hook | Why at this stage |
|---|---|---|
| pre-commit | whitespace, YAML/TOML syntax, merge markers, large files, `breakpoint()`, private keys | instant |
| pre-commit | `ruff-check --fix`, `ruff-format` | sub-second |
| pre-commit | `gitleaks` (staged changes) | a secret is cheapest to stop before it is in history |
| pre-commit | `zizmor`, `actionlint` | workflow mistakes, only when workflow files change |
| pre-commit | `pytest-audit` | stdlib AST scan, only when files under `tests/` change |
| pre-push | `pytest-fast` (`-x --no-cov -p no:randomly`) | seconds, not milliseconds; once per push is enough |

Setup:

```
pip install pre-commit
pre-commit install --hook-type pre-commit --hook-type pre-push
pre-commit run --all-files
```

Keep the commit stage under a couple of seconds. A slow hook gets skipped with
`--no-verify`, and then it protects nothing. `ruff-check` replaced the older `ruff`
hook id in ruff-pre-commit; both currently work, but new configs should use
`ruff-check`. `pre-commit validate-config` reported the shipped config valid.

## Claude Code hooks

Claude Code runs hooks at points in its loop. These four entries are used:

| Event | When | What our hook does |
|---|---|---|
| `SessionStart`, matcher `startup\|resume\|clear\|compact\|fork` | the session starts, resumes, is cleared, compacts, or is forked | records the current git commit as the session's base (first one wins) |
| `PostToolUse`, matcher `Edit\|Write\|MultiEdit` | right after Claude writes a file | audits the file if it is a test; runs it; reports back to Claude |
| `Stop` | when Claude is about to end its turn | runs the audit, the diff guard against the session base, and the fast suite; if any is red, or could not run, Claude keeps working |
| `SubagentStop` | when a subagent is about to finish | same script as `Stop`, so a subagent cannot hand back a gamed suite |

Hooks are deterministic: they run every time, regardless of whether the model
remembers the skill's instructions. That is the point of moving the gates here.

## Installing the Claude Code hooks

From the project root:

```
mkdir -p .claude/hooks
cp pytest-expert/assets/claude/hooks/*.py .claude/hooks/
cp pytest-expert/scripts/audit_tests.py pytest-expert/scripts/diff_guard.py .claude/hooks/
cp pytest-expert/assets/claude/settings.json .claude/settings.json
```

The three hooks import `_common.py`, so copy it with them (the `*.py` above does). The hooks
look for `audit_tests.py` and `diff_guard.py` next to themselves, and otherwise in the skill's
`scripts/` folder (the layout as shipped, `assets/claude/hooks` beside `scripts/`). If neither
has them, the check cannot run and the gate says so (see below).

If `.claude/settings.json` already exists, merge the `hooks` key instead of
overwriting the file. Then type `/hooks` in Claude Code to confirm all four entries are
listed. Project hooks run only after you accept the workspace trust prompt for the
folder.

Configuration, all optional:

| Variable | Default | Effect |
|---|---|---|
| `PYTEST_EXPERT_PYTHON` | `.venv/bin/python`, then `venv/bin/python`, then the hook's own interpreter | interpreter that has your test dependencies |
| `PYTEST_EXPERT_RUN_EDITED_TESTS` | `1` | set `0` to audit edited tests without running them |
| `PYTEST_EXPERT_MAX_BLOCKS` | `3` | consecutive Stop blocks before the gate gives up; a positive integer (`0` is invalid); a value above `CLAUDE_CODE_STOP_HOOK_BLOCK_CAP` is lowered to it, with a note in the `systemMessage` |
| `CLAUDE_CODE_STOP_HOOK_BLOCK_CAP` | `8` | read, never set, by the gate: Claude Code's own limit on consecutive Stop blocks (`0` turns it off, and then nothing is lowered) |
| `PYTEST_EXPERT_STOP_TIMEOUT` | `480` | seconds allowed for the fast suite; a positive integer |
| `PYTEST_EXPERT_CHECK_TIMEOUT` | `120` in the Stop gate, `60` in the post-edit hook | seconds allowed for each of the audit and the diff guard; a positive integer |
| `PYTEST_EXPERT_DIFF_BASE` | session base, else `HEAD` | git ref the diff guard compares with when no SessionStart base was recorded |
| `PYTEST_EXPERT_DIFF_ALLOW` | empty | comma-separated diff guard codes to accept, for a session where the user asked for a deletion |

A value that is not a positive integer is replaced by its default, and the hook says so in a
`systemMessage`. The hook command is `python3`. On Windows without `python3` on PATH, change
`command` in `settings.json` to `python` or `py`.

## How the hooks behave

**Post-edit.** Real output (2026-11-20, project path shortened to `/tmp/hookdemo`) when Claude
writes a test that computes a value and never asserts it (`result == 3` on its own line):

```json
{
    "decision": "block",
    "reason": "pytest-expert audit flagged test_total.py. Fix each finding (or add `# audit: ignore[CODE] <reason>` when it is a deliberate exception):\n/tmp/hookdemo/tests/test_total.py:1: NO_ASSERT: test_total asserts nothing that can fail\n/tmp/hookdemo/tests/test_total.py:3: BARE_COMPARISON: `result == 3` is not asserted\n\n2 finding(s): BARE_COMPARISON=1, NO_ASSERT=1\n"
}
```

For `PostToolUse` the edit has already happened; `"decision": "block"` delivers the
reason to Claude as feedback. If the audit is clean but the edited file's tests
fail, the hook returns the failure as `additionalContext` instead of a block,
because a red test is the expected state partway through test-first work.

The audit's exit status is the contract: 0 clean, 1 findings, anything else (exit 2, a crash, a
timeout, a missing script or interpreter) means it could not run. The hook then reports that
to Claude as `additionalContext` and to the user as a `systemMessage`, and never treats it as
clean. The same goes for an event that is not a JSON object, and for a crash inside the hook.
An edited file that pytest cannot run (an exit other than 0, 1 or 5) is reported to Claude
the same way; a run longer than 90 seconds is reported as slow; exit 5 (no tests in the
file) is silent.

**Stop gate.** Real output (2026-11-20) with a failing suite, the first lines of every block
message and then the problem:

```
pytest-expert: do not stop yet. Fix the code so these pass, then finish.
Do not edit, skip, or delete a failing test to make it pass, and do not special-case the code for a test's inputs. If a failing test contradicts the specification, stop and escalate instead: write .claude/pytest-expert-escalation.md with the lines 'Issue type:', 'Test:', 'Problem:', 'Tried:', 'Recommendation:'.

Fast suite failing:
F
=================================== FAILURES ===================================
_______________________________ test_cart_total ________________________________
tests/test_cart.py:2: in test_cart_total
    assert sum([1, 2]) == 4
E   assert 3 == 4
E    +  where 3 = sum([1, 2])
=========================== short test summary info ============================
FAILED tests/test_cart.py::test_cart_total - assert 3 == 4
!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!
1 failed in 0.07s
```

The gate audits the whole project root with `audit_tests.py`, runs pytest on the whole project
(so pytest decides where the tests are: `tests/`, the root, `testpaths`), and, inside a git
repository, runs `diff_guard.py`. Both hooks call the audit with no flags, so `assert_helpers` under
`[tool.pytest-expert]` in `pyproject.toml` applies to them without any hook setting. Each result is
one of three kinds:

| Situation | What the gate does |
|---|---|
| audit findings, diff guard findings, a failing or too-slow suite | blocks, with the findings |
| a check could not run: a script missing, exit 2, a crash, a timeout, an unknown diff base, pytest not installed in the interpreter, pytest exiting with a code other than 0, 1 or 5 | blocks, with a `COULD NOT CHECK:` reason naming what failed and how to fix it, and a `systemMessage` |
| not a git repository (or no commit yet), so no diff guard; pytest exit 5 (nothing collected, so the suite checked nothing); a bad numeric setting | lets Claude stop and shows a `systemMessage` warning; a tree whose pytest run exited 5 is not remembered as green |
| the event on stdin is unreadable | lets the stop through with a `systemMessage`: nothing can be checked without it |
| an unexpected exception inside the gate | blocks once, with the traceback |

When no test file exists at all, pytest exits 5 and the audit has nothing to audit; that is
one warning, not a second failure. Not finding a checker is a block, not a pass, so an install
that forgot to copy `diff_guard.py` is noticed at the first stop.

**Diff guard.** Real output (2026-11-20) after the session started at a commit with two tests,
and Claude then skipped one and committed (`tests/test_cart.py`):

```
Test-weakening edits since c84dc368ca4c (if the user asked for one, say so and they can set PYTEST_EXPERT_DIFF_ALLOW):
tests/test_cart.py:9: SKIP_ADDED: test_discount: skip/xfail added
```

The suite was green, so a gate that only ran the tests would have let Claude stop. The guard compares
with the commit recorded at `SessionStart`, not `HEAD`, so committing the change does
not hide it. Outside a git repository, or before the first commit, `diff_guard.py`
exits 2; the gate checks for that first and shows a warning instead of blocking. A base that
does not exist (a stale `PYTEST_EXPERT_DIFF_BASE`) is a block. Codes and what counts as
legitimate: `llm-written-tests.md`.

**Escalation exit.** When the agent believes a failing test is wrong, it writes the
five-line note instead of touching the test. Real output (2026-11-20), for a test calling a
function that does not exist:

```
pytest-expert: Claude escalated instead of finishing (saved to .claude/pytest-expert-escalations/20261120T210729-37617.md).

Issue type: test contradicts spec
Test: tests/test_cart.py::test_discount
Problem: apply_discount is not defined anywhere and the spec has no discount feature
Tried: searched src/ and docs/
Recommendation: delete the test or add the feature to the spec

The suite is still red:
Fast suite failing:
.F
...
E   NameError: name 'apply_discount' is not defined
...
1 failed, 1 passed in 0.07s
```

The note is archived (one note, one stop), and a note missing any of the five fields is
sent back. Why this exists, with the evidence: `llm-written-tests.md`. Add
`.claude/pytest-expert-escalation*` to `.gitignore` if you do not want notes committed.

Three guards keep it from becoming a loop or a tax:

1. **Fingerprint.** It hashes the path, size, and modification time of every `.py`
   file, and of every config file pytest or coverage reads: `pytest.toml`, `.pytest.toml`,
   `pytest.ini`, `.pytest.ini`, `pyproject.toml`, `tox.ini`, `setup.cfg`, `.coveragerc` and
   `.coveragerc.toml`
   (a config change can turn a green run red or lower the bar). If nothing changed since the
   last green run, it exits immediately.
2. **`stop_hook_active`.** Claude Code sets this when it is already continuing because
   of a Stop hook. The gate does not treat it as a reason to let a red tree stop: its own
   counter (guard 3) is the loop guard, so a red suite keeps blocking until the counter
   reaches the limit. When the field is true and the gate blocks again, the reason starts
   with "block N of MAX", so the progress toward the limit is visible. A green or unchanged
   tree is allowed through silently whatever the field says.
3. **Its own counter.** The block count is kept in the per-session state file and is reset
   only by a green run, a give-up or an escalation (a new session id starts at zero).
   After `PYTEST_EXPERT_MAX_BLOCKS` consecutive blocks (default 3) the gate lets Claude stop
   and shows a `systemMessage` saying the suite is still red. Claude Code itself overrides a
   Stop block after `CLAUDE_CODE_STOP_HOOK_BLOCK_CAP` consecutive ones (default 8, `0` turns
   the cap off; hooks reference, read 2026-11-20), so the gate lowers a larger
   `PYTEST_EXPERT_MAX_BLOCKS` to that cap and says so, which keeps its own give-up message
   ahead of the platform's override. Keep the limit low: each forced retry is another round
   of pressure, and on ImpossibleBench allowing multiple submissions raised cheating from
   33% to 38% [IB §5.3] (`papers/impossiblebench.md`).

**Optional: read-only tests for implementation tasks.** When the task is "make the
code pass these tests" rather than "write tests", deny edits to the test tree for that
session, for example with `permissions.deny` rules for `Edit(tests/**)` and
`Write(tests/**)` in `.claude/settings.local.json`. On ImpossibleBench read-only tests
stopped test edits without hurting legitimate performance, but did not stop
special-casing or operator overloading [IB §5.2]; the diff guard's source warnings
cover part of that. (Not exercised in the selftests; check the rule syntax against the
Claude Code permissions docs before relying on it.)

## Exit codes and output, the rules that bite

From the Claude Code hooks reference (re-read 2026-11-02):

| Rule | Consequence |
|---|---|
| Exit code 2 is the blocking code; exit 1 is a non-blocking error and the action continues | a policy hook that `exit 1`s enforces nothing |
| For `PostToolUse`, exit 2 shows stderr to Claude, but the tool already ran | it is feedback, not prevention |
| For `Stop`, exit 2 or `{"decision": "block"}` keeps Claude working | the reason must say what to fix |
| stdout on exit 0 is parsed as JSON when it starts with `{` and ends with `}` | print nothing else on stdout |
| Hook text over 10,000 characters is moved to a file with a 2,000-character preview | the hooks truncate to 9,000 |
| A typo in the script path makes the hook fail as non-blocking | check the first run's output |
| With an `args` array, `command` is spawned directly with no shell | no shell expansion beyond `${CLAUDE_PROJECT_DIR}` |
| `SubagentStop` accepts the same `decision: "block"` as `Stop` | one script serves both |
| `SessionStart` runs on every start, resume, `/clear`, and compaction, and supports only `command` and `mcp_tool` hooks | keep it fast; ours runs one `git rev-parse` |

Our hooks exit 0 and speak JSON, which is the structured route the reference
recommends over mixing exit codes and output.

That covers a hook that runs. A hook that cannot start (a wrong path after a partial copy,
no `python3` on the PATH) or that times out is a hook failure, and Claude Code carries out
the action when a hook fails unless the hook sets `"onFailure": "block"` (hooks reference,
"Block the action when a hook fails", read 2026-11-20; needs Claude Code v2.1.295 or later).
The `Stop` and `SubagentStop` entries in `assets/claude/settings.json` set it, so a gate that
cannot run blocks the stop instead of waving it through; the post-edit and session-start
entries leave it unset, because they advise and must not block work when they fail. Not
exercised against a live Claude Code session: the reference shows the field on `PreToolUse`,
and its effect on `Stop` is taken from the general rule.

## Testing hooks

`selftest/test_hooks.py` drives the three hooks exactly as Claude Code does: JSON on
stdin, JSON on stdout, in a temporary project. Its 111 tests (collected 2026-11-20) cover the
audit block, red tests as context, silence on clean files, non-test files ignored, a post-edit
audit that exits 2, crashes, times out or is missing, the Stop block, the fingerprint skip
(and its rerun after a `.py` or config change), a red suite that keeps blocking through the whole
budget when `stop_hook_active` is true (and the "block N of MAX" line, absent when it is false), the
give-up counter and its reset by a green run, a `PYTEST_EXPERT_MAX_BLOCKS` above
`CLAUDE_CODE_STOP_HOOK_BLOCK_CAP` lowered to it (and left alone when the cap is `0`), bad
numeric settings, every could-not-check case in the table above, tests outside `tests/`,
diff-guard cases in a temporary git repo (deleted test, added skip, base surviving a commit,
`PYTEST_EXPERT_DIFF_ALLOW`, an unknown base, `SubagentStop`), the shipped layout finding the
scripts in the skill's `scripts/` folder, and the escalation cases (block message names the
exit, complete note lets Claude stop, one note per stop, incomplete note sent back).
`selftest/test_diff_guard.py` covers the guard itself.

```
pytest selftest/test_hooks.py -q
```

## Sources

- Claude Code hooks reference: https://code.claude.com/docs/en/hooks (fetched 2026-11-02: `args`, `SubagentStop`, `SessionStart` input and matchers, 10,000-character cap).
- Stop-hook loop report: https://github.com/anthropics/claude-code/issues/54360
- pre-commit: https://pre-commit.com/
