# Changelog

## 1.0.1 (2026-11-19)

### Fixed: tests

- `tests/test_examples.py`: the mixed-currency addition inside `pytest.raises` is now assigned to `_`, so Pyright no longer reports it as an unused expression (`reportUnusedExpression`). The assertion is unchanged.

## 1.0.0 (2026-11-16)

First versioned release. Fixes the findings of the 2026-11-16 audit (`audits/design-patterns-expert/2026-11-16.md` in the My_Skills repo, local only); item ids below refer to that report.

### Fixed: behaviour

- `scripts/design_smells.py` exited 0 and printed "recommend no pattern" when given a path that does not exist. It now exits 2 with `error: no such file or directory`, as its docstring promises for bad arguments (I-FIX-F1-0028; test `test_missing_path_is_an_error_not_a_clean_report`).
- `tests/test_grade.py` always skipped: the grader it tests was never committed. `evals/grade.py` and `evals/evals.json` now exist, built to that test, so the suite runs it (I-W4-0220).

### Fixed: wrong claims

- SKILL.md: `register()` is no longer listed as something only ABCs have; Protocol classes inherit it from `ABCMeta` (I-W1-0078). Same correction in `references/python-idioms.md` and `references/selection.md`.
- SKILL.md: `check_layers.py` is described as checking direct imports, not as proving a layer clean; transitive chains and `importlib` calls need import-linter (I-W1-0100).
- SKILL.md: the heavy-pattern warnings are no longer called "the consensus of the people who named the patterns" (I-W1-0045).
- SKILL.md: Event Sourcing replay: Fowler says internal bug fixes are easy to reprocess and external gateways are the complication; the old text said bug fixes break replay (I-FIX-F1-0001).
- SKILL.md: Fowler's cost of repair belongs to a needed feature built wrong, not to a wrong guess (I-FIX-F2-0001).
- SKILL.md and `references/python-idioms.md`: Protocol `isinstance` cost is now given per version (about 1.4 µs on 3.11, about 0.12 to 0.14 µs on 3.12 to 3.15, Apple M4 Pro) instead of one range that did not hold on 3.11 (I-W1-0075, I-W4-0021); "several runs and machines" removed (I-W4-0022).
- SKILL.md: antipatterns row no longer says every pattern has a failure entry (I-W1-0097).
- `references/python-idioms.md`: the opening no longer claims every fact is from the docs and every executed fact is marked (I-W4-0001, I-W4-0002); the P6 entry notes the 3.15 `DeprecationWarning`.
- `references/behavioral.md`: a `Callable` alias is checked by mypy and pyright (arity, parameter and return types); a Protocol adds names, keyword-only parameters, overloads and attributes (I-W2-0014).
- `references/creational.md`: slotted dataclass vs `NamedTuple` read times replaced with the 2026-11-16 measurements and a ratio; 3.15 added (I-W2-0098, I-W2-0099).
- `references/resilience.md`: Hystrix's 99.5th-percentile timeout rule added beside AWS's (I-W3-0196); in-process deadline propagation exists (nested `asyncio.timeout`, AnyIO) (I-W3-0200); unsourced "most systems get it wrong" removed (I-FIX-F2-0011); Full Jitter no longer called AWS's default (I-FIX-F2-0012); bulkhead primitives: `httpx.Limits` is third party (I-FIX-F3-0011); IETF Idempotency-Key draft no longer called "best available" and "widely followed" (I-FIX-F2-0017, I-FIX-F2-0018); at-least-once brokers "can" redeliver, not "guarantee" duplicates (I-FIX-F2-0019); stamina 26.2.0's two gaps stated: additive 0 to 1 s jitter and no retry budget (I-FIX-F2-0023, I-FIX-F2-0024).
- `references/architectural.md`: no longer says every entry leads with its warning or that all these patterns are hard to back out of (I-FIX-F2-0006, I-FIX-F3-0009); CQRS read models can widen inconsistency (I-FIX-F2-0009); saga countermeasure is "version files", as Azure names it (I-FIX-F1-0022).
- `references/structural.md`: ACL costs: the two items that are this skill's view, not Azure's, are marked (I-FIX-F2-0010); the `sqlite3` injection names the flag `check_layers.py` needs.
- `references/more-patterns.md`: the `assert_never` error text is labelled as coming from adding `OrderCancelled`, not removing a case (I-FIX-F1-0007).
- SKILL.md: the dated test result pinned Python 3.13 and import-linter 2.15 (I-W1-0102); results now live here with a stamp.

### Added

- `metadata.version` and `metadata.test-command`; negative scope in the description (hands off to async-expert and negative-space-programming).
- `evals/evals.json` (7 cases, one control), `evals/trigger_evals.json` (10 should, 9 should not), `evals/grade.py` with tests.
- `tests/test_skill_structure.py`: frontmatter, backticked paths, the reference table, the 13-kinds and 20-checks counts, and a test for every script.
- `references/sources.md`, `AUDIT.md`, an "Extend this skill" section.

### Results

- Tests: 104 passed, 0 skipped, on 2026-11-16 on macOS arm64 with Python 3.13.2 (uv), pytest 9.2.0, mypy 2.4.1, import-linter 2.16, via `metadata.test-command`.
- `scripts/verify_python_claims.py`: 20/20 on CPython 3.11.18, 3.12.16, 3.13.18, 3.14.10 and 3.15.2 (2026-11-16, macOS arm64).
- Evals: not run for this version.
- `check_skill.py --strict`: 0 errors. Two SB013 warnings remain, both em dashes inside verbatim Rhodes quotes (`references/creational.md`, `references/structural.md`), left as quoted; `--strict --ignore SB013` exits 0. `package_skill.py`: 46 files, tests passed in the unpacked copy.

## Before 1.0.0 (unversioned)

- 2026-11-03: 76 passed, 1 skipped (`evals/` not shipped), macOS arm64, Python 3.13, mypy 2.4.1, import-linter 2.15, as recorded in SKILL.md at the time. Reproduced on 2026-11-16 with the same counts.
