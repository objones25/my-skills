# Skill audit: patterns, gaps and an outline for a skill builder

Audit date: 2026-11-03. This records the skills as they were received, before any fixes.

## 1. Summary

- **The most recent skills share a recognizable shape.** The six skills packaged on 2026-11-02 include all five that have a test guarding their own structure, and all the root-level `CHANGELOG.md` and `AUDIT.md` files. Three have all eight components on your checklist: `pytest-expert`, `pytorch` and `transformer-architecture`. Those six came from one day, so this may reflect one batch of work as much as a trend.
- **The largest gap is evaluation.** 7 of the 17 committed skills have no evals, and no skill measures whether it triggers correctly. The 10 that have evals use five different formats, and none of them keeps a log of results.
- **Running the tests found problems reading had not caught.** Of the 15 committed suites run, 2 had failing tests that the skill's documentation does not mention: `pytorch`'s training template crashes on `--resume` on a non-CPU device (observed on Apple Silicon), and a `massively-parallel-rl` learning test did not pass here.
- **"Successful" is inferred, not measured.** There is no outcome data showing that any pattern below improves results. Patterns are ranked by how often they appear, how recently they were adopted, and judgment. An evaluation framework is what would test them.

### How to read the evidence

| Label | Meaning |
| --- | --- |
| Measured | Produced by running a command for this audit |
| Read | Read from the files and open to checking at the cited path |
| Judgment | An opinion about why something helps or what to do next |
| Self-report | A skill's own claim that this audit did not reproduce |

### Scope and limits

- 21 archives were received, and 17 skills are committed. `accessible-course-materials`, `lesson-design`, `find-docs` and `study-notes` are held back at the owner's request.
- Each skill had its own read-only investigator. An independent reviewer who wrote none of this checked the draft and found 18 errors and 19 overstated claims. These are corrected here, except for one finding that was checked and rejected (the `lesson-design` comparison in section 7).
- Prevalence counts cover the 17 committed skills. Counts come from pattern searches, so treat them as lower bounds. "Most", "several" and "a few" are estimates that were not counted.
- Not tested: whether a skill's instructions improve an agent's work, and any eval run against a model. Citations inside reference files were not re-checked against their sources.

## 2. Component matrix

Key: **Y** present in the standard place, **~** present somewhere non-standard, **-** missing, **n/a** would not earn its place (investigator's judgment).

| Skill | Packaged | references | examples | scripts | tests | assets | CHANGELOG | AUDIT.md | evals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pytest-expert | 11-02 | Y | Y | Y | ~ `selftest/` | Y | Y | Y | Y |
| pytorch | 11-02 | Y | Y | Y | Y | Y | Y | Y | Y |
| transformer-architecture | 11-02 | Y | Y | Y | Y | Y | Y | Y | Y |
| robot-learning-expert | 11-02 | Y | Y | Y | Y | Y | ~ `references/` | - | Y |
| building-effective-agents | 11-02 | Y | Y | Y | Y | n/a | - | Y | Y |
| design-patterns-expert | 11-02 | Y | Y | Y | Y | n/a | - | - | - |
| massively-parallel-rl | 10-27 | Y | - | Y | Y | n/a | - | - | ~ `case.yaml` |
| hook-writer | 10-25 | Y | - | Y | - | Y | - | - | - |
| negative-space-programming | 10-23 | Y | ~ `assets/examples/` | Y | ~ `scripts/` | Y | - | - | - |
| async-expert | 10-22 | Y | ~ `assets/examples/` | Y | ~ `scripts/` | Y | - | - | Y |
| bash-scripting-specialist | 10-22 | Y | - | Y | ~ `scripts/tests/` | Y | - | - | - |
| documentation-expert | 10-22 | Y | - | Y | ~ `scripts/` | n/a | - | - | Y |
| observability-expert | 10-22 | Y | ~ `assets/examples/` | Y | ~ `assets/tests/` | Y | - | - | Y |
| scientific-computing | 10-22 | Y | - | Y | ~ `scripts/tests/` | n/a | - | - | Y |
| annotate-research-paper | 10-20 | Y | - | Y | - | n/a | - | - | - |
| dockerfile-expert | 10-19 | Y | - | Y | ~ `scripts/tests/` | Y | - | - | - |
| kubernetes-expert | 10-19 | Y | - | Y | ~ `scripts/tests/` | Y | - | - | - |

"Packaged" is the archive's modification date in 2026.

Totals for the 17 committed skills (Measured):

| Component | Present anywhere | Standard place |
| --- | --- | --- |
| references | 17 | 17 |
| scripts | 17 | 17 |
| tests | 15 | 6 |
| assets | 11 | 11 |
| examples | 9 | 6 |
| evals | 10 | 9 |
| CHANGELOG | 4 | 3 |
| AUDIT.md | 4 | 4 |

## 3. Test runs

Measured on 2026-11-03 on macOS (Darwin 25.5, Apple Silicon arm64), with Python 3.13 through `uv`, in throwaway environments on scratch copies. Package versions are the latest available, except `massively-parallel-rl`, which used the versions it pins. A failure counts as a skill problem only when it survived supplying the dependencies the skill names.

| Skill | Test path | Result | Skill's own claim |
| --- | --- | --- | --- |
| async-expert | `scripts/test_aioutil.py`, `scripts/test_audit_async.py` | 23 passed | - |
| bash-scripting-specialist | `scripts/tests` | 24 passed | - |
| building-effective-agents | `tests` | 39 passed, 1 skipped (`inspect_ai` absent) | 40 tests: matches |
| design-patterns-expert | `tests` | 74 passed, 3 skipped (`evals/` not shipped, `import-linter`, `mypy`) | - |
| dockerfile-expert | `scripts/tests` | 2 passed | - |
| documentation-expert | `scripts/test_scripts.py` | 21 passed | - |
| kubernetes-expert | `scripts/tests` | 4 passed | - |
| **massively-parallel-rl** | `tests` | **24 passed, 1 failed** | "Tested on" these versions on CPU: this test did not pass here |
| negative-space-programming | `scripts/test_audit.py`, `scripts/test_types.py` | 9 passed, 3 skipped (reasons not recorded) | - |
| observability-expert | `assets/tests`, with `PYTHONPATH=assets` and `wandb` | 43 passed | 43 tests in `SKILL.md`; 42 in `references/verified-behavior.md` |
| pytest-expert | `selftest` | 252 passed, 3 skipped | 255 passed: same total, 3 skipped here |
| **pytorch** | `tests` | **369 passed, 3 failed, 4 skipped** | 375 passed, 1 skipped, machine not recorded |
| robot-learning-expert | `tests` | 184 passed, 5 skipped (reasons not recorded) | - |
| scientific-computing | `scripts/tests` (needs network, about 2 minutes) | 5 passed | - |
| transformer-architecture | `tests` | 181 passed | 181 passed: matches |

Not run:

- `annotate-research-paper` and `hook-writer` have no tests.
- `bash-scripting-specialist`'s `assets/template.bats` and `scripts/test_matrix.sh` need `bats` and extra bash versions, which are not installed.
- `pytest-expert/examples/wallet/tests` was not run.

### What the failures mean

- **`pytorch`, 2 failures, a real bug.** `assets/train_template.py --resume` crashes with `TypeError: RNG state must be a torch.ByteTensor` on `mps`. Line 203 loads the checkpoint with `map_location=device`, which moves the saved RNG state off the CPU, and line 210 passes it to `torch.set_rng_state`. Reproduced outside pytest. `pick_device` selects any accelerator, so CUDA is probably affected too, but CUDA was not run.
- **`pytorch`, 1 failure, a machine-specific document.** The walkthrough in `references/01-devices-amp.md` documents `device: cpu`, and this machine prints `device: mps`.
- **`pytorch`, 4 skips versus the claimed 1.** The 3 extra skips were not explained.
- **`massively-parallel-rl`, 1 failure, cause undetermined.** `test_pqn_learns_cartpole` requires every seed to finish above 400. Seed 2 reaches 500 at the halfway point and then collapses: greedy evaluation gives `[500.0, 53.7]`. Runs are deterministic per seed, so repeats add no evidence. With 6 seeds the reviewer got `[500.0, 53.7, 500.0, 500.0, 398.9, 500.0]`, so 2 of 6 finish below the threshold on this machine.

## 4. Pattern catalog

Counts are out of the 17 committed skills. "Exemplars" are good places to copy from.

### 4.1 Description and triggering

| Pattern | Count | Exemplars | Why it likely helps (Judgment) |
| --- | --- | --- | --- |
| Says what the skill does and when to use it, using the user's words and symptoms rather than only topic names | Most; quality varies | `building-effective-agents/SKILL.md:3`, `transformer-architecture/SKILL.md:3` | The description decides whether a skill loads, and users describe symptoms ("my loss is spiking") more often than topics |
| Negative scope or a named handoff to another skill | 9 | `documentation-expert/SKILL.md:3`, `dockerfile-expert/SKILL.md:3` | Stops overlapping skills competing. The 8 without it: async-expert, design-patterns-expert, hook-writer, negative-space-programming, observability-expert, pytest-expert, robot-learning-expert, transformer-architecture |
| Description kept to 1024 characters with no angle brackets, enforced by a test | 5 | `transformer-architecture/tests/test_skill_structure.py:32` | Current descriptions run from about 519 to 1020 characters, so some are near the limit |

**Watch for:** descriptions that spend their budget listing bundled files, such as `bash-scripting-specialist` and `kubernetes-expert`. That text does little for triggering (Judgment).

### 4.2 SKILL.md structure

| Pattern | Count | Exemplars | Why it likely helps (Judgment) |
| --- | --- | --- | --- |
| Short `SKILL.md` that routes to references through a table saying when to read each file | Most | `robot-learning-expert/SKILL.md:26-47` (102 lines fronting 37 reference files), `negative-space-programming/SKILL.md:178-190` | Keeps always-loaded context small while depth stays reachable |
| Symptom-indexed routing | Several | `building-effective-agents/SKILL.md:83-102`, `scientific-computing/SKILL.md:62-64` | Matches how users arrive: with an error, not a topic |
| Workflow steps ending in a checkable "Done when" | Phrase in 4; used per step in 2 | `pytest-expert/SKILL.md:37-93`, `bash-scripting-specialist/SKILL.md:23-24` | Gives the agent an objective stopping point and discourages premature "done" |
| Hard rules, each with its reason | Most of the larger skills | `kubernetes-expert/SKILL.md:66-77`, `dockerfile-expert/SKILL.md:86-96`, `async-expert/SKILL.md:111-116` | A reason lets the model apply a rule to cases it does not literally cover |
| Hard rules mapped to a machine check | Several | `observability-expert/SKILL.md:103-121`, `pytorch/SKILL.md:83-107`, `pytest-expert/SKILL.md:123-177` | Separates what is enforced from what is advised |
| Gate before acting, including permission to recommend nothing | A few | `design-patterns-expert/SKILL.md:20-32`, `bash-scripting-specialist/SKILL.md:19-22`, `hook-writer/SKILL.md:16-23` | Counters over-application, the main failure of advice skills |
| Fixed output contract | Several | Verdict first: `building-effective-agents/SKILL.md:12-16`. States what was not verified: `dockerfile-expert/SKILL.md:41-43` | Makes output consistent and gradable |
| "Extend this skill" maintenance recipe | 3 | `pytorch/SKILL.md:160-174`, `transformer-architecture/SKILL.md:375-389` | Tells a future editor which test or table each new fact must update |

### 4.3 Mechanical enforcement

These are the skills' most distinctive habit: turning advice into something that can fail.

| Pattern | Count | Exemplars | Why it likely helps (Judgment) |
| --- | --- | --- | --- |
| Custom audit or lint script with stable rule codes | Common among code skills | `pytest-expert/scripts/audit_tests.py`, `bash-scripting-specialist/scripts/audit_bash.py`, `pytorch/scripts/lint_pytorch.py` | Findings can be cited, suppressed with a reason, and regression-tested |
| Exit codes separating "failed" from "could not check" | Rare: 1 clear case | `pytorch/scripts/verify_api.py:15` | Stops a missing tool reading as a pass. Most scripts use 2 for usage errors only. Counter-example: `hook-writer/scripts/dry_run.py` exits 0 whatever the hooks decide |
| Fixture of deliberate violations that the checker must catch | 8 by search | `observability-expert` (`bad_example.py`), `pytorch/examples/README.md:19-28`, `dockerfile-expert/scripts/tests` | Proves the checker can fail. A known-good fixture also catches false positives |
| Scripts that compute answers so the model does not estimate | Several | `transformer-architecture/scripts/config_budget.py`, `robot-learning-expert/scripts/` | Removes plausible-but-wrong arithmetic |
| Honest-skip contract: missing tools are reported, never silently passed | Several | `bash-scripting-specialist/SKILL.md:40-41`, `negative-space-programming/scripts/test_types.py:49-58` | Prevents false "all clean" reports |
| Build gate that rejects bad output and says how to fix it | A few | `annotate-research-paper/SKILL.md:238` | Gives the agent a loop to iterate on instead of relying on compliance |
| Blind verification by a fresh subagent, with copy-paste briefs | A few | `annotate-research-paper/SKILL.md:247-266`, `documentation-expert/SKILL.md:182-189` | An author reviewing its own work misses what it assumed |

### 4.4 Self-testing skills

| Pattern | Count | Exemplars | Why it likely helps (Judgment) |
| --- | --- | --- | --- |
| Test that checks the skill's own structure | 5 | `pytest-expert/selftest/test_skill_structure.py`, `transformer-architecture/tests/test_skill_structure.py`, `building-effective-agents/tests/test_skill_integrity.py`, `pytorch/tests/test_skill_structure.py`, `robot-learning-expert/tests/test_skill_structure.py` | Cheap protection against drift between separately edited files. Typical checks: paths resolve, references routed, scripts tested, line cap, eval shape |
| Documentation executed as tests | A few | `pytorch/tests/test_walkthroughs.py` | Prose cannot drift from behaviour unnoticed |
| Published or recorded results pinned as tests | A few | `massively-parallel-rl/tests/test_scripts.py:5-6`, `kubernetes-expert` (`pss_server_verdicts.json`), `scientific-computing/scripts/tests/test_verify_api.py:33-64` | Turns "verified against X" from a sentence into a regression test |
| Reference-versus-installed drift checker, with a rule that the script outranks the document | 2 | `pytorch/scripts/verify_api.py`, `scientific-computing/scripts/verify_api.py` | Makes a fast-aging reference detect its own staleness. `design-patterns-expert/scripts/verify_python_claims.py` is a related form that re-checks claims on the current interpreter |
| Tested examples and templates | Several | `design-patterns-expert/tests/test_examples.py`, `pytorch/tests/test_assets.py` | Stops the part users copy from being the part that is broken |

**Limits of structure tests, found in this audit:** `transformer-architecture`'s missed two dangling filenames because its path regex only matches prefixed paths, and `kubernetes-expert`'s "every rule is exercised" test exempts K022 by hand.

### 4.5 Provenance

| Pattern | Count | Exemplars | Why it likely helps (Judgment) |
| --- | --- | --- | --- |
| Versions and dates stated next to the claims that depend on them | At least 11 by search | `pytest-expert/SKILL.md:14-17`, `kubernetes-expert/SKILL.md:114-115` | Makes staleness visible |
| Source file recording how deeply each source was checked | Several | `building-effective-agents/references/sources.md:3`, `pytest-expert/references/sources.md:7-29`, `pytorch/references/sources.md:3-16` | Stops "I saw the abstract" reading as "I verified it" |
| Separating executed knowledge from recalled knowledge | Several | `observability-expert/references/verified-behavior.md:1-8`, `hook-writer/references/gotchas.md:24` | Lets a reader see which claims were observed |
| Dedicated "Not verified" section | 4 | `pytest-expert/AUDIT.md:77-89`, `transformer-architecture/AUDIT.md:43-48` | Stops silent overclaiming |
| Located citations and a marker for inference | A few | `transformer-architecture/SKILL.md:24-26`, `robot-learning-expert/SKILL.md:18-22` | Each claim can be checked at its source |
| Examples marked illustrative, enforced by a test | 4 use the marker | `building-effective-agents/tests/test_skill_integrity.py:66-68` | Stops invented scenarios leaking into answers as facts |
| Graded evidence with admitted weak spots | 1 | `annotate-research-paper/references/evidence.md:11-14` | Honest about extrapolation |
| Live documentation fetched first for fast-moving APIs | 1 | `hook-writer/SKILL.md:25-38` | Avoids trusting training data on a schema that changes |

### 4.6 Evals

| Pattern | Count | Exemplars | Why it likely helps (Judgment) |
| --- | --- | --- | --- |
| Expectations containing specific checkable facts | Several | `transformer-architecture/evals/evals.json` | Grading is reliable when the answer is an exact number |
| `verified_facts` recording the executed truth behind each case | 4 | `async-expert`, `documentation-expert`, `observability-expert`, `scientific-computing` | The grader's key is itself evidence |
| Cases targeting previously found errors | 2+ | `building-effective-agents/evals/evals.json` (`targets` field) | A regression suite for content accuracy |
| A control case where the right answer is "no change" | Several | `async-expert/evals/evals.json`, `observability-expert/evals/evals.json` | Catches over-correction |
| Atomic one-claim graders | 1 | `massively-parallel-rl/evals/*/case.yaml` | Failures are diagnosable |

### 4.7 Records

| Pattern | Count | Exemplars | Why it likely helps (Judgment) |
| --- | --- | --- | --- |
| `AUDIT.md` using claim, evidence and fix tables, with a "Not verified" section | 4 | `pytorch/AUDIT.md`, `pytest-expert/AUDIT.md`, `transformer-architecture/AUDIT.md`, `building-effective-agents/AUDIT.md` | Leaves a trail a reviewer or successor can follow |
| `CHANGELOG.md` with a "verification at release" table | 3 at root | `pytest-expert/CHANGELOG.md`, `transformer-architecture/CHANGELOG.md` | Ties each release to evidence |

## 5. Recommended standard

Not every skill needs every component, so requirements depend on what the skill does. This is a proposal (Judgment).

| Tier | Applies to | Required |
| --- | --- | --- |
| Core | Every skill | `SKILL.md` with valid YAML frontmatter and `name` matching the folder; a description covering what, when and negative scope; a statement of how to verify the work; `CHANGELOG.md`; `evals/` with trigger and output cases |
| Tooling | Ships scripts | Stable rule codes where relevant; exit codes 0, 1 and 2 meaning pass, finding, and could-not-check; a tests folder; a failing fixture and a clean fixture; a run command stated in `SKILL.md` |
| Knowledge | States facts that can age or be wrong | `references/sources.md` recording check depth; dates and versions beside claims; `AUDIT.md` with a "Not verified" section |
| Large | More than one reference or about 150+ lines | A routing table saying when to read each file; a structure test; an "Extend this skill" recipe |
| Optional | When it earns its place | `examples/` (marked illustrative, indexed by a README); `assets/` (tested if users copy them); `INSTALL.md` |

**Machine-specific results.** Any result that could differ between machines states its date, architecture and versions, and conflicting results are kept side by side: "On 2026-11-03 on macOS arm64 with torch 2.15.0: X. On an earlier date, machine not recorded: Y."

### Conventions to settle

| Topic | Today | Proposal |
| --- | --- | --- |
| Test location | `tests/`, `scripts/`, `scripts/tests/`, `assets/tests/`, `selftest/` | `tests/` at the skill root. Drop-in packages may keep their tests beside the package if `SKILL.md` gives the run command |
| Test dependencies | Often documented in prose, which a harness cannot read | A machine-readable form, such as a one-line `uv run --with ...` command |
| Eval format | Five forms (section 6) | One schema |
| Expectation key | `expectations` and `assertions` | One name |
| Eval fixtures | Paths inside prompt text; every `files` array is empty in the 4 `evals.json` files that have the field | Use `files` or remove the field |
| Example location | `examples/`, `assets/examples/` | `examples/` |
| Changelog location | Root and `references/` | Root |
| Frontmatter | 12 skills use only `name` and `description`; 5 add `license`, `compatibility` or `metadata` | Decide where the version and `verified-against` live. `pytorch/SKILL.md` is the fullest example |
| Packaging | Caches shipped in one committed skill's archive, `evals/` left out of another | A package check that unzips into a clean directory and runs the tests |
| Licensing | 4 skills declare MIT; no repo-level or skill-level `LICENSE` file exists | Decide before making the repo public |

### Pitfalls this audit found

- **Counts that drift.** `pytest-expert/INSTALL.md:108` says 31 hard gates where `SKILL.md` lists 34. `observability-expert` says both 43 and 42 tests. Prefer a test that compares such counts, or leave them out.
- **Checks with holes.** See the limits listed in section 4.4.
- **Machine-specific expectations.** `pytorch`'s walkthrough assumes CPU, and its resume bug appears off CPU. Test on more than one machine type, or say which was used.
- **Frontmatter only lenient parsers accept.** `negative-space-programming/SKILL.md:3` has an unquoted `: `, so strict YAML parsing fails, though Claude Code still lists the skill.
- **Dead references.** `transformer-architecture/references/papers/README.md:24` names two files that do not exist.

## 6. Eval formats today

Case counts come from parsing the files (Measured). The notes are Judgment.

| Form | Skills | Cases | Notes |
| --- | --- | --- | --- |
| `evals.json` with `expectations`, `verified_facts` | async-expert, documentation-expert, observability-expert, scientific-computing | 6 each | Strongest ground truth |
| `evals.json` with `expectations`, `files` | pytest-expert (17), pytorch (17), transformer-architecture (16) | 50 | `files` is empty in every case |
| `evals.json` with `expectations` only | robot-learning-expert | 10 | |
| `evals.json` with `assertions`, `targets`, `name` | building-effective-agents | 20 | A/B protocol described in a note, with no runner |
| `case.yaml` per case, with graders | massively-parallel-rl | 3 | |

That is 107 structured cases across 10 skills. Missing entirely: annotate-research-paper, bash-scripting-specialist, design-patterns-expert (its `evals/` was deliberately left out of the archive, per `tests/test_grade.py:6`), dockerfile-expert, hook-writer, kubernetes-expert, negative-space-programming. A search of the committed skills found no trigger or should-not-trigger cases.

## 7. Per-skill update list

Priorities are recommendations (Judgment). Skills are in suggested working order. "Problems" includes confirmed defects and undetermined failures, and says which is which. Every skill also needs trigger cases, since none has them.

| Skill | Problems to resolve | Components to add |
| --- | --- | --- |
| pytorch | **High, confirmed:** `--resume` crash off CPU (`assets/train_template.py:203`, `:210`). Low: walkthrough output is machine-specific; `CHANGELOG.md:60-62` 1.1.0 is a stub | - |
| massively-parallel-rl | **Med, cause undetermined:** `test_pqn_learns_cartpole` did not pass here. Low: "22 repos" against 20-21 in `papers.md`; adapted Apache-2.0 code has attribution but no NOTICE | CHANGELOG, AUDIT.md, machine-readable test dependencies |
| negative-space-programming | **Med, confirmed:** invalid strict YAML in the frontmatter (`SKILL.md:3`), which any future conformance linter would reject | Evals, AUDIT.md, negative scope |
| hook-writer | Med: `hookspec.py`, used by `check_hooks.py` and `dry_run.py`, hardcodes facts from 2026-10-25 with no drift check. The fetch-first rule covers the agent, not these scripts. Low: `dry_run.py` exits 0 whatever the hooks decide | **Tests (about 1,100 lines of untested logic)**, evals, AUDIT.md, negative scope |
| annotate-research-paper | Low: no dependency manifest | **Tests (about 3,100 lines of untested logic)**, evals, CHANGELOG, example inputs |
| transformer-architecture | Med, confirmed: `references/papers/README.md:24` names two files that do not exist, and the structure test misses it | Negative scope |
| design-patterns-expert | Low: tests expect an `evals/` folder deliberately left out of the archive; find it and restore it for this repo. Low: "type-checked" claim depends on optional tools | Evals, CHANGELOG, negative scope |
| dockerfile-expert | Low: `scripts/verify_image.sh:3` documents an argument the script ignores | **Evals (sibling boundary with kubernetes-expert is unmeasured)**, tests for the verifier |
| kubernetes-expert | Low: K022 is exempted from "every rule is exercised" (`scripts/tests/test_check_manifests.py:40`); description oversells the GPU Job template, whose GPU lines are commented out | Evals, CHANGELOG |
| bash-scripting-specialist | None found | Evals, AUDIT.md |
| documentation-expert | Low: "tested on Python 3.10-3.14" has no matrix behind it | CHANGELOG, AUDIT.md |
| async-expert | Low: stale `/home/claude` path in the evals note | CHANGELOG, AUDIT.md, negative scope |
| observability-expert | Low: 43 versus 42 test count; `INSTALL.md:26` refers to an absent `.skill` bundle; `evals/files/django_queue/` is unused by any case | CHANGELOG, negative scope |
| scientific-computing | Low: "No network access required" holds for the scripts but not the tests | CHANGELOG, AUDIT.md |
| building-effective-agents | Low: hardcoded `/tmp` log path in `scripts/inspect_eval_template.py:142` | CHANGELOG, eval runner |
| robot-learning-expert | Low, privacy: the owner's full name is in `references/papers/massively-parallel-survey.md:3` | Move CHANGELOG to root, AUDIT.md, negative scope |
| pytest-expert | Low: `.ruff_cache/` in the archive (excluded from this repo); `INSTALL.md:108` stale count; `INSTALL.md:19` refers to an absent `.skill` file | Negative scope |

Three skills (`massively-parallel-rl`, `pytorch`, `scientific-computing`) carry the owner's GitHub handle as `metadata.author`, which is normal attribution.

### Held-back skills

Reported so nothing is lost. These are not committed.

- **lesson-design:** the Oct 4 `SKILL.md` scores 12 of 20 on `evals/scripts/coverage.py`, and the Sep 29 `SKILL.md` (from the earlier archive, in `~/.Trash/lesson-design.zip`) scores 18 of 20 (Measured). Two failures in both are checks on `accessible-course-materials`, so for `lesson-design` alone that is 12 of 18 versus 18 of 18. The checks match literal strings, so some of the drop is rewording. For example, finger voting remains but is narrowed to specific misconceptions (`SKILL.md:131`, `:247`). `grade.py` still expects formats `SKILL.md` never defines (`[task]` headings, `Click N, beside Line`), so the evals predate the current text. `SKILL.md:14` points to `week_1/` and `week_2/` as reference examples, and they are not bundled.
- **accessible-course-materials:** hardcoded personal contact details and caches in the archive. It has no evals, CHANGELOG, AUDIT.md or references, and its suite was not run because veraPDF is not installed.
- **find-docs:** no defects above Low. It would benefit from trigger evals because its trigger is broad.
- **study-notes:** 63 tests passed but pytest exited 1, because `tests/test_cards.py:206` never closes a `sqlite3` connection and `pyproject.toml` sets `filterwarnings = ["error"]`. `references/evidence.md:59` and `:288` contradict `SKILL.md:18-20`, and its handoff names `library-reference` and `cliff-notes` are in neither this repo nor the current install. It also hardcodes a named person's personal details, so review that before publishing it anywhere.

## 8. Outline for a skill builder

### Create mode

1. **Scope.** State the job, the trigger phrases, the negative scope and the sibling skills. Decide which tiers from section 5 apply.
2. **Evals first.** Write trigger and should-not-trigger prompts, then output cases with checkable expectations, before writing the skill.
3. **Ground truth.** Collect primary sources and installed versions, and record how deeply each was checked. Run what can be run.
4. **Draft.** Write a short `SKILL.md`, route depth to references, attach a reason to each rule, and end each workflow step with a "Done when".
5. **Mechanize.** Turn every rule that can be checked into a script with stable codes and exit codes, with failing and clean fixtures.
6. **Self-test.** Add a structure test and test every script, template and example.
7. **Independent check.** Have a fresh agent fact-check the claims, and record the findings in `AUDIT.md`.
8. **Evaluate.** Run the evals with and without the skill, and record the results.
9. **Package.** Unzip into a clean directory, run the tests there, and confirm no caches shipped and nothing was lost.

### Update mode

1. Run a conformance check against the tier requirements and report the gaps.
2. Run the existing tests on the current environment, before changing anything.
3. Recheck dated claims against current versions.
4. Fix, add a regression test or eval for each fix, and update CHANGELOG and AUDIT.md.

### Assets the builder would need

- A conformance linter. The checks already exist in pieces across the five structure tests.
- A shared structure test, parameterized by tier.
- Templates for `SKILL.md`, `CHANGELOG.md`, `AUDIT.md`, `references/sources.md` and `evals/evals.json`.
- A single eval schema.
- A packaging script with a clean-room test run.

## 9. Open decisions

1. Do the tiers in section 5 match what you want, particularly requiring evals and a CHANGELOG for every skill?
2. Should existing tests move to `tests/`, or should skills only declare a run command?
