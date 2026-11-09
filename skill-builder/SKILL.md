---
name: skill-builder
description: "Create a new Claude Code skill, or bring an existing one up to standard, with evals, tests, a changelog and a conformance check. Use when the user asks to build, write, scaffold, audit, update, harden or fix a skill, says a skill is stale or will not trigger, or asks what a skill is missing. Not for tuning a description or benchmarking through repeated runs; use skill-creator."
metadata:
  version: 0.2.0
  test-command: "uv run --no-project --with pytest --with pyyaml python -m pytest tests"
---

# Skill builder

A skill is finished when its claims have been checked, not when it reads well. Agents working without this skill tend to write good instructions and scripts, then leave out what makes a skill maintainable: evals, trigger cases, a changelog, negative scope, and dated results. This workflow builds those in.

`<this-skill>` below means this skill's folder.

## Pick the mode

| The user has | Mode |
|---|---|
| An idea, and no skill yet | Create |
| An existing skill to audit, fix, or update | Update |
| A description that fires wrongly, or two versions to compare | Hand off to `skill-creator`, which runs that loop |
| One small correction and no wider request | Make it test-first, add a CHANGELOG line, and stop |

## Create mode

1. **Scope it.** Write down the job, the user's own trigger phrases, nearby requests that belong elsewhere, and sibling skills. Choose tiers with `references/standard.md`.
   *Done when:* you can name what the skill is not for.

2. **Scaffold it.**
   ```bash
   python3 <this-skill>/scripts/new_skill.py NAME --dest DIR [--tier tooling] [--tier knowledge]
   ```
   *Done when:* the folder exists and `check_skill.py` reports no errors. It warns (SB014) until each `TODO` is gone, and packaging refuses while any remain.

3. **Write the evals first.** Fill in `evals/trigger_evals.json` with at least 8 cases that should trigger and 8 near misses that should not. Fill in `evals/evals.json` with cases whose expectations are specific facts, including one control where the right answer is to change nothing. Read `references/evals.md`.
   *Done when:* neither file has a `TODO` left.

4. **Collect the ground truth.** Install what the skill is about and run it. Record each source and how much of it you read in `references/sources.md`. Read `references/provenance.md`.
   *Done when:* every fact you plan to state has a command or source behind it.

5. **Write `SKILL.md`.** Use the template's sections. Give each rule its reason, end each workflow step with a "Done when", and move depth into `references/` behind a table saying when to read each file. Read `references/description.md` for the description.
   *Done when:* a reader can tell from `SKILL.md` alone what to load for a given task.

6. **Make rules executable.** If a rule can be checked by a script, write the test first, including a fixture the script must flag and a clean one it must pass. Use exit code 0 for pass, 1 for findings, and 2 for could not check. See `references/patterns.md`.
   *Done when:* each script has a test that you watched fail first, and `metadata.test-command` runs them all.

7. **Turn claims into tests.** Any sentence a skill says about itself ("audits clean", "tested with X", "17 checks") becomes a test or gets removed.
   *Done when:* no self-claim depends on someone having once checked it.

8. **Get an independent review, once the tests pass.** Give a fresh agent that wrote nothing the skill's files, and ask it to break the claims. Wait for its full report. Check each finding yourself, since reviewers are sometimes wrong. Record the results in `AUDIT.md`, including what was not verified.
   *Done when:* every finding is fixed or rejected with a reason.

9. **Evaluate and record.** Run the evals with and without the skill, following `references/evals.md`, on the reviewed version. Add the results to the CHANGELOG with the output of `machine_stamp.py`.
   *Done when:* the CHANGELOG states the results with date and machine, or says that evals were not run.

10. **Run the final gate last.** Any change after this step means running it again.
    ```bash
    python3 <this-skill>/scripts/check_skill.py SKILL_DIR --strict
    python3 <this-skill>/scripts/package_skill.py SKILL_DIR --out DIR
    ```
    *Done when:* both exit 0, and the packager says the tests passed in the unpacked copy. If it says "tests not run", the skill has no `metadata.test-command`, so say that in your report. Do not report the skill as finished before this.

## Update mode

1. **Measure before changing anything.**
   ```bash
   python3 <this-skill>/scripts/check_skill.py SKILL_DIR
   ```
   Then run the skill's tests with every optional tool installed, and read each skip reason. A skip hides whatever it skips: in one audited skill, installing the optional tools exposed a failing test.
   *Done when:* you have a list of findings, failures and skips, with today's stamp from `machine_stamp.py`.

2. **Run its promises.** If the `audit-skill` skill is installed, run it on this skill (it is read-only) and work from its report: its verdicts, quotes and coverage line replace your own claim-checking, and the items it did not check stay "not verified". Without it, search the skill for claims about itself (counts, "clean", "tested on", "verified") and run each one, and recheck facts that depend on a version against what is installed.
   *Done when:* each claim is confirmed, corrected, or marked not verified, and the audit report's path (if one was made) is recorded in `AUDIT.md`.

3. **Rank the problems.** Handle broken behaviour first, then wrong claims, then missing components. Ask before adding a component that would not earn its place: `references/standard.md` says which do.
   *Done when:* the user has seen the list, unless they already asked you to fix everything.

4. **Fix test-first.** For each defect, write a test, watch it fail for the right reason, fix it, and run the whole suite. If a fix already exists, undo it temporarily to see the test fail. Do not weaken a test until you understand why it fails. If the test is what's wrong, say so and show the evidence.
   *Done when:* the full suite passes and each fix has a test or eval.

5. **Record it.** Update the CHANGELOG and `AUDIT.md`. Keep earlier results beside new ones, each labelled with its date and machine. Write "machine not recorded" when you do not know; never guess.
   *Done when:* `check_skill.py --strict` and `package_skill.py` have been run after the last change, and each remaining warning has a stated reason.

## Rules

- **Check, do not assume.** `check_skill.py` catches what is mechanical. It cannot tell whether the skill is good, so passing it is necessary and not sufficient.
- **State machine-specific results as such.** Results that can differ between machines carry a date, an architecture and versions, because a bare "375 passed" cannot be compared with anything. Use `machine_stamp.py` for the wording.
- **Add components only when they earn their place.** A checklist item with nothing in it is worse than an absent one, because it looks like coverage.
- **Report what was not done.** If evals were not run, say so.

## Reference files

| Read | When |
|---|---|
| `references/standard.md` | Choosing tiers and conventions, or looking up a checker code |
| `references/description.md` | Writing or fixing a description |
| `references/evals.md` | Writing or running evals |
| `references/provenance.md` | Recording sources, or stating results and self-claims |
| `references/patterns.md` | Looking for a proven way to do something, with examples to copy |
| `references/sources.md` | Checking where this skill's facts came from |
| `examples/README.md` | Seeing what a finished run looks like |

## Scripts

| Script | Purpose |
|---|---|
| `scripts/check_skill.py` | Conformance check: `--json`, `--strict`, `--ignore CODE`, `--list-codes` |
| `scripts/new_skill.py` | Scaffold a skill from `assets/templates/` |
| `scripts/package_skill.py` | Zip without junk, then test the unpacked copy |
| `scripts/machine_stamp.py` | Print the date, machine and versions for a result |

## Extend this skill

1. A new checker rule: add a failing test in `tests/test_check_skill.py`, add the rule to `RULES`, add its fix to `references/standard.md` (a test requires this), and run it on real skills to look for false positives.
2. A new pattern: add it to `references/patterns.md` with a path to copy from.
3. A new fact about Claude Code or skill-creator: add it to `references/sources.md` with the date checked.
4. After any change: run the final gate from Create mode, and add the result to the CHANGELOG.

## Reporting

Lead with the state of the skill: what passes, what does not, and what was not checked. Then list what changed. Give the checker's error and warning counts, and the test result with its stamp.
