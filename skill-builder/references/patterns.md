# Patterns worth copying

Patterns found across the skills in the `my-skills` repository, with the place to look. Prevalence figures and evidence are in `docs/SKILL_AUDIT.md` in that repository. These are inferred from which skills use them; no outcome data shows that any one of them improves results.

If this skill was copied out of that repository, the paths below will not resolve from here.

## Structure

| Pattern | Look at |
|---|---|
| Short `SKILL.md` routing to references by situation | `robot-learning-expert/SKILL.md` (102 lines, 37 references) |
| Symptom-indexed routing | `building-effective-agents/SKILL.md`, `scientific-computing/SKILL.md` |
| A "Done when" on each workflow step | `pytest-expert/SKILL.md` |
| A gate before acting, where doing nothing is a valid answer | `design-patterns-expert/SKILL.md`, `hook-writer/SKILL.md` |
| Fixed output contract, with what was not verified | `building-effective-agents/SKILL.md`, `dockerfile-expert/SKILL.md` |
| "Extend this skill" recipe | `pytorch/SKILL.md`, `transformer-architecture/SKILL.md` |

## Enforcement

| Pattern | Look at |
|---|---|
| Audit script with stable rule codes | `pytest-expert/scripts/audit_tests.py`, `bash-scripting-specialist/scripts/audit_bash.py` |
| Exit code separating "could not check" from "failed" | `pytorch/scripts/verify_api.py` |
| Fixture of deliberate violations | `observability-expert/assets/examples/bad_example.py`, `dockerfile-expert/scripts/tests/` |
| Scripts that compute so the model does not estimate | `transformer-architecture/scripts/config_budget.py` |
| Missing tools reported, never passed silently | `bash-scripting-specialist/scripts/check.sh`, `negative-space-programming/scripts/test_types.py` |
| Build gate that says how to fix what it rejects | `annotate-research-paper/scripts/annotate.py` |
| Blind review by a fresh agent, with a copy-paste brief | `annotate-research-paper/SKILL.md`, `documentation-expert/SKILL.md` |

## Self-testing

| Pattern | Look at |
|---|---|
| A test of the skill's own structure | `pytest-expert/selftest/test_skill_structure.py`, `transformer-architecture/tests/test_skill_structure.py` |
| Documentation executed as tests | `pytorch/tests/test_walkthroughs.py` |
| Recorded ground truth compared in tests | `kubernetes-expert/scripts/tests/pss_server_verdicts.json` |
| Drift checker against what is installed | `pytorch/scripts/verify_api.py`, `scientific-computing/scripts/verify_api.py` |
| Live drift check against a dated table | `hook-writer/tests/test_hookspec.py` |
| Tested templates and examples | `design-patterns-expert/tests/test_examples.py`, `pytorch/tests/test_assets.py` |

## Provenance and evals

| Pattern | Look at |
|---|---|
| Source table with check depth | `building-effective-agents/references/sources.md` |
| Executed knowledge kept apart from recalled knowledge | `observability-expert/references/verified-behavior.md` |
| Examples marked illustrative, enforced by a test | `building-effective-agents/tests/test_skill_integrity.py` |
| Evals with executed ground truth | `async-expert/evals/evals.json` (`verified_facts`) |
| Evals targeting past errors | `building-effective-agents/evals/evals.json` (`targets`). It uses `assertions` where the standard uses `expectations`, so `check_skill.py` reports SB011 on it |

## Limits to expect

A structure test is only as good as its patterns. In one skill a path regex missed bare filenames, and in another a test named "every rule is exercised" exempted a rule by hand. A green structure test does not prove there are no dead references.
