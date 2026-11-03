# My Skills

Claude Code skills I develop and maintain. Each folder is a self-contained skill with a `SKILL.md` at its root, plus whatever references, scripts, tests, assets, examples and evals it needs.

The patterns these skills share, a proposed standard, and what each skill still needs are in [docs/SKILL_AUDIT.md](docs/SKILL_AUDIT.md). A plan for measuring them is in [docs/EVAL_FRAMEWORK.md](docs/EVAL_FRAMEWORK.md).

## Skills

### Python and software engineering

| Skill | What it does |
| --- | --- |
| [async-expert](async-expert/) | Writes, reviews and debugs concurrent Python (asyncio, httpx, anyio, FastAPI), with numbered rules, an AST audit script and drop-in helpers. |
| [design-patterns-expert](design-patterns-expert/) | Helps choose, or decline, a design pattern for Python code, with runnable examples and a script that checks the skill's version-dependent claims. |
| [documentation-expert](documentation-expert/) | Writes and audits READMEs, API references and docstrings, using scripts that check doc claims against the source. |
| [negative-space-programming](negative-space-programming/) | Adds runtime checks for what must never happen (assertions, bounded loops, the split between programmer and operating errors), with an audit script. |
| [observability-expert](observability-expert/) | Sets up Python logging, structured JSON logs, correlation IDs and Weights & Biases tracking, with a 17-check audit script and a drop-in `obslog` package. |
| [pytest-expert](pytest-expert/) | Builds, audits and hardens pytest suites, with scripts that catch tests that cannot fail and tests weakened to get green, plus mutation, flaky-test, CI and hook tooling. |

### Infrastructure and tooling

| Skill | What it does |
| --- | --- |
| [bash-scripting-specialist](bash-scripting-specialist/) | Writes, reviews and ports bash scripts, with a strict-mode template, a bats suite, a custom audit script and a multi-version test runner. |
| [dockerfile-expert](dockerfile-expert/) | Writes and reviews Dockerfiles and `.dockerignore` files, with Python, Node and Go templates, a linter and an image verifier. |
| [hook-writer](hook-writer/) | Designs, writes and debugs Claude Code hooks, starting from the live hooks reference, with a linter, a dry-run tool and handler templates. |
| [kubernetes-expert](kubernetes-expert/) | Writes and reviews Kubernetes manifests, with a kustomize base, a Job template and a policy checker tested against recorded API-server verdicts. |

### Machine learning and scientific computing

| Skill | What it does |
| --- | --- |
| [massively-parallel-rl](massively-parallel-rl/) | Scales RL to thousands or millions of vectorized environments, with runnable JAX scaffolds, scaling-law tools and tests pinned to published results. |
| [pytorch](pytorch/) | PyTorch 2.15 engineering reference with a linter, a training triage tool, memory sizing and an API drift checker. |
| [robot-learning-expert](robot-learning-expert/) | Explains RL concepts and helps design and debug robot policies, using a routing `SKILL.md`, paper digests and tested numerical oracle scripts. |
| [scientific-computing](scientific-computing/) | NumPy and SciPy engineering reference with a symptom-indexed debugging guide and a version-drift checker. |
| [transformer-architecture](transformer-architecture/) | Sizes, implements and debugs decoder-only transformer LLMs, with exact-count budget scripts, property tests and paper digests. |

### Agents, research and skills

| Skill | What it does |
| --- | --- |
| [annotate-research-paper](annotate-research-paper/) | Writes expert margin notes, highlights and critique onto a paper's PDF through a build-gated pipeline that splits the work across subagents. |
| [skill-builder](skill-builder/) | Creates new skills and brings existing ones up to standard, with a conformance checker, templates and a packaging check that tests the unpacked zip. |
| [building-effective-agents](building-effective-agents/) | Diagnoses and designs LLM agents and their harnesses, routing each symptom to a starting reference, with mostly stdlib scripts and an A/B eval set. |

## Using a skill

From the repo root, copy a skill's folder into `~/.claude/skills/`, or into `.claude/skills/` in a project:

```bash
mkdir -p ~/.claude/skills
cp -R pytest-expert ~/.claude/skills/
```

Claude Code picks up the skill from its `SKILL.md`. Some skills need extra tools, which they list in their own `SKILL.md` or `INSTALL.md`.

## Running a skill's tests

Test locations and dependencies differ between skills. [Section 3 of the audit](docs/SKILL_AUDIT.md#3-test-runs) lists the path for each one and its result at the time of the audit. For example:

```bash
cd transformer-architecture
uv run --no-project --with pytest --with torch --with numpy --with scipy python -m pytest
```

## Repository layout

Source `.zip` archives, tool caches and skills kept out of this repo are excluded by [.gitignore](.gitignore).
