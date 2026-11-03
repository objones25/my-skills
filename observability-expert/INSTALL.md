# Installing observability-expert

Two things ship here: a skill Claude loads, and a small package plus audit script you can
use on their own.

## Claude Code (a skill Claude loads automatically)

```bash
# available in every project
cp -r observability-expert ~/.claude/skills/

# available only in this repo, and committable so a team shares it
mkdir -p .claude/skills && cp -r observability-expert .claude/skills/
```

Restart Claude Code, or run `/doctor` to confirm it loaded. Claude picks it up on any
request about logging setup, structured logs, replacing prints, request ids, instrumenting
a training loop, or W&B.

```bash
ls ~/.claude/skills/observability-expert/SKILL.md   # confirm it registered
```

## Claude Cowork / claude.ai

Use the `observability-expert.skill` file: its card in the conversation has a **Save
skill** button when your organization allows it.

## Using the tooling directly, without Claude

The audit is stdlib-only; nothing here needs Claude:

```bash
python observability-expert/scripts/audit_observability.py src/
python observability-expert/scripts/audit_observability.py --list-codes .
python observability-expert/scripts/audit_observability.py src/ --format json
python observability-expert/scripts/audit_observability.py a.py b.py   # several paths
```

Wire the package into a project:

```bash
# 1. the package (copy, or add assets/ to your path)
cp -r observability-expert/assets/obslog your_project/

# 2. configure once, in each entry point
#    from obslog.setup import configure, install_excepthook
#    configure(level="INFO", fmt="json", service="your-service")
#    install_excepthook()

# 3. run the gate
python observability-expert/scripts/audit_observability.py your_project/ \
    --entrypoint 'your_project/jobs/*.py'

# 4. (optional) the plumbing's own tests, to confirm it behaves on your Python
pip install pytest wandb
PYTHONPATH=observability-expert/assets python -m pytest \
    observability-expert/assets/tests -q
```

### Environment variables `configure()` reads

| Variable | Effect |
|---|---|
| `LOG_LEVEL` | Console handler level. Default `INFO`. |
| `LOG_FORMAT` | `console` (human) or `json` (JSON Lines). Default `console`. |
| `LOG_FILE` | Also write a rotating JSON file at this path. |
| `LOG_QUEUE` | `0` disables the background-thread queue. Default on (Python 3.12+). |
| `SERVICE_NAME` / `SERVICE_VERSION` / `ENVIRONMENT` | Stamped on every JSON record. |

W&B reads its own: `WANDB_API_KEY`, `WANDB_MODE`, `WANDB_PROJECT`, `WANDB_ENTITY`,
`WANDB_DIR`, `WANDB_RUN_ID`, `WANDB_BASE_URL` (since 0.31.0 the default server is
`https://api.forge.coreweave.com`; keys stored for `api.wandb.ai` still work).

### Adjust before committing

- The `library_levels` dict in `obslog/setup.py`: add the libraries that are noisy in
  your stack, remove the ones you do not use.
- `extra_keys` on `RedactingFilter` for field names specific to your domain.
- `--entrypoint` globs for any module that legitimately configures logging or prints.
- On Python 3.9-3.11 call `configure_queue_manually([...])` if you want the queue; the
  dictConfig auto-wiring is 3.12+.

### Pre-commit hook

```yaml
- repo: local
  hooks:
    - id: observability-audit
      name: observability audit
      entry: python observability-expert/scripts/audit_observability.py
      language: system
      types: [python]
      pass_filenames: true
```

## What is in the box

```
observability-expert/
├── SKILL.md                        workflow + the 17 hard rules
├── INSTALL.md                      this file
├── references/
│   ├── rules.md                    every rule with rationale and sources
│   ├── stdlib-logging.md           pipeline, dictConfig, context, queues, testing
│   ├── wandb.md                    runs, config, metrics, artifacts, sweeps, offline
│   ├── services.md                 request-scoped logging and framework wiring
│   ├── ml-training.md              what to log from a training loop, and how often
│   └── verified-behavior.md        the surprising behaviors, with real output
├── assets/
│   ├── obslog/                     setup · json_formatter · context · redact · wandb_run
│   ├── examples/                   train_example · service_example · bad_example
│   └── tests/test_obslog.py        43 tests
└── scripts/audit_observability.py  17-check AST audit; exit 1 on any finding
```

## Verified against

Python 3.11.17, 3.12.13, 3.13.16 and 3.14.6; wandb 0.31.0 (and 0.28.2). October 2026.

API signatures were introspected from the installed packages rather than recalled, both
examples run clean, the test suite passes on 3.11 to 3.14 (and 3.15.0b4), and each of the
17 audit codes was confirmed to fire exactly once on `assets/examples/bad_example.py` and
not at all on the shipped code.
