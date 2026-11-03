# Sources

Checked means how much was read.

| Source | Date checked | Checked | Supports |
|---|---|---|---|
| https://code.claude.com/docs/en/skills.md | 2026-11-03 | Fetched by a research agent; summary used | 500-line guidance, 1,536-character listing truncation, listing budget, malformed YAML behaviour |
| https://code.claude.com/docs/en/plugin-evals.md | 2026-11-03 | Fetched by a research agent; summary used | `claude plugin eval` layout and its plugin requirement |
| skill-creator plugin, `references/schemas.md` and `scripts/quick_validate.py` | 2026-11-03 | Read locally | `evals.json` and trigger formats, 64-character name and 1024-character description limits, angle-bracket rule |
| superpowers `writing-skills` skill, version 7.0.0 | 2026-11-03 | Read locally | Advice to leave the workflow out of descriptions, and to get a baseline first. Its testing is that skill's own report, not reproduced here |
| `docs/SKILL_AUDIT.md` in the `my-skills` repository | 2026-11-03 | Written and reviewed for this work | Tiers, patterns, pitfalls |

## Observed, not documented

On 2026-11-03, `negative-space-programming` appeared in the Claude Code skill listing with its description, while its frontmatter failed a strict YAML parse. The docs say malformed YAML loads a skill with empty metadata, so Claude Code's parser appears more lenient than `yaml.safe_load`. Machine: macOS arm64. Claude Code version not recorded. `check_skill.py` still reports this as SB002, because other tools may be stricter.
