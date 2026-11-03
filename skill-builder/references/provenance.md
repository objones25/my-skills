# Provenance: sources, results and self-claims

A reader should be able to tell which claims were observed, which came from a source, and which are judgment.

## Sources

Keep `references/sources.md` with this table. "Checked" states how much was read.

| Source | Date checked | Checked | Supports |
|---|---|---|---|
| https://example.org/paper | 2026-11-02 | Abstract only | The headline claim, not its numbers |
| torch 2.15.0 | 2026-11-02 | Installed, via `inspect` | Every signature in `references/api.md` |

Use the installed package ahead of the docs and the docs ahead of memory. When they disagree, follow the code and say so.

## Machine-specific results

A result is machine-specific if it could differ on another computer: test counts, timings, sizes, throughput, or anything touching hardware. API facts and arithmetic are not.

Get the wording from `machine_stamp.py --package NAME`, and keep differing results side by side:

| Date | Machine | Versions | Result |
|---|---|---|---|
| 2026-11-03 | macOS arm64 (Apple Silicon) | torch 2.15.0 | 375 passed, 4 skipped |
| 2026-11-02 | Not recorded | torch 2.15.0 | 375 passed, 1 skipped |

Rules:

- **Do not guess.** If the machine is unknown, write "machine not recorded".
- **Do not delete an old result** just because a new one differs. The difference may be the finding.
- **Keep expected output honest.** If a documented output depends on the machine, say which machine produced it, and make any test that compares it aware of that. `pytorch`'s `cpu-output` fence is an example.

## Self-claims

Claims a skill makes about itself go stale fastest: "17 checks", "audits clean", "tested on 3.10 to 3.14". Handle each one in the first of these ways that works:

1. Delete the number if nothing depends on it.
2. Add a test that fails if the claim stops being true.
3. State it with a date and machine, as a record of one run.

## Labelling

| Mark | Meaning |
|---|---|
| Located tag, such as `[Paper §3]` | Checkable at that location |
| `[inference]` or "Judgment" | Reasoned, not sourced |
| "Not verified" | Looked for, could not confirm |
| "Observed, not documented" | Seen happening, with no source saying it should |

## `AUDIT.md`

Write this from an independent review. Sections: errors found (claim, evidence, fix), how things were checked, and not verified. The "Not verified" section is the most useful part for a later reader.
