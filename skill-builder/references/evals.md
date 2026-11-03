# Evals

Evals answer two separate questions: does the skill load when it should (triggering), and does loading it make the work better (output quality)? A skill can pass one and fail the other, so write both.

## Formats

Use skill-creator's formats, so its tools can run these files without conversion.

### `evals/trigger_evals.json`

```json
[
  {"query": "my pytest suite fails randomly in CI", "should_trigger": true},
  {"query": "write a bash script to rotate logs", "should_trigger": false}
]
```

Write at least 8 of each kind. Queries should sound like real requests, with concrete details. Should-not-trigger cases should be near misses that share vocabulary with the skill, since obviously unrelated requests test nothing.

### `evals/evals.json`

```json
{
  "skill_name": "my-skill",
  "evals": [
    {
      "id": 1,
      "prompt": "A realistic request",
      "expected_output": "What a good result looks like",
      "files": ["evals/files/sample.py"],
      "expectations": ["A statement a grader can check against the output"]
    }
  ]
}
```

Optional fields that have proved useful in existing skills:

- `verified_facts`: the executed result behind the expectations, so the key is itself evidence.
- `targets`: the audit finding this case guards against.

## Writing good cases

- **Check facts:** "states the total as 7,298,617,344" beats "gives the right parameter count".
- **Include a control:** a case where the right answer is that nothing is wrong. This catches over-correction.
- **Turn each bug into a case:** when the skill gets something wrong, add a case first and then fix it.
- **Keep one claim per expectation,** so a failure says what failed.
- **Put fixtures in `evals/files/`,** and list them in `files`.

## Running them

| Goal | Tool |
|---|---|
| Trigger rates and description tuning | skill-creator's `run_loop.py` and `run_eval.py`, which read `{query, should_trigger}` |
| With-skill and without-skill comparison, using `evals.json` | skill-creator's benchmark workflow |
| Isolated runs with graders, in a plugin | `claude plugin eval`. Output-quality use is untested here. It uses a folder per case, holding a prompt file, a `case.yaml`, or both, and needs a plugin, and its format is not interchangeable with skill-creator's |
| A quick check with no other tools | Run each prompt in a fresh subagent, once with the skill and once without, and grade the expectations yourself |

### Settings that change the answer

Seen on one skill on 2026-11-03 on macOS arm64 with Claude Code 2.1.317. Small samples.

- **`run_eval.py` tests the description, not the skill.** It writes a stub command holding only the description. Do not have the real skill installed while running it, or the two compete.
- **Run it with `--num-workers 1`, from a throwaway project that has its own `.claude` folder.** With 4 workers, 4 of 17 should-trigger runs fired. With 1 worker, 42 of 47 did. Its default is 10 workers. It also writes temporary commands to the nearest `.claude` folder, which can be `~/.claude`.
- **Errors count as misses.** It discards stderr and scores errors and timeouts as not triggered, so a passing should-not-trigger result does not prove that the runs worked. Check that a known-good query fires.
- **Expect noise.** At 1 worker, one query fired 0 of 2 in one run and 3 of 3 in others.
- **Grant `claude plugin eval` the tools the skill needs, and keep the transcripts.** Its default grants exclude shell and write tools. In one case the with-skill runs scored below the baseline, and the cause could not be found once the traces were gone.
- **Its regex graders do not accept `(?i)`.** Use `flags: i`.
- **Point `tool_used: Skill` at your skill with `input_match`,** otherwise any skill counts.

See `docs/EVAL_FRAMEWORK.md` in the `my-skills` repository for the full comparison.

Fresh agents matter: an agent that wrote the skill already knows the answers. Grade from the output, and record the results in the CHANGELOG with the output of `machine_stamp.py`. If the evals were not run, the CHANGELOG says so.

The one-machine scope and the sample size belong in the record. Three runs of a case show direction, and do not prove an improvement.
