# 01: Support agent reports actions that didn't happen

*Illustrative system.* A ticketing agent sometimes tells users "I've opened ticket T-3 for you" when no such ticket exists. The team has added "never claim a ticket was created unless it was" to the prompt three times.

## Verdict

Stop editing the prompt. This is a harness problem: nothing checks the claim against what the tools returned. A prompt is a request; a check in code is a constraint.

## Candidate causes

All look identical from the user's side, and each needs a different fix (`verification.md`):

1. The model narrated a call it never made.
2. The call ran and failed inside a 200 response (GraphQL-style `errors` array), and the client only checked the status code.
3. The ticket exists under a different id than the one the model reported.
4. A timeout triggered a retry, so the action ran twice or half-applied.
5. A lookup returned nothing and the model filled the gap.

## Cheapest test

Pull the tool-call log for a sample of failing runs and sort them. With logs in the shape `scripts/triage_transcripts.py` expects:

```
$ python scripts/triage_transcripts.py tests/fixtures/false_success_runs.jsonl
never_called       1   e.g. r1
silent_failure     1   e.g. r2
wrong_id           1   e.g. r3
partial_retry      1   e.g. r4
empty_result       1   e.g. r5
verified           1   e.g. r6
```

The fixture has one run per bucket for demonstration. On real data, one bucket usually dominates, and that decides the fix.

## Fixes by bucket

- `never_called`: the final answer may only cite tool results. Give the agent a `finish` tool whose `evidence` field must list tool-call ids, and reject "done" whose evidence isn't a successful call (`scripts/agent_loop.py`, `evidence_exists`).
- `silent_failure`: parse response bodies and raise on error payloads before the model sees "success".
- `wrong_id`: render ids from the API response into the reply template; never let the model compose them.
- `partial_retry`: idempotency keys on create calls, then reconcile duplicates.
- `empty_result`: a sanctioned `could_not_complete` outcome the harness treats as normal, so "not found" is an acceptable answer.

## Confirm the fix

Turn the failing transcripts into eval tasks, include a "ticket can't be created" negative case, and re-run several trials per task before and after (`evals.md`). `scripts/inspect_eval_template.py` shows a scorer that requires both the right answer and the tool call that supports it.
