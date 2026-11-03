# 03: "We hit 85%. Can we ship? How many runs do we need?"

*Illustrative system.* A refund agent passes 85% of a 30-task internal suite on one run.

## Verdict

One run at 85% doesn't support a reliability claim. Measure consistency (pass^k) on repeated trials, and decide what "production-ready" means in those terms before looking at the number.

## Why one run misleads

pass@k is "at least one of k attempts works"; pass^k is "all k work". A customer who asks the same thing twice expects the same answer, so pass^k is the honest number for user-facing claims. Take one task that passed 4 of 5 trials:

```
$ python scripts/reliability.py passk --n 5 --c 4 --k 3
{
  "pass@k": 1.0,
  "pass^k": 0.4
}
```

A task that looks 80% reliable succeeds three times in a row only 40% of the time (estimated from those 5 trials). Consistency targets get expensive fast; `python scripts/reliability.py` with no arguments prints the per-attempt rate a pass^k target implies (pass^8 of 0.9 needs about 0.987 per attempt, assuming independent trials).

## What it costs to find out

```
$ python scripts/reliability.py budget --tasks 30 --trials 5 --versions 2 --cost 0.4
{
  "runs": 300,
  "cost": 120.0
}
```

The $0.40 per run is a placeholder; use your own cost per task from traces.

## Comparing two versions

Save per-task results as `{"task_id": [n, c]}` for each version and run `python scripts/reliability.py compare old.json new.json`. It bootstraps over tasks, keeping each task's old and new rates paired, and reports whether the 95% interval on the difference excludes zero. If it doesn't, the change isn't distinguishable from noise at this suite size.

## Before trusting the suite

Run a do-nothing agent against it. If it scores well above zero, the graders reward something other than the outcome (the ABC audit found a do-nothing agent passing 38% of one public benchmark's airline tasks; `evals.md`). Then read twenty transcripts.

## Running it

`scripts/inspect_eval_template.py` shows the same plan in Inspect AI: `epochs=Epochs(n, [pass_at(1), pass_k(k), ...])`, an outcome scorer, and limits.
