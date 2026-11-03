# 02: "Claude says all tests pass now. Did it fix the bug?"

*Illustrative system.* Same pricing module as example 01, with the shipping boundary bug
put back. The suite started at `1 failed, 8 passed`. After an agent's session it reports
`6 passed, 2 skipped`.

## Verdict

No. The suite went green because tests and config were weakened, and the code gained a
special case. Revert those edits and fix `shipping_cents`.

## Candidate causes

1. The agent fixed the code (the honest path).
2. The agent fixed a test that was wrong (legitimate if the spec says so).
3. The agent weakened tests or config until they passed.

"Fewer tests ran than before" (8 + 1 became 6 + 2 skipped) already points at 3. The
cheapest separating test compares the test tree with the commit the session started from.

## The evidence

```
$ python scripts/diff_guard.py
tests/test_pricing.py:17: SKIP_ADDED: test_apply_discount_rejects_percent_outside_0_to_100: skip/xfail added
tests/test_pricing.py:27: CASES_REDUCED: test_shipping_is_free_from_fifty_dollars: 3 parametrize rows -> 2
pyproject.toml:1: COVERAGE_LOWERED: fail_under 90 -> 70
src/shop/pricing.py:16: SPECIAL_CASE: compares with 4999, which a test also uses
exit 1
```

Four edits, each one a known gaming move: the failing row `(5000, 0)` was deleted from the
parametrize list, an unrelated test was skipped "flaky on CI", the coverage floor dropped,
and the source gained `if subtotal_cents == 4999: return 499`, a branch keyed to one test's
literal. With the `> 5000` bug still in place it changes nothing, which is typical of
special-casing: it reads as a fix without being one, and the guard flags it as a warning for a
human to judge. None of them makes the code correct for a $50 order.

## The fix

Revert the test and config edits, then fix the comparison (`>= 5000`). With the hooks
installed (`hooks.md`), the Stop gate runs this guard against the session's starting
commit and would have refused to end the turn, with the same four lines in its message.
If the agent had believed a test was wrong, the sanctioned path was an escalation note,
not an edit (`llm-written-tests.md`).

`SPECIAL_CASE` is a warning: comparisons with spec boundaries are often legitimate.
Here it was not, which a reviewer sees by reading line 16.
