# 01: "Everything is green and coverage is 100%, but customers at exactly $50 still pay shipping"

*Illustrative system.* A small pricing module (`src/shop/pricing.py`): free shipping
"at 50.00 and above", whole-percent discounts. Four tests, all passing, 100% branch
coverage.

## Verdict

The tests cannot fail. Don't read the coverage number; run the audit, then rewrite the
tests from the spec, then prove they can fail with the mutation check.

## Candidate causes

1. Tests that assert nothing or only existence (the suite is green by construction).
2. Tests whose expected values were copied from the code's output (they lock in the bug).
3. A missing boundary case (5000 itself was never tested).

The cheapest test that separates them is the audit: it reads the tests without running
anything.

## The evidence

```
$ pytest --cov=shop --cov-branch
src/shop/pricing.py       10      0      4      0   100%
4 passed

$ python scripts/audit_tests.py tests/
tests/test_pricing.py:4: WEAK_ASSERT: test_apply_discount: every assertion only checks existence or type  [warning]
tests/test_pricing.py:9: NO_ASSERT: test_bad_percent asserts nothing
tests/test_pricing.py:12: SWALLOWED_FAILURE: except Exception without re-raise hides failures
tests/test_pricing.py:17: PRINT: print() in a test; assert instead, or use capsys/caplog  [warning]
tests/test_pricing.py:21: NO_ASSERT: test_shipping_free asserts nothing
tests/test_pricing.py:22: BARE_COMPARISON: `shipping_cents(5000) == 0` is not asserted

6 finding(s): BARE_COMPARISON=1, NO_ASSERT=2, PRINT=1, SWALLOWED_FAILURE=1, WEAK_ASSERT=1
```

`test_shipping_free` computed `shipping_cents(5000) == 0` and threw the result away.
Asserted, it would have failed on day one. The mutation check on the old tests (after
the code fix below) says the same thing in numbers:

```
2/16 mutants killed (12%), baseline 2.3s
SURVIVED line 16: >= -> >  <- no test fails when this changes
SURVIVED line 16: 5000 -> 5001  <- no test fails when this changes
...
```

## The fix

Tests rewritten with expected values worked out by hand from the spec (10% of 1000 is
100, so 900), parametrized over the boundaries, with an anchored `match=` on the error:

```
$ pytest -q
E       assert 499 == 0
E        +  where 499 = shipping_cents(5000)
FAILED tests/test_pricing.py::test_shipping_is_free_from_fifty_dollars[exactly-50-is-free]
1 failed, 8 passed
```

That is the production bug (`> 5000` where the spec says "and above"). After changing it
to `>= 5000`:

```
$ pytest -q
9 passed
$ python scripts/audit_tests.py tests/
clean: no findings
$ python scripts/mutate_check.py src/shop/pricing.py -- python -m pytest -x -q -p no:randomly tests/
16/16 mutants killed (100%), baseline 2.3s
```

References: `rules.md` (rules 6, 28, 29, 31), `mutation-testing.md`, `llm-written-tests.md` (where expected values come from).
