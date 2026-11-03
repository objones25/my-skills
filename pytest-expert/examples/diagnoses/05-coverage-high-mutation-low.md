# 05: "Coverage is 100% and the audit is clean. Are we done?"

*Illustrative system.* The pricing module from example 01, with tests that do assert
something: the audit finds nothing to complain about.

## Verdict

No. Every line runs and every test asserts, but the assertions are loose enough that
10 of 16 deliberate bugs go unnoticed. Tighten the expected values at the boundaries.

## Candidate causes

1. Inequalities where exact values are known (`< 1000` instead of `== 900`).
2. No test at a boundary (`5000` itself, `0` and `100` percent).
3. Error tests that match a prefix but not the value.

Coverage cannot tell these apart, because each line already runs. The mutation check
can: each survivor names a change no test notices.

## The evidence

```
$ pytest --cov=shop --cov-branch
src/shop/pricing.py       10      0      4      0   100%
4 passed

$ python scripts/audit_tests.py tests/
clean: no findings

$ python scripts/mutate_check.py src/shop/pricing.py -- pytest -x -q -p no:randomly tests/
6/16 mutants killed (38%), baseline 2.2s
SURVIVED line 9: < -> <=  <- no test fails when this changes
SURVIVED line 9: 0 -> 1  <- no test fails when this changes
SURVIVED line 9: > -> >=  <- no test fails when this changes
SURVIVED line 9: 100 -> 101  <- no test fails when this changes
SURVIVED line 11: // -> *  <- no test fails when this changes
SURVIVED line 11: * -> //  <- no test fails when this changes
SURVIVED line 11: 100 -> 101  <- no test fails when this changes
SURVIVED line 16: >= -> >  <- no test fails when this changes
SURVIVED line 16: 5000 -> 5001  <- no test fails when this changes
SURVIVED line 18: 499 -> 500  <- no test fails when this changes
```

Each survivor is a test to write. `* -> //` survives because `1000 - 1000 // 10 // 100`
is 999, still `< 1000`. `5000 -> 5001` survives because nothing tests 5000. `0 -> 1`
on line 9 survives because nothing tests a discount of exactly 0 percent.

## The fix

The parametrized tests from example 01 (exact values at 0, 10, 50, 100 percent; 4999,
5000, 6000 cents; `-1` and `101` rejected with an anchored `match=`) kill 16 of 16. The
rule this example illustrates is in `mutation-testing.md`: coverage is a floor,
mutation score is the signal. Meta's ACH found 49% of the tests it kept for killing a
mutant added no line coverage (`papers/meta-ach.md`).
