# Security testing with pytest

Security tests turn an attack class into a regression test. Each pattern pairs a
fixed table of known payloads (documents what you defend against) with a property
over generated input (finds the payload you did not think of). Templates:
`assets/test_security_template.py`. Worked tests: `examples/wallet/tests/security/`.

## Contents

- [Threat to test map](#threat-to-test-map)
- [Input boundaries: parse, do not validate](#input-boundaries-parse-do-not-validate)
- [Path traversal](#path-traversal)
- [SQL injection and wildcard injection](#sql-injection-and-wildcard-injection)
- [Deserialization](#deserialization)
- [Resource limits](#resource-limits)
- [Exception contracts](#exception-contracts)
- [Authorization matrices](#authorization-matrices)
- [Static analysis, secrets, and dependencies](#static-analysis-secrets-and-dependencies)
- [What tests cannot do](#what-tests-cannot-do)

## Threat to test map

| Attack class | Test pattern | Tools | Example |
|---|---|---|---|
| Malformed input crashes or corrupts state | exception contract + postcondition | Hypothesis, Atheris | `tests/property/test_money_properties.py` |
| Path traversal | payload table + "never escapes base" property | pytest, Hypothesis | `tests/security/test_store_security.py` |
| SQL injection, LIKE wildcards | payload table + "results match the filter" property | pytest, Hypothesis, ruff `S608` | same file |
| Unsafe deserialization | feed a pickle payload, expect a clean rejection | pytest | same file |
| Oversized input (memory or CPU DoS) | huge input rejected before parsing | pytest | `tests/security/test_money_security.py` |
| Unicode and encoding tricks | non-ASCII digits, fullwidth dots, NUL, trailing newline | pytest | both security files |
| Missing authorization check | full role x action matrix | pytest parametrize | `assets/test_security_template.py` |
| Logic hidden behind exact values | solver search | CrossHair | `tests/symbolic/` |
| Vulnerable dependencies | audit installed packages | pip-audit | `assets/github/workflows/security.yml` |
| Committed secrets | scan staged changes and history | gitleaks | `assets/.pre-commit-config.yaml` |
| Dangerous code patterns | static rules | ruff `S` (bandit port) | `security.yml` |

## Input boundaries: parse, do not validate

Convert untrusted input into a typed value once, at the edge, and make the rest of
the code accept only that type. The example's `Wallet` accepts only `Decimal`, and
`parse_amount` is the only way text becomes one. A float, NaN, or infinity never
reaches the balance.

Three regex traps that tests should pin down, each verified on Python 3.12:

| Trap | Fact | Test |
|---|---|---|
| `$` matches before a trailing newline | `re.match(r"^[a-z]+$", "alice\n")` matches; `re.fullmatch(r"[a-z]+", "alice\n")` does not | `test_dollar_anchor_lets_a_trailing_newline_through` |
| `\d` is not `[0-9]` | `\d` matches Arabic-Indic digits, and `Decimal("١٢")` evaluates to `12` | `test_hostile_amount_text_is_rejected[arabic-indic-digits]` |
| Decimal accepts special values | `NaN`, `Infinity`, and exponents like `1e999999999` all parse | `[nan]`, `[infinity]`, `[huge-exponent]` |

Use `re.fullmatch` with explicit ASCII classes for anything security-relevant.

## Path traversal

```python
TRAVERSAL = ["../etc/passwd", "..", "/etc/passwd", "a/../../b", "a\\..\\b",
             "alice\n", "alice\x00", "\uff0e\uff0e", "%2e%2e%2f", "Alice", ""]

@pytest.mark.parametrize("name", TRAVERSAL, ids=repr)
def test_wallet_path_rejects_hostile_names(tmp_path, name):
    with pytest.raises(StoreError, match="invalid wallet name"):
        wallet_path(tmp_path, name)

@given(st.text(max_size=60))
def test_wallet_path_never_escapes_the_base_directory(tmp_path_factory, name):
    base = tmp_path_factory.getbasetemp() / "store"
    try:
        path = wallet_path(base, name)
    except StoreError:
        return
    assert path.is_relative_to(base.resolve())
```

The implementation uses two layers: an allow-list regex, then
`resolve()` plus `is_relative_to()` so a symlink inside the store cannot point out.
The property test uses `tmp_path_factory` because a function-scoped `tmp_path`
would be shared across all generated examples.

## SQL injection and wildcard injection

The v0 store builds SQL with an f-string. The demo test result:

```
E   assert ('alice', '10.00') is None
FAILED demo/test_v0_findings.py::test_get_returns_nothing_for_injection_payloads[tautology]
FAILED demo/test_v0_findings.py::test_get_returns_nothing_for_injection_payloads[like-wildcard]
```

Static analysis flags the same line without running anything:

```
examples/wallet/demo/store_v0.py:13:32: S608 Possible SQL injection vector through string-based query construction
```

Parameterized queries stop injection but not wildcard injection: with
`WHERE owner LIKE ? || '%'`, a user-supplied `%` or `_` still matches other rows,
and SQLite's `LIKE` is case-insensitive for ASCII. The fixed store avoids `LIKE`:

```python
"SELECT owner, balance FROM wallets WHERE substr(owner, 1, ?) = ? ORDER BY owner", (len(prefix), prefix)
```

and the property checks the semantics directly:

```python
@given(st.text(max_size=12))
def test_prefix_search_only_returns_matching_owners(prefix):
    ...
    assert owners == sorted(o for o in ("alice", "bob") if o.startswith(prefix))
```

## Deserialization

Never `pickle.loads`, `yaml.load` (without `SafeLoader`), or `eval` bytes from
outside the process. The test feeds a pickle payload to the loader and expects the
same clean `StoreError` as any other corrupt file:

```python
def test_load_wallet_refuses_pickle_payloads(tmp_path):
    (tmp_path / "mallory.json").write_bytes(pickle.dumps({"owner": "mallory"}))
    with pytest.raises(StoreError, match="corrupt wallet file"):
        load_wallet(tmp_path, "mallory")
```

## Resource limits

Check size before doing work. `parse_amount` rejects anything over 32 characters
before the regex runs, so a 100,000-character input costs one `len()`:

```python
("1" * 100_000, "too long"),
```

For regexes, avoid nested quantifiers like `(\d+)+`, which can backtrack
exponentially on near-miss input. Assert on behavior (rejected), not on timing:
timing assertions are flaky on shared CI runners. If a time bound is the contract,
use `pytest-timeout` (`@pytest.mark.timeout(2)`) as a hang detector.

## Exception contracts

A parser that leaks an internal exception (`decimal.InvalidOperation` instead of
`AmountError`) turns bad input into a 500 error or a crashed worker. Both Hypothesis
and Atheris found exactly this in v0 within seconds. The test shape is in
`property-fuzz-symbolic.md`; catch only the documented error.

## Authorization matrices

List every role and action, and assert each pair explicitly. A separate test checks
the table itself is complete, so adding a role without deciding its permissions
fails the suite:

```python
def test_matrix_covers_every_pair():
    assert set(EXPECTED) == {(r, a) for r in ROLES for a in ACTIONS}

@pytest.mark.parametrize(("role", "action"), sorted(EXPECTED), ids=lambda v: str(v))
def test_authorization_matrix(role, action):
    assert can(role, action) is EXPECTED[(role, action)]
```

## Static analysis, secrets, and dependencies

| Check | Command | Runs in |
|---|---|---|
| Bandit rules via ruff | `ruff check --select S src/` | `security.yml`, every push |
| Secrets in staged changes | gitleaks pre-commit hook | `.pre-commit-config.yaml` |
| Secrets in full history | `pre-commit run gitleaks --all-files` with `fetch-depth: 0` | `security.yml` |
| Private keys | `detect-private-key` hook | `.pre-commit-config.yaml` |
| Vulnerable dependencies | `pip-audit --skip-editable --strict` | `security.yml`, weekly too |
| New vulnerable deps in a PR | `actions/dependency-review-action` | `security.yml`, PRs only |

Test fixtures must never contain real credentials. Use obviously fake values
(`"test-token-not-real"`) so a secret scanner hit is always worth reading.

For reproducible installs, lock dependencies with hashes (`pip install
--require-hashes -r requirements.lock`, or a `uv.lock`). Without a lock, CI installs
whatever is newest that day, and a compromised release goes straight into the test
run.

## What tests cannot do

Tests prove the defenses you thought of still hold. They do not find a missing
defense nobody wrote a test for, and they do not replace threat modeling or code
review. Fuzzing and solver search narrow that gap for parsers and pure logic; they
do not reach authorization design, secrets handling, or deployment configuration.
