"""The grader must pass good answers and fail bad ones; otherwise eval results mean nothing."""
from __future__ import annotations

import pytest

grade = pytest.importorskip("grade", reason="evals/ is not shipped in the .skill package")

GOOD_429 = '''Yes, retry the 429s, honour `Retry-After`, and cap the total wait.

```python
async def get_with_retry(client, url, deadline):
    resp = await client.get(url)
    if resp.status_code == 429:
        await asyncio.sleep(float(resp.headers["Retry-After"]))
    return resp
```

For the POSTs that create records, send an idempotency key or do not retry them.'''

BAD_429 = '''Never retry 429s; that only makes overload worse. This keeps the job cleaner.

```python
def broken(:
```'''


def by_text(result: dict) -> dict[str, bool]:
    return {e["text"]: e["passed"] for e in result["expectations"]}


def test_retry_eval_passes_good_and_fails_bad() -> None:
    good, bad = by_text(grade.grade(4, GOOD_429)), by_text(grade.grade(4, BAD_429))
    assert all(good.values()), good
    assert not bad["[auto] Does not say never retry 429"]
    assert not bad["[auto] All Python code blocks parse"]
    assert not bad["[auto] No unfalsifiable mood claims"]


def test_protocol_eval_catches_the_audited_misclaims() -> None:
    bad = "Protocols cannot share code with subclasses, and isinstance is about 160x slower."
    res = by_text(grade.grade(3, bad))
    assert not res["[auto] No ~100x or ~160x isinstance claim"]
    assert not res["[auto] Does not say Protocols cannot share code"]
    good = "Explicitly subclass the Protocol; mypy flags a missing method, and @abstractmethod raises at runtime."
    assert all(by_text(grade.grade(3, good)).values())


def test_singleton_eval_rejects_new_based_singleton() -> None:
    bad = "```python\nclass Model:\n    _i = None\n    def __new__(cls):\n        return cls._i\n```"
    good = ("Load it in the FastAPI lifespan, or guard it with a Lock. Note @cache can run the "
            "loader more than once under threads.\n\n```python\n_lock = threading.Lock()\n```")
    assert not by_text(grade.grade(6, bad))["[auto] No Singleton class"]
    assert all(by_text(grade.grade(6, good)).values())


def test_refactor_eval_checks_names_smtp_and_transcript() -> None:
    good = '''```python
def export_report(rows: list[ReportRow], kind: str, send) -> str:
    body = FORMATTERS[kind](rows)
    send(body)
    return body

def main():
    sender = SmtpSender(smtplib.SMTP("smtp.internal", 25))
```'''
    bad = good.replace("send(body)", "smtplib.SMTP('smtp.internal').sendmail('a', ['b'], body)")
    assert all(by_text(grade.grade(2, good, transcript="$ python scripts/design_smells.py exporters.py")).values())
    res = by_text(grade.grade(2, bad, transcript=""))
    assert not res["[auto] export_report no longer constructs SMTP"]
    assert not res["[auto] Scanner was run"]


def test_every_eval_has_auto_checks_and_matches_evals_json() -> None:
    import json
    from pathlib import Path
    data = json.loads((Path(grade.__file__).parent / "evals.json").read_text())
    ids = {e["id"] for e in data["evals"]}
    assert ids == set(grade.CHECKS)
    for e in data["evals"]:
        for f in e["files"]:
            assert (Path(grade.__file__).parent.parent / f).exists(), f
