# Checking scripts, examples and tests

A quality finding needs proof the reviewer can rerun. "Might not handle unusual input" is not a finding. Work only in `RUN/workspace/`, never in the skill itself (AS002 catches any change there).

## Order

1. **Run the skill's own tests first.** Use `metadata.test-command` from its `SKILL.md`, with every optional tool installed, and read each skip reason: a skip hides whatever it skips. If the suite fails, that failure is the finding (`kind: test`, `verdict: wrong`, with the command and output), and mutation scoring stops for that suite, because mutation testing needs a green baseline.
2. **Run every script on its documented examples.** A documented command whose real output differs from the documented output is a `wrong` example item, with both shown.
3. **Probe for missed edge cases.** For each script, try the inputs its job implies: an empty file, an empty list, one element, a missing file, unicode and spaces in paths, very large input, the boundary values named in its own docs, a non-zero exit from a subprocess it calls. A finding includes the exact input, the command and the failing output.
4. **Look for cheap workarounds.** A hand-rolled regex where the tool has a flag, a broad `except`, a sleep instead of a wait, a hard-coded path or version, a check that only tests the happy path. A finding names the workaround, shows the robust alternative, and backs it with the source ranking (code, then docs, then research). Without that source, it is a suggestion.
5. **Mutation-test the tests.** For each Python script with tests:

   ```bash
   cd RUN/workspace
   python3 <this-skill>/scripts/mutate_check.py scripts/foo.py -- python3 -m pytest -x -q -p no:randomly tests/test_foo.py
   ```

   Every surviving mutant gets one outcome with a one-line reason: **missing test** (a real gap: an item with `verdict: wrong` on the test), **equivalent** (no input can tell the mutant apart from the original), or **unproductive** (killable, but only by pinning an arbitrary detail such as a buffer size). Do not report every survivor as a gap; the reason line is what makes the outcome checkable. For a whole package, `mutmut` in the throwaway environment is the alternative.
6. **Check that each test can fail.** A test is only worth counting if breaking the code makes it fail. Watch for tautologies (`assert x or True`), assertions on a value the test itself set, and `pytest.raises` blocks whose match is too loose.

## What goes where

| You have | It is |
|---|---|
| A failing input, command and output | A finding: `script`, `example` or `test` item with `verdict: wrong` |
| A robust alternative backed by a source | A finding, with the source as evidence |
| A concern with neither | A line in `suggestions.jsonl` |
| A test suite that fails before you change anything | A finding, and the end of mutation scoring for that suite |
