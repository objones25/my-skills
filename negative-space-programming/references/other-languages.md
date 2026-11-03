# Mapping the rules to other languages

The ten rules do not change. What changes is which mechanism survives the release build, and how
much of the negative space the type system can absorb before you have to write a check.

## The one question per language

Every language has a default assertion that a release build deletes. Find it, then decide what you
use instead for checks that must survive.

| Language | Deleted in release | Survives release |
|---|---|---|
| Python | `assert` (removed under `-O` / `PYTHONOPTIMIZE`) | `if not cond: raise`, i.e. the `require()` helper |
| C / C++ | `assert()` (disabled by `NDEBUG`) | a custom `CHECK()` macro that always compiles; `std::terminate` / `std::abort` |
| Rust | `debug_assert!` | `assert!`, `assert_eq!`, `panic!`, `unreachable!`, `expect` |
| Zig | `std.debug.assert` (elided in `ReleaseFast`/`ReleaseSmall`) | build in `ReleaseSafe`, or a custom assert that cannot be optimized out |
| TypeScript | nothing by default; types erase entirely at runtime | an explicit `invariant(cond, msg)` helper; a runtime schema validator at boundaries |
| Go | no assert in the language | explicit `if !cond { panic(...) }`; `errors.Is/As` for operating errors |

## C++

- `assert()` from `<cassert>` vanishes under `NDEBUG`, which most release builds define. Write a
  `CHECK(cond)` macro that expands to an `if` + `std::abort()` regardless of build type, and keep
  `assert()` for the checks you are willing to lose.
- Prefer `gsl::span` / `std::span` over pointer+length pairs: the bound travels with the data
  instead of living in a comment.
- `static_assert` is free: assert the relationships between constants, struct sizes, and alignments
  at compile time (rule: a check that runs before the program does is the cheapest one).
- Compile at the strictest warning level and treat warnings as errors; the compiler is asserting the
  negative space for you.
- Contracts (P2900) are still evolving; check the status for your standard version before designing
  around them.

## Rust

- `assert!` stays in release builds; `debug_assert!` does not. Use `assert!` for contracts.
- Much of the negative space is unrepresentable already: `Option` for absence, `Result` for
  operating errors, `NonZeroUsize`, newtypes for units, exhaustive `match`. Rule 9 does most of the
  work here, so assertion density is legitimately lower than in Python.
- `unreachable!()` for match arms that cannot occur; `expect("why this cannot fail")` where the
  message documents the invariant.
- `unsafe` blocks are where the negative space must be written by hand: every safety requirement
  of the called function becomes an assertion or a `// SAFETY:` comment stating why it holds.

## Zig

The source article's examples are Zig. Note its caveat: `std.debug.assert` is elided in
`ReleaseFast` and `ReleaseSmall`, so either build `ReleaseSafe` or write a custom assert that is
not optimized out. Otherwise rule 7 (assertions on in production) is silently violated.

- Errors are values: `!T` and `try` make operating errors explicit at every call site, which is
  rule 3 enforced by the compiler.
- `defer` / `errdefer` group allocation with release so leaks are visible in one screenful.
- TigerBeetle's own tooling (`src/tidy.zig`) enforces mechanical limits (function length, line
  width) in CI. That is the same idea as the audit script here.

## TypeScript

- Types are erased at runtime, so a `Mode` union does not stop a bad string from a JSON payload.
  Validate at every boundary (a schema validator such as zod at the parse point), assert inside.
- Write `function invariant(cond: unknown, msg: string): asserts cond`. The `asserts` return type
  gives you type narrowing *and* a runtime check, so one call does both jobs.
- `strict: true`, `noUncheckedIndexedAccess: true`, and no `any` are the cheapest negative-space
  checks available; they run on every build.
- `catch {}` with an empty body is the TypeScript form of rule 8's swallowed exception, and it is
  common in async code.

## Go

- No assert statement, so contracts are explicit `if !cond { panic(fmt.Sprintf(...)) }`. Reserve
  panic for programmer errors; return `error` for operating errors; the language already draws
  rule 3's line for you.
- `if err != nil` with an empty body, or `_ = err`, is the swallowed exception. `errcheck` in CI
  catches it.
- Bound goroutine fan-out with a semaphore or worker pool; an unbounded `go` in a loop is rule 4's
  unbounded loop wearing a different hat.

## What the audit script covers

`scripts/audit_negative_space.py` parses Python only (it uses the `ast` module). For other
languages the equivalent checks are usually available from existing tooling: `clang-tidy` and
`-Wall -Wextra -Werror` for C++, `clippy` for Rust, `zig build -Doptimize=ReleaseSafe` plus a tidy
script for Zig, `eslint` with `no-empty` and `@typescript-eslint/strict` for TypeScript,
`errcheck` and `go vet` for Go. Wire them into CI at the same gate.
