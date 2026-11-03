# API reference and docstrings

Contents:

- [First: find out who owns the reference](#first-find-out-who-owns-the-reference)
- [What reference material is](#what-reference-material-is)
- [The entry pattern](#the-entry-pattern)
- [What to document and what to skip](#what-to-document-and-what-to-skip)
- [Docstring formats](#docstring-formats)
- [Describing parameters, returns, and errors](#describing-parameters-returns-and-errors)
- [Deprecation](#deprecation)
- [HTTP API reference](#http-api-reference)
- [Validate before you report](#validate-before-you-report)
- [Sources](#sources)

## First: find out who owns the reference

Run `scripts/scan_repo.py` and read its "Generated reference" section before
writing a single reference page.

If Sphinx autodoc, TypeDoc, rustdoc, godoc, Doxygen, or an OpenAPI spec is
present, the reference is generated from source. The work is then in the
docstrings and in the structure that feeds the generator, never in a parallel
set of hand-written pages. Two copies of the same reference diverge at the first
refactor, and the generated one is the one that stays right, so the hand-written
one becomes a trap that outranks it in search.

When no generator exists, hand-written reference is legitimate. Say so
explicitly in your plan, because it is a maintenance commitment the project is
taking on.

## What reference material is

Diataxis puts it precisely: reference describes the machinery, and its structure
mirrors the structure of the product. Its tone is "austere and uncompromising" -
neutral, objective, factual.

That has three consequences you can check.

**No instructions.** Reference says what a thing is, not how to accomplish a
goal with it. Steps belong in a how-to guide; link out.

**No opinion or marketing.** "Recommended", "powerful", "the best way to" do not
appear. A reader consulting reference is mid-task and needs a fact.

**Examples illustrate, they do not teach.** One short, consistent example per
entry, in the same shape every time, so the reader learns where to look once.
Examples that grow into explanation are the standard way reference decays; the
explanation belongs in its own document.

Reference is the mode where completeness matters most and prose matters least.
If it reads as boring, that is the correct outcome.

## The entry pattern

Every entry, every time, in the same order. Omit a field that does not apply
rather than writing "N/A", which trains readers to skip fields.

1. Name and signature
2. One sentence on what it does, in the present tense, that does not restate the
   name
3. When to use it, and which alternative to use otherwise
4. Parameters, with types, defaults, and constraints
5. Returns, with type and meaning
6. Raises or errors, with the condition that produces each
7. One runnable example
8. Caveats: side effects, thread safety, rate limits, auth requirements,
   performance characteristics that would surprise someone

Item 3 is the one that gets dropped and the one that has the most value. The
signature is already visible in the editor; which of six similar functions to
call is not.

Numbers carry provenance. A latency figure, a memory bound, or a throughput
claim needs a dated measurement behind it. Without one, delete it.

## What to document and what to skip

Document: public API, anything a caller can reach, non-obvious behavior, error
conditions, side effects, defaults, units, mutation, and thread or async safety.

Skip: private helpers, obvious getters and setters, parameter descriptions that
repeat the parameter name, and any restatement of what the type annotation
already says. `user_id: int` does not need "The user ID as an integer."

The distinction is whether the reader could have derived it. Volume is not the
goal; resolving ambiguity is.

## Docstring formats

Match the format already used in the file you are editing. Sphinx's `napoleon`
extension parses Google and NumPy styles side by side, so mixing them does not
break the build, but it makes entries inconsistent for readers and for
style-specific checkers such as `numpydoc` validation.

### Python, Google style

```python
def resize(image: Image, width: int, height: int, *, keep_aspect: bool = True) -> Image:
    """Scale an image to fit within a bounding box.

    The image is never enlarged: an image already smaller than the box is
    returned unchanged.

    Args:
        image: Source image. Not modified.
        width: Bounding box width in pixels. Must be positive.
        height: Bounding box height in pixels. Must be positive.
        keep_aspect: If True, preserve the aspect ratio and fit inside the box.
            If False, stretch to exactly width by height.

    Returns:
        A new image. The source is left untouched.

    Raises:
        ValueError: If width or height is not positive.

    Example:
        >>> resized = resize(photo, 800, 600)
        >>> resized.size
        (800, 450)
    """
```

`photo` is not defined inside the docstring, so `doctest` fails this example
with `NameError` until something supplies it. Under pytest, add it in
`conftest.py` with the `doctest_namespace` fixture, or build the input on a
line of the example itself.

### Python, NumPy style

Common in scientific projects. Sphinx renders it through the `napoleon`
extension, the same as Google style.

```python
def resize(image, width, height, keep_aspect=True):
    """Scale an image to fit within a bounding box.

    Parameters
    ----------
    image : Image
        Source image. Not modified.
    width : int
        Bounding box width in pixels. Must be positive.
    height : int
        Bounding box height in pixels. Must be positive.
    keep_aspect : bool, optional
        Preserve the aspect ratio. Default is True.

    Returns
    -------
    Image
        A new image. The source is left untouched.

    Raises
    ------
    ValueError
        If width or height is not positive.
    """
```

### TypeScript and JavaScript, JSDoc or TSDoc

In TypeScript, the types live in the signature. Repeating them in `@param {type}`
is the classic redundancy; describe the meaning instead. TSDoc reserves `{`
for inline tags, so `@throws {RangeError}` (JSDoc syntax) is a TSDoc parse
error; name the exception with `{@link RangeError}` on the `@throws` line.

```typescript
/**
 * Scale an image to fit within a bounding box.
 *
 * The image is never enlarged: an image already smaller than the box is
 * returned unchanged.
 *
 * @param image - Source image. Not modified.
 * @param width - Bounding box width in pixels. Must be positive.
 * @param keepAspect - Preserve the aspect ratio. Defaults to true.
 * @returns A new image. The source is left untouched.
 * @throws {@link RangeError}
 * If width or height is not positive.
 *
 * @example
 * const resized = resize(photo, 800, 600);
 * console.log(resized.width); // 800
 */
```

### Rust

rustdoc renders `///` comments as Markdown, and it compiles and runs the
examples during `cargo test`. That makes the example a test, which is why Rust
reference examples go stale less often than everyone else's.

```rust
/// Scale an image to fit within a bounding box.
///
/// The image is never enlarged: an image already smaller than the box is
/// returned unchanged.
///
/// # Errors
///
/// Returns [`ResizeError::InvalidBounds`] if `width` or `height` is zero.
///
/// # Examples
///
/// ```
/// # use imgkit::{resize, Image, ResizeError};
/// # fn main() -> Result<(), ResizeError> {
/// # let photo = Image::new(1600, 1200);
/// let resized = resize(&photo, 800, 600)?;
/// assert_eq!(resized.width(), 800);
/// # Ok(())
/// # }
/// ```
```

Lines starting with `# ` compile but are hidden from the rendered page. A
doctest is compiled as an external crate with its own `main`, so it needs the
`use` line, its inputs, and a `main` returning `Result` before `?` works;
without them `cargo test --doc` fails with "cannot find function `resize`".

### Go

By convention a doc comment is a full sentence that begins with the name of the
thing it documents (Go Code Review Comments). pkg.go.dev builds the reference
from these directly.

```go
// Resize scales an image to fit within a bounding box. The image is never
// enlarged: an image already smaller than the box is returned unchanged.
//
// Resize returns ErrInvalidBounds if width or height is not positive.
func Resize(img *Image, width, height int) (*Image, error) {
```

## Describing parameters, returns, and errors

Google's API reference comment guidance gives conventions worth following
because they make entries scannable:

- Method summaries start with a present-tense verb: "Adds", "Returns",
  "Deletes". Not "This method will add".
- Class summaries state the purpose, and do not repeat the class name.
- Boolean parameters describe both branches: "If true, validates the certificate.
  If false, trusts any certificate."
- Non-boolean parameter descriptions start with "The" or "A", capitalize the
  first word, and end with a period.
- Note defaults explicitly as "Default: 30 seconds", including the unit.
- Boolean returns read "True if the file exists; false otherwise."

Two things readers need that these conventions do not cover: units and mutation.
State the unit on every duration, size, and rate. State whether an argument is
modified in place, because that is the bug the reader will otherwise write.

## Deprecation

A deprecation notice needs three parts, and the third is the one usually
missing:

1. That it is deprecated, and since which version
2. What to use instead, linked
3. How to migrate, concretely

```python
def get_cached_user(user_id: int) -> User | None:
    """Return the cached user record.

    .. deprecated:: 4.2
        Use :func:`get_user` with ``cache=True``. The two differ in that
        ``get_user`` raises ``UserNotFound`` where this returned ``None``.
    """
```

Naming the behavioral difference is what makes the notice actionable. "Use X
instead" without it leaves the reader to discover the difference in production.

## HTTP API reference

If an OpenAPI spec exists, it is the source of truth and the reference is
generated from it. Improve the spec, not the rendered page.

Each endpoint needs: method and path, what it does, auth requirement, path and
query parameters with types and constraints, request body schema, a success
response with a real example body, the error responses with their status codes
and the condition that triggers each, rate limits, and pagination behavior.

A `400` with no list of what triggers it is not documentation.

## Validate before you report

Reference material is the easiest kind to check mechanically, so check it.

| Ecosystem | Command | What it checks |
|---|---|---|
| Python | `python -m doctest -v path/to/module.py` or `pytest --doctest-modules` | Runs `>>>` examples in docstrings |
| TypeScript | `npx tsc --noEmit` | Type-checks source only; `@example` blocks in comments are not compiled, so keep runnable examples in `.ts` files that `tsc` covers |
| Rust | `cargo test --doc` | Compiles and runs every doc example |
| Go | `go test ./...` | Runs `Example` functions and compares their `// Output:` comments |
| Go | `staticcheck -checks ST1020,ST1021,ST1022 ./...` | Doc comments start with the declared name (these checks are off by default; `go vet` has no doc-comment check) |
| OpenAPI | `npx @redocly/cli lint openapi.yaml` | Spec validity and lint rules |
| Sphinx | `sphinx-build -W -n -b html docs docs/_build` | `-W` turns warnings into errors; `-n` (nitpicky) is what reports broken `:func:` and `:class:` references, which `-W` alone lets pass |

If a validation step fails, fix the example and run it again before reporting.
If a tool is not available in the environment, say which check you did not run
rather than implying the examples are verified.

Then run the skill's own claim verifier over any prose reference pages:

```bash
python3 scripts/verify_claims.py docs/api.md
```

## Sources

- Diataxis, reference - https://diataxis.fr/reference/
- Google developer documentation style guide, API reference comments - https://developers.google.com/style/api-reference-comments
- Uddin and Robillard, "How API Documentation Fails", IEEE Software 2015 - https://www.cs.mcgill.ca/~martin/papers/ieeesw2015.pdf
- Sphinx napoleon extension - https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
- Go, doc comments - https://go.dev/doc/comment
- Go Code Review Comments, comment sentences - https://go.dev/wiki/CodeReviewComments#comment-sentences
- Go, testable examples - https://go.dev/blog/examples
- staticcheck ST1020 to ST1022 - https://staticcheck.dev/docs/checks/
- TSDoc, @throws - https://tsdoc.org/pages/tags/throws/
- pytest, doctest_namespace fixture - https://docs.pytest.org/en/stable/how-to/doctest.html
- Sphinx, sphinx-build options - https://www.sphinx-doc.org/en/master/man/sphinx-build.html
- rustdoc, documentation tests - https://doc.rust-lang.org/rustdoc/write-documentation/documentation-tests.html
- TSDoc - https://tsdoc.org/
- OpenAPI specification - https://spec.openapis.org/oas/latest.html
