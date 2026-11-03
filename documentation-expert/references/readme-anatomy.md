# README and CONTRIBUTING anatomy

Contents:

- [The completeness test](#the-completeness-test)
- [Section order](#section-order)
- [Section by section](#section-by-section)
- [Depth tiers](#depth-tiers)
- [Ecosystem conventions](#ecosystem-conventions)
- [CONTRIBUTING](#contributing)
- [Sources](#sources)

## The completeness test

standard-readme opens its Background with Ken Williams's bar: *"Your
documentation is complete when someone can use your module without ever having
to look at its code."*

That is a test, not a slogan. Apply it to a draft by asking, for each thing the
reader must supply, whether the document told them where to get it: the runtime
version, the credential, the config file, the service that must be running.
Every one of those you assumed is a prerequisite the reader will discover as a
failure.

A second test, from the same spec: Install and Usage must each contain a code
block. A README a reader cannot copy from has not finished the job.

## Section order

standard-readme fixes the order for open-source libraries. Deviating is fine when
there is a reason; the default exists so readers can find things without
reading.

| Order | Section | Status | Hard requirement from the spec |
|---|---|---|---|
| 1 | Title | Required | Matches the repository, folder, and package manager name |
| 2 | Banner | Optional | No heading of its own; image lives in this repository |
| 3 | Badges | Optional | No heading of its own; newline delimited |
| 4 | Short description | Required | No heading of its own; under 120 characters; matches the package manager's and GitHub's description |
| 5 | Long description | Optional | No heading of its own |
| 6 | Table of contents | Required over 100 lines | Links every level-two heading |
| 7 | Security | Optional | Only when it needs prominence |
| 8 | Background | Optional | Covers motivation and abstract dependencies |
| 9 | Install | Required | Contains a code block |
| 10 | Usage | Required | Contains a code block; a `CLI` subsection if the project has one |
| 11 | Extra sections | Optional | After Usage, before API |
| 12 | API | Optional | Signatures and return types |
| 13 | Maintainers | Optional | Titled `Maintainer` or `Maintainers` |
| 14 | Thanks | Optional | Titled `Thanks`, `Credits`, or `Acknowledgements` |
| 15 | Contributing | Required | States where to ask questions and whether PRs are accepted |
| 16 | License | Required | Last section; names an SPDX identifier and the license owner |

On GitHub specifically: a README is looked up in `.github/`, then the repository
root, then `docs/`. Use relative links so they survive cloning. Content past
500 KiB is truncated in the web UI, which is a size ceiling worth knowing before
a README grows into a manual.

## Section by section

### Title and short description

The first screen decides whether the reader continues. It answers what this is,
who it is for, and what it does that alternatives do not.

Write the short description as one sentence under 120 characters, in the
reader's vocabulary, naming the thing the project is rather than the category it
belongs to.

- Weak: "A powerful, flexible library for modern developers."
- Strong: "Route HTTP requests to Python functions with type-checked path
  parameters."

Common mistake: unexplained jargon in the opening paragraph. Count the technical
terms a newcomer would not know; more than two and the opening has failed. This
is not a request to dumb anything down, only to define the terms on first use or
link them.

### Badges

Badges are metadata, not content. Keep to the few that tell a reader something
they would otherwise have to go find: build status, published version, license.
More than about six and they push the description below the fold, which is the
opposite of what they were added for.

### Background

Include it when the reader needs the motivation to evaluate the project, and
skip it when they do not. Its real job is stating what problem existed before
this project, and what the alternatives are. A Background section that only
restates the description is deletable.

### Install

Cover, in order: prerequisites with versions, the install command, and the
verification command that proves it worked.

The prerequisites line is the most frequently skipped and the most frequently
needed. State the runtime version range, not a single version, and state where
it comes from. The verification command matters because it turns a silent
failure into an immediate one.

Requires Python 3.8 or newer.

```bash
python -m pip install httpx
python -c "import httpx; print(httpx.__version__)"
```

Put the prerequisite in prose, not as a `#` line inside the block: zsh, the
macOS default shell, runs a pasted `# ...` line as a command unless
`interactivecomments` is set, and prints `command not found: #`.

Common mistake: an install command with no verification and no version floor.
The reader on an old runtime gets a stack trace from inside your library and no
way to connect it to the cause.

### Usage

The first example is the most important text in the document. It should be the
shortest thing that produces a real result, runnable by copy and paste, with the
output shown.

Show the output. An example without its result forces the reader to run it just
to learn what it does, and it hides the case where the example no longer works.

```python
import httpx

response = httpx.get("https://example.com")
print(response.status_code)
# 200
```

Order examples by how many readers need them: the common case first, then
configuration, then the advanced case. Google's guidance is to precede each
sample with a sentence saying what it does, and to mark elisions with a comment
in the sample's own language rather than with `...`, which is ambiguous in
Python and invalid in most others.

### API

In a README, this section is a map, not a manual. List the exported names with
one line each and link to the full reference. If a generator produces the
reference, link to the generated output rather than restating it here; see
[api-reference.md](api-reference.md).

Common mistake: pasting the full signature list into the README. In Uddin and
Robillard's survey, no respondent ranked "excess structural information" (detail
"which could be easily obtained through modern IDEs") as a top priority. The README's job is the part the editor
cannot show: which of these you actually need, and when.

### Configuration

Every environment variable and config key the program reads, with its type,
default, and effect. `scripts/scan_repo.py` lists the ones the source reads that no
document mentions, which is usually where this section comes from.

A table works better than prose here because readers scan it for one row.

### Honest limits

The section most projects omit and most readers want. State what the project
does not do, what it is not ready for, and when a reader should choose something
else. Naming the case where your project is the wrong choice buys more trust
than any benchmark, and it prevents the bug reports that come from mismatched
expectations.

### Contributing and License

Contributing states where to ask questions and whether pull requests are
accepted, then links to `CONTRIBUTING.md` for the detail. License names the SPDX
identifier and goes last.

## Depth tiers

Match the README to the project. Over-documenting a small utility is its own
failure mode: it buries the two things the reader needed.

**Small utility, one purpose.** Title, description, install, one usage example,
license. Roughly 40 lines. Anything more is padding.

**Library or framework.** The full standard-readme order, a table of contents,
several usage examples ordered by frequency, a configuration table, an API map
linking to reference, and honest limits. Roughly 150 to 300 lines.

**Platform or application.** The README becomes a routing document: what this
is, how to run it locally, and where everything else lives. The detail moves
into `docs/`. A README that has grown past roughly 400 lines is usually a docs
tree that has not been split out yet.

## Ecosystem conventions

Readers arrive with expectations set by the ecosystem. Meeting them costs
nothing and missing them looks careless.

**npm.** The package name matches `package.json`. Show the install for one
package manager and mention the others exist. Note the module system (ESM or
CommonJS) and the minimum Node version, because both routinely break installs.

**PyPI.** Show `python -m pip install pkg`, and add the `uv add pkg` (project
dependency) or `uv tool install pkg` / `pipx install pkg` (command-line tool)
form when readers are likely to use uv. Quote extras: `pip install 'pkg[extra]'`,
because zsh treats unquoted brackets as a glob and fails with `no matches found`.
Take the supported Python range from `requires-python` in `pyproject.toml`
(PEP 621), not from classifiers or memory. If the package ships type hints,
say so; the marker is a `py.typed` file in the package (PEP 561).

**Cargo.** Show the `Cargo.toml` dependency line as well as `cargo add`. Name the
feature flags and which are default. Link to docs.rs rather than restating the
API.

**Go.** For a library, show `go get` with the module path, run inside the
reader's module. For a command, show `go install module/path/cmd@latest`:
since Go 1.18, `go get` only edits `go.mod` and fails outside a module. Take the
minimum Go version from the `go` line in `go.mod`. Reference lives on
pkg.go.dev, generated from doc comments.

**CLI tools.** Show the invocation with real flags and real output. Include
`--help` output only if it is short; otherwise link to the command reference.
Say what the exit codes mean, because scripts depend on them.

**Services and applications.** Prerequisites become the dominant section:
runtime, database, message broker, credentials. Give a single command that
brings the whole thing up locally, even if it is a compose file, and give the
health check that proves it is up.

**Research and ML projects.** Add a citation block if there is a paper, name the
dataset and where to get it, and state hardware requirements and expected
runtime. Pin the versions that reproducibility depends on.

## CONTRIBUTING

A contributing guide answers the questions a first-time contributor hits in the
order they hit them.

1. **Is this contribution wanted?** Say whether PRs are accepted, what kinds are
   welcome, and whether to open an issue first. This one line prevents the most
   common wasted effort in open source.
2. **How do I set up a development environment?** The exact commands, including
   the ones that differ from the user-facing install. Include the verification
   step.
3. **How do I run the tests?** The command, how long it takes, and how to run a
   subset. A contributor who cannot run the tests will not write any.
4. **What are the conventions?** Formatter, linter, commit message format, branch
   naming. Give the command that applies them rather than describing the rules,
   where a command exists.
5. **What happens to my PR?** Who reviews, roughly how long, what CI must pass.
6. **Where do I ask?** The actual channel, linked.

Link the code of conduct rather than restating it. For conventions a formatter
or linter enforces, give the command that runs it rather than a description of
its rules, because the description and the tool will drift apart, and the tool
wins.

The GitHub Open Source Survey found process documentation of this kind is valued
more highly by people from underrepresented groups, which makes an unclear
contributing guide an access problem rather than only an inconvenience.

## Sources

- standard-readme specification - https://github.com/RichardLitt/standard-readme
- GitHub, About READMEs - https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes
- Google developer documentation style guide, code samples - https://developers.google.com/style/code-samples
- Write the Docs, beginner's guide to documentation - https://www.writethedocs.org/guide/writing/beginners-guide-to-docs/
- Uddin and Robillard, "How API Documentation Fails", IEEE Software 2015 - https://www.cs.mcgill.ca/~martin/papers/ieeesw2015.pdf
- GitHub Open Source Survey 2017 - https://opensourcesurvey.org/2017/
- Python Packaging User Guide, writing pyproject.toml - https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
- Python Packaging User Guide, installing packages (quoted extras) - https://packaging.python.org/en/latest/tutorials/installing-packages/
- PEP 561, distributing type information - https://peps.python.org/pep-0561/
- Go 1.18 release notes, `go get` - https://go.dev/doc/go1.18
