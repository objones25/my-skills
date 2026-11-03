# Documentation anti-patterns

Contents:

- [How to use this file](#how-to-use-this-file)
- [The document is wrong](#the-document-is-wrong)
- [The document is missing something](#the-document-is-missing-something)
- [The document is unusable](#the-document-is-unusable)
- [The document is dishonest](#the-document-is-dishonest)
- [Failure modes specific to generated documentation](#failure-modes-specific-to-generated-documentation)
- [Sources](#sources)

## How to use this file

Read the group that matches the section you are about to write. Each entry names
the failure, what it costs the reader, and the fix.

The ordering is deliberate. The empirical work on this puts content problems
firmly ahead of presentation problems: developers named incompleteness,
ambiguity, and incorrectness as most in need of attention, and no surveyed
developer ranked excess structural information or tangled information as a top
priority (Uddin and Robillard 2015). Fix truth before you fix polish.

## The document is wrong

### The lie

Instructions that no longer work. The install command references a package name
that changed, the example calls a function that was renamed, the config key was
removed two releases ago.

Cost: the highest of any failure here. The reader trusts the document, follows
it, fails, and cannot tell whether they made a mistake or the document did.
Write the Docs states the principle directly: incorrect documentation is worse
than missing documentation.

Fix: `scripts/verify_claims.py` catches the mechanical cases. For the rest, open the
definition and compare the signature, do not just confirm the name exists.

### Obsoleteness

The document describes a previous version and never says so. Distinct from the
lie because each individual statement was true once.

Cost: the reader on the current version debugs a difference the document does
not acknowledge.

Fix: state the version the document describes. Where behavior changed, say what
it changed from and in which release. Prefer version-agnostic phrasing for
anything that does not need pinning.

### Inconsistency

Two documents, or two sections, disagree. The README says the default timeout is
30 seconds; the API reference says 60.

Cost: the reader cannot resolve it without reading source, which is the outcome
documentation exists to prevent.

Fix: one fact, one place, linked from everywhere else. Write the Docs calls this
`Unique`: eliminate content overlap between sources. Where a fact must appear
twice, generate the second copy rather than typing it.

### Pseudocode fantasy

An example that looks runnable and is not: missing imports, invented helper
functions, a variable that appears from nowhere.

Cost: the reader copies it, it fails, and now they are debugging your example
instead of using your library.

Fix: run every example. Paste real output. `scripts/verify_claims.py` parses code blocks
and resolves imported symbols, which catches the common cases.

### The stale screenshot

An image of a UI, a terminal, or an output that no longer looks like that.

Cost: images are never checked by any tool and rot silently. The reader assumes
a screenshot is current because nobody publishes an old one on purpose.

Fix: prefer text output in a code block, which can be diffed and verified. When
an image is necessary, date it and note the version it shows.

## The document is missing something

### Install and pray

An install command with no prerequisites, no version floor, and no verification
step.

Cost: readers on the wrong runtime get an error from deep inside the library,
with nothing connecting it to the cause.

Fix: state the runtime version range before the install command, and give a
command whose output proves the install succeeded.

### The ghost town

A README with a title, badges, and nothing a reader can act on. Sometimes it
says `documentation coming soon`.

Cost: the reader leaves. A project that cannot describe itself reads as
abandoned regardless of its commit history.

Fix: the minimum viable README is a description, an install block, one usage
example with output, and a license. Roughly 40 lines, and it beats an empty
scaffold every time.

### Undocumented failure modes

The document covers only the happy path. Errors, exit codes, rate limits, and
what happens when a dependency is down are absent.

Cost: the reader finds out what fails, and how, in production instead of in the
docs.

Fix: list the errors with the condition that triggers each. For a CLI, document
the exit codes. `scripts/scan_repo.py` reports environment variables and public names
the docs never mention; the same discipline applies to raised exceptions.

### Ambiguity

The description is present but does not resolve the question. "Sets the cache
mode" without saying what the modes are. "Timeout in seconds" without saying
whether it covers connection, read, or total.

Cost: in Uddin and Robillard's survey, ambiguity was the problem most often
ranked a top priority (51.9 percent of respondents). It looks like
documentation, so nobody notices it is missing.

Fix: state the units, the defaults, the valid values, and whether the argument
is modified. Then read the sentence as an outsider and ask what it still leaves
open.

## The document is unusable

### The novel

Correct, complete, and forty screens long with no structure. Everything is here
and nothing can be found.

Cost: bloat and fragmentation are real reported problems, just lower priority
than wrongness. A reader who cannot find the answer is in the same position as
one for whom it does not exist.

Fix: split by reader need rather than by topic size. A README past roughly 400
lines is usually a docs tree that has not been separated yet.

### The API dump

Every public symbol listed with its signature and a description that restates
the signature. `get_user(user_id: int) -> User` documented as "Gets the user by
user ID."

Cost: no surveyed developer ranked it a top priority, and it crowds out the
information they rank highest. Editors already show signatures.

Fix: document what the signature cannot say. Why this function rather than the
other one, what it costs, what it mutates, when it fails.

### Badge soup

Fifteen badges above the fold, pushing the description below it.

Cost: the first screen decides whether the reader continues, and it has been
spent on shields.

Fix: keep the few that tell the reader something: build status, version,
license.

### The orphan section

A section nothing links to, covering something the project no longer does, kept
because deleting felt risky.

Cost: it is indexed, it is found, and it is wrong.

Fix: delete it. Version control has the history. Deleting is documentation work.

### The FAQ

A growing list of unrelated questions used in place of writing the actual
sections.

Cost: Write the Docs names this one directly: FAQs go stale, accumulate
unrelated content, become hard to search, and let authors avoid writing real
documentation.

Fix: move each answer into the section where a reader would look for it. Keep an
FAQ only for questions that genuinely span sections.

### Empty scaffolding

Headings created for content that does not exist yet. `## Advanced usage`
followed by the next heading.

Cost: it promises coverage the project does not have, and it makes the table of
contents lie.

Fix: Diataxis is explicit that creating empty structures is a misuse of the
framework. Write the section or omit the heading. If the gap matters, say what
is missing and why, in prose.

## The document is dishonest

### The resume

A README that sells the project to a hiring manager instead of explaining it to
a user. Superlatives, no measurements, achievements framed as features.

Cost: an experienced reader discounts everything else in the document.

Fix: replace each adjective with the number behind it, or delete it. "Fast"
becomes a benchmark with a date and a machine.

### Unmarked aspiration

Features described in the present tense that are planned, partial, or behind a
flag.

Cost: the reader integrates against something that does not exist.

Fix: mark status explicitly. Experimental, planned, and behind-a-flag are all
honest and all useful. Silence is not.

### Missing non-goals

No statement of what the project does not do or when to choose something else.

Cost: mismatched expectations, which arrive later as bug reports and as
frustration.

Fix: a short "when not to use this" section. It costs a few lines and saves the
reader who should have picked something else.

### The template zombie

A generated scaffold with the placeholders still in it: `YOUR_PROJECT_NAME`,
`TODO: describe`, `{{ description }}`.

Cost: it signals that nobody has read the document, which invites the reader to
stop reading it too.

Fix: `scripts/lint_prose.py` rules `HONEST001` and `HONEST002` catch these. Run it before
publishing.

## Failure modes specific to generated documentation

Documentation written by a language model, including yours, fails in
characteristic ways. These are worth checking for by name.

### Plausible invention

A config key, CLI flag, environment variable, response field, or method that
fits the project's naming conventions perfectly and does not exist. This is the
single most dangerous failure in this file, because it is undetectable by
reading: it looks more consistent than the real API.

Fix: `scripts/verify_claims.py` exists for this. Run it on anything you generate, not
only on documents you audit. Write only names you have resolved to a source
line.

### Restating the code

A description that paraphrases the implementation line by line, adding a layer
without adding information.

Fix: before writing a sentence, ask what it tells a reader who has the source
open. If the answer is nothing, cut it.

### Universal middle-of-the-road structure

Every README ends up with the same eight sections in the same order regardless
of what the project is, because that is the average of the training data. A
one-function utility gets a Roadmap and an Acknowledgements section.

Fix: choose the depth tier deliberately, and delete sections the project does
not need. See [readme-anatomy.md](readme-anatomy.md).

### Over-hedging

Every statement wrapped in "may", "typically", "in most cases" because certainty
was not established.

Fix: go and establish it. If you cannot, say specifically what you did not
verify. A precise admission is more useful than a vague hedge.

### Parallel reference

Hand-written API pages created next to a live generator, because writing pages
is what the request seemed to ask for.

Fix: check for a generator first. Improve the docstrings instead. See
[api-reference.md](api-reference.md).

### Silent truncation

Documenting the first N of something and presenting it as the whole, or quietly
skipping the parts that were hard to verify.

Fix: state the bound. "Covers the twelve public functions in `core`; the
`experimental` module is not documented here" is honest and lets the reader plan.

## Sources

- Uddin and Robillard, "How API Documentation Fails", IEEE Software 2015 - https://www.cs.mcgill.ca/~martin/papers/ieeesw2015.pdf
- Write the Docs, documentation principles - https://www.writethedocs.org/guide/writing/docs-principles/
- Write the Docs, beginner's guide to documentation - https://www.writethedocs.org/guide/writing/beginners-guide-to-docs/
- Diataxis, how to use Diataxis - https://diataxis.fr/how-to-use-diataxis/
- GitHub Open Source Survey 2017 - https://opensourcesurvey.org/2017/
- standard-readme specification - https://github.com/RichardLitt/standard-readme
