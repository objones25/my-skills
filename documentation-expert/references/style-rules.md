# Style rules and mode discipline

Contents:

- [Person, voice, tense](#person-voice-tense)
- [Headings](#headings)
- [Lists and procedures](#lists-and-procedures)
- [Code in text and in blocks](#code-in-text-and-in-blocks)
- [Link text](#link-text)
- [Accessibility](#accessibility)
- [Words to cut or replace](#words-to-cut-or-replace)
- [Writing for readers who translate](#writing-for-readers-who-translate)
- [Mode discipline](#mode-discipline)
- [What the linter enforces](#what-the-linter-enforces)
- [Sources](#sources)

Everything here is from the Google developer documentation style guide unless
marked otherwise. The word table in [Words to cut or replace](#words-to-cut-or-replace)
is the exception: it mixes Google's word list with this skill's own choices,
and its last column says which. Where `scripts/lint_prose.py` checks a rule, the rule ID is named so
an audit can cite it.

## Person, voice, tense

**Second person.** Address the reader as "you", not "we" or "us". "The following
sections describe how you can create a website", not "how we can create a
website". Reserve "we" for the authoring organization, and use it rarely.

**Imperative for instructions.** "Click **Submit**." "Run `npm test`." Not "You
should click Submit" and not "The user clicks Submit".

**Active voice.** Name who does the thing. "Send a query to the service. The
server sends an acknowledgment", not "The service is queried, and an
acknowledgment is sent". Passive is acceptable when the actor is genuinely
irrelevant: "Over 50 conflicts were found."

**Present tense.** "The server returns a 404", not "will return". Reserve "will"
for events that are genuinely later in time: "The file is archived the next time
the backup process runs."

## Headings

- Sentence case. "Configure the database", not "Configure The Database".
  (`DOC005`)
- Exactly one level-one heading per document. (`DOC001`)
- Never skip levels: no `h3` directly under an `h1`. (`DOC002`)
- Every heading is followed by content, or by subsections. (`DOC003`)
- No links inside headings. (`DOC004`)
- No duplicate heading text, which makes anchors ambiguous. (`DOC006`)
- Task headings use the bare infinitive: "Create an instance", not "Creating an
  instance". Conceptual headings are noun phrases: "Migration to the new API".
- Avoid starting a heading with an `-ing` verb.
- Do not number headings to imply sequence; the reader arrives from a link.
- Optional sections read "Optional: Customize your alias", not "Customize your
  alias (optional)".
- Minimize punctuation. Punctuation in a heading usually means the heading is
  carrying two ideas.

## Lists and procedures

- Numbered when order matters, bulleted when it does not.
- A one-item list is not a list; make it a sentence.
- Introduce a list with a complete sentence, ending in a colon when the list
  follows immediately.
- Keep items parallel in structure.
- Capitalize the first word unless case is meaningful, as with code.
- End items with a period unless the item is a single word, has no verb, is
  entirely code, or is entirely link text.

For procedures specifically:

- One action per step, each starting with an imperative verb.
- State the location before the action: "In the project settings, click
  **Integrations**", not "Click Integrations in the project settings". The
  reader needs to know where to look before being told what to do.
- State the goal before the action: "To rotate the key, run ...".
- A single-step procedure is a bullet, not a numbered list of one.
- Optional steps begin with "Optional:".

## Code in text and in blocks

Use code font for: filenames and paths, environment variables, class, method,
and function names, package names, HTTP verbs and status codes, query
parameters, port numbers, data types, literal text the user types, and command
output.

Do not use code font for product names, domain names in prose, or URLs the
reader is meant to open, which should be links.

Do not inflect code as English: "Call the `close` method", not "`Close` the file".

For code blocks:

- Every fence declares its language. (`CODE001`) A fence without one loses
  syntax highlighting and tells the reader nothing about what they are looking
  at.
- Mark elisions with a comment in the sample's own language, not with `...` or
  an ellipsis character. (`CODE002`) In Python `...` is valid syntax, which
  makes the ambiguity worse than cosmetic.
- Wrap at 80 characters where the language allows it. (`CODE003`)
- Precede every sample with a sentence saying what it does.
- Never paste a real credential, token, or key. Use an obvious placeholder and
  say where the real value comes from.

## Link text

- Descriptive and unique. "See the [authentication guide]", not "click
  [here]". (`LINK001`)
- Meaningful out of context: a screen reader user may hear only the link text.
- Never use a bare URL as link text. (`LINK002`)
- Do not reuse the same link text for two different destinations. (`LINK003`)
- Standard phrasing: "For more information, see [page title]."
- Front-load the important words; do not make a whole sentence a link.

## Accessibility

- Every image has alt text that summarizes its intent. Decorative images get an
  empty alt. Never put information only in an image. (`A11Y001`)
- No directional language for document position. "In the preceding diagram", not
  `in the diagram above`. Position changes with rendering, and it means nothing
  to a screen reader. (`A11Y002`)
- Keep sentences to fewer than 26 words. (`A11Y003`)
- Tables: introduce them in the preceding text, use header cells for the first
  row or column only, and avoid merged cells.
- Never use color as the only carrier of information.
- Define acronyms on first use.

## Words to cut or replace

`scripts/lint_prose.py` flags these. The point is not word policing: each one either
makes a claim the document cannot support, or replaces a fact with a feeling.
"Google" in the last column means the word has its own entry in Google's word
list or tone page; "skill" means this skill added it.

| Word or phrase | Why | Instead | Source |
|---|---|---|---|
| easy, simple, simply | Judges the reader's experience | Give the step count or the time | Google |
| trivial, straightforward, obviously, of course | Same; tells a stuck reader they are slow | Delete | skill |
| blazing fast, lightning fast | Unmeasured | Give the benchmark and its date | skill |
| seamless, effortless, powerful, robust | Marketing | Name the specific property | skill |
| cutting-edge, state-of-the-art, revolutionary | Marketing | Delete | skill |
| just | Filler | Delete | Google |
| really, very, basically | Filler | Delete | skill |
| please note, note that, it should be noted | Placeholder phrase | Delete; the sentence stands alone | Google tone page ("please note"); rest skill |
| utilize | Longer than the word it replaces | use | Google |
| leverage | Same | use, build on | Google |
| in order to | Same | to | Google |
| e.g., i.e. | Often confused | for example, that is | Google |
| click on | Non-standard | click | Google |
| allows you to | Roundabout | lets you | Google |
| enables you to | Roundabout | lets you | skill |
| currently | Implied | Delete | Google |
| please | Not used in instructions | Delete | Google |
| terminate, abort, kill | Harsh and imprecise | stop, cancel, exit | Google |
| whitelist, blacklist | Exclusionary | allowlist, denylist | Google |
| master, slave | Exclusionary | primary, replica | Google |
| dummy (variable) | Exclusionary when it means placeholder | placeholder | Google |
| sanity check | Ableist | quick check, validation | Google |
| and/or | Ambiguous | Rephrase with "or" | Google |
| via | Vague | with, through | Google |
| dumb down, cripple, lame, you guys | Ableist or exclusionary | Rewrite | Google |

Exclamation marks: one is a lot. (`TONE006`)

## Writing for readers who translate

Much of your audience reads English as a second language or through machine
translation. These cost nothing and help everyone:

- Use the short word. "Start", not "commence".
- Avoid phrasal verbs, except the established technical ones: set up, log in,
  sign in.
- Do not stack more than two nouns as modifiers.
- Replace ambiguous pronouns with the noun.
- Keep the optional "that": "If the key is not found, then the default is
  returned."
- No idioms, no humor that depends on wordplay, no culture-specific or
  season-specific references.
- Use identical terminology throughout. Pick "endpoint" and never also say
  "route", "URL", or "path" for the same thing.

## Mode discipline

Diataxis names four kinds of documentation and observes that blurring them is
what produces structurally bad docs. The compass is two questions:

| The content informs | Serving the reader's | So it is |
|---|---|---|
| action | acquisition of skill | a tutorial |
| action | application of skill | a how-to guide |
| cognition | application of skill | reference |
| cognition | acquisition of skill | explanation |

Or in one line each: a tutorial is a lesson, a how-to guide is a recipe,
reference is the information on the back of the packet, explanation is the
article about the cuisine.

In a README these modes sit next to each other, which is why READMEs go wrong:

| README section | Mode | The discipline |
|---|---|---|
| Quickstart | Tutorial | One managed path, no choices, guaranteed to work |
| Usage examples | How-to | Assumes competence, may branch, no teaching |
| API map, configuration | Reference | Austere, factual, mirrors the code structure |
| Background, design notes | Explanation | May discuss alternatives and reasons |

The two failures to watch for, both named by Diataxis:

**Tutorial and how-to collapse**, the most common conflation in software
documentation. A tutorial offers no choices and cannot fail; a how-to guide
assumes the reader knows the goal and must handle the real world's branches. A
quickstart that says "you can also configure X, or use Y instead" has stopped
being a tutorial and become a worse how-to guide.

**Explanation leaking into reference.** Examples are enjoyable to write and grow
into essays. The result is reference the reader cannot scan and explanation that
never gets written properly. Diataxis notes that intuition is not reliable
enough here, so check deliberately: material that is boring, listlike, and
tabular is reference; material you could imagine reading away from a keyboard is
explanation.

One more, from the same source: build structure only around content that
exists. A heading for a tutorial you have not written is worse than its absence,
because it tells the reader the document covers something it does not.

## What the linter enforces

```bash
python3 scripts/lint_prose.py --list-rules
```

Rules are grouped `DOC` (structure), `CODE` (code blocks), `LINK` (link text),
`A11Y` (accessibility), `TONE` (word choice), `HONEST` (unfinished content), and
`RM` (README-specific, from standard-readme). Severities are `error`, `warn`,
`info`; `--fail-on` and `--disable` control CI behavior.

The rules the linter cannot check are the ones that matter most: whether the
first example is the one most readers need, whether a section earns its place,
whether the document answers the question the reader arrived with. Those stay
with you.

## Sources

- Google developer documentation style guide - https://developers.google.com/style
- Highlights - https://developers.google.com/style/highlights
- Voice and tone - https://developers.google.com/style/tone
- Person - https://developers.google.com/style/person
- Headings - https://developers.google.com/style/headings
- Lists - https://developers.google.com/style/lists
- Procedures - https://developers.google.com/style/procedures
- Code in text - https://developers.google.com/style/code-in-text
- Code samples - https://developers.google.com/style/code-samples
- Link text - https://developers.google.com/style/link-text
- Accessibility - https://developers.google.com/style/accessibility
- Writing for a global audience - https://developers.google.com/style/translation
- Word list - https://developers.google.com/style/word-list
- Diataxis, the compass - https://diataxis.fr/compass/
- Diataxis, tutorials and how-to guides - https://diataxis.fr/tutorials-how-to/
- Diataxis, reference and explanation - https://diataxis.fr/reference-explanation/
- Diataxis, how to use it - https://diataxis.fr/how-to-use-diataxis/
