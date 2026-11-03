# 05: "Should we add a memory layer?"

*Illustrative systems.* Team A runs a coding agent on one repo. Team B runs a personal scheduling assistant that people use every day. Both are considering a memory product.

## Verdict

Team A probably shouldn't; Team B should, but the hard part is updates, not storage.

## Team A: coding agent

The repo and a progress file already hold the state that matters, and the agent can grep both. In MemGym, memory was roughly neutral on coding for that reason, and a controlled comparison of 11 memory substrates found several that added large latency on a coding suite while scoring below no memory (`memory.md`). What Team A needs is a notes file the agent re-reads, plus compaction or tool-result clearing for long sessions (`context.md`). Test that before buying anything.

## Team B: scheduling assistant

Preferences (meeting hours, time zone, people to avoid double-booking) must survive across sessions, and re-asking every day is the product failing. Memory pays here. Design points:

- **Schema first.** Preferences are key-value with provenance (`said by user`, `inferred`, date). The calendar itself stays in the calendar API, read at use time, never copied into memory.
- **Updates replace, they don't append.** When the user says "I moved to Denver," write a supersession and remove or mark the old time zone. Benchmarks find append-only memory doing worse than no memory once facts reverse, and a prompt rule to prefer the newest fact barely helps.
- **Read narrowly.** Load the few preferences relevant to the request, not the whole profile on every turn.
- **User-visible.** A "what do you remember about me" view with delete.

## Eval before shipping

Tasks that write in session 1 and use the fact in session 4, with a reversal in session 3 and distractor sessions in between. Grade the outcome (the meeting landed in the right time zone). Include a no-memory baseline; if memory doesn't beat it, it doesn't ship.
