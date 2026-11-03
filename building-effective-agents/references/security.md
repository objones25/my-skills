# Security: prompt injection and blast radius

Assume a prompt injection will sometimes get through, and design so that a compromised session can't do much. Tool results, fetched pages, emails, documents, and other agents' output are data, not instructions. Filtering them helps; it is not a guarantee.

## The Rule of Two

Meta's *Agents Rule of Two* (Oct 2025): until models can reliably detect and refuse prompt injection, a session should have at most two of these three properties.

- **[A]** It processes untrustworthy input.
- **[B]** It can reach sensitive systems or private data.
- **[C]** It can change state or communicate externally.

If a task needs all three, Meta says the agent should not run autonomously: put a human approval (or another reliable validation) on the step that combines them, or start a fresh session so the combination never exists in one context. Meta credits Simon Willison's "lethal trifecta" (*The lethal trifecta for AI agents*, Jun 2025: private data, untrusted content, and external communication) as an inspiration; it is the same idea framed around data theft.

Exfiltration rarely looks like an obvious send. Willison's point is that any tool able to make an HTTP request (to an API, to load an image, or by giving the user a link to click) can carry stolen data out. In practice that means a rendered image URL with data in its query string, a parameter on an API call, or a link in a generated document. Count all of those as [C].

## Detection doesn't hold; structure does

Filters, classifiers, and prompt tricks lower attack rates on static test sets and then fail against an attacker who adapts. *The Attacker Moves Second* (Nasr, Carlini et al., arXiv 2510.09023) bypassed 12 published defenses, most with over 90% attack success, though most had originally reported near-zero; spotlighting and prompt sandwiching went from about 1% to over 95% under adaptive search, and a human red team broke every challenge. Anthropic reports the same pattern for its own models: about 0.1% injection success on a single attempt and about 5 to 6% after 100 adaptive attempts, and in a red-team test a user-originated exfiltration succeeded 24 of 25 times where only deterministic egress and filesystem controls would have stopped it (*How we contain Claude across products*, May 2026). OpenAI's guidance makes the same turn: injections increasingly look like social engineering, so limit what a fooled agent can do rather than betting on detection (*Designing AI agents to resist prompt injection*, Mar 2026).

Use detectors as one layer. Put the guarantee in structure:

- **Plan, then execute.** The privileged model fixes the plan before it sees untrusted data, so injected text can't add steps. It can still change arguments, such as the body of an email being sent (Beurer-Kellner et al., *Design Patterns for Securing LLM Agents against Prompt Injections*, arXiv 2506.08837).
- **Dual LLM / quarantine.** A quarantined model reads untrusted content and returns values in a fixed schema; the privileged model handles only references to those values. CaMeL builds this with explicit data-flow tracking and capability checks: in AgentDojo it solved 77% of tasks with provable security against 84% undefended, at about 2.8x the tokens, with large utility drops on some suites (Debenedetti et al., arXiv 2503.18813; Google-authored, AgentDojo only).
- **Action selector, map-reduce, context minimization, code-then-execute**: the other patterns in the same paper, each trading flexibility for a narrower attack surface.
- **Egress control.** Allowlist outbound domains at the network layer; deny by default from sandboxes. Anthropic reports an OS sandbox with network denied by default cut permission prompts 84% (prompts that users approved about 93% of the time, which is itself a warning about approval fatigue).

## Credentials

Scope credentials to the acting user, then go further. Scoping still assumes the model can't misuse a limited token. In Anthropic's Managed Agents, an injection in the old coupled design only had to convince Claude to read its own environment. The fix was structural: tokens are never reachable from the sandbox where model-written code runs. Git tokens are wired into the remote when the sandbox is set up, and MCP calls go through a proxy that fetches OAuth tokens from a vault, so the harness never sees them (*Scaling Managed Agents*, Apr 2026).

## Design rules

- Split work so no single session holds A, B, and C. Pass results between sessions as data with a fixed schema, not as free text the next agent will obey.
- Disclose destructive and open-world tools with MCP tool annotations and gate them; risk tiers and approval rules live in `verification.md`.
- Run model-written code in a real sandbox. Hugging Face states that the smolagents local Python executor is not a security sandbox.
- Treat MCP servers, skills, and plugins you didn't write as untrusted code that also feeds untrusted text into context. Tool descriptions can carry injections (tool poisoning) and can change after you approved them; pin versions and diff definitions on update (`mcp.md`).
- Don't let memory launder untrusted text. A fact written to long-term memory from a web page must keep its provenance, or it will later be read back as trusted (`memory.md`).
- Watch for approval fatigue. If humans approve nearly every prompt, the gate is a formality; move routine actions into a sandbox and keep prompts for the few that matter.
- Log every tool call with its arguments. Injection shows up in the tool-call record, not in the model's summary of it.

## Audit script

`scripts/rule_of_two_audit.py manifest.json` takes a manifest of tools (which of A, B, C each confers) and sessions (tools held, human gates, inputs from other sessions) and reports sessions holding all three. A session receiving free text from a session that read untrusted input inherits [A]; a fixed-schema handoff does not.

## Diagnosing

| Symptom | Likely cause | First check |
|---|---|---|
| Agent did something a document, page, or email told it to | Untrusted content in the same session as authority to act | Which of A, B, and C the session held |
| Data appeared somewhere it shouldn't | An outbound channel counted as harmless | Rendered URLs, links, and API parameters in the transcript |
| Agent read or used a secret | Credentials reachable from model-run code | Where tokens live relative to the sandbox |
| Behavior changed after an MCP server update | Tool description changed (rug pull or poisoning) | Diff of tool definitions between versions |
| Defense scored near zero in testing, fails in the field | Tested against static attacks only | Rerun with adaptive attacks; add a structural control |
