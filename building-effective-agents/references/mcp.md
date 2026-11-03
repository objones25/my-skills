# MCP servers: design, current spec, and safety

The Model Context Protocol (MCP) is a standard way for an agent host to discover and call tools, read resources, and fetch prompts from separate servers. Everything in `tools.md` about tool design applies to MCP tools; this file covers what is specific to the protocol. For writing a server's code, use the mcp-builder skill.

## When MCP is worth it

Build an MCP server when more than one host or team will use the tools, or when you want the tools to live in a separate process with their own credentials. A tool used by one agent in one codebase is often simpler as a plain function in the harness. MCP adds a process boundary, a transport, auth, and versioning; take those on for reuse or isolation, not by default.

## Spec 2026-07-28 (current as of 2026-11-02)

The 2026-07-28 revision is still the latest; the draft changelog was empty when checked. It changed enough that servers written against 2025 revisions need review.

- **Stateless protocol.** Protocol sessions, the `Mcp-Session-Id` header, and the `initialize` handshake are gone. Each request carries the protocol version and client capabilities in `_meta`; servers implement `server/discover`. Keep cross-call state in server-minted handles passed as ordinary tool arguments.
- **Handles are not credentials.** The security guidance replaces "session hijacking" with state handle hijacking: servers must not treat holding a handle as authentication, and should use random handles bound to the user.
- **Long-running work uses the Tasks extension** (`io.modelcontextprotocol/tasks`): the server returns a task with `taskId`, `ttlMs`, and `pollIntervalMs`; clients poll with `tasks/get`, send input with `tasks/update`, and cancel cooperatively with `tasks/cancel`.
- **Multi round-trip requests replace server-initiated requests.** When a tool needs more input mid-call, the server returns `resultType: "input_required"` with `inputRequests`; the client gathers answers and retries the original request with `inputResponses` (and any `requestState`). Elicitation still exists, but it travels inside `inputRequests` instead of as a server-to-client `elicitation/create` call.
- **Deprecated: Sampling, Roots, Logging** (functional for at least 12 months). Call the LLM provider directly instead of Sampling, pass directories as tool parameters or configuration instead of Roots, log to stderr or OpenTelemetry instead of Logging. The HTTP+SSE transport and OAuth dynamic client registration are deprecated too.
- **Caching.** Servers SHOULD return tools in a deterministic order, which keeps the host's prompt prefix cacheable (`production.md`). `tools/list` must not vary per connection, though it may vary by the authorization on the request. List and read results carry `ttlMs` and `cacheScope`.
- **Tracing.** W3C trace context (`traceparent`, `tracestate`, `baggage`) rides in `_meta`, so tool calls can join the host's trace.
- **Tool annotations** remain: `readOnlyHint` (default false), `destructiveHint` (default true), `idempotentHint` (default false), `openWorldHint` (default true). The spec says clients MUST treat annotations as untrusted unless they come from trusted servers. Use them to drive approval UI for servers you trust; never to relax checks on servers you don't.

## Designing MCP tools

- Shape tools around workflows, not endpoints, and budget their output (`tools.md`). MCP hosts cap output too; Claude Code warns above 10,000 tokens and truncates at 25,000 by default.
- Parameter schemas steer selection more than people expect. In controlled single-factor edits across 5 models, schema changes moved tool selection more than name or description changes (Li et al., arXiv 2610.14334, Oct 2026, abstract only, effect sizes not reported there).
- Most published descriptions are weak. Of 856 MCP tool descriptions studied, 97.1% had at least one defect and 56% didn't state their purpose clearly; augmenting descriptions raised success by a median 5.85 points but added steps and made 16.67% of cases worse (Hasan et al., arXiv 2602.14878). Rewrite descriptions against your own eval, not a checklist.
- Retrieval over large catalogs is the bottleneck. In LiveMCPBench (70 servers, 527 tools), retrieval errors were nearly half of failures (Mo et al., arXiv 2508.01780). Prefer on-demand tool loading with a small always-loaded core.
- Expect low consistency on real CRUD work. On MCPMark's 127 tasks the best model reached 52.56% pass@1 and 33.86% pass^4 (Wu et al., arXiv 2509.24002). Plan verification accordingly (`verification.md`).

## Safety: servers are code and text you didn't write

Treat every third-party server as untrusted code that also writes text into your model's context.

- **Tool poisoning.** Instructions hidden in tool descriptions or results steer the agent. MCPTox measured attack success up to 72.8% across 20 agents, with more capable models often more exposed and refusal rates under 3% (Wang et al., arXiv 2508.14925, template-generated attacks). Pin server versions and diff tool definitions on update, since a description can change after you approved it.
- **Spec-named attacks** (2026-07-28 security best practices): confused deputy through OAuth proxies, token passthrough (servers must not accept tokens not issued for them), SSRF through planted discovery URLs, state handle hijacking, local server compromise through malicious startup commands, and authorization URL injection. Clients must show the full command before one-click local installs and should sandbox local servers.
- **Credentials.** Keep OAuth tokens out of the agent's sandbox; route calls through a proxy that attaches them (`security.md`).
- **Scopes.** Request minimal scopes and step up through `WWW-Authenticate` challenges rather than asking for everything up front.
- **Registry.** The official MCP Registry is metadata only and still in preview; it leaves security scanning to package registries and aggregators. Being listed is not a review.

## Testing an MCP server

- Contract tests on the server alone: schemas validate, errors are instructive, outputs respect the size cap, tools list in a stable order.
- Tool-selection evals through a real host and model on realistic multi-call tasks (`tools.md`).
- A poisoning check: run the agent with a deliberately malicious description in a test server and confirm your approval gates and Rule of Two split hold (`security.md`).
