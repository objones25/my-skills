# 04: Wrong tool picks, and a bill that jumped after a "small" change

*Illustrative system.* A CRM agent alternates between `search_customers` and `find_customer` unpredictably, and after someone added today's date to the system prompt, the monthly bill rose noticeably.

## Verdict

Two separate interface problems, neither fixed by prompting. Merge or separate the overlapping tools; move the timestamp out of the cached prefix.

## Wrong tool: lint the definitions

```
$ python scripts/tool_lint.py tests/fixtures/tools_overlapping.json
4 tools, ~238 definition tokens
[overlap] search_customers, find_customer: description similarity 0.56, name similarity 0.00
[boundary] search_customers: overlaps find_customer but never says when not to use it
[boundary] find_customer: overlaps search_customers but never says when not to use it
[thin-desc] search_customers: 11 words
[thin-desc] find_customer: 9 words
[vague-param] find_customer: rename ['user'] (e.g. user -> user_id)
[write-unmarked] delete_customer: side-effect name with no annotations (destructiveHint / idempotentHint / openWorldHint)
```

The overlap fails the ambiguity test (`tools.md`): an engineer couldn't say which to call either. Merge them into one search tool, or give each a disjoint job and a sentence saying when not to use it. Rename `user` to the value actually expected. Mark `delete_customer` destructive and gate it. The lint is a screen; confirm with a tool-selection eval on realistic multi-call tasks, since descriptions and schemas can move selection in ways a word-overlap score can't see.

## Bill went up: check cache reuse

Log the request bodies for one run and compare consecutive calls:

```
$ python scripts/cache_prefix_check.py tests/fixtures/requests_cache_break.jsonl
ok  call 0->1: 100% shared
LOW call 1->2: 36% shared; diverges in system: timestamp in system prompt: move it to the end or drop it
     ...'nt. Today is 2026-11-02 10:05:13.' -> ...'nt. Today is 2026-11-02 10:05:41.'
LOW call 2->3: 4% shared; diverges in tools: tool order changed: list tools in a fixed order
```

A timestamp with seconds near the top changes every call, so nothing after it is reused. Tool lists built from a dict or set can reorder between calls with the same effect; MCP's 2026-07-28 spec asks servers to list tools in a deterministic order for this reason. Fix both, then measure hit rate from the provider's usage fields before and after.
