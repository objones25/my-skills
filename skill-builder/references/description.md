# Writing the description

The description is the only part of a skill that is always in context, and Claude uses it to decide whether to load the skill. A good skill with a poor description does nothing.

## Shape

Three parts, in this order:

1. **What:** one clause.
2. **When:** "Use when" followed by situations, symptoms, and the user's own words. "Why is my loss spiking at step 4000" triggers better than "training stability".
3. **Not when:** "Not for X" or "For X use other-skill". Name the sibling that overlaps most.

Example:

> Write and review Dockerfiles to production practice. Use whenever the user asks to containerize, dockerize, shrink, harden or debug a Dockerfile or image, or says "make this run in a container". For Kubernetes manifests use kubernetes-expert.

## Rules

- **Use the words people use.** Include symptoms, error text and tool names. Add a catch-all for requests that never name the topic, such as "even if they never say best practices".
- **Leave out the workflow.** A description summarizing the steps can lead an agent to follow the summary and skip the skill body, according to testing reported in the superpowers `writing-skills` skill.
- **Do not list bundled files.** It spends the budget without helping the decision. State a capability only if it would make someone choose the skill.
- **Pair a proactive trigger with limits.** "Use on any code that awaits" fires often, so it needs negative scope.

## Limits

- Keep it to 1024 characters with no `<` or `>`, which is what skill-creator's validator enforces. `->` contains `>`.
- Claude Code truncates the combined `description` and `when_to_use` at 1,536 characters in its listing. The listing has an overall budget, and when it overflows, descriptions of the least-used skills are dropped.
- Quote the value if it contains `: `. The docs say malformed YAML loads a skill with empty metadata, which would lose the description.

## Testing it

Do not judge a description by reading it. Write `evals/trigger_evals.json` as described in `evals.md`, and run skill-creator's description loop. Near misses are the useful cases: requests sharing the skill's words that belong to something else.
