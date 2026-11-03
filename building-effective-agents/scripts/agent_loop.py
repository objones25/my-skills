#!/usr/bin/env python3
"""A small, framework-free agent harness that shows where each control lives.

Read this to see the parts a framework hides; copy pieces into your own loop.
Everything that must not fail is in code here, not in the prompt.

  loop          model call -> tool dispatch -> append result -> repeat, capped by max_steps
  finish tool   the only way to end: status "done" or "could_not_complete" (sanctioned failure)
  budgets       every tool result passes through tool_budget.budget_text
  errors        tool exceptions come back as short, instructive text, not tracebacks
  approval      tools marked needs_approval call an approver before running
  verification  a verify() hook checks "done" against the event log; failure is fed back
  event log     append-only JSONL outside the context window; resume() rebuilds the run

Provider-agnostic: a model is any callable (messages, tool_specs) -> ModelTurn.
ScriptedModel drives tests offline. AnthropicModel is a thin adapter for the
Messages API tool-use format; check it against the current SDK before relying on it.

Usage
  python agent_loop.py --demo         offline run with a scripted model
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

sys.path.insert(0, str(Path(__file__).parent))
from tool_budget import budget_text  # noqa: E402

FINISH = {
    "name": "finish",
    "description": ("End the task. Use status 'done' only when the evidence field cites tool results "
                    "that show the outcome. Use 'could_not_complete' when data is missing or an action "
                    "failed; this is an accepted outcome, not a penalty."),
    "input_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["done", "could_not_complete"]},
            "summary": {"type": "string"},
            "evidence": {"type": "array", "items": {"type": "string"},
                         "description": "tool_call ids whose results support the summary"},
        },
        "required": ["status", "summary"],
    },
}


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class ModelTurn:
    text: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    usage_tokens: int = 0


class Model(Protocol):
    def __call__(self, messages: list[dict], tools: list[dict]) -> ModelTurn: ...


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[..., Any]
    needs_approval: bool = False
    max_result_tokens: int = 2_000

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


@dataclass
class Outcome:
    status: str                  # done | could_not_complete | step_limit | budget_limit | verify_failed
    summary: str
    steps: int
    tokens: int
    log_path: str


class EventLog:
    """Append-only JSONL. The context window is a view over this, never the record."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, kind: str, **data: Any) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps({"t": time.time(), "kind": kind, **data}, default=str) + "\n")

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]


class Agent:
    def __init__(self, model: Model, tools: list[Tool], system: str, *,
                 max_steps: int = 20, max_tokens: int = 200_000,
                 approver: Callable[[ToolCall], bool] | None = None,
                 verify: Callable[[dict, list[dict]], tuple[bool, str]] | None = None,
                 max_verify_retries: int = 2):
        self.model, self.system = model, system
        self.tools = {t.name: t for t in tools}
        self.max_steps, self.max_tokens = max_steps, max_tokens
        self.approver = approver or (lambda call: False)   # deny by default
        self.verify = verify
        self.max_verify_retries = max_verify_retries

    # -- public -----------------------------------------------------------
    def run(self, task: str, log_path: str | os.PathLike) -> Outcome:
        log = EventLog(log_path)
        log.append("task", task=task, system=self.system)
        return self._loop(log, [{"role": "user", "content": task}], steps=0, tokens=0, verify_fails=0)

    def resume(self, log_path: str | os.PathLike) -> Outcome:
        """Rebuild messages from the event log and continue from the last event."""
        log = EventLog(log_path)
        ev = log.events()
        if not ev or ev[0]["kind"] != "task":
            raise ValueError("log has no task event")
        if any(e["kind"] == "outcome" for e in ev):
            last = [e for e in ev if e["kind"] == "outcome"][-1]
            return Outcome(last["status"], last["summary"], last["steps"], last["tokens"], str(log.path))
        messages = [{"role": "user", "content": ev[0]["task"]}]
        steps = tokens = verify_fails = 0
        for e in ev[1:]:
            if e["kind"] == "assistant":
                messages.append({"role": "assistant", "content": e["text"], "calls": e["calls"]})
                steps += 1; tokens += e.get("tokens", 0)
            elif e["kind"] == "tool_result":
                messages.append({"role": "tool", "call_id": e["call_id"], "content": e["content"],
                                 "is_error": e.get("is_error", False)})
            elif e["kind"] == "verify_failed":
                verify_fails += 1
                messages.append({"role": "user", "content": e["feedback"]})
        # tool calls issued but never answered (crash mid-dispatch): answer them now
        answered = {m.get("call_id") for m in messages if m["role"] == "tool"}
        pending = [c for m in messages if m["role"] == "assistant" for c in m["calls"]
                   if c["id"] not in answered and c["name"] != "finish"]
        for c in pending:
            self._dispatch(ToolCall(c["id"], c["name"], c["args"]), log, messages)
        log.append("resumed", steps=steps)
        return self._loop(log, messages, steps, tokens, verify_fails)

    # -- internals --------------------------------------------------------
    def _loop(self, log: EventLog, messages: list[dict], steps: int, tokens: int,
              verify_fails: int) -> Outcome:
        specs = [t.spec() for t in self.tools.values()] + [FINISH]
        while True:
            if steps >= self.max_steps:
                return self._end(log, "step_limit", f"stopped after {steps} steps", steps, tokens)
            if tokens >= self.max_tokens:
                return self._end(log, "budget_limit", f"stopped at ~{tokens} tokens", steps, tokens)
            turn = self.model(messages, specs)
            steps += 1; tokens += turn.usage_tokens
            calls = [c.__dict__ for c in turn.calls]
            log.append("assistant", text=turn.text, calls=calls, tokens=turn.usage_tokens)
            messages.append({"role": "assistant", "content": turn.text, "calls": calls})
            if not turn.calls:
                messages.append({"role": "user", "content":
                                 "No tool was called. Continue, or call finish with a status."})
                continue
            for call in turn.calls:
                if call.name == "finish":
                    result = self._finish(call, log, messages, verify_fails)
                    if result is None:          # verification failed, keep going
                        verify_fails += 1
                        if verify_fails > self.max_verify_retries:
                            return self._end(log, "verify_failed", "claimed done; checks kept failing",
                                             steps, tokens)
                        break
                    return self._end(log, result[0], result[1], steps, tokens)
                self._dispatch(call, log, messages)

    def _dispatch(self, call: ToolCall, log: EventLog, messages: list[dict]) -> None:
        tool = self.tools.get(call.name)
        is_error = False
        if tool is None:
            content, is_error = (f"Unknown tool '{call.name}'. Available: {', '.join(self.tools)}.", True)
        elif tool.needs_approval and not self.approver(call):
            content, is_error = (f"'{call.name}' needs human approval and was not approved. "
                                 "Do not retry it; choose another step or finish with could_not_complete.", True)
        else:
            try:
                raw = tool.fn(**call.args)
                text = raw if isinstance(raw, str) else json.dumps(raw, default=str)
                content = budget_text(text, max_tokens=tool.max_result_tokens).text
            except TypeError as e:
                content, is_error = (f"Bad arguments for {call.name}: {e}. Schema: "
                                     f"{json.dumps(tool.input_schema)}", True)
            except Exception as e:  # report compactly; keep the failure in context
                content, is_error = (f"{call.name} failed: {type(e).__name__}: {str(e)[:300]}", True)
        log.append("tool_result", call_id=call.id, name=call.name, args=call.args,
                   content=content, is_error=is_error)
        messages.append({"role": "tool", "call_id": call.id, "content": content, "is_error": is_error})

    def _finish(self, call: ToolCall, log: EventLog, messages: list[dict],
                verify_fails: int) -> tuple[str, str] | None:
        status = call.args.get("status", "could_not_complete")
        summary = call.args.get("summary", "")
        if status == "done" and self.verify is not None:
            ok, reason = self.verify(call.args, log.events())
            if not ok:
                feedback = (f"Verification failed: {reason}. Fix it, or finish with "
                            "status could_not_complete and say why.")
                log.append("verify_failed", reason=reason, feedback=feedback)
                messages.append({"role": "tool", "call_id": call.id, "content": feedback, "is_error": True})
                return None
        return status, summary

    @staticmethod
    def _end(log: EventLog, status: str, summary: str, steps: int, tokens: int) -> Outcome:
        log.append("outcome", status=status, summary=summary, steps=steps, tokens=tokens)
        return Outcome(status, summary, steps, tokens, str(log.path))


# -- verification helpers ---------------------------------------------------
def evidence_exists(final: dict, events: list[dict]) -> tuple[bool, str]:
    """Every cited evidence id must be a tool result that did not error."""
    cited = final.get("evidence") or []
    if not cited:
        return False, "status done with no evidence ids"
    ok_ids = {e["call_id"] for e in events if e["kind"] == "tool_result" and not e.get("is_error")}
    missing = [c for c in cited if c not in ok_ids]
    return (not missing, f"evidence ids not found among successful tool results: {missing}" if missing else "")


# -- models -------------------------------------------------------------------
class ScriptedModel:
    """Replays a fixed list of turns. For offline tests and demos."""

    def __init__(self, turns: list[ModelTurn]):
        self.turns = list(turns)
        self.seen: list[list[dict]] = []

    def __call__(self, messages: list[dict], tools: list[dict]) -> ModelTurn:
        self.seen.append(list(messages))
        if not self.turns:
            raise RuntimeError("ScriptedModel ran out of turns")
        return self.turns.pop(0)


class AnthropicModel:
    """Adapter for the Anthropic Messages API tool-use format (requires `anthropic`).

    Pass the model id explicitly; ids change, so none is hard-coded here.
    """

    def __init__(self, model: str, system: str, max_tokens: int = 4_096):
        import anthropic  # deferred so tests run without it
        self.client, self.model, self.system, self.max_tokens = anthropic.Anthropic(), model, system, max_tokens

    def __call__(self, messages: list[dict], tools: list[dict]) -> ModelTurn:
        api_msgs: list[dict] = []
        for m in messages:
            if m["role"] == "user":
                api_msgs.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks = ([{"type": "text", "text": m["content"]}] if m["content"] else [])
                blocks += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]}
                           for c in m["calls"]]
                api_msgs.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["call_id"], "content": m["content"],
                         "is_error": m.get("is_error", False)}
                if api_msgs and api_msgs[-1]["role"] == "user" and isinstance(api_msgs[-1]["content"], list):
                    api_msgs[-1]["content"].append(block)
                else:
                    api_msgs.append({"role": "user", "content": [block]})
        r = self.client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                        system=self.system, messages=api_msgs, tools=tools)
        text = "".join(b.text for b in r.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in r.content if b.type == "tool_use"]
        return ModelTurn(text, calls, r.usage.input_tokens + r.usage.output_tokens)


def _demo() -> None:
    orders = {"A1": {"status": "shipped", "carrier": "UPS"}}

    def get_order(order_id: str) -> dict:
        return orders.get(order_id) or {"found": False, "hint": "check the id format, e.g. A1"}

    tools = [Tool("get_order", "Look up one order by its id (format A<digits>).",
                  {"type": "object", "properties": {"order_id": {"type": "string"}},
                   "required": ["order_id"]}, get_order)]
    model = ScriptedModel([
        ModelTurn("Looking up B7.", [ToolCall("c1", "get_order", {"order_id": "B7"})], 50),
        ModelTurn("", [ToolCall("c2", "finish", {"status": "done", "summary": "B7 shipped yesterday"})], 20),
        ModelTurn("", [ToolCall("c3", "finish", {"status": "could_not_complete",
                                                 "summary": "No order B7 found; asked customer to confirm id",
                                                 "evidence": ["c1"]})], 20),
    ])
    log = Path(os.environ.get("TMPDIR", "/tmp")) / f"agent-demo-{uuid.uuid4().hex[:6]}.jsonl"
    out = Agent(model, tools, "Support agent.", verify=evidence_exists).run("Where is order B7?", log)
    print(out)
    print("events:", [e["kind"] for e in EventLog(log).events()])


if __name__ == "__main__":
    if "--demo" in sys.argv:
        _demo()
    else:
        print(__doc__)
