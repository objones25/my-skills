"""Inspect AI template for an outcome-graded, multi-trial agent eval.

Checked against inspect_ai 0.3.287 (signatures read with Python's `inspect`
module on 2026-11-02). Inspect moves fast; re-check names after upgrading.

What it shows
  - samples drawn from real failures, including a negative case (the right
    answer is "not found")
  - a react() agent with a tool and a sandbox-free fake backend
  - a scorer that grades the outcome (final answer AND the tool-call record),
    not the exact path, with partial credit
  - several epochs per sample reduced with pass_at(k) and pass_k(k) (pass^k)
  - message / token limits so a looping agent ends instead of burning budget

Run against a real model
  inspect eval scripts/inspect_eval_template.py --model anthropic/<model-id>
  inspect view                       # browse transcripts; read them, don't just read the score

Offline smoke test (no API key): python scripts/inspect_eval_template.py --smoke
For code-writing agents add sandbox="docker" to Task and use the bash()/python()
tools, which run inside the sandbox rather than on the host.
"""
from __future__ import annotations

import tempfile

import sys

from inspect_ai import Epochs, Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
from inspect_ai.scorer import (CORRECT, INCORRECT, PARTIAL, Score, Target, accuracy,
                               pass_at, pass_k, scorer, stderr)
from inspect_ai.solver import TaskState
from inspect_ai.tool import tool

ORDERS = {"A100": "shipped", "A200": "processing"}


@tool
def get_order():
    async def execute(order_id: str) -> str:
        """Look up one order's status.

        Args:
            order_id: Order id in the form A<digits>, e.g. A100. Returns NOT_FOUND if no such order.
        """
        return ORDERS.get(order_id, "NOT_FOUND")

    return execute


def dataset() -> list[Sample]:
    return [
        Sample(id="shipped", input="Customer asks: where is order A100?",
               target="shipped", metadata={"order_id": "A100"}),
        Sample(id="processing", input="Has order A200 gone out yet?",
               target="processing", metadata={"order_id": "A200"}),
        # negative case: the correct outcome is to say it does not exist
        Sample(id="missing", input="Where is my order A999?",
               target="not found", metadata={"order_id": "A999"}),
    ]


def looked_up(state: TaskState, order_id: str) -> bool:
    """True if the agent actually called get_order with this id (tool-call record, not its claim)."""
    for m in state.messages:
        if isinstance(m, ChatMessageAssistant) and m.tool_calls:
            for c in m.tool_calls:
                if c.function == "get_order" and c.arguments.get("order_id") == order_id:
                    return True
    return False


@scorer(metrics=[accuracy(), stderr()])
def order_outcome():
    async def score(state: TaskState, target: Target) -> Score:
        answer = (state.output.completion or "").lower()
        did_lookup = looked_up(state, state.metadata["order_id"])
        right = target.text in answer
        if target.text == "not found":
            # fabricating a status for a missing order is the failure we care most about
            right = any(p in answer for p in ("not found", "no order", "couldn't find", "could not find"))
            right = right and not any(s in answer for s in ORDERS.values())
        if right and did_lookup:
            value = CORRECT
        elif right or did_lookup:
            value = PARTIAL   # right answer without evidence, or looked up but answered wrong
        else:
            value = INCORRECT
        return Score(value=value, answer=answer[:200],
                     explanation=f"lookup={did_lookup} answer_ok={right}")

    return score


def reducers_for(epochs: int) -> list:
    reducers = [pass_at(1)]
    if epochs >= 2:
        reducers += [pass_at(2), pass_k(2)]
    if epochs > 2:
        reducers.append(pass_k(epochs))
    return reducers


@task
def order_status(epochs: int = 4):
    return Task(
        dataset=dataset(),
        solver=react(
            prompt="You answer order-status questions. Always look the order up; "
                   "if the tool returns NOT_FOUND, say the order was not found.",
            tools=[get_order()],
        ),
        scorer=order_outcome(),
        # pass_at / pass_k need k <= epochs. PARTIAL counts as 0.5, below the 1.0 threshold.
        epochs=Epochs(epochs, reducers_for(epochs)),
        message_limit=20,
        token_limit=50_000,
    )


def _smoke() -> None:
    """Run offline with scripted mock outputs to check the plumbing (not the agent)."""
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import ModelOutput, get_model

    import re
    answers = {"A100": "Order A100 has shipped.",
               "A200": "Order A200 is still processing.",
               "A999": "Order A999 shipped yesterday."}   # a fabrication; the scorer should fail it

    def scripted(input, tools, tool_choice, config):
        # Decide from the conversation, not from call order: samples and epochs run concurrently.
        text = " ".join(str(m.content) for m in input)
        oid = re.search(r"A\d+", text).group(0)
        if not any(isinstance(m, ChatMessageTool) for m in input):
            return ModelOutput.for_tool_call("mockllm/model", "get_order", {"order_id": oid})
        return ModelOutput.for_tool_call("mockllm/model", "submit", {"answer": answers[oid]})

    model = get_model("mockllm/model", custom_outputs=scripted)
    t = order_status(epochs=2)
    logs = inspect_eval(t, model=model, display="none", log_dir=tempfile.mkdtemp(prefix="inspect-smoke-"))
    log = logs[0]
    print("status:", log.status)
    for s in log.results.scores:
        print(s.reducer, {k: round(v.value, 3) for k, v in s.metrics.items()})
    return log


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        _smoke()
    else:
        print(__doc__)
