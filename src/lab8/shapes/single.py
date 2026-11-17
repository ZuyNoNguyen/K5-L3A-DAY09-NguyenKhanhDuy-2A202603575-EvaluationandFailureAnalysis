"""Shape B — one ReAct agent. Student work.

Use `lab8.runtime.run_react` with max_steps=8. `decide` may pick a read tool.
It must not call `issue_refund`. After the loop, the host validates the quote
and is the only writer.

A decide() that trusts ticket text over the order row will fail hidden claims
where the customer overstates the delay or cites a superseded clause.

Per-order cap: an order may already have been refunded by an earlier claim.
The read tool `lookup_refund_history` shows how much. The host still enforces
the cap, so pass the full quote to `durability.issue_refund`, not a net amount.

Caps: stop at 8 steps. Count model calls in the Budget so cost stays comparable.

Retry/crash: forward crash_at to the host and let InjectedCrash propagate.
Retry against the same store; the host owns the refund:{claim_id} key and
checkpoint recovery. Never create a new refund key from the seed or run_id.

Trace contract v1 (docs/contracts.md): return a trace_path to a JSON file written in this run,
{"run_id": <same run_id>, "events": [...]}, on every return path including early
ones. `lab8.trace.Trace` does this; your own tracer is fine if it matches.
"""

from __future__ import annotations

from typing import Any

from lab8.db import connect
from lab8.durability import issue_refund
from lab8.mock_llm import complete
from lab8.runtime import run_react
from lab8.schemas import Budget, RunResult, make_run_id
from lab8.shapes.common import finish
from lab8.tools import compute_compensation, get_claim_view, tool_cost
from lab8.trace import Trace


def run_claim(
    claim_id: str,
    *,
    seed: int,
    store_path: str,
    crash_at: str | None = None,
) -> RunResult:
    conn = connect(store_path)
    budget = Budget()
    trace = Trace(make_run_id("single", claim_id, seed))

    def result(status: str, amount: int = 0, reason: str | None = None) -> RunResult:
        trace.add("decision", {"status": status, "proposal_amount": amount, "reason": reason})
        return finish(
            conn, shape="single", claim_id=claim_id, seed=seed,
            status=status, proposal_amount=amount, budget=budget,
            trace=trace, error_reason=reason,
        )

    try:
        budget.add_tool(tool_cost())
        claim = get_claim_view(conn, claim_id)
        # Do not log the ticket: user text can contain secrets or instructions.
        trace.add("claim", {"found": claim["found"]})
        if not claim["found"]:
            return result("failed", reason="unknown_claim")
        if not claim.get("order_id"):
            return result("needs_info", reason="missing_order_id")

        order: dict[str, Any] | None = None

        def decide(observation: dict[str, Any]) -> dict[str, Any]:
            nonlocal order
            last_tool = observation.get("last_tool")
            # MockLLM has no structured tool-call API. This bounded adapter
            # selects read actions from observations; its text cannot authorize
            # writes, change ownership, or override facts in the database.
            complete(budget, seed=seed, purpose="plan", prompt=f"claim review after {last_tool or 'start'}")
            if last_tool is None:
                action = {
                    "tool": "lookup_order",
                    "args": {"order_id": claim["order_id"], "principal_id": claim["principal_id"]},
                }
            elif last_tool == "lookup_order":
                view = observation["last_result"]
                trace.add("order_scope", {"visible": view["visible"], "reason": view["reason"]})
                if not view["visible"]:
                    return {"final": {"status": "escalated", "error_reason": view["reason"]}}
                order = view["order"]
                action = {"tool": "lookup_policy", "args": {}}
            else:
                trace.add("policy", {"policy_count": len(observation["last_result"]["policies"])})
                return {"final": {"status": "ready"}}
            trace.add("read_tool", {"name": action["tool"]})
            return action

        review = run_react(
            conn, budget, observation={"claim": claim}, decide=decide,
            max_steps=8, worker=True,
        )
        final = review["final"]
        if review["stopped"] != "final":
            return result("failed", reason="step_cap")
        if final["status"] != "ready":
            return result(final["status"], reason=final.get("error_reason"))
        if order is None:
            return result("failed", reason="missing_order_observation")

        # Money is computed from the authorized DB view, never ticket text.
        budget.add_tool(tool_cost())
        quote = compute_compensation(conn, order)
        proposal = int(quote["amount_vnd"])
        trace.add("quote", quote)
        if proposal <= 0:
            return result("denied", reason=quote["reason"])

        # Only the host writes. Pass the full quote so it can enforce approval,
        # per-order caps and retry safety in one transaction. Let crashes escape.
        # Persist the attempt before the effect so even an interrupted run has
        # a current trace. This event does not claim that a refund committed.
        trace.add("refund_attempt", {"proposal_amount": proposal})
        trace.write()
        outcome = issue_refund(conn, claim_id, proposal, crash_at=crash_at)
        trace.add("refund_outcome", {"ok": outcome["ok"], "reason": outcome["reason"]})
        if not outcome["ok"]:
            return result("failed" if outcome["conflict"] else "denied", proposal, outcome["reason"])
        return result("resolved", proposal)
    finally:
        conn.close()
