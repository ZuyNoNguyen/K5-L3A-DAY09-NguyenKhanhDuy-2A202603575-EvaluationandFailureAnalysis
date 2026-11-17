"""Shape A — deterministic workflow. Student work.

Signature (do not change):

    run_claim(claim_id, *, seed, store_path, crash_at=None) -> RunResult

Suggested steps. Money stays in code (`lab8.tools.compute_compensation`).

1. Open the store. Read the claim through `get_claim_view` (not SELECT *).
2. Missing order_id -> status `needs_info`. Do not write the ledger.
3. `lookup_order`. If not visible -> `escalated`. Do not copy phone, address, item, internal note.
4. Quote from the order row. Ignore delay numbers written in the ticket.
5. Quote amount 0 -> `denied`. Do not call issue_refund.
6. Quote amount > 0 -> `durability.issue_refund(conn, claim_id, <full quote>)`.
   Pass the FULL quote. The host checks the claim's own approval and subtracts
   what earlier claims already got for the same order (T13, T14). Do not
   subtract it yourself or the host subtracts twice.
   Auth failure or order already refunded -> `denied` (the host writes `policy_denied`).
   `proposal_amount` stays the full quote. Amount conflict -> `failed`.
7. Success -> `resolved`. One ledger row, one `refund_committed` audit.
8. Use MockLLM only to draft the reply text, if you want a model call at all.

The model must not choose the refund amount and must not commit the refund.

Trace contract v1 (docs/contracts.md): return a trace_path to a JSON file written in this run,
{"run_id": <same run_id>, "events": [...]}, on every return path including early
ones. `lab8.trace.Trace` does this; your own tracer is fine if it matches.
"""

from __future__ import annotations

import sqlite3

from lab8.db import connect
from lab8.durability import InjectedCrash, issue_refund
from lab8.schemas import Budget, RunResult, make_run_id
from lab8.shapes.common import finish
from lab8.tools import compute_compensation, get_claim_view, lookup_order, tool_cost
from lab8.trace import Trace


def run_claim(
    claim_id: str,
    *,
    seed: int,
    store_path: str,
    crash_at: str | None = None,
) -> RunResult:
    conn = connect(store_path)
    try:
        return _run(conn, claim_id, seed=seed, crash_at=crash_at)
    finally:
        conn.close()


def _run(
    conn: sqlite3.Connection,
    claim_id: str,
    *,
    seed: int,
    crash_at: str | None,
) -> RunResult:
    budget = Budget()
    trace = Trace(make_run_id("workflow", claim_id, seed))

    # 1. Read the claim through the allowlist. Never log the ticket text: it is
    #    untrusted user data and may carry PII or a prompt injection.
    claim = get_claim_view(conn, claim_id)
    budget.add_tool(tool_cost())
    trace.add(
        "claim_view",
        {key: claim.get(key) for key in ("found", "claim_id", "order_id")},
    )
    if not claim["found"]:
        return _finish(conn, claim_id, seed, budget, trace, "failed", 0, "unknown_claim")

    # 2. Missing order_id -> needs_info. Do not invent an order id from the ticket.
    order_id = claim.get("order_id")
    if not order_id:
        return _finish(conn, claim_id, seed, budget, trace, "needs_info", 0, "missing_order_id")

    # 3. Ownership + existence are checked by the host read tool.
    order_view = lookup_order(conn, order_id, claim["principal_id"])
    budget.add_tool(tool_cost())
    trace.add("order_view", order_view)
    if not order_view["visible"]:
        return _finish(conn, claim_id, seed, budget, trace, "escalated", 0, order_view["reason"])

    # 4. Money is computed by code from structured order fields, never from the ticket.
    quote_payload = compute_compensation(conn, order_view["order"])
    budget.add_tool(tool_cost())
    proposal = int(quote_payload["amount_vnd"])
    trace.add("quote", quote_payload)

    # 5. Quote 0 -> denied without touching the ledger (no policy_denied audit).
    if proposal <= 0:
        return _finish(conn, claim_id, seed, budget, trace, "denied", 0, quote_payload.get("reason"))

    # 6. Host is the only writer. Pass the FULL quote; the host subtracts what the
    #    order already received and caps at approval/paid. Forward crash_at and let
    #    InjectedCrash propagate so the crash window is real.
    try:
        outcome = issue_refund(conn, claim_id, proposal, crash_at=crash_at)
    except InjectedCrash:
        trace.add("crash", {"at": crash_at})
        trace.write()
        raise

    if not outcome["ok"]:
        status = "failed" if outcome["conflict"] else "denied"
        return _finish(conn, claim_id, seed, budget, trace, status, proposal, outcome["reason"])

    return _finish(conn, claim_id, seed, budget, trace, "resolved", proposal, None)


def _finish(
    conn: sqlite3.Connection,
    claim_id: str,
    seed: int,
    budget: Budget,
    trace: Trace,
    status: str,
    proposal_amount: int,
    error_reason: str | None,
) -> RunResult:
    trace.add("decision", {"status": status, "reason": error_reason})
    return finish(
        conn,
        shape="workflow",
        claim_id=claim_id,
        seed=seed,
        status=status,
        proposal_amount=proposal_amount,
        budget=budget,
        trace=trace,
        error_reason=error_reason,
    )
