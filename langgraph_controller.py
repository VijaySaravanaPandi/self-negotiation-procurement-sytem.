"""
langgraph_controller.py  —  LangGraph StateGraph Multi-Agent Orchestrator
══════════════════════════════════════════════════════════════════════════
Orchestrates a 4-Agent Autonomous Procurement Network using LangGraph:
  1. Factory Agent      (Port 8001): Buyer Goal & CSP Requirements
  2. Supplier Agent     (Port 8002): Generative LLM Bargaining (Groq)
  3. Auditor Agent      (Port 8003): Symbolic Horn-Clause Verification
  4. Logistics Agent    (Port 8004): Freight Feasibility & Optimal Carrier Selection

Persistence:
  State transitions, round traces, and multilateral contracts are
  persisted to MongoDB (with resilient local JSON fallback).

Usage:
  python langgraph_controller.py --scenario baseline
  python langgraph_controller.py --scenario clash
  python langgraph_controller.py --scenario shock
  python langgraph_controller.py --scenario lowball
  python langgraph_controller.py --scenario stress
  python langgraph_controller.py --scenario all
  python langgraph_controller.py --mermaid
══════════════════════════════════════════════════════════════════════════
"""

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import httpx
import yaml
from langgraph.graph import StateGraph, END

from mongo_store import mongo_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [LangGraph] %(message)s")
logger = logging.getLogger("langgraph_controller")

# ── Load config ──────────────────────────────────────────────────
_cfg_path = Path(__file__).parent / "config.yaml"
with open(_cfg_path, "r", encoding="utf-8") as f:
    _CFG = yaml.safe_load(f)

PORTS = {
    "factory": 8001,
    "supplier": 8002,
    "auditor": 8003,
    "logistics": 8004,
}
FACTORY_URL   = f"http://localhost:{PORTS['factory']}"
SUPPLIER_URL  = f"http://localhost:{PORTS['supplier']}"
AUDITOR_URL   = f"http://localhost:{PORTS['auditor']}"
LOGISTICS_URL = f"http://localhost:{PORTS['logistics']}"

ROUND_LIMIT = _CFG.get("negotiation", {}).get("round_limit", 6)
SHOCK_CFG   = _CFG.get("shock", {})

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "description": "Ample inventory, low pressure → fast consensus with optimal logistics",
        "factory_override": {},
        "supplier_override": {"pricing_pressure": "low", "cost_floor_init": 60.0},
        "shock_enabled": False,
        "adversarial_prompt": False,
    },
    "clash": {
        "description": "Factory price ceiling ($50) below Supplier cost floor ($60) → Horn-clause impasse",
        "factory_override": {"max_price": 50.0},
        "supplier_override": {"pricing_pressure": "low", "cost_floor_init": 60.0},
        "shock_enabled": False,
        "adversarial_prompt": False,
    },
    "shock": {
        "description": "Price shock at round 3 alters supplier cost basis mid-negotiation",
        "factory_override": {},
        "supplier_override": {"pricing_pressure": "low", "cost_floor_init": 55.0},
        "shock_enabled": True,
        "adversarial_prompt": False,
    },
    "lowball": {
        "description": "Suspicious below-cost offer under high pressure → flagged by Auditor",
        "factory_override": {},
        "supplier_override": {"pricing_pressure": "high", "cost_floor_init": 70.0},
        "shock_enabled": False,
        "adversarial_prompt": True,
    },
    "stress": {
        "description": "Impossible constraints ($1 price, 1-day deadline) → tested against round limits",
        "factory_override": {"max_price": 1.0, "deadline": 1},
        "supplier_override": {"pricing_pressure": "low", "cost_floor_init": 60.0},
        "shock_enabled": False,
        "adversarial_prompt": False,
    },
}


# ── Typed State for LangGraph ────────────────────────────────────

class NegotiationGraphState(TypedDict):
    session_id: str
    scenario: str
    round_num: int
    round_limit: int
    shock_enabled: bool
    shock_applied: bool
    shock_trigger_round: int
    adversarial_prompt: bool

    # Factory constraint facts
    item: str
    min_qty: int
    max_price: float
    deadline: int

    # Current offer exchanged
    offer_price: Optional[float]
    offer_qty: Optional[int]
    offer_lead_time: Optional[int]
    pricing_pressure: str
    cost_floor: float
    nl_text: str

    # Auditor evaluation
    audit_verdict: Optional[str]
    violated_clauses: List[str]
    suspicious: bool
    audit_explanation: Optional[str]

    # Logistics evaluation
    logistics_feasible: Optional[bool]
    logistics_status: Optional[str]
    logistics_recommended_carrier: Optional[Dict[str, Any]]
    logistics_feedback: Optional[str]

    # Factory decision
    factory_decision: Optional[str]     # ACCEPT | REJECT

    # Episode resolution
    status: str                         # IN_PROGRESS | SUCCESS | FAILED
    final_contract: Optional[Dict[str, Any]]
    round_traces: List[Dict[str, Any]]


# ── HTTP Helpers ─────────────────────────────────────────────────

def _safe_post(url: str, path: str, payload: Dict[str, Any], timeout: float = 35.0) -> Dict[str, Any]:
    with httpx.Client(timeout=timeout) as client:
        r = client.post(f"{url}{path}", json=payload)
        r.raise_for_status()
        return r.json()


def _check_all_agents() -> bool:
    for url, name in [
        (FACTORY_URL, "Factory (8001)"),
        (SUPPLIER_URL, "Supplier (8002)"),
        (AUDITOR_URL, "Auditor (8003)"),
        (LOGISTICS_URL, "Logistics (8004)"),
    ]:
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(f"{url}/health")
                r.raise_for_status()
        except Exception as e:
            logger.error("Service %s is not reachable at %s: %s", name, url, e)
            return False
    return True


# ── LangGraph Nodes ──────────────────────────────────────────────

def init_session_node(state: NegotiationGraphState) -> NegotiationGraphState:
    """Reset all 4 agent microservices and initialize requirements for the chosen scenario."""
    sid = state["session_id"]
    scenario = state["scenario"]
    sc = SCENARIOS.get(scenario, SCENARIOS["baseline"])

    logger.info("═" * 60)
    logger.info("LangGraph Node: [InitSession] Session=%s Scenario=%s", sid, scenario.upper())
    logger.info("═" * 60)

    # 1. Reset all 4 microservices with session_id
    for url in [FACTORY_URL, SUPPLIER_URL, AUDITOR_URL, LOGISTICS_URL]:
        try:
            _safe_post(url, "/reset", {"session_id": sid})
        except Exception as e:
            logger.warning("Reset warning for %s: %s", url, e)

    # 2. Reset Factory requirements to defaults + scenario overrides
    f_defaults = {
        "item": _CFG["factory"]["item"],
        "min_qty": _CFG["factory"]["min_qty"],
        "max_price": _CFG["factory"]["max_price"],
        "deadline": _CFG["factory"]["deadline"],
    }
    f_defaults.update(sc.get("factory_override", {}))
    _safe_post(FACTORY_URL, "/set_requirement", f_defaults)

    f_req = _safe_post(FACTORY_URL, "/requirement", {"session_id": sid})

    # 3. Dispatch requirements to Supplier and constraints to Auditor
    _safe_post(SUPPLIER_URL, "/receive_requirement", {
        "session_id": sid,
        "item": f_req["item"],
        "min_qty": f_req["min_qty"],
        "max_price": f_req["max_price"],
        "deadline": f_req["deadline"],
        "round_limit": f_req.get("round_limit", ROUND_LIMIT),
    })

    _safe_post(AUDITOR_URL, "/set_constraints", {
        "session_id": sid,
        "factory_max_price": f_req["max_price"],
        "factory_min_qty": f_req["min_qty"],
        "factory_deadline": f_req["deadline"],
    })

    state["item"] = f_req["item"]
    state["min_qty"] = f_req["min_qty"]
    state["max_price"] = f_req["max_price"]
    state["deadline"] = f_req["deadline"]
    state["round_num"] = 0
    state["round_traces"] = []
    state["status"] = "IN_PROGRESS"

    # Persist session init document to MongoDB
    mongo_store.log_session({
        "session_id": sid,
        "scenario": scenario,
        "status": "INITIALIZED",
        "orchestrator": "LangGraph",
        "constraints": f_req,
    })

    return state


def supplier_node(state: NegotiationGraphState) -> NegotiationGraphState:
    """Supplier generates or revises an offer using the Groq LLM."""
    state["round_num"] += 1
    rn = state["round_num"]
    sid = state["session_id"]

    logger.info("─" * 50)
    logger.info("LangGraph Node: [Supplier] Round %d / %d", rn, state["round_limit"])

    is_shock_round = False
    if (
        state["shock_enabled"]
        and SHOCK_CFG.get("enabled", True)
        and rn == state["shock_trigger_round"]
        and not state["shock_applied"]
    ):
        logger.warning("⚡ Applying mid-negotiation price shock at round %d!", rn)
        shock_resp = _safe_post(SUPPLIER_URL, "/shock", {
            "session_id": sid,
            "round_num": rn,
        })
        _safe_post(FACTORY_URL, "/shock_alert", {
            "round_num": rn,
            "new_pressure": shock_resp.get("pricing_pressure", "high"),
            "cost_floor_multiplier": SHOCK_CFG.get("cost_floor_multiplier", 1.25),
        })
        state["shock_applied"] = True
        is_shock_round = True

    if rn == 1:
        offer = _safe_post(SUPPLIER_URL, "/generate_offer", {
            "session_id": sid,
            "round_num": rn,
        })
    else:
        # Pass feedback from Auditor and Logistics to prompt intelligent revision
        violated = state.get("violated_clauses", [])
        fb_parts = []
        if violated:
            fb_parts.append(f"Auditor: {', '.join(violated)}")
        if state.get("logistics_feedback"):
            fb_parts.append(f"Logistics: {state['logistics_feedback']}")

        feedback_text = " | ".join(fb_parts) or ", ".join(violated)

        offer = _safe_post(SUPPLIER_URL, "/revise_offer", {
            "session_id": sid,
            "round_num": rn,
            "violated_clauses": violated,
            "previous_offer_price": state.get("offer_price") or 0.0,
            "previous_offer_qty": state.get("offer_qty") or 0,
            "previous_offer_lead_time": state.get("offer_lead_time") or 0,
            "feedback_text": feedback_text,
        })

    state["offer_price"] = float(offer.get("offer_price") or 0.0)
    state["offer_qty"] = int(offer.get("offer_qty") or 0)
    state["offer_lead_time"] = int(offer.get("offer_lead_time") or 0)
    state["pricing_pressure"] = offer.get("pricing_pressure", "low")
    state["cost_floor"] = float(offer.get("cost_floor", 60.0))
    state["nl_text"] = offer.get("nl_text", "")

    logger.info("Supplier generated NL offer: %s...", state["nl_text"][:120].replace('\n', ' '))
    logger.info("Parsed: price=$%.2f, qty=%d, lead_time=%d days",
                state["offer_price"], state["offer_qty"], state["offer_lead_time"])

    return state


def auditor_node(state: NegotiationGraphState) -> NegotiationGraphState:
    """Auditor performs symbolic forward-chaining Horn-clause inference."""
    sid = state["session_id"]
    rn = state["round_num"]

    audit_resp = _safe_post(AUDITOR_URL, "/audit", {
        "session_id": sid,
        "round_num": rn,
        "offer_price": state["offer_price"],
        "offer_qty": state["offer_qty"],
        "offer_lead_time": state["offer_lead_time"],
        "pricing_pressure": state["pricing_pressure"],
        "cost_floor": state["cost_floor"],
        "nl_text": state["nl_text"],
        "is_shock_round": state["shock_applied"] and (rn == state["shock_trigger_round"]),
    })

    verdict = audit_resp.get("verdict", "REJECTED")
    violated = audit_resp.get("violated_clauses", [])
    suspicious = audit_resp.get("suspicious", False)

    state["audit_verdict"] = verdict
    state["violated_clauses"] = violated
    state["suspicious"] = suspicious
    state["audit_explanation"] = audit_resp.get("explanation", "")

    logger.info("Auditor verdict: %s | Violated: %s | Suspicious: %s", verdict, violated, suspicious)

    # Persist audit inference event to MongoDB
    mongo_store.log_audit_trace({
        "session_id": sid,
        "round_num": rn,
        "verdict": verdict,
        "violated_clauses": violated,
        "suspicious": suspicious,
        "facts": {
            "price": state["offer_price"],
            "qty": state["offer_qty"],
            "lead_time": state["offer_lead_time"],
            "pressure": state["pricing_pressure"],
            "cost_floor": state["cost_floor"],
        },
    })

    return state


def logistics_node(state: NegotiationGraphState) -> NegotiationGraphState:
    """Logistics Agent evaluates carrier options, transit duration, and delivery feasibility."""
    sid = state["session_id"]
    rn = state["round_num"]

    log_resp = _safe_post(LOGISTICS_URL, "/evaluate_shipment", {
        "session_id": sid,
        "item": state["item"],
        "quantity": state["offer_qty"],
        "supplier_lead_time_days": state["offer_lead_time"],
        "factory_deadline_days": state["deadline"],
    })

    feasible = log_resp.get("feasible", False)
    rec_carrier = log_resp.get("recommended_carrier")
    explanation = log_resp.get("explanation", "")

    state["logistics_feasible"] = feasible
    state["logistics_status"] = log_resp.get("status", "UNKNOWN")
    state["logistics_recommended_carrier"] = rec_carrier
    state["logistics_feedback"] = explanation

    carrier_name = rec_carrier.get("carrier_name") if rec_carrier else "None"
    transit = rec_carrier.get("transit_days", 0) if rec_carrier else 0
    logger.info("Logistics evaluation: Feasible=%s | Carrier=%s (%dd transit)", feasible, carrier_name, transit)

    # If logistics route is infeasible, inform downstream that delivery constraint is violated
    if not feasible and "valid_delivery" not in state["violated_clauses"]:
        state["violated_clauses"].append("valid_delivery")

    # Factory receives offer and registers decision
    factory_resp = _safe_post(FACTORY_URL, "/receive_offer", {
        "session_id": sid,
        "round_num": rn,
        "offer_price": state["offer_price"],
        "offer_qty": state["offer_qty"],
        "offer_lead_time": state["offer_lead_time"],
        "verdict": state["audit_verdict"],
        "violated_clauses": state["violated_clauses"],
        "nl_text": state["nl_text"],
    })
    state["factory_decision"] = factory_resp.get("factory_decision", "REJECT")

    # Save round trace snapshot
    trace_item = {
        "round_num": rn,
        "offer_price": state["offer_price"],
        "offer_qty": state["offer_qty"],
        "offer_lead_time": state["offer_lead_time"],
        "verdict": state["audit_verdict"],
        "violated_clauses": state["violated_clauses"],
        "suspicious": state["suspicious"],
        "logistics_feasible": feasible,
        "recommended_carrier": carrier_name,
        "factory_decision": state["factory_decision"],
        "nl_text": state["nl_text"],
    }
    state["round_traces"].append(trace_item)

    return state


def finalize_contract_node(state: NegotiationGraphState) -> NegotiationGraphState:
    """Finalize the 4-way contract across Factory, Supplier, Auditor, and Logistics."""
    sid = state["session_id"]
    rn = state["round_num"]

    logger.info("LangGraph Node: [FinalizeContract] Agreement ratified in Round %d!", rn)

    # 1. Auditor signs and emits contract
    _safe_post(AUDITOR_URL, "/emit_contract", {
        "session_id": sid,
        "round_num": rn,
        "offer_price": state["offer_price"],
        "offer_qty": state["offer_qty"],
        "offer_lead_time": state["offer_lead_time"],
        "scenario": state["scenario"],
    })

    # 2. Logistics books freight consignment
    booking = {}
    if state.get("logistics_recommended_carrier"):
        cid = state["logistics_recommended_carrier"]["carrier_id"]
        booking = _safe_post(LOGISTICS_URL, "/book_shipment", {
            "session_id": sid,
            "carrier_id": cid,
            "item": state["item"],
            "quantity": state["offer_qty"],
            "contract_price_per_unit": state["offer_price"],
        })

    state["status"] = "SUCCESS"
    final_contract = {
        "session_id": sid,
        "scenario": state["scenario"],
        "status": "SUCCESS",
        "rounds_to_agreement": rn,
        "final_price": state["offer_price"],
        "final_qty": state["offer_qty"],
        "final_lead_time": state["offer_lead_time"],
        "logistics_booking": booking,
        "orchestrator": "LangGraph",
        "traces": state["round_traces"],
    }
    state["final_contract"] = final_contract

    # Persist finalized multilateral agreement to MongoDB
    mongo_store.log_contract(final_contract)
    mongo_store.log_session(final_contract)

    return state


def abort_negotiation_node(state: NegotiationGraphState) -> NegotiationGraphState:
    """Terminate negotiation when constraints cannot be satisfied within round limit."""
    sid = state["session_id"]
    rn = state["round_num"]

    logger.warning("LangGraph Node: [AbortNegotiation] Round limit %d exhausted. Negotiation aborted.", rn)

    state["status"] = "FAILED"
    failed_doc = {
        "session_id": sid,
        "scenario": state["scenario"],
        "status": "FAILED",
        "rounds_exhausted": rn,
        "orchestrator": "LangGraph",
        "last_violated_clauses": state.get("violated_clauses", []),
        "last_logistics_status": state.get("logistics_status"),
        "traces": state["round_traces"],
    }
    mongo_store.log_session(failed_doc)

    return state


# ── Conditional Routing Function ──────────────────────────────────

def consensus_router(state: NegotiationGraphState) -> str:
    """
    Evaluates whether contract consensus has been achieved:
    Requires:
      1. Auditor Horn-clause verdict == "VALID"
      2. Factory decision == "ACCEPT"
      3. Logistics route feasible == True
    """
    is_audit_valid = (state.get("audit_verdict") == "VALID")
    is_factory_accept = (state.get("factory_decision") == "ACCEPT")
    is_logistics_ok = bool(state.get("logistics_feasible"))

    if is_audit_valid and is_factory_accept and is_logistics_ok:
        return "finalize_contract"

    if state["round_num"] >= state["round_limit"]:
        return "abort_negotiation"

    return "continue_negotiating"


# ── LangGraph Workflow Construction ───────────────────────────────

def build_procurement_graph() -> StateGraph:
    """Constructs the cyclic StateGraph for the 4-agent network."""
    workflow = StateGraph(NegotiationGraphState)

    # Register Nodes
    workflow.add_node("init_session", init_session_node)
    workflow.add_node("supplier_generate", supplier_node)
    workflow.add_node("auditor_eval", auditor_node)
    workflow.add_node("logistics_eval", logistics_node)
    workflow.add_node("finalize_contract", finalize_contract_node)
    workflow.add_node("abort_negotiation", abort_negotiation_node)

    # Initial linear flow
    workflow.set_entry_point("init_session")
    workflow.add_edge("init_session", "supplier_generate")
    workflow.add_edge("supplier_generate", "auditor_eval")
    workflow.add_edge("auditor_eval", "logistics_eval")

    # Conditional Routing based on Multilateral Consensus
    workflow.add_conditional_edges(
        "logistics_eval",
        consensus_router,
        {
            "finalize_contract": "finalize_contract",
            "abort_negotiation": "abort_negotiation",
            "continue_negotiating": "supplier_generate",
        },
    )

    # Terminal transitions
    workflow.add_edge("finalize_contract", END)
    workflow.add_edge("abort_negotiation", END)

    return workflow.compile()


# ── Runner ────────────────────────────────────────────────────────

def run_scenario(scenario_name: str) -> Dict[str, Any]:
    """Executes a single scenario through the LangGraph StateGraph."""
    if not _check_all_agents():
        print("ERROR: Not all 4 microservices are running.")
        print("Please ensure the following ports are active:")
        print("  - Port 8001: Factory Agent   (uvicorn factory_agent:app --port 8001)")
        print("  - Port 8002: Supplier Agent  (uvicorn supplier_agent:app --port 8002)")
        print("  - Port 8003: Auditor Agent   (uvicorn auditor_agent:app --port 8003)")
        print("  - Port 8004: Logistics Agent (uvicorn logistics_agent:app --port 8004)")
        sys.exit(1)

    sc = SCENARIOS.get(scenario_name, SCENARIOS["baseline"])
    sid = str(uuid.uuid4())[:8]

    initial_state: NegotiationGraphState = {
        "session_id": sid,
        "scenario": scenario_name,
        "round_num": 0,
        "round_limit": ROUND_LIMIT,
        "shock_enabled": sc.get("shock_enabled", False),
        "shock_applied": False,
        "shock_trigger_round": SHOCK_CFG.get("trigger_round", 3),
        "adversarial_prompt": sc.get("adversarial_prompt", False),
        "item": "steel_rods",
        "min_qty": 500,
        "max_price": 85.0,
        "deadline": 7,
        "offer_price": None,
        "offer_qty": None,
        "offer_lead_time": None,
        "pricing_pressure": "low",
        "cost_floor": 60.0,
        "nl_text": "",
        "audit_verdict": None,
        "violated_clauses": [],
        "suspicious": False,
        "audit_explanation": None,
        "logistics_feasible": None,
        "logistics_status": None,
        "logistics_recommended_carrier": None,
        "logistics_feedback": None,
        "factory_decision": None,
        "status": "IN_PROGRESS",
        "final_contract": None,
        "round_traces": [],
    }

    graph = build_procurement_graph()
    logger.info("Executing LangGraph State Machine for Scenario: %s", scenario_name.upper())

    final_state = graph.invoke(initial_state)

    print("\n" + "=" * 70)
    print(f"LANGGRAPH EXECUTION SUMMARY: {scenario_name.upper()}")
    print(f"Session ID : {final_state['session_id']}")
    print(f"Status     : {final_state['status']}")
    print(f"Rounds     : {final_state['round_num']}")
    if final_state["status"] == "SUCCESS":
        c = final_state.get("final_contract", {})
        bk = c.get("logistics_booking", {})
        print(f"Agreed Deal: ${final_state['offer_price']:.2f}/unit | {final_state['offer_qty']} units | {final_state['offer_lead_time']} days production")
        if bk:
            print(f"Freight    : {bk.get('carrier_name')} ({bk.get('mode')}) | Transit: {bk.get('transit_days')}d | Cost: ${bk.get('freight_cost_per_unit'):.2f}/unit")
    else:
        print(f"Outcome    : Impasse reached. Last Violated Clauses: {final_state.get('violated_clauses')}")
    print("=" * 70 + "\n")

    return final_state


def print_mermaid_graph() -> None:
    """Outputs the Mermaid visualization syntax for the compiled LangGraph."""
    graph = build_procurement_graph()
    print("\n--- LangGraph Mermaid Flowchart ---")
    try:
        mermaid = graph.get_graph().draw_mermaid()
        print(mermaid)
    except Exception as e:
        print(f"Mermaid generation note: {e}")
        print("""
graph TD
    init_session[Init Session Node] --> supplier_generate[Supplier LLM Bargaining]
    supplier_generate --> auditor_eval[Horn-Clause Contract Auditor]
    auditor_eval --> logistics_eval[Logistics Freight Route Eval]
    logistics_eval -->|Valid & Feasible| finalize_contract[Finalize Contract & Book Freight]
    logistics_eval -->|Limit Reached| abort_negotiation[Abort Negotiation]
    logistics_eval -->|Revisions Needed| supplier_generate
    finalize_contract --> END
    abort_negotiation --> END
        """)
    print("-----------------------------------\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph Multi-Agent Procurement Orchestrator")
    parser.add_argument(
        "--scenario",
        default="baseline",
        choices=["baseline", "clash", "shock", "lowball", "stress", "all"],
        help="Negotiation scenario to execute",
    )
    parser.add_argument(
        "--mermaid",
        action="store_true",
        help="Print the Mermaid diagram of the LangGraph State Machine",
    )
    args = parser.parse_args()

    if args.mermaid:
        print_mermaid_graph()
        return

    if args.scenario == "all":
        results = {}
        for sc in ["baseline", "clash", "shock", "lowball", "stress"]:
            st = run_scenario(sc)
            results[sc] = st
        print("\n" + "=" * 70)
        print("LANGGRAPH ALL SCENARIOS SUMMARY")
        print("=" * 70)
        print(f"{'Scenario':<12} {'Status':<10} {'Rounds':<8} {'Price':<10} {'Carrier'}")
        print("-" * 70)
        for name, r in results.items():
            status = r["status"]
            rounds = r["round_num"]
            price_str = f"${r['offer_price']:.2f}" if status == "SUCCESS" and r.get("offer_price") else "—"
            c_info = "—"
            if status == "SUCCESS" and r.get("logistics_recommended_carrier"):
                c_info = r["logistics_recommended_carrier"].get("carrier_name", "—")
            print(f"{name:<12} {status:<10} {rounds:<8} {price_str:<10} {c_info}")
        print("=" * 70 + "\n")
    else:
        run_scenario(args.scenario)


if __name__ == "__main__":
    main()
