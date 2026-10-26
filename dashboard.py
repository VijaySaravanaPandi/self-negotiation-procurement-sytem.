"""
dashboard.py  —  Streamlit Interactive Control & Inspection Dashboard
══════════════════════════════════════════════════════════════════════
Interactive 4-Agent Control Dashboard:
  - 4 Dynamic Agent State Cards (Factory 8001, Supplier 8002, Auditor 8003, Logistics 8004)
    * Starting: Displays default / initial state for each port
    * Every Search: Dynamically updates all 4 agent answers generated for that run!
  - Real-time dynamic input forms for Factory procurement requirements
  - Live 4-agent negotiation execution (Factory, Supplier, Horn Auditor, Logistics)
  - Multilateral agreement certification & carrier booking
  - MongoDB Live Document Store & SQLite historical trace analysis
══════════════════════════════════════════════════════════════════════
"""

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import pandas as pd
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).parent))
from mongo_store import mongo_store

# Setup page layout
st.set_page_config(
    page_title="Self-Negotiating Procurement Network Dashboard",
    page_icon="🛡️",
    layout="wide",
)

_DB_PATH = Path(__file__).parent / "logs" / "negotiation_log.sqlite"

FACTORY_URL   = "http://localhost:8001"
SUPPLIER_URL  = "http://localhost:8002"
AUDITOR_URL   = "http://localhost:8003"
LOGISTICS_URL = "http://localhost:8004"


def fetch_agent_state(url: str) -> dict:
    try:
        r = httpx.get(f"{url}/state", timeout=3.0)
        return r.json()
    except Exception as e:
        return {"error": str(e), "status": "offline"}


# Initialize session state for dynamic answers
if "agent_answers" not in st.session_state:
    st.session_state.agent_answers = None

if "last_execution_result" not in st.session_state:
    st.session_state.last_execution_result = None


# ─────────────────────────────────────────────────────────────
# 🛡️ TITLE & HEADER
# ─────────────────────────────────────────────────────────────
st.title("🛡️ Self-Negotiating Procurement Network")
st.caption("4-Agent Horn-Clause-Audited & Logistics-Certified Autonomous Procurement Ecosystem")

top_col_title, top_col_btn = st.columns([4, 1])
with top_col_title:
    if st.session_state.agent_answers:
        st.info("💡 **Display Mode**: Showing dynamic answers generated from the latest negotiation search.")
    else:
        st.info("💡 **Display Mode**: Showing default / initial state for all 4 microservice ports.")

with top_col_btn:
    if st.session_state.agent_answers:
        if st.button("↺ Reset Cards to Defaults"):
            st.session_state.agent_answers = None
            st.session_state.last_execution_result = None
            st.rerun()

# ─────────────────────────────────────────────────────────────
# 🤖 4-AGENT STATUS & ANSWER CARDS (DYNAMIC PER SEARCH)
# ─────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

# 1. Factory Agent (Port 8001)
with col1:
    st.subheader("Factory Agent (Port 8001)")
    if st.session_state.agent_answers:
        dyn_f = st.session_state.agent_answers["factory"]
        f_status = dyn_f.get("status", "ACCEPTED")
        if f_status == "ACCEPTED":
            st.success("Status: ACCEPTED (Deal Closed)")
        else:
            st.error("Status: REJECTED (Impasse)")
        st.json(dyn_f)
    else:
        f_state = fetch_agent_state(FACTORY_URL)
        if "error" in f_state:
            st.error(f"Offline: {f_state['error']}")
        else:
            st.success(f"Status: {f_state.get('status', 'RUNNING')}")
            st.json(f_state)

# 2. Supplier Agent (Port 8002)
with col2:
    st.subheader("Supplier Agent (Port 8002)")
    if st.session_state.agent_answers:
        dyn_s = st.session_state.agent_answers["supplier"]
        st.success("Offer Generated (Groq LLM)")
        st.json(dyn_s)
    else:
        s_state = fetch_agent_state(SUPPLIER_URL)
        if "error" in s_state:
            st.error(f"Offline: {s_state['error']}")
        else:
            st.success("Running (Partial Observability Enforced)")
            st.json(s_state)

# 3. Auditor Agent (Port 8003)
with col3:
    st.subheader("Auditor Agent (Port 8003)")
    if st.session_state.agent_answers:
        dyn_a = st.session_state.agent_answers["auditor"]
        if dyn_a.get("verdict") == "VALID":
            st.success("Verdict: VALID (Horn-Certified)")
        else:
            st.warning("Verdict: REJECTED (Clauses Violated)")
        st.json(dyn_a)
    else:
        a_state = fetch_agent_state(AUDITOR_URL)
        if "error" in a_state:
            st.error(f"Offline: {a_state['error']}")
        else:
            st.success("Running (Horn Engine Active)")
            st.json(a_state)

# 4. Logistics Agent (Port 8004)
with col4:
    st.subheader("Logistics Agent (Port 8004)")
    if st.session_state.agent_answers:
        dyn_l = st.session_state.agent_answers["logistics"]
        if dyn_l.get("shipping_feasibility") == "FEASIBLE":
            st.success("Status: FEASIBLE (Carrier Booked)")
        else:
            st.error("Status: INFEASIBLE (Route Delayed)")
        st.json(dyn_l)
    else:
        l_state = fetch_agent_state(LOGISTICS_URL)
        if "error" in l_state:
            st.error(f"Offline: {l_state['error']}")
        else:
            st.success("Running (Fleet Route Engine Active)")
            st.json(l_state)

st.divider()

# ─────────────────────────────────────────────────────────────
# 🎛️ DYNAMIC USER INPUT & LIVE EXECUTION PANEL
# ─────────────────────────────────────────────────────────────
st.header("🎛️ Dynamic Requirement Configuration & Live Execution")
st.markdown(
    "Specify custom procurement requirements at runtime. "
    "All **4 Agents** will autonomously interact across ports **8001, 8002, 8003, and 8004**:\n"
    "- **Factory (8001)** specifies CSP goals $\\rightarrow$\n"
    "- **Supplier (8002)** generates generative NL bargains via Groq $\\rightarrow$\n"
    "- **Auditor (8003)** logically certifies constraints via Horn-clause inference $\\rightarrow$\n"
    "- **Logistics (8004)** evaluates carrier routes, transit feasibility, and books freight $\\rightarrow$\n"
    "- **MongoDB & SQLite** persist the multilateral audit trail!"
)

with st.form("dynamic_procurement_form"):
    c1, c2, c3, c4, c5 = st.columns(5)

    items_list = [
        "aluminum_tubes",
        "steel_rods",
        "copper_pipes",
        "titanium_plates",
        "brass_fittings",
        "stainless_sheets",
        "nickel_wire",
        "carbon_fiber_rods",
    ]

    with c1:
        selected_item = st.selectbox("Raw Material Item", items_list, index=0)
    with c2:
        min_qty = st.number_input("Required Quantity (units)", min_value=1, max_value=10000, value=300, step=50)
    with c3:
        max_price = st.number_input("Max Budget Ceiling ($/unit)", min_value=1.0, max_value=10000.0, value=95.0, step=5.0)
    with c4:
        deadline = st.number_input("Delivery Deadline (days)", min_value=1, max_value=30, value=7, step=1)
    with c5:
        scenario = st.selectbox(
            "Negotiation Scenario",
            ["baseline", "shock", "clash", "lowball", "stress"],
            index=0,
            help="Choose baseline or price-shock/clash scenarios",
        )

    submit_btn = st.form_submit_button("🚀 Run Live 4-Agent Negotiation", type="primary")


# ── Live 4-Agent Orchestration Function ──────────────────────────────
def execute_4agent_negotiation(item_name: str, qty: int, budget: float, max_days: int, scenario_type: str) -> dict:
    session_id = str(uuid.uuid4())[:8]

    # 1. Reset all 4 microservices
    for u in [FACTORY_URL, SUPPLIER_URL, AUDITOR_URL, LOGISTICS_URL]:
        httpx.post(f"{u}/reset", json={"session_id": session_id}, timeout=10.0)

    # 2. Configure Factory goals
    factory_goals = {
        "item": item_name,
        "min_qty": qty,
        "max_price": budget,
        "deadline": max_days,
    }
    # Apply scenario overrides if clash or stress
    if scenario_type == "clash":
        factory_goals["max_price"] = 50.0
    elif scenario_type == "stress":
        factory_goals["max_price"] = 1.0
        factory_goals["deadline"] = 1

    httpx.post(f"{FACTORY_URL}/set_requirement", json=factory_goals, timeout=10.0)
    req = httpx.post(f"{FACTORY_URL}/requirement", json={"session_id": session_id}, timeout=10.0).json()

    # 3. Inform Supplier and Auditor
    httpx.post(f"{SUPPLIER_URL}/receive_requirement", json={
        "session_id": session_id,
        "item": req["item"],
        "min_qty": req["min_qty"],
        "max_price": req["max_price"],
        "deadline": req["deadline"],
        "round_limit": 6,
    }, timeout=10.0)

    httpx.post(f"{AUDITOR_URL}/set_constraints", json={
        "session_id": session_id,
        "factory_max_price": req["max_price"],
        "factory_min_qty": req["min_qty"],
        "factory_deadline": req["deadline"],
    }, timeout=10.0)

    round_traces = []
    success = False
    final_offer = None
    final_carrier_booking = None
    last_violated = []
    last_verdict = "REJECTED"
    last_suspicious = False
    last_logistics_resp = None
    round_limit = 6

    for r_num in range(1, round_limit + 1):
        # Handle mid-negotiation price shock
        if scenario_type == "shock" and r_num == 3:
            shock_r = httpx.post(f"{SUPPLIER_URL}/shock", json={"session_id": session_id, "round_num": r_num}, timeout=10.0).json()
            httpx.post(f"{FACTORY_URL}/shock_alert", json={
                "round_num": r_num,
                "new_pressure": shock_r.get("pricing_pressure", "high"),
                "cost_floor_multiplier": 1.25,
            }, timeout=10.0)

        # Supplier generate / revise offer
        if r_num == 1:
            offer = httpx.post(f"{SUPPLIER_URL}/generate_offer", json={
                "session_id": session_id,
                "round_num": r_num,
                "adversarial": (scenario_type == "lowball"),
            }, timeout=25.0).json()
        else:
            fb_parts = [f"Auditor: {', '.join(last_violated)}"] if last_violated else []
            if last_logistics_resp and not last_logistics_resp.get("feasible"):
                fb_parts.append(f"Logistics: {last_logistics_resp.get('explanation', '')}")
            fb_text = " | ".join(fb_parts) or ", ".join(last_violated)

            offer = httpx.post(f"{SUPPLIER_URL}/revise_offer", json={
                "session_id": session_id,
                "round_num": r_num,
                "violated_clauses": last_violated,
                "previous_offer_price": final_offer["offer_price"] if final_offer else 0.0,
                "previous_offer_qty": final_offer["offer_qty"] if final_offer else 0,
                "previous_offer_lead_time": final_offer["offer_lead_time"] if final_offer else 0,
                "feedback_text": fb_text,
            }, timeout=25.0).json()

        final_offer = offer
        p = float(offer.get("offer_price") or 0.0)
        q = int(offer.get("offer_qty") or 0)
        lt = int(offer.get("offer_lead_time") or 0)

        # Auditor verification
        audit_resp = httpx.post(f"{AUDITOR_URL}/audit", json={
            "session_id": session_id,
            "round_num": r_num,
            "offer_price": p,
            "offer_qty": q,
            "offer_lead_time": lt,
            "pricing_pressure": offer.get("pricing_pressure", "low"),
            "cost_floor": offer.get("cost_floor", 60.0),
            "nl_text": offer.get("nl_text", ""),
            "is_shock_round": (scenario_type == "shock" and r_num == 3),
        }, timeout=15.0).json()

        last_verdict = audit_resp.get("verdict", "REJECTED")
        last_violated = audit_resp.get("violated_clauses", [])
        last_suspicious = audit_resp.get("suspicious", False)

        # Logistics evaluation
        log_resp = httpx.post(f"{LOGISTICS_URL}/evaluate_shipment", json={
            "session_id": session_id,
            "item": req["item"],
            "quantity": q,
            "supplier_lead_time_days": lt,
            "factory_deadline_days": req["deadline"],
        }, timeout=15.0).json()

        last_logistics_resp = log_resp
        log_feasible = log_resp.get("feasible", False)
        if not log_feasible and "valid_delivery" not in last_violated:
            last_violated.append("valid_delivery")

        # Factory review
        factory_resp = httpx.post(f"{FACTORY_URL}/receive_offer", json={
            "session_id": session_id,
            "round_num": r_num,
            "offer_price": p,
            "offer_qty": q,
            "offer_lead_time": lt,
            "verdict": last_verdict,
            "violated_clauses": last_violated,
            "nl_text": offer.get("nl_text", ""),
        }, timeout=10.0).json()

        fac_decision = factory_resp.get("factory_decision", "REJECT")

        step_record = {
            "round_num": r_num,
            "offer_price": p,
            "offer_qty": q,
            "offer_lead_time": lt,
            "verdict": last_verdict,
            "violated_clauses": last_violated,
            "suspicious": last_suspicious,
            "nl_text": offer.get("nl_text", ""),
            "logistics_feasible": log_feasible,
            "carrier_name": log_resp.get("recommended_carrier", {}).get("carrier_name") if log_resp.get("recommended_carrier") else None,
            "transit_days": log_resp.get("recommended_carrier", {}).get("transit_days", 0) if log_resp.get("recommended_carrier") else 0,
            "freight_cost_per_unit": log_resp.get("recommended_carrier", {}).get("freight_cost_per_unit", 0.0) if log_resp.get("recommended_carrier") else 0.0,
            "factory_decision": fac_decision,
        }
        round_traces.append(step_record)

        # Agreement Condition
        if last_verdict == "VALID" and log_feasible and fac_decision == "ACCEPT":
            success = True
            # Auditor contract
            httpx.post(f"{AUDITOR_URL}/emit_contract", json={
                "session_id": session_id,
                "round_num": r_num,
                "offer_price": p,
                "offer_qty": q,
                "offer_lead_time": lt,
                "scenario": scenario_type,
            }, timeout=10.0)

            # Logistics booking
            if log_resp.get("recommended_carrier"):
                cid = log_resp["recommended_carrier"]["carrier_id"]
                final_carrier_booking = httpx.post(f"{LOGISTICS_URL}/book_shipment", json={
                    "session_id": session_id,
                    "carrier_id": cid,
                    "item": req["item"],
                    "quantity": q,
                    "contract_price_per_unit": p,
                }, timeout=10.0).json()
            break

    # Persist contract to MongoDB
    final_contract_doc = {
        "session_id": session_id,
        "scenario": scenario_type,
        "status": "SUCCESS" if success else "FAILED",
        "rounds_to_agreement": len(round_traces) if success else round_limit,
        "final_price": final_offer.get("offer_price") if success and final_offer else None,
        "final_qty": final_offer.get("offer_qty") if success and final_offer else None,
        "final_lead_time": final_offer.get("offer_lead_time") if success and final_offer else None,
        "logistics_booking": final_carrier_booking,
        "traces": round_traces,
    }
    mongo_store.log_session(final_contract_doc)
    if success:
        mongo_store.log_contract(final_contract_doc)

    # Persist summary to SQLite
    try:
        con = sqlite3.connect(_DB_PATH)
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS episode_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT, scenario TEXT, status TEXT,
                rounds_to_agreement INTEGER, final_price REAL,
                final_qty INTEGER, final_lead_time INTEGER,
                shock_at_round INTEGER, timestamp REAL
            )
        """)
        cur.execute(
            "INSERT INTO episode_summary (session_id, scenario, status, rounds_to_agreement, final_price, final_qty, final_lead_time, shock_at_round, timestamp) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                session_id, scenario_type, "SUCCESS" if success else "FAILED",
                len(round_traces),
                final_offer.get("offer_price") if success and final_offer else None,
                final_offer.get("offer_qty") if success and final_offer else None,
                final_offer.get("offer_lead_time") if success and final_offer else None,
                3 if scenario_type == "shock" else None,
                time.time()
            )
        )
        con.commit()
        con.close()
    except Exception:
        pass

    # Build dynamic answers dictionary for each port
    carrier_info = (last_logistics_resp or {}).get("recommended_carrier") or {}
    dynamic_answers = {
        "factory": {
            "session_id": session_id,
            "item": req["item"],
            "requested_qty": req["min_qty"],
            "max_price_ceiling": req["max_price"],
            "delivery_deadline": req["deadline"],
            "final_price_accepted": final_offer.get("offer_price") if success and final_offer else None,
            "round_count": len(round_traces),
            "status": "ACCEPTED" if success else "FAILED",
            "decision": "ACCEPT" if success else "REJECT",
        },
        "supplier": {
            "session_id": session_id,
            "scenario": scenario_type,
            "final_offered_price": final_offer.get("offer_price") if final_offer else None,
            "final_offered_qty": final_offer.get("offer_qty") if final_offer else None,
            "final_lead_time_days": final_offer.get("offer_lead_time") if final_offer else None,
            "rounds_negotiated": len(round_traces),
            "latest_nl_offer": (final_offer.get("nl_text") or "") if final_offer else "",
            "pricing_pressure": final_offer.get("pricing_pressure") if final_offer else "low",
            "llm_model": "qwen/qwen3.8-27b",
        },
        "auditor": {
            "session_id": session_id,
            "round_count": len(round_traces),
            "final_verdict": last_verdict,
            "violated_clauses": last_violated,
            "suspicious_offer_flagged": last_suspicious,
            "horn_rules_verified": [
                "valid_price: offer_price <= factory_max_price",
                "valid_delivery: offer_lead_time <= factory_deadline",
                "valid_quantity: offer_qty >= factory_min_qty",
                "not_suspicious: ¬(pricing_pressure = high ∧ offer_price < cost_floor)",
                "contract_valid: valid_price ∧ valid_delivery ∧ valid_quantity ∧ ¬suspicious_offer"
            ] if success else [],
            "contract_certified": success,
        },
        "logistics": {
            "session_id": session_id,
            "shipping_feasibility": "FEASIBLE" if (last_logistics_resp and last_logistics_resp.get("feasible")) else "INFEASIBLE",
            "selected_carrier": carrier_info.get("carrier_name", "None"),
            "transport_mode": carrier_info.get("transport_mode", "None"),
            "carrier_transit_days": carrier_info.get("transit_days", 0),
            "total_delivery_days": carrier_info.get("total_delivery_days", 0),
            "freight_cost_per_unit": carrier_info.get("freight_cost_per_unit", 0.0),
            "total_freight_cost": round(carrier_info.get("freight_cost_per_unit", 0.0) * req["min_qty"], 2),
            "carbon_emissions_kg": carrier_info.get("estimated_carbon_kg", 0.0),
            "booking_id": final_carrier_booking.get("booking_id") if final_carrier_booking else None,
            "destination": "Factory Plant Alpha",
        },
    }

    return {
        "session_id": session_id,
        "status": "SUCCESS" if success else "FAILED",
        "rounds_to_agreement": len(round_traces),
        "final_price": final_offer.get("offer_price") if final_offer and success else None,
        "final_qty": final_offer.get("offer_qty") if final_offer and success else None,
        "final_lead_time": final_offer.get("offer_lead_time") if final_offer and success else None,
        "round_trace": round_traces,
        "carrier_booking": final_carrier_booking,
        "dynamic_answers": dynamic_answers,
    }


if submit_btn:
    st.subheader("🔄 Live Negotiation Execution Trace")
    with st.spinner("All 4 Agents negotiating live (Factory 8001 + Supplier Groq 8002 + Auditor Horn 8003 + Logistics 8004)..."):
        try:
            exec_res = execute_4agent_negotiation(
                item_name=selected_item,
                qty=int(min_qty),
                budget=float(max_price),
                max_days=int(deadline),
                scenario_type=scenario,
            )
            st.session_state.agent_answers = exec_res["dynamic_answers"]
            st.session_state.last_execution_result = exec_res
            st.rerun()

        except Exception as err:
            st.error(f"Error running 4-agent negotiation: {err}")


# Display results if a run just took place
if st.session_state.last_execution_result:
    res = st.session_state.last_execution_result
    st.subheader("🔄 Last Negotiation Execution Trace")

    if res["status"] == "SUCCESS":
        st.balloons()
        bk = res.get("carrier_booking") or {}
        st.success(
            f"✅ **CONTRACT CERTIFIED & SHIPMENT BOOKED!** Agreed in **{res['rounds_to_agreement']} round(s)** | "
            f"Final Price: **${res['final_price']:.2f}/unit** | Qty: **{res['final_qty']} units** | Lead Time: **{res['final_lead_time']} days**\n\n"
            f"🚚 **Freight Carrier**: {bk.get('carrier_name')} ({bk.get('mode')}) | Transit: {bk.get('transit_days')}d | "
            f"Booking ID: `{bk.get('booking_id')}` | Total Freight Cost: ${bk.get('total_freight_cost', 0.0):.2f}"
        )
    else:
        st.error(
            f"❌ **NEGOTIATION FAILED / IMPASSE** — Reached round limit ({res['rounds_to_agreement']} rounds) without satisfying all constraints."
        )

    st.markdown("### 📜 Round-by-Round Step Trace (All 4 Agents)")
    for step in res["round_trace"]:
        verdict_icon = "✅ VALID" if step["verdict"] == "VALID" else "❌ REJECTED"
        carrier_icon = f"🚚 {step['carrier_name']} ({step['transit_days']}d transit)" if step.get("carrier_name") else "⚠️ No Feasible Carrier"
        with st.expander(f"Round {step['round_num']}: Offer ${step['offer_price']:.2f} — {verdict_icon} | {carrier_icon}"):
            st.markdown(f"**💬 Supplier NL Bargaining Text (Groq LLM):**")
            st.info(f"\"{step['nl_text']}\"")

            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.markdown("**🔍 Parsed Terms:**")
                st.write(f"- Price: `${step['offer_price']:.2f}`/unit")
                st.write(f"- Quantity: `{step['offer_qty']}` units")
                st.write(f"- Lead Time: `{step['offer_lead_time']}` days")
            with sc2:
                st.markdown("**⚖️ Horn-Clause Auditor (8003):**")
                if step["violated_clauses"]:
                    st.warning(f"Violated: {step['violated_clauses']}")
                else:
                    st.success("All Horn clauses satisfied!")
            with sc3:
                st.markdown("**🚚 Logistics Route Engine (8004):**")
                if step.get("logistics_feasible"):
                    st.success(f"Feasible via {step['carrier_name']} (${step['freight_cost_per_unit']:.2f}/unit)")
                else:
                    st.error("Route exceeds delivery deadline!")

st.divider()

# ─────────────────────────────────────────────────────────────
# 📦 SUPPORTED RAW MATERIALS INVENTORY LIST
# ─────────────────────────────────────────────────────────────
st.header("📦 Domain Items & Materials Catalog")
st.markdown("Supported raw-material items for dynamic procurement negotiations:")

item_df = pd.DataFrame(
    [
        {"Item Key": "aluminum_tubes", "Material Name": "Aluminum Tubes (Grade 6061)", "Unit": "Tubes", "Category": "Lightweight Metals", "Default Cost Floor": "$60.00"},
        {"Item Key": "steel_rods", "Material Name": "Steel Rods (High-Tensile Structural)", "Unit": "Rods", "Category": "Heavy Metals", "Default Cost Floor": "$60.00"},
        {"Item Key": "copper_pipes", "Material Name": "Copper Pipes (HVAC Grade)", "Unit": "Pipes", "Category": "Conductive Piping", "Default Cost Floor": "$75.00"},
        {"Item Key": "titanium_plates", "Material Name": "Titanium Alloy Plates (Grade 5)", "Unit": "Plates", "Category": "Aerospace Metals", "Default Cost Floor": "$120.00"},
        {"Item Key": "brass_fittings", "Material Name": "Brass Hex Fittings", "Unit": "Pieces", "Category": "Plumbing Hardware", "Default Cost Floor": "$35.00"},
        {"Item Key": "stainless_sheets", "Material Name": "Stainless Steel Sheets (304)", "Unit": "Sheets", "Category": "Heavy Metals", "Default Cost Floor": "$80.00"},
        {"Item Key": "nickel_wire", "Material Name": "Nickel Alloy Wire (Inconel)", "Unit": "Spools", "Category": "Specialty Alloys", "Default Cost Floor": "$95.00"},
        {"Item Key": "carbon_fiber_rods", "Material Name": "Carbon Fiber Composite Rods", "Unit": "Rods", "Category": "Composites", "Default Cost Floor": "$110.00"},
    ]
)
st.dataframe(item_df, width="stretch" if hasattr(st, "dataframe") else None)

st.divider()

# ─────────────────────────────────────────────────────────────
# 📊 HISTORICAL LOG ANALYSIS & MONGODB LIVE DOCUMENT STORE
# ─────────────────────────────────────────────────────────────
st.header("📊 Negotiation Trace History & MongoDB Document Store")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Round Trace Log (SQLite)",
    "📜 Episode Summaries",
    "🍃 MongoDB Live Document Store",
    "⚖️ Horn Rules & KB"
])

with tab1:
    st.subheader("Historical Negotiation Rounds")
    if not _DB_PATH.exists():
        st.info("No database file found yet. Run a negotiation above to populate data.")
    else:
        con = sqlite3.connect(_DB_PATH)
        rounds_df = pd.read_sql_query("SELECT * FROM negotiation_rounds ORDER BY id DESC", con)
        con.close()
        if rounds_df.empty:
            st.write("No round data logged yet.")
        else:
            sessions = rounds_df["session_id"].unique().tolist()
            selected_session = st.selectbox("Select Session ID to Filter", sessions)
            session_rounds = rounds_df[rounds_df["session_id"] == selected_session].sort_values("round_num")
            st.line_chart(session_rounds.set_index("round_num")[["offer_price"]])
            st.dataframe(session_rounds, width="stretch" if hasattr(st, "dataframe") else None)

with tab2:
    st.subheader("Episode Summary History")
    if _DB_PATH.exists():
        con = sqlite3.connect(_DB_PATH)
        summary_df = pd.read_sql_query("SELECT * FROM episode_summary ORDER BY id DESC", con)
        con.close()
        if summary_df.empty:
            st.write("No episode summaries found.")
        else:
            st.dataframe(summary_df, width="stretch" if hasattr(st, "dataframe") else None)

with tab3:
    st.subheader("🍃 Live MongoDB Document Collections (`procurement_network`)")
    mongo_status = "🟢 Connected (mongodb://localhost:27017)" if mongo_store.is_connected else "🟡 Local JSON Fallback Mode"
    st.caption(f"MongoDB Status: {mongo_status}")

    mongo_col1, mongo_col2 = st.columns(2)
    with mongo_col1:
        st.markdown("#### 📜 Finalized Contracts Collection (`contracts`)")
        contracts = mongo_store.get_contracts()
        if contracts:
            st.json(contracts[-10:])
        else:
            st.info("No contracts saved in MongoDB yet.")

    with mongo_col2:
        st.markdown("#### 🔄 Negotiation Sessions Collection (`negotiation_sessions`)")
        sessions = mongo_store.get_sessions(limit=10)
        if sessions:
            st.json(sessions)
        else:
            st.info("No sessions saved in MongoDB yet.")

with tab4:
    st.subheader("Horn Knowledge Base Rules")
    a_state = fetch_agent_state(AUDITOR_URL)
    if "horn_rules" in a_state:
        for rule in a_state["horn_rules"]:
            st.code(rule, language="prolog")
    else:
        st.write("Auditor offline — rules unavailable.")
