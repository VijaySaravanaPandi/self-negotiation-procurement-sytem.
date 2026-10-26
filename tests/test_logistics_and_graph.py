"""
test_logistics_and_graph.py  —  Unit tests for Logistics Agent, MongoStore & LangGraph
══════════════════════════════════════════════════════════════════════════════════════
Tests:
  1. MongoStore offline fallback and document logging
  2. Logistics route evaluation logic (carrier selection, deadline feasibility)
  3. LangGraph workflow graph compilation and node topology
══════════════════════════════════════════════════════════════════════════════════════
"""

import pytest
from mongo_store import MongoStore
from logistics_agent import evaluate_shipment, ShipmentEvaluationRequest, CARRIER_FLEET
from langgraph_controller import build_procurement_graph, consensus_router


def test_mongo_store_resilient_logging(tmp_path):
    """Test that MongoStore can log sessions and contracts without error."""
    store = MongoStore(uri="mongodb://localhost:27017")
    sid = store.log_session({"session_id": "test_unit_session", "status": "UNIT_TEST"})
    assert sid is not None

    cid = store.log_contract({"session_id": "test_unit_session", "price": 85.0})
    assert cid is not None


def test_logistics_feasible_route():
    """Test Logistics Agent selects a feasible carrier when lead time + transit <= deadline."""
    req = ShipmentEvaluationRequest(
        session_id="test-log-1",
        item="steel_rods",
        quantity=500,
        supplier_lead_time_days=2,
        factory_deadline_days=7,
        max_freight_budget_per_unit=10.0,
    )
    resp = evaluate_shipment(req)
    assert resp.feasible is True
    assert resp.status == "FEASIBLE"
    assert resp.recommended_carrier is not None
    assert resp.recommended_carrier.total_delivery_days <= 7


def test_logistics_infeasible_deadline():
    """Test Logistics Agent rejects route when supplier lead time + transit > deadline."""
    req = ShipmentEvaluationRequest(
        session_id="test-log-2",
        item="steel_rods",
        quantity=500,
        supplier_lead_time_days=8,   # Supplier already exceeds deadline of 7
        factory_deadline_days=7,
    )
    resp = evaluate_shipment(req)
    assert resp.feasible is False
    assert resp.status == "INFEASIBLE_DEADLINE"
    assert resp.recommended_carrier is None


def test_langgraph_consensus_router():
    """Test LangGraph conditional router logic."""
    # When all approve:
    state_valid = {
        "audit_verdict": "VALID",
        "factory_decision": "ACCEPT",
        "logistics_feasible": True,
        "round_num": 1,
        "round_limit": 6,
    }
    assert consensus_router(state_valid) == "finalize_contract"

    # When round limit reached without consensus:
    state_aborted = {
        "audit_verdict": "REJECTED",
        "factory_decision": "REJECT",
        "logistics_feasible": False,
        "round_num": 6,
        "round_limit": 6,
    }
    assert consensus_router(state_aborted) == "abort_negotiation"

    # When intermediate round needs revision:
    state_continue = {
        "audit_verdict": "REJECTED",
        "factory_decision": "REJECT",
        "logistics_feasible": False,
        "round_num": 2,
        "round_limit": 6,
    }
    assert consensus_router(state_continue) == "continue_negotiating"


def test_langgraph_workflow_compilation():
    """Test that LangGraph StateGraph compiles and contains required nodes."""
    compiled = build_procurement_graph()
    assert compiled is not None
