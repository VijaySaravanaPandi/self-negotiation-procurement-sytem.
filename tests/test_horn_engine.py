"""
tests/test_horn_engine.py  —  Unit Tests for Horn-Clause Engine
════════════════════════════════════════════════════════════════
Tests each rule in isolation and tests forward chaining combinations.
  1. valid_price
  2. valid_delivery
  3. valid_quantity
  4. suspicious_offer
  5. contract_valid
════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.horn_rules import HornKnowledgeBase, AuditResult


def test_valid_price_rule():
    kb = HornKnowledgeBase()
    
    # Case A: Price within limit
    facts_good = {
        "offer_price": 80.0,
        "factory_max_price": 85.0,
        "offer_lead_time": 5,
        "factory_deadline": 7,
        "offer_qty": 500,
        "factory_min_qty": 500,
        "pricing_pressure": "low",
        "cost_floor": 60.0,
    }
    res: AuditResult = kb.forward_chain(facts_good)
    assert res.derived_facts.get("valid_price") is True
    assert "valid_price" not in res.violated_clauses

    # Case B: Price exceeds ceiling
    facts_bad = dict(facts_good, offer_price=95.0)
    res_bad: AuditResult = kb.forward_chain(facts_bad)
    assert "valid_price" in res_bad.violated_clauses
    assert res_bad.contract_valid is False


def test_valid_delivery_rule():
    kb = HornKnowledgeBase()
    
    facts = {
        "offer_price": 80.0,
        "factory_max_price": 85.0,
        "offer_lead_time": 10, # Exceeds deadline of 7
        "factory_deadline": 7,
        "offer_qty": 500,
        "factory_min_qty": 500,
        "pricing_pressure": "low",
        "cost_floor": 60.0,
    }
    res = kb.forward_chain(facts)
    assert "valid_delivery" in res.violated_clauses
    assert res.contract_valid is False


def test_valid_quantity_rule():
    kb = HornKnowledgeBase()
    
    facts = {
        "offer_price": 80.0,
        "factory_max_price": 85.0,
        "offer_lead_time": 5,
        "factory_deadline": 7,
        "offer_qty": 300, # Below min_qty of 500
        "factory_min_qty": 500,
        "pricing_pressure": "low",
        "cost_floor": 60.0,
    }
    res = kb.forward_chain(facts)
    assert "valid_quantity" in res.violated_clauses
    assert res.contract_valid is False


def test_suspicious_offer_rule():
    kb = HornKnowledgeBase()
    
    # High pricing pressure + price below cost floor = suspicious
    facts = {
        "offer_price": 45.0, # Below cost_floor 60.0
        "factory_max_price": 85.0,
        "offer_lead_time": 5,
        "factory_deadline": 7,
        "offer_qty": 500,
        "factory_min_qty": 500,
        "pricing_pressure": "high",
        "cost_floor": 60.0,
    }
    res = kb.forward_chain(facts)
    assert res.suspicious is True
    assert "suspicious_offer" in res.violated_clauses
    assert res.contract_valid is False


def test_contract_valid_master_rule():
    kb = HornKnowledgeBase()
    
    # Fully valid offer
    facts = {
        "offer_price": 82.0,
        "factory_max_price": 85.0,
        "offer_lead_time": 6,
        "factory_deadline": 7,
        "offer_qty": 600,
        "factory_min_qty": 500,
        "pricing_pressure": "low",
        "cost_floor": 60.0,
    }
    res = kb.forward_chain(facts)
    assert res.contract_valid is True
    assert len(res.violated_clauses) == 0
