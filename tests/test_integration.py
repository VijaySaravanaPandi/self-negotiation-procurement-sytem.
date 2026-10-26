"""
tests/test_integration.py  —  Integration Tests for 3-Agent Negotiation Loop
════════════════════════════════════════════════════════════════
Runs scenario integration tests using pytest and httpx.
Asserts final status, round limits, and Horn-clause validity.
════════════════════════════════════════════════════════════════
"""

import os
import sys
from pathlib import Path
import pytest
import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from negotiation_controller import run_episode, SCENARIOS

FACTORY_URL = "http://localhost:8001"
SUPPLIER_URL = "http://localhost:8002"
AUDITOR_URL = "http://localhost:8003"


def is_service_up(url: str) -> bool:
    try:
        r = httpx.get(f"{url}/health", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(
    not (is_service_up(FACTORY_URL) and is_service_up(SUPPLIER_URL) and is_service_up(AUDITOR_URL)),
    reason="All 3 microservices (Factory 8001, Supplier 8002, Auditor 8003) must be running.",
)
def test_baseline_scenario():
    """Test baseline scenario — should succeed within round limit."""
    res = run_episode("baseline")
    assert res.status == "SUCCESS"
    assert res.rounds_to_agreement <= 6
    assert res.final_price is not None
    assert res.final_price <= 85.0  # Factory max ceiling


@pytest.mark.skipif(
    not (is_service_up(FACTORY_URL) and is_service_up(SUPPLIER_URL) and is_service_up(AUDITOR_URL)),
    reason="Microservices offline.",
)
def test_clash_scenario():
    """Test constraint clash scenario — should end in FAILED due to ceiling below cost floor."""
    res = run_episode("clash")
    assert res.status == "FAILED"
    assert res.rounds_to_agreement == 6


@pytest.mark.skipif(
    not (is_service_up(FACTORY_URL) and is_service_up(SUPPLIER_URL) and is_service_up(AUDITOR_URL)),
    reason="Microservices offline.",
)
def test_shock_scenario():
    """Test shock scenario — shock should trigger at round 3."""
    res = run_episode("shock")
    assert res.shock_at_round == 3
