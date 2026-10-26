"""
factory_agent.py  —  Agent 1: Factory Agent
════════════════════════════════════════════════════════════════
Port  : 8001
Role  : States raw-material needs under hard price & deadline
        constraints. Model-based reflex agent — reacts to each
        incoming offer by comparing it against fixed constraints.
Agent type: Model-based Reflex Agent (CSP goal generator)

Endpoints:
    GET  /state                  — current constraint state
    POST /requirement            — (controller calls) get requirement dict
    POST /receive_offer          — evaluate an offer, return accept/reject
    POST /shock_alert            — receive shock broadcast
════════════════════════════════════════════════════════════════
"""

import logging
import time
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ─── logging ────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [Factory] %(message)s")
logger = logging.getLogger("factory_agent")

# ─── load config ────────────────────────────────────────────
_cfg_path = Path(__file__).parent / "config.yaml"
with open(_cfg_path) as f:
    _CFG = yaml.safe_load(f)

_FACTORY_CFG = _CFG["factory"]
_ROUND_LIMIT  = _CFG["negotiation"]["round_limit"]

# ─── FastAPI app ─────────────────────────────────────────────
app = FastAPI(
    title="Factory Agent",
    description="CSP constraint-satisfaction procurement agent",
    version="1.0.0",
)

# ─── Agent internal state (model-based) ─────────────────────
class FactoryState:
    def __init__(self) -> None:
        self.item: str         = _FACTORY_CFG["item"]
        self.min_qty: int      = _FACTORY_CFG["min_qty"]
        self.max_price: float  = _FACTORY_CFG["max_price"]
        self.deadline: int     = _FACTORY_CFG["deadline"]

        self.round_count: int          = 0
        self.best_offer: Optional[dict] = None
        self.shock_received: bool       = False
        self.negotiation_status: str    = "PENDING"   # PENDING | ACCEPTED | FAILED
        self.session_id: Optional[str]  = None

    def reset(self, session_id: str) -> None:
        """Reset for a new negotiation episode."""
        self.round_count      = 0
        self.best_offer       = None
        self.shock_received   = False
        self.negotiation_status = "PENDING"
        self.session_id       = session_id
        logger.info("Factory reset for session %s", session_id)

    def to_dict(self) -> dict:
        return {
            "item":         self.item,
            "min_qty":      self.min_qty,
            "max_price":    self.max_price,
            "deadline":     self.deadline,
            "round_limit":  _ROUND_LIMIT,
            "round_count":  self.round_count,
            "status":       self.negotiation_status,
            "shock_received": self.shock_received,
            "session_id":   self.session_id,
        }

_state = FactoryState()

# ─── Pydantic models ─────────────────────────────────────────

class OfferIn(BaseModel):
    session_id: str
    round_num: int
    offer_price: float
    offer_qty: int
    offer_lead_time: int
    verdict: str        # "VALID" or "REJECTED"
    violated_clauses: list[str] = []
    nl_text: str = ""

class ResetIn(BaseModel):
    session_id: str

class SetRequirementIn(BaseModel):
    item: Optional[str] = None
    min_qty: Optional[int] = None
    max_price: Optional[float] = None
    deadline: Optional[int] = None

class ShockAlertIn(BaseModel):
    round_num: int
    new_pressure: str
    cost_floor_multiplier: float

# ─── Endpoints ───────────────────────────────────────────────

@app.get("/state", summary="Get Factory Agent state")
def get_state() -> dict:
    return _state.to_dict()


@app.post("/reset", summary="Reset for new negotiation session")
def reset(body: ResetIn) -> dict:
    _state.reset(body.session_id)
    return {"ok": True, "state": _state.to_dict()}


@app.post("/set_requirement", summary="Dynamically set Factory procurement requirements")
def set_requirement(body: SetRequirementIn) -> dict:
    """Allows user to dynamically update Factory goals at runtime."""
    if body.item is not None:
        _state.item = body.item
    if body.min_qty is not None:
        _state.min_qty = body.min_qty
    if body.max_price is not None:
        _state.max_price = body.max_price
    if body.deadline is not None:
        _state.deadline = body.deadline
    logger.info(
        "Updated requirement: %s × %d units @ max $%.2f, deadline %d days",
        _state.item, _state.min_qty, _state.max_price, _state.deadline,
    )
    return {"ok": True, "state": _state.to_dict()}


@app.post("/requirement", summary="Return the Factory's procurement requirement")
def get_requirement(body: ResetIn) -> dict:
    """
    Called by the controller at the start of each episode.
    Returns the structured requirement that will be sent to the Supplier.
    """
    if _state.session_id != body.session_id:
        _state.reset(body.session_id)
    req = {
        "session_id":  body.session_id,
        "item":        _state.item,
        "min_qty":     _state.min_qty,
        "max_price":   _state.max_price,
        "deadline":    _state.deadline,
        "round_limit": _ROUND_LIMIT,
        "timestamp":   time.time(),
    }
    logger.info(
        "Requirement dispatched: %s × %d units @ max $%.2f, deadline %d days",
        req["item"], req["min_qty"], req["max_price"], req["deadline"],
    )
    return req


@app.post("/receive_offer", summary="Evaluate an audited offer")
def receive_offer(body: OfferIn) -> dict:
    """
    Model-based reflex decision:
      - If Auditor already marked VALID → perform own sanity check → accept/reject
      - If Auditor marked REJECTED     → propagate rejection (Factory never overrides Auditor)
    Factory only accepts when Auditor verdict == VALID AND its own constraint check passes.
    """
    _state.round_count = body.round_num

    # Factory never accepts an Auditor-rejected offer
    if body.verdict != "VALID":
        logger.info(
            "Round %d: Offer $%.2f rejected by Auditor (%s)",
            body.round_num, body.offer_price, body.violated_clauses,
        )
        return {
            "factory_decision": "REJECT",
            "reason": "Auditor verdict: REJECTED",
            "violated_clauses": body.violated_clauses,
        }

    # Own CSP hard-constraint check
    violations: list[str] = []
    if body.offer_price > _state.max_price:
        violations.append(f"price ${body.offer_price:.2f} > ceiling ${_state.max_price:.2f}")
    if body.offer_lead_time > _state.deadline:
        violations.append(f"lead_time {body.offer_lead_time}d > deadline {_state.deadline}d")
    if body.offer_qty < _state.min_qty:
        violations.append(f"qty {body.offer_qty} < min_qty {_state.min_qty}")

    if violations:
        logger.warning("Round %d: Factory CSP check failed: %s", body.round_num, violations)
        return {
            "factory_decision": "REJECT",
            "reason": "Factory CSP constraint violated",
            "violated_clauses": violations,
        }

    # Accept
    _state.best_offer      = body.dict()
    _state.negotiation_status = "ACCEPTED"
    logger.info(
        "Round %d: ✅ Factory ACCEPTS — $%.2f × %d units, %d days lead time",
        body.round_num, body.offer_price, body.offer_qty, body.offer_lead_time,
    )
    return {
        "factory_decision": "ACCEPT",
        "reason": "All constraints satisfied",
        "accepted_offer": _state.best_offer,
    }


@app.post("/shock_alert", summary="Receive price-shock broadcast")
def shock_alert(body: ShockAlertIn) -> dict:
    """
    Factory acknowledges the shock event (it cannot see Supplier internals,
    but the controller broadcasts a shock notification to all agents).
    """
    _state.shock_received = True
    logger.warning(
        "⚡ SHOCK received at round %d — market pressure → %s",
        body.round_num, body.new_pressure,
    )
    return {"ok": True, "shock_acknowledged": True}


@app.get("/health")
def health() -> dict:
    return {"agent": "factory", "status": "running"}
