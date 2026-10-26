"""
supplier_agent.py  —  Agent 2: Supplier Agent
════════════════════════════════════════════════════════════════
Port  : 8002
Role  : Uses Groq LLM to generate natural-language offers
        conditioned on private inventory + pricing-pressure state.
        Revises offers when the Auditor returns a rejection +
        violated-clause feedback.
Agent type: Utility-based (Generative) Agent

Private state (NEVER exposed in /state or shared with Factory):
    inventory, cost_floor, pricing_pressure

Endpoints:
    GET  /state               — public state (hides private data)
    POST /receive_requirement — store Factory's requirement
    POST /generate_offer      — call Groq LLM → return NL offer + parsed fields
    POST /revise_offer        — revise with feedback → new NL offer + parsed fields
    POST /shock               — trigger price-shock mutation
════════════════════════════════════════════════════════════════
"""

import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException
from groq import Groq
from pydantic import BaseModel

# ─── logging ────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [Supplier] %(message)s")
logger = logging.getLogger("supplier_agent")

# ─── load config ────────────────────────────────────────────
_cfg_path = Path(__file__).parent / "config.yaml"
with open(_cfg_path) as f:
    _CFG = yaml.safe_load(f)

_SUPPLIER_CFG = _CFG["supplier"]
_GROQ_CFG     = _CFG["groq"]
_SHOCK_CFG    = _CFG["shock"]

# ─── Groq client ────────────────────────────────────────────
_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not _GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not set — LLM calls will fail!")

groq_client = Groq(api_key=_GROQ_API_KEY or "dummy_key")

# ─── FastAPI app ─────────────────────────────────────────────
app = FastAPI(
    title="Supplier Agent",
    description="LLM-backed generative offer agent with private inventory",
    version="1.0.0",
)

# ─── Agent private state ─────────────────────────────────────
class SupplierPrivateState:
    """
    Contains all private supplier information.
    Partial observability: these fields are NEVER returned
    in the /state endpoint or any public response.
    """
    def __init__(self) -> None:
        self.inventory: int              = _SUPPLIER_CFG["inventory"]
        self.cost_floor: float           = _SUPPLIER_CFG["cost_floor"]
        self.pricing_pressure: str       = _SUPPLIER_CFG["initial_pricing_pressure"]
        self.shock_applied: bool         = False
        self.requirement: Optional[dict] = None
        self.session_id: Optional[str]   = None
        self.offer_history: list[dict]   = []

    def reset(self, session_id: str) -> None:
        self.inventory        = _SUPPLIER_CFG["inventory"]
        self.cost_floor       = _SUPPLIER_CFG["cost_floor"]
        self.pricing_pressure = _SUPPLIER_CFG["initial_pricing_pressure"]
        self.shock_applied    = False
        self.requirement      = None
        self.session_id       = session_id
        self.offer_history    = []

    def apply_shock(self) -> dict:
        """Apply price shock — raises cost floor and pricing pressure."""
        if self.shock_applied:
            return {"already_applied": True}
        mult = _SHOCK_CFG["cost_floor_multiplier"]
        old_floor = self.cost_floor
        self.cost_floor       = round(self.cost_floor * mult, 2)
        self.pricing_pressure = _SHOCK_CFG["pressure_after_shock"]
        self.shock_applied    = True
        logger.warning(
            "⚡ SHOCK applied: cost_floor $%.2f → $%.2f, pressure → %s",
            old_floor, self.cost_floor, self.pricing_pressure,
        )
        return {
            "shock_applied": True,
            "cost_floor_before": old_floor,
            "cost_floor_after": self.cost_floor,
            "pricing_pressure": self.pricing_pressure,
        }

_private = SupplierPrivateState()

# ─── Pydantic models ─────────────────────────────────────────

class ResetIn(BaseModel):
    session_id: str

class RequirementIn(BaseModel):
    session_id: str
    item: str
    min_qty: int
    max_price: float
    deadline: int
    round_limit: int

class GenerateOfferIn(BaseModel):
    session_id: str
    round_num: int

class ReviseOfferIn(BaseModel):
    session_id: str
    round_num: int
    violated_clauses: list[str]
    previous_offer_price: Optional[float] = 0.0
    previous_offer_qty: Optional[int] = 0
    previous_offer_lead_time: Optional[int] = 0
    feedback_text: str = ""

class ShockIn(BaseModel):
    session_id: str
    round_num: int

# ─── NL offer parsing ────────────────────────────────────────

_PRICE_RE     = re.compile(r"\$?\s*(\d+(?:\.\d+)?)\s*(?:per\s+unit|/unit|each|USD)?", re.I)
_QTY_RE       = re.compile(r"(\d+)\s*(?:units?|pieces?|rods?|pcs?)", re.I)
_LEAD_RE      = re.compile(r"(\d+)\s*(?:business\s+)?days?\s+(?:lead\s+time|delivery|within|for)", re.I)
_LEAD_RE2     = re.compile(r"(?:lead\s+time|delivery|deliver\s+within)\s*[:\-]?\s*(\d+)\s*days?", re.I)


def parse_offer_nl(text: str) -> dict:
    """
    Extract (price, qty, lead_time) from the LLM-generated NL offer.
    """
    price = None
    qty = None
    lead_time = None

    # Price: prioritize explicit dollar sign ($82.50)
    m_price_dollar = re.search(r"\$\s*(\d+(?:\.\d+)?)", text)
    if m_price_dollar:
        price = float(m_price_dollar.group(1))
    else:
        # Fallback to per unit / USD mentions
        m_price_alt = re.search(r"(\d+(?:\.\d+)?)\s*(?:per\s+unit|/unit|USD)", text, re.I)
        if m_price_alt:
            price = float(m_price_alt.group(1))

    # Quantity: look for numbers preceding units/pcs/rods
    m_qty = re.search(r"(\d+)\s*(?:units?|pieces?|rods?|pcs?)", text, re.I)
    if m_qty:
        qty = int(m_qty.group(1))

    # Lead time: look for numbers near days
    m_lead = re.search(r"(\d+)\s*(?:business\s+)?days?", text, re.I)
    if m_lead:
        lead_time = int(m_lead.group(1))

    return {"offer_price": price, "offer_qty": qty, "offer_lead_time": lead_time}


# ─── LLM offer generation ─────────────────────────────────────

def _system_prompt() -> str:
    return (
        "You are a steel-rod supplier negotiating a procurement deal. "
        "Your goal is to maximise your profit margin while remaining "
        "plausible to the buyer and acceptable to a strict contract auditor. "
        "Always respond with a short, professional business offer (2-4 sentences). "
        "The offer MUST explicitly state: price per unit (in USD), quantity (in units), "
        "and lead time (in days). Use the format: "
        "'I can offer [qty] units of steel rods at $[price] per unit with a lead time of [days] days.' "
        "Do not reveal your internal cost floor or inventory levels."
    )


def _user_prompt_initial(req: dict, private: SupplierPrivateState) -> str:
    pressure_hints = {
        "low":    "Market conditions are stable. Open at a high profitable price near the buyer's ceiling.",
        "medium": "Material costs have risen. Open at a high price to protect your margin.",
        "high":   "Severe market shortage. Open at a premium price near or slightly above the buyer's maximum budget ceiling.",
    }
    hint = pressure_hints.get(private.pricing_pressure, "")
    return (
        f"The factory needs {req['min_qty']} units of {req['item']} at most "
        f"${req['max_price']:.2f}/unit, delivery within {req['deadline']} days.\n"
        f"Your private minimum cost floor is ${private.cost_floor:.2f}/unit.\n"
        f"STRATEGY: You want to maximize profit margin. Open your initial offer at a HIGH price "
        f"close to (or slightly above) the buyer's max budget of ${req['max_price']:.2f}/unit. "
        f"Do NOT start immediately at your minimum cost floor of ${private.cost_floor:.2f}/unit!\n"
        f"Generate your first negotiation offer."
    )


def _user_prompt_revision(
    req: dict,
    private: SupplierPrivateState,
    violated: list[str],
    prev_price: float,
    prev_qty: int,
    prev_lead: int,
) -> str:
    feedback_lines = []
    if "valid_price" in violated:
        feedback_lines.append(
            f"  - Your price ${prev_price:.2f} exceeded the buyer's ceiling ${req['max_price']:.2f}. "
            f"Lower your price if possible, but NEVER go below your minimum cost floor of ${private.cost_floor:.2f}/unit."
        )
    if "valid_delivery" in violated:
        feedback_lines.append(
            f"  - Your lead time {prev_lead} days exceeded the deadline {req['deadline']} days. "
            f"You MUST offer lead time ≤ {req['deadline']} days."
        )
    if "valid_quantity" in violated:
        feedback_lines.append(
            f"  - Your quantity {prev_qty} units was below the minimum {req['min_qty']} units. "
            f"You MUST offer at least {req['min_qty']} units."
        )
    if "suspicious_offer" in violated:
        feedback_lines.append(
            "  - Your offer was flagged as suspiciously low given current market pressure. "
            f"Adjust your price to a realistic level above ${private.cost_floor:.2f}/unit."
        )

    pressure_hints = {
        "low":    "Market is stable — small concessions are acceptable.",
        "medium": "Costs are rising — revise carefully.",
        "high":   "Market under severe pressure — protect margins but meet hard constraints.",
    }
    hint = pressure_hints.get(private.pricing_pressure, "")

    feedback_str = "\n".join(feedback_lines) if feedback_lines else "  - General constraint violation."
    return (
        f"Your previous offer (${prev_price:.2f}/unit, {prev_qty} units, {prev_lead} days) "
        f"was REJECTED for the following reasons:\n{feedback_str}\n"
        f"Market note: {hint}\n"
        f"CRITICAL RULE: Your private minimum cost floor is ${private.cost_floor:.2f}/unit. "
        f"You MUST NEVER offer below ${private.cost_floor:.2f}/unit.\n"
        f"Generate a revised offer."
    )


def _call_groq(system: str, user: str) -> str:
    """Call Groq LLM and return the raw text response."""
    api_key = os.environ.get("GROQ_API_KEY", "") or _GROQ_API_KEY
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=_GROQ_CFG["model"],
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=_GROQ_CFG["temperature"],
        max_tokens=_GROQ_CFG["max_tokens"],
    )
    return response.choices[0].message.content.strip()


def _build_offer_response(nl_text: str, round_num: int, session_id: str) -> dict:
    """Parse NL text and build the standard offer response dict."""
    parsed = parse_offer_nl(nl_text)
    return {
        "session_id":        session_id,
        "round_num":         round_num,
        "nl_text":           nl_text,
        "offer_price":       parsed["offer_price"],
        "offer_qty":         parsed["offer_qty"],
        "offer_lead_time":   parsed["offer_lead_time"],
        "pricing_pressure":  _private.pricing_pressure,   # revealed to Auditor for suspicious check
        "cost_floor":        _private.cost_floor,          # revealed to Auditor for suspicious check
        "timestamp":         time.time(),
    }


# ─── Endpoints ───────────────────────────────────────────────

@app.get("/state", summary="Supplier public state (private fields hidden)")
def get_state() -> dict:
    """Partial observability: inventory, cost_floor, pricing_pressure are private."""
    return {
        "session_id":     _private.session_id,
        "shock_applied":  _private.shock_applied,
        "offer_count":    len(_private.offer_history),
        # NOTE: inventory, cost_floor, pricing_pressure intentionally withheld
        "public_note": (
            "Supplier private state (inventory, cost_floor, pricing_pressure) "
            "is not shared. Partial observability is enforced."
        ),
    }


@app.post("/reset", summary="Reset supplier for a new session")
def reset(body: ResetIn) -> dict:
    _private.reset(body.session_id)
    return {"ok": True, "session_id": body.session_id}


@app.post("/receive_requirement", summary="Receive Factory procurement requirement")
def receive_requirement(body: RequirementIn) -> dict:
    if _private.session_id != body.session_id:
        _private.reset(body.session_id)
    _private.requirement = body.dict()
    logger.info(
        "Requirement received: %s × %d @ max $%.2f, deadline %d days",
        body.item, body.min_qty, body.max_price, body.deadline,
    )
    return {"ok": True, "session_id": body.session_id}


@app.post("/generate_offer", summary="Generate first NL offer via LLM")
def generate_offer(body: GenerateOfferIn) -> dict:
    if _private.requirement is None:
        raise HTTPException(400, "No requirement received yet. Call /receive_requirement first.")
    if _private.session_id != body.session_id:
        raise HTTPException(400, "Session mismatch.")

    req = _private.requirement
    system  = _system_prompt()
    user    = _user_prompt_initial(req, _private)

    logger.info("Round %d: Calling Groq LLM for initial offer...", body.round_num)
    nl_text = _call_groq(system, user)
    logger.info("Round %d: LLM offer: %s", body.round_num, nl_text[:120])

    offer = _build_offer_response(nl_text, body.round_num, body.session_id)
    _private.offer_history.append(offer)
    return offer


@app.post("/revise_offer", summary="Revise offer based on Auditor rejection feedback")
def revise_offer(body: ReviseOfferIn) -> dict:
    if _private.requirement is None:
        raise HTTPException(400, "No requirement received yet.")
    if _private.session_id != body.session_id:
        raise HTTPException(400, "Session mismatch.")

    req = _private.requirement
    system = _system_prompt()
    user   = _user_prompt_revision(
        req, _private,
        body.violated_clauses,
        body.previous_offer_price,
        body.previous_offer_qty,
        body.previous_offer_lead_time,
    )

    logger.info(
        "Round %d: Revising offer. Violated: %s", body.round_num, body.violated_clauses
    )
    nl_text = _call_groq(system, user)
    logger.info("Round %d: Revised offer: %s", body.round_num, nl_text[:120])

    offer = _build_offer_response(nl_text, body.round_num, body.session_id)
    _private.offer_history.append(offer)
    return offer


@app.post("/shock", summary="Apply price-shock to Supplier state")
def apply_shock(body: ShockIn) -> dict:
    result = _private.apply_shock()
    return result


@app.get("/health")
def health() -> dict:
    return {"agent": "supplier", "status": "running"}
