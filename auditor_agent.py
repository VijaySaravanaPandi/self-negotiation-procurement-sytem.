"""
auditor_agent.py  —  Agent 3: Contract Auditor Agent
════════════════════════════════════════════════════════════════
Port  : 8003
Role  : Parses the Supplier's NL offer into structured facts,
        runs them against the Horn-clause rule base via
        forward-chaining inference, returns VALID / REJECTED
        with the violated clause(s).
Agent type: Pure Logic / Knowledge-Based Agent

Endpoints:
    GET  /state           — KB state + rule list
    POST /set_constraints — store Factory constraint facts
    POST /audit           — audit a Supplier offer → verdict
    POST /emit_contract   — finalise and record accepted deal
    GET  /contracts       — list all finalised contracts this session
════════════════════════════════════════════════════════════════
"""

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Horn-clause engine ────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from rules.horn_rules import audit_offer, list_rules, AuditResult

# ─── logging ────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [Auditor] %(message)s")
logger = logging.getLogger("auditor_agent")

# ─── load config ────────────────────────────────────────────
_cfg_path = Path(__file__).parent / "config.yaml"
with open(_cfg_path) as f:
    _CFG = yaml.safe_load(f)

# ─── SQLite log path ─────────────────────────────────────────
_DB_PATH = Path(__file__).parent / "logs" / "negotiation_log.sqlite"
_DB_PATH.parent.mkdir(exist_ok=True)


def _init_db() -> None:
    con = sqlite3.connect(_DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS negotiation_rounds (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id       TEXT,
            round_num        INTEGER,
            offer_price      REAL,
            offer_qty        INTEGER,
            offer_lead_time  INTEGER,
            verdict          TEXT,
            violated_clauses TEXT,
            suspicious       INTEGER,
            shock_round      INTEGER DEFAULT 0,
            wall_time        REAL,
            nl_text          TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS contracts (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id       TEXT,
            round_num        INTEGER,
            offer_price      REAL,
            offer_qty        INTEGER,
            offer_lead_time  INTEGER,
            scenario         TEXT,
            signed_at        REAL
        )
    """)
    con.commit()
    con.close()

_init_db()

# ─── FastAPI app ─────────────────────────────────────────────
app = FastAPI(
    title="Contract Auditor Agent",
    description="Horn-clause forward-chaining contract validation agent",
    version="1.0.0",
)

# ─── Auditor state ───────────────────────────────────────────
class AuditorState:
    def __init__(self) -> None:
        self.factory_constraints: dict[str, Any] = {}
        self.session_id: Optional[str] = None
        self.round_count: int = 0
        self.shock_round: Optional[int] = None   # round when shock was applied

    def reset(self, session_id: str) -> None:
        self.session_id   = session_id
        self.round_count  = 0
        self.shock_round  = None
        self.factory_constraints = {}

_state = AuditorState()

# ─── Pydantic models ─────────────────────────────────────────

class ConstraintSetIn(BaseModel):
    session_id: str
    factory_max_price: float
    factory_min_qty: int
    factory_deadline: int

class AuditIn(BaseModel):
    session_id: str
    round_num: int
    offer_price: float
    offer_qty: int
    offer_lead_time: int
    pricing_pressure: str
    cost_floor: float
    nl_text: str = ""
    is_shock_round: bool = False

class ContractEmitIn(BaseModel):
    session_id: str
    round_num: int
    offer_price: float
    offer_qty: int
    offer_lead_time: int
    scenario: str = "default"

class ResetIn(BaseModel):
    session_id: str

# ─── SQLite helpers ──────────────────────────────────────────

def _log_round(
    session_id: str,
    round_num: int,
    offer_price: float,
    offer_qty: int,
    offer_lead_time: int,
    verdict: str,
    violated: list[str],
    suspicious: bool,
    is_shock_round: bool,
    nl_text: str,
) -> None:
    con = sqlite3.connect(_DB_PATH)
    con.execute(
        """
        INSERT INTO negotiation_rounds
            (session_id, round_num, offer_price, offer_qty, offer_lead_time,
             verdict, violated_clauses, suspicious, shock_round, wall_time, nl_text)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            session_id, round_num, offer_price, offer_qty, offer_lead_time,
            verdict,
            ",".join(violated),
            int(suspicious),
            int(is_shock_round),
            time.time(),
            nl_text,
        ),
    )
    con.commit()
    con.close()


def _log_contract(
    session_id: str,
    round_num: int,
    offer_price: float,
    offer_qty: int,
    offer_lead_time: int,
    scenario: str,
) -> None:
    con = sqlite3.connect(_DB_PATH)
    con.execute(
        """
        INSERT INTO contracts
            (session_id, round_num, offer_price, offer_qty, offer_lead_time,
             scenario, signed_at)
        VALUES (?,?,?,?,?,?,?)
        """,
        (session_id, round_num, offer_price, offer_qty, offer_lead_time, scenario, time.time()),
    )
    con.commit()
    con.close()


# ─── Endpoints ───────────────────────────────────────────────

@app.get("/state", summary="Auditor state + rule list")
def get_state() -> dict:
    return {
        "session_id":          _state.session_id,
        "round_count":         _state.round_count,
        "factory_constraints": _state.factory_constraints,
        "shock_round":         _state.shock_round,
        "horn_rules":          list_rules(),
    }


@app.post("/reset", summary="Reset for a new negotiation session")
def reset(body: ResetIn) -> dict:
    _state.reset(body.session_id)
    return {"ok": True}


@app.post("/set_constraints", summary="Store Factory hard constraints as KB facts")
def set_constraints(body: ConstraintSetIn) -> dict:
    if _state.session_id != body.session_id:
        _state.reset(body.session_id)
    _state.factory_constraints = {
        "factory_max_price": body.factory_max_price,
        "factory_min_qty":   body.factory_min_qty,
        "factory_deadline":  body.factory_deadline,
    }
    logger.info(
        "Constraints set: max_price=$%.2f, min_qty=%d, deadline=%dd",
        body.factory_max_price, body.factory_min_qty, body.factory_deadline,
    )
    return {"ok": True, "constraints": _state.factory_constraints}


@app.post("/audit", summary="Audit a Supplier offer via Horn-clause forward chaining")
def audit(body: AuditIn) -> dict:
    """
    Core auditor logic:
    1. Build a fact dict from Factory constraints + Supplier offer fields.
    2. Run forward_chain() on the Horn KB.
    3. Return verdict (VALID/REJECTED) + violated clauses + explanation.
    4. Log every round to SQLite.
    """
    if not _state.factory_constraints:
        raise HTTPException(400, "Factory constraints not set. Call /set_constraints first.")

    _state.round_count = body.round_num
    if body.is_shock_round:
        _state.shock_round = body.round_num

    # Validate that offer fields are parseable
    if body.offer_price is None or body.offer_qty is None or body.offer_lead_time is None:
        violated = ["parse_failure — one or more offer fields could not be extracted from NL text"]
        logger.warning("Round %d: NL parse failure", body.round_num)
        _log_round(
            body.session_id, body.round_num,
            body.offer_price or 0, body.offer_qty or 0, body.offer_lead_time or 0,
            "REJECTED", violated, False, body.is_shock_round, body.nl_text,
        )
        return {
            "verdict":         "REJECTED",
            "violated_clauses": violated,
            "derived_facts":   {},
            "suspicious":      False,
            "explanation":     "NL parsing failed — structured fields missing.",
        }

    # Build fact dict
    facts: dict[str, Any] = {
        **_state.factory_constraints,
        "offer_price":      body.offer_price,
        "offer_qty":        body.offer_qty,
        "offer_lead_time":  body.offer_lead_time,
        "pricing_pressure": body.pricing_pressure,
        "cost_floor":       body.cost_floor,
    }

    # Run Horn-clause forward chaining
    result: AuditResult = audit_offer(facts)
    verdict = "VALID" if result.contract_valid else "REJECTED"

    logger.info(
        "Round %d: Audit → %s | Violated: %s | Suspicious: %s",
        body.round_num, verdict, result.violated_clauses, result.suspicious,
    )
    logger.debug("Horn-clause trace:\n%s", result.explanation)

    # Log to SQLite
    _log_round(
        body.session_id, body.round_num,
        body.offer_price, body.offer_qty, body.offer_lead_time,
        verdict,
        result.violated_clauses,
        result.suspicious,
        body.is_shock_round,
        body.nl_text,
    )

    return {
        "verdict":          verdict,
        "violated_clauses": result.violated_clauses,
        "derived_facts":    result.derived_facts,
        "suspicious":       result.suspicious,
        "explanation":      result.explanation,
    }


@app.post("/emit_contract", summary="Finalise and record an accepted deal")
def emit_contract(body: ContractEmitIn) -> dict:
    """
    Called by the controller when Factory ACCEPTS.
    Records the signed contract to SQLite and returns a contract receipt.
    """
    _log_contract(
        body.session_id,
        body.round_num,
        body.offer_price,
        body.offer_qty,
        body.offer_lead_time,
        body.scenario,
    )
    contract_id = f"{body.session_id}_{body.round_num}"
    logger.info(
        "📄 Contract signed: %s | $%.2f × %d units | %d days",
        contract_id, body.offer_price, body.offer_qty, body.offer_lead_time,
    )
    return {
        "contract_id":    contract_id,
        "status":         "SIGNED",
        "offer_price":    body.offer_price,
        "offer_qty":      body.offer_qty,
        "offer_lead_time": body.offer_lead_time,
        "signed_at":      time.time(),
    }


@app.get("/contracts", summary="List all contracts for this session")
def list_contracts() -> dict:
    con = sqlite3.connect(_DB_PATH)
    rows = con.execute(
        "SELECT * FROM contracts WHERE session_id = ? ORDER BY signed_at",
        (_state.session_id,),
    ).fetchall()
    con.close()
    cols = ["id","session_id","round_num","offer_price","offer_qty",
            "offer_lead_time","scenario","signed_at"]
    return {"contracts": [dict(zip(cols, r)) for r in rows]}


@app.get("/rounds", summary="Get all round logs for current session")
def list_rounds() -> dict:
    con = sqlite3.connect(_DB_PATH)
    rows = con.execute(
        "SELECT * FROM negotiation_rounds WHERE session_id = ? ORDER BY round_num",
        (_state.session_id,),
    ).fetchall()
    con.close()
    cols = ["id","session_id","round_num","offer_price","offer_qty",
            "offer_lead_time","verdict","violated_clauses","suspicious",
            "shock_round","wall_time","nl_text"]
    return {"rounds": [dict(zip(cols, r)) for r in rows]}


@app.get("/health")
def health() -> dict:
    return {"agent": "auditor", "status": "running"}
