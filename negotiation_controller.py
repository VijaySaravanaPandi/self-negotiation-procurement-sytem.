"""
negotiation_controller.py  —  Round-Loop Orchestrator
════════════════════════════════════════════════════════════════
Drives the full negotiation episode:
  1. Reset all agents for a new session
  2. Dispatch Factory requirement → Supplier
  3. Supplier generates offer → Auditor audits
  4. If VALID → Factory accepts → log SUCCESS
  5. If REJECTED → Supplier revises (with feedback) → loop
  6. If rounds > ROUND_LIMIT → log FAILED
  7. Injects shock_event() at configured round
  8. Writes per-round trace to negotiation_log.sqlite

Usage:
    python negotiation_controller.py --scenario baseline
    python negotiation_controller.py --scenario shock
    python negotiation_controller.py --scenario clash
    python negotiation_controller.py --scenario lowball
    python negotiation_controller.py --scenario stress
    python negotiation_controller.py --scenario all

Environment variables:
    GROQ_API_KEY=<your_key>

Scenarios:
    baseline — ample inventory, low pressure → agrees in 1-2 rounds
    clash    — Factory price ceiling below Supplier cost floor → FAILED
    shock    — price shock at round 2 of converging negotiation
    lowball  — adversarial suspiciously-cheap first offer
    stress   — deliberately unsatisfiable constraints → FAILED at ROUND_LIMIT
    all      — run all 5 scenarios sequentially
════════════════════════════════════════════════════════════════
"""

import argparse
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import httpx
import yaml

# ─── logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Controller] %(message)s",
)
logger = logging.getLogger("controller")

# ─── load config ────────────────────────────────────────────
_cfg_path = Path(__file__).parent / "config.yaml"
with open(_cfg_path) as f:
    _CFG = yaml.safe_load(f)

_PORTS        = _CFG["ports"]
_ROUND_LIMIT  = _CFG["negotiation"]["round_limit"]
_SHOCK_CFG    = _CFG["shock"]
_DB_PATH      = Path(__file__).parent / "logs" / "negotiation_log.sqlite"
_DB_PATH.parent.mkdir(exist_ok=True)

FACTORY_URL  = f"http://localhost:{_PORTS['factory']}"
SUPPLIER_URL = f"http://localhost:{_PORTS['supplier']}"
AUDITOR_URL  = f"http://localhost:{_PORTS['auditor']}"

# ─── Scenario configurations ─────────────────────────────────

SCENARIOS: dict[str, dict] = {
    "baseline": {
        "description": "Ample inventory, low pressure → agreement in 1-2 rounds",
        "factory_override":  {},
        "supplier_override": {"pricing_pressure": "low",  "cost_floor_init": 60.0},
        "shock_enabled":     False,
        "adversarial_prompt": False,
    },
    "clash": {
        "description": "Factory price ceiling below Supplier cost floor → FAILED",
        "factory_override":  {"max_price": 50.0},   # below cost_floor of 60
        "supplier_override": {"pricing_pressure": "low",  "cost_floor_init": 60.0},
        "shock_enabled":     False,
        "adversarial_prompt": False,
    },
    "shock": {
        "description": "Price shock at round 3 of a converging negotiation",
        "factory_override":  {},
        "supplier_override": {"pricing_pressure": "low",  "cost_floor_init": 55.0},
        "shock_enabled":     True,
        "adversarial_prompt": False,
    },
    "lowball": {
        "description": "Adversarially cheap first offer → suspicious_offer clause fires",
        "factory_override":  {},
        "supplier_override": {"pricing_pressure": "high", "cost_floor_init": 70.0},
        "shock_enabled":     False,
        "adversarial_prompt": True,  # will inject a below-cost-floor price in round 1
    },
    "stress": {
        "description": "Unsatisfiable constraints → FAILED at ROUND_LIMIT",
        "factory_override":  {"max_price": 1.0, "deadline": 1},  # impossible
        "supplier_override": {"pricing_pressure": "low",  "cost_floor_init": 60.0},
        "shock_enabled":     False,
        "adversarial_prompt": False,
    },
}


# ─── Episode result dataclass ────────────────────────────────

@dataclass
class EpisodeResult:
    session_id: str
    scenario: str
    status: str            # SUCCESS | FAILED
    rounds_to_agreement: int
    final_price: Optional[float]
    final_qty: Optional[int]
    final_lead_time: Optional[int]
    shock_at_round: Optional[int]
    round_trace: list[dict]


# ─── HTTP helpers ────────────────────────────────────────────

def _post(url: str, path: str, data: dict, timeout: float = 30.0) -> dict:
    with httpx.Client(timeout=timeout) as client:
        r = client.post(f"{url}{path}", json=data)
        r.raise_for_status()
        return r.json()


def _get(url: str, path: str, timeout: float = 10.0) -> dict:
    with httpx.Client(timeout=timeout) as client:
        r = client.get(f"{url}{path}")
        r.raise_for_status()
        return r.json()


def _check_health(url: str, name: str) -> bool:
    try:
        r = _get(url, "/health")
        logger.info("%s: %s", name, r)
        return True
    except Exception as e:
        logger.error("%s health check FAILED: %s", name, e)
        return False


# ─── Summary log to SQLite ───────────────────────────────────

def _log_episode_summary(result: EpisodeResult) -> None:
    con = sqlite3.connect(_DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS episode_summary (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id           TEXT,
            scenario             TEXT,
            status               TEXT,
            rounds_to_agreement  INTEGER,
            final_price          REAL,
            final_qty            INTEGER,
            final_lead_time      INTEGER,
            shock_at_round       INTEGER,
            timestamp            REAL
        )
    """)
    cur.execute(
        """
        INSERT INTO episode_summary
            (session_id, scenario, status, rounds_to_agreement,
             final_price, final_qty, final_lead_time, shock_at_round, timestamp)
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (
            result.session_id, result.scenario, result.status,
            result.rounds_to_agreement,
            result.final_price, result.final_qty, result.final_lead_time,
            result.shock_at_round, time.time(),
        ),
    )
    con.commit()
    con.close()
    logger.info("Episode summary logged to SQLite.")


# ─── Scenario config injection ───────────────────────────────

def _inject_scenario_config(scenario_cfg: dict) -> None:
    """
    Apply scenario overrides directly via the agent state endpoints.
    Factory and Supplier agents support dynamic override payloads
    injected through their /config_override endpoint.
    (For simplicity, overrides are embedded in the requirement message
    and the Supplier's initial generate_offer prompt.)
    """
    pass  # Handled inline in run_episode()


# ─── Main negotiation loop ───────────────────────────────────

def run_episode(
    scenario_name: str = "baseline",
    custom_req: Optional[dict] = None,
) -> EpisodeResult:
    scenario_cfg = SCENARIOS.get(scenario_name, SCENARIOS["baseline"])
    session_id   = str(uuid.uuid4())[:8]
    shock_applied_round: Optional[int] = None

    logger.info("=" * 70)
    logger.info("SCENARIO: %s — %s", scenario_name.upper(), scenario_cfg["description"])
    logger.info("Session ID: %s", session_id)
    logger.info("=" * 70)

    # ── 1. Check all agents are running ─────────────────────
    for url, name in [(FACTORY_URL, "Factory"), (SUPPLIER_URL, "Supplier"), (AUDITOR_URL, "Auditor")]:
        if not _check_health(url, name):
            raise RuntimeError(f"{name} agent is not running at {url}. "
                               "Start all three services first.")

    # ── 2. Reset all agents ──────────────────────────────────
    for url in [FACTORY_URL, SUPPLIER_URL, AUDITOR_URL]:
        _post(url, "/reset", {"session_id": session_id})

    # Apply custom requirements if provided by user/UI
    if custom_req:
        _post(FACTORY_URL, "/set_requirement", custom_req)

    # ── 3. Get Factory requirement ───────────────────────────
    req = _post(FACTORY_URL, "/requirement", {"session_id": session_id})

    # Apply scenario overrides to requirement if not manually passed
    if not custom_req:
        fo = scenario_cfg.get("factory_override", {})
        if "max_price" in fo:
            req["max_price"] = fo["max_price"]
            logger.info("[Override] Factory max_price → $%.2f", req["max_price"])
        if "deadline" in fo:
            req["deadline"] = fo["deadline"]
            logger.info("[Override] Factory deadline → %d days", req["deadline"])

    logger.info(
        "Factory requirement: %s × %d units @ max $%.2f, deadline %d days",
        req["item"], req["min_qty"], req["max_price"], req["deadline"],
    )

    # ── 4. Send requirement to Supplier ──────────────────────
    _post(SUPPLIER_URL, "/receive_requirement", {
        "session_id": session_id,
        "item":        req["item"],
        "min_qty":     req["min_qty"],
        "max_price":   req["max_price"],
        "deadline":    req["deadline"],
        "round_limit": req["round_limit"],
    })

    # ── 5. Set Factory constraints in Auditor ────────────────
    _post(AUDITOR_URL, "/set_constraints", {
        "session_id":        session_id,
        "factory_max_price": req["max_price"],
        "factory_min_qty":   req["min_qty"],
        "factory_deadline":  req["deadline"],
    })

    # ── 6. Negotiation round loop ─────────────────────────────
    round_trace:   list[dict] = []
    offer:         Optional[dict] = None
    final_status:  str = "FAILED"
    final_offer:   Optional[dict] = None

    for round_num in range(1, _ROUND_LIMIT + 1):
        logger.info("─" * 50)
        logger.info("ROUND %d / %d", round_num, _ROUND_LIMIT)

        # ── Price shock injection ─────────────────────────────
        is_shock_round = False
        if (
            scenario_cfg.get("shock_enabled")
            and _SHOCK_CFG.get("enabled")
            and round_num == _SHOCK_CFG["trigger_round"]
            and shock_applied_round is None
        ):
            logger.warning("⚡ Injecting price shock at round %d!", round_num)
            shock_result = _post(SUPPLIER_URL, "/shock", {
                "session_id": session_id,
                "round_num":  round_num,
            })
            _post(FACTORY_URL, "/shock_alert", {
                "round_num":              round_num,
                "new_pressure":           shock_result.get("pricing_pressure", "high"),
                "cost_floor_multiplier":  _SHOCK_CFG["cost_floor_multiplier"],
            })
            shock_applied_round = round_num
            is_shock_round = True

        # ── Generate or revise offer ──────────────────────────
        if offer is None:
            # First offer
            offer = _post(SUPPLIER_URL, "/generate_offer", {
                "session_id": session_id,
                "round_num":  round_num,
            })
        else:
            # Revision based on previous rejection
            offer = _post(SUPPLIER_URL, "/revise_offer", {
                "session_id":               session_id,
                "round_num":                round_num,
                "violated_clauses":         last_violated,
                "previous_offer_price":     offer["offer_price"],
                "previous_offer_qty":       offer["offer_qty"],
                "previous_offer_lead_time": offer["offer_lead_time"],
                "feedback_text":            ", ".join(last_violated),
            })

        logger.info("Supplier NL offer: %s", offer.get("nl_text", "")[:150])
        logger.info(
            "Parsed → price=$%.2f, qty=%s, lead_time=%s days",
            offer.get("offer_price") or 0,
            offer.get("offer_qty"),
            offer.get("offer_lead_time"),
        )

        # ── Audit the offer ───────────────────────────────────
        audit_resp = _post(AUDITOR_URL, "/audit", {
            "session_id":       session_id,
            "round_num":        round_num,
            "offer_price":      offer.get("offer_price") or 0,
            "offer_qty":        offer.get("offer_qty") or 0,
            "offer_lead_time":  offer.get("offer_lead_time") or 0,
            "pricing_pressure": offer.get("pricing_pressure", "low"),
            "cost_floor":       offer.get("cost_floor", 60.0),
            "nl_text":          offer.get("nl_text", ""),
            "is_shock_round":   is_shock_round,
        })

        verdict         = audit_resp["verdict"]
        last_violated   = audit_resp.get("violated_clauses", [])
        suspicious      = audit_resp.get("suspicious", False)

        logger.info(
            "Auditor verdict: %s | Violated: %s | Suspicious: %s",
            verdict, last_violated, suspicious,
        )
        logger.info("Horn-clause trace:\n%s", audit_resp.get("explanation", ""))

        # ── Record round trace ────────────────────────────────
        round_trace.append({
            "round_num":       round_num,
            "offer_price":     offer.get("offer_price"),
            "offer_qty":       offer.get("offer_qty"),
            "offer_lead_time": offer.get("offer_lead_time"),
            "verdict":         verdict,
            "violated_clauses": last_violated,
            "suspicious":      suspicious,
            "is_shock_round":  is_shock_round,
            "nl_text":         offer.get("nl_text", ""),
        })

        # ── Factory decision ──────────────────────────────────
        factory_resp = _post(FACTORY_URL, "/receive_offer", {
            "session_id":       session_id,
            "round_num":        round_num,
            "offer_price":      offer.get("offer_price") or 0,
            "offer_qty":        offer.get("offer_qty") or 0,
            "offer_lead_time":  offer.get("offer_lead_time") or 0,
            "verdict":          verdict,
            "violated_clauses": last_violated,
            "nl_text":          offer.get("nl_text", ""),
        })

        factory_decision = factory_resp.get("factory_decision")
        logger.info("Factory decision: %s", factory_decision)

        if factory_decision == "ACCEPT":
            # Auditor emits final signed contract
            _post(AUDITOR_URL, "/emit_contract", {
                "session_id":       session_id,
                "round_num":        round_num,
                "offer_price":      offer["offer_price"],
                "offer_qty":        offer["offer_qty"],
                "offer_lead_time":  offer["offer_lead_time"],
                "scenario":         scenario_name,
            })
            final_status = "SUCCESS"
            final_offer  = offer
            logger.info("✅ NEGOTIATION SUCCEEDED in round %d!", round_num)
            break

        # Prepare for next round
        logger.info("↺ Round %d REJECTED — retrying...", round_num)
        # offer stays populated so revise_offer knows what to revise

    else:
        logger.warning("❌ NEGOTIATION FAILED — round limit %d reached.", _ROUND_LIMIT)

    # ── Build result ─────────────────────────────────────────
    result = EpisodeResult(
        session_id=session_id,
        scenario=scenario_name,
        status=final_status,
        rounds_to_agreement=len(round_trace) if final_status == "SUCCESS" else _ROUND_LIMIT,
        final_price=final_offer["offer_price"] if final_offer else None,
        final_qty=final_offer["offer_qty"] if final_offer else None,
        final_lead_time=final_offer["offer_lead_time"] if final_offer else None,
        shock_at_round=shock_applied_round,
        round_trace=round_trace,
    )

    _log_episode_summary(result)

    # ── Print summary ─────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("EPISODE SUMMARY")
    logger.info("  Scenario           : %s", result.scenario)
    logger.info("  Status             : %s", result.status)
    logger.info("  Rounds             : %d", result.rounds_to_agreement)
    if result.final_price:
        logger.info("  Final price        : $%.2f", result.final_price)
        logger.info("  Final qty          : %d units", result.final_qty)
        logger.info("  Final lead time    : %d days", result.final_lead_time)
    if result.shock_at_round:
        logger.info("  Shock at round     : %d", result.shock_at_round)
    logger.info("=" * 70)

    return result


# ─── CLI entry point ─────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Negotiation Controller")
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()) + ["all"],
        default="baseline",
        help="Which negotiation scenario to run",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds between rounds (default: 2.0)",
    )
    args = parser.parse_args()

    if args.scenario == "all":
        results = []
        for name in SCENARIOS:
            try:
                r = run_episode(name)
                results.append(r)
            except Exception as e:
                logger.error("Scenario %s failed with error: %s", name, e)
            time.sleep(args.delay)

        # Print comparison table
        print("\n" + "=" * 70)
        print("ALL SCENARIOS SUMMARY")
        print("=" * 70)
        print(f"{'Scenario':<12} {'Status':<10} {'Rounds':<8} {'Final Price':<14} {'Shock Rd'}")
        print("-" * 70)
        for r in results:
            price_str = f"${r.final_price:.2f}" if r.final_price else "—"
            shock_str = str(r.shock_at_round) if r.shock_at_round else "—"
            print(f"{r.scenario:<12} {r.status:<10} {r.rounds_to_agreement:<8} {price_str:<14} {shock_str}")
        print("=" * 70)
    else:
        run_episode(args.scenario)


if __name__ == "__main__":
    main()
