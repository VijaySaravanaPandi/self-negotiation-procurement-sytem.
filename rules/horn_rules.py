"""
rules/horn_rules.py
═══════════════════════════════════════════════════════════════════
Pure-Python Forward-Chaining Horn-Clause Engine
Contract Auditor Agent — Knowledge Base

Each rule is a strict Horn clause:
    head ← body₁ ∧ body₂ ∧ ... ∧ [¬negated_fact]

Facts are stored in a flat dict:  { fact_name: bool | numeric_value }

Forward chaining fires rules whose bodies are all satisfied,
deriving new facts until a fixed point (no new facts added).
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Data types
# ─────────────────────────────────────────────────────────────

@dataclass
class HornRule:
    """
    A single Horn clause:  head ← body_predicates

    Each predicate in `body` is a callable(facts: dict) → bool.
    `negated_heads` is a set of derived-fact names that must NOT
    be true for this rule's head to fire (NAF — negation as failure).
    """
    head: str                                           # name of derived fact
    body: list[Callable[[dict[str, Any]], bool]]        # list of predicate callables
    negated_heads: list[str] = field(default_factory=list)  # facts that must be absent
    description: str = ""                               # human-readable description

    def fires(self, facts: dict[str, Any]) -> bool:
        """Return True iff all body predicates hold and no negated head is derived."""
        try:
            body_satisfied = all(p(facts) for p in self.body)
            negation_satisfied = all(facts.get(n) is not True for n in self.negated_heads)
            return body_satisfied and negation_satisfied
        except (KeyError, TypeError):
            return False


@dataclass
class AuditResult:
    """Outcome returned by forward_chain()."""
    contract_valid: bool
    derived_facts: dict[str, bool]        # which intermediate facts were derived
    violated_clauses: list[str]           # names of failed required clauses
    suspicious: bool                      # whether suspicious_offer fired
    explanation: str                      # human-readable summary


# ─────────────────────────────────────────────────────────────
#  Knowledge Base
# ─────────────────────────────────────────────────────────────

class HornKnowledgeBase:
    """
    Maintains the rule set and runs forward-chaining inference.

    Required input fact keys (set by Auditor from parsed offer + Factory constraints):
        offer_price       (float)   — price per unit in the offer
        offer_qty         (int)     — quantity offered
        offer_lead_time   (int)     — lead time in days
        factory_max_price (float)   — Factory's hard price ceiling
        factory_deadline  (int)     — Factory's hard delivery deadline (days)
        factory_min_qty   (int)     — Factory's minimum acceptable quantity
        cost_floor        (float)   — Supplier's cost floor (shared with Auditor for audit)
        pricing_pressure  (str)     — "low" | "medium" | "high"
    """

    def __init__(self) -> None:
        self.rules: list[HornRule] = self._build_rules()

    # ── Rule definitions ────────────────────────────────────────

    def _build_rules(self) -> list[HornRule]:
        return [
            # ── Rule 1: valid_price ──────────────────────────────
            HornRule(
                head="valid_price",
                body=[
                    lambda f: "offer_price" in f and "factory_max_price" in f,
                    lambda f: float(f["offer_price"]) <= float(f["factory_max_price"]),
                ],
                description="valid_price ← offer_price ≤ factory_max_price",
            ),

            # ── Rule 2: valid_delivery ───────────────────────────
            HornRule(
                head="valid_delivery",
                body=[
                    lambda f: "offer_lead_time" in f and "factory_deadline" in f,
                    lambda f: int(f["offer_lead_time"]) <= int(f["factory_deadline"]),
                ],
                description="valid_delivery ← offer_lead_time ≤ factory_deadline",
            ),

            # ── Rule 3: valid_quantity ───────────────────────────
            HornRule(
                head="valid_quantity",
                body=[
                    lambda f: "offer_qty" in f and "factory_min_qty" in f,
                    lambda f: int(f["offer_qty"]) >= int(f["factory_min_qty"]),
                ],
                description="valid_quantity ← offer_qty ≥ factory_min_qty",
            ),

            # ── Rule 4: suspicious_offer (adversarial lowball) ───
            # Fires when the supplier's pricing pressure is high yet it
            # offers below its own cost floor — a logically implausible bid.
            HornRule(
                head="suspicious_offer",
                body=[
                    lambda f: str(f.get("pricing_pressure", "low")).lower() == "high",
                    lambda f: "offer_price" in f and "cost_floor" in f,
                    lambda f: float(f["offer_price"]) < float(f["cost_floor"]),
                ],
                description=(
                    "suspicious_offer ← pricing_pressure = high "
                    "∧ offer_price < cost_floor"
                ),
            ),

            # ── Rule 5: contract_valid (master rule) ─────────────
            # valid_price ∧ valid_delivery ∧ valid_quantity ∧ ¬suspicious_offer
            HornRule(
                head="contract_valid",
                body=[
                    lambda f: f.get("valid_price") is True,
                    lambda f: f.get("valid_delivery") is True,
                    lambda f: f.get("valid_quantity") is True,
                ],
                negated_heads=["suspicious_offer"],
                description=(
                    "contract_valid ← valid_price ∧ valid_delivery "
                    "∧ valid_quantity ∧ ¬suspicious_offer"
                ),
            ),
        ]

    # ── Forward-chaining inference ──────────────────────────────

    def forward_chain(self, base_facts: dict[str, Any]) -> AuditResult:
        """
        Run forward chaining on base_facts.

        Returns an AuditResult with:
            - contract_valid bool
            - dict of derived boolean facts
            - list of violated clause names
            - suspicious flag
            - plain-English explanation
        """
        facts: dict[str, Any] = dict(base_facts)   # working copy
        derived: dict[str, bool] = {}

        # Iterate until fixed point
        changed = True
        max_iterations = 20  # safety cap
        iteration = 0
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            for rule in self.rules:
                if facts.get(rule.head) is True:
                    continue  # already derived
                if rule.fires(facts):
                    facts[rule.head] = True
                    derived[rule.head] = True
                    changed = True
                    logger.debug("Derived: %s  [%s]", rule.head, rule.description)

        # Determine violated required clauses
        required = ["valid_price", "valid_delivery", "valid_quantity"]
        violated: list[str] = [
            c for c in required if facts.get(c) is not True
        ]
        suspicious = facts.get("suspicious_offer") is True
        if suspicious:
            violated.append("suspicious_offer")

        contract_valid = facts.get("contract_valid") is True

        # Build explanation
        lines = []
        for rule in self.rules:
            status = "✓" if facts.get(rule.head) is True else "✗"
            lines.append(f"  [{status}] {rule.description}")
        explanation = "\n".join(lines)

        return AuditResult(
            contract_valid=contract_valid,
            derived_facts=derived,
            violated_clauses=violated,
            suspicious=suspicious,
            explanation=explanation,
        )

    def rule_descriptions(self) -> list[str]:
        """Return human-readable descriptions of all rules."""
        return [r.description for r in self.rules]


# ─────────────────────────────────────────────────────────────
#  Convenience helper used by auditor_agent.py
# ─────────────────────────────────────────────────────────────

_KB = HornKnowledgeBase()   # module-level singleton


def audit_offer(facts: dict[str, Any]) -> AuditResult:
    """Public entry point — runs forward chaining on a fact dict."""
    return _KB.forward_chain(facts)


def list_rules() -> list[str]:
    """Return all rule descriptions."""
    return _KB.rule_descriptions()
