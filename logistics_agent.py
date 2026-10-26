"""
logistics_agent.py  —  Agent 4: Logistics & Freight Agent
════════════════════════════════════════════════════════════════════
Port  : 8004
Role  : Evaluates shipping feasibility, transit times, carrier
        capacities, and freight costs. Ensures total fulfillment
        (supplier lead time + freight transit) does not breach
        the Factory's critical delivery deadline.
Agent type: Utility-based Optimization & Route Verification Agent

Endpoints:
    GET  /health              — service health check
    GET  /state               — active carrier fleet & statistics
    POST /evaluate_shipment   — assess route feasibility & select optimal carrier
    POST /book_shipment       — confirm and book freight consignment
    GET  /shipments           — list all confirmed shipments
    POST /reset               — reset state for new negotiation session
════════════════════════════════════════════════════════════════════
"""

import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mongo_store import mongo_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Logistics] %(message)s")
logger = logging.getLogger("logistics_agent")

app = FastAPI(
    title="Logistics & Freight Agent",
    description="Agent 4: Autonomous freight evaluation, carrier selection, and transit verification",
    version="1.0.0",
)

# ── Available Freight Carriers ────────────────────────────────────
CARRIER_FLEET = [
    {
        "id": "CARRIER-LOCAL-00",
        "name": "Local Hub Direct Dispatch",
        "mode": "Same-Day Direct Cargo",
        "transit_days": 0,
        "cost_per_unit": 2.20,
        "max_capacity": 3000,
        "co2_kg_per_unit": 0.20,
        "reliability_score": 0.99,
    },
    {
        "id": "CARRIER-AIR-01",
        "name": "AeroFreight Express",
        "mode": "Air Express",
        "transit_days": 1,
        "cost_per_unit": 7.50,
        "max_capacity": 2000,
        "co2_kg_per_unit": 1.25,
        "reliability_score": 0.98,
    },
    {
        "id": "CARRIER-ROAD-02",
        "name": "TransNational Road Cargo",
        "mode": "Standard Road Freight",
        "transit_days": 3,
        "cost_per_unit": 3.20,
        "max_capacity": 5000,
        "co2_kg_per_unit": 0.45,
        "reliability_score": 0.94,
    },
    {
        "id": "CARRIER-RAIL-03",
        "name": "Intermodal Rail Logistics",
        "mode": "Economy Rail Cargo",
        "transit_days": 5,
        "cost_per_unit": 1.60,
        "max_capacity": 15000,
        "co2_kg_per_unit": 0.18,
        "reliability_score": 0.89,
    },
]

# In-memory storage for active session
_state: Dict[str, Any] = {
    "carrier_fleet": CARRIER_FLEET,
    "evaluated_routes_count": 0,
    "booked_shipments": [],
}


# ── Request / Response Models ──────────────────────────────────────

class ShipmentEvaluationRequest(BaseModel):
    session_id: str
    item: str = "steel_rods"
    quantity: int = Field(..., gt=0)
    supplier_lead_time_days: int = Field(..., ge=0)
    factory_deadline_days: int = Field(..., gt=0)
    max_freight_budget_per_unit: Optional[float] = 10.0
    destination_hub: str = "Factory Plant Alpha"


class CarrierOption(BaseModel):
    carrier_id: str
    carrier_name: str
    transport_mode: str
    transit_days: int
    freight_cost_per_unit: float
    total_delivery_days: int
    feasible: bool
    rejection_reason: Optional[str] = None
    estimated_carbon_kg: float


class ShipmentEvaluationResponse(BaseModel):
    session_id: str
    feasible: bool
    status: str                         # FEASIBLE | INFEASIBLE_DEADLINE | INFEASIBLE_CAPACITY
    recommended_carrier: Optional[CarrierOption]
    all_options: List[CarrierOption]
    min_possible_delivery_days: int
    explanation: str


class ShipmentBookingRequest(BaseModel):
    session_id: str
    carrier_id: str
    item: str
    quantity: int
    contract_price_per_unit: float
    destination_hub: str = "Factory Plant Alpha"


# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/health")
def health() -> Dict[str, str]:
    return {"agent": "logistics", "status": "running", "port": "8004"}


@app.get("/state")
def get_state() -> Dict[str, Any]:
    return {
        "agent": "logistics_agent",
        "port": 8004,
        "evaluated_routes_count": _state["evaluated_routes_count"],
        "booked_shipments_count": len(_state["booked_shipments"]),
        "carriers_available": len(CARRIER_FLEET),
        "fleet": CARRIER_FLEET,
    }


@app.post("/reset")
def reset_session() -> Dict[str, str]:
    _state["evaluated_routes_count"] = 0
    _state["booked_shipments"] = []
    logger.info("Logistics agent state reset.")
    return {"status": "reset", "agent": "logistics"}


@app.post("/evaluate_shipment", response_model=ShipmentEvaluationResponse)
def evaluate_shipment(req: ShipmentEvaluationRequest) -> ShipmentEvaluationResponse:
    """
    Evaluates all freight carrier alternatives for the given supplier lead time and quantity.
    Determines if total delivery days (supplier production + carrier transit) <= factory deadline.
    """
    _state["evaluated_routes_count"] += 1
    options: List[CarrierOption] = []
    feasible_options: List[CarrierOption] = []

    min_transit = min(c["transit_days"] for c in CARRIER_FLEET)
    min_total = req.supplier_lead_time_days + min_transit

    for carrier in CARRIER_FLEET:
        total_days = req.supplier_lead_time_days + carrier["transit_days"]
        carbon_est = round(carrier["co2_kg_per_unit"] * req.quantity, 2)

        is_feasible = True
        rejection_reason = None

        if total_days > req.factory_deadline_days:
            is_feasible = False
            rejection_reason = (
                f"Total fulfillment time ({total_days}d = {req.supplier_lead_time_days}d supplier + "
                f"{carrier['transit_days']}d transit) exceeds factory deadline ({req.factory_deadline_days}d)"
            )
        elif carrier["max_capacity"] < req.quantity:
            is_feasible = False
            rejection_reason = (
                f"Order quantity ({req.quantity}) exceeds carrier max capacity ({carrier['max_capacity']})"
            )
        elif req.max_freight_budget_per_unit and carrier["cost_per_unit"] > req.max_freight_budget_per_unit:
            is_feasible = False
            rejection_reason = (
                f"Freight rate (${carrier['cost_per_unit']:.2f}) exceeds budget (${req.max_freight_budget_per_unit:.2f})"
            )

        option = CarrierOption(
            carrier_id=carrier["id"],
            carrier_name=carrier["name"],
            transport_mode=carrier["mode"],
            transit_days=carrier["transit_days"],
            freight_cost_per_unit=carrier["cost_per_unit"],
            total_delivery_days=total_days,
            feasible=is_feasible,
            rejection_reason=rejection_reason,
            estimated_carbon_kg=carbon_est,
        )
        options.append(option)
        if is_feasible:
            feasible_options.append(option)

    # Sort feasible options by lowest freight cost per unit (cost utility optimization)
    feasible_options.sort(key=lambda o: o.freight_cost_per_unit)

    if feasible_options:
        chosen = feasible_options[0]
        status = "FEASIBLE"
        explanation = (
            f"Logistics APPROVED: Selected '{chosen.carrier_name}' ({chosen.transport_mode}). "
            f"Transit: {chosen.transit_days}d, Total delivery: {chosen.total_delivery_days}d "
            f"(<= deadline {req.factory_deadline_days}d), Freight: ${chosen.freight_cost_per_unit:.2f}/unit."
        )
    else:
        chosen = None
        status = "INFEASIBLE_DEADLINE" if min_total > req.factory_deadline_days else "INFEASIBLE_CAPACITY"
        explanation = (
            f"Logistics REJECTED: Fastest route requires {min_total} days total fulfillment "
            f"({req.supplier_lead_time_days}d production + {min_transit}d express transit), "
            f"violating factory delivery deadline of {req.factory_deadline_days} days."
        )

    response = ShipmentEvaluationResponse(
        session_id=req.session_id,
        feasible=bool(feasible_options),
        status=status,
        recommended_carrier=chosen,
        all_options=options,
        min_possible_delivery_days=min_total,
        explanation=explanation,
    )

    # Log route evaluation to MongoDB
    mongo_store.log_shipment({
        "session_id": req.session_id,
        "event": "route_evaluation",
        "feasible": response.feasible,
        "status": status,
        "recommended_carrier": chosen.model_dump() if chosen else None,
        "supplier_lead_time_days": req.supplier_lead_time_days,
        "factory_deadline_days": req.factory_deadline_days,
        "quantity": req.quantity,
    })

    logger.info("[%s] Evaluation: %s — %s", req.session_id, status, explanation)
    return response


@app.post("/book_shipment")
def book_shipment(req: ShipmentBookingRequest) -> Dict[str, Any]:
    """Books a verified freight shipment upon finalized contract agreement."""
    carrier = next((c for c in CARRIER_FLEET if c["id"] == req.carrier_id), None)
    if not carrier:
        raise HTTPException(status_code=404, detail=f"Carrier {req.carrier_id} not found")

    booking = {
        "booking_id": f"BK-{req.session_id[:8].upper()}-{carrier['mode'][:3].upper()}",
        "session_id": req.session_id,
        "carrier_id": carrier["id"],
        "carrier_name": carrier["name"],
        "mode": carrier["mode"],
        "transit_days": carrier["transit_days"],
        "freight_cost_per_unit": carrier["cost_per_unit"],
        "total_freight_cost": round(carrier["cost_per_unit"] * req.quantity, 2),
        "quantity": req.quantity,
        "item": req.item,
        "destination_hub": req.destination_hub,
        "status": "CONFIRMED",
    }
    _state["booked_shipments"].append(booking)

    # Persist booked shipment manifest to MongoDB
    mongo_store.log_shipment({
        "session_id": req.session_id,
        "event": "shipment_booking",
        "booking": booking,
    })

    logger.info("Shipment BOOKED: %s for session %s", booking["booking_id"], req.session_id)
    return booking


@app.get("/shipments")
def list_shipments() -> Dict[str, Any]:
    return {
        "count": len(_state["booked_shipments"]),
        "shipments": _state["booked_shipments"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("logistics_agent:app", host="0.0.0.0", port=8004, reload=False)
