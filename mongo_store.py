"""
mongo_store.py  —  MongoDB Document Store & Multi-Agent Audit Trail
════════════════════════════════════════════════════════════════════
Provides schema-free, document-oriented persistent storage for:
  1. Negotiation Sessions & LangGraph state transitions
  2. Finalized Contracts & Multilateral Agreements
  3. Logistics & Freight manifests
  4. Horn-Clause Audit verification traces

Resilience:
  Auto-detects live MongoDB (local or Atlas via MONGO_URI in .env).
  If MongoDB is unavailable, gracefully falls back to local JSON
  document storage at logs/mongo_audit_store.json without crashing.
════════════════════════════════════════════════════════════════════
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("mongo_store")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [MongoStore] %(message)s")

_FALLBACK_FILE = Path(__file__).parent / "logs" / "mongo_audit_store.json"
_FALLBACK_FILE.parent.mkdir(exist_ok=True)


class MongoStore:
    """Document store for multi-agent procurement traces with graceful offline fallback."""

    def __init__(self, uri: Optional[str] = None, db_name: str = "procurement_network"):
        self.uri = uri or os.environ.get("MONGO_URI", "mongodb://localhost:27017")
        self.db_name = db_name
        self.is_connected = False
        self._client = None
        self._db = None

        self._connect()

    def _connect(self) -> None:
        try:
            import pymongo
            # Short server selection timeout so offline state is detected rapidly
            client = pymongo.MongoClient(self.uri, serverSelectionTimeoutMS=1500)
            # Test connection with ping
            client.admin.command("ping")
            self._client = client
            self._db = client[self.db_name]
            self.is_connected = True
            logger.info("Connected to MongoDB at %s (DB: %s)", self.uri, self.db_name)
        except Exception as e:
            self.is_connected = False
            logger.info(
                "MongoDB not connected at %s (%s). Using resilient local JSON document store: %s",
                self.uri,
                e.__class__.__name__,
                _FALLBACK_FILE,
            )

    def _read_fallback(self) -> Dict[str, List[Dict[str, Any]]]:
        if not _FALLBACK_FILE.exists():
            return {
                "negotiation_sessions": [],
                "contracts": [],
                "logistics_shipments": [],
                "audit_traces": [],
            }
        try:
            with open(_FALLBACK_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {
                "negotiation_sessions": [],
                "contracts": [],
                "logistics_shipments": [],
                "audit_traces": [],
            }

    def _write_fallback(self, data: Dict[str, List[Dict[str, Any]]]) -> None:
        try:
            with open(_FALLBACK_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error("Failed to write to fallback store: %s", e)

    def log_session(self, session_data: Dict[str, Any]) -> str:
        """Insert or update a full negotiation session document."""
        session_data = dict(session_data)
        session_data.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

        if self.is_connected and self._db is not None:
            try:
                coll = self._db["negotiation_sessions"]
                session_id = session_data.get("session_id")
                if session_id:
                    res = coll.update_one({"session_id": session_id}, {"$set": session_data}, upsert=True)
                    return str(res.upserted_id or session_id)
                else:
                    res = coll.insert_one(session_data)
                    return str(res.inserted_id)
            except Exception as e:
                logger.warning("MongoDB write failed, saving to fallback: %s", e)

        # Fallback
        store = self._read_fallback()
        session_id = session_data.get("session_id")
        replaced = False
        if session_id:
            for idx, existing in enumerate(store.get("negotiation_sessions", [])):
                if existing.get("session_id") == session_id:
                    store["negotiation_sessions"][idx] = session_data
                    replaced = True
                    break
        if not replaced:
            store.setdefault("negotiation_sessions", []).append(session_data)
        self._write_fallback(store)
        return str(session_id or len(store["negotiation_sessions"]))

    def log_contract(self, contract_data: Dict[str, Any]) -> str:
        """Log a finalized agreement contract."""
        contract_data = dict(contract_data)
        contract_data.setdefault("finalized_at", datetime.now(timezone.utc).isoformat())

        if self.is_connected and self._db is not None:
            try:
                res = self._db["contracts"].insert_one(contract_data)
                return str(res.inserted_id)
            except Exception as e:
                logger.warning("MongoDB contract write failed, saving to fallback: %s", e)

        store = self._read_fallback()
        store.setdefault("contracts", []).append(contract_data)
        self._write_fallback(store)
        return str(len(store["contracts"]))

    def log_shipment(self, shipment_data: Dict[str, Any]) -> str:
        """Log a logistics freight booking or route evaluation."""
        shipment_data = dict(shipment_data)
        shipment_data.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

        if self.is_connected and self._db is not None:
            try:
                res = self._db["logistics_shipments"].insert_one(shipment_data)
                return str(res.inserted_id)
            except Exception as e:
                logger.warning("MongoDB shipment write failed, saving to fallback: %s", e)

        store = self._read_fallback()
        store.setdefault("logistics_shipments", []).append(shipment_data)
        self._write_fallback(store)
        return str(len(store["logistics_shipments"]))

    def log_audit_trace(self, trace_data: Dict[str, Any]) -> str:
        """Log a Horn-clause audit inference step."""
        trace_data = dict(trace_data)
        trace_data.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

        if self.is_connected and self._db is not None:
            try:
                res = self._db["audit_traces"].insert_one(trace_data)
                return str(res.inserted_id)
            except Exception as e:
                logger.warning("MongoDB audit trace write failed, saving to fallback: %s", e)

        store = self._read_fallback()
        store.setdefault("audit_traces", []).append(trace_data)
        self._write_fallback(store)
        return str(len(store["audit_traces"]))

    def get_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent negotiation sessions."""
        if self.is_connected and self._db is not None:
            try:
                cursor = self._db["negotiation_sessions"].find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
                return list(cursor)
            except Exception as e:
                logger.warning("MongoDB read failed, falling back: %s", e)

        store = self._read_fallback()
        sessions = store.get("negotiation_sessions", [])
        return sessions[-limit:]

    def get_contracts(self) -> List[Dict[str, Any]]:
        """Retrieve all finalized contracts."""
        if self.is_connected and self._db is not None:
            try:
                cursor = self._db["contracts"].find({}, {"_id": 0})
                return list(cursor)
            except Exception as e:
                logger.warning("MongoDB read failed, falling back: %s", e)

        store = self._read_fallback()
        return store.get("contracts", [])


# Singleton instance
mongo_store = MongoStore()
