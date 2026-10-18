"""FastAPI interface for real-time evaluation, what-if simulation, and audit."""
from __future__ import annotations

import hashlib
import hmac
import math
import os
from contextlib import asynccontextmanager
from typing import Any

import joblib
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from src.config import AUDIT_LOG_PATH, DUPLICATE_LEDGER_PATH, MODEL_PATH
from src.inference_engine import score_transaction
from src.pgvector_store import PgVectorVendorStore
from src.runtime_store import HashChainAuditLog, InvoiceIdLedger
from src.simulator import simulate_transaction


class InvoicePayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    invoice_id: str | None = None
    supplier_id: str | None = None
    invoice_amount: float | None = None
    currency: str | None = None
    invoice_date: str | None = None
    submission_hour: int | None = None


class SimulationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transaction: dict[str, Any] = Field(default_factory=dict)
    changes: dict[str, Any] = Field(default_factory=dict)


class VendorProfilePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    supplier_id: str
    vendor_name: str
    address: str = ""
    country: str = ""
    registered_bank_account: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = joblib.load(MODEL_PATH) if os.path.isfile(MODEL_PATH) else None
    app.state.audit_log = HashChainAuditLog(AUDIT_LOG_PATH)
    app.state.invoice_ledger = InvoiceIdLedger(DUPLICATE_LEDGER_PATH)
    app.state.vendor_store = None
    if os.environ.get("FINTRACE_DATABASE_URL"):
        store = PgVectorVendorStore()
        store.initialize()
        app.state.vendor_store = store
    yield


app = FastAPI(
    title="FinTrace Payment Firewall API",
    description="Hybrid AP fraud scoring, policy enforcement, and what-if simulation.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for browser access and external dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    """Require a configured API key; unset means local development mode only."""
    expected = os.environ.get("FINTRACE_API_KEY")
    if expected and (x_api_key is None or not hmac.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="A valid X-API-Key header is required")


def _validate_transaction(transaction: dict, require_invoice_id: bool = False) -> None:
    """Fail closed on missing or invalid monetary fields before model scoring."""
    if require_invoice_id and not str(transaction.get("invoice_id", "")).strip():
        raise HTTPException(status_code=422, detail="invoice_id is required for payment evaluation")
    if transaction.get("invoice_amount") is None:
        raise HTTPException(status_code=422, detail="invoice_amount is required")
    try:
        amount = float(transaction["invoice_amount"])
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="invoice_amount must be numeric") from exc
    if not math.isfinite(amount) or amount < 0:
        raise HTTPException(status_code=422, detail="invoice_amount must be a finite non-negative number")


def _bundle_or_503(request: Request) -> dict:
    bundle = request.app.state.bundle
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail=f"No trained model artifact found at {MODEL_PATH}. Run `python main.py` first.",
        )
    return bundle


def _invoice_hash(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return hashlib.sha256(str(value).strip().encode("utf-8")).hexdigest()


def _write_audit(request: Request, operation: str, payload: dict, result: dict) -> None:
    safe_event = {
        "operation": operation,
        "invoice_id_hash": _invoice_hash(payload.get("invoice_id")),
        "supplier_id_hash": _invoice_hash(payload.get("supplier_id")),
        "decision": result.get("decision") or result.get("simulated", {}).get("decision"),
        "risk_score": result.get("risk_score") if "risk_score" in result else result.get("simulated", {}).get("risk_score"),
        "hard_rules": result.get("hard_rules", []),
        "simulation_only": bool(result.get("simulation_only", False)),
    }
    request.app.state.audit_log.append(safe_event)


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model_loaded": request.app.state.bundle is not None,
        "classifier_backend": request.app.state.bundle.get("classifier_backend") if request.app.state.bundle else None,
        "pgvector_enabled": request.app.state.vendor_store is not None,
    }


@app.post("/evaluate")
def evaluate(payload: InvoicePayload, request: Request, _: None = Depends(require_api_key)):
    bundle = _bundle_or_503(request)
    transaction = payload.model_dump(exclude_none=True)
    _validate_transaction(transaction, require_invoice_id=True)
    duplicate_detected = False
    invoice_id = transaction.get("invoice_id")

    vendor_store = request.app.state.vendor_store
    if vendor_store is not None and transaction.get("supplier_id") and transaction.get("vendor_name"):
        match = vendor_store.compare_profile(
            supplier_id=transaction["supplier_id"],
            vendor_name=str(transaction.get("vendor_name", "")),
            address=str(transaction.get("vendor_address", transaction.get("address", ""))),
            country=str(transaction.get("supplier_country", "")),
            payment_bank=transaction.get("payment_bank", transaction.get("payment_bank_account")),
        )
        if match.get("found"):
            if match.get("name_similarity") is not None:
                transaction["vendor_name_similarity"] = float(match["name_similarity"])
            if match.get("bank_match") is False:
                transaction["bank_mismatch_flag"] = 1
            elif match.get("bank_match") is True:
                transaction["bank_mismatch_flag"] = 0
            transaction["vendor_profile_found"] = True
        else:
            transaction["vendor_profile_found"] = False

    if invoice_id:
        duplicate_detected = not request.app.state.invoice_ledger.reserve(invoice_id)
    result = score_transaction(transaction, bundle, duplicate_detected=duplicate_detected)
    _write_audit(request, "evaluate", transaction, result)
    return {"invoice_id": invoice_id, **result}


@app.post("/simulate")
def simulate(payload: SimulationPayload, request: Request, _: None = Depends(require_api_key)):
    bundle = _bundle_or_503(request)
    transaction = dict(payload.transaction)
    _validate_transaction(transaction)
    proposed = dict(transaction)
    proposed.update(payload.changes)
    _validate_transaction(proposed)
    result = simulate_transaction(transaction, payload.changes, bundle)
    _write_audit(request, "simulate", transaction, result)
    return result


@app.get("/audit")
def audit(
    request: Request,
    limit: int = Query(default=100, ge=1, le=1000),
    verify: bool = False,
    _: None = Depends(require_api_key),
):
    log = request.app.state.audit_log
    response = {"records": log.read_recent(limit)}
    if verify:
        response["integrity_check"] = log.verify()
    return response


@app.post("/vendor/profile")
def upsert_vendor_profile(payload: VendorProfilePayload, request: Request, _: None = Depends(require_api_key)):
    store = request.app.state.vendor_store
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="pgvector storage is disabled; set FINTRACE_DATABASE_URL and install psycopg + pgvector.",
        )
    try:
        result = store.upsert_profile(
            supplier_id=payload.supplier_id,
            vendor_name=payload.vendor_name,
            address=payload.address,
            country=payload.country,
            bank_account=payload.registered_bank_account,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    request.app.state.audit_log.append({
        "operation": "vendor_profile_upsert",
        "supplier_id_hash": _invoice_hash(payload.supplier_id),
        "decision": "PROFILE_UPDATED",
        "bank_fingerprint_stored": result.get("bank_fingerprint_stored", False),
    })
    return result