"""Optional PostgreSQL/pgvector store for vendor name/address fingerprints.

Raw bank-account values are never persisted by this class. A keyed HMAC is stored
instead and FINTRACE_BANK_HMAC_KEY must be configured before bank matching is used.
"""
from __future__ import annotations

import hashlib
import hmac
import os
from typing import Any

from src.vendor_similarity import EMBEDDING_DIMENSIONS, make_text_embedding, normalize_bank_identifier


class PgVectorVendorStore:
    def __init__(self, database_url: str | None = None, bank_hmac_key: str | None = None):
        self.database_url = database_url or os.environ.get("FINTRACE_DATABASE_URL", "")
        self.bank_hmac_key = (bank_hmac_key or os.environ.get("FINTRACE_BANK_HMAC_KEY", "")).encode("utf-8")
        if not self.database_url:
            raise ValueError("FINTRACE_DATABASE_URL is required to initialize pgvector storage")

    def _connect(self):
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as exc:
            raise RuntimeError("Optional dependencies are required: install psycopg and pgvector") from exc
        connection = psycopg.connect(self.database_url)
        register_vector(connection)
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute(
                f"CREATE TABLE IF NOT EXISTS fintrace_vendor_profiles ("
                f"supplier_id TEXT PRIMARY KEY, embedding vector({EMBEDDING_DIMENSIONS}) NOT NULL, "
                "bank_hmac TEXT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            connection.commit()

    @staticmethod
    def _profile_text(vendor_name: str, address: str, country: str) -> str:
        return "\n".join([str(vendor_name or "").strip(), str(address or "").strip(), str(country or "").strip()])

    def _bank_hmac(self, bank_account: Any) -> str | None:
        normalized = normalize_bank_identifier(bank_account)
        if not normalized:
            return None
        if not self.bank_hmac_key:
            raise ValueError("Set FINTRACE_BANK_HMAC_KEY before storing or comparing bank-account values")
        return hmac.new(self.bank_hmac_key, normalized.encode("utf-8"), hashlib.sha256).hexdigest()

    def upsert_profile(self, supplier_id: str, vendor_name: str, address: str = "", country: str = "", bank_account: str | None = None) -> dict:
        if not str(supplier_id or "").strip():
            raise ValueError("supplier_id is required")
        profile_text = self._profile_text(vendor_name, address, country)
        if not profile_text.strip():
            raise ValueError("At least one vendor profile text field is required")
        embedding = make_text_embedding(profile_text)
        bank_fingerprint = self._bank_hmac(bank_account)
        try:
            from pgvector import Vector
        except ImportError as exc:
            raise RuntimeError("Install pgvector to create vector values for PostgreSQL") from exc
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO fintrace_vendor_profiles (supplier_id, embedding, bank_hmac, updated_at) "
                "VALUES (%s, %s, %s, CURRENT_TIMESTAMP) "
                "ON CONFLICT (supplier_id) DO UPDATE SET embedding = EXCLUDED.embedding, "
                "bank_hmac = COALESCE(EXCLUDED.bank_hmac, fintrace_vendor_profiles.bank_hmac), "
                "updated_at = CURRENT_TIMESTAMP",
                (str(supplier_id).strip(), Vector(embedding), bank_fingerprint),
            )
            connection.commit()
        return {"stored": True, "supplier_id": str(supplier_id).strip(), "bank_fingerprint_stored": bank_fingerprint is not None}

    def compare_profile(self, supplier_id: str, vendor_name: str, address: str = "", country: str = "", payment_bank: str | None = None) -> dict:
        try:
            from pgvector import Vector
        except ImportError as exc:
            raise RuntimeError("Install pgvector to query vector values in PostgreSQL") from exc
        query_embedding = make_text_embedding(self._profile_text(vendor_name, address, country))
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 - (embedding <=> %s) AS similarity, bank_hmac "
                "FROM fintrace_vendor_profiles WHERE supplier_id = %s",
                (Vector(query_embedding), str(supplier_id).strip()),
            ).fetchone()
        if row is None:
            return {"found": False, "name_similarity": None, "bank_match": None}
        similarity = float(row[0]) if row[0] is not None else None
        stored_fingerprint = row[1]
        submitted_fingerprint = self._bank_hmac(payment_bank)
        bank_match = None
        if stored_fingerprint is not None and submitted_fingerprint is not None:
            bank_match = hmac.compare_digest(str(stored_fingerprint), submitted_fingerprint)
        return {"found": True, "name_similarity": similarity, "bank_match": bank_match}
