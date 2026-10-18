"""Development runtime storage: persistent duplicate guard and hash-chained audit log.

For production multi-node deployment, replace the SQLite duplicate guard with a
transactional PostgreSQL unique constraint and place audit records in a protected,
append-only/WORM or cryptographically signed enterprise audit system.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any


class InvoiceIdLedger:
    """Atomic local idempotency ledger that stores only a hash of invoice IDs."""
    def __init__(self, database_path: str):
        self.database_path = database_path
        os.makedirs(os.path.dirname(os.path.abspath(database_path)), exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS invoice_id_ledger ("
                "invoice_id_hash TEXT PRIMARY KEY, first_seen_utc TEXT NOT NULL)"
            )
            connection.commit()

    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=15.0, isolation_level="IMMEDIATE")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    @staticmethod
    def _digest(invoice_id: str) -> str:
        return hashlib.sha256(str(invoice_id).strip().encode("utf-8")).hexdigest()

    def reserve(self, invoice_id: Any) -> bool:
        """Return True for the first occurrence; False when this invoice ID was seen."""
        if invoice_id is None or not str(invoice_id).strip():
            return True
        digest = self._digest(str(invoice_id))
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO invoice_id_ledger (invoice_id_hash, first_seen_utc) VALUES (?, ?)",
                (digest, timestamp),
            )
            connection.commit()
            return cursor.rowcount == 1


class HashChainAuditLog:
    """Append-only JSONL audit trail with a SHA-256 predecessor hash per record."""
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def _canonical(value: dict) -> bytes:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")

    def _last_hash(self) -> str:
        if not os.path.isfile(self.path):
            return "0" * 64
        last_line = ""
        with open(self.path, "rb") as stream:
            for line in stream:
                if line.strip():
                    last_line = line.decode("utf-8").strip()
        if not last_line:
            return "0" * 64
        try:
            return str(json.loads(last_line)["record_hash"])
        except (json.JSONDecodeError, KeyError, TypeError):
            raise ValueError("Audit log's last record is malformed; refusing to append")

    def append(self, event: dict) -> dict:
        """Append a sanitized event and flush it to disk before returning."""
        with self._lock:
            previous_hash = self._last_hash()
            body = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "previous_hash": previous_hash,
                "event": event,
            }
            record_hash = hashlib.sha256(self._canonical(body)).hexdigest()
            record = {**body, "record_hash": record_hash}
            encoded = self._canonical(record) + b"\n"
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(descriptor, encoded)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return record

    def read_recent(self, limit: int = 100) -> list[dict]:
        if not os.path.isfile(self.path):
            return []
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        with open(self.path, "r", encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        return records[-limit:]

    def verify(self) -> dict:
        """Verify the full hash chain; this is tamper-evident, not WORM immutability."""
        if not os.path.isfile(self.path):
            return {"valid": True, "records_checked": 0, "reason": "audit file does not exist yet"}
        previous_hash = "0" * 64
        count = 0
        with open(self.path, "r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    recorded_hash = record.pop("record_hash")
                except (json.JSONDecodeError, KeyError, TypeError):
                    return {"valid": False, "records_checked": count, "failed_line": line_number, "reason": "invalid JSON or missing record_hash"}
                if record.get("previous_hash") != previous_hash:
                    return {"valid": False, "records_checked": count, "failed_line": line_number, "reason": "previous hash does not match"}
                calculated = hashlib.sha256(self._canonical(record)).hexdigest()
                if calculated != recorded_hash:
                    return {"valid": False, "records_checked": count, "failed_line": line_number, "reason": "record hash mismatch"}
                previous_hash = recorded_hash
                count += 1
        return {"valid": True, "records_checked": count, "reason": "hash chain is internally consistent"}
