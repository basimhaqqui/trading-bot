from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from trading_bot.core.audit import AuditLedger
from trading_bot.core.serialization import canonical_json, parse_datetime, require_aware, sha256_digest, utc_now
from trading_bot.execution.alpaca import AlpacaAccount, AlpacaOrder, AlpacaPaperClient, AlpacaPosition


CONTROL_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_execution_control (
    environment TEXT PRIMARY KEY CHECK (environment = 'paper'),
    enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
    kill_switch_active INTEGER NOT NULL CHECK (kill_switch_active IN (0, 1)),
    reason TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_control_events (
    event_id TEXT PRIMARY KEY,
    action TEXT NOT NULL,
    reason TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    digest TEXT NOT NULL
);

CREATE TRIGGER IF NOT EXISTS paper_control_events_no_update
BEFORE UPDATE ON paper_control_events BEGIN
    SELECT RAISE(ABORT, 'paper_control_events is append-only');
END;

CREATE TRIGGER IF NOT EXISTS paper_control_events_no_delete
BEFORE DELETE ON paper_control_events BEGIN
    SELECT RAISE(ABORT, 'paper_control_events is append-only');
END;
"""


LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_account_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    observed_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    digest TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_order_events (
    event_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    client_order_id TEXT NOT NULL,
    status TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    remote_updated_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    digest TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_paper_order_events_order
ON paper_order_events(order_id, remote_updated_at, status);

CREATE TRIGGER IF NOT EXISTS paper_account_snapshots_no_update
BEFORE UPDATE ON paper_account_snapshots BEGIN
    SELECT RAISE(ABORT, 'paper_account_snapshots is append-only');
END;

CREATE TRIGGER IF NOT EXISTS paper_account_snapshots_no_delete
BEFORE DELETE ON paper_account_snapshots BEGIN
    SELECT RAISE(ABORT, 'paper_account_snapshots is append-only');
END;

CREATE TRIGGER IF NOT EXISTS paper_order_events_no_update
BEFORE UPDATE ON paper_order_events BEGIN
    SELECT RAISE(ABORT, 'paper_order_events is append-only');
END;

CREATE TRIGGER IF NOT EXISTS paper_order_events_no_delete
BEFORE DELETE ON paper_order_events BEGIN
    SELECT RAISE(ABORT, 'paper_order_events is append-only');
END;
"""


@dataclass(frozen=True)
class PaperControlStatus:
    enabled: bool
    kill_switch_active: bool
    reason: str
    updated_at: datetime

    @property
    def ready(self) -> bool:
        return self.enabled and not self.kill_switch_active


@dataclass(frozen=True)
class ReconciliationResult:
    observed_at: datetime
    account_snapshot_added: bool
    order_events_added: int
    remote_orders: int
    open_orders: int
    missing_remote_client_order_ids: tuple[str, ...]
    unexpected_remote_client_order_ids: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not self.missing_remote_client_order_ids and not self.unexpected_remote_client_order_ids


class PaperControlStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.executescript(CONTROL_SCHEMA)
            connection.execute(
                """
                INSERT OR IGNORE INTO paper_execution_control (
                    environment, enabled, kill_switch_active, reason, updated_at
                ) VALUES ('paper', 0, 1, 'paper execution defaults locked', ?)
                """,
                (now.isoformat(),),
            )

    def status(self) -> PaperControlStatus:
        self.initialize()
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM paper_execution_control WHERE environment = 'paper'"
            ).fetchone()
        if row is None:
            raise RuntimeError("paper control state is missing")
        return PaperControlStatus(
            bool(row["enabled"]),
            bool(row["kill_switch_active"]),
            row["reason"],
            parse_datetime(row["updated_at"]),
        )

    def enable(self, *, confirmation: str, reason: str, now: datetime | None = None) -> PaperControlStatus:
        self._require_confirmation(confirmation)
        return self._change(enabled=True, reason=reason, action="enable", now=now)

    def disable(self, *, reason: str, now: datetime | None = None) -> PaperControlStatus:
        return self._change(enabled=False, reason=reason, action="disable", now=now)

    def activate_kill_switch(
        self, *, reason: str, now: datetime | None = None
    ) -> PaperControlStatus:
        return self._change(
            enabled=False,
            kill_switch_active=True,
            reason=reason,
            action="kill",
            now=now,
        )

    def release_kill_switch(
        self, *, confirmation: str, reason: str, now: datetime | None = None
    ) -> PaperControlStatus:
        self._require_confirmation(confirmation)
        return self._change(
            kill_switch_active=False,
            reason=reason,
            action="release_kill_switch",
            now=now,
        )

    def _change(
        self,
        *,
        reason: str,
        action: str,
        now: datetime | None,
        enabled: bool | None = None,
        kill_switch_active: bool | None = None,
    ) -> PaperControlStatus:
        if not reason.strip():
            raise ValueError("control changes require a reason")
        current = self.status()
        changed_at = require_aware(now or utc_now(), "now")
        next_enabled = current.enabled if enabled is None else enabled
        next_kill = current.kill_switch_active if kill_switch_active is None else kill_switch_active
        payload = {
            "action": action,
            "previous_enabled": current.enabled,
            "previous_kill_switch_active": current.kill_switch_active,
            "enabled": next_enabled,
            "kill_switch_active": next_kill,
            "reason": reason,
            "occurred_at": changed_at,
        }
        event_id = sha256_digest(payload)
        payload_json = canonical_json(payload)
        digest = sha256_digest(json.loads(payload_json))
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE paper_execution_control
                SET enabled = ?, kill_switch_active = ?, reason = ?, updated_at = ?
                WHERE environment = 'paper'
                """,
                (int(next_enabled), int(next_kill), reason, changed_at.isoformat()),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO paper_control_events (
                    event_id, action, reason, occurred_at, payload_json, digest
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (event_id, action, reason, changed_at.isoformat(), payload_json, digest),
            )
        return PaperControlStatus(next_enabled, next_kill, reason, changed_at)

    @staticmethod
    def _require_confirmation(value: str) -> None:
        if value != "PAPER-ONLY":
            raise PermissionError("paper control requires confirmation PAPER-ONLY")


class PaperExecutionLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(LEDGER_SCHEMA)

    def append_account(
        self,
        account: AlpacaAccount,
        positions: tuple[AlpacaPosition, ...],
    ) -> bool:
        self.initialize()
        payload = {"account": account, "positions": positions}
        snapshot_id = sha256_digest(payload)
        payload_json = canonical_json(payload)
        digest = sha256_digest(json.loads(payload_json))
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO paper_account_snapshots (
                    snapshot_id, observed_at, payload_json, digest
                ) VALUES (?, ?, ?, ?)
                """,
                (snapshot_id, account.observed_at.isoformat(), payload_json, digest),
            )
        return cursor.rowcount > 0

    def append_order(self, order: AlpacaOrder, *, observed_at: datetime) -> bool:
        self.initialize()
        observed_at = require_aware(observed_at, "observed_at")
        event_payload = {"order": order, "observed_at": observed_at}
        event_id = sha256_digest(event_payload)
        payload_json = canonical_json(event_payload)
        digest = sha256_digest(json.loads(payload_json))
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO paper_order_events (
                    event_id, order_id, client_order_id, status, observed_at,
                    remote_updated_at, payload_json, digest
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    order.order_id,
                    order.client_order_id,
                    order.status,
                    observed_at.isoformat(),
                    order.updated_at.isoformat(),
                    payload_json,
                    digest,
                ),
            )
        return cursor.rowcount > 0

    def verify_integrity(self) -> int:
        self.initialize()
        total = 0
        with self.connect() as connection:
            for table in ("paper_account_snapshots", "paper_order_events"):
                rows = connection.execute(
                    f"SELECT payload_json, digest FROM {table}"
                ).fetchall()
                total += len(rows)
                for row in rows:
                    if sha256_digest(json.loads(row["payload_json"])) != row["digest"]:
                        raise RuntimeError(f"digest mismatch in {table}")
        return total


class PaperReconciler:
    def __init__(
        self,
        client: AlpacaPaperClient,
        ledger: PaperExecutionLedger,
        audit: AuditLedger,
    ) -> None:
        self.client = client
        self.ledger = ledger
        self.audit = audit

    def run(self, *, observed_at: datetime | None = None) -> ReconciliationResult:
        observed_at = require_aware(observed_at or utc_now(), "observed_at")
        account = self.client.account(observed_at=observed_at)
        positions = self.client.positions()
        orders = self.client.orders(status="all")
        account_added = self.ledger.append_account(account, positions)
        order_events_added = sum(
            int(self.ledger.append_order(order, observed_at=observed_at)) for order in orders
        )
        expected = {
            receipt.client_order_id
            for receipt in self.audit.execution_receipts()[-500:]
            if receipt.client_order_id
        }
        remote = {order.client_order_id for order in orders if order.client_order_id.startswith("tb-")}
        missing = tuple(sorted(expected - remote))
        unexpected = tuple(sorted(remote - expected))
        open_statuses = {
            "new",
            "accepted",
            "pending_new",
            "partially_filled",
            "pending_cancel",
            "pending_replace",
            "accepted_for_bidding",
            "held",
        }
        return ReconciliationResult(
            observed_at,
            account_added,
            order_events_added,
            len(orders),
            sum(order.status in open_statuses for order in orders),
            missing,
            unexpected,
        )
