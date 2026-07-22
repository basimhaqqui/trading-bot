from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from trading_bot.core.serialization import require_aware, utc_now
from trading_bot.execution.risk import ApprovalSigner
from trading_bot.execution.schemas import ApprovedOrderIntent, ExecutionEnvironment


@dataclass(frozen=True)
class ExecutionReceipt:
    intent_id: str
    environment: ExecutionEnvironment
    status: str
    executed_at: datetime
    venue_order_id: str | None = None
    client_order_id: str | None = None
    filled_quantity: float = 0.0
    average_fill_price: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "executed_at", require_aware(self.executed_at, "executed_at"))
        if not self.intent_id or not self.status:
            raise ValueError("receipt intent_id and status are required")
        if self.filled_quantity < 0:
            raise ValueError("filled_quantity cannot be negative")
        if self.average_fill_price is not None and self.average_fill_price <= 0:
            raise ValueError("average_fill_price must be positive")


class ExecutionAdapter(Protocol):
    environment: ExecutionEnvironment

    def submit(self, approval: ApprovedOrderIntent, *, now: datetime) -> ExecutionReceipt:
        ...


class PaperLedgerAdapter:
    def __init__(self, environment: ExecutionEnvironment = ExecutionEnvironment.SHADOW) -> None:
        if environment is ExecutionEnvironment.LIVE:
            raise ValueError("PaperLedgerAdapter cannot run live")
        self.environment = environment
        self.receipts: list[ExecutionReceipt] = []

    def submit(self, approval: ApprovedOrderIntent, *, now: datetime) -> ExecutionReceipt:
        receipt = ExecutionReceipt(
            intent_id=approval.intent.intent_id,
            environment=self.environment,
            status="recorded",
            executed_at=now,
        )
        self.receipts.append(receipt)
        return receipt


class DeterministicExecutor:
    def __init__(self, signer: ApprovalSigner, adapter: ExecutionAdapter) -> None:
        self.signer = signer
        self.adapter = adapter
        self._executed_intents: set[str] = set()

    def execute(
        self, approval: ApprovedOrderIntent, *, now: datetime | None = None
    ) -> ExecutionReceipt:
        now = require_aware(now or utc_now(), "now")
        if not self.signer.verify(approval):
            raise PermissionError("risk approval signature is invalid")
        if now >= approval.approval_expires_at:
            raise PermissionError("risk approval has expired")
        if approval.intent.environment is not self.adapter.environment:
            raise PermissionError("execution environment does not match adapter")
        if approval.intent.intent_id in self._executed_intents:
            raise PermissionError("intent has already been executed")
        receipt = self.adapter.submit(approval, now=now)
        self._executed_intents.add(approval.intent.intent_id)
        return receipt
