from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from trading_bot.core.schemas import AssetClass, Instrument
from trading_bot.core.serialization import parse_datetime, require_aware, utc_now
from trading_bot.execution.control import ExecutionReceipt
from trading_bot.execution.schemas import (
    ApprovedOrderIntent,
    ExecutionEnvironment,
    OrderSide,
    OrderType,
    PortfolioSnapshot,
    Position,
    TimeInForce,
)


JsonValue = dict[str, Any] | list[Any]


class AlpacaPaperError(RuntimeError):
    pass


class TradingTransport(Protocol):
    def get_json(
        self, path: str, *, query: Mapping[str, str | int | float | bool | None] | None = None
    ) -> JsonValue:
        ...

    def post_json(self, path: str, payload: Mapping[str, Any]) -> JsonValue:
        ...

    def delete_json(self, path: str) -> JsonValue:
        ...


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


@dataclass(frozen=True)
class PinnedTradingHttpTransport:
    api_key_id: str
    api_secret_key: str
    base_url: str = "https://paper-api.alpaca.markets"
    allowed_host: str = "paper-api.alpaca.markets"
    timeout_seconds: float = 10.0
    max_response_bytes: int = 5_000_000

    def __post_init__(self) -> None:
        if not self.api_key_id or not self.api_secret_key:
            raise ValueError("Alpaca paper credentials are required")
        for value in (self.api_key_id, self.api_secret_key):
            if "\r" in value or "\n" in value:
                raise ValueError("credential values cannot contain line breaks")
        parts = urlsplit(self.base_url)
        if (
            parts.scheme != "https"
            or parts.hostname != self.allowed_host
            or parts.port not in (None, 443)
            or parts.username is not None
            or parts.password is not None
            or self.allowed_host != "paper-api.alpaca.markets"
        ):
            raise ValueError("paper transport must be pinned to paper-api.alpaca.markets")
        if self.timeout_seconds <= 0 or self.max_response_bytes < 1:
            raise ValueError("timeout and response limit must be positive")

    def get_json(
        self, path: str, *, query: Mapping[str, str | int | float | bool | None] | None = None
    ) -> JsonValue:
        return self._request("GET", path, query=query)

    def post_json(self, path: str, payload: Mapping[str, Any]) -> JsonValue:
        return self._request("POST", path, payload=payload)

    def delete_json(self, path: str) -> JsonValue:
        return self._request("DELETE", path)

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, str | int | float | bool | None] | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> JsonValue:
        path_parts = urlsplit(path)
        if not path.startswith("/") or path.startswith("//") or path_parts.scheme or path_parts.netloc:
            raise ValueError("path must remain on the configured paper host")
        url = urljoin(self.base_url.rstrip("/") + "/", path.lstrip("/"))
        final = urlsplit(url)
        if (
            final.scheme != "https"
            or final.hostname != self.allowed_host
            or final.port not in (None, 443)
        ):
            raise ValueError("request escaped the configured paper host")
        values = {key: value for key, value in (query or {}).items() if value is not None}
        if values:
            url = f"{url}?{urlencode(values)}"
        body = None
        headers = {
            "Accept": "application/json",
            "Cache-Control": "no-cache",
            "User-Agent": "trading-bot-paper-executor/0.1",
            "APCA-API-KEY-ID": self.api_key_id,
            "APCA-API-SECRET-KEY": self.api_secret_key,
        }
        if payload is not None:
            body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(url, data=body, headers=headers, method=method)
        try:
            with build_opener(_NoRedirects()).open(request, timeout=self.timeout_seconds) as response:
                raw = response.read(self.max_response_bytes + 1)
        except HTTPError as exc:
            raise AlpacaPaperError(
                f"{method} failed for paper-api.alpaca.markets with HTTP {exc.code}"
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise AlpacaPaperError(
                f"{method} failed for paper-api.alpaca.markets"
            ) from exc
        if len(raw) > self.max_response_bytes:
            raise AlpacaPaperError("paper API response exceeded configured size limit")
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AlpacaPaperError("paper API response was not valid JSON") from exc
        if not isinstance(parsed, (dict, list)):
            raise AlpacaPaperError("paper API response must be an object or list")
        _validate_json(parsed)
        return parsed


@dataclass(frozen=True)
class AlpacaAccount:
    account_id: str
    status: str
    equity: float
    last_equity: float
    cash: float
    buying_power: float
    trading_blocked: bool
    account_blocked: bool
    trade_suspended_by_user: bool
    observed_at: datetime

    @property
    def daily_return(self) -> float:
        if self.last_equity <= 0:
            return 0.0
        return self.equity / self.last_equity - 1

    @property
    def can_trade(self) -> bool:
        return (
            self.status.upper() == "ACTIVE"
            and not self.trading_blocked
            and not self.account_blocked
            and not self.trade_suspended_by_user
        )


@dataclass(frozen=True)
class AlpacaPosition:
    symbol: str
    asset_class: AssetClass
    quantity: float
    side: str
    market_value: float
    average_entry_price: float
    current_price: float
    unrealized_pl: float


@dataclass(frozen=True)
class AlpacaOrder:
    order_id: str
    client_order_id: str
    symbol: str
    asset_class: AssetClass
    side: OrderSide
    order_type: str
    time_in_force: str
    status: str
    quantity: float
    filled_quantity: float
    limit_price: float | None
    average_fill_price: float | None
    submitted_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class PaperOrderRequest:
    client_order_id: str
    symbol: str
    asset_class: AssetClass
    side: OrderSide
    order_type: OrderType
    time_in_force: TimeInForce
    quantity: float
    limit_price: float | None = None

    def __post_init__(self) -> None:
        if not self.client_order_id or len(self.client_order_id) > 48:
            raise ValueError("client_order_id must contain 1 to 48 characters")
        if not self.symbol or self.quantity <= 0:
            raise ValueError("symbol and positive quantity are required")
        if self.asset_class not in {AssetClass.EQUITY, AssetClass.OPTION}:
            raise ValueError("Alpaca paper execution supports equities and options only")
        if self.order_type not in {OrderType.MARKET, OrderType.LIMIT}:
            raise ValueError("Alpaca paper execution supports market and limit orders only")
        if self.order_type is OrderType.LIMIT and (
            self.limit_price is None or self.limit_price <= 0
        ):
            raise ValueError("limit orders require a positive limit price")
        if self.asset_class is AssetClass.OPTION:
            if not self.quantity.is_integer():
                raise ValueError("option orders require a whole contract quantity")
            if self.time_in_force is not TimeInForce.DAY:
                raise ValueError("option orders require day time in force")

    def payload(self) -> dict[str, str]:
        payload = {
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.order_type.value,
            "time_in_force": self.time_in_force.value,
            "qty": _decimal_string(self.quantity),
        }
        if self.limit_price is not None:
            payload["limit_price"] = _decimal_string(self.limit_price)
        return payload


class AlpacaPaperClient:
    def __init__(
        self,
        api_key_id: str,
        api_secret_key: str,
        *,
        transport: TradingTransport | None = None,
    ) -> None:
        if not api_key_id or not api_secret_key:
            raise ValueError("Alpaca paper credentials are required")
        self.transport = transport or PinnedTradingHttpTransport(api_key_id, api_secret_key)

    def account(self, *, observed_at: datetime | None = None) -> AlpacaAccount:
        raw = _object(self.transport.get_json("/v2/account"), "account")
        return AlpacaAccount(
            account_id=_text(raw, "id"),
            status=_text(raw, "status"),
            equity=_number(raw, "equity"),
            last_equity=_number(raw, "last_equity"),
            cash=_number(raw, "cash"),
            buying_power=_number(raw, "buying_power"),
            trading_blocked=_boolean(raw, "trading_blocked"),
            account_blocked=_boolean(raw, "account_blocked"),
            trade_suspended_by_user=_boolean(raw, "trade_suspended_by_user"),
            observed_at=require_aware(observed_at or utc_now(), "observed_at"),
        )

    def positions(self) -> tuple[AlpacaPosition, ...]:
        values = _array(self.transport.get_json("/v2/positions"), "positions")
        return tuple(self._position(_object(item, "position")) for item in values)

    def orders(self, *, status: str = "all", limit: int = 500) -> tuple[AlpacaOrder, ...]:
        if status not in {"open", "closed", "all"}:
            raise ValueError("order status must be open, closed, or all")
        if not 1 <= limit <= 500:
            raise ValueError("order limit must be between 1 and 500")
        values = _array(
            self.transport.get_json(
                "/v2/orders",
                query={"status": status, "limit": limit, "direction": "desc", "nested": True},
            ),
            "orders",
        )
        return tuple(self._order(_object(item, "order")) for item in values)

    def order_by_client_id(self, client_order_id: str) -> AlpacaOrder | None:
        try:
            raw = self.transport.get_json(
                "/v2/orders:by_client_order_id",
                query={"client_order_id": client_order_id},
            )
        except AlpacaPaperError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise
        return self._order(_object(raw, "order"))

    def submit_order(self, request: PaperOrderRequest) -> AlpacaOrder:
        return self._order(
            _object(self.transport.post_json("/v2/orders", request.payload()), "order")
        )

    def cancel_open_orders(self) -> tuple[dict[str, Any], ...]:
        response = self.transport.delete_json("/v2/orders")
        return tuple(_object(item, "cancel result") for item in _array(response, "cancel results"))

    def portfolio_snapshot(
        self,
        instruments_by_symbol: Mapping[str, Instrument],
        *,
        observed_at: datetime | None = None,
    ) -> PortfolioSnapshot:
        account = self.account(observed_at=observed_at)
        positions: list[Position] = []
        for remote in self.positions():
            instrument = instruments_by_symbol.get(remote.symbol)
            if instrument is None:
                raise AlpacaPaperError(
                    f"paper position {remote.symbol} is missing from the instrument master"
                )
            sign = -1.0 if remote.side == "short" else 1.0
            positions.append(
                Position(
                    instrument.instrument_id,
                    instrument.venue,
                    instrument.asset_class,
                    sign * abs(remote.market_value),
                )
            )
        return PortfolioSnapshot(
            account.observed_at,
            account.equity,
            max(0.0, account.buying_power),
            tuple(positions),
        )

    @staticmethod
    def _position(raw: Mapping[str, Any]) -> AlpacaPosition:
        return AlpacaPosition(
            symbol=_text(raw, "symbol"),
            asset_class=_asset_class(_text(raw, "asset_class")),
            quantity=_number(raw, "qty"),
            side=_text(raw, "side"),
            market_value=_number(raw, "market_value"),
            average_entry_price=_number(raw, "avg_entry_price"),
            current_price=_number(raw, "current_price"),
            unrealized_pl=_number(raw, "unrealized_pl"),
        )

    @staticmethod
    def _order(raw: Mapping[str, Any]) -> AlpacaOrder:
        submitted_at = parse_datetime(_text(raw, "submitted_at"))
        updated_value = raw.get("updated_at") or raw.get("filled_at") or raw.get("canceled_at")
        updated_at = parse_datetime(str(updated_value)) if updated_value else submitted_at
        return AlpacaOrder(
            order_id=_text(raw, "id"),
            client_order_id=_text(raw, "client_order_id"),
            symbol=_text(raw, "symbol"),
            asset_class=_asset_class(_text(raw, "asset_class")),
            side=OrderSide(_text(raw, "side")),
            order_type=_text(raw, "type"),
            time_in_force=_text(raw, "time_in_force"),
            status=_text(raw, "status"),
            quantity=_number(raw, "qty"),
            filled_quantity=_number(raw, "filled_qty"),
            limit_price=_optional_number(raw.get("limit_price")),
            average_fill_price=_optional_number(raw.get("filled_avg_price")),
            submitted_at=submitted_at,
            updated_at=updated_at,
        )


class AlpacaPaperAdapter:
    environment = ExecutionEnvironment.PAPER

    def __init__(
        self,
        client: AlpacaPaperClient,
        instrument_resolver: Callable[[str], Instrument],
        *,
        trading_enabled: bool = False,
    ) -> None:
        self.client = client
        self.instrument_resolver = instrument_resolver
        self.trading_enabled = trading_enabled

    def submit(self, approval: ApprovedOrderIntent, *, now: datetime) -> ExecutionReceipt:
        now = require_aware(now, "now")
        if not self.trading_enabled:
            raise PermissionError("paper order submission is not explicitly enabled")
        intent = approval.intent
        if intent.environment is not ExecutionEnvironment.PAPER:
            raise PermissionError("AlpacaPaperAdapter accepts paper intents only")
        instrument = self.instrument_resolver(intent.instrument_id)
        if instrument.venue != "alpaca":
            raise PermissionError("AlpacaPaperAdapter accepts Alpaca instruments only")
        if intent.quantity is None:
            raise ValueError("paper orders require an explicit quantity")
        order_type = _select_order_type(intent.allowed_order_types)
        limit_price = None
        if order_type is OrderType.LIMIT:
            limit_price = intent.max_price if intent.side is OrderSide.BUY else intent.min_price
            if limit_price is None:
                raise ValueError("limit intent is missing its executable price bound")
        client_order_id = _client_order_id(intent.intent_id)
        existing = self.client.order_by_client_id(client_order_id)
        order = existing or self.client.submit_order(
            PaperOrderRequest(
                client_order_id,
                instrument.symbol,
                instrument.asset_class,
                intent.side,
                order_type,
                intent.time_in_force,
                intent.quantity,
                limit_price,
            )
        )
        return ExecutionReceipt(
            intent.intent_id,
            self.environment,
            order.status,
            now,
            order.order_id,
            order.client_order_id,
            order.filled_quantity,
            order.average_fill_price,
        )


def _select_order_type(allowed: tuple[OrderType, ...]) -> OrderType:
    if OrderType.LIMIT in allowed:
        return OrderType.LIMIT
    if OrderType.MARKET in allowed:
        return OrderType.MARKET
    raise ValueError("intent does not allow a supported Alpaca order type")


def _client_order_id(intent_id: str) -> str:
    digest = hashlib.sha256(intent_id.encode("utf-8")).hexdigest()[:32]
    return f"tb-{digest}"


def _asset_class(value: str) -> AssetClass:
    mapping = {
        "us_equity": AssetClass.EQUITY,
        "us_option": AssetClass.OPTION,
        "crypto": AssetClass.CRYPTO,
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise AlpacaPaperError(f"unsupported Alpaca asset class: {value}") from exc


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AlpacaPaperError(f"{name} must be an object")
    return value


def _array(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise AlpacaPaperError(f"{name} must be a list")
    return value


def _text(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise AlpacaPaperError(f"{key} must be a non-empty string")
    return item


def _boolean(value: Mapping[str, Any], key: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise AlpacaPaperError(f"{key} must be a boolean")
    return item


def _number(value: Mapping[str, Any], key: str) -> float:
    item = _optional_number(value.get(key))
    if item is None:
        raise AlpacaPaperError(f"{key} must be numeric")
    return item


def _optional_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _decimal_string(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".")


def _validate_json(value: object, *, depth: int = 0, budget: list[int] | None = None) -> None:
    if budget is None:
        budget = [100_000]
    budget[0] -= 1
    if budget[0] < 0 or depth > 30:
        raise AlpacaPaperError("paper API JSON exceeded safety limits")
    if value is None or isinstance(value, (int, float, bool)):
        return
    if isinstance(value, str):
        if len(value) > 1_000_000:
            raise AlpacaPaperError("paper API JSON string exceeded safety limit")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json(item, depth=depth + 1, budget=budget)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise AlpacaPaperError("paper API JSON keys must be strings")
            _validate_json(item, depth=depth + 1, budget=budget)
        return
    raise AlpacaPaperError("paper API returned an unsupported JSON value")
