from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any, Mapping
from urllib.parse import quote

from trading_bot.core.schemas import AssetClass, Instrument, MarketEvent, MarketEventType
from trading_bot.core.serialization import require_aware, utc_now
from trading_bot.data.collectors.common import (
    CollectorPayloadError,
    observed_event_time,
    parse_time,
    require_object,
    stable_event_id,
)
from trading_bot.data.http import ReadOnlyHttpTransport, ReadOnlyTransport
from trading_bot.data.quality import inspect_events
from trading_bot.data.schemas import (
    CollectionBatch,
    DataQualityDiagnostic,
    DiagnosticCode,
    DiagnosticSeverity,
)


OCC_PATTERN = re.compile(r"^([A-Z0-9.]{1,6})(\d{6})([CP])(\d{8})$")


class AlpacaOptionsCollector:
    venue = "alpaca"

    def __init__(
        self,
        api_key_id: str,
        api_secret_key: str,
        transport: ReadOnlyTransport | None = None,
    ) -> None:
        if not api_key_id or not api_secret_key:
            raise ValueError("Alpaca market-data credentials are required")
        self.transport = transport or ReadOnlyHttpTransport(
            "https://data.alpaca.markets/v1beta1/options",
            "data.alpaca.markets",
            headers={
                "APCA-API-KEY-ID": api_key_id,
                "APCA-API-SECRET-KEY": api_secret_key,
            },
        )

    def collect_chain(
        self,
        underlying_symbol: str,
        *,
        collected_at: datetime | None = None,
        feed: str = "indicative",
        limit: int = 100,
        page_token: str | None = None,
        contract_type: str | None = None,
        expiration_date: str | None = None,
        expiration_date_gte: str | None = None,
        expiration_date_lte: str | None = None,
        strike_price_gte: float | None = None,
        strike_price_lte: float | None = None,
        updated_since: datetime | None = None,
    ) -> CollectionBatch:
        override = require_aware(collected_at, "collected_at") if collected_at else None
        if not re.fullmatch(r"[A-Z0-9.]{1,12}", underlying_symbol):
            raise ValueError("invalid underlying symbol")
        if feed not in {"opra", "indicative"}:
            raise ValueError("feed must be opra or indicative")
        if contract_type not in {None, "call", "put"}:
            raise ValueError("type must be call or put")
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        for field_name, value in (
            ("expiration_date", expiration_date),
            ("expiration_date_gte", expiration_date_gte),
            ("expiration_date_lte", expiration_date_lte),
        ):
            if value is None:
                continue
            try:
                parsed = date.fromisoformat(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field_name} must be an ISO date") from exc
            if parsed.isoformat() != value:
                raise ValueError(f"{field_name} must be an ISO date")
        for field_name, value in (
            ("strike_price_gte", strike_price_gte),
            ("strike_price_lte", strike_price_lte),
        ):
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{field_name} must be a positive finite number")
        if (
            strike_price_gte is not None
            and strike_price_lte is not None
            and strike_price_lte < strike_price_gte
        ):
            raise ValueError("strike price bounds are reversed")
        if updated_since is not None:
            updated_since = require_aware(updated_since, "updated_since")
        response = self.transport.get_json(
            f"/snapshots/{quote(underlying_symbol, safe='')}",
            query={
                "feed": feed,
                "limit": limit,
                "page_token": page_token,
                "type": contract_type,
                "expiration_date": expiration_date,
                "expiration_date_gte": expiration_date_gte,
                "expiration_date_lte": expiration_date_lte,
                "strike_price_gte": strike_price_gte,
                "strike_price_lte": strike_price_lte,
                "updated_since": updated_since.isoformat() if updated_since else None,
            },
        )
        collected_at = override or utc_now()
        snapshots = require_object(response.get("snapshots"), "snapshots")
        instruments: list[Instrument] = []
        events: list[MarketEvent] = []
        diagnostics: list[DataQualityDiagnostic] = []
        if feed == "indicative":
            diagnostics.append(
                DataQualityDiagnostic(
                    DiagnosticCode.INDICATIVE_FEED,
                    DiagnosticSeverity.WARNING,
                    "Alpaca indicative options trades are delayed and quotes are modified",
                )
            )
        for symbol, raw in snapshots.items():
            if not isinstance(symbol, str):
                raise CollectorPayloadError("snapshot symbol must be a string")
            snapshot = require_object(raw, f"snapshots.{symbol}")
            instrument = _option_instrument(symbol, underlying_symbol)
            instruments.append(instrument)
            quote_event = self._quote_event(
                instrument, snapshot, collected_at, feed, diagnostics
            )
            if quote_event:
                events.append(quote_event)
            trade_event = self._trade_event(
                instrument, snapshot, collected_at, feed, diagnostics
            )
            if trade_event:
                events.append(trade_event)
        diagnostics.extend(inspect_events(events))
        next_page_token = response.get("next_page_token")
        if next_page_token is not None and not isinstance(next_page_token, str):
            raise CollectorPayloadError("next_page_token must be a string")
        return CollectionBatch(
            self.venue,
            tuple(instruments),
            tuple(events),
            tuple(diagnostics),
            next_page_token or None,
            {
                "feed": feed,
                "underlying_symbol": underlying_symbol,
                "expiration_date_gte": expiration_date_gte,
                "expiration_date_lte": expiration_date_lte,
                "strike_price_gte": strike_price_gte,
                "strike_price_lte": strike_price_lte,
                "updated_since": updated_since.isoformat() if updated_since else None,
            },
        )

    def _quote_event(
        self,
        instrument: Instrument,
        snapshot: Mapping[str, Any],
        collected_at: datetime,
        feed: str,
        diagnostics: list[DataQualityDiagnostic],
    ) -> MarketEvent | None:
        raw = snapshot.get("latestQuote")
        if raw is None:
            return None
        quote_data = require_object(raw, "latestQuote")
        source_time = parse_time(quote_data.get("t"), "latestQuote.t")
        event_time = observed_event_time(
            source_time,
            collected_at,
            instrument_id=instrument.instrument_id,
            diagnostics=diagnostics,
        )
        payload = {
            "bid_price": quote_data.get("bp"),
            "bid_size": quote_data.get("bs"),
            "ask_price": quote_data.get("ap"),
            "ask_size": quote_data.get("as"),
            "feed": feed,
            "greeks": snapshot.get("greeks"),
            "implied_volatility": snapshot.get("impliedVolatility"),
            "raw": dict(quote_data),
        }
        return MarketEvent(
            stable_event_id(
                "alpaca:option-quote",
                {"symbol": instrument.symbol, "source_time": source_time, "payload": payload},
            ),
            MarketEventType.QUOTE,
            self.venue,
            instrument.instrument_id,
            event_time,
            collected_at,
            f"alpaca-options-{feed}-snapshot-v1beta1",
            payload,
            ingested_at=collected_at,
        )

    def _trade_event(
        self,
        instrument: Instrument,
        snapshot: Mapping[str, Any],
        collected_at: datetime,
        feed: str,
        diagnostics: list[DataQualityDiagnostic],
    ) -> MarketEvent | None:
        raw = snapshot.get("latestTrade")
        if raw is None:
            return None
        trade = require_object(raw, "latestTrade")
        source_time = parse_time(trade.get("t"), "latestTrade.t")
        event_time = observed_event_time(
            source_time,
            collected_at,
            instrument_id=instrument.instrument_id,
            diagnostics=diagnostics,
        )
        payload = {
            "price": trade.get("p"),
            "size": trade.get("s"),
            "exchange": trade.get("x"),
            "conditions": trade.get("c"),
            "feed": feed,
            "raw": dict(trade),
        }
        return MarketEvent(
            stable_event_id(
                "alpaca:option-trade",
                {"symbol": instrument.symbol, "source_time": source_time, "payload": payload},
            ),
            MarketEventType.TRADE,
            self.venue,
            instrument.instrument_id,
            event_time,
            collected_at,
            f"alpaca-options-{feed}-snapshot-v1beta1",
            payload,
            ingested_at=collected_at,
        )


def _option_instrument(symbol: str, requested_underlying: str) -> Instrument:
    match = OCC_PATTERN.fullmatch(symbol)
    if not match:
        raise CollectorPayloadError(f"invalid OCC option symbol: {symbol}")
    root, expiration, side, strike_code = match.groups()
    if root != requested_underlying:
        raise CollectorPayloadError(
            f"option root {root} did not match requested underlying {requested_underlying}"
        )
    expiration_date = f"20{expiration[:2]}-{expiration[2:4]}-{expiration[4:]}"
    return Instrument(
        instrument_id=f"alpaca:option:{symbol}",
        venue="alpaca",
        symbol=symbol,
        asset_class=AssetClass.OPTION,
        quote_currency="USD",
        multiplier=100.0,
        settlement="physical",
        metadata={
            "underlying_symbol": root,
            "expiration_date": expiration_date,
            "option_type": "call" if side == "C" else "put",
            "strike_price": int(strike_code) / 1000,
        },
    )
