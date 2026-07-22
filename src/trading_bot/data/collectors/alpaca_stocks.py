from __future__ import annotations

import re
from datetime import datetime, timedelta
from urllib.parse import quote

from trading_bot.core.schemas import AssetClass, Instrument, MarketEvent, MarketEventType
from trading_bot.core.serialization import require_aware, utc_now
from trading_bot.data.collectors.common import (
    CollectorPayloadError,
    observed_event_time,
    parse_time,
    require_list,
    require_object,
    stable_event_id,
)
from trading_bot.data.http import ReadOnlyHttpTransport, ReadOnlyTransport
from trading_bot.data.quality import inspect_events
from trading_bot.data.schemas import CollectionBatch, DataQualityDiagnostic


class AlpacaStockCollector:
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
            "https://data.alpaca.markets/v2/stocks",
            "data.alpaca.markets",
            headers={
                "APCA-API-KEY-ID": api_key_id,
                "APCA-API-SECRET-KEY": api_secret_key,
            },
        )

    def collect_daily_bars(
        self,
        symbol: str,
        *,
        collected_at: datetime | None = None,
        feed: str = "iex",
        lookback_days: int = 45,
        limit: int = 1000,
        page_token: str | None = None,
    ) -> CollectionBatch:
        override = require_aware(collected_at, "collected_at") if collected_at else None
        if not re.fullmatch(r"[A-Z0-9.]{1,12}", symbol):
            raise ValueError("invalid stock symbol")
        if feed not in {"iex", "sip", "delayed_sip"}:
            raise ValueError("stock feed must be iex, sip, or delayed_sip")
        if not 2 <= lookback_days <= 3660:
            raise ValueError("lookback_days must be between 2 and 3660")
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        request_time = override or utc_now()
        response = self.transport.get_json(
            f"/{quote(symbol, safe='')}/bars",
            query={
                "feed": feed,
                "timeframe": "1Day",
                "start": (request_time - timedelta(days=lookback_days)).date().isoformat(),
                "end": request_time.date().isoformat(),
                "limit": limit,
                "page_token": page_token,
                "adjustment": "raw",
            },
        )
        receipt_time = override or utc_now()
        response_symbol = response.get("symbol", symbol)
        if response_symbol != symbol:
            raise CollectorPayloadError("bar response symbol did not match the request")
        bars = require_list(response.get("bars"), "bars")
        instrument = Instrument(
            instrument_id=f"alpaca:equity:{symbol}",
            venue=self.venue,
            symbol=symbol,
            asset_class=AssetClass.EQUITY,
            quote_currency="USD",
            metadata={},
        )
        events: list[MarketEvent] = []
        diagnostics: list[DataQualityDiagnostic] = []
        for raw in bars:
            bar = require_object(raw, "bar")
            source_time = parse_time(bar.get("t"), "bar.t")
            event_time = observed_event_time(
                source_time,
                receipt_time,
                instrument_id=instrument.instrument_id,
                diagnostics=diagnostics,
            )
            payload = {
                "open": bar.get("o"),
                "high": bar.get("h"),
                "low": bar.get("l"),
                "close": bar.get("c"),
                "volume": bar.get("v"),
                "trade_count": bar.get("n"),
                "vwap": bar.get("vw"),
                "feed": feed,
                "timeframe": "1Day",
                "adjustment": "raw",
                "raw": dict(bar),
            }
            events.append(
                MarketEvent(
                    event_id=stable_event_id(
                        "alpaca:stock-bar",
                        {
                            "symbol": symbol,
                            "feed": feed,
                            "timeframe": "1Day",
                            "source_time": source_time,
                            "payload": payload,
                        },
                    ),
                    event_type=MarketEventType.BAR,
                    venue=self.venue,
                    instrument_id=instrument.instrument_id,
                    event_time=event_time,
                    available_at=receipt_time,
                    source=f"alpaca-stocks-{feed}-bars-v2",
                    payload=payload,
                    ingested_at=receipt_time,
                )
            )
        diagnostics.extend(
            inspect_events(events, stale_after=timedelta(days=lookback_days + 1))
        )
        next_page_token = response.get("next_page_token")
        if next_page_token is not None and not isinstance(next_page_token, str):
            raise CollectorPayloadError("next_page_token must be a string")
        return CollectionBatch(
            self.venue,
            (instrument,),
            tuple(events),
            tuple(diagnostics),
            next_page_token or None,
            {
                "feed": feed,
                "timeframe": "1Day",
                "adjustment": "raw",
                "lookback_days": lookback_days,
            },
        )
