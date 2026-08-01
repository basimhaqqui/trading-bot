from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Mapping

from trading_bot.core.schemas import AssetClass, Instrument, MarketEvent, MarketEventType
from trading_bot.core.serialization import require_aware, utc_now
from trading_bot.data.collectors.common import (
    CollectorPayloadError,
    observed_event_time,
    parse_time,
    require_list,
    require_object,
    require_string,
    stable_event_id,
)
from trading_bot.data.http import ReadOnlyHttpTransport, ReadOnlyTransport
from trading_bot.data.quality import inspect_events
from trading_bot.data.schemas import CollectionBatch, DataQualityDiagnostic


class KalshiCollector:
    venue = "kalshi"

    def __init__(self, transport: ReadOnlyTransport | None = None) -> None:
        self.transport = transport or ReadOnlyHttpTransport(
            "https://external-api.kalshi.com/trade-api/v2",
            "external-api.kalshi.com",
        )

    def collect_markets(
        self,
        *,
        collected_at: datetime | None = None,
        status: str | None = "open",
        limit: int = 100,
        cursor: str | None = None,
        tickers: tuple[str, ...] = (),
        mve_filter: str | None = None,
        min_close_ts: int | None = None,
        max_close_ts: int | None = None,
        active_only: bool = False,
    ) -> CollectionBatch:
        override = require_aware(collected_at, "collected_at") if collected_at else None
        if status not in {None, "unopened", "open", "paused", "closed", "settled"}:
            raise ValueError("unsupported Kalshi status filter")
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        if len(tickers) > 100 or len(set(tickers)) != len(tickers):
            raise ValueError("Kalshi ticker filters must contain at most 100 unique values")
        if any(not re.fullmatch(r"[A-Z0-9._-]{1,200}", ticker) for ticker in tickers):
            raise ValueError("invalid Kalshi ticker filter")
        if mve_filter not in {None, "only", "exclude"}:
            raise ValueError("mve_filter must be only or exclude")
        if (min_close_ts is None) != (max_close_ts is None):
            raise ValueError("both close timestamp filters are required together")
        if min_close_ts is not None and (
            isinstance(min_close_ts, bool)
            or isinstance(max_close_ts, bool)
            or max_close_ts <= min_close_ts
        ):
            raise ValueError("close timestamp filters are invalid")
        if not isinstance(active_only, bool):
            raise ValueError("active_only must be boolean")
        response = self.transport.get_json(
            "/markets",
            query={
                "status": status,
                "limit": limit,
                "cursor": cursor,
                "tickers": ",".join(tickers) if tickers else None,
                "mve_filter": mve_filter,
                "min_close_ts": min_close_ts,
                "max_close_ts": max_close_ts,
            },
        )
        collected_at = override or utc_now()
        markets = require_list(response.get("markets"), "markets")
        instruments: list[Instrument] = []
        events: list[MarketEvent] = []
        diagnostics: list[DataQualityDiagnostic] = []
        malformed_markets_skipped = 0
        for raw in markets:
            # The bounded public page is untrusted. One malformed market must
            # not abort a rapid observation or suppress valid later contracts,
            # but a partially parsed record must never become an instrument or
            # a point-in-time event.
            try:
                market = require_object(raw, "market")
                if active_only and str(market.get("status", "")).lower() != "active":
                    continue
                ticker = require_string(market.get("ticker"), "market.ticker")
                instrument = self._instrument(ticker, market)
                source_time = self._market_time(market, collected_at)
                market_diagnostics: list[DataQualityDiagnostic] = []
                event_time = observed_event_time(
                    source_time,
                    collected_at,
                    instrument_id=instrument.instrument_id,
                    diagnostics=market_diagnostics,
                )
                event_payload = dict(market)
                market_events = [
                    MarketEvent(
                        event_id=stable_event_id(
                            "kalshi:contract-rule",
                            {
                                "ticker": ticker,
                                "source_time": source_time,
                                "collected_at": collected_at,
                                "payload": event_payload,
                            },
                        ),
                        event_type=MarketEventType.CONTRACT_RULE,
                        venue=self.venue,
                        instrument_id=instrument.instrument_id,
                        event_time=event_time,
                        available_at=collected_at,
                        source="kalshi-rest-markets-v2",
                        payload=event_payload,
                        ingested_at=collected_at,
                    )
                ]
                settlement_event = self._settlement_event(
                    instrument, market, collected_at, market_diagnostics
                )
                if settlement_event is not None:
                    market_events.append(settlement_event)
                book_event = self._market_book_event(instrument, market, collected_at)
                if book_event is not None:
                    market_events.append(book_event)
            except (CollectorPayloadError, TypeError, ValueError):
                malformed_markets_skipped += 1
                continue
            instruments.append(instrument)
            events.extend(market_events)
            diagnostics.extend(market_diagnostics)
        diagnostics.extend(inspect_events(events))
        next_cursor = response.get("cursor")
        if next_cursor is not None and not isinstance(next_cursor, str):
            raise CollectorPayloadError("cursor must be a string")
        return CollectionBatch(
            self.venue,
            tuple(instruments),
            tuple(events),
            tuple(diagnostics),
            next_cursor or None,
            {
                "status": status,
                "ticker_filter_count": len(tickers),
                "mve_filter": mve_filter,
                "active_only": active_only,
                "min_close_ts": min_close_ts,
                "max_close_ts": max_close_ts,
                "malformed_markets_skipped": malformed_markets_skipped,
            },
        )

    def collect_orderbook(
        self,
        ticker: str,
        *,
        collected_at: datetime | None = None,
        depth: int = 100,
    ) -> CollectionBatch:
        override = require_aware(collected_at, "collected_at") if collected_at else None
        if not ticker or "/" in ticker:
            raise ValueError("invalid Kalshi ticker")
        if not 1 <= depth <= 100:
            raise ValueError("depth must be between 1 and 100")
        response = self.transport.get_json(
            f"/markets/{ticker}/orderbook", query={"depth": depth}
        )
        collected_at = override or utc_now()
        raw_book = response.get("orderbook_fp", response.get("orderbook"))
        book = require_object(raw_book, "orderbook_fp")
        yes_bids = require_list(
            book.get("yes_dollars", book.get("yes", [])), "orderbook.yes_dollars"
        )
        no_bids = require_list(
            book.get("no_dollars", book.get("no", [])), "orderbook.no_dollars"
        )
        instrument = self._instrument(ticker, {})
        payload = {
            "yes_bids": yes_bids,
            "no_bids": no_bids,
            "raw": dict(book),
            "depth": depth,
        }
        event = MarketEvent(
            event_id=stable_event_id(
                "kalshi:book", {"ticker": ticker, "collected_at": collected_at, "book": book}
            ),
            event_type=MarketEventType.BOOK_SNAPSHOT,
            venue=self.venue,
            instrument_id=instrument.instrument_id,
            event_time=collected_at,
            available_at=collected_at,
            source="kalshi-rest-orderbook-v2",
            payload=payload,
            ingested_at=collected_at,
        )
        return CollectionBatch(
            self.venue,
            (instrument,),
            (event,),
            inspect_events([event]),
            metadata={"depth": depth},
        )

    def collect_trades(
        self,
        *,
        collected_at: datetime | None = None,
        ticker: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CollectionBatch:
        override = require_aware(collected_at, "collected_at") if collected_at else None
        if ticker is not None and (not ticker or "/" in ticker):
            raise ValueError("invalid Kalshi ticker")
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        response = self.transport.get_json(
            "/markets/trades",
            query={"ticker": ticker, "limit": limit, "cursor": cursor},
        )
        collected_at = override or utc_now()
        trades = require_list(response.get("trades"), "trades")
        instruments: dict[str, Instrument] = {}
        events: list[MarketEvent] = []
        diagnostics: list[DataQualityDiagnostic] = []
        for raw in trades:
            trade = require_object(raw, "trade")
            trade_id = require_string(trade.get("trade_id"), "trade.trade_id")
            market_ticker = require_string(trade.get("ticker"), "trade.ticker")
            instrument = self._instrument(market_ticker, {})
            instruments[instrument.instrument_id] = instrument
            source_time = parse_time(trade.get("created_time"), "trade.created_time")
            event_time = observed_event_time(
                source_time,
                collected_at,
                instrument_id=instrument.instrument_id,
                diagnostics=diagnostics,
            )
            events.append(
                MarketEvent(
                    event_id=f"kalshi:trade:{trade_id}",
                    event_type=MarketEventType.TRADE,
                    venue=self.venue,
                    instrument_id=instrument.instrument_id,
                    event_time=event_time,
                    available_at=collected_at,
                    source="kalshi-rest-trades-v2",
                    payload=dict(trade),
                    ingested_at=collected_at,
                )
            )
        diagnostics.extend(inspect_events(events))
        next_cursor = response.get("cursor")
        if next_cursor is not None and not isinstance(next_cursor, str):
            raise CollectorPayloadError("cursor must be a string")
        return CollectionBatch(
            self.venue,
            tuple(instruments.values()),
            tuple(events),
            tuple(diagnostics),
            next_cursor or None,
        )

    def _instrument(self, ticker: str, market: Mapping[str, Any]) -> Instrument:
        return Instrument(
            instrument_id=f"kalshi:prediction:{ticker}",
            venue=self.venue,
            symbol=ticker,
            asset_class=AssetClass.PREDICTION,
            quote_currency="USD",
            settlement="cash",
            metadata={"market_type": "binary"},
        )

    def _settlement_event(
        self,
        instrument: Instrument,
        market: Mapping[str, Any],
        collected_at: datetime,
        diagnostics: list[DataQualityDiagnostic],
    ) -> MarketEvent | None:
        result = str(market.get("result", "")).lower()
        settlement_ts = market.get("settlement_ts")
        # A market can expose a result while it is merely determined (and may
        # still be disputed or amended).  Kalshi's lifecycle documentation
        # identifies ``finalized`` as the terminal, paid-out state, so only a
        # finalized response may become a research outcome label.
        if (
            str(market.get("status", "")).lower() != "finalized"
            or result not in {"yes", "no", "scalar"}
            or not settlement_ts
        ):
            return None
        source_time = parse_time(settlement_ts, "market.settlement_ts")
        event_time = observed_event_time(
            source_time,
            collected_at,
            instrument_id=instrument.instrument_id,
            diagnostics=diagnostics,
        )
        payload = {
            "result": result,
            "event_ticker": market.get("event_ticker"),
            "occurrence_datetime": market.get("occurrence_datetime"),
            "settlement_value_dollars": market.get("settlement_value_dollars"),
            "settlement_ts": settlement_ts,
            "status": market.get("status"),
            "expiration_value": market.get("expiration_value"),
            "raw_market": dict(market),
        }
        return MarketEvent(
            event_id=stable_event_id(
                "kalshi:settlement",
                {
                    "ticker": instrument.symbol,
                    "settlement_ts": settlement_ts,
                    "result": result,
                    "settlement_value_dollars": market.get("settlement_value_dollars"),
                    "collected_at": collected_at,
                },
            ),
            event_type=MarketEventType.SETTLEMENT,
            venue=self.venue,
            instrument_id=instrument.instrument_id,
            event_time=event_time,
            available_at=collected_at,
            source="kalshi-rest-markets-v2",
            payload=payload,
            ingested_at=collected_at,
        )

    def _market_book_event(
        self,
        instrument: Instrument,
        market: Mapping[str, Any],
        collected_at: datetime,
    ) -> MarketEvent | None:
        yes_bid = market.get("yes_bid_dollars")
        no_bid = market.get("no_bid_dollars")
        if yes_bid in (None, "") or no_bid in (None, ""):
            return None
        payload = {
            "yes_bids": [[yes_bid, market.get("yes_bid_size_fp")]],
            "no_bids": [[no_bid, market.get("no_bid_size_fp")]],
            "yes_ask_dollars": market.get("yes_ask_dollars"),
            "no_ask_dollars": market.get("no_ask_dollars"),
            "source_snapshot": "market_list_top_of_book",
            "raw_market": dict(market),
        }
        return MarketEvent(
            event_id=stable_event_id(
                "kalshi:market-book",
                {
                    "ticker": instrument.symbol,
                    "collected_at": collected_at,
                    "yes_bid": yes_bid,
                    "no_bid": no_bid,
                    "yes_ask": market.get("yes_ask_dollars"),
                    "no_ask": market.get("no_ask_dollars"),
                },
            ),
            event_type=MarketEventType.BOOK_SNAPSHOT,
            venue=self.venue,
            instrument_id=instrument.instrument_id,
            event_time=collected_at,
            available_at=collected_at,
            source="kalshi-rest-markets-top-of-book-v2",
            payload=payload,
            ingested_at=collected_at,
        )

    @staticmethod
    def _market_time(market: Mapping[str, Any], fallback: datetime) -> datetime:
        for field in ("updated_time", "created_time"):
            if market.get(field):
                return parse_time(market[field], f"market.{field}")
        return fallback
