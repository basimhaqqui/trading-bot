from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
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


class CoinbaseCollector:
    venue = "coinbase"

    CANDLE_GRANULARITIES = {
        "ONE_MINUTE": 60,
        "FIVE_MINUTE": 300,
        "FIFTEEN_MINUTE": 900,
        "THIRTY_MINUTE": 1800,
        "ONE_HOUR": 3600,
        "TWO_HOUR": 7200,
        "FOUR_HOUR": 14400,
        "SIX_HOUR": 21600,
        "ONE_DAY": 86400,
    }

    def __init__(self, transport: ReadOnlyTransport | None = None) -> None:
        self.transport = transport or ReadOnlyHttpTransport(
            "https://api.coinbase.com/api/v3/brokerage",
            "api.coinbase.com",
        )

    def collect_products(
        self,
        *,
        collected_at: datetime | None = None,
        product_type: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> CollectionBatch:
        override = require_aware(collected_at, "collected_at") if collected_at else None
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        response = self.transport.get_json(
            "/market/products",
            query={"product_type": product_type, "limit": limit, "cursor": cursor},
        )
        collected_at = override or utc_now()
        products = require_list(response.get("products"), "products")
        instruments: list[Instrument] = []
        events: list[MarketEvent] = []
        for raw in products:
            product = require_object(raw, "product")
            instrument = self._instrument(product)
            instruments.append(instrument)
            events.append(
                self._observation_event(
                    instrument,
                    MarketEventType.CONTRACT_RULE,
                    collected_at,
                    {"product": dict(product)},
                )
            )
            best_bid = product.get("best_bid_price")
            best_ask = product.get("best_ask_price")
            if best_bid not in (None, "") and best_ask not in (None, ""):
                events.append(
                    self._observation_event(
                        instrument,
                        MarketEventType.BOOK_SNAPSHOT,
                        collected_at,
                        {
                            "bid_price": best_bid,
                            "ask_price": best_ask,
                            "source_snapshot": "product_list_top_of_book",
                            "raw_product": dict(product),
                        },
                    )
                )
            funding = product.get("funding_rate")
            details = product.get("future_product_details")
            perpetual: Mapping[str, Any] = {}
            if isinstance(details, dict) and isinstance(details.get("perpetual_details"), dict):
                perpetual = details["perpetual_details"]
            if funding in (None, ""):
                if isinstance(details, dict):
                    funding = details.get("funding_rate")
            if funding in (None, ""):
                funding = perpetual.get("funding_rate")
            if funding not in (None, ""):
                payload = {
                    "funding_rate": funding,
                    "funding_time": (
                        product.get("funding_time")
                        or (details.get("funding_time") if isinstance(details, dict) else None)
                        or perpetual.get("funding_time")
                    ),
                    "product_id": instrument.symbol,
                }
                events.append(
                    self._observation_event(
                        instrument,
                        MarketEventType.FUNDING,
                        collected_at,
                        payload,
                    )
                )
            open_interest = product.get("open_interest")
            if open_interest in (None, ""):
                if isinstance(details, dict):
                    open_interest = details.get("open_interest") or perpetual.get("open_interest")
            if open_interest not in (None, ""):
                events.append(
                    self._observation_event(
                        instrument,
                        MarketEventType.OPEN_INTEREST,
                        collected_at,
                        {"open_interest": open_interest, "product_id": instrument.symbol},
                    )
                )
        next_cursor = response.get("cursor")
        pagination = response.get("pagination")
        if next_cursor is None and isinstance(pagination, dict) and pagination.get("has_next", True):
            next_cursor = pagination.get("next_cursor")
        if next_cursor is not None and not isinstance(next_cursor, str):
            raise CollectorPayloadError("cursor must be a string")
        return CollectionBatch(
            self.venue,
            tuple(instruments),
            tuple(events),
            inspect_events(events),
            next_cursor or None,
            {"public_endpoint": True},
        )

    def collect_product_book(
        self,
        product_id: str,
        *,
        collected_at: datetime | None = None,
        limit: int = 100,
    ) -> CollectionBatch:
        override = require_aware(collected_at, "collected_at") if collected_at else None
        if not product_id or "/" in product_id:
            raise ValueError("invalid Coinbase product_id")
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        response = self.transport.get_json(
            "/market/product_book", query={"product_id": product_id, "limit": limit}
        )
        product_response = self.transport.get_json(f"/market/products/{product_id}")
        collected_at = override or utc_now()
        instrument = self._instrument(product_response)
        if instrument.symbol != product_id:
            raise CollectorPayloadError("product detail did not match the requested product_id")
        pricebook = require_object(response.get("pricebook"), "pricebook")
        response_product = require_string(
            pricebook.get("product_id", product_id), "pricebook.product_id"
        )
        if response_product != product_id:
            raise CollectorPayloadError("pricebook product_id did not match the request")
        bids = require_list(pricebook.get("bids"), "pricebook.bids")
        asks = require_list(pricebook.get("asks"), "pricebook.asks")
        diagnostics: list[DataQualityDiagnostic] = []
        source_time = (
            parse_time(pricebook.get("time"), "pricebook.time")
            if pricebook.get("time")
            else collected_at
        )
        event_time = observed_event_time(
            source_time,
            collected_at,
            instrument_id=instrument.instrument_id,
            diagnostics=diagnostics,
        )
        payload: dict[str, Any] = {
            "bids": bids,
            "asks": asks,
            "bid_price": _level_price(bids[0]) if bids else None,
            "ask_price": _level_price(asks[0]) if asks else None,
            "raw": dict(pricebook),
        }
        event = MarketEvent(
            stable_event_id(
                "coinbase:book",
                {"product_id": product_id, "source_time": source_time, "pricebook": pricebook},
            ),
            MarketEventType.BOOK_SNAPSHOT,
            self.venue,
            instrument.instrument_id,
            event_time,
            collected_at,
            "coinbase-advanced-public-product-book-v3",
            payload,
            ingested_at=collected_at,
        )
        diagnostics.extend(inspect_events([event]))
        return CollectionBatch(
            self.venue,
            (instrument,),
            (event,),
            tuple(diagnostics),
            metadata={"public_endpoint": True, "limit": limit},
        )

    def collect_candles(
        self,
        product_id: str,
        *,
        collected_at: datetime | None = None,
        granularity: str = "ONE_HOUR",
        limit: int = 30,
    ) -> CollectionBatch:
        override = require_aware(collected_at, "collected_at") if collected_at else None
        if not product_id or "/" in product_id:
            raise ValueError("invalid Coinbase product_id")
        if granularity not in self.CANDLE_GRANULARITIES:
            raise ValueError("unsupported Coinbase candle granularity")
        if not 1 <= limit <= 350:
            raise ValueError("candle limit must be between 1 and 350")
        collected_at = override or utc_now()
        seconds = self.CANDLE_GRANULARITIES[granularity]
        end_epoch = int(collected_at.timestamp()) // seconds * seconds
        start_epoch = end_epoch - limit * seconds
        response = self.transport.get_json(
            f"/market/products/{product_id}/candles",
            query={
                "start": str(start_epoch),
                "end": str(end_epoch),
                "granularity": granularity,
                "limit": limit,
            },
        )
        product_response = self.transport.get_json(f"/market/products/{product_id}")
        instrument = self._instrument(product_response)
        if instrument.symbol != product_id:
            raise CollectorPayloadError("product detail did not match the requested product_id")
        raw_candles = require_list(response.get("candles"), "candles")
        events: list[MarketEvent] = []
        for raw in raw_candles:
            candle = require_object(raw, "candle")
            try:
                bucket_start = int(require_string(candle.get("start"), "candle.start"))
            except ValueError as exc:
                raise CollectorPayloadError("candle.start must be a UNIX timestamp") from exc
            candle_start = datetime.fromtimestamp(bucket_start, timezone.utc)
            candle_end = candle_start + timedelta(seconds=seconds)
            if candle_end > collected_at:
                continue
            payload = {
                "start": candle_start.isoformat(),
                "end": candle_end.isoformat(),
                "open": candle.get("open"),
                "high": candle.get("high"),
                "low": candle.get("low"),
                "close": candle.get("close"),
                "volume": candle.get("volume"),
                "granularity": granularity,
                "granularity_seconds": seconds,
                "raw": dict(candle),
            }
            _validate_candle(payload)
            events.append(
                MarketEvent(
                    event_id=stable_event_id(
                        "coinbase:candle",
                        {
                            "instrument_id": instrument.instrument_id,
                            "candle_start": bucket_start,
                            "payload": payload,
                        },
                    ),
                    event_type=MarketEventType.BAR,
                    venue=self.venue,
                    instrument_id=instrument.instrument_id,
                    event_time=candle_end,
                    available_at=collected_at,
                    source="coinbase-advanced-public-candles-v3",
                    payload=payload,
                    ingested_at=collected_at,
                )
            )
        events.sort(key=lambda item: (item.event_time, item.event_id))
        return CollectionBatch(
            self.venue,
            (instrument,),
            tuple(events),
            metadata={
                "public_endpoint": True,
                "granularity": granularity,
                "requested_limit": limit,
            },
        )

    def _instrument(self, product: Mapping[str, Any]) -> Instrument:
        product_id = require_string(product.get("product_id"), "product.product_id")
        product_type = str(product.get("product_type", "SPOT")).upper()
        details = product.get("future_product_details")
        expiry_type = ""
        if isinstance(details, dict):
            expiry_type = str(details.get("contract_expiry_type", "")).upper()
        display_name = str(
            product.get("display_name")
            or (details.get("contract_display_name") if isinstance(details, dict) else "")
        ).upper()
        has_funding_mechanism = bool(
            isinstance(details, dict)
            and (details.get("funding_interval") or details.get("funding_rate"))
        )
        if product_type == "FUTURE" and (
            expiry_type == "PERPETUAL"
            or has_funding_mechanism
            or display_name.endswith(" PERP")
        ):
            asset_class = AssetClass.PERPETUAL
        elif product_type == "FUTURE":
            asset_class = AssetClass.FUTURE
        else:
            asset_class = AssetClass.CRYPTO
        quote_currency = product.get("quote_currency_id") or _quote_currency(product_id)
        return Instrument(
            instrument_id=f"coinbase:product:{product_id}",
            venue=self.venue,
            symbol=product_id,
            asset_class=asset_class,
            quote_currency=require_string(quote_currency, "product.quote_currency_id"),
            settlement="cash" if asset_class in {AssetClass.FUTURE, AssetClass.PERPETUAL} else None,
            metadata={},
        )

    def _observation_event(
        self,
        instrument: Instrument,
        event_type: MarketEventType,
        collected_at: datetime,
        payload: Mapping[str, Any],
    ) -> MarketEvent:
        return MarketEvent(
            event_id=stable_event_id(
                f"coinbase:{event_type.value}",
                {
                    "instrument_id": instrument.instrument_id,
                    "collected_at": collected_at,
                    "payload": payload,
                },
            ),
            event_type=event_type,
            venue=self.venue,
            instrument_id=instrument.instrument_id,
            event_time=collected_at,
            available_at=collected_at,
            source="coinbase-advanced-public-products-v3",
            payload=payload,
            ingested_at=collected_at,
        )


def _quote_currency(product_id: str) -> str:
    for separator in ("-", "/"):
        if separator in product_id:
            return product_id.rsplit(separator, 1)[1]
    raise CollectorPayloadError("could not infer quote currency from product_id")


def _level_price(level: object) -> object:
    if isinstance(level, dict):
        return level.get("price")
    if isinstance(level, list) and level:
        return level[0]
    raise CollectorPayloadError("pricebook level must contain a price")


def _validate_candle(payload: Mapping[str, Any]) -> None:
    values: dict[str, float] = {}
    for name in ("open", "high", "low", "close", "volume"):
        try:
            value = float(payload.get(name))
        except (TypeError, ValueError) as exc:
            raise CollectorPayloadError(f"candle.{name} must be numeric") from exc
        if not math.isfinite(value) or value < 0 or (name != "volume" and value == 0):
            raise CollectorPayloadError(f"candle.{name} is invalid")
        values[name] = value
    if values["low"] > min(values["open"], values["close"]):
        raise CollectorPayloadError("candle low exceeds open or close")
    if values["high"] < max(values["open"], values["close"]):
        raise CollectorPayloadError("candle high is below open or close")
