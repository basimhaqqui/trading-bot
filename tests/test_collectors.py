import unittest
from datetime import datetime, timedelta, timezone

from trading_bot.core.schemas import AssetClass, MarketEventType
from trading_bot.data.collectors.alpaca import AlpacaOptionsCollector
from trading_bot.data.collectors.alpaca_stocks import AlpacaStockCollector
from trading_bot.data.collectors.coinbase import CoinbaseCollector
from trading_bot.data.collectors.common import CollectorPayloadError
from trading_bot.data.collectors.kalshi import KalshiCollector
from trading_bot.data.schemas import DiagnosticCode


class FakeTransport:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get_json(self, path, *, query=None):
        self.calls.append((path, query or {}))
        return self.responses[path]


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.collected = datetime(2026, 7, 21, 20, tzinfo=timezone.utc)

    def test_kalshi_contract_trades_and_book_preserve_availability(self):
        transport = FakeTransport(
            {
                "/markets": {
                    "markets": [
                        {
                            "ticker": "KXTEST-YES",
                            "event_ticker": "KXTEST",
                            "market_type": "binary",
                            "updated_time": "2026-07-21T19:55:00Z",
                            "close_time": "2026-07-22T00:00:00Z",
                            "rules_primary": "Use the published final value.",
                            "yes_bid_dollars": "0.54",
                            "yes_bid_size_fp": "10.00",
                            "no_bid_dollars": "0.44",
                            "no_bid_size_fp": "8.00",
                        }
                    ],
                    "cursor": "next",
                },
                "/markets/trades": {
                    "trades": [
                        {
                            "trade_id": "trade-1",
                            "ticker": "KXTEST-YES",
                            "count_fp": "2.00",
                            "yes_price_dollars": "0.55",
                            "no_price_dollars": "0.45",
                            "created_time": "2026-07-21T19:59:00Z",
                        }
                    ],
                    "cursor": "",
                },
                "/markets/KXTEST-YES/orderbook": {
                    "orderbook_fp": {
                        "yes_dollars": [["0.60", "10.00"]],
                        "no_dollars": [["0.45", "5.00"]],
                    }
                },
            }
        )
        collector = KalshiCollector(transport)
        markets = collector.collect_markets(collected_at=self.collected)
        trades = collector.collect_trades(collected_at=self.collected)
        book = collector.collect_orderbook("KXTEST-YES", collected_at=self.collected)

        self.assertEqual(markets.cursor, "next")
        self.assertEqual(markets.events[0].event_type, MarketEventType.CONTRACT_RULE)
        self.assertIn(
            MarketEventType.BOOK_SNAPSHOT,
            {event.event_type for event in markets.events},
        )
        self.assertEqual(trades.events[0].available_at, self.collected)
        self.assertLess(trades.events[0].event_time, trades.events[0].available_at)
        self.assertEqual(trades.events[0].event_id, "kalshi:trade:trade-1")
        self.assertIn(DiagnosticCode.CROSSED_BOOK, {item.code for item in book.diagnostics})

    def test_kalshi_market_filters_support_binary_universe_and_exact_tickers(self):
        transport = FakeTransport({"/markets": {"markets": [], "cursor": ""}})

        KalshiCollector(transport).collect_markets(
            collected_at=self.collected,
            status=None,
            limit=2,
            tickers=("KXONE-YES", "KXTWO-NO"),
            mve_filter="exclude",
        )

        self.assertEqual(
            transport.calls[0][1],
            {
                "status": None,
                "limit": 2,
                "cursor": None,
                "tickers": "KXONE-YES,KXTWO-NO",
                "mve_filter": "exclude",
                "min_close_ts": None,
                "max_close_ts": None,
            },
        )
        with self.assertRaisesRegex(ValueError, "unique values"):
            KalshiCollector(transport).collect_markets(
                tickers=("KXONE-YES", "KXONE-YES")
            )
        with self.assertRaisesRegex(ValueError, "invalid Kalshi ticker"):
            KalshiCollector(transport).collect_markets(tickers=("KX/BAD",))

    def test_kalshi_close_window_filters_out_nonactive_markets(self):
        transport = FakeTransport(
            {
                "/markets": {
                    "markets": [
                        {
                            "ticker": "KXACTIVE-YES",
                            "status": "active",
                            "updated_time": "2026-07-21T19:55:00Z",
                            "yes_bid_dollars": "0.50",
                            "no_bid_dollars": "0.48",
                        },
                        {
                            "ticker": "KXINITIALIZED-YES",
                            "status": "initialized",
                            "updated_time": "2026-07-21T19:55:00Z",
                            "yes_bid_dollars": "0.50",
                            "no_bid_dollars": "0.48",
                        },
                    ],
                    "cursor": "",
                }
            }
        )

        batch = KalshiCollector(transport).collect_markets(
            collected_at=self.collected,
            status=None,
            min_close_ts=100,
            max_close_ts=200,
            active_only=True,
        )

        self.assertEqual([item.symbol for item in batch.instruments], ["KXACTIVE-YES"])
        self.assertEqual(transport.calls[0][1]["min_close_ts"], 100)
        self.assertEqual(transport.calls[0][1]["max_close_ts"], 200)

    def test_coinbase_maps_spot_and_perpetual_observations(self):
        transport = FakeTransport(
            {
                "/market/products": {
                    "products": [
                        {
                            "product_id": "BTC-USD",
                            "product_type": "SPOT",
                            "base_currency_id": "BTC",
                            "quote_currency_id": "USD",
                            "best_bid_price": "59990.00",
                            "best_ask_price": "60010.00",
                        },
                        {
                            "product_id": "BTC-PERP-USDC",
                            "product_type": "FUTURE",
                            "base_currency_id": "BTC",
                            "quote_currency_id": "USDC",
                            "future_product_details": {
                                "contract_expiry_type": "EXPIRING",
                                "contract_display_name": "BTC PERP",
                                "funding_interval": "3600s",
                                "funding_rate": "0.0001",
                                "funding_time": "2026-07-21T21:00:00Z",
                                "perpetual_details": {
                                    "open_interest": "1234.5",
                                },
                            },
                        },
                    ],
                    "pagination": {
                        "has_next": True,
                        "next_cursor": "coinbase-next",
                    },
                }
            }
        )
        batch = CoinbaseCollector(transport).collect_products(collected_at=self.collected)
        classes = {item.symbol: item.asset_class for item in batch.instruments}
        self.assertEqual(classes["BTC-USD"], AssetClass.CRYPTO)
        self.assertEqual(classes["BTC-PERP-USDC"], AssetClass.PERPETUAL)
        self.assertEqual(batch.cursor, "coinbase-next")
        self.assertEqual(
            {event.event_type for event in batch.events},
            {
                MarketEventType.CONTRACT_RULE,
                MarketEventType.BOOK_SNAPSHOT,
                MarketEventType.FUNDING,
                MarketEventType.OPEN_INTEREST,
            },
        )
        self.assertTrue(all(event.available_at == self.collected for event in batch.events))

    def test_coinbase_rejects_symbol_confusion_in_book_response(self):
        transport = FakeTransport(
            {
                "/market/product_book": {
                    "pricebook": {
                        "product_id": "ETH-USD",
                        "time": "2026-07-21T19:59:59Z",
                        "bids": [],
                        "asks": [],
                    }
                },
                "/market/products/BTC-USD": {
                    "product_id": "BTC-USD",
                    "product_type": "SPOT",
                    "quote_currency_id": "USD",
                },
            }
        )
        with self.assertRaises(CollectorPayloadError):
            CoinbaseCollector(transport).collect_product_book(
                "BTC-USD", collected_at=self.collected
            )

    def test_coinbase_candles_keep_only_completed_bars_and_receipt_availability(self):
        completed_start = int(
            datetime(2026, 7, 21, 19, tzinfo=timezone.utc).timestamp()
        )
        incomplete_start = int(
            datetime(2026, 7, 21, 20, tzinfo=timezone.utc).timestamp()
        )
        transport = FakeTransport(
            {
                "/market/products/BTC-USD/candles": {
                    "candles": [
                        {
                            "start": str(incomplete_start),
                            "low": "100",
                            "high": "110",
                            "open": "102",
                            "close": "108",
                            "volume": "10",
                        },
                        {
                            "start": str(completed_start),
                            "low": "90",
                            "high": "105",
                            "open": "95",
                            "close": "100",
                            "volume": "20",
                        },
                    ]
                },
                "/market/products/BTC-USD": {
                    "product_id": "BTC-USD",
                    "product_type": "SPOT",
                    "quote_currency_id": "USD",
                },
            }
        )
        batch = CoinbaseCollector(transport).collect_candles(
            "BTC-USD",
            collected_at=self.collected,
            granularity="ONE_HOUR",
            limit=30,
        )
        self.assertEqual(len(batch.events), 1)
        candle = batch.events[0]
        self.assertEqual(candle.event_type, MarketEventType.BAR)
        self.assertEqual(candle.event_time, self.collected)
        self.assertEqual(candle.available_at, self.collected)
        self.assertEqual(candle.payload["close"], "100")
        self.assertEqual(
            transport.calls[0][1]["granularity"],
            "ONE_HOUR",
        )

    def test_alpaca_option_chain_maps_occ_identity_and_flags_indicative_feed(self):
        symbol = "AAPL260918C00200000"
        transport = FakeTransport(
            {
                "/snapshots/AAPL": {
                    "snapshots": {
                        symbol: {
                            "latestQuote": {
                                "t": "2026-07-21T19:59:59Z",
                                "bp": 4.9,
                                "bs": 10,
                                "ap": 5.1,
                                "as": 8,
                            },
                            "latestTrade": {
                                "t": "2026-07-21T19:59:58Z",
                                "p": 5.0,
                                "s": 1,
                                "x": "P",
                            },
                            "greeks": {"delta": 0.51},
                            "impliedVolatility": 0.25,
                        }
                    },
                    "next_page_token": "page-2",
                }
            }
        )
        batch = AlpacaOptionsCollector("key", "secret", transport).collect_chain(
            "AAPL",
            collected_at=self.collected,
            feed="indicative",
            expiration_date_gte="2026-07-21",
            expiration_date_lte="2026-08-04",
            strike_price_gte=180.0,
            strike_price_lte=220.0,
            updated_since=self.collected - timedelta(hours=2),
        )
        instrument = batch.instruments[0]
        self.assertEqual(instrument.asset_class, AssetClass.OPTION)
        self.assertEqual(instrument.multiplier, 100)
        self.assertEqual(instrument.metadata["strike_price"], 200)
        self.assertEqual(batch.cursor, "page-2")
        self.assertEqual(len(batch.events), 2)
        self.assertIn(
            DiagnosticCode.INDICATIVE_FEED, {item.code for item in batch.diagnostics}
        )
        self.assertEqual(
            transport.calls[0][1],
            {
                "feed": "indicative",
                "limit": 100,
                "page_token": None,
                "type": None,
                "expiration_date": None,
                "expiration_date_gte": "2026-07-21",
                "expiration_date_lte": "2026-08-04",
                "strike_price_gte": 180.0,
                "strike_price_lte": 220.0,
                "updated_since": "2026-07-21T18:00:00+00:00",
            },
        )

    def test_alpaca_option_chain_rejects_unsafe_filter_bounds(self):
        collector = AlpacaOptionsCollector(
            "key", "secret", FakeTransport({"/snapshots/AAPL": {"snapshots": {}}})
        )
        with self.assertRaisesRegex(ValueError, "ISO date"):
            collector.collect_chain("AAPL", expiration_date_gte="07/21/2026")
        with self.assertRaisesRegex(ValueError, "positive finite"):
            collector.collect_chain("AAPL", strike_price_gte=float("nan"))
        with self.assertRaisesRegex(ValueError, "bounds are reversed"):
            collector.collect_chain(
                "AAPL", strike_price_gte=220.0, strike_price_lte=180.0
            )

    def test_source_clock_ahead_is_visible_and_never_creates_future_information(self):
        transport = FakeTransport(
            {
                "/markets/trades": {
                    "trades": [
                        {
                            "trade_id": "future",
                            "ticker": "KXFUTURE",
                            "created_time": (self.collected + timedelta(seconds=2)).isoformat(),
                        }
                    ],
                    "cursor": "",
                }
            }
        )
        batch = KalshiCollector(transport).collect_trades(collected_at=self.collected)
        self.assertEqual(batch.events[0].event_time, self.collected)
        self.assertIn(
            DiagnosticCode.SOURCE_CLOCK_AHEAD, {item.code for item in batch.diagnostics}
        )

    def test_alpaca_stock_bars_preserve_raw_feed_and_receipt_availability(self):
        transport = FakeTransport(
            {
                "/AAPL/bars": {
                    "symbol": "AAPL",
                    "bars": [
                        {
                            "t": "2026-07-20T04:00:00Z",
                            "o": 210.0,
                            "h": 214.0,
                            "l": 209.0,
                            "c": 213.0,
                            "v": 1000000,
                            "n": 12000,
                            "vw": 212.4,
                        }
                    ],
                    "next_page_token": None,
                }
            }
        )
        batch = AlpacaStockCollector("key", "secret", transport).collect_daily_bars(
            "AAPL", collected_at=self.collected, feed="iex", lookback_days=30
        )
        self.assertEqual(batch.instruments[0].instrument_id, "alpaca:equity:AAPL")
        self.assertEqual(batch.events[0].payload["close"], 213.0)
        self.assertEqual(batch.events[0].payload["feed"], "iex")
        self.assertEqual(batch.events[0].payload["adjustment"], "raw")
        self.assertEqual(batch.events[0].available_at, self.collected)
        self.assertEqual(transport.calls[0][1]["end"], "2026-07-21")

    def test_kalshi_finalized_market_emits_public_settlement_label(self):
        transport = FakeTransport(
            {
                "/markets": {
                    "markets": [
                        {
                            "ticker": "KXSETTLED-YES",
                            "event_ticker": "KXSETTLED",
                            "market_type": "binary",
                            "status": "finalized",
                            "updated_time": "2026-07-21T18:00:00Z",
                            "settlement_ts": "2026-07-21T19:00:00Z",
                            "occurrence_datetime": "2026-07-21T18:45:00Z",
                            "result": "yes",
                            "settlement_value_dollars": "1.0000",
                            "rules_primary": "Use the named source.",
                        }
                    ],
                    "cursor": "",
                }
            }
        )
        batch = KalshiCollector(transport).collect_markets(
            collected_at=self.collected, status="settled"
        )
        settlements = [
            event for event in batch.events if event.event_type is MarketEventType.SETTLEMENT
        ]
        self.assertEqual(len(settlements), 1)
        self.assertEqual(settlements[0].payload["result"], "yes")
        self.assertEqual(settlements[0].payload["event_ticker"], "KXSETTLED")
        self.assertEqual(
            settlements[0].payload["occurrence_datetime"],
            "2026-07-21T18:45:00Z",
        )
        self.assertEqual(
            settlements[0].event_time,
            datetime(2026, 7, 21, 19, tzinfo=timezone.utc),
        )
        self.assertEqual(settlements[0].available_at, self.collected)
        repeated = KalshiCollector(transport).collect_markets(
            collected_at=self.collected + timedelta(minutes=30), status="settled"
        )
        self.assertNotEqual(batch.events[0].event_id, repeated.events[0].event_id)
        repeated_settlement = next(
            event
            for event in repeated.events
            if event.event_type is MarketEventType.SETTLEMENT
        )
        self.assertNotEqual(settlements[0].event_id, repeated_settlement.event_id)


if __name__ == "__main__":
    unittest.main()
