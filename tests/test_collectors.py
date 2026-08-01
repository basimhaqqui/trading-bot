import unittest
import base64
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from trading_bot.core.schemas import AssetClass, MarketEventType
from trading_bot.data.collectors.alpaca import AlpacaOptionsCollector
from trading_bot.data.collectors.alpaca_stocks import AlpacaStockCollector
from trading_bot.data.collectors.coinbase import CoinbaseCollector
from trading_bot.data.collectors.dexscreener import DexscreenerCollector
from trading_bot.data.collectors.common import CollectorPayloadError
from trading_bot.data.collectors.kalshi import KalshiCollector
from trading_bot.data.collectors.solana import SolanaMintAuthorityCollector
from trading_bot.data.schemas import DiagnosticCode


class FakeTransport:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get_json(self, path, *, query=None):
        self.calls.append((path, query or {}))
        return self.responses[path]

    def get_json_array(self, path, *, query=None):
        self.calls.append((path, query or {}))
        return self.responses[path]


class FakeSolanaTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def call(self, method, params):
        self.calls.append((method, params))
        if method in self.response:
            return self.response[method]
        return self.response


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.collected = datetime(2026, 7, 21, 20, tzinfo=timezone.utc)

    def test_solana_mint_address_validation_rejects_untrusted_profile_values(self):
        self.assertTrue(
            SolanaMintAuthorityCollector.is_valid_mint_address(
                "11111111111111111111111111111111"
            )
        )
        self.assertFalse(SolanaMintAuthorityCollector.is_valid_mint_address("ExampleMint"))
        self.assertFalse(
            SolanaMintAuthorityCollector.is_valid_mint_address(
                "O" * 32
            )
        )
        # Character and text-length checks alone accept these, but neither
        # base58 value decodes to Solana's required 32-byte public-key form.
        self.assertFalse(SolanaMintAuthorityCollector.is_valid_mint_address("z" * 32))
        self.assertFalse(SolanaMintAuthorityCollector.is_valid_mint_address("z" * 44))

    def test_dexscreener_solana_profiles_are_point_in_time_and_safety_blocked(self):
        mint = "11111111111111111111111111111111"
        raw_profile = {
            "url": f"https://dexscreener.com/solana/{mint}",
            "chainId": "solana",
            "tokenAddress": mint,
            "description": "untrusted profile text",
            "links": [{"type": "website", "url": "https://example.invalid"}],
        }
        transport = FakeTransport(
            {
                "/token-profiles/latest/v1": [
                    {"chainId": "solana", "tokenAddress": "ExampleMint"},
                    raw_profile,
                    {
                        "chainId": "ethereum",
                        "tokenAddress": "0xNotASolanaToken",
                    },
                ]
            }
        )

        batch = DexscreenerCollector(transport).collect_token_profiles(
            collected_at=self.collected, limit=25
        )

        self.assertEqual(transport.calls, [("/token-profiles/latest/v1", {})])
        self.assertEqual(len(batch.instruments), 1)
        self.assertEqual(batch.instruments[0].asset_class, AssetClass.MEMECOIN)
        self.assertEqual(batch.instruments[0].symbol, mint)
        event = batch.events[0]
        self.assertIs(event.event_type, MarketEventType.ONCHAIN_STATE)
        self.assertEqual(event.event_time, self.collected)
        self.assertEqual(event.available_at, self.collected)
        self.assertEqual(event.payload["raw_profile"], raw_profile)
        self.assertEqual(event.payload["safety_status"], "blocked_unverified")
        self.assertTrue(all(reason.endswith("_unobserved") for reason in event.payload["safety_reasons"]))
        self.assertFalse(event.payload["wallet_or_transaction_authority"])
        self.assertEqual(batch.metadata["invalid_solana_addresses_skipped"], 1)

    def test_dexscreener_pool_observations_are_bounded_and_remain_blocked(self):
        mint = "11111111111111111111111111111111"
        raw_profile = {
            "chainId": "solana",
            "tokenAddress": mint,
        }
        smaller_pool = {
            "chainId": "solana",
            "pairAddress": "SmallPool",
            "baseToken": {"address": mint},
            "quoteToken": {"address": "USDC"},
            "liquidity": {"usd": 5},
        }
        selected_pool = {
            "chainId": "solana",
            "pairAddress": "LargePool",
            "baseToken": {"address": mint},
            "quoteToken": {"address": "USDC"},
            "liquidity": {"usd": 100_000},
            "pairCreatedAt": 1_700_000_000_000,
            "priceUsd": "0.12",
        }
        transport = FakeTransport(
            {
                "/token-profiles/latest/v1": [raw_profile],
                f"/tokens/v1/solana/{mint}": [smaller_pool, selected_pool],
            }
        )

        batch = DexscreenerCollector(transport).collect_token_profiles(
            collected_at=self.collected, limit=25, include_pool_observations=True
        )

        self.assertEqual(
            transport.calls,
            [
                ("/token-profiles/latest/v1", {}),
                (f"/tokens/v1/solana/{mint}", {}),
            ],
        )
        self.assertEqual(len(batch.instruments), 1)
        self.assertEqual(len(batch.events), 2)
        pool_event = batch.events[1]
        self.assertEqual(pool_event.instrument_id, batch.instruments[0].instrument_id)
        self.assertEqual(pool_event.event_time, self.collected)
        self.assertEqual(pool_event.available_at, self.collected)
        self.assertEqual(pool_event.payload["pair_address"], "LargePool")
        self.assertEqual(pool_event.payload["raw_pair"], selected_pool)
        self.assertEqual(pool_event.payload["safety_status"], "blocked_unverified")
        self.assertFalse(pool_event.payload["wallet_or_transaction_authority"])
        self.assertFalse(pool_event.payload["forecast_created"])
        self.assertFalse(pool_event.payload["shadow_intent_created"])
        self.assertEqual(batch.metadata["pool_observations_seen"], 1)

    def test_dexscreener_duplicate_profiles_do_not_amplify_discovery_or_pool_reads(self):
        mint = "11111111111111111111111111111111"
        first_profile = {
            "chainId": "solana",
            "tokenAddress": mint,
            "description": "first point-in-time profile",
        }
        duplicate_profile = {
            "chainId": "solana",
            "tokenAddress": mint,
            "description": "later duplicate profile",
        }
        pool = {
            "chainId": "solana",
            "pairAddress": "OnlyPool",
            "baseToken": {"address": mint},
            "quoteToken": {"address": "USDC"},
            "liquidity": {"usd": 100_000},
        }
        transport = FakeTransport(
            {
                "/token-profiles/latest/v1": [first_profile, duplicate_profile],
                f"/tokens/v1/solana/{mint}": [pool],
            }
        )

        batch = DexscreenerCollector(transport).collect_token_profiles(
            collected_at=self.collected, limit=25, include_pool_observations=True
        )

        self.assertEqual(len(batch.instruments), 1)
        self.assertEqual(len(batch.events), 2)
        self.assertEqual(batch.events[0].payload["raw_profile"], first_profile)
        self.assertEqual(batch.metadata["solana_profiles_seen"], 1)
        self.assertEqual(batch.metadata["duplicate_profiles_skipped"], 1)
        self.assertEqual(
            transport.calls,
            [
                ("/token-profiles/latest/v1", {}),
                (f"/tokens/v1/solana/{mint}", {}),
            ],
        )

    def test_dexscreener_skips_malformed_discovery_and_pool_records(self):
        mint = "11111111111111111111111111111111"
        profile = {"chainId": "solana", "tokenAddress": mint}
        pool = {
            "chainId": "solana",
            "pairAddress": "ValidPool",
            "baseToken": {"address": mint},
            "quoteToken": {"address": "USDC"},
            "liquidity": {"usd": 50_000},
        }
        transport = FakeTransport(
            {
                "/token-profiles/latest/v1": [
                    "not-an-object",
                    {"chainId": "solana"},
                    {"chainId": 123, "tokenAddress": mint},
                    profile,
                ],
                f"/tokens/v1/solana/{mint}": ["not-an-object", pool],
            }
        )

        batch = DexscreenerCollector(transport).collect_token_profiles(
            collected_at=self.collected, include_pool_observations=True
        )

        self.assertEqual(len(batch.instruments), 1)
        self.assertEqual(len(batch.events), 2)
        self.assertEqual(batch.metadata["malformed_profiles_skipped"], 3)
        self.assertEqual(
            batch.events[1].payload["malformed_pool_records_skipped"], 1
        )
        self.assertEqual(batch.events[1].payload["pair_address"], "ValidPool")

    def test_solana_finalized_authority_observation_stays_safety_blocked(self):
        address = "11111111111111111111111111111111"
        mint = bytearray(82)
        mint[0:4] = (0).to_bytes(4, "little")
        mint[46:50] = (1).to_bytes(4, "little")
        transport = FakeSolanaTransport(
            {
                "result": {
                    "context": {"slot": 123},
                    "value": [
                        {
                            "owner": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
                            "data": [base64.b64encode(mint).decode("ascii"), "base64"],
                        }
                    ],
                }
            }
        )

        batch = SolanaMintAuthorityCollector(transport).collect_mint_authorities(
            (address,), collected_at=self.collected
        )

        self.assertEqual(
            transport.calls,
            [("getMultipleAccounts", [[address], {"commitment": "finalized", "encoding": "base64"}])],
        )
        event = batch.events[0]
        self.assertEqual(event.venue, "solana")
        self.assertTrue(event.payload["onchain_authorities_observed"])
        self.assertFalse(event.payload["mint_authority_active"])
        self.assertTrue(event.payload["freeze_authority_active"])
        self.assertTrue(event.payload["transfer_behavior_observed"])
        self.assertFalse(event.payload["token_2022_extensions_observed"])
        self.assertEqual(event.payload["safety_status"], "blocked_unverified")
        self.assertIn("freeze_authority_active", event.payload["safety_reasons"])
        self.assertFalse(event.payload["wallet_or_transaction_authority"])

    def test_solana_read_only_provider_endpoint_is_host_pinned(self):
        collector = SolanaMintAuthorityCollector(
            environment={
                "SOLANA_READ_ONLY_RPC_URL": (
                    "https://mainnet.provider.example/v1/tenant?api-key=secret"
                )
            }
        )
        transport = collector.transport
        self.assertEqual(transport.allowed_host, "mainnet.provider.example")
        self.assertEqual(
            transport.allowed_methods,
            frozenset(
                {
                    "getMultipleAccounts",
                    "getSignaturesForAddress",
                    "getTokenLargestAccounts",
                    "getTokenSupply",
                }
            ),
        )
        with self.assertRaisesRegex(ValueError, "HTTPS endpoint"):
            SolanaMintAuthorityCollector(
                environment={"SOLANA_READ_ONLY_RPC_URL": "http://provider.example"}
            )

    def test_solana_holder_activity_is_aggregate_finalized_and_never_a_safety_pass(self):
        mint = "11111111111111111111111111111111"
        account_one = "11111111111111111111111111111112"
        account_two = "11111111111111111111111111111113"
        transport = FakeSolanaTransport(
            {
                "getTokenLargestAccounts": {
                    "result": {
                        "context": {"slot": 456},
                        "value": [
                            {"address": account_one, "amount": "70"},
                            {"address": account_two, "amount": "20"},
                        ],
                    }
                },
                "getSignaturesForAddress": {
                    "result": [
                        {"confirmationStatus": "finalized", "err": None},
                        {"confirmationStatus": "finalized", "err": {"InstructionError": [0, "x"]}},
                    ]
                },
            }
        )

        event = SolanaMintAuthorityCollector(transport).collect_holder_activity(
            (mint,), collected_at=self.collected
        ).events[0]

        self.assertEqual(
            transport.calls,
            [
                ("getTokenLargestAccounts", [mint, {"commitment": "finalized"}]),
                (
                    "getSignaturesForAddress",
                    [account_one, {"commitment": "finalized", "limit": 10}],
                ),
                (
                    "getSignaturesForAddress",
                    [account_two, {"commitment": "finalized", "limit": 10}],
                ),
            ],
        )
        self.assertTrue(event.payload["holder_activity_observed"])
        self.assertEqual(event.payload["sampled_token_account_count"], 2)
        self.assertEqual(event.payload["sampled_successful_finalized_reference_count"], 2)
        self.assertFalse(event.payload["transfer_behavior_observed"])
        self.assertFalse(event.payload["round_trip_simulation_observed"])
        self.assertEqual(event.payload["safety_status"], "blocked_unverified")
        self.assertFalse(event.payload["wallet_or_transaction_authority"])
        self.assertFalse(event.payload["shadow_intent_created"])

    def test_solana_holder_activity_fails_closed_on_unparseable_signature_response(self):
        mint = "11111111111111111111111111111111"
        account = "11111111111111111111111111111112"
        transport = FakeSolanaTransport(
            {
                "getTokenLargestAccounts": {
                    "result": {
                        "context": {"slot": 456},
                        "value": [{"address": account, "amount": "70"}],
                    }
                },
                "getSignaturesForAddress": {
                    "result": [{"confirmationStatus": "confirmed", "err": None}]
                },
            }
        )

        event = SolanaMintAuthorityCollector(transport).collect_holder_activity(
            (mint,), collected_at=self.collected
        ).events[0]

        self.assertFalse(event.payload["holder_activity_observed"])
        self.assertIn("holder_activity_unobserved", event.payload["safety_reasons"])
        self.assertEqual(event.payload["safety_status"], "blocked_unverified")

    def test_solana_token_2022_transfer_controls_are_structurally_observed_and_blocked(self):
        address = "11111111111111111111111111111111"
        mint = bytearray(82)

        def extension(extension_type, value):
            return (
                extension_type.to_bytes(2, "little")
                + len(value).to_bytes(2, "little")
                + value
            )

        mint.extend(extension(12, b"\x01" * 32))
        mint.extend(extension(14, b"\x00" * 32 + b"\x02" * 32))
        mint.extend(extension(26, b"\x03" * 32 + b"\x01"))
        mint.extend(extension(6, b"\x02"))
        mint.extend(extension(9, b""))
        mint.extend(extension(999, b"\x00"))
        transport = FakeSolanaTransport(
            {
                "result": {
                    "context": {"slot": 124},
                    "value": [
                        {
                            "owner": "TokenzQdBNbLqP5VEhdk7u6RRsJbMpbB5R2r7mS1rG",
                            "data": [base64.b64encode(mint).decode("ascii"), "base64"],
                        }
                    ],
                }
            }
        )

        event = SolanaMintAuthorityCollector(transport).collect_mint_authorities(
            (address,), collected_at=self.collected
        ).events[0]

        self.assertEqual(event.source, "solana-rpc-get-multiple-accounts-finalized-v2")
        self.assertTrue(event.payload["onchain_authorities_observed"])
        self.assertTrue(event.payload["transfer_behavior_observed"])
        self.assertTrue(event.payload["token_2022_extensions_observed"])
        self.assertEqual(event.payload["token_2022_extension_types"], (12, 14, 26, 6, 9, 999))
        self.assertEqual(event.payload["unknown_token_2022_extension_types"], (999,))
        self.assertTrue(event.payload["permanent_delegate_active"])
        self.assertTrue(event.payload["transfer_hook_active"])
        self.assertTrue(event.payload["pausable_active"])
        self.assertTrue(event.payload["transfers_currently_paused"])
        self.assertTrue(event.payload["non_transferable_active"])
        self.assertTrue(event.payload["default_account_frozen"])
        self.assertIn("transfer_hook_active", event.payload["safety_reasons"])
        self.assertIn("default_token_accounts_frozen", event.payload["safety_reasons"])
        self.assertEqual(event.payload["safety_status"], "blocked_unverified")
        self.assertFalse(event.payload["wallet_or_transaction_authority"])

    def test_solana_malformed_transfer_control_fails_closed(self):
        address = "11111111111111111111111111111111"
        mint = bytearray(82)
        mint.extend((12).to_bytes(2, "little") + (31).to_bytes(2, "little") + b"\x01" * 31)
        transport = FakeSolanaTransport(
            {
                "result": {
                    "context": {"slot": 125},
                    "value": [
                        {
                            "owner": "TokenzQdBNbLqP5VEhdk7u6RRsJbMpbB5R2r7mS1rG",
                            "data": [base64.b64encode(mint).decode("ascii"), "base64"],
                        }
                    ],
                }
            }
        )

        event = SolanaMintAuthorityCollector(transport).collect_mint_authorities(
            (address,), collected_at=self.collected
        ).events[0]

        self.assertFalse(event.payload["onchain_authorities_observed"])
        self.assertFalse(event.payload["transfer_behavior_observed"])
        self.assertIn("transfer_behavior_unobserved", event.payload["safety_reasons"])
        self.assertEqual(event.payload["safety_status"], "blocked_unverified")

    def test_solana_holder_concentration_is_finalized_and_remains_blocked(self):
        address = "11111111111111111111111111111111"
        transport = FakeSolanaTransport(
            {
                "getTokenLargestAccounts": {
                    "result": {
                        "context": {"slot": 126},
                        "value": [{"amount": "700"}, {"amount": "200"}],
                    }
                },
                "getTokenSupply": {
                    "result": {"context": {"slot": 126}, "value": {"amount": "1000"}}
                },
            }
        )

        batch = SolanaMintAuthorityCollector(transport).collect_holder_concentrations(
            (address,), collected_at=self.collected
        )

        self.assertEqual(
            transport.calls,
            [
                ("getTokenLargestAccounts", [address, {"commitment": "finalized"}]),
                ("getTokenSupply", [address, {"commitment": "finalized"}]),
            ],
        )
        event = batch.events[0]
        self.assertEqual(event.source, "solana-rpc-token-holder-concentration-finalized-v1")
        self.assertTrue(event.payload["holder_concentration_observed"])
        self.assertEqual(event.payload["top_holder_share_bps"], 7000)
        self.assertEqual(event.payload["top_twenty_holder_share_bps"], 9000)
        self.assertEqual(event.payload["safety_status"], "blocked_unverified")
        self.assertIn("round_trip_simulation_unobserved", event.payload["safety_reasons"])
        self.assertFalse(event.payload["wallet_or_transaction_authority"])

    @patch("trading_bot.data.collectors.solana.sleep")
    def test_solana_holder_reads_are_paced_without_widening_the_bound(self, sleep):
        addresses = (
            "11111111111111111111111111111111",
            "11111111111111111111111111111112",
        )
        transport = FakeSolanaTransport(
            {
                "getTokenLargestAccounts": {
                    "result": {
                        "context": {"slot": 126},
                        "value": [{"amount": "700"}],
                    }
                },
                "getTokenSupply": {
                    "result": {"context": {"slot": 126}, "value": {"amount": "1000"}}
                },
            }
        )

        batch = SolanaMintAuthorityCollector(transport).collect_holder_concentrations(
            addresses, collected_at=self.collected
        )

        self.assertEqual(len(batch.events), 2)
        self.assertEqual(len(transport.calls), 4)
        self.assertEqual(
            sleep.call_args_list,
            [
                unittest.mock.call(SolanaMintAuthorityCollector.HOLDER_REQUEST_SPACING_SECONDS),
                unittest.mock.call(SolanaMintAuthorityCollector.HOLDER_REQUEST_SPACING_SECONDS),
                unittest.mock.call(SolanaMintAuthorityCollector.HOLDER_REQUEST_SPACING_SECONDS),
            ],
        )

    def test_solana_holder_concentration_fails_closed_on_slot_mismatch(self):
        address = "11111111111111111111111111111111"
        transport = FakeSolanaTransport(
            {
                "getTokenLargestAccounts": {
                    "result": {"context": {"slot": 126}, "value": [{"amount": "700"}]}
                },
                "getTokenSupply": {
                    "result": {"context": {"slot": 127}, "value": {"amount": "1000"}}
                },
            }
        )

        event = SolanaMintAuthorityCollector(transport).collect_holder_concentrations(
            (address,), collected_at=self.collected
        ).events[0]

        self.assertFalse(event.payload["holder_concentration_observed"])
        self.assertIsNone(event.payload["top_holder_share_bps"])
        self.assertIn("holder_concentration_unobserved", event.payload["safety_reasons"])

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

    def test_kalshi_skips_malformed_market_records_without_partial_evidence(self):
        valid_market = {
            "ticker": "KXVALID-YES",
            "event_ticker": "KXVALID",
            "updated_time": "2026-07-21T19:55:00Z",
            "yes_bid_dollars": "0.54",
            "no_bid_dollars": "0.44",
        }
        transport = FakeTransport(
            {
                "/markets": {
                    "markets": [
                        "not-an-object",
                        {"ticker": "KXBROKEN-YES", "updated_time": "not-a-time"},
                        valid_market,
                    ],
                    "cursor": "next",
                }
            }
        )

        batch = KalshiCollector(transport).collect_markets(
            collected_at=self.collected
        )

        self.assertEqual([item.symbol for item in batch.instruments], ["KXVALID-YES"])
        self.assertEqual(
            [event.instrument_id for event in batch.events],
            ["kalshi:prediction:KXVALID-YES", "kalshi:prediction:KXVALID-YES"],
        )
        self.assertEqual(batch.metadata["malformed_markets_skipped"], 2)
        self.assertEqual(batch.cursor, "next")

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

    def test_alpaca_stock_latest_quote_preserves_executable_book_and_receipt(self):
        transport = FakeTransport(
            {
                "/SPY/quotes/latest": {
                    "symbol": "SPY",
                    "quote": {
                        "t": "2026-07-21T19:59:58.123456789Z",
                        "bp": 630.10,
                        "bs": 4,
                        "bx": "V",
                        "ap": 630.14,
                        "as": 2,
                        "ax": "V",
                        "c": ["R"],
                        "z": "B",
                    },
                }
            }
        )
        batch = AlpacaStockCollector("key", "secret", transport).collect_latest_quote(
            "SPY", collected_at=self.collected, feed="iex"
        )
        self.assertEqual(transport.calls[0][1]["feed"], "iex")
        self.assertEqual(batch.instruments[0].instrument_id, "alpaca:equity:SPY")
        self.assertEqual(len(batch.events), 1)
        event = batch.events[0]
        self.assertIs(event.event_type, MarketEventType.QUOTE)
        self.assertEqual(event.payload["bid_price"], 630.10)
        self.assertEqual(event.payload["bid_size"], 4)
        self.assertEqual(event.payload["ask_price"], 630.14)
        self.assertEqual(event.payload["ask_size"], 2)
        self.assertEqual(event.payload["feed"], "iex")
        self.assertEqual(event.available_at, self.collected)
        self.assertLess(event.event_time, self.collected)

    def test_alpaca_stock_latest_quote_rejects_mismatched_symbol(self):
        transport = FakeTransport(
            {
                "/SPY/quotes/latest": {
                    "symbol": "QQQ",
                    "quote": {"t": "2026-07-21T19:59:58Z", "bp": 1.0, "ap": 1.1},
                }
            }
        )
        with self.assertRaises(CollectorPayloadError):
            AlpacaStockCollector("key", "secret", transport).collect_latest_quote(
                "SPY", collected_at=self.collected
            )

    def test_alpaca_stock_latest_quote_flags_crossed_books(self):
        transport = FakeTransport(
            {
                "/SPY/quotes/latest": {
                    "symbol": "SPY",
                    "quote": {
                        "t": "2026-07-21T19:59:58Z",
                        "bp": 631.0,
                        "bs": 1,
                        "ap": 630.0,
                        "as": 1,
                    },
                }
            }
        )
        batch = AlpacaStockCollector("key", "secret", transport).collect_latest_quote(
            "SPY", collected_at=self.collected
        )
        self.assertTrue(
            any(
                diagnostic.code is DiagnosticCode.CROSSED_BOOK
                for diagnostic in batch.diagnostics
            )
        )

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
