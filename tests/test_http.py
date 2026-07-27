import unittest
from email.message import Message
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from trading_bot.data.http import (
    ReadOnlyHttpError,
    ReadOnlyHttpTransport,
    ReadOnlyJsonRpcTransport,
)


class ReadOnlyHttpTransportTests(unittest.TestCase):
    def test_transport_requires_pinned_https_host(self):
        with self.assertRaises(ValueError):
            ReadOnlyHttpTransport("http://api.coinbase.com", "api.coinbase.com")
        with self.assertRaises(ValueError):
            ReadOnlyHttpTransport("https://example.com", "api.coinbase.com")
        with self.assertRaises(ValueError):
            ReadOnlyHttpTransport("https://api.coinbase.com:444", "api.coinbase.com")

    def test_transport_rejects_privileged_headers(self):
        with self.assertRaises(ValueError):
            ReadOnlyHttpTransport(
                "https://api.coinbase.com",
                "api.coinbase.com",
                headers={"Authorization": "Bearer trading-credential"},
            )
        with self.assertRaises(ValueError):
            ReadOnlyHttpTransport(
                "https://data.alpaca.markets",
                "data.alpaca.markets",
                headers={"APCA-API-KEY-ID": "safe\r\nX-Evil: injected"},
            )

    def test_request_cannot_escape_host_and_has_no_mutating_api(self):
        transport = ReadOnlyHttpTransport(
            "https://api.coinbase.com/api/v3/brokerage", "api.coinbase.com"
        )
        self.assertFalse(hasattr(transport, "post_json"))
        with self.assertRaises(ValueError):
            transport.get_json("https://evil.example/steal")
        with self.assertRaises(ValueError):
            transport.get_json("//evil.example/steal")

    def test_json_rpc_transport_only_allows_explicit_read_methods(self):
        transport = ReadOnlyJsonRpcTransport(
            "https://api.mainnet-beta.solana.com",
            "api.mainnet-beta.solana.com",
            frozenset({"getMultipleAccounts", "getTokenLargestAccounts", "getTokenSupply"}),
        )
        with self.assertRaisesRegex(ValueError, "not allowed"):
            transport.call("sendTransaction", [])
        with self.assertRaises(ValueError):
            ReadOnlyJsonRpcTransport(
                "https://api.mainnet-beta.solana.com",
                "api.mainnet-beta.solana.com",
                frozenset({"sendTransaction"}),
            )

    def test_json_rpc_provider_endpoint_requires_explicit_opt_in(self):
        endpoint = "https://mainnet.provider.example/v1/key?api-key=secret"
        with self.assertRaises(ValueError):
            ReadOnlyJsonRpcTransport(
                endpoint,
                "mainnet.provider.example",
                frozenset({"getTokenSupply"}),
            )
        transport = ReadOnlyJsonRpcTransport(
            endpoint,
            "mainnet.provider.example",
            frozenset({"getTokenSupply"}),
            allow_endpoint_path=True,
        )
        self.assertEqual(transport.allowed_host, "mainnet.provider.example")

    @patch("trading_bot.data.http.build_opener")
    def test_json_rpc_failure_does_not_echo_provider_endpoint(self, build_opener):
        endpoint = "https://mainnet.provider.example/v1/key?api-key=secret"
        build_opener.return_value.open.side_effect = HTTPError(
            endpoint,
            400,
            "Bad Request",
            Message(),
            None,
        )
        transport = ReadOnlyJsonRpcTransport(
            endpoint,
            "mainnet.provider.example",
            frozenset({"getTokenSupply"}),
            allow_endpoint_path=True,
        )
        with self.assertRaises(ReadOnlyHttpError) as error:
            transport.call("getTokenSupply", ["mint", {"commitment": "finalized"}])
        self.assertIn("mainnet.provider.example", str(error.exception))
        self.assertNotIn("api-key", str(error.exception))
        self.assertNotIn("secret", str(error.exception))

    @patch("trading_bot.data.http.build_opener")
    def test_array_response_remains_bounded_to_read_only_get(self, build_opener):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'[{"ok": true}]'
        build_opener.return_value.open.return_value = response
        transport = ReadOnlyHttpTransport("https://api.example.com", "api.example.com")

        self.assertEqual(transport.get_json_array("/profiles"), [{"ok": True}])
        self.assertEqual(build_opener.return_value.open.call_args.args[0].method, "GET")

    @patch("trading_bot.data.http.sleep")
    @patch("trading_bot.data.http.build_opener")
    def test_transient_rate_limit_retries_bounded_read_only_get(
        self, build_opener, sleep
    ):
        headers = Message()
        headers["Retry-After"] = "2"
        rate_limit = HTTPError(
            "https://api.coinbase.com/test",
            429,
            "Too Many Requests",
            headers,
            None,
        )
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"ok": true}'
        opener = build_opener.return_value
        opener.open.side_effect = (rate_limit, response)
        transport = ReadOnlyHttpTransport(
            "https://api.coinbase.com",
            "api.coinbase.com",
        )

        self.assertEqual(transport.get_json("/test"), {"ok": True})
        self.assertEqual(opener.open.call_count, 2)
        sleep.assert_called_once_with(2.0)
        for call in opener.open.call_args_list:
            self.assertEqual(call.args[0].method, "GET")

    @patch("trading_bot.data.http.sleep")
    @patch("trading_bot.data.http.build_opener")
    def test_non_retryable_http_error_fails_immediately(self, build_opener, sleep):
        opener = build_opener.return_value
        opener.open.side_effect = HTTPError(
            "https://api.coinbase.com/test",
            400,
            "Bad Request",
            Message(),
            None,
        )
        transport = ReadOnlyHttpTransport(
            "https://api.coinbase.com",
            "api.coinbase.com",
        )

        with self.assertRaises(ReadOnlyHttpError):
            transport.get_json("/test")

        self.assertEqual(opener.open.call_count, 1)
        sleep.assert_not_called()

    @patch("trading_bot.data.http.sleep")
    @patch("trading_bot.data.http.build_opener")
    def test_transient_retries_stop_at_configured_attempts(
        self, build_opener, sleep
    ):
        opener = build_opener.return_value
        opener.open.side_effect = HTTPError(
            "https://api.coinbase.com/test",
            503,
            "Service Unavailable",
            None,
            None,
        )
        transport = ReadOnlyHttpTransport(
            "https://api.coinbase.com",
            "api.coinbase.com",
            max_attempts=3,
            retry_backoff_seconds=0.25,
        )

        with self.assertRaises(ReadOnlyHttpError):
            transport.get_json("/test")

        self.assertEqual(opener.open.call_count, 3)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            [0.25, 0.5],
        )


if __name__ == "__main__":
    unittest.main()
