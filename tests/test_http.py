import unittest

from trading_bot.data.http import ReadOnlyHttpTransport


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


if __name__ == "__main__":
    unittest.main()
