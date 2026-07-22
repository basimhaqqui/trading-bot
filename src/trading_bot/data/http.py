from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


JsonObject = Mapping[str, Any]


class ReadOnlyTransport(Protocol):
    def get_json(
        self, path: str, *, query: Mapping[str, str | int | float | bool | None] | None = None
    ) -> JsonObject:
        ...


class ReadOnlyHttpError(RuntimeError):
    pass


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


@dataclass(frozen=True)
class ReadOnlyHttpTransport:
    base_url: str
    allowed_host: str
    headers: Mapping[str, str] | None = None
    timeout_seconds: float = 10.0
    max_response_bytes: int = 5_000_000

    _ALLOWED_HEADERS = frozenset(
        {
            "accept",
            "cache-control",
            "user-agent",
            "apca-api-key-id",
            "apca-api-secret-key",
        }
    )

    def __post_init__(self) -> None:
        parts = urlsplit(self.base_url)
        if (
            parts.scheme != "https"
            or parts.hostname != self.allowed_host
            or parts.port not in (None, 443)
            or parts.username is not None
            or parts.password is not None
        ):
            raise ValueError("base_url must be HTTPS and match allowed_host")
        if self.timeout_seconds <= 0 or self.max_response_bytes < 1:
            raise ValueError("timeout and response limit must be positive")
        for name, value in (self.headers or {}).items():
            if name.lower() not in self._ALLOWED_HEADERS:
                raise ValueError(f"header is not allowed for read-only transport: {name}")
            if "\r" in value or "\n" in value:
                raise ValueError("header values cannot contain line breaks")

    def get_json(
        self, path: str, *, query: Mapping[str, str | int | float | bool | None] | None = None
    ) -> JsonObject:
        path_parts = urlsplit(path)
        if not path.startswith("/") or path.startswith("//") or path_parts.scheme or path_parts.netloc:
            raise ValueError("path must be an absolute path on the configured host")
        url = urljoin(self.base_url.rstrip("/") + "/", path.lstrip("/"))
        final_parts = urlsplit(url)
        if (
            final_parts.scheme != "https"
            or final_parts.hostname != self.allowed_host
            or final_parts.port not in (None, 443)
        ):
            raise ValueError("request escaped the configured HTTPS host")
        values = {key: value for key, value in (query or {}).items() if value is not None}
        if values:
            url = f"{url}?{urlencode(values)}"
        request_headers = {
            "Accept": "application/json",
            "Cache-Control": "no-cache",
            "User-Agent": "trading-bot-observer/0.3",
            **(self.headers or {}),
        }
        request = Request(url, headers=request_headers, method="GET")
        try:
            with build_opener(_NoRedirects()).open(request, timeout=self.timeout_seconds) as response:
                body = response.read(self.max_response_bytes + 1)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise ReadOnlyHttpError(f"GET failed for {self.allowed_host}: {exc}") from exc
        if len(body) > self.max_response_bytes:
            raise ReadOnlyHttpError("JSON response exceeded configured size limit")
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReadOnlyHttpError("response was not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ReadOnlyHttpError("top-level JSON response must be an object")
        _validate_json_shape(payload)
        return payload


def _validate_json_shape(
    value: object,
    *,
    depth: int = 0,
    budget: list[int] | None = None,
) -> None:
    if budget is None:
        budget = [100_000]
    budget[0] -= 1
    if budget[0] < 0:
        raise ReadOnlyHttpError("JSON response contains too many values")
    if depth > 30:
        raise ReadOnlyHttpError("JSON response nesting is too deep")
    if isinstance(value, str):
        if len(value) > 1_000_000:
            raise ReadOnlyHttpError("JSON string exceeds configured safety limit")
        return
    if value is None or isinstance(value, (int, float, bool)):
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_shape(item, depth=depth + 1, budget=budget)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ReadOnlyHttpError("JSON object keys must be strings")
            _validate_json_shape(item, depth=depth + 1, budget=budget)
        return
    raise ReadOnlyHttpError(f"unsupported JSON value: {type(value).__name__}")
