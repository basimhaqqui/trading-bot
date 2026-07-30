from __future__ import annotations

import json
from dataclasses import dataclass
from time import sleep
from typing import Any, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


JsonObject = Mapping[str, Any]
JsonArray = list[Any]


class ReadOnlyTransport(Protocol):
    def get_json(
        self, path: str, *, query: Mapping[str, str | int | float | bool | None] | None = None
    ) -> JsonObject:
        ...

    def get_json_array(
        self, path: str, *, query: Mapping[str, str | int | float | bool | None] | None = None
    ) -> JsonArray:
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
    max_attempts: int = 3
    retry_backoff_seconds: float = 0.5
    max_retry_after_seconds: float = 5.0

    _RETRYABLE_HTTP_STATUS = frozenset({429, 502, 503, 504})

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
        if (
            self.timeout_seconds <= 0
            or self.max_response_bytes < 1
            or not 1 <= self.max_attempts <= 5
            or not 0 <= self.retry_backoff_seconds <= 10
            or not 0 <= self.max_retry_after_seconds <= 60
        ):
            raise ValueError("transport limits are invalid")
        for name, value in (self.headers or {}).items():
            if name.lower() not in self._ALLOWED_HEADERS:
                raise ValueError(f"header is not allowed for read-only transport: {name}")
            if "\r" in value or "\n" in value:
                raise ValueError("header values cannot contain line breaks")

    def get_json(
        self, path: str, *, query: Mapping[str, str | int | float | bool | None] | None = None
    ) -> JsonObject:
        payload = self._get_payload(path, query=query)
        if not isinstance(payload, dict):
            raise ReadOnlyHttpError("top-level JSON response must be an object")
        return payload

    def get_json_array(
        self, path: str, *, query: Mapping[str, str | int | float | bool | None] | None = None
    ) -> JsonArray:
        payload = self._get_payload(path, query=query)
        if not isinstance(payload, list):
            raise ReadOnlyHttpError("top-level JSON response must be an array")
        return payload

    def _get_payload(
        self, path: str, *, query: Mapping[str, str | int | float | bool | None] | None = None
    ) -> JsonObject | JsonArray:
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
        body: bytes | None = None
        for attempt in range(self.max_attempts):
            try:
                with build_opener(_NoRedirects()).open(
                    request, timeout=self.timeout_seconds
                ) as response:
                    body = response.read(self.max_response_bytes + 1)
                break
            except HTTPError as exc:
                if (
                    exc.code not in self._RETRYABLE_HTTP_STATUS
                    or attempt + 1 >= self.max_attempts
                ):
                    raise ReadOnlyHttpError(
                        f"GET failed for {self.allowed_host}: {exc}"
                    ) from exc
                retry_after = (
                    exc.headers.get("Retry-After") if exc.headers is not None else None
                )
                sleep(self._retry_delay(attempt, retry_after))
            except (URLError, TimeoutError) as exc:
                if attempt + 1 >= self.max_attempts:
                    raise ReadOnlyHttpError(
                        f"GET failed for {self.allowed_host}: {exc}"
                    ) from exc
                sleep(self._retry_delay(attempt))
        if body is None:
            raise ReadOnlyHttpError(f"GET failed for {self.allowed_host}")
        if len(body) > self.max_response_bytes:
            raise ReadOnlyHttpError("JSON response exceeded configured size limit")
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReadOnlyHttpError("response was not valid JSON") from exc
        _validate_json_shape(payload)
        return payload

    def _retry_delay(self, attempt: int, retry_after: str | None = None) -> float:
        if retry_after is not None:
            try:
                requested = float(retry_after)
            except (TypeError, ValueError):
                requested = -1
            if requested >= 0:
                return min(requested, self.max_retry_after_seconds)
        return min(
            self.retry_backoff_seconds * (2**attempt),
            self.max_retry_after_seconds,
        )


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


@dataclass(frozen=True)
class ReadOnlyJsonRpcTransport:
    """A bounded JSON-RPC reader restricted to explicitly allowlisted methods.

    JSON-RPC uses POST even for read methods.  Keeping the method allowlist on
    this transport prevents a collector from accidentally reaching signing,
    transaction, or other state-changing RPC methods.
    """

    base_url: str
    allowed_host: str
    allowed_methods: frozenset[str]
    timeout_seconds: float = 10.0
    max_response_bytes: int = 5_000_000
    max_attempts: int = 3
    retry_backoff_seconds: float = 0.5
    max_retry_after_seconds: float = 5.0
    allow_endpoint_path: bool = False

    _RETRYABLE_HTTP_STATUS = frozenset({429, 502, 503, 504})

    def __post_init__(self) -> None:
        parts = urlsplit(self.base_url)
        endpoint_path_allowed = self.allow_endpoint_path
        if (
            parts.scheme != "https"
            or parts.hostname != self.allowed_host
            or parts.port not in (None, 443)
            or parts.username is not None
            or parts.password is not None
            or parts.fragment
            or (not endpoint_path_allowed and parts.path not in ("", "/"))
            or (not endpoint_path_allowed and parts.query)
        ):
            raise ValueError("base_url must be an HTTPS JSON-RPC origin on the allowed host")
        if not isinstance(endpoint_path_allowed, bool):
            raise ValueError("allow_endpoint_path must be boolean")
        if endpoint_path_allowed and (
            len(parts.path) > 2048 or len(parts.query) > 4096
        ):
            raise ValueError("JSON-RPC endpoint path or query exceeds the safety limit")
        if not self.allowed_methods or any(
            method
            not in {
                "getAccountInfo",
                "getMultipleAccounts",
                "getSignaturesForAddress",
                "getTokenLargestAccounts",
                "getTokenSupply",
            }
            for method in self.allowed_methods
        ):
            raise ValueError("JSON-RPC methods must be an explicit read-only allowlist")
        if (
            self.timeout_seconds <= 0
            or self.max_response_bytes < 1
            or not 1 <= self.max_attempts <= 5
            or not 0 <= self.retry_backoff_seconds <= 10
            or not 0 <= self.max_retry_after_seconds <= 60
        ):
            raise ValueError("transport limits are invalid")

    def call(self, method: str, params: list[object]) -> JsonObject:
        if method not in self.allowed_methods:
            raise ValueError(f"JSON-RPC method is not allowed by this read-only transport: {method}")
        if not isinstance(params, list):
            raise ValueError("JSON-RPC params must be a list")
        _validate_json_shape(params)
        request_payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        body = json.dumps(request_payload, separators=(",", ":")).encode("utf-8")
        if len(body) > 100_000:
            raise ValueError("JSON-RPC request exceeds the read-only size limit")
        request = Request(
            self.base_url,
            data=body,
            headers={
                "Accept": "application/json",
                "Cache-Control": "no-cache",
                "Content-Type": "application/json",
                "User-Agent": "trading-bot-observer/0.3",
            },
            method="POST",
        )
        response_body: bytes | None = None
        for attempt in range(self.max_attempts):
            try:
                with build_opener(_NoRedirects()).open(
                    request, timeout=self.timeout_seconds
                ) as response:
                    response_body = response.read(self.max_response_bytes + 1)
                break
            except HTTPError as exc:
                if exc.code not in self._RETRYABLE_HTTP_STATUS or attempt + 1 >= self.max_attempts:
                    raise ReadOnlyHttpError(
                        f"read-only JSON-RPC call failed for {self.allowed_host}: HTTP {exc.code}"
                    ) from exc
                retry_after = exc.headers.get("Retry-After") if exc.headers is not None else None
                sleep(self._retry_delay(attempt, retry_after))
            except (URLError, TimeoutError) as exc:
                if attempt + 1 >= self.max_attempts:
                    raise ReadOnlyHttpError(
                        f"read-only JSON-RPC call failed for {self.allowed_host}: "
                        f"{type(exc).__name__}"
                    ) from exc
                sleep(self._retry_delay(attempt))
        if response_body is None:
            raise ReadOnlyHttpError(f"read-only JSON-RPC call failed for {self.allowed_host}")
        if len(response_body) > self.max_response_bytes:
            raise ReadOnlyHttpError("JSON-RPC response exceeded configured size limit")
        try:
            payload = json.loads(response_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReadOnlyHttpError("JSON-RPC response was not valid JSON") from exc
        _validate_json_shape(payload)
        if not isinstance(payload, dict):
            raise ReadOnlyHttpError("JSON-RPC response must be an object")
        return payload

    def _retry_delay(self, attempt: int, retry_after: str | None = None) -> float:
        if retry_after is not None:
            try:
                requested = float(retry_after)
            except (TypeError, ValueError):
                requested = -1
            if requested >= 0:
                return min(requested, self.max_retry_after_seconds)
        return min(self.retry_backoff_seconds * (2**attempt), self.max_retry_after_seconds)
