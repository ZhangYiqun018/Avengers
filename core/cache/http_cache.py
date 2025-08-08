from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Optional

import hishel
import httpx
from hishel._utils import normalized_url


def _param_only_key(request: httpx.Request, body: Optional[bytes] = b"") -> str:
    """Generate a cache key using URL path and selected JSON body fields.

    Keeps behavior aligned with the previous implementation to avoid
    unexpected cache misses/hits across the codebase.
    """
    interested_fields = {"model", "temperature", "top_p", "n", "messages"}

    # 1) Normalize URL path (exclude host to preserve prior behavior)
    # Be defensive about input types from various clients/wrappers
    url = getattr(request, "url", request)
    try:
        if not isinstance(url, httpx.URL):
            url = httpx.URL(str(url))
        full_url = normalized_url(url)
        url_obj = full_url if isinstance(full_url, httpx.URL) else httpx.URL(str(full_url))
    except Exception:
        # Fallback to simple coercion to avoid keygen failure
        url_obj = httpx.URL(str(url))
    encoded_url = url_obj.raw_path or b"/"
    if isinstance(encoded_url, str):
        encoded_url = encoded_url.encode()

    # 2) Extract only interested fields from JSON body
    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError:
        payload = {}
    filtered = {k: payload.get(k) for k in sorted(interested_fields)}
    encoded_body = json.dumps(
        filtered, separators=(",", ":"), sort_keys=True, ensure_ascii=False
    ).encode()

    # 3) Hash
    method = getattr(request, "method", b"GET")
    if isinstance(method, str):
        method_bytes = method.encode()
    elif isinstance(method, bytes):
        method_bytes = method
    else:
        method_bytes = str(method).encode()

    key_parts = [method_bytes, encoded_url, encoded_body]
    # Hashlib compatibility across environments (avoid usedforsecurity where unsupported)
    try:
        hasher = hashlib.blake2b(digest_size=16, usedforsecurity=False)  # type: ignore[call-arg]
    except TypeError:
        try:
            hasher = hashlib.blake2b(digest_size=16)
        except Exception:
            hasher = hashlib.sha256()
    for part in key_parts:
        hasher.update(part)
    return hasher.hexdigest()


@dataclass
class HttpCache:
    """Small wrapper to create a hishel-powered HTTP cache transport/client.

    Supports:
    - local file-based storage (default)
    - remote Redis storage (host/port/user/password)

    This class mirrors the previous caching behavior while centralizing
    configuration in one place and avoiding broader changes.
    """

    mode: str  # "local" | "remote" (redis) | "mysql"
    cache_dir: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    user: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None
    ttl: Optional[float] = None
    key_generator: Callable[[httpx.Request, Optional[bytes]], str] = _param_only_key

    @staticmethod
    def local(cache_dir: str, ttl: Optional[float] = None,
              key_generator: Callable[[httpx.Request, Optional[bytes]], str] = _param_only_key) -> "HttpCache":
        return HttpCache(mode="local", cache_dir=cache_dir, ttl=ttl, key_generator=key_generator)

    @staticmethod
    def remote(host: str, port: int, user: Optional[str], password: Optional[str], ttl: Optional[float] = None,
               key_generator: Callable[[httpx.Request, Optional[bytes]], str] = _param_only_key,
               database: Optional[str] = None) -> "HttpCache":
        # Heuristic: port 3306 implies MySQL-backed cache
        mode = "mysql" if int(port or 0) == 3306 else "remote"
        return HttpCache(mode=mode, host=host, port=port, user=user, password=password, ttl=ttl,
                         database=database, key_generator=key_generator)

    def _build_storage(self):
        if self.mode == "local":
            if not self.cache_dir:
                raise ValueError("cache_dir is required for local storage")
            return hishel.FileStorage(base_path=self.cache_dir)

        if self.mode == "remote":
            # Remote storage via Redis (hishel[redis])
            try:
                import redis  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "Redis backend required for remote cache. Install 'hishel[redis]' or 'redis' package."
                ) from e

            client = redis.Redis(
                host=self.host or "localhost",
                port=int(self.port or 6379),
                username=self.user or None,
                password=self.password or None,
            )
            return hishel.RedisStorage(client=client, ttl=self.ttl)

        raise ValueError(f"Unsupported cache mode: {self.mode}")

    def create_transport(self) -> hishel.CacheTransport:
        storage = self._build_storage()
        base_transport = httpx.HTTPTransport()
        controller = hishel.Controller(
            cacheable_methods=["GET", "POST"],
            cacheable_status_codes=[200],
            allow_stale=True,
            force_cache=True,
            key_generator=self.key_generator,
        )
        return hishel.CacheTransport(storage=storage, transport=base_transport, controller=controller)

    def create_httpx_client(self) -> httpx.Client:
        if self.mode == "mysql":
            from .mysql_cache import MySQLCacheClientFactory
            ClientCls = MySQLCacheClientFactory.create(
                host=self.host or "localhost",
                port=int(self.port or 3306),
                user=self.user,
                password=self.password,
                database=self.database,
                ttl=int(self.ttl) if self.ttl is not None else None,
                key_generator=self.key_generator,
            )
            return ClientCls()
        return httpx.Client(transport=self.create_transport())
