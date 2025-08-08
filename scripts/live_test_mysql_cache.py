"""
Live test for MySQL-backed HTTP cache using your config.

Steps:
1) Read experiments.remote_cache from config/experts.yaml
2) Create MySQL-backed client via MySQLCacheClientFactory
3) Write a test cache record using the client's internal store
4) Read it back to verify cache hit

Run: python scripts/live_test_mysql_cache.py
"""
import os
import time
import json
from pathlib import Path

import yaml


def load_remote_cfg():
    cfg_path = Path("config/experts.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    rc = (data or {}).get("experiments", {}).get("remote_cache", {})
    host = rc.get("host")
    port = int(rc.get("port", 3306))
    user = rc.get("user")
    password = rc.get("password")
    database = rc.get("database", "avengers_cache")
    ttl = rc.get("ttl")
    return host, port, user, password, database, ttl


def main():
    host, port, user, password, database, ttl = load_remote_cfg()
    if not host or not user:
        raise SystemExit("remote_cache.host and remote_cache.user are required in config/experts.yaml")

    # Import factory directly from file to avoid importing core.cache.__init__
    import importlib.util
    mysql_path = Path("core/cache/mysql_cache.py").resolve()
    spec = importlib.util.spec_from_file_location("core.cache.mysql_cache", str(mysql_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    # prepare pkg namespace
    import types
    if "core" not in globals():
        import sys as _sys
        pkg_core = types.ModuleType("core")
        pkg_core.__path__ = []
        _sys.modules["core"] = pkg_core
        pkg_cache = types.ModuleType("core.cache")
        pkg_cache.__path__ = []
        _sys.modules["core.cache"] = pkg_cache
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    MySQLCacheClientFactory = mod.MySQLCacheClientFactory

    ClientCls = MySQLCacheClientFactory.create(
        host=host, port=port, user=user, password=password, database=database,
        ttl=ttl, key_generator=lambda req, body: ""
    )
    client = ClientCls()

    # Use a unique key and payload
    cache_key = f"live_test:{int(time.time())}"
    status = 200
    headers = [("x-live-test", "1"), ("content-type", "application/json")]
    body = json.dumps({"ok": True, "t": time.time()}).encode()

    # Write
    client._store_cache(cache_key, status, headers, body)

    # Read back
    row = client._fetch_cache(cache_key)
    if not row:
        raise SystemExit("Cache read failed: no row returned")
    r_status, r_headers, r_body = row

    assert int(r_status) == status, "Status mismatch"
    assert isinstance(r_headers, list), "Headers should be a list"
    assert bytes(r_body) == body, "Body mismatch"

    print("Live MySQL cache write/read OK.")
    print(f"Host={host} DB={database} Key={cache_key}")


if __name__ == "__main__":
    main()
