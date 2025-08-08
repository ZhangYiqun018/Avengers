from __future__ import annotations

import json
import time
from typing import Optional, Callable, Any, Tuple

from loguru import logger


class MySQLCacheClientFactory:
    """
    Factory for creating a lightweight HTTP caching client backed by MySQL.

    This returns an httpx.Client subclass that intercepts send(), checks
    a MySQL table for a cached response by a provided key generator, and
    stores responses on cache miss.

    Requirements at runtime: PyMySQL (or mysqlclient compatible import as pymysql).
    Table will be created automatically if missing.
    """

    TABLE_SQL = (
        "CREATE TABLE IF NOT EXISTS http_cache ("
        "  cache_key VARCHAR(128) PRIMARY KEY,"
        "  status INT NOT NULL,"
        "  headers JSON NOT NULL,"
        "  body LONGBLOB NOT NULL,"
        "  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "  ttl_sec INT NULL,"
        "  expires_at TIMESTAMP NULL,"
        "  INDEX (expires_at)"
        ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"
    )

    @staticmethod
    def create(
        host: str,
        port: int,
        user: Optional[str],
        password: Optional[str],
        database: Optional[str],
        ttl: Optional[int],
        key_generator: Callable[[Any, Optional[bytes]], str],
    ):
        try:
            import pymysql  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "MySQL cache requires the 'PyMySQL' package. Please install it (e.g., pip install PyMySQL)."
            ) from e

        import httpx

        logger.info(
            f"MySQLCache: connecting host={host} port={int(port or 3306)} db={database or 'avengers_cache'}"
        )
        # Ensure database exists; create if missing (best-effort)
        dbname = database or "avengers_cache"
        try:
            admin_conn = pymysql.connect(host=host, port=int(port or 3306), user=user, password=password, autocommit=True)
            with admin_conn.cursor() as cur:
                cur.execute(f"CREATE DATABASE IF NOT EXISTS `{dbname}` DEFAULT CHARACTER SET utf8mb4")
            logger.debug("MySQLCache: ensured database exists")
        except Exception as e:
            logger.warning(f"Could not ensure database exists: {e}")
        finally:
            try:
                admin_conn.close()
            except Exception:
                pass

        def _connect():
            return pymysql.connect(
                host=host,
                port=int(port or 3306),
                user=user,
                password=password,
                database=dbname,
                autocommit=True,
                charset="utf8mb4",
            )

        # Use a dedicated connection to ensure the table exists, then close it.
        tmp_conn = _connect()
        try:
            with tmp_conn.cursor() as cur:
                cur.execute(MySQLCacheClientFactory.TABLE_SQL)
            logger.debug("MySQLCache: ensured http_cache table exists")
        finally:
            try:
                tmp_conn.close()
            except Exception:
                pass

        class MySQLCachedClient(httpx.Client):
            def __init__(self, *_args, **_kwargs):
                super().__init__(*_args, **_kwargs)
                self._ttl = int(ttl) if ttl is not None else None
                self._keygen = key_generator
                # 使用线程安全的计数器
                import threading
                self._hits = 0
                self._misses = 0
                self._reconnects = 0
                self._stats_lock = threading.Lock()  # 保护计数器的锁
                # one connection per thread to avoid packet sequence issues
                self._local = threading.local()
                # register exit summary
                import atexit, weakref
                _selfref = weakref.ref(self)
                def _report():
                    s = _selfref()
                    if s is None:
                        return
                    try:
                        with s._stats_lock:  # 读取统计时也要加锁
                            logger.info(
                                f"MySQLCache Summary: hits={s._hits} misses={s._misses} reconnects={s._reconnects}"
                            )
                    except Exception:
                        pass
                atexit.register(_report)

            def _get_conn(self):
                try:
                    c = getattr(self._local, "conn", None)
                    if c is None:
                        c = _connect()
                        self._local.conn = c
                    try:
                        # If ping fails, reconnect and count it
                        c.ping(reconnect=True)
                    except Exception:
                        c = _connect()
                        with self._stats_lock:  # 线程安全的计数器更新
                            self._reconnects += 1
                        self._local.conn = c
                    return c
                except Exception as e:
                    logger.warning(f"MySQLCache: connection error, creating new: {e}")
                    with self._stats_lock:  # 线程安全的计数器更新
                        self._reconnects += 1
                    c = _connect()
                    self._local.conn = c
                    return c

            def _now(self) -> float:
                return time.time()

            def _fetch_cache(self, key: str) -> Optional[Tuple[int, list, bytes]]:
                try:
                    conn = self._get_conn()
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT status, headers, body, expires_at FROM http_cache WHERE cache_key=%s",
                            (key,),
                        )
                        row = cur.fetchone()
                        if not row:
                            logger.info(f"MySQLCache MISS key={key}")
                            return None
                        status, headers_json, body, expires_at = row
                        # Expiry check (if expires_at is not null)
                        if expires_at is not None:
                            cur.execute("SELECT NOW() < %s", (expires_at,))
                            result = cur.fetchone()
                            fresh = bool(result[0]) if result is not None else False
                            if not fresh:
                                logger.info(f"MySQLCache EXPIRED key={key}")
                                return None
                        headers = json.loads(headers_json)
                        logger.info(f"MySQLCache HIT key={key} status={int(status)} size={len(body) if body else 0}")
                        return int(status), headers, body
                except Exception as e:
                    logger.debug(f"MySQL cache fetch failed, bypassing cache: {e}")
                    return None

            def _store_cache(self, key: str, status: int, headers: list, body: bytes):
                try:
                    conn = self._get_conn()
                    ttl_sec = self._ttl
                    expires = None
                    if ttl_sec is not None:
                        # Let MySQL compute expiration timestamp
                        with conn.cursor() as cur:
                            cur.execute(
                                "INSERT INTO http_cache (cache_key, status, headers, body, ttl_sec, expires_at)"
                                " VALUES (%s, %s, %s, %s, %s, NOW() + INTERVAL %s SECOND)"
                                " ON DUPLICATE KEY UPDATE status=VALUES(status), headers=VALUES(headers), body=VALUES(body), ttl_sec=VALUES(ttl_sec), expires_at=VALUES(expires_at)",
                                (key, int(status), json.dumps(headers, ensure_ascii=False), body, ttl_sec, ttl_sec),
                            )
                    else:
                        with conn.cursor() as cur:
                            cur.execute(
                                "INSERT INTO http_cache (cache_key, status, headers, body, ttl_sec, expires_at)"
                                " VALUES (%s, %s, %s, %s, %s, NULL)"
                                " ON DUPLICATE KEY UPDATE status=VALUES(status), headers=VALUES(headers), body=VALUES(body), ttl_sec=VALUES(ttl_sec), expires_at=NULL",
                                (key, int(status), json.dumps(headers, ensure_ascii=False), body, None),
                            )
                    logger.info(
                        f"MySQLCache STORE key={key} status={int(status)} size={len(body) if body else 0} ttl={ttl_sec}"
                    )
                except Exception as e:
                    logger.warning(f"MySQL cache store failed, ignoring: {e}")

            def send(self, request, *, stream=False, auth=None, follow_redirects=None):
                # Compute key from request and body
                try:
                    body_bytes = getattr(request, "content", None)
                    if body_bytes is None:
                        body_bytes = b""
                    if isinstance(body_bytes, str):
                        body_bytes = body_bytes.encode()
                    key = self._keygen(request, body_bytes)
                    # Debug key details
                    try:
                        method = getattr(request, 'method', 'GET')
                        if isinstance(method, (bytes, bytearray)):
                            method_str = method.decode('utf-8', errors='ignore')
                        else:
                            method_str = str(method)
                        url_str = str(getattr(request, 'url', ''))
                        logger.debug(
                            f"MySQLCache KEY method={method_str} url={url_str} key={key} body_len={len(body_bytes)}"
                        )
                    except Exception:
                        pass
                except Exception as e:
                    logger.debug(f"Key generation failed, bypass cache: {e}")
                    key = None

                if key:
                    cached = self._fetch_cache(key)
                    if cached is not None:
                        status, headers_list, body = cached
                        with self._stats_lock:  # 线程安全的计数器更新
                            self._hits += 1
                            current_hits = self._hits
                            current_misses = self._misses
                        method = getattr(request, 'method', 'GET')
                        if isinstance(method, (bytes, bytearray)):
                            method_str = method.decode('utf-8', errors='ignore')
                        else:
                            method_str = str(method)
                        logger.debug(
                            f"MySQLCache SEND hit method={method_str} url={str(getattr(request, 'url', ''))} hits={current_hits} misses={current_misses}"
                        )
                        # 构造完整的响应对象，确保与真实HTTP响应一致  
                        # headers_list从数据库读取时是JSON，需要转换为httpx可识别的格式
                        headers_dict = {k: v for k, v in headers_list} if isinstance(headers_list, list) else headers_list
                        response = httpx.Response(
                            status_code=int(status),
                            headers=headers_dict,
                            stream=httpx.ByteStream(body),
                            request=request,
                        )
                        # 确保响应已读取完成，避免stream相关错误
                        response._content = body
                        return response

                # Miss: forward request then store
                response = super().send(request, stream=False if stream is False else stream, auth=auth, follow_redirects=follow_redirects)
                try:
                    # Ensure body buffered prior to storing
                    content = response.content
                    headers_list = [(k, v) for k, v in response.headers.items()]
                    if key:
                        self._store_cache(key, response.status_code, headers_list, content)
                        with self._stats_lock:  # 线程安全的计数器更新
                            self._misses += 1
                            current_hits = self._hits
                            current_misses = self._misses
                        method = getattr(request, 'method', 'GET')
                        if isinstance(method, (bytes, bytearray)):
                            method_str = method.decode('utf-8', errors='ignore')
                        else:
                            method_str = str(method)
                        logger.debug(
                            f"MySQLCache SEND miss method={method_str} url={str(getattr(request, 'url', ''))} hits={current_hits} misses={current_misses}"
                        )
                except Exception as e:
                    logger.debug(f"Post-send cache store failed: {e}")
                return response

        return MySQLCachedClient
