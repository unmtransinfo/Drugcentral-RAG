"""DSN or libpq fields; ECS can inject a password without constructing a URL."""
import os
from threading import BoundedSemaphore, Lock


def _connection_arguments(source=False):
    prefix = "SOURCE_" if source else ""
    dsn = os.getenv("SOURCE_DATABASE_URL" if source else "PGVECTOR_DATABASE_URL")
    if dsn:
        return (dsn,), {"connect_timeout": 10}
    host = os.getenv(prefix + "PGHOST")
    if not host:
        raise ValueError("Configure " + ("SOURCE_DATABASE_URL or SOURCE_PGHOST" if source
                                        else "PGVECTOR_DATABASE_URL or PGHOST"))
    kwargs = {"host": host, "port": os.getenv(prefix + "PGPORT", "5432"),
              "dbname": os.environ[prefix + "PGDATABASE"],
              "user": os.environ[prefix + "PGUSER"],
              "password": os.environ[prefix + "PGPASSWORD"],
              "sslmode": os.getenv(prefix + "PGSSLMODE", "verify-full"),
              "connect_timeout": 10}
    if os.getenv(prefix + "PGSSLROOTCERT"):
        kwargs["sslrootcert"] = os.environ[prefix + "PGSSLROOTCERT"]
    return (), kwargs


def pool_size():
    """PG_POOL_SIZE > 0 enables pooling (API processes); 0/unset keeps one connection per call."""
    return min(100, max(0, int(os.getenv("PG_POOL_SIZE", "0") or 0)))


def connect_database(source=False, pooled=False):
    """A connection used as `with closing(conn), conn:`; pooled ones return to the pool on close()."""
    import psycopg2
    if pooled and not source and pool_size():
        return _borrow()
    args, kwargs = _connection_arguments(source)
    return psycopg2.connect(*args, **kwargs)


_pool = None
_pool_slots = None
_pool_lock = Lock()


class PooledConnection:
    """Borrowed psycopg2 connection: close() rolls back any open transaction and returns it."""

    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        self._conn.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._conn.__exit__(*exc_info)

    def close(self):
        conn, self._conn = self._conn, None
        if conn is None:
            return
        from psycopg2.extensions import TRANSACTION_STATUS_IDLE
        broken = bool(conn.closed)
        if not broken:
            try:
                if conn.get_transaction_status() != TRANSACTION_STATUS_IDLE:
                    conn.rollback()
            except Exception:
                broken = True
        try:
            _pool.putconn(conn, close=broken)
        finally:
            _pool_slots.release()


def _borrow():
    import psycopg2
    from psycopg2.pool import ThreadedConnectionPool
    global _pool, _pool_slots
    with _pool_lock:
        if _pool is None:
            size = pool_size()
            args, kwargs = _connection_arguments()
            # psycopg2 keeps a returned connection only while fewer than minconn are
            # idle, so minconn must equal the size or every return closes the socket.
            _pool = ThreadedConnectionPool(size, size, *args, **kwargs)
            # ThreadedConnectionPool raises when empty; the semaphore makes callers wait instead.
            _pool_slots = BoundedSemaphore(size)
    if not _pool_slots.acquire(timeout=float(os.getenv("PG_POOL_WAIT_SECONDS", "10"))):
        raise psycopg2.OperationalError("Database connection pool exhausted")
    try:
        conn = _pool.getconn()
        try:
            # Detect connections dropped by a database restart before handing them out.
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
            conn.rollback()
        except psycopg2.Error:
            _pool.putconn(conn, close=True)
            conn = _pool.getconn()
    except Exception:
        _pool_slots.release()
        raise
    return PooledConnection(conn)
