"""DSN or libpq fields; ECS can inject a password without constructing a URL."""
import os


def connect_database(source=False):
    import psycopg2
    prefix = "SOURCE_" if source else ""
    dsn = os.getenv("SOURCE_DATABASE_URL" if source else "PGVECTOR_DATABASE_URL")
    if dsn:
        return psycopg2.connect(dsn, connect_timeout=10)
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
    return psycopg2.connect(**kwargs)
