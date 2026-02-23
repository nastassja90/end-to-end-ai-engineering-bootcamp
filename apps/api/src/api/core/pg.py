from typing import Any, Optional
from psycopg2 import connect
from api.core.config import config
from api.utils.utils import parse_pg_connection_string
from psycopg2.extras import RealDictCursor


class __PostgresClient:
    """
    A private PostgreSQL client for managing a singleton database connection.

    Attributes:
        __conn (Optional[Any]): The underlying database connection object.

    Methods:
        get():
            Establishes and returns a PostgreSQL cursor if not already connected.
            Sets autocommit mode to True.
            Returns the active cursor object.

        close():
            Closes the active database connection and resets the connection attribute.
    """

    def __init__(self):
        self.__conn: Optional[Any] = None
        self.__db: Optional[str] = None

    def get(self, db: str, autocommit: bool = True) -> RealDictCursor:
        if not self.__conn or self.__db != db:
            pg = parse_pg_connection_string(config.POSTGRES_CONNECTION_STRING)
            self.__conn = connect(
                host=pg["host"],
                port=pg["port"],
                database=db,
                user=pg["user"],
                password=pg["password"],
            )
            self.__conn.autocommit = autocommit
            self.__db = db
        return self.__conn.cursor(cursor_factory=RealDictCursor)

    def commit(self):
        if self.__conn and not self.__conn.autocommit:
            self.__conn.commit()

    def rollback(self):
        if self.__conn and not self.__conn.autocommit:
            self.__conn.rollback()

    def close(self):
        if self.__conn:
            self.__conn.close()
            self.__conn = None
            self.__db = None


postgres_client = __PostgresClient()
"""Singleton instance of the Postgres client. Use `postgres_client.get()` to get the connection and `postgres_client.close()` to close it when the application shuts down."""
