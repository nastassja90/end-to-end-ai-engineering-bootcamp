from urllib.parse import urlparse


def parse_pg_connection_string(connection_string: str) -> dict:
    """
    Parses a PostgreSQL connection string into its components.

    Args:
        connection_string (str): The PostgreSQL connection string to parse.

    Returns:
        dict: A dictionary containing the following keys:
            - "user": The username specified in the connection string.
            - "password": The password specified in the connection string.
            - "host": The hostname of the PostgreSQL server.
            - "port": The port number of the PostgreSQL server.
            - "db": The database name.

    Example:
        >>> parse_pg_connection_string("postgresql://user:pass@localhost:5432/mydb")
        {'user': 'user', 'password': 'pass', 'host': 'localhost', 'port': 5432, 'db': 'mydb'}
    """
    parsed = urlparse(connection_string)
    return {
        "user": parsed.username,
        "password": parsed.password,
        "host": parsed.hostname,
        "port": parsed.port,
        "db": parsed.path.lstrip("/"),
    }
