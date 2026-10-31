import sqlite3
from contextlib import contextmanager
from pathlib import Path

from app.config import Settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(settings: Settings) -> sqlite3.Connection:
    connection = sqlite3.connect(settings.db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


@contextmanager
def db(settings: Settings):
    connection = connect(settings)
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.media_dir.mkdir(parents=True, exist_ok=True)
    with db(settings) as connection:
        connection.executescript(SCHEMA_PATH.read_text())
