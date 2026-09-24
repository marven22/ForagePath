from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "app" / "pollinator_forage.sqlite3"
SCHEMA = Path(__file__).with_name("schema.sql")


class ClosingConnection(sqlite3.Connection):
    """Preserve transaction context behavior and close when leaving a with block."""

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def connect(path: Path | str = DEFAULT_DB, *, read_only: bool = False) -> sqlite3.Connection:
    path = Path(path).resolve()
    if read_only:
        connection = sqlite3.connect(
            f"file:{path.as_posix()}?mode=ro", uri=True, factory=ClosingConnection
        )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, factory=ClosingConnection)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    if not read_only:
        connection.execute("PRAGMA journal_mode = WAL")
    return connection


def initialize(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA.read_text(encoding="utf-8"))


@contextmanager
def transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        connection.execute("BEGIN IMMEDIATE")
        yield connection
    except Exception:
        connection.rollback()
        raise
    else:
        connection.commit()
