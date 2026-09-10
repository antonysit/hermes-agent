"""Standalone verification of APIServerAdapter._persist_context_usage.

Local patch (Helm, see task.md Finding #4): the turn's context occupancy must
land on the ``sessions`` row in state.db, because the SSE ``run.completed``
usage event is lost whenever the client disconnects mid-turn (leaving the task
window / phone backgrounded).  The PM API then reads the columns to drive a
remote context gauge.

These tests run the helper's real UPDATE against an in-memory SQLite database,
so a typo in the statement or the column names fails here rather than in
production.
"""

import sqlite3
import types

import pytest

from gateway.platforms.api_server import APIServerAdapter


class _FakeDB:
    """Minimal SessionDB stand-in: applies the write fn to a real sqlite conn."""

    def __init__(self, conn):
        self._conn = conn
        self.writes = 0

    def _execute_write(self, fn, patience_s=None):
        self.writes += 1
        return fn(self._conn)


def _adapter(db):
    """An object carrying the real method, with only _ensure_session_db faked."""
    fake = types.SimpleNamespace()
    fake._ensure_session_db = lambda: db
    return fake


def _make_db():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, context_tokens INTEGER DEFAULT 0, "
        "context_window INTEGER DEFAULT 0)"
    )
    conn.execute("INSERT INTO sessions (id) VALUES ('sess-1')")
    conn.commit()
    return conn, _FakeDB(conn)


def _row(conn):
    return conn.execute(
        "SELECT context_tokens, context_window FROM sessions WHERE id = 'sess-1'"
    ).fetchone()


def test_writes_both_columns():
    conn, db = _make_db()
    APIServerAdapter._persist_context_usage(
        _adapter(db), "sess-1", {"context_tokens": 12233, "context_window": 131072}
    )
    assert _row(conn) == (12233, 131072)
    assert db.writes == 1


def test_uses_effective_rotated_session_id():
    """Compression can rotate the session id mid-turn — the caller passes the
    effective one; the write must land on it, not the original."""
    conn, db = _make_db()
    conn.execute("INSERT INTO sessions (id) VALUES ('sess-rotated')")
    conn.commit()
    APIServerAdapter._persist_context_usage(
        _adapter(db), "sess-rotated", {"context_tokens": 500, "context_window": 128000}
    )
    assert conn.execute(
        "SELECT context_tokens FROM sessions WHERE id = 'sess-rotated'"
    ).fetchone()[0] == 500
    assert _row(conn) == (0, 0)  # original untouched


def test_unknown_sentinel_does_not_overwrite():
    """0/0 means 'unknown' — it must never clobber a previously recorded value."""
    conn, db = _make_db()
    conn.execute(
        "UPDATE sessions SET context_tokens = 900, context_window = 1000 WHERE id = 'sess-1'"
    )
    conn.commit()
    APIServerAdapter._persist_context_usage(
        _adapter(db), "sess-1", {"context_tokens": 0, "context_window": 0}
    )
    assert _row(conn) == (900, 1000)
    assert db.writes == 0


@pytest.mark.parametrize(
    "usage",
    [
        None,
        "not-a-dict",
        {},
        {"context_tokens": "abc", "context_window": "def"},
        {"context_tokens": None, "context_window": None},
    ],
)
def test_bad_usage_is_swallowed(usage):
    conn, db = _make_db()
    APIServerAdapter._persist_context_usage(_adapter(db), "sess-1", usage)
    assert _row(conn) == (0, 0)


def test_missing_session_id_is_swallowed():
    conn, db = _make_db()
    APIServerAdapter._persist_context_usage(
        _adapter(db), None, {"context_tokens": 10, "context_window": 20}
    )
    assert db.writes == 0


def test_db_unavailable_is_swallowed():
    APIServerAdapter._persist_context_usage(
        _adapter(None), "sess-1", {"context_tokens": 10, "context_window": 20}
    )


def test_write_failure_is_swallowed():
    """A raising db (locked, missing column on an old store) must not surface."""

    class Boom:
        def _execute_write(self, fn, patience_s=None):
            raise sqlite3.OperationalError("database is locked")

    APIServerAdapter._persist_context_usage(
        _adapter(Boom()), "sess-1", {"context_tokens": 10, "context_window": 20}
    )
