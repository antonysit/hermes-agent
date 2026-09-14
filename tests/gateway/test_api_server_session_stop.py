"""Tests for the session-scoped stop endpoint (local Helm patch, 2026-09-14).

POST /api/sessions/{session_id}/stop — interrupt every live run on a session,
regardless of whether the caller knows the run_id (browser reloads lose it,
and runs queued behind the session turn lease have no agent yet).
"""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.api_server import (
    APIServerAdapter,
    PlatformConfig,
    cors_middleware,
    security_headers_middleware,
)


class FlakyAgent:
    """Minimal agent double with an interrupt() recorder."""

    def __init__(self):
        self.interrupt_calls = []

    def interrupt(self, message=None):
        self.interrupt_calls.append(message)

    def run_conversation(self, **kw):
        return {"final_response": "", "messages": [], "api_calls": 0}


def _make_adapter() -> APIServerAdapter:
    config = PlatformConfig(enabled=True, extra={"key": "test-key-0123456789abcdef"})
    return APIServerAdapter(config)


def _make_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post(
        "/api/sessions/{session_id}/stop", adapter._handle_session_stop
    )
    return app


def _register_run(adapter, run_id, session_id, status="running", agent=None):
    """Insert a run into the adapter's registries the way the handlers do."""
    adapter._set_run_status(run_id, status, session_id=session_id)
    if agent is not None:
        adapter._active_run_agents[run_id] = agent


async def _patch_lookup(adapter, monkeypatch):
    async def fake_get(session_id):
        return {"id": session_id}, None

    monkeypatch.setattr(adapter, "_get_existing_session_or_404", fake_get)

    def fake_auth(request):
        return None

    monkeypatch.setattr(adapter, "_check_auth", fake_auth)


@pytest.mark.asyncio
async def test_session_stop_interrupts_live_run(monkeypatch):
    adapter = _make_adapter()
    await _patch_lookup(adapter, monkeypatch)
    agent = FlakyAgent()
    _register_run(adapter, "run_A", "sess-1", agent=agent)

    async with TestClient(TestServer(_make_app(adapter))) as cli:
        resp = await cli.post("/api/sessions/sess-1/stop")
        assert resp.status == 200
        body = await resp.json()
    assert body["count"] == 1
    assert body["stopped"][0]["run_id"] == "run_A"
    assert adapter._run_statuses["run_A"]["status"] == "stopping"
    assert agent.interrupt_calls, "agent.interrupt() must have been called"


@pytest.mark.asyncio
async def test_session_stop_reaches_queued_run_without_agent(monkeypatch):
    """A run waiting on the session turn lease has no agent yet — the endpoint
    must still mark it stopping so the race guard in _run_agent honors it."""
    adapter = _make_adapter()
    await _patch_lookup(adapter, monkeypatch)
    _register_run(adapter, "run_Q", "sess-1", status="queued")

    async with TestClient(TestServer(_make_app(adapter))) as cli:
        resp = await cli.post("/api/sessions/sess-1/stop")
        assert resp.status == 200
        body = await resp.json()
    assert body["count"] == 1
    assert adapter._run_statuses["run_Q"]["status"] == "stopping"
    assert "run_Q" in adapter._stopping_run_ids


@pytest.mark.asyncio
async def test_session_stop_scopes_to_session(monkeypatch):
    adapter = _make_adapter()
    await _patch_lookup(adapter, monkeypatch)
    agent_other = FlakyAgent()
    _register_run(adapter, "run_other", "sess-2", agent=agent_other)
    _register_run(adapter, "run_done", "sess-1", status="completed")

    async with TestClient(TestServer(_make_app(adapter))) as cli:
        resp = await cli.post("/api/sessions/sess-1/stop")
        assert resp.status == 200
        body = await resp.json()
    assert body["count"] == 0
    assert not agent_other.interrupt_calls
    assert adapter._run_statuses["run_other"]["status"] == "running"


@pytest.mark.asyncio
async def test_session_stop_skips_finished_runs(monkeypatch):
    adapter = _make_adapter()
    await _patch_lookup(adapter, monkeypatch)
    for st in ("completed", "failed", "cancelled", "interrupted"):
        _register_run(adapter, f"run_{st}", "sess-1", status=st)

    async with TestClient(TestServer(_make_app(adapter))) as cli:
        resp = await cli.post("/api/sessions/sess-1/stop")
        body = await resp.json()
    assert body["count"] == 0
    for st in ("completed", "failed", "cancelled", "interrupted"):
        assert adapter._run_statuses[f"run_{st}"]["status"] == st


@pytest.mark.asyncio
async def test_session_stop_unknown_session_404(monkeypatch):
    adapter = _make_adapter()

    async def fake_get(session_id):
        return None, web.json_response({"error": {"message": "nope"}}, status=404)

    monkeypatch.setattr(adapter, "_get_existing_session_or_404", fake_get)
    monkeypatch.setattr(adapter, "_check_auth", lambda request: None)

    async with TestClient(TestServer(_make_app(adapter))) as cli:
        resp = await cli.post("/api/sessions/missing/stop")
        assert resp.status == 404


@pytest.mark.asyncio
async def test_run_agent_race_guard_interrupts_pre_marked_run(monkeypatch):
    """The _run_agent guard: if a session-stop marked the run stopping before
    the agent registered, registering must immediately interrupt it."""
    adapter = _make_adapter()
    agent = FlakyAgent()
    adapter._stopping_run_ids.add("run_pre")

    monkeypatch.setattr(adapter, "_create_agent", lambda **kw: agent)

    def safe_run(**kw):
        return {"final_response": "", "messages": [], "api_calls": 0}

    monkeypatch.setattr(agent, "run_conversation", safe_run)

    result, _usage = await adapter._run_agent(
        user_message="hi",
        conversation_history=[],
        session_id="sess-1",
        active_run_id="run_pre",
    )
    assert agent.interrupt_calls, "guard must interrupt the pre-marked run"
    # The finally block pops the registration when the turn returns — what
    # matters is the interrupt fired before run_conversation ran.
