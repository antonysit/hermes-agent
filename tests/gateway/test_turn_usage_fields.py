"""Standalone verification of _turn_usage_fields (PR #87418 cherry-pick).

Extracted from the PR's TestTurnUsageFields into a self-contained pytest so
it runs against our local-patch/live tree without the PR's larger test-file
merge. Same assertions; the helper itself lives in
gateway/platforms/api_server.py.
"""

import types

import pytest

from gateway.platforms.api_server import _turn_usage_fields


class TestTurnUsageFields:
    """The turn's usage block: the cumulative cost counters, plus the gauge
    numbers (context_tokens / context_window) a remote client cannot compute
    from them."""

    def test_reports_cost_counters_and_compressor_numbers(self):
        agent = types.SimpleNamespace(
            session_prompt_tokens=1200,
            session_completion_tokens=300,
            session_total_tokens=1500,
            context_compressor=types.SimpleNamespace(
                last_prompt_tokens=35019, context_length=1050000
            ),
        )
        assert _turn_usage_fields(agent) == {
            "input_tokens": 1200,
            "output_tokens": 300,
            "total_tokens": 1500,
            "context_tokens": 35019,
            "context_window": 1050000,
        }

    def test_anchored_figure_preferred_over_last_prompt_tokens(self):
        from agent.usage_anchor import capture_usage_anchor

        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        agent = types.SimpleNamespace(
            session_prompt_tokens=1200,
            session_completion_tokens=300,
            session_total_tokens=1500,
            context_compressor=types.SimpleNamespace(
                last_prompt_tokens=35019, context_length=1050000
            ),
            _usage_anchor=capture_usage_anchor(35019, 42, messages),
        )
        fields = _turn_usage_fields(agent, messages=messages)
        assert fields["context_tokens"] == 35019 + 42
        assert fields["context_window"] == 1050000

    def test_stale_anchor_falls_back_to_last_prompt_tokens(self):
        from agent.usage_anchor import capture_usage_anchor

        messages = [{"role": "user", "content": "hi"}]
        agent = types.SimpleNamespace(
            context_compressor=types.SimpleNamespace(
                last_prompt_tokens=35019, context_length=1050000
            ),
            _usage_anchor=capture_usage_anchor(9000, 10, messages),
        )
        # Anchor identity is a content fingerprint (#99421): a transcript
        # whose priced message was rewritten (compaction/rewind) fails the
        # match closed, so the figure falls back to last_prompt_tokens.
        reloaded = [{"role": "user", "content": "hi (compacted away)"}]
        assert _turn_usage_fields(agent, messages=reloaded)["context_tokens"] == 35019

    def test_compression_sentinel_clamps_to_zero(self):
        agent = types.SimpleNamespace(
            context_compressor=types.SimpleNamespace(
                last_prompt_tokens=-1, context_length=272000
            )
        )
        assert _turn_usage_fields(agent)["context_tokens"] == 0

    def test_missing_compressor_is_zeroed(self):
        assert _turn_usage_fields(types.SimpleNamespace()) == {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "context_tokens": 0,
            "context_window": 0,
        }

    def test_non_numeric_attributes_are_zeroed(self):
        agent = types.SimpleNamespace(
            session_prompt_tokens=None,
            session_completion_tokens=None,
            session_total_tokens=None,
            context_compressor=types.SimpleNamespace(
                last_prompt_tokens=None, context_length="unknown"
            ),
        )
        assert _turn_usage_fields(agent) == {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "context_tokens": 0,
            "context_window": 0,
        }
