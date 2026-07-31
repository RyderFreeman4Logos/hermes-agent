"""Agent teardown allocator-trim wiring."""

import inspect

from run_agent import AIAgent


def test_agent_close_forces_trim_after_resource_release():
    source = inspect.getsource(AIAgent.close)
    trim = source.index('trim_memory(force=True, reason="agent close")')
    client_release = source.index("_close_cached_request_openai_client")

    assert trim > client_release
