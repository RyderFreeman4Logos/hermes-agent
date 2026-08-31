"""Opt-in checkpoint context engine (architecture v2)."""
from agent.context_engine import ContextEngine


class CheckpointContextEngine(ContextEngine):
    @property
    def name(self):
        return "checkpoint"

    def update_from_response(self, usage):
        usage = usage if isinstance(usage, dict) else {}
        self.last_prompt_tokens = int(usage.get("prompt_tokens") or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens") or 0)
        self.last_total_tokens = int(usage.get("total_tokens") or self.last_prompt_tokens + self.last_completion_tokens)

    def should_compress(self, prompt_tokens=None):
        return bool(prompt_tokens is not None and prompt_tokens >= self.threshold_tokens > 0)

    def compress(self, messages, current_tokens=None, focus_topic=None, force=False, memory_context="", **kwargs):
        return messages
