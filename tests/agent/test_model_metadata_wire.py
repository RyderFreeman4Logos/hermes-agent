"""Structured api_content is what the wire sends, so the estimate counts that."""

from agent.model_metadata import estimate_messages_tokens_rough


def test_structured_api_content_substitutes_for_flattened_display_content():
    sidecar = [
        {"type": "text", "text": "structured replay"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    persisted = {
        "role": "user",
        "content": "structured replay\n[screenshot]",
        "api_content": sidecar,
    }
    assert 1500 <= estimate_messages_tokens_rough([persisted]) < 2000
