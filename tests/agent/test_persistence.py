from agent.checkpoint_engine import (
    CheckpointContextEngine, ContentAddressedArtifacts, DurableCheckpointStore,
)


def test_live_commit_rechecks_revision_before_publish():
    store = DurableCheckpointStore()
    messages = [{"role": "user", "content": "work", "_row_id": 1}]
    engine = CheckpointContextEngine({"mode": "live"}, store=store, session_id="s")
    engine._before_commit = lambda: store.revision("s", messages + [{"role": "user", "content": "new", "_row_id": 2}])
    assert engine.compress(messages) is messages
    assert engine.compression_count == 0
    assert engine.last_rejection


def test_artifact_store_is_content_addressed_and_verified(tmp_path):
    store = ContentAddressedArtifacts(tmp_path)
    digest = store.put("receipt")
    assert store.read(digest) == "receipt"
    assert digest == store.put("receipt")
