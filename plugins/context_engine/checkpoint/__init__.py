from agent.checkpoint_engine import CheckpointContextEngine

__all__ = ["CheckpointContextEngine"]


def create_context_engine(config):
    return CheckpointContextEngine(config=config)
