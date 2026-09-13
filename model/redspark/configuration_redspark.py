"""Configuration for the RedSpark causal language model.

The initial revision is shape-compatible with MiniCPM5-2B's standard LLaMA
decoder. Keeping a RedSpark-owned configuration gives future architectural
revisions a stable, explicit compatibility boundary.
"""

from __future__ import annotations

from transformers import LlamaConfig


class RedSparkConfig(LlamaConfig):
    """RedSpark configuration, initially compatible with ``LlamaConfig``."""

    model_type = "redspark"

    def __init__(self, architecture_revision: str = "baseline-v1", **kwargs) -> None:
        super().__init__(**kwargs)
        self.architecture_revision = architecture_revision

    @classmethod
    def from_llama_config(cls, config: LlamaConfig) -> "RedSparkConfig":
        """Create a RedSpark config without changing compatible tensor shapes."""
        values = config.to_dict()
        values.pop("model_type", None)
        values["architectures"] = ["RedSparkForCausalLM"]
        return cls(**values)
