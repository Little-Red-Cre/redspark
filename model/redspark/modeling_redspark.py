"""RedSpark Transformer implementation.

This baseline deliberately preserves MiniCPM5-2B tensor names, dimensions and
forward behaviour so its official base weights can be loaded exactly. New
attention, MLP, routing or coding-specific modules belong in the RedSpark
classes below rather than in the installed Transformers package.
"""

from __future__ import annotations

from torch import nn
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

from .configuration_redspark import RedSparkConfig


class RedSparkDecoderLayer(LlamaDecoderLayer):
    """RedSpark-owned decoder layer extension point.

    It is intentionally behaviour-identical to the upstream LLaMA layer in
    ``baseline-v1``. Structural changes start here while preserving a clear
    weight-migration surface.
    """


class RedSparkModel(LlamaModel):
    """LLaMA-compatible decoder stack constructed from RedSpark layer classes."""

    config_class = RedSparkConfig

    def __init__(self, config: RedSparkConfig) -> None:
        # Deliberately bypass LlamaModel.__init__: it would build the whole stack
        # with LlamaDecoderLayer, and the result is replaced wholesale below.
        # The assignments that follow mirror LlamaModel.__init__ verbatim, with
        # RedSpark layers substituted. Keep them in sync with upstream: they are
        # load-bearing, not redundant.
        LlamaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [RedSparkDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()


class RedSparkForCausalLM(LlamaForCausalLM):
    """Causal-LM head backed by the RedSpark decoder implementation."""

    config_class = RedSparkConfig
    _no_split_modules = ["RedSparkDecoderLayer"]

    def __init__(self, config: RedSparkConfig) -> None:
        LlamaPreTrainedModel.__init__(self, config)
        self.model = RedSparkModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
