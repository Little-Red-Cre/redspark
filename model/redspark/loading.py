"""Shared loading helpers for RedSpark training and inference."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaConfig

from .configuration_redspark import RedSparkConfig
from .modeling_redspark import RedSparkForCausalLM


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_ROOT = PROJECT_ROOT / "model" / "redspark" / "model-weights"
BASE_WEIGHTS_DIR = WEIGHTS_ROOT / "base"
FINAL_REFERENCE_WEIGHTS_DIR = WEIGHTS_ROOT / "minicpm5-2b-final-reference"
CHAT_TEMPLATE_PATH = Path(__file__).resolve().parent / "chat_template.jinja"


def load_chat_template() -> str:
    """Return the single chat template shared by training and inference.

    The template **renders** byte for byte identically to the upstream
    MiniCPM5-2B template (``model-weights/*/chat_template.jinja``); only the
    assistant branch is restructured, to add ``{% generation %}`` markers that
    ``assistant_only_loss`` needs to build its mask. Keeping one copy guarantees
    training and inference never drift apart.
    """
    if not CHAT_TEMPLATE_PATH.is_file():
        raise RuntimeError(f"RedSpark chat template is missing: {CHAT_TEMPLATE_PATH}")
    return CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")


def has_model_weights(weights_dir: Path) -> bool:
    """Return whether a local directory contains a complete weight file entry."""
    return (weights_dir / "model.safetensors").is_file() or any(weights_dir.glob("model-*.safetensors"))


# Prefer the official Base checkpoint; fall back to the preserved final upstream
# reference checkpoint when Base weights are absent. Both directories ship
# without weight binaries — fetch them with tools/model_weights.py.
DEFAULT_WEIGHTS_DIR = BASE_WEIGHTS_DIR if has_model_weights(BASE_WEIGHTS_DIR) else FINAL_REFERENCE_WEIGHTS_DIR


def load_redspark_config(weights_dir: Path) -> RedSparkConfig:
    """Load an upstream-compatible config as a RedSpark-owned configuration."""
    base_config = LlamaConfig.from_pretrained(weights_dir, local_files_only=True)
    return RedSparkConfig.from_llama_config(base_config)


def load_tokenizer(weights_dir: Path, chat_template: str | None = None) -> AutoTokenizer:
    """Load the tokenizer and attach the shared RedSpark chat template.

    The checkpoint's own template is replaced by default so that training and
    inference always agree. Pass ``chat_template`` to supply a different one, or
    an empty string to keep whatever the checkpoint ships with.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        weights_dir, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    if chat_template is None:
        chat_template = load_chat_template()
    if chat_template:
        tokenizer.chat_template = chat_template
    return tokenizer


def load_model(weights_dir: Path, dtype: torch.dtype, **kwargs) -> RedSparkForCausalLM:
    """Load compatible weights through the RedSpark model implementation."""
    config = load_redspark_config(weights_dir)
    return RedSparkForCausalLM.from_pretrained(
        weights_dir,
        config=config,
        dtype=dtype,
        local_files_only=True,
        **kwargs,
    )
