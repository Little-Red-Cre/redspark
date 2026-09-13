# RedSpark model source

This directory is the only writable model-architecture boundary in RedSpark.
`baseline-v1` mirrors MiniCPM5-2B's standard LLaMA-compatible decoder exactly:
42 decoder layers, hidden size 2048, 16 query heads, 2 KV heads and a 524288-token context window.

## Layout

- `configuration_redspark.py`: RedSpark-owned configuration and architecture revision marker.
- `modeling_redspark.py`: RedSpark decoder, decoder layer and causal-LM classes.
- `loading.py`: one shared path for weight/config/tokenizer loading used by training and inference.
- `chat_template.jinja`: the single chat template shared by training and inference.
- `model-weights/base/`: official `MiniCPM5-2B-Base` checkpoint used for architecture research and continued pre-training.
- `model-weights/minicpm5-2b-final-reference/`: preserved upstream final checkpoint, used only as a reference baseline.

Weight binaries are not stored in this checkout: each directory keeps its `config.json`, `tokenizer.json` and `chat_template.jinja` for provenance, and `tools/model_weights.py` re-downloads or removes the weights. Use `download --target base` / `--target reference` to fetch one of them.

The baseline retains upstream tensor names and shapes, so the official base weights load without conversion. Future Transformer changes must be made here, paired with an explicit checkpoint migration and a short inference/training verification.

Both RedSpark model classes bypass their immediate parent constructor by calling `LlamaPreTrainedModel.__init__` directly. The parent constructors would build a full stack of upstream layers that is then discarded, which wastes one construction and one weight-initialisation pass. The bodies of those constructors are mirrored verbatim with RedSpark classes substituted, so the initialisation semantics stay identical to upstream. Those mirrored assignments are required, not redundant.

The shared loader prefers `base/` whenever `model.safetensors` is present, and otherwise falls back to `minicpm5-2b-final-reference/`. With both directories empty it still resolves a directory, but loading a model fails for lack of weights — config and tokenizer load normally.
