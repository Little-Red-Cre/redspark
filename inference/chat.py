"""Local inference entry point for RedSpark."""

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.redspark.loading import DEFAULT_WEIGHTS_DIR, load_model as load_redspark_model, load_tokenizer


MODEL_NAME = "RedSpark Base"
DEFAULT_MODEL_DIR = DEFAULT_WEIGHTS_DIR


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def load_model(model_dir: Path, device: str = "auto", adapter_dir: Path | None = None):
    device = resolve_device(device)
    tokenizer = load_tokenizer(model_dir)
    model = load_redspark_model(
        model_dir,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else "cpu",
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir, local_files_only=True)
    return tokenizer, model.eval()


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool = True,
    enable_thinking: bool = True,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    generation_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        generation_kwargs.update(temperature=1.0, top_p=0.95)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    return tokenizer.decode(output_ids[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Run one {MODEL_NAME} inference request.")
    parser.add_argument("prompt")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--adapter", type=Path, help="Optional PEFT adapter directory produced by LoRA training.")
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help=(
            "Skip the <think> scaffold. The template still opens and closes a reasoning block, "
            "but leaves it empty so the model answers directly."
        ),
    )
    args = parser.parse_args()
    tokenizer, model = load_model(args.model_dir, args.device, args.adapter)
    print(
        generate(
            model,
            tokenizer,
            args.prompt,
            args.max_new_tokens,
            enable_thinking=not args.no_thinking,
        )
    )


if __name__ == "__main__":
    main()
