"""Local inference entry point for the MiniCPM5-based RedSpark preview."""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "RedSpark-1.0-FlashLight-Preview"
DEFAULT_MODEL_DIR = Path(__file__).parents[1] / "model" / MODEL_NAME


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def load_model(model_dir: Path, device: str = "auto"):
    device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else "cpu",
        trust_remote_code=False,
    ).eval()
    return tokenizer, model


def generate(model, tokenizer, prompt: str, max_new_tokens: int, do_sample: bool = True) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
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
    parser.add_argument("--no-thinking", action="store_true", help="Accepted for compatibility; MiniCPM5-2B always uses Think mode.")
    args = parser.parse_args()
    if args.no_thinking:
        print("MiniCPM5-2B only supports Think mode; --no-thinking is ignored.", file=sys.stderr)
    tokenizer, model = load_model(args.model_dir, args.device)
    print(generate(model, tokenizer, args.prompt, args.max_new_tokens))


if __name__ == "__main__":
    main()
