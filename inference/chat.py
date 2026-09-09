"""Local inference entry point for the RedSpark Qwen3 model."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_DIR = Path(__file__).parents[1] / "model" / "Qwen3-1.7B"


def load_model(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto", device_map="auto", trust_remote_code=False)
    return tokenizer, model


def generate(model, tokenizer, prompt: str, thinking: bool, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
    return tokenizer.decode(output_ids[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one RedSpark inference request.")
    parser.add_argument("prompt")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-thinking", action="store_true")
    args = parser.parse_args()
    tokenizer, model = load_model(args.model_dir)
    print(generate(model, tokenizer, args.prompt, not args.no_thinking, args.max_new_tokens))


if __name__ == "__main__":
    main()

