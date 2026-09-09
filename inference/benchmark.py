"""Repeatable local inference benchmark for RedSpark."""

import argparse
import os
import time
from pathlib import Path

import psutil
import torch

from chat import generate, load_model, resolve_device


MODEL_NAME = "RedSpark-1.0-FlashLight-Preview"
DEFAULT_MODEL_DIR = Path(__file__).parents[1] / "model" / MODEL_NAME


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs cannot be negative")

    device = resolve_device(args.device)
    process = psutil.Process(os.getpid())
    load_start = time.perf_counter()
    tokenizer, model = load_model(args.model_dir, device)
    load_seconds = time.perf_counter() - load_start
    prompt = "Explain retrieval augmented generation in one short sentence."
    prompt_inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_tokens = prompt_inputs.input_ids.shape[-1]
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(args.warmup_runs):
        generate(model, tokenizer, prompt, args.max_new_tokens, do_sample=False)
    if device == "cuda":
        torch.cuda.synchronize()
    outputs = []
    elapsed = 0.0
    for _ in range(args.runs):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        outputs.append(generate(model, tokenizer, prompt, args.max_new_tokens, do_sample=False))
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - start
    output_tokens = sum(len(tokenizer(output, add_special_tokens=False)["input_ids"]) for output in outputs)
    print(f"device={device}")
    print(f"model_name={MODEL_NAME}")
    print(f"load_seconds={load_seconds:.3f}")
    print(f"prompt_tokens={prompt_tokens}")
    print(f"output_tokens_total={output_tokens}")
    print(f"runs={args.runs}")
    print(f"warmup_runs={args.warmup_runs}")
    print(f"generate_seconds={elapsed:.3f}")
    print(f"tokens_per_second={output_tokens / elapsed:.2f}")
    print(f"rss_gib={process.memory_info().rss / 1024**3:.2f}")
    if device == "cuda":
        print(f"cuda_allocated_gib={torch.cuda.memory_allocated() / 1024**3:.2f}")
        print(f"cuda_reserved_gib={torch.cuda.memory_reserved() / 1024**3:.2f}")
        print(f"cuda_peak_allocated_gib={torch.cuda.max_memory_allocated() / 1024**3:.2f}")
    print(f"output={outputs[-1]!r}")


if __name__ == "__main__":
    main()
