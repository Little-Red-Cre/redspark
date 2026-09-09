"""Repeatable local inference benchmark for RedSpark."""

import argparse
import os
import time
from pathlib import Path

import psutil
import torch

from chat import generate, load_model


DEFAULT_MODEL_DIR = Path(__file__).parents[1] / "model" / "Qwen3-1.7B"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    process = psutil.Process(os.getpid())
    load_start = time.perf_counter()
    tokenizer, model = load_model(args.model_dir)
    load_seconds = time.perf_counter() - load_start
    prompt = "Explain retrieval augmented generation in one short sentence."
    prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    output = generate(model, tokenizer, prompt, False, args.max_new_tokens)
    elapsed = time.perf_counter() - start
    output_tokens = len(tokenizer(output, add_special_tokens=False)["input_ids"])
    print(f"device={model.device}")
    print(f"load_seconds={load_seconds:.3f}")
    print(f"prompt_tokens={prompt_tokens}")
    print(f"output_tokens={output_tokens}")
    print(f"generate_seconds={elapsed:.3f}")
    print(f"tokens_per_second={output_tokens / elapsed:.2f}")
    print(f"rss_gib={process.memory_info().rss / 1024**3:.2f}")
    if torch.cuda.is_available():
        print(f"cuda_allocated_gib={torch.cuda.memory_allocated() / 1024**3:.2f}")
        print(f"cuda_reserved_gib={torch.cuda.memory_reserved() / 1024**3:.2f}")
        print(f"cuda_peak_allocated_gib={torch.cuda.max_memory_allocated() / 1024**3:.2f}")
    print(f"output={output!r}")


if __name__ == "__main__":
    main()
