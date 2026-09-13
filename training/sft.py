"""Supervised fine-tuning entry point for RedSpark conversation data."""

import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.redspark.loading import DEFAULT_WEIGHTS_DIR, load_model as load_redspark_model, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune RedSpark from the MiniCPM5-2B Base checkpoint with SFT data.")
    parser.add_argument("--model", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--data", default="data/processed/smoke_sft.jsonl")
    parser.add_argument("--output", default="artifacts/sft")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--full-finetune", action="store_true", help="Disable the default LoRA adapter training.")
    args = parser.parse_args()

    # load_tokenizer attaches the shared RedSpark chat template, which carries the
    # {% generation %} markers that assistant_only_loss needs. Training and
    # inference therefore always render identical prompts.
    tokenizer = load_tokenizer(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if args.full_finetune and torch.cuda.is_available():
        available_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if available_gib < 32:
            raise RuntimeError(
                f"Full fine-tuning needs at least 32 GiB of GPU memory with this AdamW setup; found {available_gib:.1f} GiB. "
                "Use distributed ZeRO/FSDP or CPU/NVMe offload for full training. The default LoRA mode remains available for a smoke test."
            )
    model = load_redspark_model(
        args.model,
        dtype=torch.bfloat16 if use_bf16 else torch.float32,
    )
    model.config.use_cache = False
    dataset = load_dataset("json", data_files=args.data, split="train")
    if len(dataset) < 32:
        # The default is the checked-in smoke fixture, so a bare `python training/sft.py`
        # would otherwise produce a plausible-looking checkpoint from a handful of
        # samples without saying so.
        print(
            f"warning: {args.data} has only {len(dataset)} records; this is a pipeline "
            "smoke test, not a training run.",
            file=sys.stderr,
        )
    peft_config = None if args.full_finetune else LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        args=SFTConfig(
            output_dir=args.output,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-5,
            logging_steps=10,
            save_steps=500,
            bf16=use_bf16,
            use_cpu=not torch.cuda.is_available(),
            max_length=args.max_length,
            assistant_only_loss=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to="none",
        ),
    )
    trainer.train()
    trainer.save_model(Path(args.output) / "final")


if __name__ == "__main__":
    main()
