"""Supervised fine-tuning entry point for RedSpark conversation data."""

import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune RedSpark-1.0-FlashLight-Preview with SFT data.")
    parser.add_argument("--model", default="model/RedSpark-1.0-FlashLight-Preview")
    parser.add_argument("--data", default="data/processed/sft.jsonl")
    parser.add_argument("--output", default="artifacts/sft")
    parser.add_argument("--epochs", type=float, default=3)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", trust_remote_code=False)
    dataset = load_dataset("json", data_files=args.data, split="train")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        formatting_func=lambda row: tokenizer.apply_chat_template(row["messages"], tokenize=False),
        args=TrainingArguments(
            output_dir=args.output,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-5,
            logging_steps=10,
            save_steps=500,
            bf16=True,
            report_to="none",
        ),
    )
    trainer.train()
    trainer.save_model(Path(args.output) / "final")


if __name__ == "__main__":
    main()
