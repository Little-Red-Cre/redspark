"""Supervised fine-tuning entry point for RedSpark conversation data."""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


TRAINING_CHAT_TEMPLATE = (
    "{{- bos_token }}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'user' %}"
    "{{- '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- generation %}"
    "{{- message['content'] + '<|im_end|>' }}"
    "{%- endgeneration %}"
    "{{- '\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune RedSpark-1.0-FlashLight-Preview with SFT data.")
    parser.add_argument("--model", default="model/RedSpark-1.0-FlashLight-Preview")
    parser.add_argument("--data", default="data/processed/sft.jsonl")
    parser.add_argument("--output", default="artifacts/sft")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--full-finetune", action="store_true", help="Disable the default LoRA adapter training.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = TRAINING_CHAT_TEMPLATE
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if use_bf16 else torch.float32,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    dataset = load_dataset("json", data_files=args.data, split="train")
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
