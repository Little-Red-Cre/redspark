"""Small, dependency-free validation pass for the RedSpark SFT JSONL contract."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    total = 0
    for line_number, line in enumerate(args.path.open(encoding="utf-8"), 1):
        record = json.loads(line)
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"line {line_number}: messages must contain at least two turns")
        for message in messages:
            if message.get("role") not in {"system", "user", "assistant"} or not str(message.get("content", "")).strip():
                raise ValueError(f"line {line_number}: invalid role or empty content")
        total += 1
    print(f"valid_records={total}")


if __name__ == "__main__":
    main()

