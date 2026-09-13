"""Small, dependency-free validation pass for the RedSpark SFT JSONL contract.

``data/schemas/sft_record.schema.json`` is the machine-readable declaration of
the same contract, for editors and CI. This script is the executable gate: it is
a strict superset, because a JSON Schema cannot express the rules below readably.

Errors (the record is rejected):

* an answer containing a literal ``</think>`` is rejected. The chat template
  splits the assistant turn on the first and the last ``</think>``, so a stray
  one silently drops everything between them instead of failing loudly;
* a record with no assistant turn is rejected, because ``assistant_only_loss``
  would build an empty mask and the sample would contribute nothing to the loss.

Warnings (advisory, the record still passes):

* an empty reasoning block teaches the model that reasoning is always empty,
  which is the exact behaviour the thinking-format contract exists to prevent.
  It is a data-quality signal rather than a format violation, because the
  template renders it fine;
* ``<|im_start|>`` and ``<|im_end|>`` tokenize to single control tokens. Content
  containing them injects real turn boundaries into the prompt and can truncate
  generation. Not an error, because documentation about chat formats may
  legitimately quote them.
"""

import argparse
import json
import sys
from pathlib import Path

ROLES = {"system", "user", "assistant"}
RECORD_KEYS = {"messages"}
MESSAGE_KEYS = {"role", "content"}
# The only two literals the tokenizer maps to a single control token (130072 and
# 130073); everything else the template emits is ordinary text.
CONTROL_LITERALS = ("<|im_start|>", "<|im_end|>")
MAX_REPORTED_ERRORS = 20


class ContractError(ValueError):
    """Raised when a record violates the SFT contract."""


def _fail(path: Path, line_number: int, message: str) -> None:
    raise ContractError(f"{path}:{line_number}: {message}")


def parse_thinking(content: str) -> tuple[str, str, str] | None:
    """Split assistant content into ``(head, reasoning, answer)``.

    This mirrors the chat template, which partitions on the first ``<think>`` and
    then on the first ``</think>`` after it. ``head`` is everything before the
    opening tag and must be blank.

    Returns ``None`` when the content carries no ``<think>`` block at all. That
    is renderable -- the template synthesises an empty block -- but it is not the
    canonical thinking format the SFT contract requires.

    Kept here as the single parse implementation: ``tools/verify_refactor.py``
    imports it rather than re-deriving the split, so the data contract and the
    regression gate cannot drift apart.
    """
    if "<think>" not in content and "</think>" not in content:
        return None
    head, _, rest = content.partition("<think>")
    reasoning, _, answer = rest.partition("</think>")
    return head, reasoning, answer


def check_message(path: Path, line_number: int, index: int, message: object) -> tuple[str, list[str]]:
    """Validate one turn.

    Returns ``(kind, warnings)`` where ``kind`` is ``"other"``, ``"thinking"`` or
    ``"empty-thinking"``. Raises ``ContractError`` on a violation.
    """
    if not isinstance(message, dict):
        _fail(path, line_number, f"messages[{index}] must be an object")
    unknown = set(message) - MESSAGE_KEYS
    if unknown:
        _fail(path, line_number, f"messages[{index}] has unsupported keys: {sorted(unknown)}")
    role = message.get("role")
    if role not in ROLES:
        _fail(path, line_number, f"messages[{index}] role must be one of {sorted(ROLES)}, got {role!r}")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        _fail(path, line_number, f"messages[{index}] content must be a non-empty string")
    warnings = [
        f"content contains the control literal {literal}; it tokenizes to a single control "
        "token and can inject a turn boundary or truncate generation"
        for literal in CONTROL_LITERALS
        if literal in content
    ]
    if role != "assistant":
        return "other", warnings

    parsed = parse_thinking(content)
    if parsed is None:
        _fail(
            path,
            line_number,
            f"messages[{index}] assistant content has no <think> block; assistant turns must "
            "carry an inline <think>...</think> block followed by the answer",
        )
    first_close = content.find("</think>")
    if first_close != -1 and "</think>" in content[first_close + len("</think>") :]:
        _fail(
            path,
            line_number,
            f"messages[{index}] assistant content has more than one </think>; the chat template "
            "splits on the first and the last one, so everything between them would be dropped",
        )
    open_count = content.count("<think>")
    close_count = content.count("</think>")
    if open_count != 1 or close_count != 1:
        _fail(
            path,
            line_number,
            f"messages[{index}] assistant content must contain exactly one <think> and exactly "
            f"one </think>, found {open_count} and {close_count}",
        )
    head, reasoning, answer = parsed
    if head.strip():
        _fail(
            path,
            line_number,
            f"messages[{index}] assistant content must start with <think>; the chat template "
            f"drops everything before it, so {head.strip()!r} would be lost",
        )
    if not answer.strip():
        _fail(path, line_number, f"messages[{index}] assistant content has no answer after </think>")
    return ("thinking" if reasoning.strip() else "empty-thinking"), warnings


def check_record(path: Path, line_number: int, record: object) -> tuple[bool, list[str]]:
    """Validate one JSONL record, raising ``ContractError`` on a violation.

    Returns ``(thin, warnings)``. ``thin`` is False when some assistant turn has
    an empty reasoning block -- reported through this flag alone, so there is one
    channel per finding. Callers must look at both: ignoring ``thin`` is how the
    empty-reasoning regression slips through a gate unnoticed.
    """
    if not isinstance(record, dict):
        _fail(path, line_number, "record must be a JSON object")
    unknown = set(record) - RECORD_KEYS
    if unknown:
        _fail(path, line_number, f"record has unsupported keys: {sorted(unknown)}")
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        _fail(path, line_number, "messages must be a list of at least two turns")
    kinds = []
    warnings = []
    for index, message in enumerate(messages):
        kind, message_warnings = check_message(path, line_number, index, message)
        kinds.append(kind)
        warnings.extend(f"messages[{index}]: {warning}" for warning in message_warnings)
    assistant_kinds = [kind for kind in kinds if kind != "other"]
    if not assistant_kinds:
        _fail(path, line_number, "record has no assistant turn; assistant_only_loss would be empty")
    return all(kind == "thinking" for kind in assistant_kinds), warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, nargs="+")
    args = parser.parse_args()

    total = 0
    thin = 0
    errors: list[str] = []
    warnings: list[str] = []

    for path in args.path:
        if not path.is_file():
            errors.append(f"{path}: no such file")
            continue
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    errors.append(f"{path}:{line_number}: invalid JSON: {error}")
                    continue
                try:
                    thin_ok, record_warnings = check_record(path, line_number, record)
                except ContractError as error:
                    errors.append(str(error))
                    continue
                if not thin_ok:
                    thin += 1
                    warnings.append(
                        f"{path}:{line_number}: an assistant reasoning block is empty; it teaches "
                        "the model that reasoning is always empty"
                    )
                warnings.extend(f"{path}:{line_number}: {warning}" for warning in record_warnings)
                total += 1

    # Report every problem rather than stopping at the first, so a large dataset
    # can be fixed in one pass instead of one record per run.
    for warning in warnings[:MAX_REPORTED_ERRORS]:
        print(f"warning: {warning}", file=sys.stderr)
    if len(warnings) > MAX_REPORTED_ERRORS:
        print(f"warning: ... and {len(warnings) - MAX_REPORTED_ERRORS} more", file=sys.stderr)
    for error in errors[:MAX_REPORTED_ERRORS]:
        print(f"error: {error}", file=sys.stderr)
    if len(errors) > MAX_REPORTED_ERRORS:
        print(f"error: ... and {len(errors) - MAX_REPORTED_ERRORS} more", file=sys.stderr)

    print(f"valid_records={total}")
    if thin:
        print(f"empty_reasoning_records={thin}")
    if errors:
        print(f"invalid_records={len(errors)}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
