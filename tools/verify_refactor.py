"""Structural, template and numerical regression checks for the RedSpark refactor.

Usage:
    python tools/verify_refactor.py --out before.json
    python tools/verify_refactor.py --out after.json
    python tools/verify_refactor.py --compare before.json after.json
    python tools/verify_refactor.py --equivalence --device cuda --out after.json
    python tools/verify_refactor.py --data data/processed/smoke_sft.jsonl

The `--compare` mode is the gate: it fails when a refactor changes parameter
names, weight-initialisation counts, chat-template rendering, or the assistant
mask span. `--data` additionally renders a real SFT JSONL file and applies the
same assistant-span assertions to every record. Every mode exits non-zero when a
check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import LlamaConfig, LlamaForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaModel

from model.redspark.configuration_redspark import RedSparkConfig
from model.redspark.loading import CHAT_TEMPLATE_PATH, DEFAULT_WEIGHTS_DIR
from model.redspark.modeling_redspark import (
    RedSparkDecoderLayer,
    RedSparkForCausalLM,
    RedSparkModel,
)

# The thinking-format parser lives with the data contract so the two cannot drift.
DATA_DIR = PROJECT_ROOT / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from validate import ContractError, check_record, parse_thinking  # noqa: E402

SCHEMA_VERSION = 2

TINY_DIMS = {
    "vocab_size": 128,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 64,
}

FIXTURES = {
    "single_turn": [
        {"role": "system", "content": "You are RedSpark, a concise coding assistant."},
        {
            "role": "user",
            "content": "What does a Python function return when it has no return statement?",
        },
        {"role": "assistant", "content": "It returns None."},
    ],
    "multi_turn": [
        {"role": "system", "content": "You are RedSpark."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4"},
    ],
    "with_tools": [
        {"role": "user", "content": "What is the weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
                }
            ],
        },
        {"role": "tool", "content": "18C and sunny"},
        {"role": "assistant", "content": "It is 18C and sunny in Paris."},
    ],
    # Canonical SFT shape: the assistant turn carries its own <think> block, which
    # is what data/schemas/sft_record.schema.json now requires. The other fixtures
    # keep bare content on purpose, to pin the template's synthesise-an-empty-block
    # fallback as well.
    "thinking_inline": [
        {"role": "system", "content": "You are RedSpark, a concise coding assistant."},
        {
            "role": "user",
            "content": "What does a Python function return when it has no return statement?",
        },
        {
            "role": "assistant",
            "content": (
                "<think>\n"
                "A function that reaches the end of its body returns None implicitly.\n"
                "</think>\n\n"
                "It returns None."
            ),
        },
    ],
}


def measure_construction(model_cls, config) -> tuple[dict, list[str], list[str]]:
    """Construct one model while counting init activity, restoring patches after."""
    counts = {"init_weights": 0, "llama_layers": 0, "redspark_layers": 0}

    original_init_weights = PreTrainedModel.init_weights
    original_llama_layer_init = LlamaDecoderLayer.__init__
    redspark_has_own_init = "__init__" in RedSparkDecoderLayer.__dict__
    original_redspark_layer_init = RedSparkDecoderLayer.__init__

    def counted_init_weights(self):
        counts["init_weights"] += 1
        return original_init_weights(self)

    def counted_llama_layer_init(self, config, layer_idx):
        counts["llama_layers"] += 1
        return original_llama_layer_init(self, config, layer_idx)

    def counted_redspark_layer_init(self, config, layer_idx):
        counts["redspark_layers"] += 1
        return original_redspark_layer_init(self, config, layer_idx)

    PreTrainedModel.init_weights = counted_init_weights
    LlamaDecoderLayer.__init__ = counted_llama_layer_init
    RedSparkDecoderLayer.__init__ = counted_redspark_layer_init
    try:
        model = model_cls(config)
        keys = sorted(model.state_dict().keys())
        attributes = sorted(vars(model))
    finally:
        PreTrainedModel.init_weights = original_init_weights
        LlamaDecoderLayer.__init__ = original_llama_layer_init
        if redspark_has_own_init:
            RedSparkDecoderLayer.__init__ = original_redspark_layer_init
        else:
            del RedSparkDecoderLayer.__init__

    return counts, keys, attributes


def build_tiny_models() -> dict:
    """Compare tiny RedSpark and Llama models, measuring each construction apart."""
    redspark_counts, redspark_keys, redspark_attrs = measure_construction(
        RedSparkForCausalLM, RedSparkConfig(**TINY_DIMS)
    )
    llama_counts, llama_keys, llama_attrs = measure_construction(
        LlamaForCausalLM, LlamaConfig(**TINY_DIMS)
    )
    _, _, redspark_model_attrs = measure_construction(
        RedSparkModel, RedSparkConfig(**TINY_DIMS)
    )
    _, _, llama_model_attrs = measure_construction(LlamaModel, LlamaConfig(**TINY_DIMS))

    return {
        "redspark_counts": redspark_counts,
        "llama_counts": llama_counts,
        "redspark_param_keys": redspark_keys,
        "llama_param_keys": llama_keys,
        "param_keys_identical": redspark_keys == llama_keys,
        "redspark_key_count": len(redspark_keys),
        "expected_redspark_layers": TINY_DIMS["num_hidden_layers"],
        "causal_lm_attribute_keys_identical": redspark_attrs == llama_attrs,
        "causal_lm_attribute_diff": sorted(set(redspark_attrs) ^ set(llama_attrs)),
        "model_attribute_keys_identical": redspark_model_attrs == llama_model_attrs,
        "model_attribute_diff": sorted(set(redspark_model_attrs) ^ set(llama_model_attrs)),
    }


def check_loading_roundtrip() -> dict:
    """Save a tiny upstream checkpoint and load it through RedSpark (invariant I2)."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        LlamaForCausalLM(LlamaConfig(**TINY_DIMS)).save_pretrained(directory)
        config = RedSparkConfig.from_llama_config(LlamaConfig(**TINY_DIMS))
        _, info = RedSparkForCausalLM.from_pretrained(
            directory, config=config, output_loading_info=True
        )
    return {
        "missing_keys": sorted(info.get("missing_keys", [])),
        "unexpected_keys": sorted(info.get("unexpected_keys", [])),
        "mismatched_keys": sorted(info.get("mismatched_keys", [])),
    }


def render_template(tokenizer, template: str, messages: list) -> dict:
    """Render one message fixture through a template and capture text and mask."""
    tokenizer.chat_template = template
    try:
        result = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
        )
        ids = list(result["input_ids"])
        mask = list(result["assistant_masks"])
        return {
            "text": tokenizer.decode(ids, skip_special_tokens=False),
            "input_ids": ids,
            "mask": mask,
            "mask_sum": sum(mask),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def describe_mask(tokenizer, entry: dict, messages: list) -> dict:
    """Describe the assistant-span semantics of one rendered fixture.

    Absolute assertions, so they hold even when the baseline snapshot predates
    the RedSpark template. Each assistant turn must contribute exactly one span
    that starts at the ``<think>`` scaffold and ends at ``<|im_end|>``; the
    ``<|im_start|>assistant`` header must stay outside the span.

    For canonical thinking-format turns, the authored reasoning and answer are
    recorded too. The template splits the turn on ``</think>`` and reassembles it,
    so a bug there can drop text from the span without changing its boundaries;
    the expected values let ``mask_semantics_problems`` catch that.
    """
    if "error" in entry:
        return {"error": entry["error"]}
    ids, mask = entry["input_ids"], entry["mask"]
    assistant_contents = [
        message.get("content", "") for message in messages if message.get("role") == "assistant"
    ]
    regions = []
    for index, (start, end) in enumerate(_mask_regions(mask)):
        text = tokenizer.decode(ids[start : end + 1], skip_special_tokens=False)
        region = {
            "start": start,
            "end": end,
            "starts_with_think": text.startswith("<think>"),
            "ends_with_im_end": text.rstrip().endswith("<|im_end|>"),
            "text": text,
        }
        parsed = parse_thinking(assistant_contents[index]) if index < len(assistant_contents) else None
        if parsed is not None:
            _, reasoning, answer = parsed
            # Mirror the template's own normalisation (strip('\n') / lstrip('\n')),
            # so the comparison in mask_semantics_problems is like for like.
            region["expected_reasoning"] = reasoning.strip("\n")
            region["expected_answer"] = answer.lstrip("\n")
        regions.append(region)
    return {
        "assistant_turns": sum(1 for message in messages if message.get("role") == "assistant"),
        "region_count": len(regions),
        "regions": regions,
    }


def mask_semantics_problems(semantics: dict | None) -> list[str]:
    """Return a list of human-readable problems with a fixture's mask semantics."""
    if semantics is None:
        return ["mask semantics missing"]
    if "error" in semantics:
        return [f"render error: {semantics['error']}"]
    problems = []
    if semantics["region_count"] != semantics["assistant_turns"]:
        problems.append(
            f"span count {semantics['region_count']} != assistant turns {semantics['assistant_turns']}"
        )
    for index, region in enumerate(semantics["regions"]):
        if not region["starts_with_think"]:
            problems.append(f"span {index} does not start with <think>: {region['text']!r}")
        if not region["ends_with_im_end"]:
            problems.append(f"span {index} does not end with <|im_end|>: {region['text']!r}")
        # A populated authored value must survive into the span verbatim. An empty
        # one means the fixture is not canonical thinking format, so there is
        # nothing to look for.
        expected_reasoning = region.get("expected_reasoning")
        expected_answer = region.get("expected_answer")
        if not expected_reasoning and not expected_answer:
            continue
        # Compare the span's own reasoning section rather than a substring search:
        # if the authored reasoning also appears in the answer, `in` would still
        # match after the template dropped the reasoning block entirely.
        parsed_span = parse_thinking(region["text"])
        if parsed_span is None:
            problems.append(f"span {index} has no <think> block to compare against the fixture")
            continue
        _, span_reasoning, span_answer = parsed_span
        span_reasoning = span_reasoning.strip("\n")
        span_answer = span_answer.lstrip("\n")
        if expected_reasoning and span_reasoning != expected_reasoning:
            problems.append(
                f"span {index} reasoning differs from the fixture: "
                f"{span_reasoning!r} != {expected_reasoning!r}"
            )
        if expected_answer and not span_answer.startswith(expected_answer):
            problems.append(
                f"span {index} dropped the authored answer: {expected_answer!r} "
                f"not at the start of {span_answer!r}"
            )
    return problems


def load_eval_tokenizer():
    """Tokenizer used by the rendering checks: upstream weights only, no RedSpark classes."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        DEFAULT_WEIGHTS_DIR, local_files_only=True, trust_remote_code=False, use_fast=True
    )


def collect_templates() -> dict:
    """Render every fixture through the upstream and the RedSpark chat templates."""
    tokenizer = load_eval_tokenizer()
    upstream = (DEFAULT_WEIGHTS_DIR / "chat_template.jinja").read_text(encoding="utf-8")
    redspark = CHAT_TEMPLATE_PATH.read_text(encoding="utf-8") if CHAT_TEMPLATE_PATH.is_file() else None

    rendered = {}
    for name, messages in FIXTURES.items():
        redspark_entry = render_template(tokenizer, redspark, messages) if redspark else None
        entry = {
            "upstream": render_template(tokenizer, upstream, messages),
            "redspark": redspark_entry,
        }
        if redspark_entry is not None:
            entry["mask_semantics"] = describe_mask(tokenizer, redspark_entry, messages)
        rendered[name] = entry

    return {
        "weights_dir": str(DEFAULT_WEIGHTS_DIR),
        "redspark_template_path": str(CHAT_TEMPLATE_PATH),
        "redspark_template_exists": redspark is not None,
        "fixtures": rendered,
    }


def check_data_file(path: Path) -> tuple[list[str], list[str]]:
    """Check an SFT JSONL file end to end: the data contract, then the rendering.

    ``data/validate.py`` owns the data contract and is imported rather than
    reimplemented, so the two cannot drift. Its ``check_record`` catches a record
    whose assistant turn is not canonical thinking format; the span assertions
    below catch a record that the contract accepts but the template renders badly.

    Returns ``(problems, warnings)``. ``problems`` fail the run. ``warnings`` are
    the contract's advisory findings, including the empty-reasoning case that the
    thinking-format contract exists to prevent -- dropping them here is how that
    regression would slip through the gate unnoticed.
    """
    tokenizer = load_eval_tokenizer()
    template = CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")
    problems = []
    warnings = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            problems.append(f"{path}:{line_number}: invalid JSON: {error}")
            continue
        try:
            thin_ok, record_warnings = check_record(path, line_number, record)
        except ContractError as error:
            problems.append(str(error))
            continue
        if not thin_ok:
            warnings.append(f"{path}:{line_number}: an assistant reasoning block is empty")
        warnings.extend(f"{path}:{line_number}: {warning}" for warning in record_warnings)
        messages = record["messages"]
        entry = render_template(tokenizer, template, messages)
        if "error" in entry:
            problems.append(f"{path}:{line_number}: render failed: {entry['error']}")
            continue
        if entry["mask_sum"] <= 0:
            problems.append(f"{path}:{line_number}: assistant mask is empty")
        for problem in mask_semantics_problems(describe_mask(tokenizer, entry, messages)):
            problems.append(f"{path}:{line_number}: {problem}")
    return problems, warnings


def check_equivalence(device: str) -> dict:
    """Load the real checkpoint twice and compare RedSpark logits against upstream.

    RedSparkDecoderLayer is currently an empty subclass of LlamaDecoderLayer, so
    this check proves the wiring is faithful to upstream. It is not a substitute
    for comparing against the previous RedSpark revision.
    """
    import gc

    import torch

    from model.redspark.loading import load_model

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    input_ids = torch.tensor([[0, 100, 200, 300, 400, 500, 600, 700]], device=device)

    def forward_logits(model):
        with torch.no_grad():
            logits = model(input_ids).logits
        return logits.float().cpu()

    redspark = load_model(DEFAULT_WEIGHTS_DIR, dtype, device_map=device)
    redspark.eval()
    redspark_logits = forward_logits(redspark)
    del redspark
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    upstream = LlamaForCausalLM.from_pretrained(
        DEFAULT_WEIGHTS_DIR, dtype=dtype, device_map=device, local_files_only=True
    )
    upstream.eval()
    upstream_logits = forward_logits(upstream)
    del upstream
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "device": device,
        "dtype": str(dtype),
        "logits_shape": list(redspark_logits.shape),
        "max_abs_logit_diff": (redspark_logits - upstream_logits).abs().max().item(),
        "argmax_equal": bool((redspark_logits.argmax(-1) == upstream_logits.argmax(-1)).all()),
        "next_token_redspark": int(redspark_logits[0, -1].argmax()),
        "next_token_upstream": int(upstream_logits[0, -1].argmax()),
    }


def snapshot() -> dict:
    structural = build_tiny_models()
    templates = collect_templates()
    roundtrip = check_loading_roundtrip()
    counts = structural["redspark_counts"]

    checks = {
        # Precondition: every `all(...)` below is vacuously true over an empty set.
        "V0_fixture_set_nonempty": bool(templates["fixtures"]),
        "V1_param_keys_identical": structural["param_keys_identical"],
        "V2_no_missing_or_unexpected_keys": not any(
            roundtrip[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys")
        ),
        "V3_no_attribute_drift_causal_lm": structural["causal_lm_attribute_keys_identical"],
        "V3_no_attribute_drift_model": structural["model_attribute_keys_identical"],
        "V4_no_llama_layers_built": counts["llama_layers"] == 0,
        "V5_render_identical": all(
            entry["redspark"] is not None
            and entry["redspark"].get("text") == entry["upstream"].get("text")
            for entry in templates["fixtures"].values()
        ),
        "V6_redspark_mask_nonzero": all(
            entry["redspark"] is not None and entry["redspark"].get("mask_sum", 0) > 0
            for entry in templates["fixtures"].values()
        ),
        "V7_mask_semantics_correct": all(
            not mask_semantics_problems(entry.get("mask_semantics"))
            for entry in templates["fixtures"].values()
        ),
    }
    info = {
        "upstream_mask_is_zero_by_design": all(
            entry["upstream"].get("mask_sum", 0) == 0 for entry in templates["fixtures"].values()
        ),
        "init_weights_delta_vs_upstream": counts["init_weights"]
        - structural["llama_counts"]["init_weights"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "structural": structural,
        "templates": templates,
        "roundtrip": roundtrip,
        "checks": checks,
        "info": info,
    }


def _mask_regions(mask: list[int]) -> list[list[int]]:
    """Collapse a 0/1 mask into inclusive index ranges of the 1s."""
    regions: list[list[int]] = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            regions.append([start, index - 1])
            start = None
    if start is not None:
        regions.append([start, len(mask) - 1])
    return regions


def compare(before_path: Path, after_path: Path) -> int:
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))

    for label, data in (("before", before), ("after", after)):
        if data.get("schema_version") != SCHEMA_VERSION:
            print(f"ERROR: {label} snapshot schema_version is not {SCHEMA_VERSION}")
            return 2

    failures = []
    bs, as_ = before["structural"], after["structural"]

    if bs["redspark_param_keys"] != as_["redspark_param_keys"]:
        failures.append("redspark parameter key set changed")
    if not as_["param_keys_identical"]:
        failures.append("redspark parameter keys no longer match LlamaForCausalLM")
    if not as_["causal_lm_attribute_keys_identical"]:
        failures.append(f"attribute drift on RedSparkForCausalLM: {as_['causal_lm_attribute_diff']}")
    if not as_["model_attribute_keys_identical"]:
        failures.append(f"attribute drift on RedSparkModel: {as_['model_attribute_diff']}")

    before_counts, after_counts = bs["redspark_counts"], as_["redspark_counts"]
    if after_counts["llama_layers"] != 0:
        failures.append(
            f"LlamaDecoderLayer still built during RedSpark construction: {after_counts['llama_layers']}"
        )
    if after_counts["redspark_layers"] != as_["expected_redspark_layers"]:
        failures.append(
            f"RedSparkDecoderLayer build count {after_counts['redspark_layers']} "
            f"!= {as_['expected_redspark_layers']}"
        )
    if after_counts["init_weights"] != as_["llama_counts"]["init_weights"]:
        failures.append(
            f"RedSpark init_weights={after_counts['init_weights']} "
            f"!= upstream Llama init_weights={as_['llama_counts']['init_weights']}"
        )
    if bs["llama_counts"] != as_["llama_counts"]:
        failures.append("baseline Llama construction counts changed")

    if after["roundtrip"] != before["roundtrip"]:
        failures.append(f"loading round-trip changed: {after['roundtrip']}")
    if any(after["roundtrip"][key] for key in after["roundtrip"]):
        failures.append(f"loading round-trip reported keys: {after['roundtrip']}")

    # `compare` only walks the after snapshot, so a fixture deleted between the two
    # snapshots would silently drop its coverage -- and `all(...)` over an empty
    # fixture set is vacuously true, so deleting every fixture would pass V5/V6/V7.
    before_fixtures = set(before["templates"]["fixtures"])
    after_fixtures = set(after["templates"]["fixtures"])
    removed = sorted(before_fixtures - after_fixtures)
    if removed:
        failures.append(f"fixtures present in the baseline are missing from the new snapshot: {removed}")
    if not after_fixtures:
        failures.append("no template fixtures to check; V5/V6/V7 would pass vacuously")

    for name, after_entry in after["templates"]["fixtures"].items():
        before_entry = before["templates"]["fixtures"].get(name)
        upstream, redspark = after_entry["upstream"], after_entry["redspark"]
        if redspark is None:
            failures.append(f"[{name}] redspark template missing")
            continue
        if "error" in redspark:
            failures.append(f"[{name}] redspark template failed to render: {redspark['error']}")
            continue
        if redspark["text"] != upstream["text"]:
            failures.append(f"[{name}] redspark rendering differs from upstream")
        if redspark["input_ids"] != upstream["input_ids"]:
            failures.append(f"[{name}] redspark tokenization differs from upstream")
        if redspark["mask_sum"] <= 0:
            failures.append(f"[{name}] redspark assistant mask is empty")
        for problem in mask_semantics_problems(after_entry.get("mask_semantics")):
            failures.append(f"[{name}] mask semantics: {problem}")
        if before_entry is not None and before_entry["redspark"] is not None:
            if redspark["mask"] != before_entry["redspark"]["mask"]:
                failures.append(
                    f"[{name}] assistant mask changed: "
                    f"{_mask_regions(before_entry['redspark']['mask'])} -> {_mask_regions(redspark['mask'])}"
                )
        if before_entry is not None and upstream != before_entry["upstream"]:
            failures.append(f"[{name}] upstream rendering changed")

    equivalence = after.get("equivalence")
    if equivalence is not None:
        if equivalence["max_abs_logit_diff"] != 0.0:
            failures.append(f"logit equivalence broken: max_abs_diff={equivalence['max_abs_logit_diff']}")
        if not equivalence["argmax_equal"]:
            failures.append("argmax mismatch against upstream Llama")

    print("=== before ===")
    print(json.dumps(before["checks"], indent=2))
    print(json.dumps(before_counts))
    print("=== after ===")
    print(json.dumps(after["checks"], indent=2))
    print(json.dumps(after_counts))
    print("=== assistant mask regions (after) ===")
    for name, entry in after["templates"]["fixtures"].items():
        if entry["redspark"] and "mask" in entry["redspark"]:
            print(f"  {name}: {_mask_regions(entry['redspark']['mask'])}")
    print()

    if failures:
        print("FAILURES:")
        for item in failures:
            print(" -", item)
        return 1
    print("ALL COMPARISONS PASSED")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    parser.add_argument(
        "--equivalence",
        action="store_true",
        help="Also load the real checkpoint twice and compare RedSpark logits against upstream Llama.",
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--data",
        type=Path,
        help="Also render every record of this SFT JSONL file and check its assistant spans.",
    )
    args = parser.parse_args()

    if args.compare:
        raise SystemExit(compare(*args.compare))

    data = snapshot()
    if args.equivalence:
        data["equivalence"] = check_equivalence(args.device)
        print(json.dumps(data["equivalence"], indent=2))

    if args.out:
        args.out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.out}")
    print(json.dumps(data["checks"], indent=2))
    print(json.dumps(data["info"], indent=2))
    print("redspark_counts:", json.dumps(data["structural"]["redspark_counts"]))
    print("llama_counts   :", json.dumps(data["structural"]["llama_counts"]))
    print("roundtrip      :", json.dumps(data["roundtrip"]))

    problems = [
        f"[{name}] {problem}"
        for name, entry in data["templates"]["fixtures"].items()
        for problem in mask_semantics_problems(entry.get("mask_semantics"))
    ]
    for problem in problems:
        print(f"mask semantics: {problem}", file=sys.stderr)

    data_problems, data_warnings = check_data_file(args.data) if args.data else ([], [])
    for warning in data_warnings:
        print(f"data: warning: {warning}", file=sys.stderr)
    for problem in data_problems:
        print(f"data: {problem}", file=sys.stderr)
    if args.data and not data_problems:
        print(f"data: {args.data} rendered cleanly")

    failed = sorted(name for name, ok in data["checks"].items() if not ok)
    if failed or data_problems:
        # A check that prints `false` must also fail the run, otherwise the tool is
        # only a gate in `--compare` mode and silently passes in `--out` mode.
        if failed:
            print(f"FAILED: {', '.join(failed)}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
