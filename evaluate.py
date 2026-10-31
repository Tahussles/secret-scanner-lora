"""Evaluate a model (optionally with a LoRA adapter) on the secret-scanner data.

For each row the model sees the row's system and user messages (the user message
is the raw snippet) and generates with greedy decoding. Its output is parsed
strictly: after stripping leading and trailing whitespace it must be exactly
"SAFE" or "LEAKED: <type>" with <type> one of the nine leak types. Anything else
(wrong case, extra text, extra lines, trailing punctuation, unknown type) is
INVALID. There is no other normalisation.

LEAKED is the positive class. An INVALID output is never a positive prediction:
it is a false negative when the gold label is LEAKED, and simply wrong when the
gold label is SAFE. It always counts as wrong for accuracy.

Writes:
  results/<run>_metrics.json       metrics and run config (committed)
  results/<run>_predictions.jsonl  every row: raw output and parsed label (gitignored)
  results/<run>_errors.jsonl       misclassified rows, with the snippet (gitignored)

Usage:
  python evaluate.py --run-name baseline
  python evaluate.py --run-name lora_a --adapter adapters/lora_a
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = ROOT / "configs" / "system_prompt.txt"
RESULTS_DIR = ROOT / "results"

LEAK_TYPES = [
    "aws_access_key", "github_token", "stripe_key", "slack_token", "private_key",
    "jwt", "database_url", "password", "generic_api_key",
]
SAFE_CATEGORIES = [
    "env_var", "secret_manager", "placeholder", "redacted", "hash", "uuid",
    "commit_sha", "public_key", "base64_non_secret", "empty_value",
    "keyword_non_secret", "db_url_no_secret", "plain_code",
]
FORMATS = ["python", "js_ts", "yaml", "env", "json", "shell", "dockerfile"]

SAFE = "SAFE"
INVALID = "INVALID"
VALID_LABELS = [SAFE] + [f"LEAKED: {t}" for t in LEAK_TYPES]
MAX_TOKENS = 32


# ---------------------------------------------------------------------------
# Parsing and metrics (no MLX needed, so test_evaluate.py can test these alone)
# ---------------------------------------------------------------------------

def parse_output(raw):
    """Strip surrounding whitespace, then require an exact valid label."""
    text = raw.strip()
    return text if text in VALID_LABELS else INVALID


def is_leaked(label):
    return label.startswith("LEAKED: ")


def ratio(numerator, denominator):
    """A rate, or None when it is undefined (nothing to divide by)."""
    return numerator / denominator if denominator else None


def compute_metrics(golds, preds):
    """Headline metrics for one set of rows. LEAKED is the positive class."""
    pairs = list(zip(golds, preds))
    tp = sum(1 for g, p in pairs if is_leaked(g) and is_leaked(p))
    fp = sum(1 for g, p in pairs if g == SAFE and is_leaked(p))
    fn = sum(1 for g, p in pairs if is_leaked(g) and not is_leaked(p))  # includes INVALID
    tn = sum(1 for g, p in pairs if g == SAFE and p == SAFE)
    invalid = sum(1 for _, p in pairs if p == INVALID)
    both_leaked = [(g, p) for g, p in pairs if is_leaked(g) and is_leaked(p)]

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "n": len(pairs),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "invalid": invalid,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": ratio(sum(1 for g, p in pairs if g == p), len(pairs)),  # exact label, so a wrong type is wrong
        "binary_accuracy": ratio(tp + tn, len(pairs)),  # SAFE vs LEAKED only; INVALID is always wrong
        "type_accuracy": ratio(sum(1 for g, p in both_leaked if g == p), len(both_leaked)),
        "format_error_rate": ratio(invalid, len(pairs)),
    }


def binary_confusion(golds, preds):
    """Gold SAFE/LEAKED by predicted SAFE/LEAKED/INVALID."""
    matrix = {gold: {SAFE: 0, "LEAKED": 0, INVALID: 0} for gold in [SAFE, "LEAKED"]}
    for g, p in zip(golds, preds):
        matrix["LEAKED" if is_leaked(g) else SAFE]["LEAKED" if is_leaked(p) else p] += 1
    return matrix


def label_confusion(golds, preds):
    """Gold label by predicted label, including INVALID."""
    matrix = {gold: {pred: 0 for pred in VALID_LABELS + [INVALID]} for gold in VALID_LABELS}
    for g, p in zip(golds, preds):
        matrix[g][p] += 1
    return matrix


def subset_metrics(golds, preds, meta, keep):
    rows = [(g, p) for g, p, m in zip(golds, preds, meta) if keep(m)]
    return compute_metrics([g for g, _ in rows], [p for _, p in rows])


def build_metrics(golds, preds, meta):
    per_leak_type = {}
    for leak_type in LEAK_TYPES:
        rows = [p for g, p in zip(golds, preds) if g == f"LEAKED: {leak_type}"]
        flagged = sum(1 for p in rows if is_leaked(p))
        exact = sum(1 for p in rows if p == f"LEAKED: {leak_type}")
        per_leak_type[leak_type] = {
            "n": len(rows), "flagged": flagged, "recall": ratio(flagged, len(rows)),
            "exact_type": exact, "exact_type_rate": ratio(exact, len(rows)),
            "invalid": rows.count(INVALID),
        }

    per_safe_category = {}
    for category in SAFE_CATEGORIES:
        rows = [p for p, m in zip(preds, meta) if m["category"] == category]
        false_positives = sum(1 for p in rows if is_leaked(p))
        per_safe_category[category] = {
            "n": len(rows), "false_positives": false_positives,
            "false_positive_rate": ratio(false_positives, len(rows)), "invalid": rows.count(INVALID),
        }

    return {
        "overall": compute_metrics(golds, preds),
        "binary_confusion": binary_confusion(golds, preds),
        "label_confusion": label_confusion(golds, preds),
        "per_leak_type": per_leak_type,
        "per_safe_category": per_safe_category,
        "held_out_vs_seen": {
            "held_out": subset_metrics(golds, preds, meta, lambda m: m["held_out"]),
            "seen": subset_metrics(golds, preds, meta, lambda m: not m["held_out"]),
        },
        "per_format": {fmt: subset_metrics(golds, preds, meta, lambda m, fmt=fmt: m["format"] == fmt)
                       for fmt in FORMATS},
    }


# ---------------------------------------------------------------------------
# Running the model
# ---------------------------------------------------------------------------

def generate_outputs(model_name, adapter, rows):
    # Imported here so the parsing and metric functions above can be tested without MLX.
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_name, adapter_path=adapter)
    greedy = make_sampler(temp=0.0)  # in mlx-lm 0.32.2, temp 0 returns its argmax sampler
    outputs = []
    start = time.time()
    for i, row in enumerate(rows, 1):
        prompt = tokenizer.apply_chat_template(row["messages"][:2], add_generation_prompt=True)
        outputs.append(generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS, sampler=greedy))
        if i % 50 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} rows, {time.time() - start:.0f}s", flush=True)
    return outputs


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def fmt(value):
    return "n/a" if value is None else f"{value:.3f}"


SHORT_LABELS = {"SAFE": "SAFE", "INVALID": "INVALID"}
SHORT_LABELS.update({f"LEAKED: {t}": t.split("_")[0] for t in LEAK_TYPES})
SHORT_LABELS["LEAKED: private_key"] = "pkey"
SHORT_LABELS["LEAKED: database_url"] = "db_url"


def print_metric_rows(title, subsets):
    print(f"\n{title}")
    print(f"  {'':<18}{'n':>5}{'prec':>8}{'recall':>8}{'f1':>8}{'acc':>8}{'bin_acc':>9}{'type_acc':>10}{'invalid':>9}")
    for name, m in subsets.items():
        print(f"  {name:<18}{m['n']:>5}{fmt(m['precision']):>8}{fmt(m['recall']):>8}{fmt(m['f1']):>8}"
              f"{fmt(m['accuracy']):>8}{fmt(m['binary_accuracy']):>9}{fmt(m['type_accuracy']):>10}{m['invalid']:>9}")


def print_summary(metrics, config):
    o = metrics["overall"]
    print(f"\n== {config['run_name']}: {o['n']} rows")
    print(f"precision {fmt(o['precision'])}  recall {fmt(o['recall'])}  F1 {fmt(o['f1'])}  "
          f"accuracy {fmt(o['accuracy'])}  binary accuracy {fmt(o['binary_accuracy'])}  "
          f"type accuracy {fmt(o['type_accuracy'])}  format error rate {fmt(o['format_error_rate'])} ({o['invalid']} INVALID)")

    print("\nBinary confusion (rows: gold, columns: predicted)")
    print(f"  {'':<8}{'SAFE':>8}{'LEAKED':>8}{'INVALID':>9}")
    for gold, row in metrics["binary_confusion"].items():
        print(f"  {gold:<8}{row[SAFE]:>8}{row['LEAKED']:>8}{row[INVALID]:>9}")

    columns = VALID_LABELS + [INVALID]
    print("\nLabel confusion (rows: gold, columns: predicted; leak types shortened in the header)")
    print(f"  {'':<25}" + "".join(f"{SHORT_LABELS[c]:>9}" for c in columns))
    for gold, row in metrics["label_confusion"].items():
        print(f"  {gold:<25}" + "".join(f"{row[c]:>9}" for c in columns))

    print("\nRecall per leak type (flagged = predicted any LEAKED label)")
    print(f"  {'':<18}{'n':>5}{'flagged':>9}{'recall':>8}{'exact type':>12}{'invalid':>9}")
    for leak_type, m in metrics["per_leak_type"].items():
        print(f"  {leak_type:<18}{m['n']:>5}{m['flagged']:>9}{fmt(m['recall']):>8}{m['exact_type']:>12}{m['invalid']:>9}")

    print("\nFalse-positive rate per SAFE category")
    print(f"  {'':<20}{'n':>5}{'false pos':>11}{'FPR':>8}{'invalid':>9}")
    for category, m in metrics["per_safe_category"].items():
        print(f"  {category:<20}{m['n']:>5}{m['false_positives']:>11}{fmt(m['false_positive_rate']):>8}{m['invalid']:>9}")

    print_metric_rows("Held-out vs seen templates", metrics["held_out_vs_seen"])
    print_metric_rows("Per format", metrics["per_format"])
    print(f"\nRuntime: {config['runtime_seconds']:.1f}s total, {config['generation_seconds']:.1f}s generating")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default=None, help="path to a LoRA adapter directory")
    parser.add_argument("--data", default="data/test.jsonl")
    parser.add_argument("--meta", default="data/test_meta.jsonl")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit", type=int, default=None, help="only evaluate the first N rows")
    return parser.parse_args()


def main():
    args = parse_args()
    # The model weights are already cached; never contact the Hugging Face Hub (decision 6).
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    start = time.time()

    rows, meta = read_jsonl(args.data), read_jsonl(args.meta)
    if len(rows) != len(meta):
        sys.exit(f"{args.data} has {len(rows)} lines but {args.meta} has {len(meta)}")
    if args.limit:
        rows, meta = rows[:args.limit], meta[:args.limit]
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    if any(row["messages"][0]["content"] != system_prompt for row in rows):
        sys.exit("a system message in the data differs from configs/system_prompt.txt")

    print(f"Evaluating {args.model} (adapter: {args.adapter}) on {len(rows)} rows of {args.data}")
    generation_start = time.time()
    raw_outputs = generate_outputs(args.model, args.adapter, rows)
    generation_seconds = time.time() - generation_start

    golds = [row["messages"][2]["content"] for row in rows]
    preds = [parse_output(raw) for raw in raw_outputs]
    metrics = build_metrics(golds, preds, meta)
    config = {
        "run_name": args.run_name,
        "model": args.model,
        "adapter": args.adapter,
        "data": args.data,
        "meta": args.meta,
        "limit": args.limit,
        "n_rows": len(rows),
        "max_tokens": MAX_TOKENS,
        "decoding": "greedy: mlx_lm.sample_utils.make_sampler(temp=0.0)",
        "system_prompt_sha256": sha256_of(SYSTEM_PROMPT_PATH),
        "data_sha256": sha256_of(args.data),
        "meta_sha256": sha256_of(args.meta),
        "mlx_version": version("mlx"),
        "mlx_lm_version": version("mlx-lm"),
        "python_version": platform.python_version(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generation_seconds": round(generation_seconds, 1),
        "runtime_seconds": round(time.time() - start, 1),
    }

    predictions = [
        {"line": i, "meta": m, "gold": g, "raw_output": raw, "parsed": p}
        for i, (m, g, raw, p) in enumerate(zip(meta, golds, raw_outputs, preds), 1)
    ]
    errors = [dict(record, snippet=row["messages"][1]["content"])
              for record, row in zip(predictions, rows) if record["parsed"] != record["gold"]]

    RESULTS_DIR.mkdir(exist_ok=True)
    metrics_path = RESULTS_DIR / f"{args.run_name}_metrics.json"
    metrics_path.write_text(json.dumps({"config": config, **metrics}, indent=2) + "\n", encoding="utf-8")
    write_jsonl(RESULTS_DIR / f"{args.run_name}_predictions.jsonl", predictions)
    write_jsonl(RESULTS_DIR / f"{args.run_name}_errors.jsonl", errors)

    print_summary(metrics, config)
    print(f"Wrote {metrics_path.relative_to(ROOT)}, "
          f"results/{args.run_name}_predictions.jsonl, results/{args.run_name}_errors.jsonl ({len(errors)} errors)")


if __name__ == "__main__":
    main()
