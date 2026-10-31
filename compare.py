"""Compare evaluation runs side by side.

Reads results/<run>_metrics.json for each run named on the command line and
prints the same tables for all of them, with a column for the change between
each pair of neighbouring runs.

For every pair of runs evaluated on the same data file, it also runs McNemar's
exact test on results/<run>_predictions.jsonl, matching rows by line number:
  b = rows the first run got right and the second got wrong
  c = rows the second run got right and the first got wrong
  p = two-sided exact binomial test of b vs c (n = b + c, probability 0.5)
once for exact-label correctness and once for binary (SAFE vs LEAKED) correctness.

Usage: python compare.py baseline lora_a lora_b
"""

import json
import math
import sys
from itertools import combinations
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# McNemar's exact test
# ---------------------------------------------------------------------------

def mcnemar_counts(correct_1, correct_2):
    """b = rows run 1 got right and run 2 got wrong; c = the reverse."""
    b = sum(1 for x, y in zip(correct_1, correct_2) if x and not y)
    c = sum(1 for x, y in zip(correct_1, correct_2) if y and not x)
    return b, c


def mcnemar_exact_p(b, c):
    """Two-sided exact McNemar p-value: twice the binomial tail of min(b, c), capped at 1."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def is_leaked(label):
    return label.startswith("LEAKED: ")


def binary_correct(gold, parsed):
    """SAFE vs LEAKED only; INVALID is always wrong."""
    return (is_leaked(gold) and is_leaked(parsed)) or (gold == "SAFE" and parsed == "SAFE")


def load_predictions(run):
    path = RESULTS_DIR / f"{run}_predictions.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {row["line"]: row for row in rows}


def print_mcnemar(runs, metrics):
    print("\nMcNemar's exact test (pairs of runs on the same data, rows matched by line number)")
    print(f"  {'run 1':<14}{'run 2':<14}{'correctness':<13}{'b':>6}{'c':>6}{'p':>12}")
    for (run_1, m_1), (run_2, m_2) in combinations(zip(runs, metrics), 2):
        if m_1["config"]["data_sha256"] != m_2["config"]["data_sha256"]:
            continue
        preds_1, preds_2 = load_predictions(run_1), load_predictions(run_2)
        if preds_1 is None or preds_2 is None:
            print(f"  {run_1:<14}{run_2:<14}skipped: a predictions file is missing")
            continue
        lines = sorted(preds_1)
        if lines != sorted(preds_2) or any(preds_1[i]["gold"] != preds_2[i]["gold"] for i in lines):
            sys.exit(f"{run_1} and {run_2} predictions do not cover the same rows")
        for kind, correct in [("exact label", lambda r: r["parsed"] == r["gold"]),
                              ("binary", lambda r: binary_correct(r["gold"], r["parsed"]))]:
            b, c = mcnemar_counts([correct(preds_1[i]) for i in lines], [correct(preds_2[i]) for i in lines])
            print(f"  {run_1:<14}{run_2:<14}{kind:<13}{b:>6}{c:>6}{mcnemar_exact_p(b, c):>12.3g}")


# ---------------------------------------------------------------------------
# Side-by-side tables
# ---------------------------------------------------------------------------

def load_run(run):
    path = RESULTS_DIR / f"{run}_metrics.json"
    if not path.exists():
        sys.exit(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def change(first, second):
    if not isinstance(first, (int, float)) or not isinstance(second, (int, float)):
        return "n/a"
    if isinstance(first, float) or isinstance(second, float):
        return f"{second - first:+.3f}"
    return f"{second - first:+d}"


def print_table(title, rows, runs):
    """rows is a list of (row name, [one value per run])."""
    pairs = list(zip(runs, runs[1:]))
    change_headers = [f"{later}-{earlier}" for earlier, later in pairs]
    width = max(len(name) for name, _ in rows) + 2
    column = max([14] + [len(h) + 2 for h in change_headers])
    print(f"\n{title}")
    print(" " * width + "".join(f"{run:>{column}}" for run in runs) + "".join(f"{h:>{column}}" for h in change_headers))
    for name, values in rows:
        line = f"{name:<{width}}" + "".join(f"{fmt(v):>{column}}" for v in values)
        line += "".join(f"{change(a, b):>{column}}" for a, b in zip(values, values[1:]))
        print(line)


def gap(subsets):
    held_out, seen = subsets["held_out"]["f1"], subsets["seen"]["f1"]
    return None if held_out is None or seen is None else held_out - seen


def main():
    runs = sys.argv[1:]
    if not runs:
        sys.exit(__doc__)
    metrics = [load_run(run) for run in runs]

    print("Runs")
    for run, m in zip(runs, metrics):
        c = m["config"]
        print(f"  {run}: model={c['model']} adapter={c['adapter']} data={c['data']} rows={c['n_rows']} "
              f"data_sha256={c['data_sha256'][:12]} prompt_sha256={c['system_prompt_sha256'][:12]}")
    if len({m["config"]["data_sha256"] for m in metrics}) > 1:
        print("  WARNING: the runs were evaluated on different data files")
    if len({m["config"]["system_prompt_sha256"] for m in metrics}) > 1:
        print("  WARNING: the runs used different system prompts")

    overall_keys = ["precision", "recall", "f1", "accuracy", "binary_accuracy", "type_accuracy",
                    "format_error_rate", "tp", "fp", "fn", "tn", "invalid", "n"]
    print_table("Overall (LEAKED is the positive class)",
                [(key, [m["overall"][key] for m in metrics]) for key in overall_keys], runs)

    confusion_rows = []
    for gold in ["SAFE", "LEAKED"]:
        for pred in ["SAFE", "LEAKED", "INVALID"]:
            confusion_rows.append((f"gold {gold} -> pred {pred}", [m["binary_confusion"][gold][pred] for m in metrics]))
    print_table("Binary confusion", confusion_rows, runs)

    leak_types = list(metrics[0]["per_leak_type"])
    print_table("Recall per leak type (flagged as any LEAKED label)",
                [(t, [m["per_leak_type"][t]["recall"] for m in metrics]) for t in leak_types], runs)
    print_table("Exact type rate per leak type (flagged with the right type)",
                [(t, [m["per_leak_type"][t]["exact_type_rate"] for m in metrics]) for t in leak_types], runs)

    categories = list(metrics[0]["per_safe_category"])
    print_table("False-positive rate per SAFE category",
                [(c, [m["per_safe_category"][c]["false_positive_rate"] for m in metrics]) for c in categories], runs)

    held_rows = []
    for subset in ["held_out", "seen"]:
        for key in ["n", "precision", "recall", "f1", "accuracy", "invalid"]:
            held_rows.append((f"{subset} {key}", [m["held_out_vs_seen"][subset][key] for m in metrics]))
    held_rows.append(("F1 gap (held_out - seen)", [gap(m["held_out_vs_seen"]) for m in metrics]))
    print_table("Held-out vs seen templates", held_rows, runs)

    formats = list(metrics[0]["per_format"])
    print_table("F1 per format", [(f, [m["per_format"][f]["f1"] for m in metrics]) for f in formats], runs)
    print_table("Accuracy per format", [(f, [m["per_format"][f]["accuracy"] for m in metrics]) for f in formats], runs)

    error_rows = [("INVALID, all rows", [m["overall"]["invalid"] for m in metrics]),
                  ("format error rate", [m["overall"]["format_error_rate"] for m in metrics]),
                  ("INVALID, gold SAFE", [m["binary_confusion"]["SAFE"]["INVALID"] for m in metrics]),
                  ("INVALID, gold LEAKED", [m["binary_confusion"]["LEAKED"]["INVALID"] for m in metrics])]
    error_rows += [(f"INVALID, {f}", [m["per_format"][f]["invalid"] for m in metrics]) for f in formats]
    print_table("Format errors", error_rows, runs)

    if len(runs) > 1:
        print_mcnemar(runs, metrics)


if __name__ == "__main__":
    main()
