"""Compare evaluation runs side by side.

Reads results/<run>_metrics.json for each run named on the command line and
prints the same tables for all of them. With exactly two runs, a "change"
column shows the second run minus the first.

Usage: python compare.py baseline lora_a
"""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


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


def change(values):
    first, second = values
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return f"{second - first:+.3f}" if isinstance(first, float) or isinstance(second, float) else f"{second - first:+d}"
    return "n/a"


def print_table(title, rows, runs):
    """rows is a list of (row name, [one value per run])."""
    show_change = len(runs) == 2
    width = max(len(name) for name, _ in rows) + 2
    print(f"\n{title}")
    print(" " * width + "".join(f"{run:>14}" for run in runs) + (f"{'change':>10}" if show_change else ""))
    for name, values in rows:
        line = f"{name:<{width}}" + "".join(f"{fmt(v):>14}" for v in values)
        if show_change:
            line += f"{change(values):>10}"
        print(line)


def gap(subsets):
    held_out, seen = subsets["held_out"]["f1"], subsets["seen"]["f1"]
    return None if held_out is None or seen is None else held_out - seen


def main():
    runs = sys.argv[1:]
    if len(runs) < 2:
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


if __name__ == "__main__":
    main()
