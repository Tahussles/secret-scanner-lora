"""Integrity checks for the generated dataset.

Reads data/{train,valid,test}.jsonl and the line-aligned data/*_meta.jsonl
files, prints the template inventory and per-split tables, then checks that:
- held-out template_ids never appear in train or valid
- no snippet (whitespace-normalised) appears in two splits
- every assistant message is exactly SAFE or LEAKED: <valid type>
- every system message equals configs/system_prompt.txt
plus a few consistency checks between the data and metadata files.

Exits with status 1 if any check fails.

Usage: python check_data.py
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SYSTEM_PROMPT_PATH = ROOT / "configs" / "system_prompt.txt"
SPLITS = ["train", "valid", "test"]

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
VALID_LABELS = {"SAFE"} | {f"LEAKED: {t}" for t in LEAK_TYPES}

# The labelling rule from generate_data.py: only explicit markers make a value a placeholder.
PLACEHOLDER_MARKERS = re.compile(r"example|dummy|xxxx|your[_-]\S*[_-]here|<[^<>]+>", re.IGNORECASE)

failures = []


def check(ok, message):
    print(f"[{'PASS' if ok else 'FAIL'}] {message}")
    if not ok:
        failures.append(message)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def normalise(snippet):
    return " ".join(snippet.split())


def load_splits():
    splits = {}
    for name in SPLITS:
        rows = read_jsonl(DATA_DIR / f"{name}.jsonl")
        meta = read_jsonl(DATA_DIR / f"{name}_meta.jsonl")
        splits[name] = {"rows": rows, "meta": meta}
    return splits


def print_inventory(splits):
    all_meta = [m for name in SPLITS for m in splits[name]["meta"]]
    template_ids = sorted({m["template_id"] for m in all_meta})
    held_ids = sorted({m["template_id"] for m in all_meta if m["held_out"]})
    print("== Template inventory (from the metadata: templates that produced at least one example)")
    print(f"{len(template_ids)} template_ids, {len(held_ids)} held out")
    print(f"{'category':<20} {'templates':>9}  held-out template_ids")
    for category in LEAK_TYPES + SAFE_CATEGORIES:
        used = sorted({m["template_id"] for m in all_meta if m["category"] == category})
        held = sorted({m["template_id"] for m in all_meta if m["category"] == category and m["held_out"]})
        print(f"{category:<20} {len(used):>9}  {', '.join(held)}")
    print("All held-out template_ids:", ", ".join(held_ids))


def print_split_tables(splits):
    counts = {}
    for name in SPLITS:
        meta = splits[name]["meta"]
        labels = [row["messages"][-1]["content"] for row in splits[name]["rows"]]
        counts[name] = Counter()
        counts[name]["total"] = len(meta)
        counts[name]["SAFE"] = labels.count("SAFE")
        counts[name]["LEAKED"] = len(labels) - labels.count("SAFE")
        for m in meta:
            counts[name]["category:" + m["category"]] += 1
            counts[name]["format:" + m["format"]] += 1
            counts[name]["held-out templates" if m["held_out"] else "seen templates"] += 1

    def row(label, key):
        print(f"{label:<22}" + "".join(f"{counts[name][key]:>8}" for name in SPLITS))

    print("\n== Per-split tables")
    print(f"{'':<22}" + "".join(f"{name:>8}" for name in SPLITS))
    for key in ["total", "SAFE", "LEAKED"]:
        row(key, key)
    print("-- leak types")
    for category in LEAK_TYPES:
        row(category, "category:" + category)
    print("-- SAFE categories")
    for category in SAFE_CATEGORIES:
        row(category, "category:" + category)
    print("-- formats")
    for fmt in FORMATS:
        row(fmt, "format:" + fmt)
    print("-- held-out vs seen templates")
    row("held-out templates", "held-out templates")
    row("seen templates", "seen templates")


def run_checks(splits):
    print("\n== Checks")
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()

    for name in SPLITS:
        rows, meta = splits[name]["rows"], splits[name]["meta"]
        check(len(rows) == len(meta), f"{name}: data and meta files are line-aligned ({len(rows)} vs {len(meta)} lines)")
        bad_roles = [i for i, r in enumerate(rows) if [m["role"] for m in r["messages"]] != ["system", "user", "assistant"]]
        check(not bad_roles, f"{name}: every row is [system, user, assistant] ({len(bad_roles)} bad)")
        bad_system = [i for i, r in enumerate(rows) if r["messages"][0]["content"] != system_prompt]
        check(not bad_system, f"{name}: every system message equals configs/system_prompt.txt ({len(bad_system)} differ)")
        bad_labels = [r["messages"][2]["content"] for r in rows if r["messages"][2]["content"] not in VALID_LABELS]
        check(not bad_labels, f"{name}: every assistant message is SAFE or LEAKED: <valid type> ({len(bad_labels)} invalid)")
        mismatched = [
            i for i, (r, m) in enumerate(zip(rows, meta))
            if r["messages"][2]["content"] != (f"LEAKED: {m['category']}" if m["category"] in LEAK_TYPES else "SAFE")
            or m["category"] not in LEAK_TYPES + SAFE_CATEGORIES
        ]
        check(not mismatched, f"{name}: every label agrees with its metadata category ({len(mismatched)} disagree)")
        bad_length = [i for i, r in enumerate(rows) if not 1 <= len(r["messages"][1]["content"].split("\n")) <= 6]
        check(not bad_length, f"{name}: every snippet is 1 to 6 lines ({len(bad_length)} outside)")
        leaked_with_marker = [
            i for i, (r, m) in enumerate(zip(rows, meta))
            if m["category"] in LEAK_TYPES and PLACEHOLDER_MARKERS.search(r["messages"][1]["content"])
        ]
        check(not leaked_with_marker, f"{name}: no LEAKED snippet contains a placeholder marker ({len(leaked_with_marker)} do)")
        placeholder_without_marker = [
            i for i, (r, m) in enumerate(zip(rows, meta))
            if m["category"] == "placeholder" and not PLACEHOLDER_MARKERS.search(r["messages"][1]["content"])
        ]
        check(not placeholder_without_marker,
              f"{name}: every placeholder snippet contains a marker ({len(placeholder_without_marker)} do not)")

    # A template is either held out everywhere or nowhere.
    flags = {}
    for name in SPLITS:
        for m in splits[name]["meta"]:
            flags.setdefault(m["template_id"], set()).add(m["held_out"])
    inconsistent = sorted(t for t, values in flags.items() if len(values) > 1)
    check(not inconsistent, f"held_out flag is consistent per template_id ({inconsistent})")

    ids = {name: {m["template_id"] for m in splits[name]["meta"]} for name in SPLITS}
    held_ids = {t for t, values in flags.items() if True in values}
    check(not held_ids & ids["train"], f"held-out template_ids in train: {sorted(held_ids & ids['train'])}")
    check(not held_ids & ids["valid"], f"held-out template_ids in valid: {sorted(held_ids & ids['valid'])}")

    texts = {name: [normalise(r["messages"][1]["content"]) for r in splits[name]["rows"]] for name in SPLITS}
    for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        shared = set(texts[a]) & set(texts[b])
        check(not shared, f"identical snippets shared by {a} and {b}: {len(shared)}")
    for name in SPLITS:
        duplicates = len(texts[name]) - len(set(texts[name]))
        check(duplicates == 0, f"duplicate snippets within {name}: {duplicates}")


def main():
    splits = load_splits()
    print_inventory(splits)
    print_split_tables(splits)
    run_checks(splits)
    if failures:
        print(f"\n{len(failures)} check(s) FAILED")
        sys.exit(1)
    print("\nAll checks passed")


if __name__ == "__main__":
    main()
