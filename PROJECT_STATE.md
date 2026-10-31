# PROJECT_STATE

## Current step
Round 2 complete: seeded synthetic dataset generator (`generate_data.py`) and the generated splits. No evaluation or training yet. Waiting for Round 3.

## Done
- Round 1
  - Repo created at `~/secret-scanner-lora`.
  - Python checked: native arm64 (not Rosetta).
  - git initialised on branch `main`, `.gitignore` covers `.venv/`, `data/`, `adapters/`, `__pycache__/`, `.DS_Store`.
  - `.venv` created, `mlx-lm` installed.
  - Skeleton: `configs/`, `results/` (with `.gitkeep`).
  - System prompt v1 saved to `configs/system_prompt.txt`.
  - Base model smoke test on two snippets with v1.
- Round 1b
  - Root commit amended before any push so local-only files are not in history.
  - System prompt v2 written to `configs/system_prompt.txt` and frozen.
  - Base model smoke test rerun on five snippets with v2 (outputs below).
  - Labelling rule for fake values settled (decision 7).
- Round 2
  - `generate_data.py` written: standard library only, one `random.Random(42)`, 90 named templates across 7 formats.
  - Generated `data/{train,valid,test}.jsonl` plus line-aligned `data/*_meta.jsonl` (gitignored, not committed).
  - Verified: held-out templates never in train or valid, 0 identical snippets across or within splits, byte-identical output on two runs, no token-shaped literals in the source.

## Environment
| Item | Value |
|---|---|
| Machine | MacBook Pro M3 Max, 64 GB, macOS 26.5 (arm64) |
| Python | 3.12.16 (Homebrew, arm64) in `.venv` |
| mlx | 0.32.4 |
| mlx-lm | 0.32.2 |
| transformers | 5.22.0 |
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` (bf16 weights from Hugging Face, revision `989aa79`) |

## Decisions
1. Use Homebrew Python 3.12 for the venv. The only Python on PATH was Apple's 3.9.6, and mlx-lm 0.32.2 requires Python 3.11 or newer.
2. Commits use the global git identity (Taha Hussain, GitHub noreply email). No repo-local `user.name` / `user.email` is set.
3. Default branch is `main` (matches GitHub's default for the future remote).
4. System prompt v2 is frozen in `configs/system_prompt.txt` and used verbatim (read with `.strip()`) for every run, base and fine-tuned. It was changed once, from v1, after a 2-snippet smoke test showed a likely SAFE bias (v1 called `db_password = 'hunter2_prod_2024'` SAFE), and before any dataset existed. v1 is kept in the first commit. No further changes, whatever later results show.
5. Greedy decoding means `temp=0.0` (`mlx_lm.sample_utils.make_sampler(temp=0.0)`, which is argmax).
6. Model weights downloaded manually and SHA-256 verified; run all inference with `HF_HUB_OFFLINE=1`. Python's Hugging Face downloader failed on this network (Xet transfer error, then repeated DNS errors with Xet disabled), so `model.safetensors` (3,087,467,144 bytes) was fetched with `curl` into the standard Hugging Face cache. Its SHA-256 matches the Hub's published hash (`dd924a11...c6d3ee`).
7. Only explicit placeholder markers count as fake/SAFE: values containing EXAMPLE, dummy, xxxx, your_..._here, or <...>. Realistic fake values are labelled LEAKED.
8. The user message is the raw snippet with no wrapper. The system message is read at runtime from `configs/system_prompt.txt` (never duplicated in code). The assistant message is exactly `SAFE` or `LEAKED: <type>`.
9. Labelling precedence: a connection string with an embedded password is `database_url`, not `password`. A private key block is `private_key`; a public key or certificate is SAFE (`public_key`). Placeholder markers are matched case-insensitively. Stripe test-mode keys are real credentials, so they are LEAKED.
10. Template hold-out scheme: 90 templates, 22 held out (24.4%), chosen by hand so every leak type and every SAFE category has at least one held-out and at least one training template (the generator asserts this). Held-out examples go only to test. Per category the split is 70/15/15, and the test share is half held-out-template examples and half seen-template examples with fresh random values.
11. Metadata (`template_id`, `category`, `format`, `held_out`) lives in separate line-aligned `data/*_meta.jsonl` files, so the main JSONL files stay pure mlx-lm chat format.
12. Value-slot templates: the same skeleton produces LEAKED or SAFE depending on what fills `{value}` (fake secret, placeholder, redacted, empty, passwordless DB URL, public key). Structural SAFE cases (env reads, secret manager calls, hashes, UUIDs, ...) have their own templates.
13. Deduplication is by whitespace-normalised snippet text across all splits. A duplicate is discarded and regenerated, so every category hits its target count exactly (10 duplicates were regenerated).
14. Dataset size: 167 examples per leak type and 115 per SAFE category (1,503 LEAKED, 1,495 SAFE, 2,998 total).

## Open issues
- No remote yet, so nothing is pushed.
- Package versions are recorded here but there is no `requirements.txt` yet.
- Test format mix differs from train because several held-out templates are JSON, YAML or JS (JSON is 17% of test vs 8% of train). Keep this in mind when reading per-format test results.
- The system prompt is repeated in all 2,998 examples. mlx-lm's `--mask-prompt` option is off by default, so `lora_a` (defaults) will also train on prompt tokens. A candidate setting for `lora_b`.

## Dataset (seed 42, `python generate_data.py`)
| Split | Total | SAFE | LEAKED | Per leak type | Per SAFE category | Held-out-template rows |
|---|---|---|---|---|---|---|
| train | 2,093 | 1,040 | 1,053 | 117 | 80 | 0 |
| valid | 446 | 221 | 225 | 25 | 17 | 0 |
| test | 459 | 234 | 225 | 25 | 18 | 225 (seen-template rows: 234) |

Formats, train: python 551, yaml 369, shell 316, js_ts 278, env 249, json 176, dockerfile 154.
Formats, test: yaml 80, js_ts 78, json 78, python 74, shell 72, dockerfile 41, env 36.

## Latest metrics
None yet (no evaluation run). Smoke test with prompt v2 (greedy `temp=0.0`, `max_tokens=64`, `HF_HUB_OFFLINE=1`, ran twice with identical outputs):

| Snippet | Expected | Raw output (repr) | Correct |
|---|---|---|---|
| a) `db_password = 'hunter2_prod_2024'` | `LEAKED: password` | `'LEAKED: password'` | yes |
| b) `db_password = os.getenv("DB_PASS")` | `SAFE` | `'SAFE'` | yes |
| c) `STRIPE_KEY = "YOUR_STRIPE_KEY_HERE"` | `SAFE` | `'LEAKED: stripe_key'` | no |
| d) `api_token: "****"` | `SAFE` | `'SAFE'` | yes |
| e) `DATABASE_URL=postgres://admin:s3cretPass@db.internal:5432/app` | `LEAKED: database_url` | `'SAFE'` | no |

3 of 5 correct, 0 format errors. For reference, v1 in Round 1 gave `'SAFE'` for both a) and b).
