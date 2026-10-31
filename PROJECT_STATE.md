# PROJECT_STATE

## Current step
Application email. Round 7 complete: the repo is public at https://github.com/Tahussles/secret-scanner-lora. Final model `lora_b` (decision 30): test F1 0.973 (baseline 0.189, lora_a 0.885), valid F1 0.998 (lora_a 0.954, McNemar p = 3.8e-6).

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
- Round 2b
  - `check_data.py` added (standard library only): prints the template inventory and per-split tables from `data/*.jsonl` + `data/*_meta.jsonl`, runs 33 checks, exits 1 if any fails. Verified the failure path on a corrupted copy of the data (3 FAIL lines, exit code 1).
  - Label audit: one train example per category (22 rows), contrast pairs and held-out vs train skeletons reviewed by eye. A rule-based audit of all 2,998 rows (token shapes in SAFE rows, literal passwords in SAFE URLs, password vs database_url precedence, private vs public key blocks) found 0 problems.
  - Generator rerun: all 6 data files byte-identical to the Round 2 files. `generate_data.py` unchanged.
- Round 3
  - `check_data.py` gained two checks: zero U+2022 (bullet) characters in the six data files (raw or JSON-escaped), and a complete value for every typed leak (167/167 for each of AWS, GitHub, Stripe, Slack, JWT, private key). Both verified to fail on a bullet-masked copy.
  - `evaluate.py`: strict parser, metrics, confusion matrices, breakdowns by leak type, SAFE category, held-out vs seen and format; writes `results/<run>_metrics.json`, `_predictions.jsonl`, `_errors.jsonl`.
  - `test_evaluate.py` (unittest, 6 tests, all pass): parser accepts all gold labels and rejects near misses; gold-as-prediction gives F1 = accuracy = 1.0; a hand-made 8-row case gives the exact expected precision 2/3, recall 1/2, F1 4/7.
  - Baseline: two `--limit 50` runs gave byte-identical predictions; full run on 459 test rows in 52.4 s.
- Round 4
  - `configs/lora_a.yaml`: every mlx-lm 0.32.2 training default written out, each with its source location. mlx-lm's own merge logic confirmed no default is missing and only `model`, `data`, `adapter_path` and `train: true` differ in effect.
  - Trained with `HF_HUB_OFFLINE=1 mlx_lm.lora -c configs/lora_a.yaml` (no `--test`): 1,000 iterations in 780 s. Log in `results/lora_a_train_log.txt`.
  - Evaluated the final adapter on test (`lora_a`) and valid (`lora_a_valid`). The adapter loaded: 173 of 459 test predictions differ from baseline.
  - `compare.py` added: side-by-side tables from `results/*_metrics.json`.
- Round 5
  - `configs/lora_b.yaml` = `configs/lora_a.yaml` with `mask_prompt: true` and `adapter_path: adapters/lora_b` (plus header comments). mlx-lm's merge logic confirms only those two settings differ.
  - Verified masking in the installed source (tuner/datasets.py ChatDataset.process, lines 59-79; tuner/trainer.py iterate_batches lines 147-172 and default_loss lines 88-101) and by decoding the loss tokens of two training rows. Found one padding target per row (decision 27); kept mlx-lm's behaviour unchanged and documented it.
  - Trained with `HF_HUB_OFFLINE=1 /usr/bin/time -l mlx_lm.lora -c configs/lora_b.yaml`: 1,000 iterations, 806 s, peak memory footprint 11.15 GB. Log in `results/lora_b_train_log.txt`.
  - Evaluated the final adapter on test (`lora_b`) and valid (`lora_b_valid`); 39 of 459 test predictions differ from `lora_a`.
  - `compare.py` now takes any number of runs and runs McNemar's exact test for every pair of runs on the same data (exact-label and binary correctness). `test_evaluate.py` has 3 McNemar tests (9 tests in total, all pass).
  - Tested the env-read shortcut hypothesis (marker counts in train, scratch-only ablation on test copies); result under Findings.
- Round 6
  - Follow-up ablation (V1: `secretAccessKey: secretKey`; V2: V1 plus `region: process.env.AWS_REGION`) on the same 12 `ts_s3_client` rows: the trigger is the `secretAccessKey` field, not the env read (Findings).
  - `requirements.txt` with exact pins from the venv (mlx, mlx-metal, mlx-lm, transformers and their runtime dependencies), each checked against the installed version.
  - `README.md` written. Every number in it was cross-checked by a script against `results/*_metrics.json`, the training logs, the configs, the data files and recomputed McNemar tests: 0 mismatches, 0 unverified numbers. The ablation table is the only content sourced from scratch runs.
  - Pre-publish checks on the full history: file list, authors, secret patterns, no em dashes; tests and data checks pass.
- Round 7
  - README wording: the result-source note now names both `compare.py` commands, and the "What I learned" findings are reframed (masking as the one training-setup change; the remaining misses as a training-data gap, with the lesson that synthetic data must cover the labelling policy's edge cases).
  - Local cleanup: expired the reflog and ran `git gc --prune=now`, so the two pre-amend Round 1b commits no longer exist locally. `data/` and `adapters/` untouched.
  - Final pre-push checks on the full history passed (21 files, one author and committer, no secret patterns, no em dashes, 9 tests and 40 data checks pass).
  - Published as a public repo: https://github.com/Tahussles/secret-scanner-lora, with topics lora, mlx, fine-tuning, llm-evaluation, secret-detection, synthetic-data. GitHub push protection accepted the push. The remote tree matches `git ls-files` exactly.

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
15. Run `python check_data.py` after every regeneration of the dataset. It reads only the output files (it does not import `generate_data.py`), so its inventory counts templates that actually produced examples. That is why placeholder shows 44 templates there vs 45 defined: held-out `yaml_integration_key` is defined for placeholder but the 9 held-out placeholder slots went to the other held-out templates.
16. Parsing rule: strip leading and trailing whitespace only, then the output must be exactly one of the 10 valid labels (`SAFE` or `LEAKED: <type>` with one of the 9 types). Anything else (wrong case, extra text or lines, trailing punctuation, unknown type) is INVALID. No other normalisation.
17. INVALID handling: an INVALID output is never a positive prediction. It is a false negative when gold is LEAKED, and an incorrect answer (neither a false positive nor a true negative) when gold is SAFE. It always counts as wrong for accuracy and binary accuracy. Rates with a zero denominator are reported as null, not 0.
18. Committed result files: only `results/<run>_metrics.json` (aggregate metrics plus run config: model, adapter, max_tokens, decoding, sha256 of the system prompt and data files, mlx/mlx-lm versions, timestamp, runtime). `results/*_predictions.jsonl` and `results/*_errors.jsonl` are gitignored because they contain snippets and raw outputs with token-shaped strings that GitHub push protection would block.
19. Evaluation decoding: the chat template is applied to [system, user] with `add_generation_prompt=True`; greedy via `make_sampler(temp=0.0)` (checked in the mlx-lm 0.32.2 source: temp 0 returns its argmax sampler); `max_tokens=32`. `evaluate.py` sets `HF_HUB_OFFLINE=1` unless it is already set.
20. Secret-completeness patterns: the given patterns were widened only where the generator legitimately produces other shapes: AWS also accepts the `ASIA` prefix (temporary STS key IDs, 85 of 167), GitHub also accepts fine-grained `github_pat_` tokens (35 of 167), and private key blocks may be separated by literal `\n` escapes (JSON, .env), have indented body lines (YAML) or start after other text on the BEGIN line (Python string). Stripe, Slack and JWT patterns are unchanged.
21. Before every commit, grep all staged files for the six secret patterns and confirm no matches.
22. The final adapter is always the one written after the last iteration (`adapters/<run>/adapters.safetensors`; for `lora_a` and `lora_b` its sha256 equals the iteration-1000 checkpoint). No checkpoint is ever picked by test performance.
23. The validation set is used for decisions (choosing the `lora_b` change, any later tuning); the test set is used only for reporting.
24. Training runs use `mlx_lm.lora -c configs/<run>.yaml`, where the config lists every mlx-lm default explicitly (`python -m mlx_lm.lora` is deprecated in 0.32.2). `configs/lora_b.yaml` differs from `configs/lora_a.yaml` in exactly two settings: `mask_prompt` (the experiment) and `adapter_path` (its own output directory), plus header comments.
25. Training logs are saved with rich's ANSI colour codes stripped (text otherwise identical; the raw log is kept outside the repo). mlx-lm 0.32.2 computes peak memory but only passes it to `report_to` callbacks, so it is not in the log; it was measured separately with a 100-iteration replica of the `lora_a` config (same losses at iterations 10, 50 and 100): 10.10 GB mlx peak memory, 11.05 GB macOS peak memory footprint.
26. `lora_b` = `lora_a` with exactly one change: `mask_prompt: true` (loss on the answer tokens only). Chosen after Round 4 from the training log and valid results only: with the default, the answer label is 2.0% of the loss tokens (5.5 of 273.5 per example), train loss flattened at about 0.5 from iteration 20, and valid `database_url` recall was 0.48. `lora_b` is compared with `lora_a` on valid F1, not validation loss, because masked and unmasked losses cover different tokens.
27. mlx-lm 0.32.2 includes one padding target per row in the training loss: batches are zero-padded with at least one extra column (tuner/trainer.py lines 160 and 163) and `default_loss` keeps target positions up to and including the sequence length (line 95), so the target after the last real token, token id 0 (`!`), is counted. This affects both runs: 1 of about 273.5 loss targets per row in `lora_a` (0.4%), 1 of about 6.5 in `lora_b` (2,093 of 13,637 targets, 15.3%). Generation stops at `<|im_end|>`, so it cannot change an answer. I documented it instead of patching mlx-lm, so both runs use the library as shipped and still differ by one setting.
28. Paired significance: McNemar's exact test on per-row correctness (rows matched by line number), two-sided, `p = min(1, 2 * sum_{k<=min(b,c)} C(n,k) / 2^n)` with `n = b + c`, computed for exact-label and for binary (SAFE vs LEAKED) correctness. Valid results decide; test results are reported.
29. From `lora_b` on, training runs under `/usr/bin/time -l` so the log records the macOS peak memory footprint. Under `mask_prompt: true` the log's `tok/s` and `tokens` columns count loss tokens only (tuner/trainer.py lines 98 and 344), so they are not processing throughput; compare runs by wall-clock time instead.
30. The final model is `lora_b` (`configs/lora_b.yaml`, adapter `adapters/lora_b/adapters.safetensors`). Chosen after Round 5 on valid results: F1 0.998 vs 0.954 for `lora_a`, with McNemar b = 0, c = 19, p = 3.8e-6. Test results (F1 0.973, held-out F1 0.941) are reported, not used to choose. The write-up states its known weaknesses: the 12 held-out `ts_s3_client` misses (the `secretAccessKey`-field shortcut, see Findings), the 3 INVALID outputs on SSH public keys, and that valid contains only seen templates.
31. Remote: `origin` = https://github.com/Tahussles/secret-scanner-lora (public, branch `main`), and every commit is pushed right after it is made. Pushes use the GitHub CLI's token (`gh auth git-credential`), verified to belong to Tahussles (user id 173319783, the same id as the commit email). `gh auth status` shows an outdated local account label, and no global git credential helper is configured, so the helper is passed per command rather than relying on the keychain.

## Open issues
- Test format mix differs from train because several held-out templates are JSON, YAML or JS (JSON is 17% of test vs 8% of train). Keep this in mind when reading per-format test results.
- Held-out generalisation gap (test, reporting only, decision 23): `lora_a` F1 0.785 held-out vs 0.965 seen; `lora_b` 0.941 vs 1.000. All 12 `lora_b` false negatives are the held-out `ts_s3_client` template (see Findings), and its 3 INVALID outputs are on the held-out `yaml_cloud_init_ssh` template (invented types `LEAKED: ssh_key`, `LEAKED: ssh_authorized_keys` for SSH public keys).
- `lora_b`'s test precision of 1.000 relies on decision 17: its 3 INVALID outputs on SAFE rows all start with `LEAKED:`, so a lenient parser would count them as false positives (precision 213/216 = 0.986).
- Cosmetic, not a labelling issue: DB host names are drawn independently of the URL scheme (for example `postgresql+psycopg2://...@mysql.staging.local:5432/app`), and a passwordless URL can look like `postgresql://postgres:5432/billing` (host `postgres`, port 5432, no user). Labels are correct; only realism is affected.

## Findings
- Shortcut behind the `ts_s3_client` misses (Rounds 5 and 6). All 12 `lora_b` false negatives (and 12 of `lora_a`'s 45) are this held-out JS/TS template, which hardcodes an AWS access key ID next to `secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY`.
  - Hypothesis tested first: "env-var reads appear only in SAFE training rows, so the model learned 'env read present means SAFE'."
  - Train counts (snippets containing any of `process.env`, `os.getenv`, `os.environ`, `ENV[`, `getenv(`, `${`, `$(`): 13 LEAKED (1.2% of LEAKED rows, all `js_slack_bolt`, where a Slack token sits next to `signingSecret: process.env...`) vs 129 SAFE (12.4% of SAFE rows). Env reads are not exclusive to SAFE rows, but they are about 10 times more common there. None of the 117 LEAKED `aws_access_key` training rows contains the word "secret".
  - Ablations (scratch copies, test files unchanged; same prompt and greedy decoding through `evaluate.py`), rows flagged as LEAKED out of 12, same for `lora_a` and `lora_b`: original 0/12; `secretAccessKey` field deleted 12/12 (Round 5); V1 `secretAccessKey: secretKey` (no env read) 0/12; V2 = V1 plus `region: process.env.AWS_REGION` 0/12 (Round 6).
  - Conclusion: the env read is not the trigger. Keeping the field with a plain variable still gives SAFE, and only deleting the `secretAccessKey` field flips the answer, so the trigger is a non-literal `secretAccessKey` field next to the hardcoded key, a combination that never occurs in training. V2 cannot test whether an env read elsewhere matters, because V1 is already SAFE. This supersedes the Round 5 conclusion, which credited the env read before the two were separated.

## Dataset (seed 42, `python generate_data.py`)
| Split | Total | SAFE | LEAKED | Per leak type | Per SAFE category | Held-out-template rows |
|---|---|---|---|---|---|---|
| train | 2,093 | 1,040 | 1,053 | 117 | 80 | 0 |
| valid | 446 | 221 | 225 | 25 | 17 | 0 |
| test | 459 | 234 | 225 | 25 | 18 | 225 (seen-template rows: 234) |

Formats, train: python 551, yaml 369, shell 316, js_ts 278, env 249, json 176, dockerfile 154.
Formats, test: yaml 80, js_ts 78, json 78, python 74, shell 72, dockerfile 41, env 36.

## Latest metrics
Test set (`data/test.jsonl`, 459 rows), greedy, `max_tokens=32`. Full numbers in `results/<run>_metrics.json`; side by side with `python compare.py baseline lora_a lora_b`.

| Run | Precision | Recall | F1 | Accuracy | Binary acc | Type acc | INVALID | Held-out F1 | Seen F1 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.625 | 0.111 | 0.189 | 0.525 | 0.525 | 1.000 | 6 | 0.130 | 0.239 |
| lora_a | 0.989 | 0.800 | 0.885 | 0.895 | 0.898 | 0.994 | 0 | 0.785 | 0.965 |
| lora_b | 1.000 | 0.947 | 0.973 | 0.967 | 0.967 | 1.000 | 3 | 0.941 | 1.000 |

Valid set (`data/valid.jsonl`, 446 rows, seen templates only; used for decisions):

| Run | Precision | Recall | F1 | Accuracy | INVALID |
|---|---|---|---|---|---|
| lora_a_valid | 0.981 | 0.929 | 0.954 | 0.955 | 0 |
| lora_b_valid | 0.996 | 1.000 | 0.998 | 0.998 | 0 |

McNemar's exact test (b = first run right and second wrong, c = the reverse):

| Data | Pair | Exact label b / c / p | Binary b / c / p |
|---|---|---|---|
| valid | lora_a vs lora_b | 0 / 19 / 3.81e-06 | 0 / 19 / 3.81e-06 |
| test | lora_a vs lora_b | 3 / 36 / 3.61e-08 | 3 / 35 / 6.68e-08 |
| test | baseline vs lora_a | 1 / 171 / 5.78e-50 | 1 / 172 / 2.91e-50 |
| test | baseline vs lora_b | 3 / 206 / 3.7e-57 | 3 / 206 / 3.7e-57 |

`lora_b` on test: binary confusion SAFE 231 / 0 / 3 (SAFE / LEAKED / INVALID), LEAKED 12 / 213 / 0. Recall 1.000 for every leak type except AWS (0.520, all 12 misses in held-out `ts_s3_client`). No false positives.

`lora_b` training (answer-token loss, not comparable to `lora_a`'s loss): 1,000 iterations, 806 s wall clock, peak memory footprint 11.15 GB (`/usr/bin/time -l`), maximum resident set size 3.49 GB. Validation loss: 4.413 (iter 1), 0.051 (200), 0.017 (400), 0.010 (600), 0.011 (800), 0.000 (1000). Train loss 1.086 at iteration 10, 0.104 at 100, mostly below 0.01 from iteration 580 with occasional spikes (0.024 at 830, 0.021 at 870), 0.000 at 1000.

`lora_a` on test: binary confusion SAFE 232 / 2 / 0 (SAFE / LEAKED / INVALID), LEAKED 45 / 180 / 0. Recall 1.000 for stripe, slack, private key, jwt and generic API keys; 0.640 password, 0.560 GitHub, 0.520 AWS, 0.480 database_url. Only false positives: 2 placeholders.

`lora_a_valid` (valid set, 446 rows, seen templates only): precision 0.981, recall 0.929, F1 0.954, accuracy 0.955, 0 INVALID. Weakest: database_url recall 0.480 (13 of the 16 missed leaks).

`lora_a` training: 1,000 iterations, batch 4, 1.09M tokens, 780 s wall clock (including model load and 6 validation passes of about 10 s each), mean 1,542 tokens/s over the 100 reports (range 1,075 to 1,667). Validation loss (100 rows each): 3.785 (iter 1), 0.517 (200), 0.533 (400), 0.492 (600), 0.515 (800), 0.464 (1000).

`baseline` details (base model, no adapter):

Binary confusion (gold by predicted): SAFE 216 / 15 / 3 (SAFE / LEAKED / INVALID); LEAKED 197 / 25 / 3.

| Subset | n | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|
| held-out templates | 225 | 0.533 | 0.074 | 0.130 | 0.516 |
| seen templates | 234 | 0.680 | 0.145 | 0.239 | 0.534 |

What the baseline shows:
- The base model mostly answers SAFE (413 of 459 rows), so recall is very low. Only private keys are caught often (recall 0.56); database_url, slack_token and generic_api_key are never flagged.
- When it does flag a leak, it names the type correctly (type accuracy 1.0 on 25 rows).
- Highest false-positive rates: public_key 0.278, placeholder 0.167.
- All 6 INVALID outputs have the form `LEAKED: <variable name from the snippet>` (for example `LEAKED: tls_cert`, `LEAKED: client_secret`).
- Runtime 52.4 s for 459 rows; two `--limit 50` runs were byte-identical.

Earlier: smoke test with prompt v2 (greedy `temp=0.0`, `max_tokens=64`, `HF_HUB_OFFLINE=1`, ran twice with identical outputs):

| Snippet | Expected | Raw output (repr) | Correct |
|---|---|---|---|
| a) `db_password = 'hunter2_prod_2024'` | `LEAKED: password` | `'LEAKED: password'` | yes |
| b) `db_password = os.getenv("DB_PASS")` | `SAFE` | `'SAFE'` | yes |
| c) `STRIPE_KEY = "YOUR_STRIPE_KEY_HERE"` | `SAFE` | `'LEAKED: stripe_key'` | no |
| d) `api_token: "****"` | `SAFE` | `'SAFE'` | yes |
| e) `DATABASE_URL=postgres://admin:s3cretPass@db.internal:5432/app` | `LEAKED: database_url` | `'SAFE'` | no |

3 of 5 correct, 0 format errors. For reference, v1 in Round 1 gave `'SAFE'` for both a) and b).
