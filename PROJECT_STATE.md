# PROJECT_STATE

## Current step
Round 1b complete: history cleanup, system prompt v2 frozen, five-snippet smoke test. Waiting for Round 2 (dataset).

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

## Open issues
- No remote yet, so nothing is pushed.
- Package versions are recorded here but there is no `requirements.txt` yet.

## Latest metrics
None yet (no dataset). Smoke test with prompt v2 (greedy `temp=0.0`, `max_tokens=64`, `HF_HUB_OFFLINE=1`, ran twice with identical outputs):

| Snippet | Expected | Raw output (repr) | Correct |
|---|---|---|---|
| a) `db_password = 'hunter2_prod_2024'` | `LEAKED: password` | `'LEAKED: password'` | yes |
| b) `db_password = os.getenv("DB_PASS")` | `SAFE` | `'SAFE'` | yes |
| c) `STRIPE_KEY = "YOUR_STRIPE_KEY_HERE"` | `SAFE` | `'LEAKED: stripe_key'` | no |
| d) `api_token: "****"` | `SAFE` | `'SAFE'` | yes |
| e) `DATABASE_URL=postgres://admin:s3cretPass@db.internal:5432/app` | `LEAKED: database_url` | `'SAFE'` | no |

3 of 5 correct, 0 format errors. For reference, v1 in Round 1 gave `'SAFE'` for both a) and b).
