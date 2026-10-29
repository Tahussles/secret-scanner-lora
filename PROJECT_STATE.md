# PROJECT_STATE

## Current step
Round 1 complete: project setup and base model smoke test. Waiting for Round 2.

## Done
- Round 1
  - Repo created at `~/secret-scanner-lora`.
  - Python checked: native arm64 (not Rosetta).
  - git initialised on branch `main`, `.gitignore` covers `.venv/`, `data/`, `adapters/`, `__pycache__/`, `.DS_Store`.
  - `.venv` created, `mlx-lm` installed.
  - Skeleton: `configs/`, `results/` (with `.gitkeep`).
  - System prompt saved to `configs/system_prompt.txt`.
  - Base model smoke test on two snippets (outputs below).

## Environment
| Item | Value |
|---|---|
| Machine | MacBook Pro M3 Max, 64 GB, macOS 26.5 (arm64) |
| Python | 3.12.16 (Homebrew, arm64) in `.venv` |
| mlx | 0.32.4 |
| mlx-lm | 0.32.2 |
| transformers | 5.22.0 |
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` (bf16 weights from Hugging Face) |

## Decisions
1. Use Homebrew Python 3.12 for the venv. The only Python on PATH was Apple's 3.9.6, and mlx-lm 0.32.2 requires Python 3.11 or newer.
2. Commits use the global git identity (Taha Hussain, GitHub noreply email). No repo-local `user.name` / `user.email` is set.
3. Default branch is `main` (matches GitHub's default for the future remote).
4. One system prompt for every run, stored in `configs/system_prompt.txt` and read with `.strip()`. It asks for exactly one line (`SAFE` or `LEAKED: <type>`), lists the 9 types, and gives a one-sentence definition of SAFE.
5. Greedy decoding means `temp=0.0` (`mlx_lm.sample_utils.make_sampler(temp=0.0)`, which is argmax).
6. Model weights (`model.safetensors`, 3,087,467,144 bytes, revision `989aa79`) were downloaded with `curl` into the standard Hugging Face cache, because Python's downloader failed on this network (the Xet transfer errored, and plain HTTP with `HF_HUB_DISABLE_XET=1` hit repeated DNS errors). The SHA-256 matches the Hub's published hash (`dd924a11...c6d3ee`). Inference runs with `HF_HUB_OFFLINE=1` so it never re-contacts the Hub.
7. Keep `configs/system_prompt.txt` exactly as written, including the "Reply SAFE when..." sentence (Taha's call after Round 1). The base model called a plain hardcoded password SAFE in the smoke test, so the baseline run will show whether that sentence biases it toward SAFE.

## Open issues
- No remote yet, so nothing is pushed.
- Package versions are recorded here but there is no `requirements.txt` yet.
- Python downloads from Hugging Face are unreliable on this network. Use `HF_HUB_OFFLINE=1` now that the weights are cached.

## Latest metrics
None yet. Smoke test only (greedy `temp=0.0`, `max_tokens=64`, ran twice with identical outputs):

| Snippet | Expected | Raw output (repr) |
|---|---|---|
| a) `db_password = 'hunter2_prod_2024'` | `LEAKED: password` | `'SAFE'` |
| b) `db_password = os.getenv("DB_PASS")` | `SAFE` | `'SAFE'` |
