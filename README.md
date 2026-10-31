# secret-scanner-lora

I fine-tuned Qwen2.5-1.5B-Instruct with LoRA on a MacBook (MLX) to read one short code or config snippet and answer with exactly one line: `SAFE` or `LEAKED: <type>`.
On a synthetic test set of 459 snippets, F1 for the LEAKED class went from 0.189 for the base model to 0.885 with mlx-lm's default LoRA settings, and to 0.973 after one change: computing the training loss on the answer tokens only.
Most remaining errors come from one code shape the model never saw in training, and an ablation traced them to a single field next to the leaked key rather than to the key itself.

Test set (459 rows, half of them from templates never seen in training):

| Run | Precision | Recall | F1 | Accuracy | Held-out-template F1 | Format errors |
|---|---|---|---|---|---|---|
| baseline (no fine-tuning) | 0.625 | 0.111 | 0.189 | 0.525 | 0.130 | 6 |
| lora_a (mlx-lm defaults) | 0.989 | 0.800 | 0.885 | 0.895 | 0.785 | 0 |
| lora_b (lora_a + loss on answer tokens only) | 1.000 | 0.947 | 0.973 | 0.967 | 0.941 | 3 |

lora_b is the final model. Its precision is 1.000 under the strict scoring described below, and 0.986 if its 3 malformed answers on SAFE snippets are counted as false alarms.

## Task

**Input:** one snippet of 1 to 6 lines of Python, JS/TS, YAML, `.env`, JSON, shell or Dockerfile. It is passed to the model as the user message, with no wrapper.

**Output:** exactly one line, either

```
SAFE
```

or

```
LEAKED: <type>
```

where `<type>` is one of 9 leak types: `aws_access_key`, `github_token`, `stripe_key`, `slack_token`, `private_key`, `jwt`, `database_url`, `password`, `generic_api_key`.

**SAFE lookalikes** (the hard part of the task), 13 categories:

| Category | Example of what it covers |
|---|---|
| env_var | `os.getenv("API_KEY")`, `process.env.STRIPE_KEY`, `${TOKEN}` |
| secret_manager | AWS Secrets Manager, GCP Secret Manager, SSM, Vault, `${{ secrets.X }}`, Kubernetes `secretKeyRef` |
| placeholder | values with an explicit marker: `YOUR_API_KEY_HERE`, `<token>`, `dummy-...`, `xxxx...`, `...EXAMPLE` |
| redacted | `****`, `[REDACTED]`, a token prefix followed by stars |
| hash | SHA-256 checksums, image digests, npm `integrity` values |
| uuid | tenant, workspace and request IDs |
| commit_sha | pinned GitHub Actions, `git checkout <sha>`, build info |
| public_key | PEM public keys and certificates, SSH public keys |
| base64_non_secret | embedded images and data URIs |
| empty_value | `API_KEY=`, `password: ""`, `ARG TOKEN`, `Optional[str] = None` |
| keyword_non_secret | `PASSWORD_MIN_LENGTH = 12`, `cache_key = f"user:{uid}"`, `api_key_header: X-API-Key` |
| db_url_no_secret | connection strings with no password, or a password read from the environment |
| plain_code | ordinary code and config with no secret-sounding names |

## Data

All data is synthetic. `generate_data.py` uses only the Python standard library and one `random.Random(42)`, so two runs produce byte-identical files.

- **Templates.** Every example comes from one of 90 named templates across the 7 formats. Each template is a skeleton with slots such as `{value}`, `{host}` or `{region}`, and harmless filler lines (imports, unrelated config) are added around it.
- **Contrast pairs.** Value-slot templates produce both labels from the same skeleton, so the model has to look at the value rather than the code around it:

  ```
  stripe.api_key = "{value}"
    {value} = a realistic fake Stripe key        -> LEAKED: stripe_key
    {value} = "YOUR_STRIPE_SECRET_KEY_HERE"      -> SAFE (placeholder)
    {value} = "********"                         -> SAFE (redacted)
    {value} = ""                                 -> SAFE (empty value)
  ```

- **Hard negatives.** Hashes under checksum names next to 40-character hex API keys under key names, public key blocks next to private key blocks, passwordless connection strings next to ones with a password, and so on.
- **Held-out templates.** 22 of the 90 templates (24.4%) are used only in the test set, with at least one per leak type and per SAFE category. Half of the test set comes from these templates and half from seen templates with fresh random values. This measures whether the model learned what a secret looks like or just memorised the training skeletons.
- **Deduplication.** Snippets are deduplicated by whitespace-normalised text across all splits; no snippet appears twice anywhere.

| Split | Total | SAFE | LEAKED | Per leak type | Per SAFE category | From held-out templates |
|---|---|---|---|---|---|---|
| train | 2,093 | 1,040 | 1,053 | 117 | 80 | 0 |
| valid | 446 | 221 | 225 | 25 | 17 | 0 |
| test | 459 | 234 | 225 | 25 | 18 | 225 |

**Labelling rules**

1. Only explicit placeholder markers make a value SAFE: values containing `EXAMPLE`, `dummy`, `xxxx`, `your_..._here` or `<...>` (case-insensitive). Realistic-looking fake values are always LEAKED, even short or simple ones, and Stripe test-mode keys count as leaks because they are real credentials.
2. A connection string with an embedded password is `database_url`, not `password`.
3. A private key block is `private_key`. A public key or certificate is SAFE.
4. Each snippet contains at most one secret, and filler lines never contain one.

**Why `data/` is not in the repo.** The data is full of realistic token-shaped strings, and GitHub push protection blocks pushes that contain them. Only the generator is committed, and it assembles those strings at runtime from split pieces, so the source contains no literal token. To rebuild and verify the data:

```
python generate_data.py && python check_data.py
```

`check_data.py` runs 40 checks and exits non-zero if any fails. Among other things, it checks that held-out templates never reach train or valid, that no snippet is shared between splits, that every label is valid, and that every typed leak contains a complete value.

## Method

- **Model:** `Qwen/Qwen2.5-1.5B-Instruct` in bf16, run with MLX on a MacBook Pro (M3 Max, 64 GB).
- **System prompt:** one prompt, frozen in `configs/system_prompt.txt` and used unchanged for every run. I changed it once, from v1 to v2, after a 2-snippet smoke test suggested the v1 wording pushed the base model towards SAFE. That was before any dataset existed.
- **Decoding:** greedy (temperature 0), at most 32 new tokens, via `mlx_lm.sample_utils.make_sampler(temp=0.0)`.
- **Strict parsing:** I strip leading and trailing whitespace, and the result must equal one of the 10 valid labels exactly. Anything else (wrong case, extra words or lines, trailing punctuation, an invented type) is INVALID. INVALID is never a positive prediction: it is a false negative when the snippet has a leak, and an incorrect answer (neither a false positive nor a true negative) when it is SAFE. It always counts as wrong for accuracy.
- **Metrics:** precision, recall and F1 with LEAKED as the positive class; accuracy on the exact label (so a wrong leak type is wrong); type accuracy on rows where both gold and prediction are LEAKED; format error rate; confusion matrices; and breakdowns by leak type, SAFE category, held-out vs seen template and format. Paired comparisons use McNemar's exact test on per-row correctness.
- **Validation vs test:** the validation set (446 rows, seen templates only) is used for every decision. The test set is only used for reporting.
- **LoRA (lora_a):** mlx-lm 0.32.2 defaults, every one written out in `configs/lora_a.yaml`: rank 8, scale 20, dropout 0, on every linear layer of the last 16 of the 28 transformer blocks; Adam at a constant learning rate of 1e-5; batch size 4; 1,000 iterations; seed 0. That trains 5.276M of 1,543.714M parameters (0.342%).
- **lora_b:** the same config with one change, `mask_prompt: true`, so the loss covers the answer instead of the whole conversation. The only other line that differs in `configs/lora_b.yaml` is the output directory.
- **Training cost:** lora_a took 780 s and lora_b 806 s. lora_b's run, measured with `/usr/bin/time -l`, had a peak memory footprint of 11.15 GB. (mlx-lm 0.32.2 does not print peak memory, so lora_a's log has no figure.)
- **Adapter choice:** I always use the adapter saved after the last iteration, and never pick a checkpoint by test score.

## Results

The test-set tables (459 rows) come from `python compare.py baseline lora_a lora_b`. The validation table and the valid row of the McNemar table come from `python compare.py lora_a_valid lora_b_valid`.

**Overall**

| | baseline | lora_a | lora_b |
|---|---|---|---|
| precision | 0.625 | 0.989 | 1.000 |
| recall | 0.111 | 0.800 | 0.947 |
| F1 | 0.189 | 0.885 | 0.973 |
| accuracy (exact label) | 0.525 | 0.895 | 0.967 |
| binary accuracy (SAFE vs LEAKED) | 0.525 | 0.898 | 0.967 |
| type accuracy | 1.000 | 0.994 | 1.000 |
| format error rate | 0.013 | 0.000 | 0.007 |
| true positives | 25 | 180 | 213 |
| false positives | 15 | 2 | 0 |
| false negatives | 200 | 45 | 12 |
| true negatives | 216 | 232 | 231 |
| INVALID | 6 | 0 | 3 |

lora_b's precision is 1.000 under the strict rules. Its 3 INVALID answers were all on SAFE snippets and all began with `LEAKED:`. If they were counted as false alarms, precision would be 213/216 = 0.986.

**Recall per leak type** (25 test snippets each)

| | baseline | lora_a | lora_b |
|---|---|---|---|
| aws_access_key | 0.080 | 0.520 | 0.520 |
| github_token | 0.120 | 0.560 | 1.000 |
| stripe_key | 0.120 | 1.000 | 1.000 |
| slack_token | 0.000 | 1.000 | 1.000 |
| private_key | 0.560 | 1.000 | 1.000 |
| jwt | 0.080 | 1.000 | 1.000 |
| database_url | 0.000 | 0.480 | 1.000 |
| password | 0.040 | 0.640 | 1.000 |
| generic_api_key | 0.000 | 1.000 | 1.000 |

**False-positive rate per SAFE category** (18 test snippets each; INVALID answers are not counted here)

| | baseline | lora_a | lora_b |
|---|---|---|---|
| env_var | 0.056 | 0.000 | 0.000 |
| secret_manager | 0.111 | 0.000 | 0.000 |
| placeholder | 0.167 | 0.111 | 0.000 |
| redacted | 0.056 | 0.000 | 0.000 |
| hash | 0.000 | 0.000 | 0.000 |
| uuid | 0.000 | 0.000 | 0.000 |
| commit_sha | 0.000 | 0.000 | 0.000 |
| public_key | 0.278 | 0.000 | 0.000 |
| base64_non_secret | 0.000 | 0.000 | 0.000 |
| empty_value | 0.111 | 0.000 | 0.000 |
| keyword_non_secret | 0.056 | 0.000 | 0.000 |
| db_url_no_secret | 0.000 | 0.000 | 0.000 |
| plain_code | 0.000 | 0.000 | 0.000 |

**Held-out vs seen templates**

| | baseline | lora_a | lora_b |
|---|---|---|---|
| held-out F1 (225 rows) | 0.130 | 0.785 | 0.941 |
| held-out recall | 0.074 | 0.657 | 0.889 |
| held-out accuracy | 0.516 | 0.822 | 0.933 |
| seen F1 (234 rows) | 0.239 | 0.965 | 1.000 |
| seen recall | 0.145 | 0.932 | 1.000 |
| seen accuracy | 0.534 | 0.966 | 1.000 |
| F1 gap (held-out minus seen) | -0.109 | -0.180 | -0.059 |

**F1 per format**

| | rows | baseline | lora_a | lora_b |
|---|---|---|---|---|
| python | 74 | 0.195 | 0.969 | 1.000 |
| js_ts | 78 | 0.205 | 0.731 | 0.778 |
| yaml | 80 | 0.308 | 1.000 | 1.000 |
| env | 36 | 0.190 | 0.875 | 1.000 |
| json | 78 | 0.000 | 0.905 | 1.000 |
| shell | 72 | 0.291 | 0.900 | 1.000 |
| dockerfile | 41 | 0.080 | 0.703 | 1.000 |

**Validation set** (446 rows, seen templates only; this is what the lora_b decision was based on)

| | precision | recall | F1 | accuracy | INVALID |
|---|---|---|---|---|---|
| lora_a | 0.981 | 0.929 | 0.954 | 0.955 | 0 |
| lora_b | 0.996 | 1.000 | 0.998 | 0.998 | 0 |

**McNemar's exact test** (b = rows the first run got right and the second got wrong, c = the reverse; two-sided)

| Data | Runs | Exact label: b / c / p | Binary: b / c / p |
|---|---|---|---|
| valid | lora_a vs lora_b | 0 / 19 / 3.81e-06 | 0 / 19 / 3.81e-06 |
| test | lora_a vs lora_b | 3 / 36 / 3.61e-08 | 3 / 35 / 6.68e-08 |
| test | baseline vs lora_a | 1 / 171 / 5.78e-50 | 1 / 172 / 2.91e-50 |
| test | baseline vs lora_b | 3 / 206 / 3.7e-57 | 3 / 206 / 3.7e-57 |

## What I learned

**Masking the prompt mattered more than anything else I changed.** Without masking, the loss is averaged over the whole conversation. In lora_a's run that was 1,094.3k tokens over 4,000 training rows (273.6 per row), most of it the fixed system prompt and the random characters inside the snippets. With `mask_prompt: true`, lora_b's loss covered 26.0k tokens over the same rows (6.5 per row, one of which is a padding token, see below), so the answer itself was about 2% of what lora_a was optimising. lora_a's training loss was 2.460 at iteration 10 and 0.704 at iteration 20, then moved between 0.344 and 0.782 for the rest of training. With masking, test recall rose from 0.800 to 0.947: database_url went from 0.480 to 1.000, password from 0.640 to 1.000, github_token from 0.560 to 1.000, and the placeholder false-positive rate fell from 0.111 to 0.000. On the validation set lora_b fixed 19 rows lora_a got wrong and broke none (p = 3.81e-06).

**The remaining misses are a shortcut, but not the one I first guessed.** All 12 of lora_b's false negatives (and 12 of lora_a's 45) come from one held-out JS/TS template that hardcodes an AWS access key ID next to `secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY`. In the training data, environment-variable reads appear in 129 SAFE rows and only 13 LEAKED rows, so my first hypothesis was "env read present means SAFE". Ablations on those 12 test snippets, run on both models (scratch copies, test set untouched):

| Variant of the snippet | lora_a flags it | lora_b flags it |
|---|---|---|
| original | 0/12 | 0/12 |
| `secretAccessKey` field deleted | 12/12 | 12/12 |
| `secretAccessKey: secretKey` (no env read) | 0/12 | 0/12 |
| as above, plus `region: process.env.AWS_REGION` | 0/12 | 0/12 |

Replacing the env read with a plain variable does not change the answer, so the env read is not the trigger. The trigger is the presence of a non-literal `secretAccessKey` field next to the key. The word "secret" does not appear in any of the 117 LEAKED AWS rows in the training data, so the model never saw a hardcoded key ID next to a secret key that is loaded some other way. (The last variant cannot test whether an env read elsewhere matters, because the variant before it is already SAFE.)

**Strict parsing kept the numbers honest.** All 6 of the base model's INVALID answers were `LEAKED: <variable name from the snippet>`, such as `LEAKED: tls_cert` or `LEAKED: client_secret`, rather than one of the 9 types. lora_b's 3 INVALID answers were on SSH public keys in a held-out YAML template (`LEAKED: ssh_key`, `LEAKED: ssh_authorized_keys`). A lenient parser that accepted any `LEAKED:` prefix would have scored these as detections on leak rows and as false alarms on SAFE rows. Strict parsing shows them as format errors instead, which is why I report lora_b's precision both ways.

**mlx-lm 0.32.2 adds one padding token to the loss of every sequence.** Before training lora_b I decoded the exact tokens that enter the masked loss for one SAFE and one LEAKED training row. They were the answer, `<|im_end|>`, the newline after it, and one more: token id 0 (`!`). Batches are zero-padded with at least one extra column, and the loss mask keeps target positions up to and including the sequence length, so the first padding position is always included. lora_a had the same extra target, as 1 of about 274 per row instead of 1 of 6.5. It cannot change a prediction, because generation stops at `<|im_end|>` and nothing after it is ever produced; it only dilutes the masked loss.

## Limitations

- **Synthetic data only.** Every number here is measured on snippets from my own generator. Real code has far more variety, and real leaks are rarer than 50%.
- **One training run per config.** Each config was trained once with seed 0, so there is no estimate of run-to-run variance. McNemar's test covers the paired comparison on fixed data, not training noise.
- **Small per-type test sets.** There are 25 test snippets per leak type and 18 per SAFE category, so one snippet moves a per-type recall by 0.04.
- **Validation has seen templates only.** lora_b's validation F1 of 0.998 overstates real performance; the held-out test F1 of 0.941 is the better estimate.
- **AWS access key IDs are labelled LEAKED.** An access key ID is only half of a key pair and is useless without the secret, but secret scanners flag it, and so does this dataset. lora_b's 12 misses are all snippets where only the ID is hardcoded and the secret is loaded separately, so under a policy that treats a lone ID as not secret, those answers would count as correct.
- **The padding target.** Both runs include one padding token per sequence in the loss (see above). I documented it rather than patching mlx-lm.

## Reproduce

Requires Apple Silicon and Python 3.12.

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download the base model once. evaluate.py sets HF_HUB_OFFLINE=1 unless it is already set,
# so every later step reads the local Hugging Face cache and never contacts the Hub.
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')"

python generate_data.py && python check_data.py
python test_evaluate.py

python evaluate.py --run-name baseline

HF_HUB_OFFLINE=1 mlx_lm.lora -c configs/lora_a.yaml
HF_HUB_OFFLINE=1 mlx_lm.lora -c configs/lora_b.yaml

python evaluate.py --adapter adapters/lora_a --run-name lora_a
python evaluate.py --adapter adapters/lora_a --data data/valid.jsonl --meta data/valid_meta.jsonl --run-name lora_a_valid
python evaluate.py --adapter adapters/lora_b --run-name lora_b
python evaluate.py --adapter adapters/lora_b --data data/valid.jsonl --meta data/valid_meta.jsonl --run-name lora_b_valid

python compare.py baseline lora_a lora_b
python compare.py lora_a_valid lora_b_valid
```

Each `results/<run>_metrics.json` records the SHA-256 of the system prompt and of the data file it was evaluated on, so a regenerated dataset can be checked against mine. If the model download fails, set `HF_HUB_OFFLINE=0` only for that step; on my network the Python downloader failed, so I fetched `model.safetensors` with `curl` into the Hugging Face cache and checked its SHA-256 against the one published on the Hub.

## Repo layout

```
generate_data.py        seeded synthetic dataset generator (standard library only)
check_data.py           dataset integrity checks (40 checks, exits 1 on failure)
evaluate.py             greedy evaluation, strict parser, metrics, per-row predictions
compare.py              side-by-side tables and McNemar's exact test
test_evaluate.py        tests for the parser, metrics and McNemar test
requirements.txt        exact package versions
configs/
  system_prompt.txt     the frozen system prompt
  lora_a.yaml           every mlx-lm 0.32.2 training default, written out
  lora_b.yaml           lora_a with mask_prompt: true
results/
  <run>_metrics.json    metrics and run config for baseline, lora_a, lora_b (+ _valid runs)
  lora_a_train_log.txt  training logs
  lora_b_train_log.txt
PROJECT_STATE.md        engineering log: decisions, findings, open issues
data/                   generated, not committed (see Data)
adapters/               trained LoRA weights, not committed
```

Per-row prediction and error files (`results/*_predictions.jsonl`, `results/*_errors.jsonl`) are also not committed, because they contain the snippets.
