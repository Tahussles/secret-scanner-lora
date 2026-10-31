"""Generate the synthetic secret-scanner dataset.

Each example is one short code or config snippet (1 to 6 lines) labelled with
exactly one line: SAFE or LEAKED: <type>.

How it works
- Every snippet comes from a named template: a skeleton string with {slots}.
  Slots are filled from one seeded random.Random, so two runs give
  byte-identical files.
- Value-slot templates have a {value} slot. The same skeleton gives a LEAKED
  example when {value} is a realistic fake credential, and a SAFE example when
  it is a placeholder, a redacted value or empty.
- Structural SAFE cases (env var reads, secret manager calls, hashes, ...) have
  their own templates.
- About 25% of templates are held out: their examples only ever go to test.

Labelling rules (in order of precedence)
1. A connection string with an embedded password is database_url, not password.
2. A private key block is private_key. A public key or certificate is SAFE.
3. Only explicit placeholder markers make a value SAFE: values containing
   EXAMPLE, dummy, xxxx, your_..._here or <...> (case-insensitive).
   Realistic-looking fake values are always LEAKED, even short or simple ones.
4. Each snippet contains at most one secret. Filler lines never contain secrets.

Token-shaped strings (AWS keys, GitHub tokens, Stripe keys, Slack tokens, PEM
headers, JWTs) are assembled at runtime from split pieces, so this file contains
no literal string that matches a real token pattern.

Usage: python generate_data.py
Writes data/train.jsonl, data/valid.jsonl, data/test.jsonl in mlx-lm chat
format, plus line-aligned data/*_meta.jsonl files with template_id, category,
format and held_out.
"""

import base64
import json
import random
import re
import string
import uuid
from dataclasses import dataclass
from pathlib import Path

SEED = 42
rng = random.Random(SEED)  # every random value in this file comes from this one generator

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SYSTEM_PROMPT_PATH = ROOT / "configs" / "system_prompt.txt"

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

EXAMPLES_PER_LEAK_TYPE = 167      # 9 x 167 = 1,503 LEAKED
EXAMPLES_PER_SAFE_CATEGORY = 115  # 13 x 115 = 1,495 SAFE
TRAIN_FRACTION = 0.70
VALID_FRACTION = 0.15             # test gets the rest: half held-out templates, half seen templates
MAX_LINES = 6
MAX_ATTEMPTS = 100                # tries per example to get a snippet that is not a duplicate


# ---------------------------------------------------------------------------
# Random building blocks
# ---------------------------------------------------------------------------

UPPER = string.ascii_uppercase
DIGITS = string.digits
ALNUM = string.ascii_letters + string.digits
HEX = "0123456789abcdef"


def rand_str(alphabet, length):
    return "".join(rng.choice(alphabet) for _ in range(length))


def b64(data):
    return base64.b64encode(data).decode()


def b64url(data):
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


PEOPLE = ["alice", "bob", "chen", "diego", "fatima", "grace", "ivan", "noor"]
DB_USERS = ["app", "admin", "svc_api", "readonly", "etl", "deploy", "reporting", "postgres"]
DB_NAMES = ["app", "orders", "users", "analytics", "billing", "inventory", "app_production"]
ORGS = ["acme-corp", "northwind", "globex", "initech", "umbrella-labs", "bluefin"]
REPOS = ["api", "web-frontend", "infra", "data-pipeline", "billing-service", "mobile-app"]
DOMAINS = ["acme-corp.com", "northwind.io", "globex.net", "initech.dev", "bluefin.app"]
SERVICES = ["payments", "notifications", "search", "billing", "analytics", "mailer", "geocoder", "worker"]
STAGES = ["prod", "staging", "dev"]
REGIONS = ["us-east-1", "us-west-2", "eu-west-1", "eu-central-1", "ap-southeast-2"]


def db_host():
    style = rng.randrange(3)
    if style == 0:
        name = rng.choice(["db", "postgres", "mysql", "primary-db", "users-db", "orders-db"])
        return name + rng.choice([".internal", ".prod.svc.cluster.local", ".staging.local"])
    if style == 1:
        return f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
    return rng.choice(["localhost", "db", "mysql", "postgres"])


# ---------------------------------------------------------------------------
# Realistic fake secrets (LEAKED values)
# ---------------------------------------------------------------------------

def aws_access_key():
    # 4-character prefix, then 16 upper-case base32 characters
    return rng.choice(["AK", "AS"]) + "IA" + rand_str(UPPER + "234567", 16)


def github_token():
    if rng.random() < 0.2:  # fine-grained personal access token
        return "github" + "_pat_" + rand_str(ALNUM, 22) + "_" + rand_str(ALNUM, 59)
    return "gh" + rng.choice("pousr") + "_" + rand_str(ALNUM, 36)


def stripe_key():
    # Test-mode keys are real credentials too, so both modes are LEAKED.
    mode = rng.choice(["live", "test"])
    return rng.choice(["sk", "rk"]) + "_" + mode + "_" + rand_str(ALNUM, rng.choice([24, 34]))


def slack_token():
    return ("xo" + "x" + rng.choice("bp") + "-" + rand_str(DIGITS, 12) + "-"
            + rand_str(DIGITS, 13) + "-" + rand_str(ALNUM, 24))


# DER prefixes make each base64 body start the way real keys do,
# for example a PKCS#1 RSA private key body starts with "MIIE".
PRIVATE_KEY_DER_PREFIXES = {
    "RSA PRIVATE KEY": bytes.fromhex("308204a40201000282010100"),
    "EC PRIVATE KEY": bytes.fromhex("30770201010420"),
    "OPENSSH PRIVATE KEY": b"openssh-key-v1\x00",
    "PRIVATE KEY": bytes.fromhex("308204be020100300d06092a864886f70d0101010500048204a8"),
}
PUBLIC_KEY_DER_PREFIXES = {
    "PUBLIC KEY": bytes.fromhex("30820122300d06092a864886f70d01010105000382010f00"),
    "RSA PUBLIC KEY": bytes.fromhex("3082010a0282010100"),
    "CERTIFICATE": bytes.fromhex("3082035e30820246a003020102"),
}


def pem_block(label, der_prefix):
    # Two 64-character body lines keep the whole block at 4 lines.
    body = b64(der_prefix + rng.randbytes(96 - len(der_prefix)))
    begin = "-----BEGIN " + label + "-----"
    end = "-----END " + label + "-----"
    return "\n".join([begin, body[:64], body[64:], end])


def private_key():
    label = rng.choice(list(PRIVATE_KEY_DER_PREFIXES))
    return pem_block(label, PRIVATE_KEY_DER_PREFIXES[label])


def jwt():
    header = {"alg": rng.choice(["HS256", "RS256"]), "typ": "JWT"}
    issued_at = rng.randint(1_700_000_000, 1_760_000_000)
    payload = {
        "sub": str(rng.randint(1000, 999999)),
        "role": rng.choice(["user", "admin", "service"]),
        "iat": issued_at,
        "exp": issued_at + 3600,
    }
    parts = [
        json.dumps(header, separators=(",", ":")).encode(),
        json.dumps(payload, separators=(",", ":")).encode(),
        rng.randbytes(32),  # signature
    ]
    return ".".join(b64url(part) for part in parts)


PASSWORD_WORDS = ["hunter", "summer", "winter", "dragon", "falcon", "rocket", "maple", "coffee",
                  "galaxy", "shadow", "phoenix", "panda", "silver", "harbor", "tiger", "cobalt"]


def password(url_safe=False):
    symbols = "_-." if url_safe else "!%_-.+"
    word = rng.choice(PASSWORD_WORDS)
    year = rng.randint(2015, 2026)
    style = rng.randrange(4)
    if style == 0:
        return word.capitalize() + str(year) + rng.choice(symbols)
    if style == 1:
        return f"{word}{rng.randint(1, 9)}_{rng.choice(['prod', 'staging', 'db', 'admin'])}_{year}"
    if style == 2:
        return word + str(rng.randint(10, 999))  # simple values still count as leaks
    return rand_str(ALNUM + symbols, rng.randint(12, 24))


def generic_api_key():
    style = rng.randrange(4)
    if style == 0:
        return rand_str(HEX, 32)
    if style == 1:
        return rand_str(HEX, 40)
    if style == 2:
        return rand_str(ALNUM, rng.choice([32, 40, 48]))
    return b64(rng.randbytes(30))


DB_PORTS = {"postgresql": 5432, "postgres": 5432, "mysql": 3306, "mongodb": 27017, "redis": 6379, "amqp": 5672}
ALL_DB_SCHEMES = list(DB_PORTS)
SQLALCHEMY_SCHEMES = ["postgresql", "postgresql+psycopg2", "mysql", "mysql+pymysql"]
POSTGRES_SCHEMES = ["postgres", "postgresql"]


def db_url(scheme, userinfo):
    port = DB_PORTS[scheme.split("+")[0]]
    path = str(rng.randint(0, 15)) if scheme == "redis" else rng.choice(DB_NAMES)
    at = userinfo + "@" if userinfo else ""
    return f"{scheme}://{at}{db_host()}:{port}/{path}"


def database_url(scheme):
    return db_url(scheme, rng.choice(DB_USERS) + ":" + password(url_safe=True))


LEAK_GENERATORS = {  # database_url is missing because it needs a scheme, see value_for()
    "aws_access_key": aws_access_key,
    "github_token": github_token,
    "stripe_key": stripe_key,
    "slack_token": slack_token,
    "private_key": private_key,
    "jwt": jwt,
    "password": password,
    "generic_api_key": generic_api_key,
}


# ---------------------------------------------------------------------------
# SAFE lookalike values for a {value} slot
# ---------------------------------------------------------------------------

PLACEHOLDER_MARKERS = re.compile(r"example|dummy|xxxx|your[_-]\S*[_-]here|<[^<>]+>", re.IGNORECASE)

PLACEHOLDER_LABELS = {
    "aws_access_key": "aws_access_key_id", "github_token": "github_token",
    "stripe_key": "stripe_secret_key", "slack_token": "slack_bot_token",
    "private_key": "private_key", "jwt": "access_token", "database_url": "database_url",
    "password": "password", "generic_api_key": "api_key",
}


def placeholder_value(kind, scheme):
    label = PLACEHOLDER_LABELS[kind]
    options = [
        "YOUR_" + label.upper() + "_HERE",
        "your_" + label + "_here",
        "<" + label.replace("_", "-") + ">",
        "dummy-" + label.replace("_", "-"),
        "x" * rng.randint(8, 32),
    ]
    # Some placeholders keep the real token shape around an explicit marker.
    if kind == "aws_access_key":
        options.append("AK" + "IA" + rand_str(UPPER + "234567", 9) + "EXAMPLE")
    elif kind == "github_token":
        options.append("gh" + "p_" + "x" * 36)
    elif kind == "stripe_key":
        options.append("sk_" + "test_" + "EXAMPLE" + rand_str(DIGITS, 8))
    elif kind == "slack_token":
        options.append("xo" + "xb-" + "your-slack-token-here")
    elif kind == "database_url":
        user = rng.choice(DB_USERS)
        options += [db_url(scheme, user + ":<password>"),
                    db_url(scheme, "<user>:<password>"),
                    db_url(scheme, user + ":your_password_here")]
    value = rng.choice(options)
    assert PLACEHOLDER_MARKERS.search(value), value
    return value


REDACTED_PREFIXES = {"aws_access_key": "AK" + "IA", "github_token": "gh" + "p_",
                     "stripe_key": "sk_" + "live_", "slack_token": "xo" + "xb-"}


def redacted_value(kind, scheme):
    if kind == "database_url":
        return db_url(scheme, rng.choice(DB_USERS) + ":" + rng.choice(["****", "********"]))
    options = ["****", "********", "[REDACTED]", "REDACTED", "***redacted***"]
    if kind in REDACTED_PREFIXES:
        options.append(REDACTED_PREFIXES[kind] + "*" * rng.randint(8, 20))
    return rng.choice(options)


def db_url_without_password(scheme):
    return db_url(scheme, rng.choice([rng.choice(DB_USERS), ""]))  # user only, or no user at all


def public_key():
    label = rng.choice(list(PUBLIC_KEY_DER_PREFIXES))
    return pem_block(label, PUBLIC_KEY_DER_PREFIXES[label])


def value_for(category, kind, scheme):
    """The {value} slot: a fake secret for a leak type, a lookalike for a SAFE category."""
    if category in LEAK_TYPES:
        while True:
            value = database_url(scheme) if kind == "database_url" else LEAK_GENERATORS[kind]()
            if not PLACEHOLDER_MARKERS.search(value):  # a LEAKED value must never carry a SAFE marker
                return value
    if category == "placeholder":
        return placeholder_value(kind, scheme)
    if category == "redacted":
        return redacted_value(kind, scheme)
    if category == "empty_value":
        return ""
    if category == "db_url_no_secret":
        return db_url_without_password(scheme)
    if category == "public_key":
        return public_key()
    raise ValueError(f"no value generator for category {category}")


# ---------------------------------------------------------------------------
# Harmless values for the other {slots}
# ---------------------------------------------------------------------------

PNG_HEADER = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
JPEG_HEADER = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00"


def ssh_public_key():
    algo = rng.choice(["ssh-ed25519", "ssh-rsa"])
    key_bytes = rng.randbytes(32 if algo == "ssh-ed25519" else 256)
    blob = len(algo).to_bytes(4, "big") + algo.encode() + len(key_bytes).to_bytes(4, "big") + key_bytes
    comment = rng.choice(PEOPLE) + "@" + rng.choice(["laptop", "ci-runner", "bastion", "workstation"])
    return f"{algo} {b64(blob)} {comment}"


def version():
    return f"{rng.randint(0, 4)}.{rng.randint(0, 20)}.{rng.randint(0, 30)}"


def timestamp():
    return (f"{rng.randint(2023, 2025)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
            f"T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00Z")


CONTEXT_SLOTS = {
    # infrastructure and code context
    "region": lambda: rng.choice(REGIONS),
    "host": db_host,
    "user": lambda: rng.choice(DB_USERS),
    "db": lambda: rng.choice(DB_NAMES),
    "org": lambda: rng.choice(ORGS),
    "repo": lambda: rng.choice(REPOS),
    "domain": lambda: rng.choice(DOMAINS),
    "service": lambda: rng.choice(SERVICES),
    "stage": lambda: rng.choice(STAGES),
    "profile": lambda: rng.choice(["default", "prod", "staging", "ci", "deploy"]),
    "bucket": lambda: rng.choice(ORGS) + "-" + rng.choice(["assets", "backups", "logs", "uploads"]),
    "project": lambda: f"{rng.choice(ORGS)}-{rng.choice(STAGES)}-{rng.randint(100000, 999999)}",
    "email": lambda: rng.choice(PEOPLE) + "@" + rng.choice(DOMAINS),
    "channel": lambda: "C" + rand_str(UPPER + DIGITS, 10),
    "message": lambda: rng.choice(["Deploy finished", "Nightly build failed", "Backup complete", "New signup"]),
    "client_class": lambda: rng.choice(["ApiClient", "SearchClient", "MailClient", "GeoClient"]),
    "base_image": lambda: rng.choice(["python:3.12-slim", "node:20-alpine", "ubuntu:22.04", "golang:1.22"]),
    "migration": lambda: f"{rng.randint(1, 120):04d}_" + rng.choice(["add_index", "create_users", "backfill_orders"]),
    "version": version,
    "branch": lambda: rng.choice(["main", "develop", "hotfix/login", "release/" + version()]),
    "timestamp": timestamp,
    "timeout": lambda: rng.randint(5, 120),
    "timeout_ms": lambda: rng.randint(1, 60) * 1000,
    "retries": lambda: rng.randint(1, 8),
    "limit": lambda: rng.choice([10, 25, 50, 100, 500]),
    "pool_size": lambda: rng.randint(2, 50),
    "app_port": lambda: rng.choice([3000, 5000, 8000, 8080, 9090]),
    "log_level": lambda: rng.choice(["debug", "info", "warn", "error"]),
    # SAFE lookalike values
    "uuid": lambda: str(uuid.UUID(int=rng.getrandbits(128), version=4)),
    "sha": lambda: rand_str(HEX, 40),
    "sha256": lambda: rand_str(HEX, 64),
    "checksum": lambda: rand_str(HEX, rng.choice([32, 40, 64])),  # md5, sha1 or sha256 length
    "integrity": lambda: "sha512-" + b64(rng.randbytes(64)),
    "b64_image": lambda: b64(PNG_HEADER + rng.randbytes(rng.randint(24, 60))),
    "b64_jpeg": lambda: b64(JPEG_HEADER + rng.randbytes(rng.randint(24, 60))),
    "ssh_public_key": ssh_public_key,
    "pw_env_ref": lambda: "${" + rng.choice(["DB_PASSWORD", "POSTGRES_PASSWORD", "PGPASSWORD"]) + "}",
    # names and values used by SAFE templates
    "hash_name": lambda: rng.choice(["EXPECTED_CHECKSUM", "ARTIFACT_DIGEST", "MODEL_FILE_HASH", "BUNDLE_SHA"]),
    "uuid_name": lambda: rng.choice(["TENANT_ID", "WORKSPACE_ID", "DEFAULT_ORG_ID", "AZURE_SUBSCRIPTION_ID", "AZURE_CLIENT_ID"]),
    "sha_var": lambda: rng.choice(["GIT_SHA", "SOURCE_COMMIT", "SENTRY_RELEASE", "BUILD_COMMIT"]),
    "icon_name": lambda: rng.choice(["FAVICON_PNG_B64", "LOGO_BASE64", "DEFAULT_AVATAR_B64", "EMPTY_STATE_ICON"]),
    "img_var": lambda: rng.choice(["fallbackImage", "spinnerIcon", "defaultAvatar", "emptyStateImage"]),
    "file_name": lambda: rng.choice(["logo.png", "favicon.png", "banner.png"]),
    "title": lambda: rng.choice(["Quarterly report", "Team offsite", "Release notes", "Onboarding guide"]),
    "package": lambda: rng.choice(["lodash", "date-fns", "zod", "express", "left-pad"]),
    "tool": lambda: rng.choice(["helm", "kubectl", "terraform", "jq"]),
    "action": lambda: rng.choice(["checkout", "setup-node", "cache", "upload-artifact"]),
    "git_verb": lambda: rng.choice(["checkout", "cherry-pick", "revert --no-edit", "reset --hard"]),
    "build_cmd": lambda: rng.choice(["npm ci && npm test", "make build", "pytest -q", "go test ./..."]),
    "min_length": lambda: rng.randint(8, 16),
    "max_age_days": lambda: rng.choice([30, 60, 90, 180]),
    "ttl": lambda: rng.choice([300, 900, 1800, 3600, 7200, 86400]),
    "storage_key": lambda: rng.choice(["auth_token", "access_token", "session", "refresh_token"]),
    "cookie_name": lambda: rng.choice(["session_token", "sid", "auth_session", "csrf_token"]),
    "cache_prefix": lambda: rng.choice(["user", "session", "profile", "cart"]),
    "pk_field": lambda: rng.choice(["id", "uuid", "order_id", "user_id"]),
    "func": lambda: rng.choice(["scale_values", "filter_scores", "normalize", "top_items"]),
    "factor": lambda: rng.randint(2, 10),
    "threshold": lambda: rng.randint(0, 100),
    "route": lambda: rng.choice(["health", "orders", "users", "metrics", "status"]),
    "table": lambda: rng.choice(["orders", "users", "products", "events"]),
    "replicas": lambda: rng.randint(1, 6),
    "apt_pkg": lambda: rng.choice(["curl", "git", "ca-certificates", "build-essential", "libpq-dev"]),
    "glob": lambda: rng.choice(["logs/*.log", "*.csv", "exports/*.json", "tmp/*.txt"]),
}


# ---------------------------------------------------------------------------
# Variable names for secrets, in snake_case. {NAME} and {camelName} are
# derived from {name}.
# ---------------------------------------------------------------------------

NAMES = {
    "aws_access_key": ["aws_access_key_id", "aws_access_key", "access_key_id"],
    "github_token": ["github_token", "gh_token", "github_pat", "git_token"],
    "stripe_key": ["stripe_secret_key", "stripe_api_key", "stripe_key"],
    "slack_token": ["slack_bot_token", "slack_token", "slack_api_token"],
    "private_key": ["private_key", "rsa_private_key", "jwt_signing_key", "tls_key", "ssh_private_key"],
    "public_key": ["public_key", "rsa_public_key", "jwt_public_key", "tls_cert", "ca_cert"],
    "jwt": ["access_token", "auth_token", "jwt", "bearer_token", "id_token", "session_token"],
    "database_url": ["database_url", "db_url", "db_uri", "connection_string", "datasource_url"],
    "password": ["db_password", "password", "admin_password", "smtp_password", "redis_password",
                 "postgres_password", "db_pass"],
    "generic_api_key": ["api_key", "api_secret", "secret_key", "client_secret", "service_api_key", "app_secret"],
}

# Secret kinds that make sense as a plain "NAME = value" assignment.
GENERIC_KINDS = ["aws_access_key", "github_token", "stripe_key", "slack_token", "jwt", "password", "generic_api_key"]
# Kinds used to name variables in structural SAFE templates (env reads, secret manager calls, ...).
NAMING_KINDS = GENERIC_KINDS + ["database_url"]


def camel_case(snake):
    first, *rest = snake.split("_")
    return first + "".join(word.capitalize() for word in rest)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

@dataclass
class Template:
    template_id: str
    fmt: str
    categories: list  # every category this template can produce
    skeleton: str
    kinds: list       # secret kinds used for {name}
    schemes: list     # URL schemes for database values
    held_out: bool    # True means its examples only go to test


# Held out from training: at least one template per leak type and per SAFE
# category. check_templates() enforces this.
HELD_OUT_TEMPLATE_IDS = {
    "ts_s3_client", "docker_npmrc_token", "sh_curl_stripe", "json_slack_notifier",
    "yaml_tls_block", "json_service_account", "js_fetch_bearer", "json_datasource",
    "sh_mysql_cli", "yaml_integration_key", "js_process_env", "py_gcp_secret_manager",
    "yaml_k8s_secret_ref", "docker_sha256_verify", "yaml_resource_uid", "json_build_info",
    "yaml_cloud_init_ssh", "json_thumbnail", "py_settings_optional", "env_keyword_settings",
    "js_db_url_from_env", "sh_plain_loop",
}


def T(template_id, fmt, categories, skeleton, kinds=None, schemes=None):
    leak_kinds = [c for c in LEAK_TYPES if c in categories]
    return Template(
        template_id=template_id,
        fmt=fmt,
        categories=list(categories),
        skeleton=skeleton.strip("\n"),
        kinds=kinds or leak_kinds or NAMING_KINDS,
        schemes=schemes or ALL_DB_SCHEMES,
        held_out=template_id in HELD_OUT_TEMPLATE_IDS,
    )


# SAFE fillers for a {value} slot.
SLOT_SAFE = ["placeholder", "redacted", "empty_value"]
SLOT_SAFE_NO_EMPTY = ["placeholder", "redacted"]  # for skeletons where an empty value would look odd
PEM_FILLS = ["private_key", "public_key", "placeholder"]
DB_FILLS = ["database_url", "db_url_no_secret", "placeholder", "redacted"]

TEMPLATES = [
    # --- aws_access_key ---
    T("py_boto3_client", "python", ["aws_access_key"] + SLOT_SAFE, """
import boto3
s3 = boto3.client("s3", region_name="{region}", aws_access_key_id="{value}")
"""),
    T("sh_aws_configure", "shell", ["aws_access_key"] + SLOT_SAFE_NO_EMPTY, """
aws configure set aws_access_key_id {value} --profile {profile}
aws configure set region {region} --profile {profile}
"""),
    T("yaml_aws_settings", "yaml", ["aws_access_key"] + SLOT_SAFE, """
aws:
  region: {region}
  {name}: "{value}"
  bucket: {bucket}
"""),
    T("ts_s3_client", "js_ts", ["aws_access_key"] + SLOT_SAFE, """
const s3 = new S3Client({
  region: "{region}",
  credentials: { accessKeyId: "{value}", secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY },
});
"""),
    # --- github_token ---
    T("sh_git_clone_token", "shell", ["github_token"] + SLOT_SAFE_NO_EMPTY, """
git clone https://x-access-token:{value}@github.com/{org}/{repo}.git
"""),
    T("js_octokit", "js_ts", ["github_token"] + SLOT_SAFE, """
const { Octokit } = require("@octokit/rest");
const octokit = new Octokit({ auth: "{value}" });
"""),
    T("py_github_api", "python", ["github_token"] + SLOT_SAFE_NO_EMPTY, """
headers = {"Authorization": "token {value}", "Accept": "application/vnd.github+json"}
resp = requests.get("https://api.github.com/repos/{org}/{repo}/pulls", headers=headers)
"""),
    T("docker_npmrc_token", "dockerfile", ["github_token"] + SLOT_SAFE_NO_EMPTY, """
RUN echo "//npm.pkg.github.com/:_authToken={value}" > .npmrc && npm ci
"""),
    # --- stripe_key ---
    T("py_stripe_module", "python", ["stripe_key"] + SLOT_SAFE, """
import stripe
stripe.api_key = "{value}"
"""),
    T("js_stripe_require", "js_ts", ["stripe_key"] + SLOT_SAFE_NO_EMPTY, """
const stripe = require("stripe")("{value}");
"""),
    T("ts_stripe_client", "js_ts", ["stripe_key"] + SLOT_SAFE_NO_EMPTY, """
import Stripe from "stripe";
const stripe = new Stripe("{value}", { apiVersion: "2024-06-20" });
"""),
    T("sh_curl_stripe", "shell", ["stripe_key"] + SLOT_SAFE_NO_EMPTY, """
curl https://api.stripe.com/v1/customers -u {value}: -d email={email}
"""),
    # --- slack_token ---
    T("py_slack_webclient", "python", ["slack_token"] + SLOT_SAFE, """
from slack_sdk import WebClient
client = WebClient(token="{value}")
client.chat_postMessage(channel="{channel}", text="{message}")
"""),
    T("js_slack_bolt", "js_ts", ["slack_token"] + SLOT_SAFE, """
const app = new App({
  token: "{value}",
  signingSecret: process.env.SLACK_SIGNING_SECRET,
});
"""),
    T("sh_curl_slack", "shell", ["slack_token"] + SLOT_SAFE_NO_EMPTY, """
curl -X POST https://slack.com/api/chat.postMessage -H "Authorization: Bearer {value}" -d channel={channel} -d text="{message}"
"""),
    T("json_slack_notifier", "json", ["slack_token"] + SLOT_SAFE, """
{
  "notifier": "slack",
  "{name}": "{value}",
  "channel": "{channel}"
}
"""),
    # --- private_key (PEM templates also give public_key contrast examples) ---
    T("py_pem_constant", "python", PEM_FILLS, '''
{NAME} = """{value}"""
'''),
    T("yaml_tls_block", "yaml", PEM_FILLS, """
tls:
  {name}: |
    {value}
"""),
    T("env_pem_escaped", "env", PEM_FILLS, """
{NAME}="{value_escaped}"
"""),
    T("json_service_account", "json", ["private_key", "placeholder"], """
{
  "type": "service_account",
  "project_id": "{project}",
  "private_key": "{value_escaped}",
  "client_email": "deployer@{project}.iam.gserviceaccount.com"
}
"""),
    T("sh_ssh_key_heredoc", "shell", ["private_key"], """
cat > ~/.ssh/id_deploy <<'EOF'
{value}
EOF
"""),
    # --- jwt ---
    T("sh_curl_bearer", "shell", ["jwt"] + SLOT_SAFE_NO_EMPTY, """
curl -s https://api.{domain}/v1/me -H "Authorization: Bearer {value}"
"""),
    T("js_fetch_bearer", "js_ts", ["jwt"] + SLOT_SAFE_NO_EMPTY, """
const res = await fetch("https://api.{domain}/v1/orders", {
  headers: { Authorization: "Bearer {value}" },
});
"""),
    T("py_requests_bearer", "python", ["jwt"] + SLOT_SAFE_NO_EMPTY, """
headers = {"Authorization": "Bearer {value}"}
resp = requests.get("https://api.{domain}/v1/orders", headers=headers, timeout={timeout})
"""),
    T("json_http_client", "json", ["jwt"] + SLOT_SAFE, """
{
  "baseUrl": "https://api.{domain}",
  "{camelName}": "{value}",
  "timeoutMs": {timeout_ms}
}
"""),
    # --- database_url (DB templates also give db_url_no_secret contrast examples) ---
    T("env_database_url", "env", DB_FILLS, """
{NAME}={value}
"""),
    T("py_sqlalchemy_engine", "python", DB_FILLS, """
from sqlalchemy import create_engine
engine = create_engine("{value}", pool_pre_ping=True)
""", schemes=SQLALCHEMY_SCHEMES),
    T("yaml_compose_db_url", "yaml", DB_FILLS, """
services:
  {service}:
    environment:
      {NAME}: {value}
"""),
    T("js_pg_pool", "js_ts", DB_FILLS, """
const { Pool } = require("pg");
const pool = new Pool({ connectionString: "{value}" });
""", schemes=POSTGRES_SCHEMES),
    T("json_datasource", "json", DB_FILLS, """
{
  "datasource": {
    "{name}": "{value}",
    "poolSize": {pool_size}
  }
}
"""),
    # --- password ---
    T("py_psycopg_connect", "python", ["password"] + SLOT_SAFE, """
import psycopg2
conn = psycopg2.connect(host="{host}", dbname="{db}", user="{user}", password="{value}")
"""),
    T("yaml_db_credentials", "yaml", ["password"] + SLOT_SAFE, """
database:
  host: {host}
  username: {user}
  {name}: "{value}"
"""),
    T("js_mysql_pool", "js_ts", ["password"] + SLOT_SAFE, """
const pool = mysql.createPool({
  host: "{host}",
  user: "{user}",
  password: "{value}",
  database: "{db}",
});
"""),
    T("sh_mysql_cli", "shell", ["password"] + SLOT_SAFE_NO_EMPTY, """
mysql -h {host} -u {user} -p'{value}' {db} < migrations/{migration}.sql
"""),
    T("docker_postgres_env", "dockerfile", ["password"] + SLOT_SAFE, """
ENV POSTGRES_USER={user} POSTGRES_DB={db} {NAME}={value}
"""),
    # --- generic_api_key ---
    T("py_api_key_header", "python", ["generic_api_key"] + SLOT_SAFE, """
session = requests.Session()
session.headers.update({"X-API-Key": "{value}"})
"""),
    T("js_api_client", "js_ts", ["generic_api_key"] + SLOT_SAFE, """
const client = new {client_class}({
  {camelName}: "{value}",
  timeout: {timeout_ms},
});
"""),
    T("yaml_integration_key", "yaml", ["generic_api_key"] + SLOT_SAFE, """
integrations:
  {service}:
    enabled: true
    {name}: "{value}"
"""),
    T("sh_curl_api_key", "shell", ["generic_api_key"] + SLOT_SAFE_NO_EMPTY, """
curl -H "x-api-key: {value}" "https://api.{domain}/v2/{service}?limit={limit}"
"""),
    T("py_django_settings", "python", ["generic_api_key"] + SLOT_SAFE, """
DEBUG = False
ALLOWED_HOSTS = ["{domain}"]
SECRET_KEY = "{value}"
"""),
    # --- generic "NAME = value" skeletons shared by several leak types ---
    T("env_assignment", "env", GENERIC_KINDS + SLOT_SAFE, """
{NAME}={value}
"""),
    T("sh_export", "shell", GENERIC_KINDS + SLOT_SAFE, """
export {NAME}="{value}"
"""),
    T("docker_env", "dockerfile", GENERIC_KINDS + SLOT_SAFE, """
ENV {NAME}={value}
"""),
    T("yaml_key_value", "yaml", GENERIC_KINDS + SLOT_SAFE, """
{name}: "{value}"
"""),
    T("json_key_value", "json", GENERIC_KINDS + SLOT_SAFE, """
{
  "{name}": "{value}",
  "region": "{region}"
}
"""),
    T("py_constant", "python", GENERIC_KINDS + SLOT_SAFE, """
{NAME} = "{value}"
"""),

    # --- env_var: the value is read from the environment ---
    T("py_os_getenv", "python", ["env_var"], """
import os
{NAME} = {py_env_read}
"""),
    T("js_process_env", "js_ts", ["env_var"], """
const {camelName} = process.env.{NAME};
"""),
    T("sh_env_reference", "shell", ["env_var"], """
curl -sf -H "Authorization: Bearer {env_ref}" https://api.{domain}/v1/health
""", kinds=["github_token", "slack_token", "jwt", "generic_api_key"]),
    T("yaml_compose_env", "yaml", ["env_var"], """
services:
  {service}:
    environment:
      {NAME}: {env_ref}
"""),
    T("docker_arg_to_env", "dockerfile", ["env_var"], """
ARG {NAME}
ENV {NAME}={env_ref}
"""),
    # --- secret_manager: the value is fetched from a secret store ---
    T("py_aws_secrets_manager", "python", ["secret_manager"], """
client = boto3.client("secretsmanager", region_name="{region}")
{name} = client.get_secret_value(SecretId="{secret_path}")["SecretString"]
"""),
    T("py_gcp_secret_manager", "python", ["secret_manager"], """
client = secretmanager.SecretManagerServiceClient()
resource = "projects/{project}/secrets/{secret_id}/versions/latest"
{name} = client.access_secret_version(name=resource).payload.data.decode()
"""),
    T("ts_aws_ssm", "js_ts", ["secret_manager"], """
const out = await ssm.send(new GetParameterCommand({ Name: "/{secret_path}", WithDecryption: true }));
const {camelName} = out.Parameter.Value;
"""),
    T("yaml_gha_secrets", "yaml", ["secret_manager"], """
jobs:
  deploy:
    env:
      {NAME}: ${{ secrets.{NAME} }}
"""),
    T("yaml_k8s_secret_ref", "yaml", ["secret_manager"], """
env:
  - name: {NAME}
    valueFrom:
      secretKeyRef:
        name: {secret_id}
        key: {name}
"""),
    T("sh_vault_cli", "shell", ["secret_manager"], """
export {NAME}="$(vault kv get -field={name} secret/{stage}/{service})"
"""),
    # --- hash ---
    T("py_checksum_constant", "python", ["hash"], """
{hash_name} = "{checksum}"
"""),
    T("docker_sha256_verify", "dockerfile", ["hash"], """
ADD https://releases.{domain}/{tool}/{tool}-{version}.tar.gz /tmp/{tool}.tar.gz
RUN echo "{sha256}  /tmp/{tool}.tar.gz" | sha256sum -c -
"""),
    T("yaml_image_digest", "yaml", ["hash"], """
containers:
  - name: {service}
    image: ghcr.io/{org}/{service}@sha256:{sha256}
"""),
    T("json_package_integrity", "json", ["hash"], """
{
  "name": "{package}",
  "version": "{version}",
  "integrity": "{integrity}"
}
"""),
    # --- uuid ---
    T("py_uuid_constant", "python", ["uuid"], """
{uuid_name} = "{uuid}"
"""),
    T("json_resource_id", "json", ["uuid"], """
{
  "id": "{uuid}",
  "name": "{service}",
  "createdAt": "{timestamp}"
}
"""),
    T("yaml_resource_uid", "yaml", ["uuid"], """
metadata:
  name: {service}
  namespace: {stage}
  uid: {uuid}
"""),
    T("js_request_id", "js_ts", ["uuid"], """
const requestId = "{uuid}";
logger.info({ requestId }, "processing {service} job");
"""),
    # --- commit_sha ---
    T("yaml_pinned_action", "yaml", ["commit_sha"], """
steps:
  - uses: actions/{action}@{sha}
  - run: {build_cmd}
"""),
    T("sh_git_checkout_sha", "shell", ["commit_sha"], """
git fetch origin {branch}
git {git_verb} {sha}
"""),
    T("json_build_info", "json", ["commit_sha"], """
{
  "version": "{version}",
  "commit": "{sha}",
  "branch": "{branch}"
}
"""),
    T("env_commit_sha", "env", ["commit_sha"], """
{sha_var}={sha}
"""),
    # --- public_key (more come from the PEM templates above) ---
    T("sh_authorized_keys", "shell", ["public_key"], """
mkdir -p ~/.ssh
echo "{ssh_public_key}" >> ~/.ssh/authorized_keys
"""),
    T("yaml_cloud_init_ssh", "yaml", ["public_key"], """
users:
  - name: {user}
    ssh_authorized_keys:
      - {ssh_public_key}
"""),
    # --- base64_non_secret ---
    T("py_embedded_image", "python", ["base64_non_secret"], """
{icon_name} = "{b64_image}"
"""),
    T("js_data_uri", "js_ts", ["base64_non_secret"], """
const {img_var} = "data:image/png;base64,{b64_image}";
"""),
    T("yaml_configmap_binary", "yaml", ["base64_non_secret"], """
kind: ConfigMap
metadata:
  name: {service}-assets
binaryData:
  {file_name}: {b64_image}
"""),
    T("json_thumbnail", "json", ["base64_non_secret"], """
{
  "title": "{title}",
  "thumbnail": "data:image/jpeg;base64,{b64_jpeg}"
}
"""),
    # --- empty_value (most come from value-slot templates above) ---
    T("docker_arg_no_default", "dockerfile", ["empty_value"], """
ARG {NAME}
"""),
    T("py_settings_optional", "python", ["empty_value"], """
class Settings(BaseSettings):
    {name}: Optional[str] = None
    debug: bool = False
"""),
    # --- keyword_non_secret: secret-sounding names with harmless values ---
    T("py_password_policy", "python", ["keyword_non_secret"], """
PASSWORD_MIN_LENGTH = {min_length}
PASSWORD_MAX_AGE_DAYS = {max_age_days}
TOKEN_EXPIRY_SECONDS = {ttl}
"""),
    T("py_cache_key", "python", ["keyword_non_secret"], """
primary_key = "{pk_field}"
cache_key = f"{cache_prefix}:{uid}"
"""),
    T("yaml_auth_settings", "yaml", ["keyword_non_secret"], """
auth:
  token_ttl_seconds: {ttl}
  password_min_length: {min_length}
  api_key_header: X-API-Key
"""),
    T("js_storage_keys", "js_ts", ["keyword_non_secret"], """
const TOKEN_STORAGE_KEY = "{storage_key}";
const PASSWORD_FIELD_ID = "password";
localStorage.setItem(TOKEN_STORAGE_KEY, response.data.token);
"""),
    T("env_keyword_settings", "env", ["keyword_non_secret"], """
SESSION_COOKIE_NAME={cookie_name}
PASSWORD_RESET_TTL={ttl}
API_KEY_HEADER=X-API-Key
"""),
    # --- db_url_no_secret: the password comes from the environment ---
    T("py_db_url_from_env", "python", ["db_url_no_secret"], """
import os
DB_USER = os.getenv("DB_USER", "{user}")
DATABASE_URL = f"postgresql://{DB_USER}:{os.environ['DB_PASSWORD']}@{host}:5432/{db}"
"""),
    T("yaml_compose_db_interpolated", "yaml", ["db_url_no_secret"], """
services:
  {service}:
    environment:
      DATABASE_URL: postgres://{user}:{pw_env_ref}@{host}:5432/{db}
"""),
    T("js_db_url_from_env", "js_ts", ["db_url_no_secret"], """
const url = `postgres://${process.env.DB_USER}:${process.env.DB_PASSWORD}@{host}:5432/{db}`;
const pool = new Pool({ connectionString: url });
"""),
    # --- plain_code ---
    T("py_plain_function", "python", ["plain_code"], """
def {func}(values):
    return [v * {factor} for v in values if v > {threshold}]
"""),
    T("js_plain_route", "js_ts", ["plain_code"], """
app.get("/{route}", async (req, res) => {
  const rows = await db.query("SELECT id, name FROM {table} LIMIT {limit}");
  res.json(rows);
});
"""),
    T("yaml_plain_deployment", "yaml", ["plain_code"], """
spec:
  replicas: {replicas}
  template:
    spec:
      containers:
        - image: ghcr.io/{org}/{service}:{version}
"""),
    T("docker_plain_steps", "dockerfile", ["plain_code"], """
RUN apt-get update && apt-get install -y --no-install-recommends {apt_pkg}
COPY . /app
EXPOSE {app_port}
"""),
    T("sh_plain_loop", "shell", ["plain_code"], """
for f in {glob}; do
  gzip -9 "$f"
done
"""),
]

# Harmless lines added before and after a snippet. They never contain secrets.
FILLER = {
    "python": (["import os", "import json", "import logging", "import requests"],
               ["logger = logging.getLogger(__name__)", "TIMEOUT_SECONDS = {timeout}", "DEBUG = False",
                "MAX_RETRIES = {retries}"]),
    "js_ts": (['"use strict";', 'const axios = require("axios");', 'import fs from "node:fs";'],
              ["const TIMEOUT_MS = {timeout_ms};", 'console.log("config loaded");']),
    "yaml": (["# {service} configuration"], ["log_level: {log_level}", "timeout: {timeout}"]),
    "env": (["# {service} settings", "NODE_ENV=production", "APP_ENV={stage}"],
            ["PORT={app_port}", "LOG_LEVEL={log_level}"]),
    "shell": (["#!/usr/bin/env bash", "set -euo pipefail"], ['echo "done"']),
    "dockerfile": (["FROM {base_image}", "WORKDIR /app"], ["COPY . .", "EXPOSE {app_port}", 'CMD ["./start.sh"]']),
    "json": ([], []),  # JSON snippets are complete objects, so they get no filler lines
}


# ---------------------------------------------------------------------------
# Filling templates
# ---------------------------------------------------------------------------

SLOT_PATTERN = re.compile(r"\{(\w+)\}")
DERIVED_SLOTS = {"value", "value_escaped", "name", "NAME", "camelName", "secret_id", "secret_path",
                 "env_ref", "py_env_read"}
LITERAL_BRACES = {"uid", "DB_USER"}  # real f-string fields inside skeletons, left as written


def fill_line(line, fill):
    """Fill the {slots} in one skeleton line.

    Returns a list of lines because a value can span several lines (a PEM
    block). Those extra lines keep the indentation of the original line.
    """
    indent = line[: len(line) - len(line.lstrip())]
    text = SLOT_PATTERN.sub(lambda m: m.group(0) if m.group(1) in LITERAL_BRACES else fill(m.group(1)), line)
    return [part.rstrip() for part in text.replace("\n", "\n" + indent).split("\n")]


def pick_some(pool, room):
    """Pick 0 to 2 lines from pool (never more than room), keeping their order."""
    count = rng.randint(0, min(2, len(pool), room))
    indices = sorted(rng.sample(range(len(pool)), count))
    return [pool[i] for i in indices]


def add_filler(lines, fmt, fill):
    before_pool, after_pool = FILLER[fmt]
    before = [fill_line(line, fill)[0] for line in pick_some(before_pool, MAX_LINES - len(lines))]
    after = [fill_line(line, fill)[0] for line in pick_some(after_pool, MAX_LINES - len(lines) - len(before))]
    # Skip filler that repeats a line already in the snippet (for example "import os").
    before = [line for line in before if line not in lines]
    after = [line for line in after if line not in lines]
    return before + lines + after


def render(template, category):
    """Fill one template for one category and return the snippet text."""
    kind = category if category in LEAK_TYPES else rng.choice(template.kinds)
    name_pool = "public_key" if category == "public_key" else kind
    scheme = rng.choice(template.schemes)
    filled = {}  # each slot is filled once, so a slot used twice gets the same value

    def fill(slot):
        if slot not in filled:
            filled[slot] = compute_slot(slot)
        return filled[slot]

    def compute_slot(slot):
        if slot == "value":
            return value_for(category, kind, scheme)
        if slot == "value_escaped":  # a PEM block on one line, as in JSON and .env files
            return fill("value").replace("\n", "\\n")
        if slot == "name":
            return rng.choice(NAMES[name_pool])
        if slot == "NAME":
            return fill("name").upper()
        if slot == "camelName":
            return camel_case(fill("name"))
        if slot == "secret_id":
            return fill("name").replace("_", "-")
        if slot == "secret_path":
            return f"{fill('stage')}/{fill('service')}/{fill('secret_id')}"
        if slot == "env_ref":
            return rng.choice(["$" + fill("NAME"), "${" + fill("NAME") + "}"])
        if slot == "py_env_read":
            env_name = fill("NAME")
            return rng.choice([f'os.getenv("{env_name}")', f'os.environ["{env_name}"]',
                               f'os.environ.get("{env_name}", "")'])
        return str(CONTEXT_SLOTS[slot]())

    lines = []
    for line in template.skeleton.split("\n"):
        lines += fill_line(line, fill)
    lines = add_filler(lines, template.fmt, fill)
    assert 1 <= len(lines) <= MAX_LINES, (template.template_id, lines)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checks and dataset building
# ---------------------------------------------------------------------------

def templates_for(category):
    return [t for t in TEMPLATES if category in t.categories]


def check_templates():
    ids = [t.template_id for t in TEMPLATES]
    assert len(ids) == len(set(ids)), "duplicate template_id"
    assert HELD_OUT_TEMPLATE_IDS <= set(ids), "unknown held-out template_id"
    known_slots = DERIVED_SLOTS | set(CONTEXT_SLOTS) | LITERAL_BRACES
    filler_lines = [line for before, after in FILLER.values() for line in before + after]
    for t in TEMPLATES:
        assert t.fmt in FORMATS, t.template_id
        assert all(c in LEAK_TYPES + SAFE_CATEGORIES for c in t.categories), t.template_id
        for slot in SLOT_PATTERN.findall(t.skeleton):
            assert slot in known_slots, (t.template_id, slot)
    for line in filler_lines:
        for slot in SLOT_PATTERN.findall(line):
            assert slot in CONTEXT_SLOTS, (line, slot)
    for category in LEAK_TYPES + SAFE_CATEGORIES:
        templates = templates_for(category)
        minimum = 4 if category in LEAK_TYPES else 3
        assert len(templates) >= minimum, f"{category} needs at least {minimum} templates"
        assert any(t.held_out for t in templates), f"{category} has no held-out template"
        assert any(not t.held_out for t in templates), f"{category} has no training template"


def print_inventory():
    held_ids = sorted(t.template_id for t in TEMPLATES if t.held_out)
    print(f"Templates: {len(TEMPLATES)} total, {len(held_ids)} held out "
          f"({len(held_ids) / len(TEMPLATES):.1%})")
    print(f"{'category':<20} {'templates':>9}  held-out template_ids")
    for category in LEAK_TYPES + SAFE_CATEGORIES:
        templates = templates_for(category)
        held = [t.template_id for t in templates if t.held_out]
        print(f"{category:<20} {len(templates):>9}  {', '.join(held)}")
    print("All held-out template_ids:", ", ".join(held_ids))


def label_for(category):
    return f"LEAKED: {category}" if category in LEAK_TYPES else "SAFE"


def build_splits():
    """Generate every example and assign it to train, valid or test.

    Examples from held-out templates go only to test. Examples from the other
    templates are shuffled and split between train, valid and test. A snippet
    that is identical (after whitespace normalisation) to any earlier one is
    thrown away and generated again, so no snippet appears twice anywhere.
    """
    splits = {"train": [], "valid": [], "test": []}
    seen_snippets = set()
    duplicates_removed = 0

    for category in LEAK_TYPES + SAFE_CATEGORIES:
        total = EXAMPLES_PER_LEAK_TYPE if category in LEAK_TYPES else EXAMPLES_PER_SAFE_CATEGORY
        n_train = round(total * TRAIN_FRACTION)
        n_valid = round(total * VALID_FRACTION)
        n_test = total - n_train - n_valid
        n_held_out = n_test // 2  # the other half of test comes from seen templates
        held = [t for t in templates_for(category) if t.held_out]
        seen = [t for t in templates_for(category) if not t.held_out]

        rows = []
        for i in range(total):
            pool = held if i < n_held_out else seen
            template = pool[i % len(pool)]  # round robin, so templates are used about equally
            for _ in range(MAX_ATTEMPTS):
                snippet = render(template, category)
                key = " ".join(snippet.split())
                if key not in seen_snippets:
                    break
                duplicates_removed += 1
            else:
                raise RuntimeError(f"{template.template_id} could not make a new {category} snippet")
            seen_snippets.add(key)
            rows.append({
                "snippet": snippet,
                "label": label_for(category),
                "template_id": template.template_id,
                "category": category,
                "format": template.fmt,
                "held_out": template.held_out,
            })

        held_rows, seen_rows = rows[:n_held_out], rows[n_held_out:]
        rng.shuffle(seen_rows)
        splits["train"] += seen_rows[:n_train]
        splits["valid"] += seen_rows[n_train:n_train + n_valid]
        splits["test"] += seen_rows[n_train + n_valid:] + held_rows

    for rows in splits.values():
        rng.shuffle(rows)
    assert not any(row["held_out"] for row in splits["train"] + splits["valid"])
    return splits, duplicates_removed


def write_split(name, rows, system_prompt):
    data_path = DATA_DIR / f"{name}.jsonl"
    meta_path = DATA_DIR / f"{name}_meta.jsonl"
    with open(data_path, "w", encoding="utf-8") as data_file, open(meta_path, "w", encoding="utf-8") as meta_file:
        for row in rows:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row["snippet"]},
                {"role": "assistant", "content": row["label"]},
            ]
            data_file.write(json.dumps({"messages": messages}) + "\n")
            meta = {key: row[key] for key in ["template_id", "category", "format", "held_out"]}
            meta_file.write(json.dumps(meta) + "\n")


def main():
    check_templates()
    print_inventory()
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    splits, duplicates_removed = build_splits()
    DATA_DIR.mkdir(exist_ok=True)
    print(f"\nDuplicate snippets removed (and regenerated): {duplicates_removed}")
    for name, rows in splits.items():
        write_split(name, rows, system_prompt)
        leaked = sum(row["label"] != "SAFE" for row in rows)
        print(f"{name}: {len(rows)} rows ({len(rows) - leaked} SAFE, {leaked} LEAKED) -> data/{name}.jsonl")


if __name__ == "__main__":
    main()
