#!/usr/bin/env python3
"""NYC Apartment Bot — Setup Wizard.

Interactive wizard that guides through setting up all external services,
configuring Modal secrets, deploying the app, and registering the webhook.

Usage:
    python setup_wizard.py            # Run the full wizard
    python setup_wizard.py --check    # Check current setup status
    python setup_wizard.py --force    # Reconfigure even if already set up
    python setup_wizard.py --dry-run  # Run through prompts/validation without changing anything

Requires Python 3.9+. No dependencies beyond stdlib (runs before pip install).
"""

import argparse
import getpass
import json
import os
import re
import ssl
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODAL_APP_NAME = "streeteasy-bot"
MODAL_APP_FILE = "modal_app.py"

SECRETS = {
    "anthropic": {
        "env_var": "ANTHROPIC_API_KEY",
        "display": "Anthropic API Key",
    },
    "apify": {
        "env_var": "APIFY_API_TOKEN",
        "display": "Apify API Token",
    },
    "streeteasy-telegram": {
        "env_var": "SE_TELEGRAM_BOT_TOKEN",
        "display": "Telegram Bot Token",
    },
}

APIFY_ACTOR = "memo23/apify-streeteasy-cheerio"

URLS = {
    "anthropic_keys": "https://console.anthropic.com/settings/keys",
    "apify_integrations": "https://console.apify.com/settings/integrations",
    "apify_actor": "https://apify.com/memo23/apify-streeteasy-cheerio",
    "telegram_botfather": "https://t.me/BotFather",
    "modal_signup": "https://modal.com/signup",
}

# Packages to check from requirements.txt (import name, pip name)
REQUIRED_PACKAGES = [
    ("modal", "modal"),
    ("pydantic", "pydantic"),
    ("httpx", "httpx"),
    ("anthropic", "anthropic"),
    ("fastapi", "fastapi"),
    ("apify_client", "apify-client"),
    ("telegraph", "telegraph"),
    ("pytest", "pytest"),
    ("pytest_asyncio", "pytest-asyncio"),
]


# ---------------------------------------------------------------------------
# Colors / Terminal Output
# ---------------------------------------------------------------------------

class Colors:
    """ANSI color codes, auto-disabled when not a TTY or NO_COLOR is set."""

    def __init__(self):
        use_color = (
            hasattr(sys.stdout, "isatty")
            and sys.stdout.isatty()
            and not os.environ.get("NO_COLOR")
        )
        if use_color:
            self.RESET = "\033[0m"
            self.BOLD = "\033[1m"
            self.DIM = "\033[2m"
            self.GREEN = "\033[32m"
            self.RED = "\033[31m"
            self.YELLOW = "\033[33m"
            self.BLUE = "\033[34m"
            self.CYAN = "\033[36m"
        else:
            self.RESET = ""
            self.BOLD = ""
            self.DIM = ""
            self.GREEN = ""
            self.RED = ""
            self.YELLOW = ""
            self.BLUE = ""
            self.CYAN = ""

    def ok(self, msg: str) -> str:
        return f"  {self.GREEN}[ok]{self.RESET} {msg}"

    def fail(self, msg: str) -> str:
        return f"  {self.RED}[!!]{self.RESET} {msg}"

    def skip(self, msg: str) -> str:
        return f"  {self.YELLOW}[--]{self.RESET} {msg}"

    def info(self, msg: str) -> str:
        return f"  {self.BLUE}[..]{self.RESET} {msg}"

    def header(self, step: int, total: int, title: str) -> str:
        line = "\u2500" * 40
        return (
            f"\n  {self.DIM}{line}{self.RESET}\n"
            f"  {self.BOLD}[{step}/{total}] {title}{self.RESET}\n"
            f"  {self.DIM}{line}{self.RESET}\n"
        )

    def banner(self, dry_run: bool = False) -> str:
        suffix = f"  {self.YELLOW}(dry run — no changes will be made){self.RESET}\n" if dry_run else ""
        return (
            f"\n  {self.BOLD}NYC Apartment Bot \u2014 Setup Wizard{self.RESET}\n"
            + suffix
        )

    def dry(self, msg: str) -> str:
        return f"  {self.CYAN}[~~]{self.RESET} {msg}"


C = Colors()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class WizardState:
    """Accumulates values across wizard steps."""
    telegram_token: Optional[str] = None
    bot_username: Optional[str] = None
    force: bool = False
    dry_run: bool = False
    # Track which steps completed
    completed_steps: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Input Helpers
# ---------------------------------------------------------------------------

def ask_yes_no(prompt: str, default: bool = True) -> bool:
    """Ask a Y/n question. Returns bool."""
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"  {prompt} {suffix} ").strip().lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("  Please enter y or n.")


def ask_secret(prompt: str) -> str:
    """Prompt for secret input (hidden). Falls back to regular input."""
    try:
        value = getpass.getpass(f"  {prompt}: ")
    except EOFError:
        value = input(f"  {prompt}: ")
    return value.strip()


def ask_input(prompt: str) -> str:
    """Prompt for regular input."""
    return input(f"  {prompt}: ").strip()


# ---------------------------------------------------------------------------
# Subprocess Helpers
# ---------------------------------------------------------------------------

def run_cmd(
    args: List[str],
    capture: bool = True,
    timeout: int = 120,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
) -> Tuple[int, str, str]:
    """Run a subprocess command. Returns (returncode, stdout, stderr)."""
    run_env = None
    if env:
        run_env = {**os.environ, **env}

    if cwd is None:
        cwd = PROJECT_DIR

    try:
        result = subprocess.run(
            args,
            capture_output=capture,
            text=True,
            timeout=timeout,
            env=run_env,
            cwd=cwd,
        )
        return result.returncode, result.stdout or "", result.stderr or ""
    except subprocess.TimeoutExpired:
        return 1, "", f"Command timed out after {timeout}s"
    except FileNotFoundError:
        return 1, "", f"Command not found: {args[0]}"


def run_interactive(
    args: List[str],
    timeout: int = 300,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
) -> int:
    """Run a command interactively (output goes to terminal). Returns exit code."""
    run_env = None
    if env:
        run_env = {**os.environ, **env}

    if cwd is None:
        cwd = PROJECT_DIR

    try:
        result = subprocess.run(
            args,
            timeout=timeout,
            env=run_env,
            cwd=cwd,
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        print(C.fail(f"Command timed out after {timeout}s"))
        return 1
    except FileNotFoundError:
        print(C.fail(f"Command not found: {args[0]}"))
        return 1


# Project directory (where this script lives)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Detection Functions
# ---------------------------------------------------------------------------

def check_packages() -> Tuple[bool, List[str]]:
    """Check if required Python packages are importable. Returns (all_ok, missing)."""
    missing = []
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    return len(missing) == 0, missing


def check_modal_auth() -> bool:
    """Check if Modal CLI is authenticated."""
    code, stdout, _ = run_cmd(["modal", "profile", "current"])
    return code == 0 and bool(stdout.strip())


def check_modal_secret(secret_name: str) -> bool:
    """Check if a Modal secret exists by parsing `modal secret list`."""
    code, stdout, _ = run_cmd(["modal", "secret", "list"])
    if code != 0:
        return False
    # Parse table output — look for the secret name in the Name column
    for line in stdout.splitlines():
        # Table rows have │ separators; the name is in the first data column
        parts = [p.strip() for p in line.split("│") if p.strip()]
        if parts and parts[0] == secret_name:
            return True
    return False


def check_deployment() -> bool:
    """Check if the Modal app is deployed."""
    code, stdout, _ = run_cmd(["modal", "app", "list"])
    if code != 0:
        return False
    for line in stdout.splitlines():
        if "streeteasy-" in line.lower() or MODAL_APP_NAME in line.lower():
            return True
    return False


# ---------------------------------------------------------------------------
# Validation Functions (via urllib — no deps needed)
# ---------------------------------------------------------------------------

def _make_request(
    url: str,
    data: Optional[bytes] = None,
    headers: Optional[Dict[str, str]] = None,
    method: str = "GET",
) -> Tuple[int, str]:
    """Make an HTTP request using urllib. Returns (status_code, body)."""
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except ssl.SSLCertVerificationError:
        # macOS sometimes has stale SSL certs — try without verification
        ctx = ssl._create_unverified_context()
        try:
            with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                return resp.status, resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError) as e:
        return 0, str(e)


def validate_format_anthropic(key: str) -> Optional[str]:
    """Check Anthropic key format. Returns error message or None."""
    if not key.startswith("sk-ant-"):
        return "Key should start with 'sk-ant-'"
    if len(key) < 20:
        return "Key seems too short"
    return None


def validate_format_apify(token: str) -> Optional[str]:
    """Check Apify token format. Returns error message or None."""
    if not token.startswith("apify_api_"):
        return "Token should start with 'apify_api_'"
    if len(token) < 15:
        return "Token seems too short"
    return None


def validate_format_telegram(token: str) -> Optional[str]:
    """Check Telegram bot token format. Returns error message or None."""
    if not re.match(r"\d{8,}:[A-Za-z0-9_-]{30,}", token):
        return "Token should match format: 123456789:ABCdef... (number:alphanumeric)"
    return None


def validate_anthropic_key(key: str) -> Tuple[bool, str]:
    """Validate Anthropic API key by making a minimal API call."""
    data = json.dumps({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}],
    }).encode("utf-8")
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    status, body = _make_request(
        "https://api.anthropic.com/v1/messages",
        data=data,
        headers=headers,
        method="POST",
    )
    if status == 0:
        return False, f"Network error: {body}"
    if status == 200:
        return True, "Key is valid"
    if status == 401:
        return False, "Invalid API key"
    if status == 403:
        return False, "Key lacks permissions"
    return False, f"Unexpected response (HTTP {status})"


def validate_apify_token(token: str) -> Tuple[bool, str]:
    """Validate Apify token and check actor access."""
    # Check token validity
    status, body = _make_request(f"https://api.apify.com/v2/users/me?token={token}")
    if status == 0:
        return False, f"Network error: {body}"
    if status != 200:
        return False, "Invalid API token"

    # Check actor access
    actor_id = APIFY_ACTOR.replace("/", "~")
    status, body = _make_request(
        f"https://api.apify.com/v2/acts/{actor_id}?token={token}"
    )
    if status == 0:
        return True, "Token valid (could not verify actor access — network error)"
    if status != 200:
        return True, f"Token valid, but actor '{APIFY_ACTOR}' not accessible (HTTP {status}). Make sure you've rented it."
    return True, "Token valid, actor accessible"


def validate_telegram_token(token: str) -> Tuple[bool, str, Optional[str]]:
    """Validate Telegram bot token. Returns (valid, message, bot_username)."""
    status, body = _make_request(f"https://api.telegram.org/bot{token}/getMe")
    if status == 0:
        return False, f"Network error: {body}", None
    if status != 200:
        return False, "Invalid bot token", None
    try:
        data = json.loads(body)
        username = data.get("result", {}).get("username", "unknown")
        return True, f"Valid — bot is @{username}", username
    except (json.JSONDecodeError, KeyError):
        return True, "Token accepted but could not parse response", None


# ---------------------------------------------------------------------------
# Wizard Steps
# ---------------------------------------------------------------------------

TOTAL_STEPS = 8


def step_1_dependencies(state: WizardState) -> bool:
    """Step 1: Install Python dependencies."""
    print(C.header(1, TOTAL_STEPS, "Install Dependencies"))

    all_ok, missing = check_packages()
    if all_ok and not state.force:
        print(C.ok("All Python packages are installed"))
        return True

    if missing:
        print(C.info(f"Missing packages: {', '.join(missing)}"))
    else:
        print(C.ok("All packages installed, but --force was specified"))

    if not ask_yes_no("Install from requirements.txt?"):
        print(C.skip("Skipped"))
        return all_ok

    req_file = os.path.join(PROJECT_DIR, "requirements.txt")
    if not os.path.exists(req_file):
        print(C.fail("requirements.txt not found"))
        return False

    if state.dry_run:
        print(C.dry(f"Would run: {sys.executable} -m pip install -r requirements.txt"))
        return True

    print(C.info("Running pip install..."))
    exit_code = run_interactive(
        [sys.executable, "-m", "pip", "install", "-r", req_file]
    )
    if exit_code != 0:
        print(C.fail("pip install failed"))
        return False

    print(C.ok("Dependencies installed"))
    return True


def step_2_modal_auth(state: WizardState) -> bool:
    """Step 2: Authenticate with Modal."""
    print(C.header(2, TOTAL_STEPS, "Authenticate with Modal"))

    if check_modal_auth() and not state.force:
        print(C.ok("Modal CLI is authenticated"))
        return True

    print(C.info("Modal CLI is not authenticated"))
    print()
    print(f"  This will open your browser to sign in to Modal.")
    print(f"  If you don't have an account: {URLS['modal_signup']}")
    print()

    if not ask_yes_no("Run 'modal token new'?"):
        print(C.skip("Skipped"))
        return False

    if state.dry_run:
        print(C.dry("Would run: modal token new"))
        return True

    exit_code = run_interactive(["modal", "token", "new"], timeout=120)
    if exit_code != 0:
        print(C.fail("Modal authentication failed"))
        return False

    if not check_modal_auth():
        print(C.fail("Modal still not authenticated after token setup"))
        return False

    print(C.ok("Modal authenticated"))
    return True


def _configure_secret(
    state: WizardState,
    step: int,
    secret_name: str,
    env_var: str,
    title: str,
    instructions: str,
    format_validator: Any,
    api_validator: Any,
    post_validate: Any = None,
) -> bool:
    """Generic secret configuration step."""
    print(C.header(step, TOTAL_STEPS, title))

    exists = check_modal_secret(secret_name)
    if exists and not state.force:
        print(C.ok(f"Secret '{secret_name}' already exists"))
        if not ask_yes_no("Reconfigure?", default=False):
            return True

    print(textwrap.dedent(instructions).strip())
    print()

    # Get the secret value
    value = ask_secret("Paste your key/token")
    if not value:
        print(C.skip("No value entered, skipping"))
        return exists  # OK if it already existed

    # Format validation
    fmt_err = format_validator(value)
    if fmt_err:
        print(C.fail(f"Format issue: {fmt_err}"))
        if not ask_yes_no("Continue anyway?", default=False):
            return exists

    # API validation
    print(C.info("Validating..."), end="", flush=True)
    result = api_validator(value)

    # Handle Telegram's 3-tuple return
    if len(result) == 3:
        valid, msg, extra = result
    else:
        valid, msg = result
        extra = None

    if valid:
        print(f"\r{C.ok(msg)}")
    else:
        print(f"\r{C.fail(msg)}")
        if not ask_yes_no("Save anyway?", default=False):
            return exists

    # Post-validate callback (e.g. store bot username)
    if post_validate:
        post_validate(state, value, extra)

    # Create Modal secret
    if state.dry_run:
        print(C.dry("Would run: modal secret create %s %s=*** --force" % (secret_name, env_var)))
        return True

    print(C.info("Creating Modal secret..."), end="", flush=True)
    code, _, stderr = run_cmd([
        "modal", "secret", "create", secret_name,
        f"{env_var}={value}",
        "--force",
    ])

    if code != 0:
        print(f"\r{C.fail('Failed to create secret')}")
        if stderr.strip():
            print(f"  {stderr.strip()}")
        return False

    msg = 'Secret "%s" saved' % secret_name
    print("\r" + C.ok(msg))
    return True


def step_3_anthropic(state: WizardState) -> bool:
    """Step 3: Configure Anthropic API key."""
    return _configure_secret(
        state=state,
        step=3,
        secret_name="anthropic",
        env_var="ANTHROPIC_API_KEY",
        title="Anthropic API Key",
        instructions=f"""
    The bot uses Claude for conversations and listing scoring.

    1. Go to: {URLS['anthropic_keys']}
    2. Create a new API key (starts with "sk-ant-")
        """,
        format_validator=validate_format_anthropic,
        api_validator=validate_anthropic_key,
    )


def step_4_apify(state: WizardState) -> bool:
    """Step 4: Configure Apify API token."""
    return _configure_secret(
        state=state,
        step=4,
        secret_name="apify",
        env_var="APIFY_API_TOKEN",
        title="Apify API Token",
        instructions=f"""
    The bot scrapes StreetEasy via the Apify platform.

    1. Go to: {URLS['apify_integrations']}
    2. Copy your Personal API Token (starts with "apify_api_")
    3. Make sure you've rented the actor: {URLS['apify_actor']}
        """,
        format_validator=validate_format_apify,
        api_validator=validate_apify_token,
    )


def _post_validate_telegram(state: WizardState, token: str, username: Optional[str]):
    """Store telegram token and bot username after validation."""
    state.telegram_token = token
    if username:
        state.bot_username = username


def step_5_telegram(state: WizardState) -> bool:
    """Step 5: Configure Telegram bot token."""
    return _configure_secret(
        state=state,
        step=5,
        secret_name="streeteasy-telegram",
        env_var="SE_TELEGRAM_BOT_TOKEN",
        title="Telegram Bot Token",
        instructions=f"""
    The bot communicates via Telegram.

    1. Open Telegram and message @BotFather: {URLS['telegram_botfather']}
    2. Send /newbot and follow the prompts
    3. Copy the bot token (format: 123456789:ABCdef...)
        """,
        format_validator=validate_format_telegram,
        api_validator=validate_telegram_token,
        post_validate=_post_validate_telegram,
    )


def step_6_deploy(state: WizardState) -> bool:
    """Step 6: Deploy to Modal."""
    print(C.header(6, TOTAL_STEPS, "Deploy to Modal"))

    deployed = check_deployment()
    if deployed and not state.force:
        print(C.ok("App is already deployed"))
        if not ask_yes_no("Redeploy?", default=False):
            return True

    app_file = os.path.join(PROJECT_DIR, MODAL_APP_FILE)
    if not os.path.exists(app_file):
        print(C.fail(f"{MODAL_APP_FILE} not found in {PROJECT_DIR}"))
        return False

    if state.dry_run:
        print(C.dry("Would run: modal deploy %s" % MODAL_APP_FILE))
        return True

    print(C.info("Deploying to Modal (this may take a minute)..."))
    print()

    exit_code = run_interactive(["modal", "deploy", MODAL_APP_FILE])
    if exit_code != 0:
        print()
        print(C.fail("Deployment failed"))
        return False

    print()
    print(C.ok("Deployed successfully"))
    return True


def step_7_webhook(state: WizardState) -> bool:
    """Step 7: Set up Telegram webhook."""
    print(C.header(7, TOTAL_STEPS, "Set Up Telegram Webhook"))

    # We need the telegram token to pass as env var
    token = state.telegram_token
    if not token:
        print(C.info("Telegram token not available from step 5"))
        token = ask_secret("Paste your Telegram bot token")
        if not token:
            print(C.skip("No token provided, skipping"))
            return False
        state.telegram_token = token
        # Also validate and get username if we don't have it
        if not state.bot_username:
            valid, msg, username = validate_telegram_token(token)
            if valid and username:
                state.bot_username = username

    if state.dry_run:
        print(C.dry("Would run: SE_TELEGRAM_BOT_TOKEN=*** modal run %s::setup" % MODAL_APP_FILE))
        return True

    print(C.info("Registering webhook with Telegram..."))
    print()

    exit_code = run_interactive(
        ["modal", "run", f"{MODAL_APP_FILE}::setup"],
        env={"SE_TELEGRAM_BOT_TOKEN": token},
    )

    if exit_code != 0:
        print()
        print(C.fail("Webhook setup failed"))
        return False

    print()
    print(C.ok("Webhook registered"))
    return True


def step_8_verify(state: WizardState) -> bool:
    """Step 8: Verify setup."""
    print(C.header(8, TOTAL_STEPS, "Verify Setup"))

    if state.bot_username:
        print(f"  Your bot is @{state.bot_username}")
        print(f"  Open: https://t.me/{state.bot_username}")
    else:
        print("  Open Telegram and find your bot.")

    print()
    print("  Send it a message like \"Hello\" to verify it's working.")
    print()

    if ask_yes_no("Did the bot respond?"):
        print()
        print(C.ok("Setup complete! Your bot is live."))
        print()
        print("  Next steps:")
        print("  - Tell the bot your apartment preferences")
        print("  - It will search StreetEasy and send daily matches")
        print(f"  - View logs: modal app logs {MODAL_APP_NAME}")
        return True
    else:
        print()
        print(C.fail("Bot didn't respond. Troubleshooting:"))
        print(f"  - Check logs: modal app logs {MODAL_APP_NAME}")
        print("  - Verify the webhook: re-run step 7")
        print("  - Make sure all secrets are configured: python setup_wizard.py --check")
        return False


# ---------------------------------------------------------------------------
# Check Mode
# ---------------------------------------------------------------------------

def run_check():
    """Run all detection checks and print status report."""
    print(C.banner())
    print("  Checking current setup...\n")

    # Dependencies
    all_ok, missing = check_packages()
    if all_ok:
        print(C.ok("Python dependencies"))
    else:
        print(C.fail(f"Python dependencies — missing: {', '.join(missing)}"))

    # Modal auth
    if check_modal_auth():
        print(C.ok("Modal CLI authenticated"))
    else:
        print(C.skip("Modal CLI — not authenticated"))

    # Secrets
    for name, info in SECRETS.items():
        if check_modal_secret(name):
            print(C.ok(f"Secret '{name}' — exists"))
        else:
            print(C.skip(f"Secret '{name}' — not found"))

    # Deployment
    if check_deployment():
        print(C.ok("App deployed"))
    else:
        print(C.skip("App not deployed"))

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_first_incomplete_step(state: WizardState) -> int:
    """Check which steps are already done, print status, return first incomplete step."""
    print("  Checking current setup...\n")

    status = {}

    # Step 1: Dependencies
    all_ok, missing = check_packages()
    status[1] = all_ok
    if all_ok:
        print(C.ok("Python dependencies"))
    else:
        print(C.skip(f"Python dependencies — missing: {', '.join(missing)}"))

    # Step 2: Modal auth
    authed = check_modal_auth()
    status[2] = authed
    if authed:
        print(C.ok("Modal CLI authenticated"))
    else:
        print(C.skip("Modal CLI — not authenticated"))

    # Steps 3-5: Secrets
    secret_steps = [
        (3, "anthropic"),
        (4, "apify"),
        (5, "streeteasy-telegram"),
    ]
    for step_num, name in secret_steps:
        exists = check_modal_secret(name)
        status[step_num] = exists
        if exists:
            print(C.ok(f"Secret '{name}' — exists"))
        else:
            print(C.skip(f"Secret '{name}' — not found"))

    # Step 6: Deployment
    deployed = check_deployment()
    status[6] = deployed
    if deployed:
        print(C.ok("App deployed"))
    else:
        print(C.skip("App not deployed"))

    # Steps 7-8 can't be detected reliably
    status[7] = False
    status[8] = False

    print()

    if state.force:
        return 1

    # Find first incomplete step
    for step in range(1, TOTAL_STEPS + 1):
        if not status.get(step, False):
            return step

    # All detectable steps done — start from webhook
    return 7


def run_wizard(state: WizardState):
    """Run the interactive setup wizard."""
    print(C.banner(dry_run=state.dry_run))

    first_step = find_first_incomplete_step(state)

    if first_step > 1 and not state.force:
        steps_done = first_step - 1
        print(f"  Steps 1-{steps_done} already complete. ", end="")
        if not ask_yes_no(f"Start from step {first_step}?"):
            first_step = 1

    steps = [
        (1, step_1_dependencies),
        (2, step_2_modal_auth),
        (3, step_3_anthropic),
        (4, step_4_apify),
        (5, step_5_telegram),
        (6, step_6_deploy),
        (7, step_7_webhook),
        (8, step_8_verify),
    ]

    for step_num, step_func in steps:
        if step_num < first_step:
            continue

        try:
            success = step_func(state)
            if success:
                state.completed_steps.append(step_num)
        except KeyboardInterrupt:
            print("\n")
            print(C.skip(f"Step {step_num} interrupted"))
            print()
            if step_num < TOTAL_STEPS:
                if not ask_yes_no("Continue to next step?", default=False):
                    print("\n  Exiting. Re-run to resume where you left off.\n")
                    return


def main():
    parser = argparse.ArgumentParser(
        description="NYC Apartment Bot — Setup Wizard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check current setup status without making changes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reconfigure all steps even if already set up",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run through prompts and validation without making any changes",
    )
    args = parser.parse_args()

    if args.check:
        run_check()
        return

    state = WizardState(force=args.force, dry_run=args.dry_run)

    try:
        run_wizard(state)
    except KeyboardInterrupt:
        print("\n\n  Exiting. Re-run to resume where you left off.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
