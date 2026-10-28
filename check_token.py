"""Check the Telegram bot token, then optionally store it as a GitHub secret.

The token is read from token.txt (recommended, paste it there with Notepad) or
typed at a normal visible prompt. Only the bot's public name/id is printed, so
the output is safe to share.

    python check_token.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import requests

REPO = "farjamfarali774-lgtm/football-bot"
ROOT = Path(__file__).resolve().parent
TOKEN_FILE = ROOT / "token.txt"

sys.stdout.reconfigure(encoding="utf-8")


def read_token() -> tuple[str, bool]:
    """Return (token, came_from_file)."""
    if TOKEN_FILE.exists():
        raw = TOKEN_FILE.read_text(encoding="utf-8-sig").strip()
        print(f"reading token from {TOKEN_FILE.name}")
        return raw.strip().strip('"').strip("'"), True
    return input("Paste the token and press Enter: ").strip().strip('"').strip("'"), False


def main() -> int:
    token, from_file = read_token()
    if not token:
        print("empty token - token.txt is empty or nothing was typed")
        return 1

    if " " in token:
        print("PROBLEM: the token contains a space - it was pasted twice")
        return 1

    head, _, tail = token.partition(":")
    if not head.isdigit() or not tail:
        print("PROBLEM: this does not look like a bot token (expected 123456789:AA...)")
        print(f"   got {len(token)} characters starting with {token[:12]!r}")
        return 1

    try:
        r = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=30)
        data = r.json()
    except requests.RequestException as exc:
        print(f"network error: {exc}")
        return 1

    if not data.get("ok"):
        print(f"REJECTED by Telegram: {data.get('description')}")
        print("-> wrong or revoked token. Get a new one: @BotFather -> /mybots -> your bot -> API Token")
        print(f"   this one starts with {head}:{tail[:4]}... and is {len(token)} characters long")
        return 1

    bot = data["result"]
    print(f"OK  bot is @{bot['username']}  (id {bot['id']}, name: {bot['first_name']})")
    print("    open exactly this bot in Telegram")

    answer = input("Store it as the GitHub secret TELEGRAM_TOKEN? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        print("skipped")
        return 0

    proc = subprocess.run(
        ["gh", "secret", "set", "TELEGRAM_TOKEN", "--repo", REPO],
        input=token, text=True,
    )
    if proc.returncode != 0:
        print("gh failed - add the secret from the GitHub website instead")
        return 1

    print("secret updated")
    if from_file:
        TOKEN_FILE.unlink(missing_ok=True)
        print("token.txt deleted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
