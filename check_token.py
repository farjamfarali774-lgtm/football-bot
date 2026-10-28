"""Check the Telegram bot token, then optionally store it as a GitHub secret.

Nothing is printed except the bot's public name/id, so it is safe to share the
output. The token itself never appears on screen and never enters the shell
history.

    python check_token.py
"""

from __future__ import annotations

import getpass
import subprocess
import sys

import requests

REPO = "farjamfarali774-lgtm/football-bot"

sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    print("Paste the token from @BotFather (it stays hidden, press Enter after):")
    token = getpass.getpass("> ").strip().strip('"').strip("'")
    if not token:
        print("empty token, nothing to check")
        return 1

    if " " in token:
        print("PROBLEM: the token contains a space - it was probably pasted twice")
        return 1

    head, _, tail = token.partition(":")
    if not head.isdigit() or not tail:
        print("PROBLEM: this does not look like a bot token (expected 123456789:AA...)")
        return 1

    try:
        r = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=30)
        data = r.json()
    except requests.RequestException as exc:
        print(f"network error: {exc}")
        return 1

    if not data.get("ok"):
        print(f"REJECTED by Telegram: {data.get('description')}")
        print("-> the token is wrong or was revoked. Get a new one with /revoke in @BotFather.")
        print(f"   token starts with {head}:{tail[:4]}..., length {len(token)}")
        return 1

    bot = data["result"]
    print(f"\nOK  bot is @{bot['username']}  (id {bot['id']}, name: {bot['first_name']})")
    print("    This is the bot you must open in Telegram.")

    answer = input("\nStore it as the GitHub secret TELEGRAM_TOKEN? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        print("skipped")
        return 0

    proc = subprocess.run(
        ["gh", "secret", "set", "TELEGRAM_TOKEN", "--repo", REPO],
        input=token, text=True,
    )
    if proc.returncode != 0:
        print("gh failed - you can add the secret from the GitHub website instead")
        return 1
    print("secret updated ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
