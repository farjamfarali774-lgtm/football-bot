"""One-time local setup: store the bot token and lock the bot to one user.

Run this yourself (double-click `setup_local.bat`), because the token must be
typed by you - it should never be pasted into a chat or committed to the repo.

    python setup_local.py

The token is typed into a hidden prompt, checked against Telegram, and written
to `.env` (which is gitignored).
"""

from __future__ import annotations

import getpass
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
ENV = ROOT / ".env"


def read_env() -> dict[str, str]:
    out: dict[str, str] = {}
    if ENV.exists():
        for line in ENV.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.split("=", 1)
                out[key.strip()] = value.strip()
    return out


def main() -> None:
    current = read_env()

    token = current.get("TELEGRAM_TOKEN", "")
    if not token:
        print("Paste the token from @BotFather (nothing is shown while you type).")
        token = getpass.getpass("TELEGRAM_TOKEN: ").strip()
    if not token:
        raise SystemExit("no token given - nothing changed")

    answer = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=30).json()
    if not answer.get("ok"):
        raise SystemExit(f"Telegram rejected the token: {answer.get('description')}")
    me = answer["result"]
    print(f"ok - the token belongs to @{me['username']} ({me.get('first_name', '')})")

    allowed = current.get("ALLOWED_USER_IDS", "")
    if not allowed:
        value = input(
            "Your Telegram user id (send /id to the bot; leave empty to keep it open): "
        ).strip()
        if value and not (value.isdigit() and len(value) <= 12):
            print(f"'{value}' is not a valid Telegram id (about 10 digits) - leaving the bot open")
        elif value:
            allowed = value

    ENV.write_text(
        f"TELEGRAM_TOKEN={token}\nALLOWED_USER_IDS={allowed}\n",
        encoding="utf-8",
    )
    print(f"wrote {ENV.name}")
    if allowed:
        print(f"bot is locked to user id {allowed}")
    else:
        print("bot is still open to everyone who knows its name")
    print("now start it with run_bot.bat (or install_task.bat to start it at every login)")


if __name__ == "__main__":
    main()
