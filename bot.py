"""Telegram bot: today's fixtures with model probabilities.

Two run modes:

    python bot.py                 # long polling, runs forever (laptop / VPS)
    python bot.py --listen 280    # poll for 280 s then exit (GitHub Actions)
    python bot.py --once          # read pending messages once and exit
    python bot.py --self-test     # print today's predictions, no Telegram

Configuration comes from the environment (a local .env file is loaded if present):

    TELEGRAM_TOKEN    bot token from @BotFather
    ALLOWED_USER_IDS  comma separated Telegram user ids allowed to use the bot

If ALLOWED_USER_IDS is empty the bot answers everyone with their own id so the
owner can collect it once, then lock the bot down.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

import predict_fixtures as PF
from fetch_data import LEAGUES, load_matches
from model import DixonColes

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
STATE = ROOT / ".state" / "offset.json"
TEHRAN = ZoneInfo("Asia/Tehran")


def load_env() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()
TOKEN = os.environ.get("TELEGRAM_TOKEN", "").strip().strip('"').strip("'")
ALLOWED = {int(x) for x in re.split(r"[,\s]+", os.environ.get("ALLOWED_USER_IDS", "")) if x.strip().isdigit()}
API = f"https://api.telegram.org/bot{TOKEN}"


# ------------------------------------------------------------------ telegram io
def api(method: str, **params):
    if not TOKEN:
        raise SystemExit("TELEGRAM_TOKEN is not set")
    r = requests.get(f"{API}/{method}", params=params, timeout=60)
    return r.json()


def split_message(text: str, limit: int = 3800) -> list[str]:
    """Telegram caps messages at 4096 chars.

    A match card is a <blockquote> spanning several lines, so split between
    cards, never inside one - otherwise the HTML is unbalanced.
    """
    if len(text) <= limit:
        return [text]

    atoms: list[str] = []
    buffer: list[str] = []
    inside = False
    for line in text.split("\n"):
        if line.startswith("<blockquote>"):
            inside = True
        buffer.append(line)
        if inside and line.endswith("</blockquote>"):
            inside = False
        if not inside:
            atoms.append("\n".join(buffer))
            buffer = []
    if buffer:
        atoms.append("\n".join(buffer))

    chunks: list[str] = []
    current = ""
    for atom in atoms:
        candidate = f"{current}\n{atom}" if current else atom
        if current and len(candidate) > limit:
            chunks.append(current)
            current = atom
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def send(chat_id: int, text: str) -> None:
    for chunk in split_message(text or "—"):
        api("sendMessage", chat_id=chat_id, text=chunk,
            parse_mode="HTML", disable_web_page_preview=True)


# ------------------------------------------------------------------ predictions
def _form(train: pd.DataFrame, team: str, n: int = 5) -> str:
    """Last n results, newest first: ✅ win, ➖ draw, ❌ loss."""
    games = train[(train["home"] == team) | (train["away"] == team)].sort_values("date").tail(n)
    marks = []
    for _, g in games.iterrows():
        if g["ftr"] == "D":
            marks.append("➖")
        elif (g["ftr"] == "H") == (g["home"] == team):
            marks.append("✅")
        else:
            marks.append("❌")
    return "".join(reversed(marks)) or "—"


def market_rows(fx: pd.DataFrame) -> pd.DataFrame:
    """Fit one model per league and predict every fixture in it."""
    if fx.empty:
        return pd.DataFrame()
    ref = pd.Timestamp.now(TEHRAN).normalize().tz_localize(None)
    out = []
    for _, group in fx.groupby("league"):
        div = group["div"].iloc[0]
        train = load_matches([div])
        train = train[(train["date"] < ref) & (train["date"] >= ref - pd.Timedelta(days=900))]
        if len(train) < 30:
            continue
        model = DixonColes().fit(train, ref)
        for team, spec in (PF.ADJ.get("adjustments") or {}).items():
            model.set_adjustment(team, spec.get("kind", "attack"), float(spec.get("factor", 1.0)))

        counts = pd.concat([train["home"], train["away"]]).value_counts()
        forms = {t: _form(train, t) for t in set(group["home"]) | set(group["away"])}
        for _, r in group.iterrows():
            row = PF.build_row(r.to_dict(), model.predict(r["home"], r["away"]))
            n_min = min(int(counts.get(r["home"], 0)), int(counts.get(r["away"], 0)))
            w = min(1.0, n_min / 30.0)
            row["final_home"] = w * row["p_home"] + (1 - w) * row["mkt_home"]
            row["final_draw"] = w * row["p_draw"] + (1 - w) * row["mkt_draw"]
            row["final_away"] = w * row["p_away"] + (1 - w) * row["mkt_away"]
            row["low_data"] = n_min < 15
            row["time"] = r.get("time", "")
            row["form_home"] = forms.get(r["home"], "—")
            row["form_away"] = forms.get(r["away"], "—")
            out.append(row)
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_values(["date", "time"]).reset_index(drop=True)


def pc(x: float) -> str:
    return "—" if pd.isna(x) else f"{x * 100:.0f}%"


LEAGUE_FLAG = {
    "Premier League": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Championship": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "League One": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "League Two": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "National League": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Scottish Premiership": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "Scottish Championship": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "Scottish League One": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "Scottish League Two": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "Bundesliga": "🇩🇪",
    "2. Bundesliga": "🇩🇪",
    "Serie A": "🇮🇹",
    "Serie B": "🇮🇹",
    "La Liga": "🇪🇸",
    "Segunda Division": "🇪🇸",
    "Ligue 1": "🇫🇷",
    "Ligue 2": "🇫🇷",
    "Eredivisie": "🇳🇱",
    "Belgian Pro League": "🇧🇪",
    "Primeira Liga": "🇵🇹",
    "Super Lig": "🇹🇷",
    "Greek Super League": "🇬🇷",
}


def odds_text(r) -> str:
    parts = []
    for key in ("odds_h", "odds_d", "odds_a"):
        value = r.get(key)
        parts.append("—" if pd.isna(value) else f"{value:.2f}")
    return "/".join(parts)


def pick_text(r) -> str:
    """Plain-language answer to 'who wins?'."""
    options = {
        "home": (r["final_home"], f"برد {r['home']}"),
        "draw": (r["final_draw"], "مساوی"),
        "away": (r["final_away"], f"برد {r['away']}"),
    }
    best = max(options, key=lambda k: options[k][0])
    chance, text = options[best]
    return f"{text} {pc(chance)}"


def odd(r, key: str) -> str:
    value = r.get(key)
    return "—" if value is None or pd.isna(value) else f"{value:.2f}"


def format_day(rows: pd.DataFrame, title: str) -> str:
    if rows.empty:
        return f"{title}\n\nبرنامه‌ای برای این بازه در لیگ‌های پوشش‌داده‌شده پیدا نشد."

    parts = [
        f"<b>{title}</b>",
        f"{len(rows)} بازی · همهٔ ساعت‌ها به وقت تهران 🕒",
        "🏠 میزبان · ✈️ مهمان · ➜ محتمل‌ترین نتیجه",
    ]
    for _, r in rows.iterrows():
        home = html.escape(str(r["home"]))
        away = html.escape(str(r["away"]))
        league = html.escape(str(r["league"]))
        flag = LEAGUE_FLAG.get(str(r["league"]), "⚽️")
        warn = " ⚠" if r.get("low_data") else ""
        goals = r.get("exp_goals")
        goals_text = "—" if goals is None or pd.isna(goals) else f"{goals:.1f}"
        parts.append(
            "<blockquote>"
            f"🕒 <b>{r['time'] or '--:--'}</b> · {flag} {league}{warn}\n"
            f"🏠 <b>{home}</b> — برد {pc(r['final_home'])} · ضریب {odd(r, 'odds_h')}\n"
            f"✈️ <b>{away}</b> — برد {pc(r['final_away'])} · ضریب {odd(r, 'odds_a')}\n"
            f"📈 فرم {home}: {r.get('form_home', '—')} · {away}: {r.get('form_away', '—')}\n"
            f"➜ <b>{html.escape(pick_text(r))}</b> · مساوی {pc(r['final_draw'])} · ⚽ {goals_text}"
            "</blockquote>"
        )
    parts.append("\n⚠ = سابقهٔ کم (کمتر از ۱۵ بازی)، خروجی به سمت قیمت بازار متمایل شده")
    return "\n".join(parts)


def today_rows() -> pd.DataFrame:
    fx = PF.load_fixtures(list(LEAGUES), force=True)
    today = pd.Timestamp.now(TEHRAN).normalize().tz_localize(None)
    return market_rows(fx[fx["date"] == today])


def tomorrow_rows() -> pd.DataFrame:
    fx = PF.load_fixtures(list(LEAGUES), force=True)
    tomorrow = (pd.Timestamp.now(TEHRAN) + pd.Timedelta(days=1)).normalize().tz_localize(None)
    return market_rows(fx[fx["date"] == tomorrow])


def live_rows() -> pd.DataFrame:
    """Matches that kicked off today and should still be running.

    football-data.co.uk has no live score feed, so this is a time window, not a
    real "in play" flag: kick-off .. kick-off + 2h05.
    """
    fx = PF.load_fixtures(list(LEAGUES), force=True)
    now = pd.Timestamp.now(TEHRAN)
    today = now.normalize().tz_localize(None)
    day = fx[fx["date"] == today].copy()
    if day.empty:
        return day

    def minutes(hhmm: str) -> int | None:
        m = re.match(r"^(\d{1,2}):(\d{2})$", str(hhmm or "").strip())
        return int(m.group(1)) * 60 + int(m.group(2)) if m else None

    now_min = now.hour * 60 + now.minute
    keep = []
    for idx, r in day.iterrows():
        start = minutes(r["time"])
        if start is not None and start <= now_min <= start + 125:
            keep.append(idx)
    if not keep:
        return day.iloc[0:0]
    return market_rows(day.loc[keep].sort_values("time"))


def find_match(text: str) -> pd.DataFrame:
    fx = PF.load_fixtures(list(LEAGUES), force=True)
    if fx.empty:
        return fx

    raw = text.strip().lower()
    if not raw:
        return fx.iloc[0:0]

    parts = [p.strip() for p in re.split(r"\s+(?:vs|v|versus)\s+|[|]", raw) if p.strip()]
    if len(parts) < 2:
        parts = raw.split()

    def hits(term: str) -> pd.Series:
        return (fx["home"].str.lower().str.contains(term, regex=False, na=False)
                | fx["away"].str.lower().str.contains(term, regex=False, na=False))

    # "celtic dundee" -> home matches one, away matches the other
    if len(parts) == 2:
        a, b = parts
        one = ((fx["home"].str.lower().str.contains(a, regex=False, na=False)
                & fx["away"].str.lower().str.contains(b, regex=False, na=False))
               | (fx["home"].str.lower().str.contains(b, regex=False, na=False)
                  & fx["away"].str.lower().str.contains(a, regex=False, na=False)))
        if one.any():
            return market_rows(fx[one])

    mask = pd.Series(False, index=fx.index)
    for term in parts:
        mask |= hits(term)
    return market_rows(fx[mask])


# ----------------------------------------------------------------------- handler
HELP = (
    "<b>ربات پیش‌بینی فوتبال</b>\n\n"
    "/today — بازی‌های امروز\n"
    "/tomorrow — بازی‌های فردا\n"
    "/live — بازی‌هایی که همین حالا در جریان‌اند\n"
    "/match Celtic Dundee — جست‌وجوی یک بازی (نام انگلیسی)\n"
    "/id — شناسهٔ تلگرام تو\n\n"
    "هر بازی دو خط دارد:\n"
    "خط اول: ساعت به وقت تهران · دو تیم · ➜ شانس برد بیشتر\n"
    "خط دوم: درصد سه حالت · ضریب · تعداد گل انتظاری\n\n"
    "تیم اول میزبان است، تیم دوم مهمان.\n"
    "⚠ یعنی یکی از دو تیم سابقهٔ کافی ندارد."
)


def handle(update: dict) -> None:
    message = update.get("message") or {}
    text = (message.get("text") or "").strip()
    chat_id = (message.get("chat") or {}).get("id")
    user_id = (message.get("from") or {}).get("id")
    if not text or chat_id is None:
        return

    uid = int(user_id) if user_id else 0
    command = text.split()[0].lower().split("@")[0]

    if ALLOWED and uid not in ALLOWED:
        send(chat_id, "🔒 این ربات خصوصی است.")
        return

    if command == "/id":
        send(chat_id, f"شناسهٔ تلگرام تو:\n<code>{uid}</code>")
        return

    if command in ("/start", "/help"):
        note = "" if ALLOWED else (
            "\n\n🔒 ربات هنوز قفل نشده. این شناسه را در "
            "<code>ALLOWED_USER_IDS</code> بگذار تا فقط خودت دسترسی داشته باشی:"
            f"\n<code>{uid}</code>"
        )
        send(chat_id, HELP + note)
        return

    try:
        if command == "/today":
            send(chat_id, format_day(today_rows(), "⚽️ بازی‌های امروز"))
        elif command == "/tomorrow":
            send(chat_id, format_day(tomorrow_rows(), "⚽️ بازی‌های فردا"))
        elif command == "/live":
            rows = live_rows()
            send(chat_id, format_day(rows, "🔴 همین حالا در جریان") if not rows.empty
                 else "🔴 الان بازی‌ای در جریان نیست (لیگ‌های پوشش‌داده‌شده).")
        elif command == "/match":
            query = text[len("/match"):].strip()
            if not query:
                send(chat_id, "مثال: <code>/match Arsenal Chelsea</code>")
            else:
                send(chat_id, format_day(find_match(query), "🔎 نتیجهٔ جست‌وجو"))
        else:
            send(chat_id, HELP)
    except Exception as exc:  # noqa: BLE001
        send(chat_id, f"خطا: <code>{html.escape(str(exc)[:300])}</code>")


def load_offset() -> int | None:
    try:
        return int(json.loads(STATE.read_text(encoding="utf-8"))["offset"])
    except Exception:  # noqa: BLE001
        return None


def save_offset(value: int) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"offset": value}), encoding="utf-8")


def poll_once() -> None:
    offset = load_offset()
    payload = {"timeout": 0, "limit": 100}
    if offset is not None:
        payload["offset"] = offset
    data = api("getUpdates", **payload)
    if not data.get("ok"):
        raise SystemExit(f"telegram error: {data}")
    updates = data.get("result", [])
    for update in updates:
        handle(update)
    if updates:
        save_offset(updates[-1]["update_id"] + 1)


def poll_forever() -> None:
    listen(None)


def listen(seconds: float | None) -> None:
    """Poll Telegram until `seconds` elapse (or forever when None)."""
    me = api("getMe")
    if not me.get("ok"):
        raise SystemExit(
            f"telegram rejected the token: {me.get('description')} "
            "(run: python check_token.py)"
        )
    print(f"connected as @{me['result']['username']}", flush=True)

    offset = load_offset()
    deadline = None if seconds is None else time.time() + seconds
    print(f"listening ({'forever' if seconds is None else f'{seconds:.0f}s'})", flush=True)
    while deadline is None or time.time() < deadline:
        wait = 50 if deadline is None else max(1, min(50, int(deadline - time.time())))
        payload = {"timeout": wait, "limit": 100}
        if offset is not None:
            payload["offset"] = offset
        try:
            data = api("getUpdates", **payload)
        except requests.RequestException as exc:
            print("network:", exc, flush=True)
            time.sleep(3)
            continue
        if not data.get("ok"):
            print("telegram error:", data.get("description"), flush=True)
            time.sleep(5)
            continue
        for update in data.get("result", []):
            offset = update["update_id"] + 1
            save_offset(offset)
            try:
                handle(update)
            except Exception as exc:  # noqa: BLE001
                print("handler error:", exc, flush=True)
    print("listening finished", flush=True)


def self_test() -> None:
    print(format_day(today_rows(), "today"))
    print()
    print("------------------------------------------------------------------")
    print(format_day(tomorrow_rows(), "tomorrow"))
    print("------------------------------------------------------------------")
    print(format_day(live_rows(), "live"))


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--self-test" in argv:
        self_test()
    elif "--once" in argv:
        poll_once()
    elif "--listen" in argv:
        i = argv.index("--listen")
        value = argv[i + 1] if len(argv) > i + 1 else "240"
        listen(float(value))
    else:
        poll_forever()
