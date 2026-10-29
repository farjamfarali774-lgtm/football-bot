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
import math
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
VALUE_STATS = ROOT / ".state" / "value_stats.json"
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
            row = PF.blend(row, w)
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


def odd(r, key: str) -> str:
    value = r.get(key)
    return "—" if value is None or pd.isna(value) else f"{value:.2f}"


def odds_text(r) -> str:
    values = [r.get(k) for k in ("odds_h", "odds_d", "odds_a")]
    if all(v is None or pd.isna(v) for v in values):
        return "بدون ضریب (فقط مدل)"
    return "/".join("—" if v is None or pd.isna(v) else f"{v:.2f}" for v in values)


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
        no_odds = bool(r.get("no_odds"))
        home_line = f"🏠 <b>{home}</b> — برد {pc(r['final_home'])}"
        away_line = f"✈️ <b>{away}</b> — برد {pc(r['final_away'])}"
        if not no_odds:
            home_line += f" · ضریب {odd(r, 'odds_h')}"
            away_line += f" · ضریب {odd(r, 'odds_a')}"
        parts.append(
            "<blockquote>"
            f"🕒 <b>{r['time'] or '--:--'}</b> · {flag} {league}{warn}\n"
            f"{home_line}\n"
            f"{away_line}\n"
            f"📈 فرم {home}: {r.get('form_home', '—')} · {away}: {r.get('form_away', '—')}\n"
            f"➜ <b>{html.escape(pick_text(r))}</b> · مساوی {pc(r['final_draw'])} · ⚽ {goals_text}"
            f"{'' if not no_odds else chr(10) + '🧮 ضریب این بازی نیست، پیش‌بینی فقط از مدل است'}"
            "</blockquote>"
        )
    parts.append("\n⚠ = سابقهٔ کم (کمتر از ۱۵ بازی) · 🧮 = بدون ضریب بازار، فقط مدل")
    return "\n".join(parts)


def upcoming_rows(days: int = 2) -> pd.DataFrame:
    """Every covered fixture from now to `days` from today (Tehran)."""
    fx = PF.load_fixtures(list(LEAGUES), force=True)
    today = pd.Timestamp.now(TEHRAN).normalize().tz_localize(None)
    return market_rows(fx[(fx["date"] >= today) & (fx["date"] < today + pd.Timedelta(days=days))])


def value_picks(rows: pd.DataFrame, min_edge: float = 0.05) -> pd.DataFrame:
    """Fixtures where our blended probability beats the market's by `min_edge`.

    The market probability is the normalised closing price, so the margin is
    removed and the two numbers are comparable.
    """
    if rows.empty:
        return pd.DataFrame()
    picks = []
    for _, r in rows.iterrows():
        best = None
        for key, prob, mkt, odd, name in (
            ("home", r["final_home"], r["mkt_home"], r["odds_h"], r["home"]),
            ("draw", r["final_draw"], r["mkt_draw"], r["odds_d"], "مساوی"),
            ("away", r["final_away"], r["mkt_away"], r["odds_a"], r["away"]),
        ):
            if pd.isna(prob) or pd.isna(mkt) or pd.isna(odd) or odd <= 1.0:
                continue
            edge = float(prob) - float(mkt)
            if edge < min_edge or prob < 0.25:
                continue
            if best is None or edge > best[1]:
                best = (key, edge, float(prob), float(mkt), float(odd), name)
        if best is not None:
            row = r.to_dict()
            (row["v_side"], row["v_edge"], row["v_prob"],
             row["v_mkt"], row["v_odd"], row["v_name"]) = best
            picks.append(row)
    if not picks:
        return pd.DataFrame()
    return pd.DataFrame(picks).sort_values("v_edge", ascending=False).reset_index(drop=True)


def format_value(rows: pd.DataFrame, total: int, min_edge: float, priced: int) -> str:
    head = f"💎 <b>ارزش‌دارها</b> — اختلاف مدل با بازار ≥ {min_edge * 100:.0f}٪"
    if rows.empty:
        return (f"{head}\n\nدر {priced} بازی ضریب‌دار (از {total} بازی ۳ روز آینده) "
                "جایی پیدا نشد که مدل با بازار این‌قدر اختلاف داشته باشد.")
    parts = [
        head,
        f"{len(rows)} مورد از {priced} بازی ضریب‌دار",
        f"{total} بازی در ۳ روز آینده بررسی شد؛ بقیه هنوز ضریب ندارند.",
        "یعنی مدل می‌گوید بازار این نتیجه را کم‌قیمت گرفته.",
        "",
    ]
    for _, r in rows.iterrows():
        flag = LEAGUE_FLAG.get(str(r["league"]), "⚽️")
        warn = " ⚠" if r.get("low_data") else ""
        parts.append(
            "<blockquote>"
            f"🕒 <b>{r['time'] or '--:--'}</b> · {flag} {html.escape(str(r['league']))}{warn}\n"
            f"🏠 {html.escape(str(r['home']))} – ✈️ {html.escape(str(r['away']))}\n"
            f"💎 ارزش روی <b>{html.escape(str(r['v_name']))}</b>\n"
            f"مدل {pc(r['v_prob'])} · بازار {pc(r['v_mkt'])} · "
            f"اختلاف +{r['v_edge'] * 100:.0f}٪ · ضریب {r['v_odd']:.2f}"
            "</blockquote>"
        )
    parts.append("\n⚠ = سابقهٔ کم · این فقط اختلاف با بازار است، نه تضمین سود.")
    record = value_track_record(min_edge)
    if record:
        parts.append(record)
    return "\n".join(parts)


def value_stats(days: int = 90, max_age_hours: int = 12) -> dict:
    """Measured record of the /value filter, so the command shows its own result.

    Betting every disagreement with the market is the intuitive idea and the
    wrong one: in 90 days of walk-forward data it returned about -13%, i.e. the
    model's "edges" are its mistakes rather than the market's.
    """
    try:
        cached = json.loads(VALUE_STATS.read_text(encoding="utf-8"))
        age = (pd.Timestamp.now() - pd.Timestamp(cached["at"])).total_seconds() / 3600.0
        if cached.get("days") == days and age < max_age_hours:
            return cached
    except Exception:  # noqa: BLE001 - a stale cache is not an error
        pass

    pred = recent_results(days)
    truth = {"H": "home", "D": "draw", "A": "away"}
    sides = []
    for _, r in pred.iterrows():
        actual = truth[r["ftr"]]
        for side, prob, mkt, odd in (
            ("home", r["f_home"], r["mkt_h"], r["odd_h"]),
            ("draw", r["f_draw"], r["mkt_d"], r["odd_d"]),
            ("away", r["f_away"], r["mkt_a"], r["odd_a"]),
        ):
            if pd.isna(prob) or pd.isna(mkt) or pd.isna(odd):
                continue
            sides.append((float(prob), float(mkt), float(odd), int(side == actual)))

    stats: dict = {"at": pd.Timestamp.now().isoformat(), "days": days,
                   "matches": int(len(pred)), "edges": {}}
    for edge in (0.03, 0.05, 0.08, 0.10):
        bets = [s for s in sides if s[0] - s[1] >= edge and s[0] >= 0.25]
        if not bets:
            stats["edges"][f"{edge:.2f}"] = {"bets": 0, "hit": None, "roi": None}
            continue
        stats["edges"][f"{edge:.2f}"] = {
            "bets": len(bets),
            "hit": sum(s[3] for s in bets) / len(bets),
            "roi": sum(s[3] * s[2] - 1.0 for s in bets) / len(bets),
        }
    try:
        VALUE_STATS.parent.mkdir(parents=True, exist_ok=True)
        VALUE_STATS.write_text(json.dumps(stats), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return stats


def value_track_record(min_edge: float, days: int = 90) -> str:
    try:
        stats = value_stats(days)
    except Exception:  # noqa: BLE001
        return ""
    edges = stats.get("edges") or {}
    entry = edges.get(f"{min_edge:.2f}") or edges.get("0.05")
    if not entry or not entry.get("bets"):
        return ""
    verdict = "ضرر می‌داد" if entry["roi"] < 0 else "سود می‌داد"
    return (f"\n📉 سابقهٔ همین فیلتر در {days} روز اخیر: {entry['bets']} سیگنال · "
            f"برد {entry['hit'] * 100:.0f}٪ · بازده {entry['roi'] * 100:+.0f}٪ ({verdict})")


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


def recent_results(days: int = 7) -> pd.DataFrame:
    """Walk-forward predictions for finished matches of the last `days` days.

    The model is refitted per league per week on data that existed before that
    week, so this is an honest out-of-sample scoreboard, not a replay.
    """
    df = load_matches(list(LEAGUES))
    ref = pd.Timestamp.now(TEHRAN).normalize().tz_localize(None)
    start = ref - pd.Timedelta(days=days)
    ev = df[(df["date"] >= start) & (df["date"] < ref)].copy()
    if ev.empty:
        return pd.DataFrame()

    ev["week"] = ev["date"].dt.to_period("W")
    rows = []
    for (league, _week), g in ev.groupby(["league", "week"]):
        first = g["date"].min()
        train = df[(df["league"] == league) & (df["date"] < first)
                   & (df["date"] >= first - pd.Timedelta(days=900))]
        if len(train) < 30:
            continue
        model = DixonColes().fit(train, first)
        counts = pd.concat([train["home"], train["away"]]).value_counts()
        for _, m in g.iterrows():
            p = model.predict(m["home"], m["away"])
            oh, od, oa = m["mkt_h"], m["mkt_d"], m["mkt_a"]
            if pd.isna(oh) or pd.isna(od) or pd.isna(oa) or min(oh, od, oa) <= 1.0:
                mh = md = ma = float("nan")
                fh, fd, fa = p["p_home"], p["p_draw"], p["p_away"]
            else:
                inv = 1.0 / oh + 1.0 / od + 1.0 / oa
                mh, md, ma = (1.0 / oh) / inv, (1.0 / od) / inv, (1.0 / oa) / inv
                n_min = min(int(counts.get(m["home"], 0)), int(counts.get(m["away"], 0)))
                w = min(1.0, n_min / 30.0)
                fh = w * p["p_home"] + (1 - w) * mh
                fd = w * p["p_draw"] + (1 - w) * md
                fa = w * p["p_away"] + (1 - w) * ma
            rows.append({
                "date": m["date"], "league": league,
                "home": m["home"], "away": m["away"], "ftr": m["ftr"],
                "hg": int(m["hg"]), "ag": int(m["ag"]),
                "p_home": p["p_home"], "p_draw": p["p_draw"], "p_away": p["p_away"],
                "mkt_h": mh, "mkt_d": md, "mkt_a": ma,
                "odd_h": oh, "odd_d": od, "odd_a": oa,
                "f_home": fh, "f_draw": fd, "f_away": fa,
            })
    return pd.DataFrame(rows)


LABEL = {"home": "برد میزبان", "draw": "مساوی", "away": "برد مهمان"}
TRUTH = {"H": "home", "D": "draw", "A": "away"}
ORDER = ("home", "draw", "away")


def _pick(probs: tuple[float, float, float]) -> str:
    return ORDER[max(range(3), key=lambda i: probs[i])]


def format_results(days: int = 7) -> str:
    pred = recent_results(days)
    if pred.empty:
        return f"📊 در {days} روز اخیر بازی تمام‌شده‌ای در لیگ‌های پوشش‌داده‌شده پیدا نشد."

    hits = 0
    ll_model = 0.0
    ll_market = 0.0
    market_n = 0
    for _, r in pred.iterrows():
        probs = (r["p_home"], r["p_draw"], r["p_away"])
        truth = TRUTH[r["ftr"]]
        picked = _pick(probs)
        hits += int(picked == truth)
        ll_model -= math.log(max(probs[ORDER.index(truth)], 1e-12))
        mkt = (r["mkt_h"], r["mkt_d"], r["mkt_a"])
        if not any(pd.isna(v) for v in mkt):
            ll_market -= math.log(max(mkt[ORDER.index(truth)], 1e-12))
            market_n += 1

    n = len(pred)
    acc = hits / n
    ll_model /= n
    ll_market = ll_market / market_n if market_n else float("nan")
    market_text = "—" if pd.isna(ll_market) else f"{ll_market:.3f}"

    parts = [
        f"📊 <b>کارنامهٔ {days} روز اخیر</b>",
        f"{n} بازی تمام‌شده بررسی شد",
        f"✅ درست <b>{hits}</b> ({acc * 100:.0f}%) · ❌ غلط {n - hits}",
        "🎲 شانس تصادفی بین سه حالت: ۳۳٪",
        "",
        "LogLoss (کمتر بهتر است):",
        f"مدل <b>{ll_model:.3f}</b> · بازار {market_text}",
    ]

    last_day = pred["date"].max()
    recent = pred[pred["date"] == last_day].tail(6)
    if len(recent):
        parts.append(f"\n<b>آخرین روز — {last_day.date()}</b>")
        for _, r in recent.iterrows():
            probs = (r["p_home"], r["p_draw"], r["p_away"])
            truth = TRUTH[r["ftr"]]
            picked = _pick(probs)
            mark = "✅" if picked == truth else "❌"
            home = html.escape(str(r["home"]))
            away = html.escape(str(r["away"]))
            parts.append(
                "<blockquote>"
                f"{mark} <b>{home}</b> {r['hg']}–{r['ag']} <b>{away}</b>\n"
                f"مدل گفت: {LABEL[picked]} {pc(probs[ORDER.index(picked)])}\n"
                f"واقعیت: {LABEL[truth]}"
                "</blockquote>"
            )
    return "\n".join(parts)
    return "\n".join(parts)


HELP = (
    "<b>ربات پیش‌بینی فوتبال</b>\n\n"
    "/today — بازی‌های امروز\n"
    "/tomorrow — بازی‌های فردا\n"
    "/live — بازی‌هایی که همین حالا در جریان‌اند\n"
    "/results — کارنامه: پیش‌بینی‌های اخیر درست بود یا نه\n"
    "/results 30 — کارنامهٔ ۳۰ روز اخیر\n"
    "/value — بازی‌هایی که مدل با بازار اختلاف دارد (ارزش‌دار)\n"
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
        if command == "/start":
            try:
                send(chat_id, format_day(today_rows(), "⚽️ بازی‌های امروز"))
            except Exception as exc:  # noqa: BLE001
                send(chat_id, f"خطا در گرفتن بازی‌های امروز: <code>{html.escape(str(exc)[:200])}</code>")
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
        elif command in ("/results", "/stats"):
            bits = text.split()
            days = int(bits[1]) if len(bits) > 1 and bits[1].isdigit() else 7
            send(chat_id, format_results(days))
        elif command == "/value":
            bits = text.split()
            edge = 0.05
            if len(bits) > 1:
                try:
                    edge = max(0.02, min(0.30, float(bits[1].replace("%", "")) / 100.0))
                except ValueError:
                    pass
            rows = upcoming_rows(3)
            priced = int(rows["mkt_home"].notna().sum()) if not rows.empty else 0
            send(chat_id, format_value(value_picks(rows, edge), len(rows), edge, priced))
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
