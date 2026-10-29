"""Predict upcoming fixtures with the Dixon-Coles model.

For every not-yet-played fixture in the next few days it fits the model on that
league's recent history and writes out outcome probabilities versus the
bookmaker's implied probabilities, plus the top scorelines.

Human scouting inputs (starters / substitutes / tactical notes) can be supplied
through a JSON file:

    {
      "adjustments": {
        "Liverpool": {"kind": "attack", "factor": 1.15},
        "Manchester City": {"kind": "attack", "factor": 0.9}
      }
    }

    python predict_fixtures.py --leagues E0 SP1 --days 14 --lineups data/lineups.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")

import fixtures_extra
from fetch_data import LEAGUES, download_fixtures, load_matches
from model import DixonColes

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
LINEUPS_PATH = ROOT / "data" / "lineups.json"
DEFAULT_LEAGUES = list(LEAGUES)

UK = ZoneInfo("Europe/London")
TEHRAN = ZoneInfo("Asia/Tehran")
KICKOFF_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*$")


def kickoff_tehran(day, time_text: str) -> tuple[pd.Timestamp, str]:
    """football-data.co.uk lists UK local kick-off times; show them in Tehran time."""
    day = pd.Timestamp(day)
    match = KICKOFF_RE.match(str(time_text or ""))
    if pd.isna(day) or not match:
        return day.normalize(), str(time_text or "").strip()
    local = pd.Timestamp(day.year, day.month, day.day,
                         int(match.group(1)), int(match.group(2)))
    try:
        aware = local.tz_localize(UK)
    except Exception:  # noqa: BLE001 - DST edges (01:00-02:00 on changeover days)
        aware = local.tz_localize(UK, nonexistent="shift_forward", ambiguous=True)
    in_tehran = aware.tz_convert(TEHRAN)
    return in_tehran.tz_localize(None).normalize(), in_tehran.strftime("%H:%M")


_TEAMS: dict = {"stamp": None, "by_div": {}}


def known_teams(divs: list[str]) -> dict[str, set[str]]:
    """Team names exactly as the training CSVs spell them, cached for 6 hours."""
    now = pd.Timestamp.now()
    if _TEAMS["stamp"] is None or now - _TEAMS["stamp"] > pd.Timedelta(hours=6):
        _TEAMS["by_div"] = {}
        _TEAMS["stamp"] = now
    for div in divs:
        if div in _TEAMS["by_div"]:
            continue
        try:
            data = load_matches([div])
            _TEAMS["by_div"][div] = set(data["home"]) | set(data["away"])
        except Exception:  # noqa: BLE001
            _TEAMS["by_div"][div] = set()
    return {div: _TEAMS["by_div"][div] for div in divs}


def merge_extra(out: pd.DataFrame, leagues: list[str]) -> pd.DataFrame:
    """Add upcoming matches the football-data CSV does not list yet."""
    wanted = {div: LEAGUES[div] for div in leagues if div in LEAGUES}
    if not wanted:
        return out
    try:
        more = fixtures_extra.fetch(wanted, known_teams(list(wanted)))
    except Exception:  # noqa: BLE001 - the second source is optional
        return out
    if more.empty:
        return out

    today = pd.Timestamp.now(TEHRAN).normalize().tz_localize(None)
    more = more[more["date"] >= today]
    if not out.empty:
        have = set(zip(out["home"], out["away"]))
        more = more[[(h, a) not in have for h, a in zip(more["home"], more["away"])]]
    if more.empty:
        return out
    return pd.concat([out, more], ignore_index=True)


def load_fixtures(leagues: list[str], force: bool = True, extra: bool = True) -> pd.DataFrame:
    path = download_fixtures(requests.Session(), force=force)
    fx = pd.read_csv(path, encoding="utf-8-sig")
    fx["date"] = pd.to_datetime(fx["Date"], dayfirst=True, errors="coerce")
    rows = []
    for _, r in fx[fx["Date"].notna()].iterrows():
        div = str(r.get("Div") or "")
        if div not in LEAGUES or div not in leagues:
            continue
        odds_h = float(r["B365H"]) if r.get("B365H") is not None and not pd.isna(r.get("B365H")) else np.nan
        odds_d = float(r["B365D"]) if r.get("B365D") is not None and not pd.isna(r.get("B365D")) else np.nan
        odds_a = float(r["B365A"]) if r.get("B365A") is not None and not pd.isna(r.get("B365A")) else np.nan
        day, hhmm = kickoff_tehran(r["date"], r.get("Time"))
        rows.append({
            "league": LEAGUES[div],
            "div": div,
            "date": day,
            "time": hhmm,
            "home": str(r.get("HomeTeam") or ""),
            "away": str(r.get("AwayTeam") or ""),
            "odds_h": odds_h,
            "odds_d": odds_d,
            "odds_a": odds_a,
        })
    out = pd.DataFrame(rows, columns=["league", "div", "date", "time",
                                      "home", "away", "odds_h", "odds_d", "odds_a"])
    if extra:
        out = merge_extra(out, leagues)
    return out.sort_values("date").reset_index(drop=True)


def market_probs(odds: tuple[float, float, float]) -> tuple[float, float, float]:
    vals = [o for o in odds if o is not None and not pd.isna(o) and o > 1.0]
    if len(vals) != 3:
        # No usable price: caller must fall back to the model alone. Returning a
        # made-up 50/30/20 split here would quietly poison the blend.
        return (float("nan"), float("nan"), float("nan"))
    inv = np.array([1.0 / o for o in vals])
    inv = inv / inv.sum()
    return float(inv[0]), float(inv[1]), float(inv[2])


def blend(row: dict, w: float) -> dict:
    """Model/market blend that degrades to the pure model when there are no odds."""
    if pd.isna(row.get("mkt_home")):
        row["final_home"] = row["p_home"]
        row["final_draw"] = row["p_draw"]
        row["final_away"] = row["p_away"]
        row["no_odds"] = True
    else:
        row["final_home"] = w * row["p_home"] + (1 - w) * row["mkt_home"]
        row["final_draw"] = w * row["p_draw"] + (1 - w) * row["mkt_draw"]
        row["final_away"] = w * row["p_away"] + (1 - w) * row["mkt_away"]
        row["no_odds"] = False
    return row


def build_row(r: dict, p: dict) -> dict:
    mkt_h, mkt_d, mkt_a = market_probs((r["odds_h"], r["odds_d"], r["odds_a"]))
    return {
        "league": r["league"],
        "div": r["div"],
        "date": r["date"],
        "home": r["home"],
        "away": r["away"],
        "p_home": p["p_home"],
        "p_draw": p["p_draw"],
        "p_away": p["p_away"],
        "p_over_2_5": p["p_over_2_5"],
        "exp_goals": p["exp_home_goals"] + p["exp_away_goals"],
        "exp_home_goals": p["exp_home_goals"],
        "exp_away_goals": p["exp_away_goals"],
        "likely_score": p["top_scores"][0][0] if p["top_scores"] else "",
        "mkt_home": mkt_h,
        "mkt_draw": mkt_d,
        "mkt_away": mkt_a,
        "implied_home": 1.0 / r["odds_h"] if r["odds_h"] else np.nan,
        "implied_draw": 1.0 / r["odds_d"] if r["odds_d"] else np.nan,
        "implied_away": 1.0 / r["odds_a"] if r["odds_a"] else np.nan,
        "odds_h": r["odds_h"],
        "odds_d": r["odds_d"],
        "odds_a": r["odds_a"],
        "top_scores": p["top_scores"],
    }


def run(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        d = pd.Timestamp(r["date"])
        train = load_matches([r["div"]])
        train = train[(train["date"] < d) & (train["date"] >= d - pd.Timedelta(days=900))]
        if len(train) < 30:
            continue
        model = DixonColes(half_life_days=180.0).fit(train, d)
        for team, spec in (ADJ.get("adjustments") or {}).items():
            model.set_adjustment(team, spec.get("kind", "attack"), float(spec.get("factor", 1.0)))
        p = model.predict(r["home"], r["away"])
        row = build_row(r, p)
        counts = pd.concat([train["home"], train["away"]]).value_counts()
        row["n_home"] = int(counts.get(r["home"], 0))
        row["n_away"] = int(counts.get(r["away"], 0))

        # When a team has little history in this division (promoted / relegated),
        # the model is guessing. Fall back towards the market instead of
        # pretending to know better.
        n_min = min(row["n_home"], row["n_away"])
        w = min(1.0, n_min / 30.0)
        row["model_weight"] = w
        row = blend(row, w)
        row["low_data"] = n_min < 15
        out.append(row)
    return pd.DataFrame(out).sort_values("date").reset_index(drop=True)


ADJ: dict = {"adjustments": {}}


def build_html(df: pd.DataFrame) -> str:
    body = []
    for _, r in df.iterrows():
        body.append(
            "<tr>"
            f"<td>{r['date'].date()}</td>"
            f"<td>{r['league']}</td>"
            f"<td>{r['home']} - {r['away']}</td>"
            f"<td>{r['final_home']:.0%}</td><td>{r['final_draw']:.0%}</td><td>{r['final_away']:.0%}</td>"
            f"<td>{r['mkt_home']:.0%}</td><td>{r['mkt_draw']:.0%}</td><td>{r['mkt_away']:.0%}</td>"
            f"<td>{r['odds_h']:.2f} / {r['odds_d']:.2f} / {r['odds_a']:.2f}</td>"
            f"<td>{'⚠' if r['low_data'] else ''}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="fa" dir="rtl"><head><meta charset="utf-8"><title>پیش‌بینی فلوت‌های آینده</title>
<style>
  body {{ font-family: Tahoma, sans-serif; background:#0e1117; color:#e6edf3; padding:24px; }}
  table {{ border-collapse: collapse; width:100%; background:#161b22; font-size:13px; }}
  th, td {{ border:1px solid #272e37; padding:7px 10px; text-align:right; }}
  th {{ background:#1c2430; color:#8b949e; }}
  .note {{ color:#8b949e; font-size:12px; margin-top:10px; }}
</style></head><body>
<h2>پیش‌بینی فلوت‌ها</h2>
<table><thead><tr>
<th>تاریخ</th><th>لیگ</th><th>بازی</th>
<th>مدل: میزبان</th><th>مساوی</th><th>مهمان</th>
<th>بازار: میزبان</th><th>مساوی</th><th>مهمان</th>
<th>ضرایب</th><th>⚠</th>
</tr></thead><tbody>{"".join(body)}</tbody></table>
<div class="note">⚠ یعنی یکی از دو تیم سابقهٔ کافی در این لیگ ندارد و خروجی به سمت قیمت بازار متمایل شده است.</div>
</body></html>"""



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", nargs="*", default=DEFAULT_LEAGUES)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--lineups", default=str(LINEUPS_PATH))
    args = ap.parse_args()

    lineups_path = Path(args.lineups)
    if lineups_path.exists():
        data = json.loads(lineups_path.read_text(encoding="utf-8"))
        ADJ["adjustments"] = data.get("adjustments") or {}

    fx = load_fixtures(args.leagues)
    fx = fx[fx["date"] >= pd.Timestamp.now() - pd.Timedelta(days=args.days)]
    rows = []
    for _, r in fx.iterrows():
        rows.append({"league": r["league"], "div": r["div"], "date": r["date"],
                     "home": r["home"], "away": r["away"],
                     "odds_h": r["odds_h"], "odds_d": r["odds_d"], "odds_a": r["odds_a"]})

    bt = run(rows)
    bt.to_csv(REPORTS / "fixtures.csv", index=False)
    (REPORTS / "fixtures.html").write_text(build_html(bt), encoding="utf-8")

    def pct(x: float) -> str:
        return "—" if pd.isna(x) else f"{x * 100:.0f}%"

    for _, r in bt.iterrows():
        flag = "⚠" if r["low_data"] else " "
        model_note = "" if r["model_weight"] >= 1 else f" [وزن مدل {r['model_weight'] * 100:.0f}%]"
        print(
            f"{flag} {r['date'].date()}  {r['home']} vs {r['away']}  "
            f"نهایی: {pct(r['final_home'])} / {pct(r['final_draw'])} / {pct(r['final_away'])}  "
            f"خام مدل: {pct(r['p_home'])} / {pct(r['p_draw'])} / {pct(r['p_away'])}  "
            f"بازار: {pct(r['mkt_home'])} / {pct(r['mkt_draw'])} / {pct(r['mkt_away'])}  "
            f"odds {r['odds_h']}/{r['odds_d']}/{r['odds_a']}{model_note}"
        )
    print(f"\nreport -> {REPORTS / 'fixtures.html'}")


if __name__ == "__main__":
    main()
