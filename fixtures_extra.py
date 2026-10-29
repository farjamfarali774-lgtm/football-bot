"""Second fixture source: fixturedownload.com.

football-data.co.uk refreshes `fixtures.csv` only a few times a week - on
2026-10-29 the file was still stamped 2026-10-27 and stopped at that same day,
so /today could be nearly empty and /tomorrow had nothing at all.
fixturedownload.com publishes a whole season as JSON for the big competitions,
so the still-upcoming matches from there are merged in.

Two rules keep the numbers honest:

* The feed has no odds, so those matches are predicted by the model alone; the
  market blend is skipped instead of inventing a fake market.
* Team names differ between sources ("AC Milan" vs "Milan"), so names are mapped
  onto the football-data.co.uk spelling and a fixture is dropped when either
  side cannot be mapped. The model treats an unknown club as an average team,
  which would look like a real prediction while being meaningless.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests

TEHRAN = ZoneInfo("Asia/Tehran")

# competition code (as used by football-data.co.uk) -> fixturedownload.com slug
SLUGS = {
    "E0": "epl",
    "E1": "championship",
    "SC0": "scottish-premiership",
    "D1": "bundesliga",
    "I1": "serie-a",
    "SP1": "la-liga",
    "F1": "ligue-1",
    "F2": "ligue-2",
    "N1": "eredivisie",
    "P1": "primeira-liga",
    "T1": "super-lig",
}

COLS = ["league", "div", "date", "time", "home", "away", "odds_h", "odds_d", "odds_a"]

NOISE = {
    "fc", "afc", "cf", "sc", "ac", "as", "ss", "ssc", "ud", "cd", "rc", "cfc",
    "club", "de", "calcio", "sad", "sv", "vfl", "vfb", "tsg", "fsv", "spvgg",
}


def _norm(name: str) -> str:
    text = re.sub(r"[^a-z0-9 ]+", " ", str(name).lower())
    return " ".join(w for w in text.split() if w and w not in NOISE)


def _season_year() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1


def _resolve(name: str, known: set[str], index: dict[str, str]) -> str | None:
    """Map a feed team name onto the training-data name, or None if unsure."""
    if name in known:
        return name
    key = _norm(name)
    if key in index:
        return index[key]
    if not key:
        return None
    hits = [k for k in index if key in k or k in key]
    if len(hits) == 1:
        return index[hits[0]]
    return None


def fetch(leagues: dict[str, str], known: dict[str, set[str]], days: int = 10) -> pd.DataFrame:
    """Upcoming fixtures (Tehran local, no odds) for the leagues we can name-map."""
    season = _season_year()
    now = pd.Timestamp.now(tz="UTC")
    horizon = now + pd.Timedelta(days=days)
    session = requests.Session()
    rows: list[dict] = []

    for div, league in leagues.items():
        slug = SLUGS.get(div)
        teams = known.get(div)
        if not slug or not teams:
            continue
        index = {_norm(t): t for t in teams}

        url = f"https://fixturedownload.com/feed/json/{slug}-{season}"
        try:
            response = session.get(url, timeout=25)
            if response.status_code != 200:
                continue
            matches = response.json()
        except Exception:  # noqa: BLE001 - a dead source must not break the bot
            continue

        for match in matches:
            if match.get("HomeTeamScore") is not None:
                continue
            stamp = str(match.get("DateUtc") or "")
            if not stamp:
                continue
            try:
                kickoff = pd.Timestamp(stamp.replace("Z", "").strip()).tz_localize("UTC")
            except Exception:  # noqa: BLE001
                continue
            if kickoff < now - pd.Timedelta(hours=3) or kickoff > horizon:
                continue

            home = _resolve(str(match.get("HomeTeam") or ""), teams, index)
            away = _resolve(str(match.get("AwayTeam") or ""), teams, index)
            if not home or not away or home == away:
                continue

            local = kickoff.tz_convert(TEHRAN)
            rows.append({
                "league": league,
                "div": div,
                "date": local.tz_localize(None).normalize(),
                "time": local.strftime("%H:%M"),
                "home": home,
                "away": away,
                "odds_h": float("nan"),
                "odds_d": float("nan"),
                "odds_a": float("nan"),
            })

    return pd.DataFrame(rows, columns=COLS)
