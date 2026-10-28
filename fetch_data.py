"""Download match results and closing odds from football-data.co.uk.

Files are cached in data/raw/ so re-running is cheap.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# Div codes used by football-data.co.uk
LEAGUES = {
    # England
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
    "EC": "National League",
    # Scotland
    "SC0": "Scottish Premiership",
    "SC1": "Scottish Championship",
    "SC2": "Scottish League One",
    "SC3": "Scottish League Two",
    # Germany / Italy / Spain / France
    "D1": "Bundesliga",
    "D2": "2. Bundesliga",
    "I1": "Serie A",
    "I2": "Serie B",
    "SP1": "La Liga",
    "SP2": "Segunda Division",
    "F1": "Ligue 1",
    "F2": "Ligue 2",
    # Rest of Europe
    "N1": "Eredivisie",
    "B1": "Belgian Pro League",
    "P1": "Primeira Liga",
    "T1": "Super Lig",
    "G1": "Greek Super League",
}

# The five biggest leagues, used as the default when speed matters.
TOP_LEAGUES = ["E0", "SP1", "D1", "I1", "F1"]

SEASONS = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2015, 2027)]

URL = "https://football-data.co.uk/mmz4281/{season}/{league}.csv"

# Columns we care about.
COLUMNS = [
    "Div", "Date", "Time", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HS", "AS", "HST", "AST",
    # average closing odds (all bookmakers) - the realistic price
    "AvgCH", "AvgCD", "AvgCA",
    # Pinnacle closing - the sharpest reference, when available
    "PSCH", "PSCD", "PSCA",
    "B365CH", "B365CD", "B365CA",
    "MaxCH", "MaxCD", "MaxCA",
]


def download(league: str, season: str, session: requests.Session, force: bool = False) -> Path:
    path = RAW / f"{league}_{season}.csv"
    if path.exists() and path.stat().st_size > 1000 and not force:
        return path

    url = URL.format(season=season, league=league)
    for attempt in range(3):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 404:
                path.write_bytes(b"")
                return path
            r.raise_for_status()
            path.write_bytes(r.content)
            return path
        except requests.RequestException as exc:
            if attempt == 2:
                print(f"  failed {league} {season}: {exc}", file=sys.stderr)
                path.write_bytes(b"")
                return path
            time.sleep(2)
    return path


def fetch_all(force: bool = False, seasons: list[str] | None = None) -> None:
    seasons = seasons or SEASONS
    session = requests.Session()
    session.headers.update({"User-Agent": "football-bot/0.1 (local research)"})
    total = len(LEAGUES) * len(seasons)
    done = 0
    for league in LEAGUES:
        for season in seasons:
            download(league, season, session, force=force)
            done += 1
            print(f"\r{league}  {done}/{total}", end="", flush=True)
    print()


def _parse_dates(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, dayfirst=True, format="%d/%m/%Y", errors="coerce")
    except ValueError:
        return pd.to_datetime(series, dayfirst=True, errors="coerce", format="mixed")


def load_matches(leagues: list[str] | None = None) -> pd.DataFrame:
    """Return one tidy dataframe with every match we have data for."""
    leagues = leagues or list(LEAGUES)
    frames = []
    for league in leagues:
        for season in SEASONS:
            path = RAW / f"{league}_{season}.csv"
            if not path.exists() or path.stat().st_size < 1000:
                continue
            for enc in ("utf-8-sig", "latin-1"):
                try:
                    df = pd.read_csv(path, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                continue

            df = df.reindex(columns=COLUMNS)
            df["season"] = season
            df["league"] = LEAGUES.get(league, league)
            frames.append(df)

    if not frames:
        raise SystemExit("No data found. Run: python fetch_data.py")

    df = pd.concat(frames, ignore_index=True)
    df["date"] = _parse_dates(df["Date"])
    df = df.rename(columns={
        "HomeTeam": "home", "AwayTeam": "away",
        "FTHG": "hg", "FTAG": "ag", "FTR": "ftr",
        "HS": "hs", "AS": "as_", "HST": "hst", "AST": "ast",
        "AvgCH": "odd_h", "AvgCD": "odd_d", "AvgCA": "odd_a",
        "PSCH": "ps_h", "PSCD": "ps_d", "PSCA": "ps_a",
        "B365CH": "b365_h", "B365CD": "b365_d", "B365CA": "b365_a",
        "MaxCH": "max_h", "MaxCD": "max_d", "MaxCA": "max_a",
    })

    keep = [
        "league", "season", "date", "home", "away", "hg", "ag", "ftr",
        "hs", "as_", "hst", "ast",
        "odd_h", "odd_d", "odd_a", "ps_h", "ps_d", "ps_a",
        "b365_h", "b365_d", "b365_a", "max_h", "max_d", "max_a",
    ]
    df = df[keep].dropna(subset=["date", "home", "away", "hg", "ag"])
    df["hg"] = df["hg"].astype(int)
    df["ag"] = df["ag"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)

    # Prefer Pinnacle closing, fall back to the average across bookmakers.
    for side in ("h", "d", "a"):
        df[f"mkt_{side}"] = df[f"ps_{side}"].fillna(df[f"odd_{side}"])
    return df


def download_fixtures(session: requests.Session, force: bool = False) -> Path:
    """Pre-match fixtures + odds from football-data.co.uk."""
    path = RAW / "fixtures.csv"
    if force or not path.exists() or path.stat().st_size < 1000:
        r = session.get("https://football-data.co.uk/fixtures.csv", timeout=30)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download every season")
    ap.add_argument("--current", action="store_true",
                    help="re-download only the two most recent seasons (daily update)")
    args = ap.parse_args()

    if args.current:
        chosen = SEASONS[-2:]
        fetch_all(force=True, seasons=chosen)
        download_fixtures(requests.Session(), force=True)
    else:
        fetch_all(force=args.force)
