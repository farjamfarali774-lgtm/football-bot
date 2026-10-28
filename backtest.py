"""Walk-forward backtest: does the model predict winners better than the market?

For every week of the evaluation seasons we refit the model on everything that
happened *before* that week, then predict the week's matches. Nothing from the
future ever leaks into the training set.

Usage:
    python backtest.py                     # all leagues, recent seasons
    python backtest.py --leagues E0 SP1
    python backtest.py --seasons 2425 2526
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_data import LEAGUES, load_matches
from model import DixonColes

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

EVAL_SEASONS = ["2324", "2425", "2526", "2627"]
WINDOW_DAYS = 900
MIN_TRAIN = 200


# --------------------------------------------------------------------------- run
def walk_forward(
    df: pd.DataFrame,
    leagues: list[str],
    eval_seasons: list[str],
    half_life: float = 180.0,
    window_days: int = WINDOW_DAYS,
    verbose: bool = True,
) -> pd.DataFrame:
    rows: list[dict] = []
    for code in leagues:
        name = LEAGUES[code]
        ldf = df[df["league"] == name].sort_values("date")
        if ldf.empty:
            continue
        for season in eval_seasons:
            ev = ldf[ldf["season"] == season]
            if ev.empty:
                continue

            weeks = ev.groupby(
                [ev["date"].dt.isocalendar().year, ev["date"].dt.isocalendar().week]
            )
            for _, wdf in weeks:
                start = wdf["date"].min()
                train = ldf[
                    (ldf["date"] < start)
                    & (ldf["date"] >= start - pd.Timedelta(days=window_days))
                ]
                if len(train) < MIN_TRAIN:
                    continue
                model = DixonColes(half_life_days=half_life).fit(train, start)

                for _, m in wdf.iterrows():
                    p = model.predict(m["home"], m["away"])
                    rows.append({
                        "league": name,
                        "season": season,
                        "date": m["date"],
                        "home": m["home"],
                        "away": m["away"],
                        "hg": int(m["hg"]),
                        "ag": int(m["ag"]),
                        "ftr": m["ftr"],
                        "p_home": p["p_home"],
                        "p_draw": p["p_draw"],
                        "p_away": p["p_away"],
                        "p_over_2_5": p["p_over_2_5"],
                        "mkt_h": m["mkt_h"],
                        "mkt_d": m["mkt_d"],
                        "mkt_a": m["mkt_a"],
                    })

            if verbose:
                print(f"  {name} {season}: {len(ev)} matches", flush=True)

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("backtest produced no rows")
    return out


def walk_forward_pooled(
    df: pd.DataFrame,
    eval_seasons: list[str],
    half_life: float = 180.0,
    window_days: int = WINDOW_DAYS,
    verbose: bool = True,
) -> pd.DataFrame:
    """One model for every league, refitted weekly on all matches before that week."""
    ev = df[df["season"].isin(eval_seasons)].sort_values("date")
    rows: list[dict] = []
    weeks = ev.groupby([ev["date"].dt.isocalendar().year, ev["date"].dt.isocalendar().week])
    for _, wdf in weeks:
        start = wdf["date"].min()
        train = df[(df["date"] < start) & (df["date"] >= start - pd.Timedelta(days=window_days))]
        if len(train) < MIN_TRAIN:
            continue
        train = train.assign(group=train["league"])
        model = DixonColes(half_life_days=half_life).fit(train, start)
        for _, m in wdf.iterrows():
            p = model.predict(m["home"], m["away"], group=m["league"])
            rows.append({
                "league": m["league"], "season": m["season"], "date": m["date"],
                "home": m["home"], "away": m["away"],
                "hg": int(m["hg"]), "ag": int(m["ag"]), "ftr": m["ftr"],
                "p_home": p["p_home"], "p_draw": p["p_draw"], "p_away": p["p_away"],
                "p_over_2_5": p["p_over_2_5"],
                "mkt_h": m["mkt_h"], "mkt_d": m["mkt_d"], "mkt_a": m["mkt_a"],
            })
        if verbose:
            print(f"  week {start.date()}: {len(wdf)} matches, {len(train)} training rows",
                  flush=True)
    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("backtest produced no rows")
    return out


# ----------------------------------------------------------------------- metrics
def outcome_index(ftr: pd.Series) -> np.ndarray:
    return ftr.map({"H": 0, "D": 1, "A": 2}).to_numpy()


def market_probs(bt: pd.DataFrame) -> pd.DataFrame:
    inv = 1.0 / bt[["mkt_h", "mkt_d", "mkt_a"]].to_numpy(dtype=float)
    total = inv.sum(axis=1, keepdims=True)
    probs = inv / total
    return pd.DataFrame(probs, columns=["mkt_p_home", "mkt_p_draw", "mkt_p_away"], index=bt.index)


def score(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    y_onehot = np.zeros_like(p)
    y_onehot[np.arange(len(y)), y] = 1.0
    ll = -np.mean(np.log(p[np.arange(len(y)), y]))
    brier = float(np.mean(np.sum((p - y_onehot) ** 2, axis=1)))
    acc = float(np.mean(p.argmax(axis=1) == y))
    return {"n": len(y), "log_loss": float(ll), "brier": brier, "accuracy": acc}


def roi_table(bt: pd.DataFrame, thresholds=(0.0, 0.02, 0.05, 0.10)) -> pd.DataFrame:
    """Flat 1-unit stakes on every outcome the model thinks is underpriced."""
    odds = bt[["mkt_h", "mkt_d", "mkt_a"]].to_numpy(dtype=float)
    p = bt[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    implied = 1.0 / odds
    y = outcome_index(bt["ftr"])

    rows = []
    for thr in thresholds:
        bets = p > implied * (1.0 + thr)
        n_bets = int(bets.sum())
        if n_bets == 0:
            rows.append({"threshold": thr, "bets": 0, "staked": 0.0, "profit": 0.0, "roi": np.nan})
            continue
        # a winning 1-unit bet returns the full odds (stake included)
        won = np.zeros_like(bets, dtype=bool)
        won[np.arange(len(y)), y] = True
        returns = np.where(bets & won, odds, 0.0)
        stakes = bets.astype(float)
        profit = float(returns.sum() - stakes.sum())
        staked = float(stakes.sum())
        rows.append({
            "threshold": thr,
            "bets": n_bets,
            "staked": staked,
            "profit": profit,
            "roi": profit / staked if staked else np.nan,
        })
    return pd.DataFrame(rows)


def calibration(bt: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    p = bt[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float).ravel()
    y = outcome_index(bt["ftr"])
    hit = np.zeros((len(y), 3), dtype=bool)
    hit[np.arange(len(y)), y] = True
    hit = hit.ravel()

    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() < 20:
            continue
        rows.append({
            "bucket": f"{edges[b]:.0%}-{edges[b + 1]:.0%}",
            "n": int(m.sum()),
            "predicted": float(p[m].mean()),
            "observed": float(hit[m].mean()),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ report
def fmt_pct(x: float) -> str:
    return "—" if pd.isna(x) else f"{x * 100:.1f}%"


def build_html(bt: pd.DataFrame, model_s: dict, mkt_s: dict, home_s: dict,
               by_league: pd.DataFrame, by_season: pd.DataFrame,
               roi: pd.DataFrame, calib: pd.DataFrame) -> str:
    def table(df: pd.DataFrame, index: bool = False) -> str:
        return df.to_html(index=index, border=0, classes="tbl", escape=True,
                          float_format=lambda v: f"{v:.3f}")

    better = model_s["log_loss"] < mkt_s["log_loss"]
    if better:
        verdict = ("مدل در این بازه از قیمت بازار بهتر عمل کرده. این شرط لازم برای داشتن مزیت است، "
                   "ولی کافی نیست: باید روی داده‌ای که مدل ندیده هم تکرار شود و سود واقعی بعد از "
                   "حاشیهٔ بوکمیکر مثبت بماند.")
        verdict_cls = "good"
    else:
        verdict = ("مدل از بازار بهتر نیست. یعنی هر شرطی که بر اساس این مدل ببندی، در بلندمدت "
                   "انتظار ضرر داری — دقیقاً همان چیزی که باید قبل از خرج کردن پول می‌فهمیدیم.")
        verdict_cls = "bad"

    roi_rows = []
    for _, r in roi.iterrows():
        roi_rows.append(
            "<tr>"
            f"<td>{r['threshold'] * 100:.0f}%</td>"
            f"<td>{int(r['bets'])}</td>"
            f"<td>{r['staked']:.0f}</td>"
            f"<td>{r['profit']:+.1f}</td>"
            f"<td class='{ 'good' if (r['roi'] or 0) > 0 else 'bad'}'>{fmt_pct(r['roi'])}</td>"
            "</tr>"
        )

    calib_rows = []
    for _, r in calib.iterrows():
        gap = r["observed"] - r["predicted"]
        calib_rows.append(
            "<tr>"
            f"<td>{r['bucket']}</td><td>{int(r['n'])}</td>"
            f"<td>{fmt_pct(r['predicted'])}</td><td>{fmt_pct(r['observed'])}</td>"
            f"<td class='{ 'good' if abs(gap) < 0.03 else 'bad'}'>{gap * 100:+.1f}</td>"
            "</tr>"
        )

    by_league_html = by_league.copy()
    by_season_html = by_season.copy()
    for frame in (by_league_html, by_season_html):
        for col in ("model_acc", "market_acc", "model_ll", "market_ll"):
            frame[col] = frame[col].map(lambda v: f"{v:.3f}" if col.endswith("ll") else fmt_pct(v))
    by_league_html = by_league_html.rename(columns={
        "league": "لیگ", "n": "بازی", "model_acc": "دقت مدل",
        "market_acc": "دقت بازار", "model_ll": "LogLoss مدل", "market_ll": "LogLoss بازار",
    })
    by_season_html = by_season_html.rename(columns={
        "season": "فصل", "n": "بازی", "model_acc": "دقت مدل",
        "market_acc": "دقت بازار", "model_ll": "LogLoss مدل", "market_ll": "LogLoss بازار",
    })

    return f"""<!doctype html>
<html lang="fa" dir="rtl"><head><meta charset="utf-8">
<title>گزارش بک‌تست مدل پیش‌بینی فوتبال</title>
<style>
  :root {{ --bg:#0e1117; --card:#161b22; --line:#272e37; --fg:#e6edf3; --muted:#8b949e;
           --good:#3fb950; --bad:#f85149; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:32px 20px 80px; background:var(--bg); color:var(--fg);
          font-family:"Segoe UI",Tahoma,system-ui,sans-serif; line-height:1.7; }}
  .wrap {{ max-width:1000px; margin:0 auto; }}
  h1 {{ font-size:24px; margin:0 0 4px; }}
  h2 {{ font-size:17px; margin:36px 0 10px; padding-bottom:6px; border-bottom:1px solid var(--line); }}
  .sub {{ color:var(--muted); font-size:13px; margin-bottom:24px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
  .card .k {{ color:var(--muted); font-size:12px; }}
  .card .v {{ font-size:22px; font-weight:600; margin-top:2px; }}
  .verdict {{ margin:20px 0; padding:16px 18px; border-radius:10px; border:1px solid var(--line);
              background:var(--card); }}
  .verdict.good {{ border-color:#238636; }}
  .verdict.bad {{ border-color:#8b2c2c; }}
  table.tbl {{ width:100%; border-collapse:collapse; font-size:13px; background:var(--card);
               border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
  table.tbl th {{ background:#1c2430; color:var(--muted); font-weight:600; text-align:right;
                  padding:9px 12px; font-size:12px; }}
  table.tbl td {{ padding:8px 12px; border-top:1px solid var(--line); }}
  .good {{ color:var(--good); }} .bad {{ color:var(--bad); }}
  .note {{ color:var(--muted); font-size:12.5px; margin-top:8px; }}
  code {{ background:#1c2430; padding:1px 5px; border-radius:4px; font-size:12px; }}
</style></head><body><div class="wrap">
<h1>گزارش بک‌تست: مدل پیش‌بینی فوتبال</h1>
<div class="sub">{len(bt):,} بازی · {bt['league'].nunique()} لیگ · {', '.join(sorted(bt['season'].unique()))}</div>

<div class="cards">
  <div class="card"><div class="k">دقت مدل</div><div class="v">{fmt_pct(model_s['accuracy'])}</div></div>
  <div class="card"><div class="k">دقت بازار</div><div class="v">{fmt_pct(mkt_s['accuracy'])}</div></div>
  <div class="card"><div class="k">LogLoss مدل</div><div class="v">{model_s['log_loss']:.3f}</div></div>
  <div class="card"><div class="k">LogLoss بازار</div><div class="v">{mkt_s['log_loss']:.3f}</div></div>
  <div class="card"><div class="k">Brier مدل</div><div class="v">{model_s['brier']:.3f}</div></div>
  <div class="card"><div class="k">Brier بازار</div><div class="v">{mkt_s['brier']:.3f}</div></div>
</div>

<div class="verdict {verdict_cls}">{verdict}</div>

<h2>مدل در مقابل بازار، به تفکیک لیگ</h2>
{table(by_league_html)}

<h2>به تفکیک فصل</h2>
{table(by_season_html)}

<h2>اگر با این مدل شرط می‌بستیم</h2>
<table class="tbl"><thead><tr>
<th>آستانهٔ مزیت</th><th>تعداد شرط</th><th>مجموع مبلغ</th><th>سود/زیان</th><th>بازده</th>
</tr></thead><tbody>{''.join(roi_rows)}</tbody></table>
<div class="note">هر شرط ۱ واحد، با ضریب بسته‌شدن بازار. آستانهٔ مزیت یعنی مدل چقدر باید از قیمت بازار خوش‌بین‌تر باشد تا وارد شویم.</div>

<h2>کالیبراسیون: وقتی مدل ۴۰٪ می‌گوید، واقعاً چند درصد اتفاق افتاده؟</h2>
<table class="tbl"><thead><tr>
<th>بازهٔ احتمال</th><th>تعداد</th><th>میانگین پیش‌بینی</th><th>واقعیت</th><th>خطا</th>
</tr></thead><tbody>{''.join(calib_rows)}</tbody></table>
<div class="note">خطای کمتر از ۳ درصد یعنی مدل کم‌وبیش صادق است.</div>

<h2>این اعداد چه معنایی دارند</h2>
<div class="note">
<b>دقت</b> فقط می‌گوید چند بار تیم برنده را درست حدس زده؛ برای پول بی‌معنی است، چون همه می‌توانند همیشه روی میزبان شرط ببندند و ۴۵٪ دقت بگیرند.<br>
<b>LogLoss</b> معیار اصلی است: کیفیت خودِ احتمال‌ها، نه فقط حدس نهایی. هرچه کمتر بهتر. پایین‌تر بودن LogLoss بازار از مدل یعنی بازار بهتر می‌داند.<br>
<b>بازده</b> تنها چیزی است که به جیب مربوط است، و برای معنادار شدن به هزاران شرط نیاز دارد.
</div>
</div></body></html>"""


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", nargs="*", default=list(LEAGUES))
    ap.add_argument("--seasons", nargs="*", default=EVAL_SEASONS)
    ap.add_argument("--half-life", type=float, default=180.0)
    ap.add_argument("--mode", choices=["pooled", "league"], default="league",
                    help="pooled: one model for all leagues; league: one model per league")
    args = ap.parse_args()

    print("loading data ...", flush=True)
    df = load_matches(args.leagues)
    print(f"{len(df):,} matches available", flush=True)

    print(f"walk-forward backtest ({args.mode}) ...", flush=True)
    if args.mode == "pooled":
        bt = walk_forward_pooled(df, args.seasons, half_life=args.half_life)
    else:
        bt = walk_forward(df, args.leagues, args.seasons, half_life=args.half_life)

    bt = bt.dropna(subset=["mkt_h", "mkt_d", "mkt_a"]).reset_index(drop=True)
    mp = market_probs(bt)
    bt = pd.concat([bt, mp], axis=1)

    y = outcome_index(bt["ftr"])
    model_p = bt[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    mkt_p = bt[["mkt_p_home", "mkt_p_draw", "mkt_p_away"]].to_numpy(dtype=float)

    model_s = score(model_p, y)
    mkt_s = score(mkt_p, y)

    # naive baseline: always back the home side
    home_p = np.tile(np.array([[1.0, 0.0, 0.0]]), (len(y), 1))
    home_s = score(home_p, y)

    def group_scores(by: str) -> pd.DataFrame:
        rows = []
        for key, g in bt.groupby(by):
            yy = outcome_index(g["ftr"])
            ms = score(g[["p_home", "p_draw", "p_away"]].to_numpy(), yy)
            ks = score(g[["mkt_p_home", "mkt_p_draw", "mkt_p_away"]].to_numpy(), yy)
            rows.append({by: key, "n": len(g), "model_acc": ms["accuracy"],
                         "market_acc": ks["accuracy"], "model_ll": ms["log_loss"],
                         "market_ll": ks["log_loss"]})
        return pd.DataFrame(rows)

    by_league = group_scores("league")
    by_season = group_scores("season")
    roi = roi_table(bt)
    calib = calibration(bt)

    print("\n=== overall ===")
    print(f"matches          {model_s['n']:,}")
    print(f"model accuracy   {model_s['accuracy']:.3f}   logloss {model_s['log_loss']:.4f}")
    print(f"market accuracy  {mkt_s['accuracy']:.3f}   logloss {mkt_s['log_loss']:.4f}")
    print(f"home-always acc  {home_s['accuracy']:.3f}   logloss {home_s['log_loss']:.4f}")
    print("\n=== roi (flat stakes on model value picks) ===")
    print(roi.to_string(index=False))

    bt.to_csv(REPORTS / "predictions.csv", index=False)
    html_out = build_html(bt, model_s, mkt_s, home_s, by_league, by_season, roi, calib)
    (REPORTS / "backtest.html").write_text(html_out, encoding="utf-8")
    print(f"\nreport -> {REPORTS / 'backtest.html'}")


if __name__ == "__main__":
    main()
