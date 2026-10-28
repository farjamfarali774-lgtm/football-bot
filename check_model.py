"""Sanity checks that must pass before any backtest number can be trusted.

1. The analytic gradient must match a finite-difference estimate -- for the
   single-league fit and for the pooled (multi-league) fit. A wrong gradient is
   silent: the model still "fits", it just fits badly.
2. When the model is fitted and then asked about the very same matches, its
   average probabilities must match what actually happened.

    python check_model.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import check_grad

import model as M
from fetch_data import load_matches


def capture_objective(df: pd.DataFrame, ref: pd.Timestamp) -> dict:
    captured: dict = {}
    original = M.minimize

    def spy(fun, x0, **kwargs):
        captured["fun"] = fun
        captured["x0"] = x0.copy()
        return original(fun, x0, **kwargs)

    M.minimize = spy  # type: ignore[assignment]
    try:
        M.DixonColes().fit(df, ref)
    finally:
        M.minimize = original  # type: ignore[assignment]
    return captured


def gradient_check(name: str, df: pd.DataFrame, ref: pd.Timestamp) -> None:
    cap = capture_objective(df, ref)
    fun = cap["fun"]
    x = cap["x0"] + 0.05 * np.random.default_rng(0).standard_normal(cap["x0"].shape)
    err = check_grad(lambda t: float(fun(t)[0]), lambda t: fun(t)[1], x)
    rel = err / max(np.linalg.norm(fun(x)[1]), 1e-12)
    print(f"gradient ({name}): relative error {rel:.2e}")
    if rel > 1e-5:
        raise SystemExit(f"FAIL: gradient mismatch ({name})")


def calibration_check() -> None:
    df = load_matches(["E0", "SP1", "D1", "I1", "F1"])
    ev = df[df["season"].isin(["2324", "2425", "2526"])]
    m = M.DixonColes().fit(ev, ev["date"].max())

    probs = np.array([[m.predict(h, a)[k] for k in ("p_home", "p_draw", "p_away")]
                      for h, a in zip(ev["home"], ev["away"])])
    actual = np.array([(ev["ftr"] == "H").mean(), (ev["ftr"] == "D").mean(),
                       (ev["ftr"] == "A").mean()])
    print(f"probabilities   {np.round(probs.mean(0), 3)}  vs actual {np.round(actual, 3)}")

    if np.abs(probs.mean(0) - actual).max() > 0.02:
        raise SystemExit("FAIL: probabilities are biased in-sample")
    print("calibration OK")


def main() -> None:
    df = load_matches(["E0"])
    ref = df["date"].max()
    single = df[df["date"] >= ref - pd.Timedelta(days=900)]
    gradient_check("single league", single, ref)

    pooled_df = load_matches(["E0", "E1", "SP1"])
    ref2 = pooled_df["date"].max()
    pooled = pooled_df[pooled_df["date"] >= ref2 - pd.Timedelta(days=900)]
    pooled = pooled.assign(group=pooled["league"])
    gradient_check("pooled leagues", pooled, ref2)

    calibration_check()


if __name__ == "__main__":
    main()
