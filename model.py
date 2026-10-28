"""Dixon-Coles bivariate Poisson model, optionally pooled across leagues.

Each team gets an attack and a defence rating. Goals for the home side are
Poisson(exp(att_home + def_away + home_adv + league_level)), goals for the away
side are Poisson(exp(att_away + def_home + league_level)).

Why the `group` column: if we fit every division separately, a team that gets
promoted or relegated has almost no history in its new division, so the model
treats it as an average team. Fitting all leagues together -- with one extra
"scoring level" parameter per league -- lets a team carry its rating with it
when it changes division.

Recent matches count more (exponential time decay), and a small correction
(rho) fixes the known bias on 0-0, 1-0, 0-1 and 1-1 scorelines.

Fitted by maximum likelihood with an analytic gradient.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

MAX_GOALS = 12


@dataclass
class DixonColes:
    half_life_days: float = 180.0
    ridge: float = 1.5
    ridge_league: float = 20.0
    teams: list[str] = field(default_factory=list)
    attack: np.ndarray | None = None
    defence: np.ndarray | None = None
    home_adv: float = 0.0
    rho: float = 0.0
    league_offset: dict[str, float] = field(default_factory=dict)
    _index: dict[str, int] = field(default_factory=dict)
    adjustments: dict[str, tuple[str, float]] = field(default_factory=dict)

    # ---------------------------------------------------------------- fitting
    def fit(self, df: pd.DataFrame, ref_date: pd.Timestamp) -> "DixonColes":
        df = df.dropna(subset=["home", "away", "hg", "ag"]).reset_index(drop=True)
        if len(df) < 30:
            raise ValueError("not enough matches to fit")

        self.teams = sorted(set(df["home"]) | set(df["away"]))
        self._index = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        hi = df["home"].map(self._index).to_numpy()
        ai = df["away"].map(self._index).to_numpy()
        hg = df["hg"].to_numpy(dtype=float)
        ag = df["ag"].to_numpy(dtype=float)

        # optional league grouping
        if "group" in df.columns and df["group"].nunique() > 1:
            groups = sorted(df["group"].dropna().unique())
            gindex = {g: i for i, g in enumerate(groups)}
            gi = df["group"].map(gindex).to_numpy()
            G = len(groups)
        else:
            groups, gindex, G = [], {}, 0
            gi = np.zeros(len(df), dtype=int)

        age = (ref_date - df["date"]).dt.days.to_numpy(dtype=float)
        w = 0.5 ** (np.clip(age, 0, None) / self.half_life_days)

        lam_const = gammaln(hg + 1.0)
        mu_const = gammaln(ag + 1.0)

        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)

        ridge = self.ridge
        ridge_league = self.ridge_league

        def unpack(theta):
            att = theta[:n]
            dfn = theta[n:2 * n]
            home = theta[2 * n]
            rho = theta[2 * n + 1]
            lg = theta[2 * n + 2:2 * n + 2 + G]
            return att, dfn, home, rho, lg

        def objective(theta):
            att, dfn, home, rho, lg = unpack(theta)
            offset = lg[gi] if G else 0.0
            lam = np.maximum(np.exp(att[hi] + dfn[ai] + home + offset), 1e-9)
            mu = np.maximum(np.exp(att[ai] + dfn[hi] + offset), 1e-9)

            tau = np.ones_like(lam)
            tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
            tau[m01] = 1.0 + lam[m01] * rho
            tau[m10] = 1.0 + mu[m10] * rho
            tau[m11] = 1.0 - rho
            tau = np.clip(tau, 1e-10, None)

            ll = (hg * np.log(lam) - lam - lam_const
                  + ag * np.log(mu) - mu - mu_const + np.log(tau))
            pen = 50.0 * att.sum() ** 2 + ridge * (att @ att + dfn @ dfn)
            if G:
                pen += ridge_league * (lg @ lg)
            f = -np.sum(w * ll) + pen

            # ---- gradient
            g_lam = hg / lam - 1.0
            g_mu = ag / mu - 1.0

            d_lam = np.zeros_like(lam)
            d_mu = np.zeros_like(lam)
            d_rho = np.zeros_like(lam)

            d_lam[m00] = -mu[m00] * rho / tau[m00]
            d_mu[m00] = -lam[m00] * rho / tau[m00]
            d_rho[m00] = -lam[m00] * mu[m00] / tau[m00]
            d_lam[m01] = rho / tau[m01]
            d_rho[m01] = lam[m01] / tau[m01]
            d_mu[m10] = rho / tau[m10]
            d_rho[m10] = mu[m10] / tau[m10]
            d_rho[m11] = -1.0 / tau[m11]

            a_home = w * (g_lam + d_lam) * lam
            a_away = w * (g_mu + d_mu) * mu

            g_att = -(np.bincount(hi, a_home, n) + np.bincount(ai, a_away, n))
            g_def = -(np.bincount(ai, a_home, n) + np.bincount(hi, a_away, n))
            g_home = -a_home.sum()
            g_rho = -np.sum(w * d_rho)

            g_att = g_att + 100.0 * att.sum() + 2.0 * ridge * att
            g_def = g_def + 2.0 * ridge * dfn
            if G:
                g_lg = -np.bincount(gi, a_home + a_away, G) + 2.0 * ridge_league * lg
                grad = np.concatenate([g_att, g_def, [g_home, g_rho], g_lg])
            else:
                grad = np.concatenate([g_att, g_def, [g_home, g_rho]])
            return f, grad

        theta0 = np.zeros(2 * n + 2 + G)
        theta0[2 * n] = 0.25
        theta0[2 * n + 1] = -0.05
        bounds = [(None, None)] * (2 * n + 2 + G)
        bounds[2 * n + 1] = (-0.35, 0.35)

        res = minimize(objective, theta0, jac=True, method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": 500, "ftol": 1e-9})

        att, dfn, home, rho, lg = unpack(res.x)
        self.attack = att.copy()
        self.defence = dfn.copy()
        self.home_adv = float(home)
        self.rho = float(rho)
        self.league_offset = {g: float(lg[i]) for i, g in enumerate(groups)} if G else {}
        return self

    # ------------------------------------------------------------- predicting
    def _params(self, team: str) -> tuple[float, float]:
        i = self._index.get(team)
        a, d = (float(self.attack[i]), float(self.defence[i])) if i is not None else (0.0, 0.0)
        adj = self.adjustments.get(team)
        if adj is not None:
            kind, factor = adj
            if kind == "attack":
                a = a * factor
            elif kind == "defense":
                d = d * factor
        return a, d

    def set_adjustment(self, team: str, kind: str, factor: float) -> None:
        """Human scouting override, e.g. a strong starting XI vs. mostly subs."""
        self.adjustments[team] = (kind, factor)

    def score_matrix(self, home: str, away: str, group: str | None = None) -> np.ndarray:
        a_h, d_h = self._params(home)
        a_a, d_a = self._params(away)
        offset = self.league_offset.get(group, 0.0) if group else 0.0
        lam = float(np.exp(a_h + d_a + self.home_adv + offset))
        mu = float(np.exp(a_a + d_h + offset))

        k = np.arange(MAX_GOALS + 1)
        grid = np.outer(poisson.pmf(k, lam), poisson.pmf(k, mu))
        grid[0, 0] *= max(1e-10, 1.0 - lam * mu * self.rho)
        grid[0, 1] *= max(1e-10, 1.0 + lam * self.rho)
        grid[1, 0] *= max(1e-10, 1.0 + mu * self.rho)
        grid[1, 1] *= max(1e-10, 1.0 - self.rho)
        return grid / grid.sum()

    def predict(self, home: str, away: str, group: str | None = None) -> dict:
        grid = self.score_matrix(home, away, group)
        goals = np.arange(MAX_GOALS + 1)
        totals = np.add.outer(goals, goals)
        return {
            "p_home": float(np.tril(grid, -1).sum()),
            "p_draw": float(np.trace(grid)),
            "p_away": float(np.triu(grid, 1).sum()),
            "exp_home_goals": float((grid.sum(axis=1) * goals).sum()),
            "exp_away_goals": float((grid.sum(axis=0) * goals).sum()),
            "p_over_2_5": float(grid[totals > 2.5].sum()),
            "top_scores": top_scores(grid, 3),
        }


def top_scores(grid: np.ndarray, n: int) -> list[tuple[str, float]]:
    flat = grid.ravel()
    idx = flat.argsort()[::-1][:n]
    return [(f"{i // grid.shape[1]}-{i % grid.shape[1]}", float(flat[i])) for i in idx]
