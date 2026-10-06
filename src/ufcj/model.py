"""Round-scoring model learned from judges' fight totals.

How it works, in plain terms
----------------------------
1. Each round gets a score  eta = w . x,  where x is the A-minus-B stat
   differential for that round and w are the weights we want to learn.
   Positive eta favours fighter A.
2. eta is turned into probabilities for the five ways a judge can score a
   round, using an ordered ("cumulative logit") model with two thresholds:

       B 10-8 | B 10-9 | 10-10 | A 10-9 | A 10-8
           -t2      -t1      +t1      +t2

   The thresholds are symmetric and there is no intercept, so the model
   cannot learn "fighter A tends to win" (fighter order leaks the result;
   see features.py).
3. We never see which round a judge gave to whom, only the fight total.
   So for each fight we add up the probability of every round-by-round
   combination that produces each possible total (a small dynamic program
   over the rounds), and score the model on how likely it makes each
   judge's ACTUAL total. The weights that make the observed cards most
   likely are the fit.

This uses every card, including ones with 10-8 and 10-10 rounds. Cards that
need a 10-7 round (impossible in this model) are skipped and counted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .features import FEATURES, FEATURE_SETS, round_features

# (A points, B points) for each of the five round outcomes, minus 8
A_OFF = np.array([0, 1, 2, 2, 2])
B_OFF = np.array([2, 2, 2, 1, 0])


def category_probs(eta: np.ndarray, t1: float, t2: float) -> np.ndarray:
    """eta (R,) -> probabilities (R, 5) in order B10-8, B10-9, 10-10, A10-9, A10-8."""
    cuts = np.array([-t2, -t1, t1, t2])
    cdf = expit(cuts[None, :] - eta[:, None])  # P(outcome <= boundary)
    cdf = np.concatenate([np.zeros((len(eta), 1)), cdf, np.ones((len(eta), 1))], axis=1)
    return np.clip(np.diff(cdf, axis=1), 1e-12, 1.0)


class Prepared:
    """Arrays laid out for fast likelihood evaluation. Built once per data subset."""

    def __init__(self, cards: pd.DataFrame, rounds: pd.DataFrame, scale: pd.Series | None = None,
                 feature_set: str = "position"):
        rounds = rounds.sort_values(["fid", "rnd"]).reset_index(drop=True)
        self.features = FEATURE_SETS[feature_set]
        self.feature_set = feature_set
        X = round_features(rounds, feature_set)
        self.scale = X.std() if scale is None else scale  # scale only: centring would add an intercept
        self.X = (X / self.scale)[self.features].to_numpy()
        self.rounds = rounds
        self.groups = {}  # n_rounds -> (fight ids, round index matrix)
        for n, g in rounds.groupby(rounds.groupby("fid").rnd.transform("size")):
            fids = g.fid.unique()
            idx = g.index.to_numpy().reshape(len(fids), n)
            self.groups[int(n)] = (fids, idx)
        self.fight_pos = {fid: (n, i) for n, (fids, _) in self.groups.items() for i, fid in enumerate(fids)}

        c = cards[cards.fid.isin(self.fight_pos)].copy()
        c["a_i"] = c.A_pts - 8 * c.n
        c["b_i"] = c.B_pts - 8 * c.n
        ok = (c.a_i >= 0) & (c.b_i >= 0)
        self.n_skipped = int((~ok).sum())
        self.cards = c[ok].reset_index(drop=True)

    def score_dist(self, eta: np.ndarray, t1: float, t2: float) -> dict[int, np.ndarray]:
        """Per fight, the full distribution over (A points - 8n, B points - 8n)."""
        P = category_probs(eta, t1, t2)
        out = {}
        for n, (fids, idx) in self.groups.items():
            size = 2 * n + 1
            D = np.zeros((len(fids), size, size))
            D[:, 0, 0] = 1.0
            for r in range(n):
                p = P[idx[:, r]]  # (F, 5)
                new = np.zeros_like(D)
                for c in range(5):
                    da, db = A_OFF[c], B_OFF[c]
                    new[:, da:, db:] += D[:, :size - da, :size - db] * p[:, c, None, None]
                D = new
            out[n] = D
        return out

    def card_loglik(self, params: np.ndarray) -> np.ndarray:
        w, t1, t2 = unpack(params)
        dist = self.score_dist(self.X @ w, t1, t2)
        ll = np.empty(len(self.cards))
        for j, c in enumerate(self.cards.itertuples()):
            n, i = self.fight_pos[c.fid]
            ll[j] = np.log(max(dist[n][i, c.a_i, c.b_i], 1e-300))
        return ll


def unpack(params):
    w = params[:-2]
    t1 = np.exp(params[-2])
    t2 = t1 + np.exp(params[-1])
    return w, t1, t2


def fit(prep: Prepared, ridge: float = 1e-3, x0=None) -> np.ndarray:
    k = prep.X.shape[1]
    x0 = np.concatenate([np.zeros(k), [np.log(0.3), np.log(2.0)]]) if x0 is None else x0

    def obj(p):
        return -prep.card_loglik(p).mean() + ridge * np.sum(p[:k] ** 2)

    res = minimize(obj, x0, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError(res.message)
    return res.x


def round_win_prob(prep: Prepared, params: np.ndarray) -> np.ndarray:
    """P(A wins the round | the round is not scored 10-10), aligned to prep.rounds.
    This is what the locked evaluation metrics consume."""
    w, t1, t2 = unpack(params)
    P = category_probs(prep.X @ w, t1, t2)
    a, b = P[:, 3] + P[:, 4], P[:, 0] + P[:, 1]
    return a / (a + b)


def card_predictions(prep: Prepared, params: np.ndarray) -> pd.DataFrame:
    """Model-native predictions per fight: P(judge has A ahead | not a drawn card),
    and the single most likely exact card."""
    w, t1, t2 = unpack(params)
    dist = prep.score_dist(prep.X @ w, t1, t2)
    rows = []
    for n, (fids, _) in prep.groups.items():
        size = 2 * n + 1
        ai, bi = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        for i, fid in enumerate(fids):
            D = dist[n][i]
            pa, pb = D[ai > bi].sum(), D[ai < bi].sum()
            top = np.unravel_index(D.argmax(), D.shape)
            rows.append({"fid": fid, "p_A_ahead": pa / (pa + pb),
                         "top_A": top[0] + 8 * n, "top_B": top[1] + 8 * n})
    return pd.DataFrame(rows)


def weights_table(params: np.ndarray, scale: pd.Series, features: list[str] = FEATURES) -> pd.DataFrame:
    """Weights in natural units (log-odds per one unit of the stat differential)."""
    w, _, _ = unpack(params)
    return pd.DataFrame({"per_sd": w, "per_unit": w / scale[features].to_numpy()}, index=features)
