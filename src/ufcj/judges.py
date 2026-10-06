"""Judge-level scoring styles, measured against the shared model.

For each judge we keep the global round model fixed and let three things move:

  a  striking multiplier   how much more (or less) this judge weights the
                           striking part of a round than the average judge
  b  grappling multiplier  same, for takedowns, control, submissions, reversals
  c  10-8 threshold shift  positive = needs a MORE dominant round to give 10-8
                           (stricter); negative = gives 10-8s more readily

  eta_judge = (1 + a) * eta_striking + (1 + b) * eta_grappling
  t2_judge  = t2 * exp(c)

All three are pulled toward zero with a ridge penalty, so a judge with few
cards stays close to the average unless the evidence is strong (the same idea
as a random effect). Each judge scores each fight once, so a judge's cards are
independent fights, which makes a likelihood-ratio test against "this judge
scores like the average judge" reasonable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

from .features import FEATURES
from .model import Prepared, unpack

STRIKING = ["dist", "clinch", "ground", "sig_missed", "nonsig", "kd"]
GRAPPLING = ["td", "td_failed", "sub", "rev", "ctrl_min"]
S_IDX = [FEATURES.index(f) for f in STRIKING]
G_IDX = [FEATURES.index(f) for f in GRAPPLING]


def judge_loglik(prep: Prepared, global_params: np.ndarray, dev: np.ndarray) -> float:
    w, t1, t2 = unpack(global_params)
    a, b, c = dev
    eta = (1 + a) * (prep.X[:, S_IDX] @ w[S_IDX]) + (1 + b) * (prep.X[:, G_IDX] @ w[G_IDX])
    dist = prep.score_dist(eta, t1, t2 * np.exp(c))
    ll = 0.0
    for card in prep.cards.itertuples():
        n, i = prep.fight_pos[card.fid]
        ll += np.log(max(dist[n][i, card.a_i, card.b_i], 1e-300))
    return ll


def fit_judge(prep: Prepared, global_params: np.ndarray, ridge: float = 2.0) -> tuple[np.ndarray, float, float]:
    """Returns (deviations, loglik at deviations, loglik at zero)."""
    def obj(d):
        return -judge_loglik(prep, global_params, d) + ridge * np.sum(d ** 2)

    res = minimize(obj, np.zeros(3), method="L-BFGS-B", bounds=[(-0.9, 3.0), (-0.9, 3.0), (-2.0, 2.0)])
    return res.x, judge_loglik(prep, global_params, res.x), judge_loglik(prep, global_params, np.zeros(3))


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values (q-values)."""
    p = np.asarray(p)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def lr_pvalue(ll_dev: float, ll_zero: float, df: int = 3) -> float:
    return float(chi2.sf(max(2 * (ll_dev - ll_zero), 0.0), df))


def anonymize(judges: list[str], seed: int = 2017) -> dict[str, str]:
    """Random, stable letters. The mapping is never written to the reports."""
    rng = np.random.default_rng(seed)
    letters = [chr(ord("A") + i) for i in range(26)]
    shuffled = rng.permutation(len(judges))
    return {j: f"Judge {letters[k]}" for j, k in zip(sorted(judges), shuffled)}
