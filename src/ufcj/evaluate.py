"""Evaluation protocol. Locked 2026-10-06, before any model is fit.

Every method under test produces one number per fight-round: the probability
that fighter A wins that round on a judge's card. From those, we score each
judge's card.

Metrics
-------
1. round_count_acc   On cards consistent with all-10-9 scoring (dev == 0), does
                     the most likely number of rounds won by A (from a
                     Poisson-binomial over the fight's rounds) equal the
                     judge's actual count? Ties in the argmax are split evenly.
2. card_winner_acc   On every non-drawn card (including cards with 10-8s), does
                     the method pick the fighter that judge had ahead? Uses
                     P(A wins a majority of rounds). Exactly 0.5 earns half credit.
3. round_count_ll    Mean log-likelihood of the judge's actual count
                     (probabilistic models only; rules get -inf when wrong).

Ceilings (in-sample oracles, so true upper bounds)
--------------------------------------------------
A method must give the same prediction to all three judges of a fight, so the
best possible score is predicting each fight's most common card:
- count ceiling: per fight, max share of its 10-9 cards that share one round count
- winner ceiling: per fight, max share of its non-drawn cards that agree on a winner

Uncertainty: 95% bootstrap intervals resampling FIGHTS (cards from one fight
are not independent).
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TEST_LOG = ROOT / "reports" / "test_access.log"


def require_test_unlock(split_name: str, reason: str = "") -> None:
    """The test set is touched once, at the end. Unlocking is logged."""
    if split_name != "test":
        return
    if os.environ.get("UFCJ_UNLOCK_TEST") != "yes":
        raise PermissionError("Test set is locked. Set UFCJ_UNLOCK_TEST=yes for the final evaluation only.")
    TEST_LOG.parent.mkdir(exist_ok=True)
    with TEST_LOG.open("a") as f:
        f.write(f"{dt.datetime.now().isoformat()}  test set accessed  {reason}\n")


def poisson_binomial(p: np.ndarray) -> np.ndarray:
    """pmf of the number of successes in independent trials with probabilities p."""
    pmf = np.array([1.0])
    for q in p:
        pmf = np.convolve(pmf, [1.0 - q, q])
    return pmf


def card_scores(cards: pd.DataFrame, rounds: pd.DataFrame, p_round: np.ndarray) -> pd.DataFrame:
    """Score every card. `p_round` is aligned row-for-row with `rounds`."""
    r = rounds[["fid", "rnd"]].copy()
    r["p"] = np.asarray(p_round, dtype=float)
    pmfs = {fid: poisson_binomial(g.sort_values("rnd").p.to_numpy()) for fid, g in r.groupby("fid")}

    out = []
    for c in cards.itertuples():
        pmf = pmfs[c.fid]
        n = len(pmf) - 1
        row = {"fid": c.fid, "judge": c.judge}
        if c.dev == 0:
            k = int(c.k_A)
            top = np.flatnonzero(np.isclose(pmf, pmf.max()))
            row["round_count_acc"] = float(k in top) / len(top)
            row["round_count_ll"] = float(np.log(pmf[k])) if pmf[k] > 0 else -np.inf
        judge_a = np.sign(c.A_pts - c.B_pts)
        if judge_a != 0:
            p_a = pmf[n // 2 + 1:].sum()  # n is odd for every decision in the data
            pick = 0.5 if np.isclose(p_a, 0.5) else float((p_a > 0.5) == (judge_a > 0))
            row["card_winner_acc"] = pick
        out.append(row)
    return pd.DataFrame(out)


def summarize(scores: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Mean per metric over cards, with fight-level bootstrap 95% intervals."""
    metrics = [m for m in ["round_count_acc", "card_winner_acc", "round_count_ll"] if m in scores]
    by_fight_sum = scores.groupby("fid")[metrics].sum(min_count=1)
    by_fight_n = scores.groupby("fid")[metrics].count()
    fids = by_fight_sum.index.to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    for m in metrics:
        s, n = by_fight_sum[m].fillna(0).to_numpy(), by_fight_n[m].to_numpy()
        est = s.sum() / n.sum()
        if not np.isfinite(est):
            rows.append({"metric": m, "mean": est, "lo": np.nan, "hi": np.nan, "cards": int(n.sum())})
            continue
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(fids), len(fids))
            boots.append(s[i].sum() / n[i].sum())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"metric": m, "mean": est, "lo": lo, "hi": hi, "cards": int(n.sum())})
    return pd.DataFrame(rows)


def ceiling_scores(cards: pd.DataFrame) -> pd.DataFrame:
    """Ceilings expressed as per-card scores so they go through the same summarize()."""
    rows = []
    for fid, g in cards.groupby("fid"):
        c109 = g[g.dev == 0]
        share_k = c109.k_A.value_counts().max() / len(c109) if len(c109) else None
        w = np.sign(g.A_pts - g.B_pts)
        share_w = w[w != 0].value_counts().max() / (w != 0).sum() if (w != 0).any() else None
        for c, wi in zip(g.itertuples(), w):
            row = {"fid": fid}
            if c.dev == 0:
                row["round_count_acc"] = share_k
            if wi != 0:
                row["card_winner_acc"] = share_w
            rows.append(row)
    return pd.DataFrame(rows)
