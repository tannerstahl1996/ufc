"""Locked data splits. Decided 2026-10-06, before any modeling.

Main analysis: fights under the 2017 Unified Rules criteria only.
  train  2017-01-01 .. 2021-12-31
  tune   2022-01-01 .. 2023-12-31
  test   2024-01-01 ..            (locked; see evaluate.require_test_unlock)

Pre-2017 fights are kept as a separate "pre2017" set for the rule-change
comparison. They are never used to fit the main model.

Note: commissions adopted the revised criteria at slightly different times
during 2017. Using 2017-01-01 as the cut is a simplification, noted as a
limitation.

The grouped random split exists only as a drift diagnostic: it keeps all
three cards of a fight together and stratifies by year, so the gap between it
and the time split estimates how much scoring changes over time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RULES_CUTOFF = pd.Timestamp("2017-01-01")
BOUNDS = {
    "train": (pd.Timestamp("2017-01-01"), pd.Timestamp("2021-12-31")),
    "tune": (pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31")),
    "test": (pd.Timestamp("2024-01-01"), pd.Timestamp("2100-01-01")),
}


def assign_time_split(fights: pd.DataFrame) -> pd.Series:
    split = pd.Series("pre2017", index=fights.index)
    for name, (lo, hi) in BOUNDS.items():
        split[(fights.date >= lo) & (fights.date <= hi)] = name
    return split


def grouped_random_split(fights: pd.DataFrame, seed: int = 0) -> pd.Series:
    """Drift diagnostic only. Same proportions as the time split, stratified by year."""
    post = fights[fights.date >= RULES_CUTOFF]
    sizes = assign_time_split(post).value_counts(normalize=True)
    rng = np.random.default_rng(seed)
    out = pd.Series("pre2017", index=fights.index)
    for _, grp in post.groupby("year"):
        idx = rng.permutation(grp.index.to_numpy())
        n_tr = int(round(len(idx) * sizes["train"]))
        n_tu = int(round(len(idx) * sizes["tune"]))
        out[idx[:n_tr]] = "train"
        out[idx[n_tr:n_tr + n_tu]] = "tune"
        out[idx[n_tr + n_tu:]] = "test"
    return out
