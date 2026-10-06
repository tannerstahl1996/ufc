import numpy as np
import pandas as pd
import pytest

from ufcj.load import load
from ufcj.features import FEATURES, round_features
from ufcj.model import Prepared, category_probs, unpack, round_win_prob


@pytest.fixture(scope="module")
def small():
    t = load()
    fids = t["fights"].fid.iloc[:60]
    return t["cards"][t["cards"].fid.isin(fids)], t["rounds"][t["rounds"].fid.isin(fids)]


def _swap(rounds: pd.DataFrame) -> pd.DataFrame:
    """Relabel fighter A as B and vice versa."""
    ren = {}
    for c in rounds.columns:
        if c.endswith("_A"):
            ren[c] = c[:-2] + "_B"
        elif c.endswith("_B"):
            ren[c] = c[:-2] + "_A"
    return rounds.rename(columns=ren)


def test_features_antisymmetric(small):
    _, r = small
    assert np.allclose(round_features(r).to_numpy(), -round_features(_swap(r)).to_numpy())


def test_category_probs_sum_to_one_and_mirror():
    eta = np.linspace(-5, 5, 21)
    P = category_probs(eta, 0.2, 3.0)
    assert np.allclose(P.sum(axis=1), 1)
    assert np.allclose(P, category_probs(-eta, 0.2, 3.0)[:, ::-1])  # no built-in edge for either fighter


def test_score_distribution_sums_to_one(small):
    c, r = small
    prep = Prepared(c, r)
    params = np.concatenate([np.random.default_rng(1).normal(size=len(FEATURES)), [np.log(0.2), np.log(2.0)]])
    w, t1, t2 = unpack(params)
    for D in prep.score_dist(prep.X @ w, t1, t2).values():
        assert np.allclose(D.sum(axis=(1, 2)), 1)


def test_swapping_fighters_mirrors_prediction(small):
    c, r = small
    prep = Prepared(c, r)
    params = np.concatenate([np.random.default_rng(2).normal(size=len(FEATURES)), [np.log(0.2), np.log(2.0)]])
    p = round_win_prob(prep, params)
    sw = c.rename(columns={"A_pts": "B_pts", "B_pts": "A_pts"})
    prep_sw = Prepared(sw, _swap(r), scale=prep.scale)
    assert np.allclose(p, 1 - round_win_prob(prep_sw, params))
