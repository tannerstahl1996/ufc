import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom

from ufcj.evaluate import poisson_binomial, card_scores, ceiling_scores, require_test_unlock


def test_poisson_binomial_matches_binomial_when_equal():
    assert np.allclose(poisson_binomial(np.full(5, 0.3)), binom.pmf(range(6), 5, 0.3))


def test_poisson_binomial_sums_to_one():
    assert np.isclose(poisson_binomial(np.array([0.1, 0.9, 0.5])).sum(), 1)


def _toy():
    rounds = pd.DataFrame({"fid": ["f"] * 3, "rnd": [1, 2, 3]})
    cards = pd.DataFrame({
        "fid": ["f"] * 3, "judge": ["x", "y", "z"], "n": 3,
        "A_pts": [29, 29, 28], "B_pts": [28, 28, 29], "dev": 0, "k_A": [2, 2, 1],
    })
    return cards, rounds


def test_perfect_rule_scores_majority_cards():
    cards, rounds = _toy()
    s = card_scores(cards, rounds, np.array([1.0, 1.0, 0.0]))  # A wins rounds 1-2
    assert s.round_count_acc.tolist() == [1, 1, 0]
    assert s.card_winner_acc.tolist() == [1, 1, 0]


def test_ceiling_is_most_common_card():
    cards, _ = _toy()
    c = ceiling_scores(cards)
    assert np.allclose(c.round_count_acc, 2 / 3) and np.allclose(c.card_winner_acc, 2 / 3)


def test_test_set_locked(monkeypatch):
    monkeypatch.delenv("UFCJ_UNLOCK_TEST", raising=False)
    with pytest.raises(PermissionError):
        require_test_unlock("test")
    require_test_unlock("tune")  # no error
