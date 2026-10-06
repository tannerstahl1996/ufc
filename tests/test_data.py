"""Data tests. If any of these fail, nothing downstream should be trusted."""
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ufcj.load import build_tables, parse_details, ROOT
from ufcj.splits import assign_time_split, grouped_random_split, RULES_CUTOFF


@pytest.fixture(scope="module")
def t():
    return build_tables()


def test_raw_files_match_pinned_commit():
    m = json.loads((ROOT / "data" / "MANIFEST.json").read_text())
    for f, sha in m["files"].items():
        assert hashlib.sha256((ROOT / "data" / "raw" / f).read_bytes()).hexdigest() == sha, f


def test_parser_handles_missing_and_glued_names():
    assert parse_details("Brian Tyler28 - 29.Felicia Oh 27 - 30.") == [("Brian Tyler", 28, 29), ("Felicia Oh", 27, 30)]
    assert parse_details("25 - 30.26 - 30.Junichiro Kamijo 27 - 30.")[0] == ("UNKNOWN", 25, 30)
    assert parse_details("Sal D'amato 47 - 48.")[0] == ("Sal D'amato", 47, 48)


def test_three_cards_per_fight(t):
    assert (t["cards"].groupby("fid").size() == 3).all()


def test_every_fight_has_complete_round_stats(t):
    have = t["rounds"].groupby("fid").rnd.nunique()
    need = t["fights"].set_index("fid").n
    assert (have.reindex(need.index) == need).all()
    assert not t["rounds"].isna().any().any()


def test_orientation_winner_is_ahead_on_majority_of_cards(t):
    c = t["cards"].merge(t["fights"][["fid", "winner"]])
    winner_pts = np.where(c.winner == "A", c.A_pts, c.B_pts)
    loser_pts = np.where(c.winner == "A", c.B_pts, c.A_pts)
    ahead = pd.Series(winner_pts > loser_pts).groupby(c.fid.values).sum()
    assert (ahead >= 2).all()


def test_card_totals_feasible(t):
    c = t["cards"]
    tot = c.A_pts + c.B_pts
    assert ((tot >= 17 * c.n) & (tot <= 20 * c.n)).all()
    assert (c[["A_pts", "B_pts"]].max(axis=1) <= 10 * c.n).all()


def test_k_A_only_defined_for_all_10_9_cards(t):
    c = t["cards"]
    assert c.loc[c.dev != 0, "k_A"].isna().all()
    k = c.loc[c.dev == 0, "k_A"]
    assert ((k >= 0) & (k <= c.loc[c.dev == 0, "n"])).all()


def test_known_fight_ufc_284(t):
    """Makhachev vs. Volkanovski: 48-47, 48-47, 49-46 Makhachev. Checked by hand."""
    f = t["fights"]
    fid = f.loc[f.bout.str.contains("Makhachev") & f.event.str.startswith("UFC 284"), "fid"].item()
    c = t["cards"][t["cards"].fid == fid]
    assert sorted(zip(c.A_pts, c.B_pts)) == [(48, 47), (48, 47), (49, 46)]
    assert t["rounds"][t["rounds"].fid == fid].rnd.tolist() == [1, 2, 3, 4, 5]


def test_exclusions_are_logged_not_silent(t):
    assert t["n_decisions_raw"] == len(t["fights"]) + t["exclusions"].fid.nunique()


def test_splits_disjoint_and_respect_rules_cutoff(t):
    f = t["fights"].copy()
    f["split"] = assign_time_split(f)
    assert set(f.split) == {"pre2017", "train", "tune", "test"}
    assert (f.loc[f.split != "pre2017", "date"] >= RULES_CUTOFF).all()
    assert f.groupby("split").date.max()["train"] < f.groupby("split").date.min()["tune"]
    assert f.groupby("split").date.max()["tune"] < f.groupby("split").date.min()["test"]


def test_drift_split_keeps_fights_whole_and_post2017(t):
    f = t["fights"].copy()
    s = grouped_random_split(f)
    assert s.index.equals(f.index)  # one label per fight, so all three cards travel together
    assert (s[f.date < RULES_CUTOFF] == "pre2017").all()


def test_public_reports_carry_no_raw_stat_columns():
    """Raw UFCStats stat lines stay local; reports hold only model output and analysis."""
    tape = pd.read_csv(ROOT / "reports" / "tape_review.csv")
    assert not any(c.endswith("_fav_opp") for c in tape.columns)
