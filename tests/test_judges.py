import numpy as np

from ufcj.judges import benjamini_hochberg, anonymize, lr_pvalue


def test_bh_matches_hand_calculation():
    q = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.20]))
    assert np.allclose(q, [0.04, 0.0533333, 0.0533333, 0.20])


def test_anonymize_is_stable_and_hides_names():
    names = ["Zed Judge", "Amy Judge", "Bob Judge"]
    k1, k2 = anonymize(names), anonymize(list(reversed(names)))
    assert k1 == k2
    assert all(v.startswith("Judge ") and len(v) == 7 for v in k1.values())
    assert len(set(k1.values())) == 3


def test_lr_pvalue_bounds():
    assert lr_pvalue(-100.0, -100.0) == 1.0
    assert lr_pvalue(-90.0, -100.0) < 0.001


def test_no_real_judge_names_in_reports():
    """Published outputs must never contain a judge's real name."""
    from pathlib import Path
    from ufcj.load import load
    root = Path(__file__).resolve().parents[1]
    names = {n for n in load()["cards"].judge.unique() if n != "UNKNOWN" and len(n) > 3}
    for f in (root / "reports").glob("*"):
        text = f.read_text()
        leaked = [n for n in names if n in text]
        assert not leaked, f"{f.name} contains judge names: {leaked[:3]}"
