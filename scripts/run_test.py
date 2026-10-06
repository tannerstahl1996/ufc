"""Final, one-time test-set evaluation, exactly as pre-registered in
reports/preregistration_test.md. Then re-runs the judge analysis on 2017-2026.

Run with:  UFCJ_UNLOCK_TEST=yes python scripts/run_test.py
Writes reports/test_results.md.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ufcj.load import load  # noqa: E402
from ufcj.splits import assign_time_split  # noqa: E402
from ufcj.evaluate import card_scores, summarize, ceiling_scores, require_test_unlock  # noqa: E402
from ufcj.baselines import BASELINES  # noqa: E402
from ufcj.features import FEATURES_TARGET  # noqa: E402
from ufcj.model import Prepared, fit, round_win_prob, card_predictions, weights_table  # noqa: E402
from ufcj.judges import fit_judge, lr_pvalue, benjamini_hochberg, anonymize  # noqa: E402

N_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 100
rng = np.random.default_rng(2024)
prereg = subprocess.run(["git", "log", "-1", "--format=%H %cI", "--", "reports/preregistration_test.md"],
                        cwd=ROOT, capture_output=True, text=True).stdout.strip()
require_test_unlock("test", reason=f"pre-registered final evaluation ({prereg[:10]})")

t = load()
fights, cards, rounds = t["fights"], t["cards"], t["rounds"]
fights["split"] = assign_time_split(fights)
dev_f = fights[fights.split.isin(["train", "tune"])].fid
tr_f = fights[fights.split == "train"].fid
te_f = fights[fights.split == "test"].fid
C = lambda f: cards[cards.fid.isin(f)]  # noqa: E731
R = lambda f: rounds[rounds.fid.isin(f)]  # noqa: E731


def fmt(s):
    return s.apply(lambda x: f"{x['mean']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}]", axis=1)


L = ["# Final test-set results (2024-01-01 to 2026-10-03)", "",
     f"Pre-registration: `reports/preregistration_test.md`, committed {prereg}.",
     "Test set accessed once by `scripts/run_test.py`.", "",
     f"Test fights: {len(te_f)}, cards: {len(C(te_f))}.", ""]

# ------------------------------------------------------------- primary
prep_dev = Prepared(C(dev_f), R(dev_f))
p_dev = fit(prep_dev)
prep_te = Prepared(C(te_f), R(te_f), scale=prep_dev.scale)
prep_tr = Prepared(C(tr_f), R(tr_f))
p_tr = fit(prep_tr)
prep_te_tr = Prepared(C(te_f), R(te_f), scale=prep_tr.scale)

rows = []
for name, rule in BASELINES.items():
    s = summarize(card_scores(C(te_f), prep_te.rounds, rule(prep_te.rounds))); s.insert(0, "method", name); rows.append(s)
s = summarize(card_scores(C(te_f), prep_te.rounds, round_win_prob(prep_te, p_dev)))
s.insert(0, "method", "MODEL (fit 2017-23) [primary]"); rows.append(s)
s = summarize(card_scores(C(te_f), prep_te_tr.rounds, round_win_prob(prep_te_tr, p_tr)))
s.insert(0, "method", "model (fit 2017-21, Phase 2)"); rows.append(s)
s = summarize(ceiling_scores(C(te_f))); s.insert(0, "method", "CEILING"); rows.append(s)
tab = pd.concat(rows)
tab = tab[tab.metric != "round_count_ll"].assign(result=lambda d: fmt(d))
order = list(BASELINES) + ["MODEL (fit 2017-23) [primary]", "model (fit 2017-21, Phase 2)", "CEILING"]
wide = tab.pivot(index="method", columns="metric", values="result").loc[order]
L += ["## Primary result", "", wide.to_markdown(), ""]

pred = card_predictions(prep_te, p_dev).set_index("fid")
cc = C(te_f).join(pred, on="fid")
exact = ((cc.top_A == cc.A_pts) & (cc.top_B == cc.B_pts)).mean()
modal = C(te_f).groupby("fid").apply(lambda g: g.groupby(["A_pts", "B_pts"]).size().max() / len(g), include_groups=False)
cc = cc[cc.A_pts != cc.B_pts].copy()
cc["y"] = (cc.A_pts > cc.B_pts).astype(int)
cc["bin"] = pd.cut(cc.p_A_ahead, np.linspace(0, 1, 11), include_lowest=True)
cal = cc.groupby("bin", observed=True).agg(cards=("y", "size"), predicted=("p_A_ahead", "mean"), observed=("y", "mean"))
ece = (cal.cards * (cal.predicted - cal.observed).abs()).sum() / cal.cards.sum()
L += [f"Exact card score: model {exact:.3f}, ceiling {modal.mean():.3f}. Calibration ECE: {ece:.3f}.", "",
      cal.round(3).to_markdown(), ""]

# ------------------------------------------------------------- H1: leg discount replicates on test-period data
prep_t1 = Prepared(C(te_f), R(te_f), feature_set="target")
p_t1 = fit(prep_t1)
w1 = weights_table(p_t1, prep_t1.scale, FEATURES_TARGET).per_unit
ratio = w1["leg"] / w1["head"]
cb_ = {f: g for f, g in C(te_f).groupby("fid")}
rb_ = {f: g for f, g in R(te_f).groupby("fid")}
ids = C(te_f).fid.unique()
rb = []
for b in range(N_BOOT):
    pick = pd.Series(rng.choice(ids, len(ids))).value_counts()
    cbb = pd.concat([cb_[f].assign(fid=f"{f}_{k}") for f, n in pick.items() for k in range(n)])
    rbb = pd.concat([rb_[f].assign(fid=f"{f}_{k}") for f, n in pick.items() for k in range(n)])
    pb = Prepared(cbb, rbb, scale=prep_t1.scale, feature_set="target")
    wb = weights_table(fit(pb, x0=p_t1), prep_t1.scale, FEATURES_TARGET).per_unit
    rb.append([wb["leg"] / wb["head"], wb["body"] / wb["head"]])
    print(f"H1 boot {b + 1}/{N_BOOT}", end="\r")
rb = np.array(rb)
lo, hi = np.percentile(rb[:, 0], [2.5, 97.5])
blo, bhi = np.percentile(rb[:, 1], [2.5, 97.5])
L += ["## H1: leg strikes worth less than head strikes (target-split model fit on 2024+ only)", "",
      f"- Leg / head weight ratio: **{ratio:.2f}** [95% CI {lo:.2f}, {hi:.2f}]. "
      f"Prediction (CI excludes 1): **{'SUPPORTED' if hi < 1 else 'NOT SUPPORTED'}**.",
      f"- Body / head ratio (descriptive): {w1['body'] / w1['head']:.2f} [{blo:.2f}, {bhi:.2f}].", ""]

# ------------------------------------------------------------- H2: misses lean on leg strikes
side = C(te_f).assign(s=np.sign(C(te_f).A_pts - C(te_f).B_pts)).groupby("fid").s.agg(
    lambda s: s.iloc[0] if (s == s.iloc[0]).all() and s.iloc[0] != 0 else 0)
pr = pred.copy()
pr["j"] = side
pr["fav"] = np.where(pr.p_A_ahead >= 0.5, 1, -1)
pr["conf"] = np.maximum(pr.p_A_ahead, 1 - pr.p_A_ahead)
u = pr[(pr.j != 0) & (pr.conf >= 0.8)].join(R(te_f).groupby("fid")[["leg_A", "leg_B", "sig_A", "sig_B"]].sum())
u["leg_share"] = np.where(u.fav == 1, u.leg_A / u.sig_A.clip(lower=1), u.leg_B / u.sig_B.clip(lower=1))
m, h = u[u.fav != u.j], u[u.fav == u.j]
d = [rng.choice(m.leg_share, len(m)).mean() - rng.choice(h.leg_share, len(h)).mean() for _ in range(5000)]
dlo, dhi = np.percentile(d, [2.5, 97.5])
L += ["## H2: confident misses lean on leg strikes", "",
      f"- Confident calls: {len(u)}; confident misses: {len(m)} ({len(m) / len(u):.1%}).",
      f"- Favoured fighter's leg share of significant strikes: misses {m.leg_share.mean():.2f}, hits {h.leg_share.mean():.2f}; "
      f"difference {m.leg_share.mean() - h.leg_share.mean():+.2f} [95% CI {dlo:+.2f}, {dhi:+.2f}]. "
      f"Prediction (positive difference, CI excludes 0): **{'SUPPORTED' if dlo > 0 else 'NOT SUPPORTED'}**.",
      "- With this few misses, the interval is wide; read it as a replication check, not a precise estimate.", ""]

# ------------------------------------------------------------- H3: target variant accuracy (descriptive)
prep_dev_t = Prepared(C(dev_f), R(dev_f), feature_set="target")
p_dev_t = fit(prep_dev_t)
prep_te_t = Prepared(C(te_f), R(te_f), scale=prep_dev_t.scale, feature_set="target")
s = summarize(card_scores(C(te_f), prep_te_t.rounds, round_win_prob(prep_te_t, p_dev_t)))
s = s[s.metric != "round_count_ll"].assign(result=lambda d: fmt(d)).set_index("metric").result
L += ["## H3: target-split variant on test (descriptive)", "",
      f"- card_winner_acc {s['card_winner_acc']}, round_count_acc {s['round_count_acc']}", ""]

# ------------------------------------------------------------- judges, 2017-2026
all_f = fights[fights.split.isin(["train", "tune", "test"])].fid
c_all, r_all = C(all_f), R(all_f)
prep_all = Prepared(c_all, r_all)
g = fit(prep_all)
vc = c_all.judge.value_counts()
judges = [j for j in vc[vc >= 100].index if j != "UNKNOWN"]
key = anonymize(judges, seed=2026)
pd.Series(key, name="label").rename_axis("judge").to_csv(ROOT / "data" / "processed" / "judge_key_full.csv")
cm = c_all.assign(s=np.sign(c_all.A_pts - c_all.B_pts))
cm["maj"] = cm.groupby("fid").s.transform(lambda s: np.sign(s.sum()))
out = []
for j in judges:
    cj = c_all[c_all.judge == j]
    pj = Prepared(cj, r_all[r_all.fid.isin(cj.fid)], scale=prep_all.scale)
    dev, ll1, ll0 = fit_judge(pj, g)
    jf = cj.fid.unique()
    cjb = {f: x for f, x in cj.groupby("fid")}
    rjb = {f: x for f, x in r_all[r_all.fid.isin(jf)].groupby("fid")}
    bs = []
    for b in range(N_BOOT):
        pick = pd.Series(rng.choice(jf, len(jf))).value_counts()
        cbb = pd.concat([cjb[f].assign(fid=f"{f}_{k}") for f, n in pick.items() for k in range(n)])
        rbb = pd.concat([rjb[f].assign(fid=f"{f}_{k}") for f, n in pick.items() for k in range(n)])
        bs.append(fit_judge(Prepared(cbb, rbb, scale=prep_all.scale), g)[0])
    bs = np.array(bs)
    tilt_b = (1 + bs[:, 1]) / (1 + bs[:, 0]) - 1
    mj = cm[cm.judge == j]
    out.append({"judge": key[j], "cards": len(cj), "a": dev[0], "a_lo": np.percentile(bs[:, 0], 2.5),
                "a_hi": np.percentile(bs[:, 0], 97.5), "b": dev[1], "b_lo": np.percentile(bs[:, 1], 2.5),
                "b_hi": np.percentile(bs[:, 1], 97.5), "tilt": (1 + dev[1]) / (1 + dev[0]) - 1,
                "t_lo": np.percentile(tilt_b, 2.5), "t_hi": np.percentile(tilt_b, 97.5),
                "c": dev[2], "c_lo": np.percentile(bs[:, 2], 2.5), "c_hi": np.percentile(bs[:, 2], 97.5),
                "p": lr_pvalue(ll1, ll0), "r108": (cj.dev < 0).mean(), "dissent": (mj.s != mj.maj).mean()})
    print("judge", key[j], " " * 20)
J = pd.DataFrame(out)
J["q"] = benjamini_hochberg(J.p.to_numpy())
J = J.sort_values("judge")


def ci(v, lo, hi, pct=True):
    f = (lambda x: f"{x:+.0%}") if pct else (lambda x: f"{x:+.2f}")
    return f"{f(v)} [{f(lo)}, {f(hi)}]" + (" *" if lo > 0 or hi < 0 else "")


show = pd.DataFrame({
    "judge": J.judge, "cards": J.cards,
    "striking vs avg": [ci(*x) for x in zip(J.a, J.a_lo, J.a_hi)],
    "grappling vs avg": [ci(*x) for x in zip(J.b, J.b_lo, J.b_hi)],
    "grappling tilt": [ci(*x) for x in zip(J.tilt, J.t_lo, J.t_hi)],
    "10-8 strictness": [ci(*x, pct=False) for x in zip(J.c, J.c_lo, J.c_hi)],
    "10-8 card rate": J.r108.map("{:.1%}".format),
    "dissents": J.dissent.map("{:.1%}".format),
    "BH q": J.q.map(lambda q: f"{q:.3f}" + (" **" if q < 0.05 else "")),
})
L += ["## Judge styles, 2017–2026 (anonymized; letters are reassigned and do not match Phase 3)", "",
      f"Judges with 100+ cards: {len(judges)} (covering {vc[judges].sum()} of {len(c_all)} cards). "
      "`*` interval excludes zero; `**` differs from average after Benjamini–Hochberg (q < 0.05).", "",
      show.to_markdown(index=False), ""]

(ROOT / "reports" / "test_results.md").write_text("\n".join(L))
print("\n".join(L))
