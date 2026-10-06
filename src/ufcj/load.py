"""Turn the raw UFCStats scrape into three clean tables.

fights  one row per decided fight (fid = UFCStats fight id)
cards   one row per judge scorecard, oriented to fighter A / fighter B
rounds  one row per fight-round, with stats for A and B side by side

Every fight that is dropped goes into an `exclusions` table with a reason,
so nothing disappears silently.

Key facts about the raw format (verified in tests/test_data.py):
- DETAILS holds the cards, e.g. "Sal D'amato 47 - 48.Mike Bell 47 - 48.Derek Cleary 46 - 49."
- In "X - Y", Y is always the score that judge gave the OFFICIAL WINNER.
- Judge names are sometimes missing or glued to the score ("Brian Tyler28 - 29").
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

STAT_COLS = {  # raw column -> short name, "x of y" columns give landed + attempted
    "SIG.STR.": "sig", "TOTAL STR.": "tot", "TD": "td",
    "HEAD": "head", "BODY": "body", "LEG": "leg",
    "DISTANCE": "dist", "CLINCH": "clinch", "GROUND": "ground",
}
COUNT_COLS = {"KD": "kd", "SUB.ATT": "sub", "REV.": "rev"}
FEATURES = (
    [f"{v}" for v in STAT_COLS.values()]
    + [f"{v}_att" for v in STAT_COLS.values()]
    + list(COUNT_COLS.values())
    + ["ctrl"]
)

SCORE_RE = re.compile(r"(\d{1,2})\s*-\s*(\d{1,2})")


def _strip(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        try:
            df[c] = df[c].str.strip()
        except (AttributeError, TypeError):
            pass
    return df


def parse_details(details: str) -> list[tuple[str, int, int]]:
    """Return [(judge, x, y), ...]. Missing judge names become 'UNKNOWN'."""
    out, prev = [], 0
    s = str(details)
    for m in SCORE_RE.finditer(s):
        name = s[prev:m.start()].strip(" .")
        out.append((name if name else "UNKNOWN", int(m.group(1)), int(m.group(2))))
        prev = m.end()
    return out


def _landed(x):
    try:
        return int(str(x).split(" of ")[0])
    except ValueError:
        return np.nan


def _attempted(x):
    try:
        return int(str(x).split(" of ")[1])
    except (ValueError, IndexError):
        return np.nan


def _seconds(x):
    try:
        m, s = str(x).split(":")
        return int(m) * 60 + int(s)
    except ValueError:
        return np.nan


def build_tables(raw: Path = RAW) -> dict[str, pd.DataFrame]:
    res = _strip(pd.read_csv(raw / "ufc_fight_results.csv"))
    ev = _strip(pd.read_csv(raw / "ufc_event_details.csv"))
    st = _strip(pd.read_csv(raw / "ufc_fight_stats.csv"))

    ev["date"] = pd.to_datetime(ev["DATE"], format="%B %d, %Y", errors="coerce")
    res = res.merge(ev[["EVENT", "date"]], on="EVENT", how="left")
    res["fid"] = res["URL"].str.rsplit("/", n=1).str[-1]

    exclusions: list[dict] = []
    # the scrape contains a few duplicated fights (same UFCStats URL listed twice)
    n_dup_fights = int(res.fid.duplicated().sum())
    res = res.drop_duplicates("fid", keep="first")
    st = st.drop_duplicates()
    conflicting = st[st.duplicated(["EVENT", "BOUT", "ROUND", "FIGHTER"], keep=False)][["EVENT", "BOUT"]]
    st = st.drop_duplicates(["EVENT", "BOUT", "ROUND", "FIGHTER"], keep=False)

    def exclude(fid, reason):
        exclusions.append({"fid": fid, "reason": reason})

    dec = res[res["METHOD"].str.startswith("Decision", na=False)].copy()
    dec[["A", "B"]] = dec["BOUT"].str.split(r"\s+vs\.\s+", n=1, expand=True)
    dec["A"], dec["B"] = dec["A"].str.strip(), dec["B"].str.strip()
    dec["n"] = pd.to_numeric(dec["ROUND"], errors="coerce").astype("Int64")
    dec["scheduled"] = dec["TIME FORMAT"].str.extract(r"(\d+) Rnd")[0].astype(float)
    dec["winner"] = np.select([dec.OUTCOME == "W/L", dec.OUTCOME == "L/W"], ["A", "B"], default="")

    bad_stats = dec.merge(conflicting.drop_duplicates(), left_on=["EVENT", "BOUT"], right_on=["EVENT", "BOUT"])
    for fid in bad_stats.fid:
        exclude(fid, "conflicting duplicate round stats")
    for fid in dec.loc[dec.winner == "", "fid"]:
        exclude(fid, "draw or no winner (card orientation unknown)")
    for fid in dec.loc[dec.date.isna(), "fid"]:
        exclude(fid, "no event date")
    for fid in dec.loc[dec.n.astype(float) != dec.scheduled, "fid"]:
        exclude(fid, "decision before scheduled final round (technical decision)")

    # ---- cards ---------------------------------------------------------------
    rows = []
    for r in dec.itertuples():
        parsed = parse_details(r.DETAILS)
        if len(parsed) != 3:
            exclude(r.fid, f"{len(parsed)} judge cards parsed")
            continue
        for judge, x, y in parsed:
            a_pts, b_pts = (y, x) if r.winner == "A" else (x, y)
            rows.append(dict(fid=r.fid, judge=judge, winner_pts=y, loser_pts=x,
                             A_pts=a_pts, B_pts=b_pts))
    cards = pd.DataFrame(rows)
    cards = cards.merge(dec[["fid", "n", "METHOD"]], on="fid")
    cards["n"] = cards["n"].astype(int)
    cards["dev"] = cards.A_pts + cards.B_pts - 19 * cards.n  # 0 => consistent with all-10-9
    cards["card_winner"] = np.sign(cards.winner_pts - cards.loser_pts)  # +1 winner, 0 draw card, -1 dissent

    # feasibility: every round totals 17-20 points, nobody above 10 per round
    infeasible = (
        (cards.A_pts + cards.B_pts < 17 * cards.n)
        | (cards.A_pts + cards.B_pts > 20 * cards.n)
        | (cards[["A_pts", "B_pts"]].max(axis=1) > 10 * cards.n)
    )
    for fid in cards.loc[infeasible, "fid"].unique():
        exclude(fid, "card total impossible for scheduled rounds")

    # orientation must agree with the announced method
    tally = cards.groupby("fid").card_winner.agg(
        fav=lambda s: (s == 1).sum(), dis=lambda s: (s == -1).sum(), tie=lambda s: (s == 0).sum()
    ).join(dec.set_index("fid")["METHOD"])
    expected = {
        "Decision - Unanimous": lambda t: t.fav == 3,
        "Decision - Split": lambda t: (t.fav == 2) & (t.dis == 1),
        "Decision - Majority": lambda t: (t.fav == 2) & (t.tie == 1),
    }
    ok = pd.Series(False, index=tally.index)
    for method, rule in expected.items():
        m = tally.METHOD == method
        ok[m] = rule(tally[m])
    for fid in ok[~ok].index:
        exclude(fid, "cards inconsistent with announced decision type")

    # ---- rounds --------------------------------------------------------------
    o = pd.DataFrame({
        "EVENT": st.EVENT, "BOUT": st.BOUT, "FIGHTER": st.FIGHTER,
        "rnd": st.ROUND.str.extract(r"(\d+)")[0].astype(float),
    })
    for raw_col, short in STAT_COLS.items():
        o[short] = st[raw_col].map(_landed)
        o[f"{short}_att"] = st[raw_col].map(_attempted)
    for raw_col, short in COUNT_COLS.items():
        o[short] = pd.to_numeric(st[raw_col], errors="coerce")
    o["ctrl"] = st["CTRL"].map(_seconds)

    keys = dec[["fid", "EVENT", "BOUT", "A", "B", "n"]]
    a = keys.merge(o.rename(columns={"FIGHTER": "A"}), on=["EVENT", "BOUT", "A"])
    b = keys.merge(o.rename(columns={"FIGHTER": "B"}), on=["EVENT", "BOUT", "B"])
    rounds = a.merge(b, on=["fid", "rnd"], suffixes=("_A", "_B"))
    rounds = rounds[["fid", "rnd"] + [f"{f}_A" for f in FEATURES] + [f"{f}_B" for f in FEATURES]]
    rounds = rounds.join(keys.set_index("fid")["n"], on="fid")
    rounds = rounds[rounds.rnd <= rounds.n].drop(columns="n")

    have = rounds.dropna().groupby("fid").rnd.nunique()
    need = dec.set_index("fid")["n"].astype(float)
    complete = have.reindex(need.index).fillna(0) == need
    for fid in complete[~complete].index:
        exclude(fid, "round stats missing or incomplete")

    # ---- assemble --------------------------------------------------------------
    excl = pd.DataFrame(exclusions).drop_duplicates()
    bad = set(excl.fid) if len(excl) else set()
    fights = dec.loc[~dec.fid.isin(bad), ["fid", "date", "EVENT", "BOUT", "A", "B", "winner",
                                          "METHOD", "n", "WEIGHTCLASS"]].rename(
        columns={"EVENT": "event", "BOUT": "bout", "METHOD": "method", "WEIGHTCLASS": "weightclass"})
    fights["n"] = fights.n.astype(int)
    fights["year"] = fights.date.dt.year
    cards = cards[~cards.fid.isin(bad)].drop(columns="METHOD").reset_index(drop=True)
    cards["k_A"] = np.where(cards.dev == 0, cards.A_pts - 9 * cards.n, np.nan)  # rounds won by A if all 10-9
    rounds = rounds[~rounds.fid.isin(bad)].sort_values(["fid", "rnd"]).reset_index(drop=True)

    return {"fights": fights.reset_index(drop=True), "cards": cards, "rounds": rounds,
            "exclusions": excl, "n_decisions_raw": len(dec), "n_duplicate_fights_dropped": n_dup_fights}


def save(tables: dict, out: Path = ROOT / "data" / "processed") -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in ["fights", "cards", "rounds", "exclusions"]:
        tables[name].to_csv(out / f"{name}.csv", index=False)


def load(processed: Path = ROOT / "data" / "processed") -> dict[str, pd.DataFrame]:
    t = {n: pd.read_csv(processed / f"{n}.csv") for n in ["fights", "cards", "rounds", "exclusions"]}
    t["fights"]["date"] = pd.to_datetime(t["fights"]["date"])
    return t


if __name__ == "__main__":
    t = build_tables()
    save(t)
    print(f"duplicate fight rows dropped: {t['n_duplicate_fights_dropped']}")
    print(f"decisions in raw data (deduplicated): {t['n_decisions_raw']}")
    print(f"usable fights: {len(t['fights'])}  cards: {len(t['cards'])}  rounds: {len(t['rounds'])}")
    print("exclusions by reason:")
    print(t["exclusions"].reason.value_counts().to_string())
