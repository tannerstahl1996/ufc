"""One-rule baselines. Each returns P(A wins the round) for every row of `rounds`.

A round goes to whoever leads on the stat; an exact tie is a coin flip (0.5).
"""
import numpy as np
import pandas as pd


def leader_rule(rounds: pd.DataFrame, stat: str) -> np.ndarray:
    diff = rounds[f"{stat}_A"] - rounds[f"{stat}_B"]
    return np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, 0.5))


BASELINES = {
    "more significant strikes": lambda r: leader_rule(r, "sig"),
    "more total strikes": lambda r: leader_rule(r, "tot"),
    "more control time": lambda r: leader_rule(r, "ctrl"),
}
