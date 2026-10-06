"""Round features: what fighter A did minus what fighter B did, per round.

The features are chosen so no stat is counted twice. UFCStats splits the same
significant strikes three ways (by target, by position, and in total), so using
more than one of those splits makes the weights impossible to interpret.

  dist, clinch, ground   significant strikes LANDED, by position (sums to total sig)
  sig_missed             significant strikes attempted but missed (volume / aggression)
  nonsig                 non-significant strikes landed (mostly short ground strikes)
  kd                     knockdowns
  td                     takedowns landed
  td_failed              takedowns attempted but stuffed
  sub                    submission attempts
  rev                    reversals
  ctrl_min               control time, minutes

There is deliberately NO feature that knows which fighter is listed first.
UFCStats lists the eventual winner first in about 62% of decisions, so fighter
order leaks the result. Every feature is A minus B, and the model has no
intercept, which makes swapping A and B exactly mirror the prediction.
"""
import numpy as np
import pandas as pd

FEATURES = ["dist", "clinch", "ground", "sig_missed", "nonsig", "kd",
            "td", "td_failed", "sub", "rev", "ctrl_min"]
# Secondary, pre-registered variant (2026-10-06, after the tape review): split landed
# significant strikes by TARGET instead of by position. Same everything else.
FEATURES_TARGET = ["head", "body", "leg"] + FEATURES[3:]
FEATURE_SETS = {"position": FEATURES, "target": FEATURES_TARGET}

UNITS = {
    "head": "head sig. strike", "body": "body sig. strike", "leg": "leg sig. strike",
    "dist": "distance sig. strike", "clinch": "clinch sig. strike", "ground": "ground sig. strike",
    "sig_missed": "missed sig. strike", "nonsig": "non-sig. strike", "kd": "knockdown",
    "td": "takedown", "td_failed": "failed takedown", "sub": "submission attempt",
    "rev": "reversal", "ctrl_min": "minute of control",
}


def side(rounds: pd.DataFrame, s: str) -> pd.DataFrame:
    return pd.DataFrame({
        "dist": rounds[f"dist_{s}"], "clinch": rounds[f"clinch_{s}"], "ground": rounds[f"ground_{s}"],
        "sig_missed": rounds[f"sig_att_{s}"] - rounds[f"sig_{s}"],
        "nonsig": rounds[f"tot_{s}"] - rounds[f"sig_{s}"],
        "kd": rounds[f"kd_{s}"], "td": rounds[f"td_{s}"],
        "td_failed": rounds[f"td_att_{s}"] - rounds[f"td_{s}"],
        "sub": rounds[f"sub_{s}"], "rev": rounds[f"rev_{s}"], "ctrl_min": rounds[f"ctrl_{s}"] / 60.0,
        "head": rounds[f"head_{s}"], "body": rounds[f"body_{s}"], "leg": rounds[f"leg_{s}"],
    })


def round_features(rounds: pd.DataFrame, feature_set: str = "position") -> pd.DataFrame:
    """A-minus-B differential for every round, in natural units."""
    return (side(rounds, "A") - side(rounds, "B"))[FEATURE_SETS[feature_set]]
