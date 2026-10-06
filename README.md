# What UFC judges actually score

An evaluation project: how well do round statistics reproduce UFC judges' scorecards, which stats do judges weight, and do individual judges score differently? Judges are anonymized in all published results.

**Status:** Phases 0–3 done: data foundations, locked evaluation and baselines, the round-scoring model (fit on 2017–21, evaluated on 2022–23), and Phase 3 analysis (confident misses, anonymized judge styles, pre/post-2017). Test set untouched.

## Data

Third-party scrape of ufcstats.com from [Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats) (GPL-3.0), pinned to commit `1ccacc5` (last event 2026-10-03). `scripts/fetch_data.py` downloads that exact version and checks the SHA-256 of every file against `data/MANIFEST.json`.

UFCStats only gives each judge's fight total, not round-by-round cards. When a card's totals are consistent with every round being 10-9, the total tells us how many rounds each fighter won (but not which ones). The model in Phase 2 learns round-level weights from those totals.

## Setup (Mac, Terminal)

```bash
cd ufc-judging
python3 -m venv .venv
source .venv/bin/activate          # run this in every new Terminal window before working
pip install -r requirements.txt
python scripts/fetch_data.py       # download + verify pinned raw data
PYTHONPATH=src python -m ufcj.load # build data/processed/*.csv and print exclusions
python -m pytest -q                # 16 data and evaluation tests; all must pass
python scripts/run_baselines.py    # writes reports/phase1_baselines.md
python scripts/run_phase2.py 100   # fit + evaluate + 100-refit bootstrap, ~10 min; use 20 for a quick run
python scripts/run_phase3.py 100   # ~30 min; writes reports/phase3_analysis.md and reports/tape_review.csv
```

## Layout

| Path | What it does |
|---|---|
| `src/ufcj/load.py` | Raw CSVs into `fights`, `cards`, `rounds`, plus an `exclusions` table with a reason for every dropped fight |
| `src/ufcj/splits.py` | Locked splits: train 2017–21, tune 2022–23, test 2024+; pre-2017 held apart; grouped random split for the drift check |
| `src/ufcj/evaluate.py` | Locked metrics, ceilings, fight-level bootstrap CIs, test-set lock |
| `src/ufcj/baselines.py` | One-rule baselines |
| `src/ufcj/features.py` | Non-overlapping A-minus-B round features; no fighter-order information |
| `src/ufcj/model.py` | Ordered five-outcome round model, learned from fight totals with a dynamic program over rounds |
| `src/ufcj/judges.py` | Per-judge striking/grappling/10-8 deviations, LR tests, Benjamini-Hochberg, anonymization |
| `tests/` | Data and evaluation tests |
| `reports/` | Generated results |

## Decisions locked before modeling (2026-10-06)

- Main analysis uses fights from 2017 onward (revised Unified Rules criteria). Pre-2017 is only used for the rule-change comparison.
- The test set (2024+) is touched once, at the end. `evaluate.require_test_unlock` raises unless `UFCJ_UNLOCK_TEST=yes` and logs every access.
- Ceiling = how often a judge matches the most common card for that fight (an in-sample oracle, so a true upper bound).
- Cards with 10-8 or 10-10 rounds are kept. They count toward card-winner accuracy now, and Phase 2 models them directly.
- Judge analysis is fit on all post-2017 fights with year as a control. Judges are published as Judge A–Z.

## Known data issues (handled in load.py)

- 25 duplicated fight rows in the scrape (same UFCStats URL listed twice): deduplicated.
- 21 exact-duplicate round-stat rows dropped. 1 fight with conflicting duplicate stats is excluded.
- Judge names are sometimes missing or glued to the score. The parser handles both; missing names become `UNKNOWN`.
- 122 of 4,138 decisions are excluded. See `data/processed/exclusions.csv` for the reason per fight.
