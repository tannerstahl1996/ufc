# What UFC judges actually score

An evaluation project: how well do round statistics reproduce UFC judges' scorecards, which stats do judges weight, and do individual judges score differently? Judges are anonymized in all published results.

**Status:** Complete through the final, pre-registered test evaluation. Phases 0–3 done: data foundations, locked evaluation and baselines, the round-scoring model (fit on 2017–21, evaluated on 2022–23), and Phase 3 analysis (confident misses, anonymized judge styles, pre/post-2017). The test set (2024+) was evaluated once, as pre-registered.

## Headline results (test set, 2024-01-01 to 2026-10-03, 715 fights, 2,145 cards)

| Method | Picks each judge's winner | Exact round count |
|---|---|---|
| More significant strikes (best one-stat rule) | 75.3% [72.4, 78.0] | 49.1% [46.3, 52.1] |
| **Model** (fit on 2017–23) | **83.3%** [81.1, 85.6] | **59.8%** [57.3, 62.8] |
| Ceiling: each fight's most common card | 94.1% [93.1, 95.0] | 82.9% [81.4, 84.3] |

Calibration error 0.027. The pre-registered secondary hypotheses both held: judges value a landed leg strike at about half a head strike (0.50 [0.33, 0.63]), and the model's confident misses lean on leg-strike volume. Full results: `reports/test_results.md`. Pre-registration: `reports/preregistration_test.md`.

Confident-miss review (`reports/tape_review_categories.csv`; per-round stat lines kept local): categorized from post-fight coverage and media scorecards, not tape. In 10 of the 25 most confident misses from 2017–23, media scorers sided with the model against all three judges.

## Data and terms of use

**This repository does not include or redistribute any UFC data.** Raw and processed data are gitignored, and the published reports contain only aggregate results and model outputs.

The analysis was run on a third-party scrape of ufcstats.com ([Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats), GPL-3.0), pinned to commit `1ccacc5` (last event 2026-10-03). `scripts/fetch_data.py` downloads that version and checks the SHA-256 of every file against `data/MANIFEST.json`, but only when run with `--i-have-read-the-terms`.

UFC's [Terms of Use](https://www.ufc.com/terms) prohibit scraping and "constructing any kind of database" from UFC website content except for personal, non-commercial use. Whether they cover ufcstats.com is unclear. If you reproduce this work, read the terms and decide for yourself. This project is non-commercial research and commentary.

UFCStats only gives each judge's fight total, not round-by-round cards. When a card's totals are consistent with every round being 10-9, the total tells us how many rounds each fighter won (but not which ones). The model learns round-level weights from those totals.

## Setup (Mac, Terminal)

```bash
cd ufc-judging
python3 -m venv .venv
source .venv/bin/activate          # run this in every new Terminal window before working
pip install -r requirements.txt
python scripts/fetch_data.py --i-have-read-the-terms
PYTHONPATH=src python -m ufcj.load
python -m pytest -q
python scripts/run_baselines.py    # writes reports/phase1_baselines.md
python scripts/run_phase2.py 100   # fit + evaluate + 100-refit bootstrap, ~10 min; use 20 for a quick run
python scripts/run_phase3.py 100   # ~30 min; writes reports/phase3_analysis.md and reports/tape_review.csv
UFCJ_UNLOCK_TEST=yes python scripts/run_test.py 100   # the one-time test evaluation (already run; re-running reproduces it)
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

## Limitations

- Labels are judges' fight totals. Round-by-round cards aren't available, so round-level conclusions come from the model, not from observed round scores.
- Point deductions appear in card totals and are read as 10-8 rounds. They can't be detected in this data.
- UFCStats counts come from a stats vendor and don't measure damage, which is the top official criterion. At least one recording error was found (a fight with missing ground strikes).
- Only fights that went to a decision are included.
- The weights are predictive associations with judges' cards, not a description of what judges consciously score.
- Judge results can't separate a style preference from reacting to things the stats don't record.
