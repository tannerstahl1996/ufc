# Final test-set results (2024-01-01 to 2026-10-03)

Pre-registration: `reports/preregistration_test.md`, committed a0e6e6f0b5ed798a2bc462a5bf00f2304717401f 2026-10-06T10:55:11-06:00.
Test set accessed once by `scripts/run_test.py`.

Test fights: 715, cards: 2145.

## Primary result

| method                        | card_winner_acc      | round_count_acc      |
|:------------------------------|:---------------------|:---------------------|
| more significant strikes      | 0.753 [0.724, 0.780] | 0.491 [0.463, 0.521] |
| more total strikes            | 0.758 [0.730, 0.785] | 0.481 [0.453, 0.511] |
| more control time             | 0.634 [0.601, 0.663] | 0.340 [0.310, 0.369] |
| MODEL (fit 2017-23) [primary] | 0.833 [0.811, 0.856] | 0.598 [0.573, 0.628] |
| model (fit 2017-21, Phase 2)  | 0.835 [0.814, 0.858] | 0.598 [0.569, 0.629] |
| CEILING                       | 0.941 [0.931, 0.950] | 0.829 [0.814, 0.843] |

Exact card score: model 0.544, ceiling 0.809. Calibration ECE: 0.027.

| bin           |   cards |   predicted |   observed |
|:--------------|--------:|------------:|-----------:|
| (-0.001, 0.1] |     355 |       0.032 |      0.028 |
| (0.1, 0.2]    |     149 |       0.149 |      0.094 |
| (0.2, 0.3]    |     174 |       0.249 |      0.213 |
| (0.3, 0.4]    |     129 |       0.35  |      0.38  |
| (0.4, 0.5]    |     165 |       0.45  |      0.442 |
| (0.5, 0.6]    |     146 |       0.551 |      0.555 |
| (0.6, 0.7]    |     162 |       0.653 |      0.654 |
| (0.7, 0.8]    |     171 |       0.749 |      0.848 |
| (0.8, 0.9]    |     208 |       0.845 |      0.885 |
| (0.9, 1.0]    |     479 |       0.963 |      0.985 |

## H1: leg strikes worth less than head strikes (target-split model fit on 2024+ only)

- Leg / head weight ratio: **0.50** [95% CI 0.33, 0.63]. Prediction (CI excludes 1): **SUPPORTED**.
- Body / head ratio (descriptive): 0.82 [0.66, 1.01].

## H2: confident misses lean on leg strikes

- Confident calls: 362; confident misses: 5 (1.4%).
- Favoured fighter's leg share of significant strikes: misses 0.27, hits 0.15; difference +0.12 [95% CI +0.02, +0.24]. Prediction (positive difference, CI excludes 0): **SUPPORTED**.
- With this few misses, the interval is wide; read it as a replication check, not a precise estimate.

## H3: target-split variant on test (descriptive)

- card_winner_acc 0.838 [0.815, 0.861], round_count_acc 0.603 [0.576, 0.634]

## Judge styles, 2017–2026 (anonymized; letters are reassigned and do not match Phase 3)

Judges with 100+ cards: 14 (covering 4526 of 7143 cards). `*` interval excludes zero; `**` differs from average after Benjamini–Hochberg (q < 0.05).

| judge   |   cards | striking vs avg     | grappling vs avg    | grappling tilt      | 10-8 strictness        | 10-8 card rate   | dissents   | BH q     |
|:--------|--------:|:--------------------|:--------------------|:--------------------|:-----------------------|:-----------------|:-----------|:---------|
| Judge A |     130 | -9% [-26%, +14%]    | -5% [-26%, +33%]    | +4% [-25%, +65%]    | +0.02 [-0.15, +0.23]   | 10.8%            | 10.0%      | 0.390    |
| Judge B |     225 | +8% [-8%, +27%]     | +2% [-17%, +25%]    | -5% [-25%, +15%]    | -0.02 [-0.16, +0.14]   | 14.7%            | 3.6%       | 0.329    |
| Judge C |     190 | +24% [+7%, +50%] *  | +4% [-19%, +28%]    | -16% [-35%, +7%]    | +0.03 [-0.06, +0.17]   | 15.8%            | 5.3%       | 0.059    |
| Judge D |     362 | +8% [-6%, +23%]     | +12% [-6%, +31%]    | +4% [-14%, +25%]    | +0.14 [+0.05, +0.26] * | 8.0%             | 5.8%       | 0.053    |
| Judge E |     834 | +7% [-3%, +20%]     | +9% [-3%, +19%]     | +2% [-14%, +13%]    | +0.10 [+0.04, +0.17] * | 11.2%            | 6.0%       | 0.022 ** |
| Judge F |     372 | +11% [-1%, +34%]    | +16% [+1%, +34%] *  | +4% [-10%, +20%]    | +0.11 [+0.02, +0.23] * | 8.6%             | 5.4%       | 0.213    |
| Judge G |     156 | -6% [-20%, +14%]    | +3% [-20%, +28%]    | +9% [-15%, +43%]    | -0.08 [-0.21, +0.06]   | 14.1%            | 8.3%       | 0.408    |
| Judge H |     268 | -18% [-33%, +1%]    | +11% [-8%, +37%]    | +36% [+6%, +76%] *  | -0.02 [-0.15, +0.11]   | 10.8%            | 5.6%       | 0.069    |
| Judge I |     470 | -11% [-19%, -1%] *  | -15% [-28%, +1%]    | -5% [-25%, +12%]    | -0.07 [-0.14, +0.02]   | 12.6%            | 7.7%       | 0.135    |
| Judge J |     144 | +24% [+2%, +57%] *  | +6% [-14%, +40%]    | -14% [-38%, +16%]   | +0.08 [-0.08, +0.33]   | 11.1%            | 2.1%       | 0.213    |
| Judge K |     194 | +23% [+4%, +49%] *  | -14% [-33%, +13%]   | -30% [-50%, -4%] *  | +0.02 [-0.10, +0.18]   | 13.9%            | 6.7%       | 0.056    |
| Judge L |     545 | -3% [-16%, +9%]     | +42% [+23%, +60%] * | +45% [+26%, +71%] * | +0.06 [-0.03, +0.14]   | 12.8%            | 7.0%       | 0.000 ** |
| Judge M |     115 | +22% [+0%, +45%] *  | -27% [-60%, +9%]    | -40% [-69%, -14%] * | +0.05 [-0.11, +0.20]   | 11.3%            | 8.7%       | 0.135    |
| Judge N |     521 | +22% [+13%, +37%] * | +28% [+13%, +44%] * | +5% [-9%, +17%]     | +0.16 [+0.09, +0.24] * | 12.5%            | 6.7%       | 0.003 ** |
