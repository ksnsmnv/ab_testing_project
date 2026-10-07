# Marketing A/B test results

Randomized 50/50 conversion experiment (`data/ab_test.csv`, 40,000 users).
Documented true ATE: **7.08 pp**.

## Design checks
- Sample ratio mismatch: share treatment = 0.5005, SRM chi-square: estimate=0.50048, p=0.8493
- Conversion control = 11.36%, treatment = 18.79%

## Average treatment effect
- Observed ATE = **7.44 pp** (absolute), relative uplift = 65.5%
- vs ground truth 7.08 pp (difference 0.36 pp)
- Welch t-test: estimate=0.07436, p=1.905e-96, 95% CI [0.06739, 0.08134]
- z-test: estimate=0.07436, p=5.525e-97, 95% CI [0.06739, 0.08134]
- two-proportion z-test: estimate=0.07436, p=6.318e-96, 95% CI [0.06739, 0.08134]
- Mann-Whitney U: estimate=214872525.00000, p=6.353e-96
- Poisson bootstrap (means): estimate=0.07436, p=0, 95% CI [0.06744, 0.08130]
- uplift of means: estimate=0.65485, p=6.566e-58, 95% CI [0.57497, 0.73474]

## Ratio-metric toolkit (here ratio = conversion / user, so it matches the mean)
- delta-method ratio: estimate=0.07436, p=1.814e-96, 95% CI [0.06739, 0.08134]
- linearization type 1: estimate=0.07436, p=1.905e-96, 95% CI [0.06739, 0.08134]
- linearization type 2: estimate=0.07436, p=1.905e-96, 95% CI [0.06739, 0.08134]
- bucket t-test (100): estimate=0.07437, p=1.212e-51, 95% CI [0.06738, 0.08136]

## Sensitivity: CUPED and stratification
- CUPED on `prior_purchases`: CUPED Welch t-test: estimate=0.07442, p=1.323e-96, 95% CI [0.06745, 0.08139]
  Pearson r = 0.013, remaining variance factor = 1.000
- CUPED on `tenure_days`: CUPED Welch t-test: estimate=0.07435, p=1.966e-96, 95% CI [0.06738, 0.08133]
  Pearson r = -0.010
- post-stratified mean diff (device): estimate=0.07438, p=3.879e-97, 95% CI [0.06741, 0.08135]
- post-stratified mean diff (country): estimate=0.07437, p=5.147e-97, 95% CI [0.06740, 0.08135]

## Planning numbers (Day 2 formulas)
- Binary-metric σ = 0.3578
- MDE at n=40000, α=0.05, power=80%, 1:1 split: **1.00 pp**
- Control size needed to detect the true 7.08 pp effect: 401 (test the same at k=1)

## AA validity (400 random reassignments)
- Share of p < 0.05 = 0.040 (target ≈ 0.05)
- KS vs Uniform: stat=0.054, p=0.189

## Heterogeneous effects (Holm-adjusted)
| segment            |   n_test |   n_control |   cr_test |   cr_control |   ate_pp |   ci_low_pp |   ci_high_pp |   pvalue |   pvalue_holm | significant_holm   |
|:-------------------|---------:|------------:|----------:|-------------:|---------:|------------:|-------------:|---------:|--------------:|:-------------------|
| tenure<30          |     2501 |        2529 |    0.2263 |       0.1218 |  10.4522 |      8.3743 |      12.5301 |   0.0000 |        0.0000 | True               |
| device=mobile      |    12478 |       12443 |    0.1852 |       0.0998 |   8.5391 |      7.6776 |       9.4005 |   0.0000 |        0.0000 | True               |
| country=CA         |     2434 |        2506 |    0.1943 |       0.1129 |   8.1401 |      6.1376 |      10.1427 |   0.0000 |        0.0000 | True               |
| country=US         |     8009 |        7825 |    0.1904 |       0.1100 |   8.0379 |      6.9331 |       9.1426 |   0.0000 |        0.0000 | True               |
| country=UK         |     2995 |        3017 |    0.1866 |       0.1090 |   7.7596 |      5.9745 |       9.5447 |   0.0000 |        0.0000 | True               |
| prior_purchases=0  |    10428 |       10209 |    0.1852 |       0.1077 |   7.7426 |      6.7846 |       8.7007 |   0.0000 |        0.0000 | True               |
| overall            |    20019 |       19981 |    0.1879 |       0.1136 |   7.4364 |      6.7389 |       8.1338 |   0.0000 |        0.0000 | True               |
| prior_purchases>=1 |     9591 |        9772 |    0.1909 |       0.1196 |   7.1281 |      6.1117 |       8.1444 |   0.0000 |        0.0000 | True               |
| country=DE         |     2042 |        2069 |    0.1836 |       0.1131 |   7.0545 |      4.8894 |       9.2197 |   0.0000 |        0.0000 | True               |
| tenure>=30         |    17518 |       17452 |    0.1824 |       0.1124 |   7.0076 |      6.2682 |       7.7469 |   0.0000 |        0.0000 | True               |
| country=IN         |     2969 |        2990 |    0.1822 |       0.1144 |   6.7835 |      4.9858 |       8.5812 |   0.0000 |        0.0000 | True               |
| device=desktop     |     7541 |        7538 |    0.1924 |       0.1362 |   5.6172 |      4.4375 |       6.7969 |   0.0000 |        0.0000 | True               |
| country=AU         |     1570 |        1574 |    0.1841 |       0.1398 |   4.4305 |      1.8578 |       7.0032 |   0.0007 |        0.0007 | True               |

The documented uplift is largest for **mobile** and **tenure < 30**. Observed ATEs should rank the same way.

## Decision
Ship the treatment for the average user: conversion rose by 7.44 pp (95% CI 6.74–8.13), close to the known 7.08 pp.
Prioritize rollout on mobile and new users, where the CATE is largest.

Figures: `reports/figures/`.