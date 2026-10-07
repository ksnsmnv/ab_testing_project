# Marketing A/B test (YSDA A/B Week)

End-to-end analysis of a randomized marketing experiment using the methods from `YSDA_ABweek-main` (ШАД A/B Week: t/z tests, Mann–Whitney, Poisson bootstrap, delta method, linearization, bucketing, CUPED, stratification, MDE, AA checks).

**Data:** [Marketing A/B Test — Synthetic (Known Uplift)](https://www.kaggle.com/datasets/krishnaharish1/marketing-ab-test-synthetic-uplift) in `data/` (`ab_test.csv`, 40,000 users; `ground_truth.json`).

Primary metric: `converted`. Documented true ATE: **7.08 pp**, larger on **mobile** and **new** users (`tenure_days < 30`).

## Layout

```
data/                 Kaggle CSV + ground truth
src/data.py           loader and derived flags
src/stats.py          YSDA statistical toolkit
scripts/run_analysis.py
notebooks/01_ab_analysis.ipynb
reports/              results after running the script
YSDA_ABweek-main/     course notebooks (reference)
```

## Setup

```bash
python3 -m pip install -r requirements.txt
python3 scripts/run_analysis.py
```

Open `notebooks/01_ab_analysis.ipynb` for the same pipeline cell by cell. Written outputs land in `reports/results.md` and `reports/figures/`.

## What the analysis does

1. SRM check on the 50/50 split.
2. ATE via Welch t-test, two-proportion z-test, Mann–Whitney, Poisson bootstrap; relative uplift CI (Day 4).
3. Ratio toolkit on conversion-per-user (delta method, two linearizations, 100 buckets).
4. CUPED with `prior_purchases` / `tenure_days`; post-stratification by device and country.
5. Sample-size / MDE formulas from Day 2.
6. Synthetic AA (label shuffle) plus segment ATEs with Holm correction vs documented CATE.
