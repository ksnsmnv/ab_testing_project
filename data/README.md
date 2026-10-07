# Marketing A/B Test — Synthetic with Known Treatment Effect

A **synthetic** randomized experiment (50/50 control vs treatment) for practicing
**A/B test analysis, significance testing, and uplift / CATE modeling**. The catch
that makes it genuinely useful: the **true treatment effect is documented**
(`ground_truth.json`), including its **heterogeneity** — the treatment helps **mobile**
and **brand-new** users most. So you can check whether your uplift model recovers the
real conditional effects, not just the average.

- **Rows:** 40,000 · **Target:** `converted` (1 = converted) · **Design:** randomized 50/50
- The documented **true average treatment effect** is in `ground_truth.json`.

## Why it's useful
- **Verifiable causal inference:** compare your estimated ATE/CATE to the truth.
- **Heterogeneous effects:** practice uplift modeling, not just a difference in means.
- **Clean & ready:** no missing values, one row per user, balanced groups.

## Files
- `ab_test.csv` — 40k rows, 8 columns
- `ground_truth.json` — the baseline conversion model + the true (heterogeneous) uplift

## Suggested tasks
1. Estimate the average treatment effect + a confidence interval; compare to ground truth.
2. Build an uplift/CATE model; check it finds the mobile and new-user segments.
3. Practice power analysis and segment-level significance.

*Synthetic data, CC0. Generated from a documented baseline model + heterogeneous uplift + noise.*
