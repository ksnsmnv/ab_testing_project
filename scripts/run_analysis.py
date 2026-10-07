#!/usr/bin/env python3
"""End-to-end A/B analysis of the marketing conversion experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_dataset, load_ground_truth, split_groups  # noqa: E402
from src.stats import (  # noqa: E402
    aa_pvalue_uniformity,
    bucket_ratio,
    cuped,
    delta_method_ratio,
    linearize_ratio,
    mann_whitney,
    mde_for_sample_size,
    poisson_bootstrap_mean_diff,
    sample_size_for_mde,
    srm_chi2,
    stratified_mean_diff,
    two_proportion_ztest,
    uplift_mean,
    welch_ttest,
    ztest_means,
)

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def fmt(res) -> str:
    ci = ""
    if res.ci_low is not None:
        ci = f", 95% CI [{res.ci_low:.5f}, {res.ci_high:.5f}]"
    return f"{res.name}: estimate={res.estimate:.5f}, p={res.pvalue:.4g}{ci}"


def segment_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    slices = {
        "overall": df.index,
        "device=mobile": df["is_mobile"] == 1,
        "device=desktop": df["is_mobile"] == 0,
        "tenure<30": df["is_new"] == 1,
        "tenure>=30": df["is_new"] == 0,
        "prior_purchases=0": df["is_zero_prior"] == 1,
        "prior_purchases>=1": df["is_zero_prior"] == 0,
    }
    for country in sorted(df["country"].unique()):
        slices[f"country={country}"] = df["country"] == country

    pvalues = []
    records = []
    for name, mask in slices.items():
        part = df.loc[mask]
        t, c = split_groups(part)
        if len(t) < 30 or len(c) < 30:
            continue
        res = welch_ttest(t, c)
        rec = {
            "segment": name,
            "n_test": int(len(t)),
            "n_control": int(len(c)),
            "cr_test": float(t.mean()),
            "cr_control": float(c.mean()),
            "ate_pp": res.estimate * 100,
            "ci_low_pp": res.ci_low * 100,
            "ci_high_pp": res.ci_high * 100,
            "pvalue": res.pvalue,
        }
        records.append(rec)
        pvalues.append(res.pvalue)

    out = pd.DataFrame(records)
    _, q, _, _ = multipletests(out["pvalue"], method="holm")
    out["pvalue_holm"] = q
    out["significant_holm"] = out["pvalue_holm"] < 0.05
    return out.sort_values("ate_pp", ascending=False)


def true_uplift_prob(df: pd.DataFrame) -> np.ndarray:
    """Reconstruct documented additive uplift on probability scale."""
    u = np.full(len(df), 0.03)
    u = np.where(df["is_mobile"] == 1, u + 0.045, u)
    u = np.where(df["is_new"] == 1, u + 0.040, u)
    u = np.where(df["is_zero_prior"] == 1, u + 0.015, u)
    return np.clip(u, 0.0, 0.25)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    gt = load_ground_truth()
    test, control = split_groups(df)
    n_test, n_control = len(test), len(control)

    srm = srm_chi2(n_test, n_control)
    welch = welch_ttest(test, control)
    zres = ztest_means(test, control)
    prop = two_proportion_ztest(test, control)
    mw = mann_whitney(test, control)
    boot = poisson_bootstrap_mean_diff(test, control, n_iterations=1500, random_state=42)
    upl = uplift_mean(test, control)
    cuped_prior = cuped(df["converted"].to_numpy(float), df["prior_purchases"].to_numpy(float), df["treatment"].to_numpy())
    cuped_tenure = cuped(df["converted"].to_numpy(float), df["tenure_days"].to_numpy(float), df["treatment"].to_numpy())
    strat_device = stratified_mean_diff(df, "converted", "device")
    strat_country = stratified_mean_diff(df, "converted", "country")

    ones_t = np.ones(n_test)
    ones_c = np.ones(n_control)
    lin1 = linearize_ratio(test, ones_t, control, ones_c, kind=1)
    lin2 = linearize_ratio(test, ones_t, control, ones_c, kind=2)
    delta_r = delta_method_ratio(test, ones_t, control, ones_c)
    buck = bucket_ratio(test, ones_t, control, ones_c, n_buckets=100)

    sigma = df["converted"].std(ddof=1)
    mde = mde_for_sample_size(sigma, len(df), k=1)
    n_for_true = sample_size_for_mde(sigma, gt["true_average_treatment_effect_pp"] / 100, k=1)

    aa = aa_pvalue_uniformity(df["converted"].to_numpy(float), n_sim=400, random_state=1)

    segs = segment_table(df)
    segs.to_csv(REPORTS / "segment_ates.csv", index=False)

    true_u = true_uplift_prob(df)
    cate_check = (
        df.assign(true_u=true_u)
        .groupby(["device", "is_new", "is_zero_prior"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "true_uplift_pp": 100 * g["true_u"].mean(),
                    "obs_ate_pp": 100
                    * (
                        g.loc[g.treatment == 1, "converted"].mean()
                        - g.loc[g.treatment == 0, "converted"].mean()
                    ),
                }
            ),
            include_groups=False,
        )
    )
    cate_check.to_csv(REPORTS / "cate_vs_truth.csv", index=False)

    # plots
    fig, ax = plt.subplots(figsize=(7, 4))
    rates = df.groupby("group")["converted"].mean().reindex(["control", "treatment"])
    ax.bar(rates.index, rates.values, color=["#4C78A8", "#F58518"])
    ax.set_ylabel("Conversion rate")
    ax.set_title("Overall conversion by group")
    for i, v in enumerate(rates.values):
        ax.text(i, v + 0.003, f"{100*v:.2f}%", ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES / "overall_conversion.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    g = df.groupby(["device", "group"])["converted"].mean().unstack()
    g.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518"])
    ax.set_ylabel("Conversion rate")
    ax.set_title("Conversion by device (true uplift is larger on mobile)")
    ax.legend(title="group")
    fig.tight_layout()
    fig.savefig(FIGURES / "conversion_by_device.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(aa["pvalues"], bins=20, range=(0, 1), color="#4C78A8", edgecolor="white")
    ax.axhline(len(aa["pvalues"]) / 20, color="black", ls="--", lw=1, label="uniform")
    ax.set_xlabel("p-value")
    ax.set_title("AA Welch p-values after random reassignment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "aa_pvalues.png", dpi=120)
    plt.close(fig)

    plot_segs = segs[segs["segment"].isin(["overall", "device=mobile", "device=desktop", "tenure<30", "tenure>=30", "prior_purchases=0", "prior_purchases>=1"])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(plot_segs))
    ax.errorbar(
        plot_segs["ate_pp"],
        y,
        xerr=[plot_segs["ate_pp"] - plot_segs["ci_low_pp"], plot_segs["ci_high_pp"] - plot_segs["ate_pp"]],
        fmt="o",
        color="#4C78A8",
        capsize=3,
    )
    ax.axvline(gt["true_average_treatment_effect_pp"], color="#E45756", ls="--", label="true ATE")
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_segs["segment"])
    ax.set_xlabel("ATE, percentage points")
    ax.set_title("Segment ATEs vs documented overall ATE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "segment_ates.png", dpi=120)
    plt.close(fig)

    true_ate = gt["true_average_treatment_effect_pp"]
    payload = {
        "n": int(len(df)),
        "n_test": n_test,
        "n_control": n_control,
        "cr_test": float(test.mean()),
        "cr_control": float(control.mean()),
        "ate_pp": welch.estimate * 100,
        "true_ate_pp": true_ate,
        "srm_pvalue": srm.pvalue,
        "welch_pvalue": welch.pvalue,
        "aa_type1": aa["share_below_0.05"],
        "aa_ks_pvalue": aa["ks_pvalue"],
        "mde_pp": mde * 100,
        "cuped_prior_var_factor": cuped_prior.extra["var_reduction"],
    }
    (REPORTS / "summary.json").write_text(json.dumps(payload, indent=2))

    lines = []
    w = lines.append
    w("# Marketing A/B test results")
    w("")
    w("Randomized 50/50 conversion experiment (`data/ab_test.csv`, 40,000 users).")
    w(f"Documented true ATE: **{true_ate:.2f} pp**.")
    w("")
    w("## Design checks")
    w(f"- Sample ratio mismatch: share treatment = {srm.estimate:.4f}, {fmt(srm)}")
    w(f"- Conversion control = {100*control.mean():.2f}%, treatment = {100*test.mean():.2f}%")
    w("")
    w("## Average treatment effect")
    w(f"- Observed ATE = **{100*welch.estimate:.2f} pp** (absolute), relative uplift = {100*upl.estimate:.1f}%")
    w(f"- vs ground truth {true_ate:.2f} pp (difference {100*welch.estimate - true_ate:.2f} pp)")
    w(f"- {fmt(welch)}")
    w(f"- {fmt(zres)}")
    w(f"- {fmt(prop)}")
    w(f"- {fmt(mw)}")
    w(f"- {fmt(boot)}")
    w(f"- {fmt(upl)}")
    w("")
    w("## Ratio-metric toolkit (here ratio = conversion / user, so it matches the mean)")
    w(f"- {fmt(delta_r)}")
    w(f"- {fmt(lin1)}")
    w(f"- {fmt(lin2)}")
    w(f"- {fmt(buck)}")
    w("")
    w("## Sensitivity: CUPED and stratification")
    w(f"- CUPED on `prior_purchases`: {fmt(cuped_prior)}")
    w(f"  Pearson r = {cuped_prior.extra['pearson_r']:.3f}, remaining variance factor = {cuped_prior.extra['var_reduction']:.3f}")
    w(f"- CUPED on `tenure_days`: {fmt(cuped_tenure)}")
    w(f"  Pearson r = {cuped_tenure.extra['pearson_r']:.3f}")
    w(f"- {fmt(strat_device)}")
    w(f"- {fmt(strat_country)}")
    w("")
    w("## Planning numbers (Day 2 formulas)")
    w(f"- Binary-metric σ = {sigma:.4f}")
    w(f"- MDE at n={len(df)}, α=0.05, power=80%, 1:1 split: **{100*mde:.2f} pp**")
    w(f"- Control size needed to detect the true {true_ate:.2f} pp effect: {n_for_true:.0f} (test the same at k=1)")
    w("")
    w("## AA validity (400 random reassignments)")
    w(f"- Share of p < 0.05 = {aa['share_below_0.05']:.3f} (target ≈ 0.05)")
    w(f"- KS vs Uniform: stat={aa['ks_stat']:.3f}, p={aa['ks_pvalue']:.3g}")
    w("")
    w("## Heterogeneous effects (Holm-adjusted)")
    w(segs.to_markdown(index=False, floatfmt=".4f"))
    w("")
    w("The documented uplift is largest for **mobile** and **tenure < 30**. Observed ATEs should rank the same way.")
    w("")
    w("## Decision")
    if welch.pvalue < 0.05 and welch.estimate > 0:
        w(
            f"Ship the treatment for the average user: conversion rose by {100*welch.estimate:.2f} pp "
            f"(95% CI {100*welch.ci_low:.2f}–{100*welch.ci_high:.2f}), close to the known {true_ate:.2f} pp."
        )
        w("Prioritize rollout on mobile and new users, where the CATE is largest.")
    else:
        w("Do not ship on the average effect; inspect segments and power.")
    w("")
    w("Figures: `reports/figures/`.")

    (REPORTS / "results.md").write_text("\n".join(lines))
    print("\n".join(lines[:40]))
    print(f"\nWrote {REPORTS / 'results.md'}")


if __name__ == "__main__":
    main()
