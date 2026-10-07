"""Statistical tools from YSDA A/B Week: t/z tests, MW, bootstrap, delta method,
linearization, bucketing, CUPED, stratification, MDE, SRM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class TestResult:
    name: str
    estimate: float
    pvalue: float
    ci_low: float | None = None
    ci_high: float | None = None
    extra: dict | None = None

    def significant(self, alpha: float = 0.05) -> bool:
        return self.pvalue < alpha


def safe_divide(x, y):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.true_divide(x, y)
        if np.ndim(out) == 0:
            return float(out) if y != 0 else np.nan
        out[~np.isfinite(out)] = np.nan
        return out


def sample_size_for_mde(sigma: float, effect: float, k: float = 1.0, alpha: float = 0.05, beta: float = 0.2) -> float:
    """Control sample size m for 1:k split (test n = m/k). Day 2 formula."""
    dist = stats.norm()
    const = (dist.ppf(1 - alpha / 2) + dist.ppf(1 - beta)) ** 2
    return const * (1 + k) * sigma**2 / effect**2


def mde_for_sample_size(sigma: float, num_users: int, k: float = 1.0, alpha: float = 0.05, beta: float = 0.2) -> float:
    """Absolute MDE given total users and split 1:k (control is the larger group)."""
    dist = stats.norm()
    const = dist.ppf(1 - alpha / 2) + dist.ppf(1 - beta)
    m = num_users * k / (k + 1)
    return const * sigma * np.sqrt((1 + k) / m)


def srm_chi2(n_test: int, n_control: int, p_test: float = 0.5) -> TestResult:
    observed = np.array([n_control, n_test], dtype=float)
    expected = np.array([1 - p_test, p_test]) * observed.sum()
    chi2, pvalue = stats.chisquare(observed, expected)
    return TestResult(
        name="SRM chi-square",
        estimate=float(n_test / observed.sum()),
        pvalue=float(pvalue),
        extra={"chi2": float(chi2), "n_test": n_test, "n_control": n_control},
    )


def welch_ttest(test: np.ndarray, control: np.ndarray, alpha: float = 0.05) -> TestResult:
    test = np.asarray(test, dtype=float)
    control = np.asarray(control, dtype=float)
    delta = test.mean() - control.mean()
    t_stat, pvalue = stats.ttest_ind(test, control, equal_var=False)
    n, m = len(test), len(control)
    se = np.sqrt(test.var(ddof=1) / n + control.var(ddof=1) / m)
    df = n + m - 2
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    return TestResult(
        name="Welch t-test",
        estimate=float(delta),
        pvalue=float(pvalue),
        ci_low=float(delta - t_crit * se),
        ci_high=float(delta + t_crit * se),
        extra={"t_stat": float(t_stat), "se": float(se)},
    )


def ztest_means(test: np.ndarray, control: np.ndarray, alpha: float = 0.05) -> TestResult:
    test = np.asarray(test, dtype=float)
    control = np.asarray(control, dtype=float)
    delta = test.mean() - control.mean()
    n, m = len(test), len(control)
    se = np.sqrt(test.var(ddof=1) / n + control.var(ddof=1) / m)
    z = delta / se
    pvalue = 2 * stats.norm.sf(abs(z))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    return TestResult(
        name="z-test",
        estimate=float(delta),
        pvalue=float(pvalue),
        ci_low=float(delta - z_crit * se),
        ci_high=float(delta + z_crit * se),
        extra={"z": float(z), "se": float(se)},
    )


def mann_whitney(test: np.ndarray, control: np.ndarray) -> TestResult:
    stat, pvalue = stats.mannwhitneyu(test, control, alternative="two-sided")
    return TestResult(name="Mann-Whitney U", estimate=float(stat), pvalue=float(pvalue))


def two_proportion_ztest(test: np.ndarray, control: np.ndarray, alpha: float = 0.05) -> TestResult:
    test = np.asarray(test, dtype=float)
    control = np.asarray(control, dtype=float)
    p_t, p_c = test.mean(), control.mean()
    n, m = len(test), len(control)
    p_pool = (test.sum() + control.sum()) / (n + m)
    se_h0 = np.sqrt(p_pool * (1 - p_pool) * (1 / n + 1 / m))
    z = (p_t - p_c) / se_h0
    pvalue = 2 * stats.norm.sf(abs(z))
    se_unpooled = np.sqrt(p_t * (1 - p_t) / n + p_c * (1 - p_c) / m)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    delta = p_t - p_c
    return TestResult(
        name="two-proportion z-test",
        estimate=float(delta),
        pvalue=float(pvalue),
        ci_low=float(delta - z_crit * se_unpooled),
        ci_high=float(delta + z_crit * se_unpooled),
        extra={"p_test": float(p_t), "p_control": float(p_c)},
    )


def cuped(y: np.ndarray, x_pre: np.ndarray, treatment: np.ndarray, alpha: float = 0.05) -> TestResult:
    y = np.asarray(y, dtype=float)
    x_pre = np.asarray(x_pre, dtype=float)
    treatment = np.asarray(treatment)
    theta = np.cov(y, x_pre, ddof=1)[0, 1] / np.var(x_pre, ddof=1)
    y_cuped = y - theta * (x_pre - x_pre.mean())
    r = np.corrcoef(y, x_pre)[0, 1]
    res = welch_ttest(y_cuped[treatment == 1], y_cuped[treatment == 0], alpha=alpha)
    raw = welch_ttest(y[treatment == 1], y[treatment == 0], alpha=alpha)
    return TestResult(
        name="CUPED Welch t-test",
        estimate=res.estimate,
        pvalue=res.pvalue,
        ci_low=res.ci_low,
        ci_high=res.ci_high,
        extra={
            "theta": float(theta),
            "pearson_r": float(r),
            "var_reduction": float(1 - r**2),
            "raw_estimate": raw.estimate,
            "raw_pvalue": raw.pvalue,
            "ci_width_raw": raw.ci_high - raw.ci_low,
            "ci_width_cuped": res.ci_high - res.ci_low,
        },
    )


def delta_var(numerator: np.ndarray, denominator: np.ndarray) -> float:
    x = np.asarray(numerator, dtype=float)
    y = np.asarray(denominator, dtype=float)
    n = len(x)
    mu_x, mu_y = x.mean(), y.mean()
    var_x, var_y = x.var(ddof=1), y.var(ddof=1)
    cov_xy = np.cov(x, y, ddof=1)[0, 1]
    return float(
        (var_x / mu_y**2 - 2 * cov_xy * mu_x / mu_y**3 + var_y * mu_x**2 / mu_y**4) / n
    )


def delta_method_ratio(
    x_num: np.ndarray,
    x_denom: np.ndarray,
    y_num: np.ndarray,
    y_denom: np.ndarray,
    alpha: float = 0.05,
) -> TestResult:
    n, m = len(x_num), len(y_num)
    test_var = delta_var(x_num, x_denom)
    control_var = delta_var(y_num, y_denom)
    sigma = np.sqrt(test_var + control_var)
    delta = safe_divide(np.mean(x_num), np.mean(x_denom)) - safe_divide(np.mean(y_num), np.mean(y_denom))
    tt = delta / sigma
    pvalue = 2 * stats.t.sf(np.abs(tt), n + m - 2)
    t_crit = stats.t.ppf(1 - alpha / 2, n + m - 2)
    return TestResult(
        name="delta-method ratio",
        estimate=float(delta),
        pvalue=float(pvalue),
        ci_low=float(delta - t_crit * sigma),
        ci_high=float(delta + t_crit * sigma),
        extra={"se": float(sigma)},
    )


def uplift_mean(test: np.ndarray, control: np.ndarray, alpha: float = 0.05) -> TestResult:
    """Relative uplift of means: mu_t / mu_c - 1 (Day 4)."""
    x = np.asarray(test, dtype=float)
    y = np.asarray(control, dtype=float)
    n, m = len(x), len(y)
    mu_x, mu_y = x.mean(), y.mean()
    var_x, var_y = x.var(ddof=1), y.var(ddof=1)
    se2 = var_x / (n * mu_y**2) + var_y * mu_x**2 / (m * mu_y**4)
    se = np.sqrt(se2)
    est = mu_x / mu_y - 1
    tt = est / se
    pvalue = 2 * stats.t.sf(np.abs(tt), n + m - 2)
    t_crit = stats.t.ppf(1 - alpha / 2, n + m - 2)
    return TestResult(
        name="uplift of means",
        estimate=float(est),
        pvalue=float(pvalue),
        ci_low=float(est - t_crit * se),
        ci_high=float(est + t_crit * se),
        extra={"se": float(se), "mu_test": float(mu_x), "mu_control": float(mu_y)},
    )


def linearize_ratio(
    x_num: np.ndarray,
    x_denom: np.ndarray,
    y_num: np.ndarray,
    y_denom: np.ndarray,
    kind: int = 1,
    alpha: float = 0.05,
) -> TestResult:
    x_num, x_denom = np.asarray(x_num, float), np.asarray(x_denom, float)
    y_num, y_denom = np.asarray(y_num, float), np.asarray(y_denom, float)
    x_hat = x_num.mean() / x_denom.mean()
    y_hat = y_num.mean() / y_denom.mean()
    if kind == 1:
        x_lin = x_num - y_hat * x_denom
        y_lin = y_num - y_hat * y_denom
        name = "linearization type 1"
    else:
        x_lin = x_hat + (1 / x_denom.mean()) * (x_num - x_hat * x_denom)
        y_lin = y_hat + (1 / y_denom.mean()) * (y_num - y_hat * y_denom)
        name = "linearization type 2"
    res = welch_ttest(x_lin, y_lin, alpha=alpha)
    return TestResult(
        name=name,
        estimate=float(x_hat - y_hat),
        pvalue=res.pvalue,
        ci_low=res.ci_low,
        ci_high=res.ci_high,
        extra={"linear_delta": res.estimate},
    )


def bucket_ratio(
    x_num: np.ndarray,
    x_denom: np.ndarray,
    y_num: np.ndarray,
    y_denom: np.ndarray,
    n_buckets: int = 100,
    alpha: float = 0.05,
) -> TestResult:
    x_num, x_denom = np.asarray(x_num, float), np.asarray(x_denom, float)
    y_num, y_denom = np.asarray(y_num, float), np.asarray(y_denom, float)

    def ratios(num, denom):
        idx = np.arange(len(num))
        out = []
        for b in range(n_buckets):
            mask = idx % n_buckets == b
            out.append(num[mask].sum() / denom[mask].sum())
        return np.asarray(out)

    t_b = ratios(x_num, x_denom)
    c_b = ratios(y_num, y_denom)
    res = welch_ttest(t_b, c_b, alpha=alpha)
    return TestResult(
        name=f"bucket t-test ({n_buckets})",
        estimate=res.estimate,
        pvalue=res.pvalue,
        ci_low=res.ci_low,
        ci_high=res.ci_high,
    )


def poisson_bootstrap_mean_diff(
    test: np.ndarray,
    control: np.ndarray,
    n_iterations: int = 2000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> TestResult:
    rng = np.random.default_rng(random_state)
    test = np.asarray(test, dtype=float)
    control = np.asarray(control, dtype=float)
    observed = test.mean() - control.mean()
    metrics = np.empty(n_iterations)
    for i in range(n_iterations):
        w_t = rng.poisson(1, size=len(test))
        w_c = rng.poisson(1, size=len(control))
        metrics[i] = (w_t @ test) / w_t.sum() - (w_c @ control) / w_c.sum()
    left, right = np.percentile(metrics, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    pvalue = 2 * min((metrics <= 0).mean(), (metrics >= 0).mean())
    pvalue = float(min(1.0, max(pvalue, 1.0 / n_iterations)))
    return TestResult(
        name="Poisson bootstrap (means)",
        estimate=float(observed),
        pvalue=pvalue,
        ci_low=float(left),
        ci_high=float(right),
    )


def poisson_bootstrap_ratio_diff(
    x_num: np.ndarray,
    x_denom: np.ndarray,
    y_num: np.ndarray,
    y_denom: np.ndarray,
    n_iterations: int = 2000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> TestResult:
    rng = np.random.default_rng(random_state)
    x_num, x_denom = np.asarray(x_num, float), np.asarray(x_denom, float)
    y_num, y_denom = np.asarray(y_num, float), np.asarray(y_denom, float)
    observed = x_num.sum() / x_denom.sum() - y_num.sum() / y_denom.sum()
    metrics = np.empty(n_iterations)
    for i in range(n_iterations):
        w_x = rng.poisson(1, size=len(x_num))
        w_y = rng.poisson(1, size=len(y_num))
        rx = (w_x @ x_num) / (w_x @ x_denom)
        ry = (w_y @ y_num) / (w_y @ y_denom)
        metrics[i] = rx - ry
    left, right = np.percentile(metrics, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    pvalue = 2 * min((metrics <= 0).mean(), (metrics >= 0).mean())
    return TestResult(
        name="Poisson bootstrap (ratio)",
        estimate=float(observed),
        pvalue=float(min(1.0, pvalue)),
        ci_low=float(left),
        ci_high=float(right),
    )


def stratified_mean_diff(
    df: pd.DataFrame,
    metric: str,
    strata: str,
    treatment_col: str = "treatment",
    alpha: float = 0.05,
) -> TestResult:
    """Post-stratified difference of means with population weights (Day 5)."""
    weights = df[strata].value_counts(normalize=True)
    deltas, vars_ = [], []
    used_w = []
    for s, w in weights.items():
        part = df[df[strata] == s]
        t = part.loc[part[treatment_col] == 1, metric].to_numpy(float)
        c = part.loc[part[treatment_col] == 0, metric].to_numpy(float)
        if len(t) < 2 or len(c) < 2:
            continue
        deltas.append(t.mean() - c.mean())
        vars_.append(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
        used_w.append(w)
    used_w = np.asarray(used_w)
    used_w = used_w / used_w.sum()
    deltas = np.asarray(deltas)
    vars_ = np.asarray(vars_)
    est = float(used_w @ deltas)
    se = float(np.sqrt((used_w**2 * vars_).sum()))
    z = est / se
    pvalue = 2 * stats.norm.sf(abs(z))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    return TestResult(
        name=f"post-stratified mean diff ({strata})",
        estimate=est,
        pvalue=float(pvalue),
        ci_low=float(est - z_crit * se),
        ci_high=float(est + z_crit * se),
        extra={"se": se, "n_strata": int(len(deltas))},
    )


def aa_pvalue_uniformity(
    y: np.ndarray,
    n_sim: int = 500,
    test_share: float = 0.5,
    random_state: int = 0,
) -> dict:
    """Synthetic AA: reshuffle treatment and collect Welch p-values (Day 2–3)."""
    rng = np.random.default_rng(random_state)
    y = np.asarray(y, dtype=float)
    pvals = []
    n_test = int(len(y) * test_share)
    for _ in range(n_sim):
        idx = rng.permutation(len(y))
        t, c = y[idx[:n_test]], y[idx[n_test:]]
        pvals.append(stats.ttest_ind(t, c, equal_var=False).pvalue)
    pvals = np.asarray(pvals)
    ks = stats.kstest(pvals, "uniform")
    return {
        "mean_pvalue": float(pvals.mean()),
        "share_below_0.05": float((pvals < 0.05).mean()),
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "pvalues": pvals,
    }
