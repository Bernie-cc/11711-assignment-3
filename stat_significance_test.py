import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

# --- your two samples (possibly different lengths) ---
baseline_results    = np.array([0.068, 0.048, 0.098, 0.057])
exp_results         = np.array([0.061, 0.038, 0.072, 0.041])


# 1) Welch’s two-sample t-test (two-sided)
t_stat, p_two_sided = ttest_ind(
    exp_results,
    baseline_results,
    equal_var=False    # Welch’s correction
)
print(f"Welch t = {t_stat:.3f}, two-sided p = {p_two_sided:.4f}")

# 1a) Convert to one-sided (H1: exp < baseline)
if t_stat < 0:
    p_one_sided = p_two_sided / 2
else:
    p_one_sided = 1 - p_two_sided / 2
print(f"one-sided p (exp < baseline) = {p_one_sided:.4f}")

# 2) 95% CI on mean difference (exp – baseline)
mean_diff = exp_results.mean() - baseline_results.mean()
se_diff   = np.sqrt(
    exp_results.var(ddof=1)/len(exp_results)
  + baseline_results.var(ddof=1)/len(baseline_results)
)
ci_low, ci_high = mean_diff - 1.96*se_diff, mean_diff + 1.96*se_diff
print(f"Mean diff = {mean_diff:.4f}, 95% CI [{ci_low:.4f}, {ci_high:.4f}]")

# 3) Nonparametric fallback: Mann–Whitney U (one-sided)
u_stat, p_mw = mannwhitneyu(
    exp_results,
    baseline_results,
    alternative='less'   # tests exp tends to be lower than baseline
)
print(f"Mann–Whitney U = {u_stat:.3f}, one-sided p = {p_mw:.4f}")

