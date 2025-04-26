import numpy as np
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel

# --- load or compute your 100 per-user metrics ---
# hits_scores = [0,1,0,1, ...]    # length 100
# ndcg_scores = [0.123, 0.045, ...]  # length 100

# 1) One-sample: is mean ≠ baseline?
#           Hits5, Hits10, NDCG5, NDCG10
baseline1 = [0.068, 0.048, 0.098, 0.057]
baseline2 = [0.065, 0.047, 0.090, 0.055]
hit5 =      [0.400]

t_h5, p_h5 = ttest_1samp(hits_scores, popmean=baseline)
t_n5, p_n5 = ttest_1samp(ndcg_scores, popmean=baseline_ndcg)

print(f"Hits@5 vs {baseline}: t={t_h5:.3f}, p={p_h5:.3f}")
print(f"NDCG@5 vs {baseline_ndcg}: t={t_n5:.3f}, p={p_n5:.3f}")

# 2) Independent two-sample: Model A vs Model B (if you have two lists)
# t_h5_ind, p_h5_ind = ttest_ind(hits_A, hits_B, equal_var=False)

# 3) Paired: same users, A vs B
# t_h5_pair, p_h5_pair = ttest_rel(hits_A, hits_B)

# 4) 95% confidence interval for your sample mean
mean_h5 = np.mean(hits_scores)
se_h5   = np.std(hits_scores, ddof=1) / np.sqrt(len(hits_scores))
ci_low, ci_high = mean_h5 - 1.96*se_h5, mean_h5 + 1.96*se_h5

print(f"Mean Hits@5 = {mean_h5:.3f} (95% CI [{ci_low:.3f}, {ci_high:.3f}])")
