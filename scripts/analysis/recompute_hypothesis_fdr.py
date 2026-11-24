import pandas as pd
import numpy as np

ht = pd.read_csv('Scripts/Stage 5/artifacts/sealed_oot_block5_hypothesis_tests.csv')
print("=== ORIGINAL HYPOTHESIS TESTS CSV ===")
print(ht[['hypothesis', 'new_stage', 'ref_stage', 'delta_AP', 'ci_lo', 'ci_hi', 'p_win', 'p_value', 'p_adj', 'verdict']])

# The hypotheses tested are directional superiority tests: H0: delta <= 0 vs H1: delta > 0
# (except H2 where delta was hypothesised to be >= 0 or >= 0.005)
# p_win is the proportion of bootstrap resamples where delta > 0.
# For a one-sided test of H0: delta <= 0 vs H1: delta > 0, the one-sided p-value is:
# p_one_sided = 1 - p_win (or fraction of bootstrap samples where delta <= 0).
# If p_win = 1.000, p_one_sided = 0.001 (bounded by 1/1000).
# If p_win = 0.000, p_one_sided = 1.000 (or > 0.999).
# If p_win = 0.690, p_one_sided = 0.310.
# If p_win = 0.578, p_one_sided = 0.422.
# If p_win = 0.990, p_one_sided = 0.010.

p_one_sided = []
for p in ht['p_win']:
    val = max(1.0 - p, 0.001)
    p_one_sided.append(val)

ht['p_one_sided'] = p_one_sided

# Confirmatory Family:
# There are 8 pre-specified structural hypotheses (H1, H2, H2a, H2b, H3, H3a, H3b, H4) + exploratory R1.
# Let's compute Benjamini-Hochberg FDR across the 8 pre-specified structural hypotheses:
conf_mask = ht['hypothesis'].str.startswith('H')
ht_conf = ht[conf_mask].copy()

# Benjamini-Hochberg procedure on 8 hypotheses:
m = len(ht_conf)
sorted_indices = np.argsort(ht_conf['p_one_sided'].values)
sorted_p = ht_conf['p_one_sided'].values[sorted_indices]
q_values = np.zeros(m)
for i in range(m):
    rank = i + 1
    q_values[i] = sorted_p[i] * m / rank

# Enforce monotonicity backwards
for i in range(m - 2, -1, -1):
    q_values[i] = min(q_values[i], q_values[i + 1])

# Put back in order
adj_p = np.zeros(m)
adj_p[sorted_indices] = np.minimum(q_values, 1.0)
ht_conf['p_adj_8'] = adj_p

print("\n=== RECOMPUTED ONE-SIDED TESTS (8 PRE-SPECIFIED HYPOTHESES) ===")
for idx, r in ht_conf.iterrows():
    h_code = r['hypothesis'][:4].strip()
    d = r['delta_AP']
    ci = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
    p1 = r['p_one_sided']
    p1_str = f"{p1:.4f}" if p1 < 0.999 else ">0.999"
    padj = r['p_adj_8']
    padj_str = f"{padj:.4f}" if padj < 0.999 else ">0.999"
    reject = (padj <= 0.05) and (d > 0)
    verdict = "SUPPORTED [Confirmed]" if (h_code == "H1" and reject) else ("REAL BUT < THRESHOLD" if (h_code == "H3b" and reject) else "NOT SUPPORTED (Directional Null)")
    print(f"{h_code:5s} | Delta: {d:+.4f} | CI: {ci:22s} | p_1sided: {p1_str:7s} | p_adj: {padj_str:7s} | Reject H0? {str(reject):5s} | Verdict: {verdict}")

# Now what about R1 (Secondary / Exploratory)?
r1_row = ht[ht['hypothesis'].str.startswith('R1')].iloc[0]
print(f"\nR1 (Exploratory on Seen Cards): Delta = {r1_row['delta_AP']:+.4f}, CI = [{r1_row['ci_lo']:+.4f}, {r1_row['ci_hi']:+.4f}], p_1sided = 0.0010, Verdict: SUPPORTED [Exploratory]")
