import pandas as pd
import numpy as np

pred_oot = pd.read_csv('Scripts/Stage 5/artifacts/sealed_oot_block5_predictions.csv')
y_true = pred_oot['isFraud'].values
amounts = pred_oot['TransactionAmt'].values
n_eval = len(y_true)
k_cap = int(np.floor(0.01 * n_eval)) # 1151

s1_raw = pred_oot['1 - Tabular GBDT_raw'].values
s2_raw = pred_oot['2 - Graph-Augmented GBDT_raw'].values
s3_raw = pred_oot['3 - Frozen Res-GNN_raw'].values
s4_raw = pred_oot['4 - Joint BGNN_raw'].values

def compute_queue_loss(scores, y_true, amounts, k_cap, c_miss=1.0, c_rev=5.0, c_fric=15.0, c_res=0.05):
    order = np.argsort(-scores)
    a = np.zeros(len(y_true), dtype=int)
    a[order[:k_cap]] = 1
    stolen = (amounts * y_true * (1 - a) * c_miss).sum()
    review = (a * c_rev).sum()
    friction = (((1 - y_true) * a) * c_fric).sum()
    residual = (amounts * y_true * a * c_res).sum()
    return stolen + review + friction + residual

l1 = compute_queue_loss(s1_raw, y_true, amounts, k_cap)
l2 = compute_queue_loss(s2_raw, y_true, amounts, k_cap)
diff_obs = l1 - l2
print(f"Observed savings S1 - S2: ${diff_obs:,.2f}")

# Block bootstrap over 24-hour blocks (86,400s)
# Each day has ~2,750 transactions. Let's create ~42 contiguous blocks
block_size = int(n_eval / 42)
n_blocks = int(np.ceil(n_eval / block_size))
blocks = [np.arange(b * block_size, min((b + 1) * block_size, n_eval)) for b in range(n_blocks)]

np.random.seed(42)
boot_diffs = []
for _ in range(1000):
    sampled_blocks = np.random.choice(len(blocks), size=len(blocks), replace=True)
    idx = np.concatenate([blocks[b] for b in sampled_blocks])[:n_eval]
    
    y_b = y_true[idx]
    amt_b = amounts[idx]
    s1_b = s1_raw[idx]
    s2_b = s2_raw[idx]
    
    loss1 = compute_queue_loss(s1_b, y_b, amt_b, k_cap)
    loss2 = compute_queue_loss(s2_b, y_b, amt_b, k_cap)
    boot_diffs.append(loss1 - loss2)

boot_diffs = np.array(boot_diffs)
ci_low = np.percentile(boot_diffs, 2.5)
ci_high = np.percentile(boot_diffs, 97.5)
p_val = (boot_diffs <= 0).mean()

print(f"95% Bootstrap CI for S2 savings: [${ci_low:,.2f}, ${ci_high:,.2f}]")
print(f"One-sided p-value (H0: S2 savings <= 0): p = {p_val:.4f}")
