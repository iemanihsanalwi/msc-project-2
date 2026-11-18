import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression

# Load Block 5 predictions
pred_oot = pd.read_csv('Scripts/Stage 5/artifacts/sealed_oot_block5_predictions.csv')
y_true = pred_oot['isFraud'].values
amounts = pred_oot['TransactionAmt'].values

# Load S3 and S4 OOF
s3_oof = pd.read_csv('Scripts/Stage 3/artifacts/stage3_oof_predictions.csv')
s4_oof = pd.read_csv('Scripts/Stage 4/artifacts/stage4_oof_predictions.csv')

# Fit calibrators on full valid OOF
valid_s3 = s3_oof['stage3_prob_raw'].notna() & s3_oof['stage3_prob_calibrated'].notna()
iso_s3 = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
iso_s3.fit(s3_oof.loc[valid_s3, 'stage3_prob_raw'].values, s3_oof.loc[valid_s3, 'stage3_prob_calibrated'].values)

valid_s4 = s4_oof['stage4_prob_raw'].notna() & s4_oof['stage4_prob_calibrated'].notna()
iso_s4 = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
iso_s4.fit(s4_oof.loc[valid_s4, 'stage4_prob_raw'].values, s4_oof.loc[valid_s4, 'stage4_prob_calibrated'].values)

# Get S3n and S4n raw
s3n_raw = pred_oot['3n - Res-GNN, No Graph_raw'].values
s4n_raw = pred_oot['4n - Joint BGNN, No Boost_raw'].values

# Calibrate S3n and S4n
s3n_cal = np.clip(iso_s3.predict(s3n_raw), 0.0, 1.0)
s4n_cal = np.clip(iso_s4.predict(s4n_raw), 0.0, 1.0)

# ECE function
def expected_calibration_error(y_true, y_prob, n_bins=20, strategy="quantile"):
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    n = len(y_true_arr)
    if strategy == "quantile":
        edges = np.unique(np.quantile(y_prob_arr, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 3:
            return expected_calibration_error(y_true_arr, y_prob_arr, n_bins, "uniform")
        bin_idx = np.clip(np.digitize(y_prob_arr, edges[1:-1], right=False), 0, len(edges) - 2)
        n_eff = len(edges) - 1
    else:
        bin_idx = np.clip(np.digitize(y_prob_arr, np.linspace(0.0, 1.0, n_bins + 1)[1:-1], right=False), 0, n_bins - 1)
        n_eff = n_bins
    total_error = sum(
        np.sum(bin_idx == b) * abs(y_prob_arr[bin_idx == b].mean() - y_true_arr[bin_idx == b].mean())
        for b in range(n_eff) if np.sum(bin_idx == b) > 0
    )
    return float(total_error / n)

# Cost rule threshold
def alert_threshold(amounts, c_rev=10.0, c_fric=25.0):
    return np.minimum((c_rev + c_fric) / (np.asarray(amounts, dtype=float) + c_fric), 1.0)

def financial_loss(preds, y_true, amounts, c_rev=10.0, c_fric=25.0):
    stolen = amounts[(y_true == 1) & (preds == 0)].sum()
    review = preds.sum() * c_rev
    friction = ((y_true == 0) & (preds == 1)).sum() * c_fric
    return float(stolen + review + friction)

athresh = alert_threshold(amounts, 10.0, 25.0)

print("=== CALIBRATION METRICS FOR S3n & S4n ===")
for name, y_raw, y_cal in [('3n - Res-GNN, No Graph', s3n_raw, s3n_cal), ('4n - Joint BGNN, No Boost', s4n_raw, s4n_cal)]:
    ap_raw = average_precision_score(y_true, y_raw)
    ap_cal = average_precision_score(y_true, y_cal)
    ece_raw = expected_calibration_error(y_true, y_raw, 20, "quantile")
    ece_cal = expected_calibration_error(y_true, y_cal, 20, "quantile")
    brier_raw = brier_score_loss(y_true, y_raw)
    brier_cal = brier_score_loss(y_true, y_cal)
    
    preds_cost_raw = (y_raw > athresh).astype(int)
    preds_cost_cal = (y_cal > athresh).astype(int)
    
    l_raw = financial_loss(preds_cost_raw, y_true, amounts, 10.0, 25.0)
    l_cal = financial_loss(preds_cost_cal, y_true, amounts, 10.0, 25.0)
    
    alert_rate_raw = preds_cost_raw.mean()
    alert_rate_cal = preds_cost_cal.mean()
    
    print(f"\nModel: {name}")
    print(f"  Raw: AP={ap_raw:.4f}, ECE={ece_raw:.4f}, Brier={brier_raw:.4f}, Loss=${l_raw:,.2f}, AlertRate={alert_rate_raw:.4%}")
    print(f"  Cal: AP={ap_cal:.4f}, ECE={ece_cal:.4f}, Brier={brier_cal:.4f}, Loss=${l_cal:,.2f}, AlertRate={alert_rate_cal:.4%}")

# DUAL REGIME EVALUATION (Item 3 in Step 1)
# Regime 1: Fixed probability threshold (tau = 0.50)
# Regime 2: Fixed 1.0% Queue Capacity Regime (Top 1.0% / Top 1,151 Highest-Scoring Transactions)
# Using c_miss=1.0, c_review=$5.00, c_friction=$15.00, c_residual=0.05
def financial_loss_dual(preds, y_true, amounts, c_miss=1.0, c_rev=5.0, c_fric=15.0, c_res=0.05):
    # Loss = sum [ y(1-a)*c_miss*A + a*c_rev + (1-y)*a*c_fric + y*a*c_res*A ]
    stolen = (amounts * y_true * (1 - preds) * c_miss).sum()
    review = (preds * c_rev).sum()
    friction = (((1 - y_true) * preds) * c_fric).sum()
    residual = (amounts * y_true * preds * c_res).sum()
    return float(stolen + review + friction + residual)

k_cap = int(np.floor(0.01 * len(y_true))) # 1,151 transactions

print("\n" + "="*110)
print(f"{'Stage':28s} | {'Regime 1: Raw (0.5)':20s} | {'Regime 1: Cal (0.5)':20s} | {'Regime 2: Top 1.0% Cap':22s} | {'Alerts (Regime 1)':18s}")
print("="*110)

models_all = [
    ('0 - Random Floor', '0 - Random Floor'),
    ('0 - Amount Only', '0 - Amount Only'),
    ('0 - Relational Only', '0 - Relational Only'),
    ('1 - Tabular GBDT', '1 - Tabular GBDT'),
    ('2 - Graph-Augmented GBDT', '2 - Graph-Augmented GBDT'),
    ('3n - Res-GNN, No Graph', None), # use s3n_cal
    ('3 - Frozen Res-GNN', '3 - Frozen Res-GNN'),
    ('4n - Joint BGNN, No Boost', None), # use s4n_cal
    ('4 - Joint BGNN', '4 - Joint BGNN')
]

for label, col in models_all:
    if col is not None:
        y_r = pred_oot[f"{col}_raw"].values
        y_c = pred_oot[f"{col}_calibrated"].values
    elif '3n' in label:
        y_r = s3n_raw
        y_c = s3n_cal
    else:
        y_r = s4n_raw
        y_c = s4n_cal
        
    # Regime 1: tau = 0.5
    preds_r_05 = (y_r >= 0.5).astype(int)
    preds_c_05 = (y_c >= 0.5).astype(int)
    loss_r_05 = financial_loss_dual(preds_r_05, y_true, amounts)
    loss_c_05 = financial_loss_dual(preds_c_05, y_true, amounts)
    
    # Regime 2: Top 1% queue cap
    order_r = np.argsort(-y_r)
    preds_top1 = np.zeros(len(y_true), dtype=int)
    preds_top1[order_r[:k_cap]] = 1
    loss_top1 = financial_loss_dual(preds_top1, y_true, amounts)
    
    print(f"{label:28s} | ${loss_r_05:18,.2f} | ${loss_c_05:18,.2f} | ${loss_top1:20,.2f} | {preds_r_05.sum():6d} ({preds_r_05.mean():.2%})")
