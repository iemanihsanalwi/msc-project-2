import pandas as pd
s3 = pd.read_csv('Scripts/Stage 3/artifacts/stage3_fold_results.csv')
print("=== Stage 3 Fold Results ===")
print(s3[['variant', 'fold', 'val_block', 'AP', 'AP_calibrated', 'ECE', 'ECE_quantile', 'ECE_calibrated', 'Brier', 'Brier_calibrated', 'Loss_cost_rule', 'Loss_cost_rule_calibrated']])

s4 = pd.read_csv('Scripts/Stage 4/artifacts/stage4_fold_results.csv')
print("\n=== Stage 4 Fold Results ===")
print(s4[['variant', 'fold', 'val_block', 'AP', 'AP_calibrated', 'ECE', 'ECE_quantile', 'ECE_calibrated', 'Brier', 'Brier_calibrated', 'Loss_cost_rule', 'Loss_cost_rule_calibrated']])
