import pandas as pd
import numpy as np

# 1. Print benchmark table
df_bench = pd.read_csv('Scripts/Stage 5/artifacts/sealed_oot_block5_benchmark.csv')
print("=== SEALED OOT BLOCK 5 BENCHMARK ===")
print(df_bench.to_string())

# 2. Print cohort breakdown table
df_cohort = pd.read_csv('Scripts/Stage 5/artifacts/sealed_oot_block5_cohort_breakdown.csv')
print("\n=== SEALED OOT BLOCK 5 COHORT BREAKDOWN ===")
print(df_cohort.to_string())

# 3. Print hypothesis tests table
df_hypo = pd.read_csv('Scripts/Stage 5/artifacts/sealed_oot_block5_hypothesis_tests.csv')
print("\n=== SEALED OOT BLOCK 5 HYPOTHESIS TESTS ===")
print(df_hypo.to_string())

# 4. Check predictions columns
pred_path = 'Scripts/Stage 5/artifacts/sealed_oot_block5_predictions.csv'
df_pred_head = pd.read_csv(pred_path, nrows=5)
print("\n=== PREDICTIONS HEAD & COLUMNS ===")
print(df_pred_head.columns.tolist())
print(df_pred_head.head(2))
