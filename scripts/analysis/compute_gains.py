import pandas as pd

fi = pd.read_csv('Scripts/Stage 2/artifacts/stage2_feature_importance.csv')
top20 = fi.head(20)
total_gain = fi['gain_mean'].sum()
top20_gain = top20['gain_mean'].sum()
rel_top20 = top20[top20['is_relational']]
rel_top20_gain = rel_top20['gain_mean'].sum()
rel_all = fi[fi['is_relational']]
rel_all_gain = rel_all['gain_mean'].sum()

print(f"Total Features: {len(fi)}")
print(f"Total Model Gain: {total_gain:,.2f}")
print(f"Top-20 Gain: {top20_gain:,.2f} ({top20_gain/total_gain:.2%} of total)")
print(f"Relational Features in Top-20: {len(rel_top20)}")
print(f"Relational Top-20 Gain: {rel_top20_gain:,.2f}")
print(f"  -> As % of Top-20: {rel_top20_gain/top20_gain:.2%}")
print(f"  -> As % of Total:  {rel_top20_gain/total_gain:.2%}")
print(f"Total Relational Features: {len(rel_all)}")
print(f"Total Relational Gain: {rel_all_gain:,.2f} ({rel_all_gain/total_gain:.2%} of Total Model Gain)")

print("\n=== TOP 20 FEATURES TABLE ROWS ===")
for rank, (idx, r) in enumerate(top20.iterrows(), 1):
    f_name = r['feature']
    g = r['gain_mean']
    is_rel = str(r['is_relational'])
    print(f'["{rank}", "{f_name}", "{g:,.2f}", "{is_rel}"],')
