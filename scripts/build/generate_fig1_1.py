import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set publication style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Background boxes
# 1. Incoming Streaming Transaction
p1 = patches.FancyBboxPatch((0.5, 3.8), 2.5, 1.8, boxstyle="round,pad=0.2", ec="#1f77b4", fc="#e8f4f8", lw=2)
ax.add_patch(p1)
ax.text(1.75, 5.2, "Incoming Transaction $v_t$\nat Timestamp $t$", ha="center", va="center", weight="bold", color="#0b3c5d", fontsize=10.5)
ax.text(1.75, 4.3, "• Amount: $x_{\\mathrm{amt}}$\n• Continuous $V$-features\n• Entity Identifiers\n(Card, Device, IP)", ha="center", va="center", fontsize=8.5, color="#333333")

# 2. Tabular Feature Space
p2 = patches.FancyBboxPatch((4.2, 3.8), 2.5, 1.8, boxstyle="round,pad=0.2", ec="#2ca02c", fc="#eafaf1", lw=2)
ax.add_patch(p2)
ax.text(5.45, 5.2, "Tabular Partitioning\n(LightGBM GBDT)", ha="center", va="center", weight="bold", color="#196f3d", fontsize=10.5)
ax.text(5.45, 4.3, "• Exact Axis-Aligned Splits\n• Non-linear Thresholds\n• Robust to Extreme Skew\n• Multiplicative Ratios", ha="center", va="center", fontsize=8.5, color="#333333")

# 3. Relational Graph Space
p3 = patches.FancyBboxPatch((4.2, 0.8), 2.5, 2.2, boxstyle="round,pad=0.2", ec="#d62728", fc="#fdedec", lw=2)
ax.add_patch(p3)
ax.text(5.45, 2.6, "Time-Causal Graph $G_{<t}$\n(Heterogeneous GNN)", ha="center", va="center", weight="bold", color="#922b21", fontsize=10.5)
ax.text(5.45, 1.6, "• Strict Lookback $t' < t$\n• Causal Query Graph\n• Card & Device Links\n• Message Passing Smoothing", ha="center", va="center", fontsize=8.5, color="#333333")

# 4. Hybrid Integration & Blending Gate
p4 = patches.FancyBboxPatch((7.5, 2.0), 2.2, 2.8, boxstyle="round,pad=0.2", ec="#9467bd", fc="#f4ecf7", lw=2)
ax.add_patch(p4)
ax.text(8.6, 4.3, "Adapted BGNN\n& Blending Gate $\\beta_t$", ha="center", va="center", weight="bold", color="#512e5f", fontsize=10.5)
ax.text(8.6, 3.1, "• Alternating Residuals $\\mathbf{R}^m$\n• Exact Tabular Fallback\n• Focal Loss Mitigation\n• Cost-Sensitive Decision", ha="center", va="center", fontsize=8.5, color="#333333")

# Arrows
arrow_style = dict(arrowstyle="->", lw=1.8, color="#2c3e50")

# Arrow from Transaction to Tabular
ax.annotate("", xy=(4.1, 4.7), xytext=(3.1, 4.7), arrowprops=arrow_style)
ax.text(3.6, 4.9, "Raw $\\mathbf{x}_t$", ha="center", va="bottom", fontsize=9, color="#2c3e50")

# Arrow from Transaction to Graph
ax.annotate("", xy=(4.1, 1.9), xytext=(2.6, 3.7), arrowprops=arrow_style)
ax.text(3.1, 2.6, "Ephemeral\nLinking", ha="center", va="center", fontsize=8.5, color="#2c3e50")

# Arrow from Tabular to Hybrid
ax.annotate("", xy=(7.4, 4.0), xytext=(6.8, 4.5), arrowprops=arrow_style)
ax.text(7.1, 4.5, "Logit / $Z_0$", ha="center", va="bottom", fontsize=8.5, color="#2c3e50")

# Arrow from Graph to Hybrid
ax.annotate("", xy=(7.4, 3.2), xytext=(6.8, 2.2), arrowprops=arrow_style)
ax.text(7.1, 2.4, "$\\mathbf{h}_{\\mathrm{graph}}$", ha="center", va="top", fontsize=8.5, color="#2c3e50")

# Operational Decision Output
ax.annotate("", xy=(9.8, 3.4), xytext=(9.7, 3.4), arrowprops=arrow_style)
ax.text(8.6, 1.3, "Fraud Risk Score $\\hat{p}_t$\n& Alert Decision $a_t$", ha="center", va="center", weight="bold", color="#1a5276", fontsize=9.5)

plt.tight_layout()
plt.savefig('figures/fig_conceptual_overview_1_1.png', dpi=300, bbox_inches='tight')
print('Generated figures/fig_conceptual_overview_1_1.png successfully!')
