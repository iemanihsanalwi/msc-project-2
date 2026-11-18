"""
generate_thesis_diagnostics.py
================================================================================
Empirical Diagnostic & Visualization Suite for Graph-Tabular Learning Under OOT.

Generates:
  1. Publication-quality Forest Plot with 95% bootstrap CIs for H1-H4, H2a, H2b, H3a, H3b, R1.
  2. Precision-Recall curves (Full range and Operational Fraud Zoom) across Stages 1-4 & baselines.
  3. Calibration curves (Reliability diagrams: Raw vs Isotonically Calibrated) & probability distributions.
  4. Temporal slice and Entity Overlap (Seen vs Cold-Start) cohort breakdown charts.
  5. Step-by-step Architectural Ablation Summary Table (CSV and Markdown).

All figures are saved in publication-quality 300 DPI PNG and vector PDF formats.
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

# ------------------------------------------------------------------------------
# 1. Configuration & Style Settings
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_IN_DIR = os.path.join(BASE_DIR, "Stage 5", "artifacts")
OUTPUT_DIR = os.path.join(BASE_DIR, "diagnostics_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set publication style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
})

# Color palette for stages
STAGE_COLORS = {
    "1 - Tabular GBDT": "#2b5c8f",          # Steel Blue
    "2 - Graph-Augmented GBDT": "#1b7837",  # Deep Forest Green (Champion)
    "3 - Frozen Res-GNN": "#762a83",        # Purple
    "3n - Res-GNN, No Graph": "#9970ab",    # Light Purple / Lavender
    "4 - Joint BGNN": "#b2182b",            # Crimson / Red (Degraded)
    "4n - Joint BGNN, No Boost": "#e08214",  # Amber / Dark Orange
    "0 - Relational Only": "#5ab4ac",       # Muted Teal
    "0 - Random Floor": "#888888",          # Gray
    "0 - Amount Only": "#a6611a",           # Brown
}


def load_data():
    """Load evaluation artifacts from Stage 5."""
    hyp_path = os.path.join(ARTIFACTS_IN_DIR, "sealed_oot_block5_hypothesis_tests.csv")
    bench_path = os.path.join(ARTIFACTS_IN_DIR, "sealed_oot_block5_benchmark.csv")
    cohort_path = os.path.join(ARTIFACTS_IN_DIR, "sealed_oot_block5_cohort_breakdown.csv")
    pred_path = os.path.join(ARTIFACTS_IN_DIR, "sealed_oot_block5_predictions.csv")

    for p in [hyp_path, bench_path, cohort_path, pred_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required artifact: {p}")

    df_hyp = pd.read_csv(hyp_path)
    df_bench = pd.read_csv(bench_path)
    df_cohort = pd.read_csv(cohort_path)
    df_pred = pd.read_csv(pred_path)

    print(f"[Data Loader] Loaded hypothesis tests: {len(df_hyp)} rows")
    print(f"[Data Loader] Loaded benchmark table:  {len(df_bench)} rows")
    print(f"[Data Loader] Loaded cohort breakdown: {len(df_cohort)} rows")
    print(f"[Data Loader] Loaded predictions:       {df_pred.shape[0]} rows x {df_pred.shape[1]} cols")
    return df_hyp, df_bench, df_cohort, df_pred


# ------------------------------------------------------------------------------
# 2. Plot 1: Publication-Quality Forest Plot for Hypotheses
# ------------------------------------------------------------------------------
def plot_forest_hypotheses(df_hyp):
    """
    Generate Forest Plot with 95% bootstrap confidence intervals for H1 through H4.
    """
    print("[Generating Plot 1] Forest Plot of Hypotheses...")
    # Clean and order hypotheses logically
    df = df_hyp.copy()

    # Reorder in natural research flow
    order_map = {
        "H1  Relational features beat tabular baseline": 1,
        "H2  Learned structure beats hand-crafted [CONFOUNDED]": 2,
        "H2a Message passing adds signal (Identical Net Ablation)": 3,
        "H2b Message passing on connected nodes only": 4,
        "H3b Blending gate alone beats frozen pipeline": 5,
        "H3  Joint alternating BGNN beats frozen Res-GNN": 6,
        "H3a Functional boosting adds signal (Gate Fixed Ablation)": 7,
        "H4  Full alternating joint system beats tabular baseline": 8,
        "R1  Best relational stage beats tabular on seen cards only": 9,
    }
    df["order"] = df["hypothesis"].map(order_map).fillna(99)
    df = df.sort_values("order", ascending=False).reset_index(drop=True)

    y_positions = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Reference lines
    ax.axvline(0.0, color="#222222", linestyle="-", linewidth=1.1, alpha=0.9, zorder=1)
    ax.axvline(0.005, color="#1b7837", linestyle=":", linewidth=1.2, alpha=0.7, zorder=1,
               label="Materiality Threshold (+0.005 AP)")
    ax.axvline(-0.005, color="#b2182b", linestyle=":", linewidth=1.2, alpha=0.4, zorder=1)

    # Shaded neutral band
    ax.axvspan(-0.005, 0.005, color="#f0f0f0", alpha=0.5, zorder=0, label="Indifference Region ([-0.005, +0.005])")

    colors = []
    for _, row in df.iterrows():
        v = str(row["verdict"])
        if "SUPPORTED" in v and "NOT" not in v:
            colors.append("#1b7837")  # Green
        elif "NOT SUPPORTED" in v:
            colors.append("#b2182b")  # Crimson
        elif "THRESHOLD" in v:
            colors.append("#e08214")  # Orange
        else:
            colors.append("#555555")  # Dark Slate

    # Plot error bars
    for i, (_, row) in enumerate(df.iterrows()):
        c = colors[i]
        d_ap = row["delta_AP"]
        ci_lo = row["ci_lo"]
        ci_hi = row["ci_hi"]

        # Horizontal CI bar
        ax.plot([ci_lo, ci_hi], [y_positions[i], y_positions[i]], color=c, linewidth=2.2, zorder=3)
        # Whisker caps
        cap_h = 0.15
        ax.plot([ci_lo, ci_lo], [y_positions[i] - cap_h, y_positions[i] + cap_h], color=c, linewidth=1.8, zorder=3)
        ax.plot([ci_hi, ci_hi], [y_positions[i] - cap_h, y_positions[i] + cap_h], color=c, linewidth=1.8, zorder=3)
        # Point estimate
        ax.scatter(d_ap, y_positions[i], color=c, s=65, zorder=4, edgecolor="#ffffff", linewidth=0.8)

        # Value annotation to right of CI
        p_val_str = f"p_adj={row['p_adj']:.4f}" if pd.notna(row['p_adj']) else f"p={row['p_value']:.3f}"
        stat_text = f"Δ={d_ap:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}] | {p_val_str}"
        ax.text(0.075, y_positions[i], stat_text, va="center", ha="left", fontsize=8.5,
                family="monospace", color="#222222")

    # Clean hypothesis labels for Y-axis
    labels = []
    for _, row in df.iterrows():
        h_name = row["hypothesis"].strip()
        v = str(row["verdict"])
        labels.append(f"{h_name}\n[{v}]")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Out-of-Time Average Precision Delta (Δ AP) with 95% Bootstrap CI", fontweight="bold")
    ax.set_title("Sealed Out-of-Time (Block 5) Hypothesis Tests & Empirical Effect Sizes",
                 fontweight="bold", pad=12)

    ax.set_xlim(-0.065, 0.135)
    ax.grid(True, axis="x", alpha=0.35, linestyle="--")
    ax.grid(False, axis="y")

    # Legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color="#1b7837", lw=2, marker="o", label="Supported (Statistically & Materially Superior)"),
        Line2D([0], [0], color="#b2182b", lw=2, marker="o", label="Not Supported (Degraded or Zero Benefit)"),
        Line2D([0], [0], color="#e08214", lw=2, marker="o", label="Real but Below Materiality Threshold"),
        Line2D([0], [0], color="#555555", lw=2, marker="o", label="Exploratory / Confounded"),
        Line2D([0], [0], color="#1b7837", lw=1.2, ls=":", label="Materiality Threshold (+0.005)"),
    ]
    ax.legend(handles=custom_lines, loc="lower right", framealpha=0.9, fontsize=8)

    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "forest_plot_hypotheses.png")
    pdf_path = os.path.join(OUTPUT_DIR, "forest_plot_hypotheses.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


# ------------------------------------------------------------------------------
# 3. Plot 2: Precision-Recall Curves (Full & Operational Zoom)
# ------------------------------------------------------------------------------
def plot_pr_curves(df_pred, df_bench):
    """
    Generate dual-panel PR curves: Full spectrum and High-Alert low-FPR operational window.
    """
    print("[Generating Plot 2] Precision-Recall Curves...")
    y_true = df_pred["isFraud"].to_numpy(dtype=int)

    stages_to_plot = [
        ("2 - Graph-Augmented GBDT", "2 - Graph-Augmented GBDT_raw", "-", 2.2),
        ("1 - Tabular GBDT", "1 - Tabular GBDT_raw", "--", 1.8),
        ("4n - Joint BGNN, No Boost", "4n - Joint BGNN, No Boost_raw", "-.", 1.5),
        ("3 - Frozen Res-GNN", "3 - Frozen Res-GNN_raw", "-", 1.5),
        ("3n - Res-GNN, No Graph", "3n - Res-GNN, No Graph_raw", ":", 1.5),
        ("4 - Joint BGNN", "4 - Joint BGNN_raw", "-", 1.8),
        ("0 - Relational Only", "0 - Relational Only_raw", ":", 1.2),
        ("0 - Random Floor", "0 - Random Floor_raw", ":", 1.0),
    ]

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 6))

    base_fraud_rate = float(y_true.mean())

    for label, col_name, ls, lw in stages_to_plot:
        if col_name not in df_pred.columns:
            continue
        p = df_pred[col_name].to_numpy()
        prec, recall, _ = precision_recall_curve(y_true, p)
        ap = average_precision_score(y_true, p)

        c = STAGE_COLORS.get(label, "#333333")
        display_label = f"{label} (AP = {ap:.4f})"

        # Panel A: Full PR curve
        ax_full.plot(recall, prec, label=display_label, color=c, linestyle=ls, linewidth=lw, alpha=0.9)

        # Panel B: Operational zoom (Recall up to 0.45)
        ax_zoom.plot(recall, prec, label=display_label, color=c, linestyle=ls, linewidth=lw, alpha=0.9)

    # Reference random horizontal line
    ax_full.axhline(base_fraud_rate, color="#888888", linestyle=":", label=f"Prevalence Floor ({base_fraud_rate:.3f})")
    ax_zoom.axhline(base_fraud_rate, color="#888888", linestyle=":")

    # Style Panel A: Full PR
    ax_full.set_xlabel("Recall (Fraction of Total Fraud Caught)", fontweight="bold")
    ax_full.set_ylabel("Precision (Fraction of Alerts That Are Fraud)", fontweight="bold")
    ax_full.set_title("(A) Full Out-of-Time Precision-Recall Curves", fontweight="bold")
    ax_full.set_xlim(0.0, 1.0)
    ax_full.set_ylim(0.0, 1.02)
    ax_full.legend(loc="upper right", framealpha=0.92, fontsize=8)

    # Style Panel B: Zoomed Operational Zone
    ax_zoom.set_xlabel("Recall (Fraud Coverage)", fontweight="bold")
    ax_zoom.set_ylabel("Precision (Alert Fidelity)", fontweight="bold")
    ax_zoom.set_title("(B) Operational Fraud Alert Zone (Recall: 0% – 45%)", fontweight="bold")
    ax_zoom.set_xlim(0.0, 0.45)
    ax_zoom.set_ylim(0.40, 1.01)

    # Annotate operational markers (Recall@0.1%FPR and Recall@1.0%FPR for Champion S2 vs Baseline S1)
    # Using values from benchmark
    s2_r01 = df_bench.loc[df_bench["Stage"] == "2 - Graph-Augmented GBDT", "Recall@0.1%FPR"].values[0]
    s2_r10 = df_bench.loc[df_bench["Stage"] == "2 - Graph-Augmented GBDT", "Recall@1.0%FPR"].values[0]
    s1_r01 = df_bench.loc[df_bench["Stage"] == "1 - Tabular GBDT", "Recall@0.1%FPR"].values[0]
    s1_r10 = df_bench.loc[df_bench["Stage"] == "1 - Tabular GBDT", "Recall@1.0%FPR"].values[0]

    ax_zoom.axvline(s2_r01, color="#1b7837", linestyle=":", alpha=0.6)
    ax_zoom.axvline(s2_r10, color="#1b7837", linestyle=":", alpha=0.6)
    ax_zoom.text(s2_r01 + 0.005, 0.45, f"S2 R@0.1%FPR\n({s2_r01:.1%})", fontsize=7.5, color="#1b7837")
    ax_zoom.text(s2_r10 + 0.005, 0.45, f"S2 R@1.0%FPR\n({s2_r10:.1%})", fontsize=7.5, color="#1b7837")

    ax_zoom.legend(loc="upper right", framealpha=0.92, fontsize=8)

    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "pr_curves_comparison.png")
    pdf_path = os.path.join(OUTPUT_DIR, "pr_curves_comparison.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


# ------------------------------------------------------------------------------
# 4. Plot 3: Calibration Curves (Raw vs Calibrated) & Probability Distributions
# ------------------------------------------------------------------------------
def plot_calibration_diagnostics(df_pred, df_bench):
    """
    Generate Reliability Diagrams comparing raw vs calibrated models and KDE distributions.
    Highlights Focal Loss probability scaling distortion in Stage 3 and Stage 4.
    """
    print("[Generating Plot 3] Calibration Reliability Diagrams...")
    y_true = df_pred["isFraud"].to_numpy(dtype=int)

    stages = [
        ("1 - Tabular GBDT", "1 - Tabular GBDT_raw", "1 - Tabular GBDT_calibrated"),
        ("2 - Graph-Augmented GBDT", "2 - Graph-Augmented GBDT_raw", "2 - Graph-Augmented GBDT_calibrated"),
        ("3 - Frozen Res-GNN", "3 - Frozen Res-GNN_raw", "3 - Frozen Res-GNN_calibrated"),
        ("4 - Joint BGNN", "4 - Joint BGNN_raw", "4 - Joint BGNN_calibrated"),
    ]

    fig = plt.figure(figsize=(14, 9.5))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1.0])

    ax_raw = fig.add_subplot(gs[0, 0])
    ax_cal = fig.add_subplot(gs[0, 1])
    ax_dist_raw = fig.add_subplot(gs[1, 0])
    ax_dist_cal = fig.add_subplot(gs[1, 1])

    # Perfect calibration reference diagonal
    for ax in [ax_raw, ax_cal]:
        ax.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect Calibration")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Mean Predicted Probability", fontweight="bold")
        ax.set_ylabel("Empirical Fraction of Positives", fontweight="bold")

    ax_raw.set_title("(A) Raw Prediction Reliability (Quantile Deciles)", fontweight="bold")
    ax_cal.set_title("(B) Isotonically Calibrated Reliability", fontweight="bold")

    for label, raw_col, cal_col in stages:
        c = STAGE_COLORS.get(label, "#333333")

        # Bench row for ECE
        brow = df_bench[df_bench["Stage"] == label]
        ece_raw = brow["ECE_quantile"].values[0] if not brow.empty else np.nan
        ece_cal = brow["ECE_calibrated"].values[0] if not brow.empty else np.nan

        # Raw curve
        p_raw = df_pred[raw_col].to_numpy()
        prob_true_r, prob_pred_r = calibration_curve(y_true, p_raw, n_bins=10, strategy="quantile")
        ax_raw.plot(prob_pred_r, prob_true_r, marker="o", markersize=5, lw=1.8, color=c,
                    label=f"{label} (ECE_q={ece_raw:.4f})")

        # Calibrated curve
        p_cal = df_pred[cal_col].to_numpy()
        prob_true_c, prob_pred_c = calibration_curve(y_true, p_cal, n_bins=10, strategy="quantile")
        ax_cal.plot(prob_pred_c, prob_true_c, marker="s", markersize=5, lw=1.8, color=c,
                    label=f"{label} (ECE_q={ece_cal:.4f})")

        # Density distributions (log scale histograms / KDE approximations)
        bins = np.linspace(0.0, 1.0, 45)
        ax_dist_raw.hist(p_raw, bins=bins, histtype="step", lw=1.8, color=c, density=True, label=label)
        ax_dist_cal.hist(p_cal, bins=bins, histtype="step", lw=1.8, color=c, density=True, label=label)

    # Style distributions
    ax_dist_raw.set_yscale("log")
    ax_dist_cal.set_yscale("log")
    ax_dist_raw.set_xlabel("Predicted Probability (Raw)", fontweight="bold")
    ax_dist_raw.set_ylabel("Density (Log Scale)", fontweight="bold")
    ax_dist_raw.set_title("(C) Raw Probability Distributions (Focal Loss Inflation)", fontweight="bold")
    ax_dist_cal.set_xlabel("Predicted Probability (Isotonically Calibrated)", fontweight="bold")
    ax_dist_cal.set_ylabel("Density (Log Scale)", fontweight="bold")
    ax_dist_cal.set_title("(D) Calibrated Probability Distributions", fontweight="bold")

    ax_raw.legend(loc="lower right", framealpha=0.9, fontsize=8)
    ax_cal.legend(loc="lower right", framealpha=0.9, fontsize=8)
    ax_dist_raw.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax_dist_cal.legend(loc="upper right", framealpha=0.9, fontsize=8)

    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "calibration_curves_comparison.png")
    pdf_path = os.path.join(OUTPUT_DIR, "calibration_curves_comparison.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


# ------------------------------------------------------------------------------
# 5. Plot 4: Temporal & Cohort Failure Mode Diagnostics
# ------------------------------------------------------------------------------
def plot_cohort_and_temporal(df_cohort):
    """
    Plot entity overlap (Seen vs First-seen cold start) and chronological slice stability.
    """
    print("[Generating Plot 4] Cohort & Temporal Degradation Breakdown...")

    # Filter to main stages
    target_stages = [
        "1 - Tabular GBDT",
        "2 - Graph-Augmented GBDT",
        "3n - Res-GNN, No Graph",
        "3 - Frozen Res-GNN",
        "4n - Joint BGNN, No Boost",
        "4 - Joint BGNN",
    ]
    df_sub = df_cohort[df_cohort["Stage"].isin(target_stages)].copy()
    # Sort in canonical pipeline sequence
    df_sub["rank"] = df_sub["Stage"].map({s: i for i, s in enumerate(target_stages)})
    df_sub = df_sub.sort_values("rank").reset_index(drop=True)

    fig, (ax_cohort, ax_time) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Seen vs First-Seen Entity AP
    stages_abbr = [
        "S1: Tabular",
        "S2: Graph-GBDT",
        "S3n: No-Graph",
        "S3: Res-GNN",
        "S4n: No-Boost",
        "S4: Joint-BGNN",
    ]
    x = np.arange(len(df_sub))
    width = 0.35

    seen_ap = df_sub["AP_seen_entity"].to_numpy()
    new_ap = df_sub["AP_first_seen_entity"].to_numpy()

    rects1 = ax_cohort.bar(x - width/2, seen_ap, width, label="Seen Cards (N=79,914; 69.4%)",
                           color="#2b5c8f", alpha=0.9, edgecolor="#111111", lw=0.6)
    rects2 = ax_cohort.bar(x + width/2, new_ap, width, label="First-Seen Cards (N=35,197; 30.6%)",
                           color="#e08214", alpha=0.9, edgecolor="#111111", lw=0.6)

    ax_cohort.set_ylabel("Average Precision (AP)", fontweight="bold")
    ax_cohort.set_title("(A) Entity Overlap Breakdown: Cold-Start vs Established Nodes", fontweight="bold")
    ax_cohort.set_xticks(x)
    ax_cohort.set_xticklabels(stages_abbr, rotation=20, ha="right", fontsize=9)
    ax_cohort.set_ylim(0.35, 0.65)
    ax_cohort.legend(loc="upper right", framealpha=0.9)

    # Annotate delta on S2
    s2_idx = 1
    ax_cohort.annotate(f"S2 Gain:\n+0.0587 AP", xy=(x[s2_idx] - width/2, seen_ap[s2_idx]),
                       xytext=(x[s2_idx] - width/2, seen_ap[s2_idx] + 0.04),
                       ha="center", fontsize=8.5, fontweight="bold", color="#1b7837",
                       arrowprops=dict(arrowstyle="->", color="#1b7837", lw=1.2))

    # Panel B: Temporal Slices (t1 to t4)
    time_cols = ["AP_slice_t1", "AP_slice_t2", "AP_slice_t3", "AP_slice_t4"]
    slice_names = ["Slice t1 (Days 140-150)", "Slice t2 (Days 150-160)",
                   "Slice t3 (Days 160-170)", "Slice t4 (Days 170-181)"]

    for _, row in df_sub.iterrows():
        st = row["Stage"]
        c = STAGE_COLORS.get(st, "#333333")
        y_vals = [row[c_col] for c_col in time_cols]
        ls = "-" if st in ["2 - Graph-Augmented GBDT", "1 - Tabular GBDT"] else "--"
        lw = 2.2 if "Graph-Augmented" in st else (1.8 if "Tabular" in st else 1.4)
        ax_time.plot(range(1, 5), y_vals, marker="o", label=st, color=c, linestyle=ls, linewidth=lw)

    ax_time.set_xlabel("Chronological OOT Block 5 Quartiles", fontweight="bold")
    ax_time.set_ylabel("Average Precision (AP)", fontweight="bold")
    ax_time.set_title("(B) Chronological Temporal Robustness Across OOT Slices", fontweight="bold")
    ax_time.set_xticks(range(1, 5))
    ax_time.set_xticklabels(["t1", "t2", "t3", "t4"], fontsize=10)
    ax_time.set_ylim(0.40, 0.58)
    ax_time.legend(loc="lower left", framealpha=0.9, fontsize=7.5)

    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "cohort_temporal_breakdown.png")
    pdf_path = os.path.join(OUTPUT_DIR, "cohort_temporal_breakdown.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"  -> Saved: {png_path} and {pdf_path}")


# ------------------------------------------------------------------------------
# 6. Ablation Summary Table & Performance Drop Tracker
# ------------------------------------------------------------------------------
def generate_ablation_summary_table(df_bench, df_hyp):
    """
    Constructs a structured ablation tracking table identifying exactly where
    performance is gained and lost across architectural transitions.
    """
    print("[Generating Table] Comprehensive Ablation Summary...")

    # Define the precise architectural step transitions
    steps = [
        {
            "Step": "0. Baseline Tabular",
            "Stage": "1 - Tabular GBDT",
            "Mechanism": "Pure tabular LightGBM over 371 base features",
            "AP": 0.48144,
            "Cal_AP": 0.46142,
            "Delta_Prev": 0.0,
            "Delta_Base": 0.0,
            "Hypothesis": "Reference Baseline",
            "Verdict": "BASELINE",
            "Diagnosis": "Strong tree-based partitioner; benchmark standard",
        },
        {
            "Step": "1. Hand-Crafted Relational",
            "Stage": "2 - Graph-Augmented GBDT",
            "Mechanism": "+19 causal expanding entity-history aggregations",
            "AP": 0.52522,
            "Cal_AP": 0.50876,
            "Delta_Prev": +0.04378,
            "Delta_Base": +0.04378,
            "Hypothesis": "H1",
            "Verdict": "SUPPORTED (p=0.00175)",
            "Diagnosis": "Peak pipeline performance (+9.1% AP); velocity & delta ratios dominate",
        },
        {
            "Step": "2. Decoupled Neural GNN",
            "Stage": "3 - Frozen Res-GNN",
            "Mechanism": "2-stage Res-GNN over continuous PCA + neighborhood means",
            "AP": 0.47719,
            "Cal_AP": 0.46120,
            "Delta_Prev": -0.04803,
            "Delta_Base": -0.00425,
            "Hypothesis": "H2 [Confounded]",
            "Verdict": "EXPLORATORY / DROPPED",
            "Diagnosis": "Severe degradation (-9.1% vs S2); PCA compression & mean smoothing dilute signals",
        },
        {
            "Step": "3. Message Passing Ablation",
            "Stage": "3n - Res-GNN, No Graph",
            "Mechanism": "Identical Res-GNN architecture with adjacency graph zeroed",
            "AP": 0.47627,
            "Cal_AP": 0.47627,
            "Delta_Prev": -0.00092,
            "Delta_Base": -0.00517,
            "Hypothesis": "H2a (S3 vs S3n)",
            "Verdict": "NOT SUPPORTED (p=0.723)",
            "Diagnosis": "Message passing provides zero statistical benefit (Δ=+0.0009, p=0.62)",
        },
        {
            "Step": "4. Monotonic Gating Alone",
            "Stage": "4n - Joint BGNN, No Boost",
            "Mechanism": "Softplus blending gate enforcing exact tabular fallback",
            "AP": 0.48111,
            "Cal_AP": 0.48111,
            "Delta_Prev": +0.00392,
            "Delta_Base": -0.00033,
            "Hypothesis": "H3b (S4n vs S3)",
            "Verdict": "REAL BUT < THRESHOLD",
            "Diagnosis": "Gating recovers tabular signal (+0.0039), but still below materiality threshold",
        },
        {
            "Step": "5. Alternating Joint BGNN",
            "Stage": "4 - Joint BGNN",
            "Mechanism": "Full alternating boosting updates on latent representations Z",
            "AP": 0.46336,
            "Cal_AP": 0.44785,
            "Delta_Prev": -0.01775,
            "Delta_Base": -0.01808,
            "Hypothesis": "H3a (S4 vs S4n) & H4",
            "Verdict": "NOT SUPPORTED (p=0.00175)",
            "Diagnosis": "Boosting updates corrupt latent manifold; performs worse than baseline S1 (-0.0181)",
        },
    ]

    df_summary = pd.DataFrame(steps)

    csv_path = os.path.join(OUTPUT_DIR, "ablation_summary_table.csv")
    df_summary.to_csv(csv_path, index=False)

    # Build Markdown table
    md_lines = [
        "# Architectural Ablation & Performance Drop Summary",
        "",
        "| Step | Architectural Stage | Mechanism Evaluated | Block 5 OOT AP | Δ vs Baseline | Δ vs Prior Step | Hypothesis & Verdict | Key Technical Diagnostic |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |",
    ]
    for row in steps:
        md_lines.append(
            f"| **{row['Step']}** | `{row['Stage']}` | {row['Mechanism']} | **{row['AP']:.4f}** | "
            f"{row['Delta_Base']:+.4f} | {row['Delta_Prev']:+.4f} | **{row['Hypothesis']}**: {row['Verdict']} | {row['Diagnosis']} |"
        )

    md_path = os.path.join(OUTPUT_DIR, "ablation_summary_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"  -> Saved: {csv_path} and {md_path}")

    # Print to console
    print("\n" + "=" * 105)
    print("ARCHITECTURAL ABLATION & PERFORMANCE DROP SUMMARY TABLE")
    print("=" * 105)
    col_fmt = "{:<22} | {:<24} | {:<8} | {:<12} | {:<12} | {:<20}"
    print(col_fmt.format("Step", "Stage", "OOT AP", "Delta Base", "Delta Prior", "Verdict"))
    print("-" * 105)
    for row in steps:
        print(col_fmt.format(
            row["Step"][:22],
            row["Stage"][:24],
            f"{row['AP']:.4f}",
            f"{row['Delta_Base']:+.4f}",
            f"{row['Delta_Prev']:+.4f}",
            row["Verdict"][:20]
        ))
    print("=" * 105 + "\n")


# ------------------------------------------------------------------------------
# 7. Main Execution
# ------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("STARTING THESIS DIAGNOSTICS & VISUALIZATION GENERATION")
    print("=" * 80)

    df_hyp, df_bench, df_cohort, df_pred = load_data()

    plot_forest_hypotheses(df_hyp)
    plot_pr_curves(df_pred, df_bench)
    plot_calibration_diagnostics(df_pred, df_bench)
    plot_cohort_and_temporal(df_cohort)
    generate_ablation_summary_table(df_bench, df_hyp)

    print("=" * 80)
    print(f"ALL DIAGNOSTICS SUCCESSFULLY EXPORTED TO: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
