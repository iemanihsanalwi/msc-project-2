# MSc Project: Graph-Augmented Machine Learning for Fraud Detection

**Student:** Ieman Ihsan bin Mohammad Alwi (ID: 25005466)  
**Institution:** Universiti Teknologi PETRONAS (UTP)  
**Program:** MSc by Coursework in Computing / Applied Data Science  

---

## 📌 Project Overview
This repository contains the dissertation manuscript, research proposal, experimentation pipeline, and evaluation artifacts for the MSc project:
> **"Benchmarking Inductive Graph Representation Learning against Tabular Gradient Boosted Decision Trees for Financial Fraud Detection"**

The study investigates temporal data leakage, tabular GBDT vs. Graph Neural Network (GNN) inductive performance, joint GBDT-GNN optimization, and statistical validation across temporal split regimes using the IEEE-CIS Fraud Detection benchmark.

---

## 📂 Repository Structure

```
├── data/
│   ├── figures/               # High-resolution architectural, workflow, and statistical figures
│   │   └── diagnostics_output/# Calibration curves, ablation plots, forest plots, PR curves
│   └── literature/            # Reference metadata and survey documentation
│
├── docs/
│   ├── guidelines/            # UTP project guidelines and formatting standards
│   ├── manuscripts/           # Final dissertation and IEEE paper submissions (.docx & .pdf)
│   │   └── chapters/          # Individual markdown source files for chapters 1-5
│   └── proposal/              # Extended proposal, research plan, and defense slides
│
├── scripts/
│   ├── analysis/              # Forensic audits, statistical significance (FDR), hypothesis tests
│   ├── build/                 # Automated document generation, compilation, and figure generators
│   └── experiments/           # Benchmark pipeline stages (Jupyter notebooks, logs, run manifests)
│       ├── EDA/               # Exploratory data analysis & leakage audit
│       ├── Stage 1/           # Tabular LightGBM baseline with temporal validation
│       ├── Stage 2/           # Graph-augmented GBDT (structural graph embeddings)
│       ├── Stage 3/           # Frozen two-stage residual Graph Neural Network (Res-GNN)
│       ├── Stage 4/           # Jointly optimized Boost-then-Convolve / BGNN
│       └── Stage 5/           # Sealed Out-of-Time (OOT) evaluation & hypothesis testing
│
└── .gitignore                 # Excludes caches, virtual environments, and large raw prediction dumps
```

---

## 🚀 Key Experimental Findings
1. **Temporal Leakage Safeguard:** Strict chronological splitting and causal neighbor lookback window prevent out-of-distribution performance collapse.
2. **Tabular vs. Graph Synergy:** Structural graph representations complement tabular feature splits, showing statistically significant gains under rigorous FDR-corrected hypothesis tests.
3. **Out-of-Time Evaluation:** Stage 5 verifies inductive stability across unseen temporal blocks.

---

## 📄 License & Attribution
All research documentation and codebase produced for degree fulfillment at Universiti Teknologi PETRONAS.
