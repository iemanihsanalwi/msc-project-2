"""
Publication-Grade Academic Tables Data Module
Synchronized with exact verified empirical ground truth from CSV artifacts,
audit_step1_findings.json, and unified cost parameterization
(c_miss = 1.0, c_review = $5.00, c_friction = $15.00, c_residual = 0.0).
All col_widths sum to 145 mm to fit UTP A4 portrait margins cleanly (210 - 40 - 25 mm).
"""

# ==============================================================================
# CHAPTER 2 TABLES
# ==============================================================================
TABLE_2_1 = {
    "caption": "TABLE 2.1. Taxonomic Summary of Related Literature across Tabular, Graph, and Hybrid Learning Paradigms",
    "headers": [
        "Paradigm / Method",
        "Key Literature",
        "Handling of Tabular Structure",
        "Handling of Relational Graph",
        "Key Limitation in Streaming Production",
    ],
    "col_widths": [24, 24, 32, 32, 33],
    "rows": [
        [
            "Standalone Tabular GBDT",
            "Chen & Guestrin (2016); Ke et al. (2017)",
            "Optimal: non-linear histogram binning, native missingness routing",
            "None: ignores graph topology and entity connection structures",
            "Blind to multi-hop relational fraud patterns and syndicates",
        ],
        [
            "Aggregated Relational Features",
            "Bahnsen et al. (2016); Dal Pozzolo et al. (2018)",
            "Good: manual velocity counts and aggregations appended as features",
            "Manual: 1-hop heuristic feature engineering",
            "Limited receptive field; cannot capture continuous structural roles",
        ],
        [
            "Homogeneous GNNs",
            "Kipf & Welling (2017); Hamilton et al. (2017)",
            "Weak: converts features to continuous embeddings; flattens anomalies",
            "Optimal for homogeneous topologies; smooths over node features",
            "Suffers from oversmoothing; assumes homophily; high latency",
        ],
        [
            "Heterogeneous GNNs",
            "Wang et al. (2019); Dou et al. (2020)",
            "Moderate: projects heterogeneous nodes to common latent dimension",
            "Good: relation-aware message passing and attention mechanisms",
            "High memory overhead; prone to transductive look-ahead leakage",
        ],
        [
            "Iterative / BGNN Couplings",
            "Ivanov & Prokhorenkova (2021)",
            "Alternating gradient updates between trees and neural representations",
            "Joint: features updated via functional boosting on graph manifold",
            "High outer iteration overhead; sensitive to covariate drift",
        ],
    ],
}

# ==============================================================================
# CHAPTER 3 TABLES (5 TABLES: 3.1 - 3.5)
# ==============================================================================
TABLE_3_1 = {
    "caption": "TABLE 3.1. Architectural Specification of the Four-Stage Experimental Ladder and Ablation Arms",
    "headers": [
        "Stage Identifier",
        "Architectural Paradigm",
        "Feature Representation",
        "Model Engine & Optimizer",
        "Structural Feedback / Coupling",
    ],
    "col_widths": [21, 25, 30, 34, 35],
    "rows": [
        [
            "Stage 1 (S1)",
            "Standalone Tabular GBDT",
            "371 raw tabular features",
            "LightGBM 4.6 (256 leaves, eta=0.03)",
            "None (orthogonal tree partitions)",
        ],
        [
            "Stage 2 (S2)",
            "Graph-Augmented GBDT",
            "371 tabular + 19 relational",
            "LightGBM 4.6 (256 leaves, eta=0.03)",
            "Two-stage manual feature augmentation",
        ],
        [
            "Stage 3 (S3)",
            "Frozen Res-GNN Pipeline",
            "Coordinate matrix Z_0 on G_<t",
            "PyTorch 2.6 Res-GNN (L=1, dim=64, SELU, lr=0.003)",
            "Decoupled neural propagation over fixed Z_0",
        ],
        [
            "Stage 3n (S3n)",
            "Res-GNN Without Graph",
            "Coordinate matrix Z_0 (A = 0)",
            "PyTorch 2.6 MLP (dim=64, SELU, lr=0.003)",
            "Zero message passing (Identical Net ablation)",
        ],
        [
            "Stage 4n (S4n)",
            "Joint BGNN (No Boost)",
            "Z_0 on G_<t + blending gate",
            "Res-GNN (L=1, SELU, lr=0.003) + Softplus Gate",
            "Gated fallback to S1 logits (No boosting)",
        ],
        [
            "Stage 4 (S4)",
            "Rank-1 Projected BGNN",
            "Z_0 on G_<t + boosted R_i^m",
            "Alternating PyTorch (SELU, lr=0.003) + LightGBM (M=4, J=2)",
            "Rank-1 empirical covariance projection coupling",
        ],
    ],
}

TABLE_3_2 = {
    "caption": "TABLE 3.2. IEEE-CIS Benchmark Dataset Ingestion and Structural Feature Profile",
    "headers": [
        "Feature Group",
        "Source Table",
        "Raw Count",
        "Tabular Pipeline (S1/S2)",
        "Neural Pipeline (S3/S4 Z_0)",
        "Missingness Strategy",
        "Operational Domain Context",
    ],
    "col_widths": [20, 18, 11, 22, 24, 21, 29],
    "rows": [
        ["Transaction Amounts", "train_transaction", "1", "1 (TransactionAmt)", "2 (amt_scaled, log_amt)", "None (100% complete)", "Direct financial exposure and non-linear loss magnitude"],
        ["Timestamps (DT)", "train_transaction", "1", "1 (TransactionDT)", "None (Ordering only)", "None (100% complete)", "Continuous clearing seconds; chronological sequence ordering"],
        ["Product Code", "train_transaction", "1", "1 (ProductCD encoded)", "None (Categorical excluded)", "None (100% complete)", "Merchant product line and transaction risk segmentation"],
        ["Card Attributes", "train_transaction", "6", "6 (card1 - card6)", "None (Proxy graph entity)", "Categorical mode fill", "Cardholder account proxy, issuing bank, card brand/type"],
        ["Address Proxies", "train_transaction", "2", "2 (addr1, addr2)", "None (Proxy graph entity)", "Missing indicator + -1", "Billing region and country geographic coordinates"],
        ["Distance & Deltas", "train_transaction", "17", "17 (dist1-2, D1-15)", "17 (scaled continuous)", "Missing indicator + -1", "Timedeltas between transaction and prior cardholder events"],
        ["Velocity Counters", "train_transaction", "14", "14 (C1 - C14)", "13 (scaled continuous)", "Zero imputation", "Transaction frequency counts across entities and time windows"],
        ["Email Domains", "train_transaction", "2", "2 (P_email, R_email)", "None (Categorical token)", "Categorical token fill", "Purchaser and recipient communication domain risk profiles"],
        ["Match Flags", "train_transaction", "9", "9 (M1 - M9)", "None (Binary flags)", "Missing indicator + -1", "Address, name, and identity verification match indicators"],
        ["Vesta Engineered", "train_transaction", "339", "278 raw continuous V", "15 (PCA-15, 69.5% var)", "Median fill / PCA", "Rich behavioral fraud indicators compressed via Block PCA"],
        ["Identity & Device", "train_identity", "40", "40 (id_01-38, DevType/Info)", "None (Proxy graph entity)", "Categorical fill (75.58% missing join)", "Hardware fingerprint, OS, browser (joined on N=144,233 rows)"],
        ["GBDT Logit / Flag", "Upstream S1", "0", "0 (Not applicable)", "2 (S1 Logit + Avail Flag)", "Deterministic", "Out-of-fold risk logit from Stage 1 tabular model"],
        ["Total Pipeline", "Unified Pipeline", "432", "371 base (+19 relational)", "49 aligned continuous Z_0", "Audit-hardened", "Comprehensive multi-modal transaction representation"],
    ],
}

TABLE_3_3 = {
    "caption": "TABLE 3.3. Chronological 5-Block Dataset Partitioning Scheme and Embargo Windows",
    "headers": [
        "Block Partition",
        "Transaction Count (N)",
        "Day Range",
        "Timestamp Interval (s)",
        "Fraud Rate (%)",
        "Role in Experimental Design",
    ],
    "col_widths": [24, 21, 22, 32, 16, 30],
    "rows": [
        ["Block 1", "118,108", "Day 0.00 to 25.74", "[86,400, 2,310,631]", "2.39%", "Training Fold 1; Graph Initialization"],
        ["Block 2", "118,108", "Day 25.74 to 63.73", "[2,310,632, 5,592,704]", "4.01%", "Training Fold 2; Expanding History"],
        ["Block 3", "118,108", "Day 63.73 to 100.22", "[5,592,705, 8,745,408]", "3.75%", "Training Fold 3; Intermediate History"],
        ["Block 4", "118,108", "Day 100.22 to 140.12", "[8,745,409, 12,192,842]", "3.90%", "Validation Fold; Hyperparameter & Calibration"],
        ["1-Day Embargo", "2,997", "Day 140.12 to 141.12", "[12,192,900, 12,279,242]", "6.61%", "Zero-leakage temporal buffer (purged)"],
        ["Block 5 (Eval OOT)", "115,111", "Day 141.12 to 182.00", "[12,279,259, 15,811,131]", "3.41%", "Sealed Out-of-Time Production Test"],
    ],
}

TABLE_3_4 = {
    "caption": "TABLE 3.4. Topological Summary of the Directed Multigraph G_<t",
    "headers": [
        "Entity / Edge Type",
        "Notation",
        "Total Entities (Dev)",
        "Block 5 Eval / Total",
        "Cold-Start / Missing %",
        "Domain Meaning & Degree Structure",
    ],
    "col_widths": [26, 14, 21, 22, 24, 38],
    "rows": [
        ["Transaction Nodes", "V_tx", "472,432", "115,111 (118,108 raw)", "--", "Individual clearing events with monetary amounts"],
        ["Card Proxy Nodes", "V_card", "109,773", "79,914 seen / 35,197 new", "30.58% cold-start cards", "Synthetic card cluster (card1-card6 + addr1); median deg=4, max=357k"],
        ["Device Proxy Nodes", "V_dev", "178,294", "19,130 dev / 98,978 null", "83.80% missing-dev supernode", "Hardware key (DeviceInfo+id_30+id_31); distinct from 75.58% identity missingness"],
        ["Transaction-Card Edges", "E_card", "472,432", "115,111 eval edges", "K_card <= 10 (30d cap)", "Directed causal edge from historical card transactions"],
        ["Transaction-Device Edges", "E_dev", "119,088", "28,777 non-null eval", "K_dev <= 5 (30d cap)", "Directed causal edge from historical device transactions"],
    ],
}

TABLE_3_5 = {
    "caption": "TABLE 3.5. Engineered Backward-Rolling Relational Features in Stage 2",
    "headers": [
        "Feature Group",
        "Count",
        "Aggregation Windows",
        "Mathematical Formulation",
        "Target Signal",
    ],
    "col_widths": [27, 11, 27, 38, 42],
    "rows": [
        ["Velocity Counters", "6", "1-hour, 24-hour, 7-day", "count(tx in W)", "High-frequency card testing and bot attacks"],
        ["Monetary Running Averages", "4", "24-hour, 30-day", "mean(amt in W)", "Abnormal spending spikes relative to card history"],
        ["Monetary Variance / Std", "3", "7-day, 30-day", "std(amt in W)", "Erratic amount deviations from baseline"],
        ["Historical Presence Flags", "4", "Unlimited expanding", "count(entity in G_<t)", "Entity maturity and cold-start differentiation"],
        ["Cross-Entity Ratios", "2", "Instantaneous vs 30-day", "amt_i / (mean_amt + 1.0)", "Multiplicative transaction scale anomalies"],
    ],
}

# ==============================================================================
# CHAPTER 4 TABLES (6 TABLES: 4.1 - 4.6)
# ==============================================================================
TABLE_4_1 = {
    "caption": "TABLE 4.1. Master Evaluation Benchmark on Sealed Out-of-Time Block 5 Holdout (Post-Embargo N = 115,111, Fraud Prevalence = 3.413%)",
    "headers": [
        "Architectural Stage / Model Variant",
        "Raw AP",
        "Cal. AP",
        "ROC-AUC",
        "Rec@0.1%FPR",
        "Rec@1.0%FPR",
        "Prec@1.0%Cap",
        "ECE (Cal.)",
        "Queue Loss (1% Cap)",
        "Bayes Loss (Corr.)",
    ],
    "col_widths": [27, 10, 10, 11, 12, 12, 12, 11, 18, 22],
    "rows": [
        ["Reference Baselines:", "", "", "", "", "", "", "", "", ""],
        ["  0 - Random Prevalence Floor", "0.0342", "0.0342", "0.5016", "0.05%", "1.04%", "3.56%", "0.4675", "$601,295.41", "$1,724,812.74"],
        ["  0 - Amount-Only Baseline", "0.0359", "0.0359", "0.4788", "0.05%", "0.87%", "3.04%", "0.0253", "$545,460.15", "$462,727.75"],
        ["  0 - Relational-Only (19 Counters)", "0.3026", "0.3026", "0.8338", "8.14%", "25.66%", "59.43%", "0.0019", "$512,543.02", "$368,538.01"],
        ["Experimental Benchmarking Ladder:", "", "", "", "", "", "", "", "", ""],
        ["  1 - Standalone Tabular LightGBM", "0.4814", "0.4614", "0.8731", "24.51%", "39.83%", "88.36%", "0.0063", "$503,596.43", "$301,621.83 (Cal) [$307,201.37 Raw]"],
        ["  2 - Graph-Augmented GBDT (S2)", "0.5252", "0.5088", "0.9053", "24.97%", "43.90%", "89.14%", "0.0034", "$490,253.81 [Champ]", "$283,904.49 [Champ] [$282,936.56 Raw]"],
        ["  3 - Frozen Two-Stage Res-GNN (S3)", "0.4772", "0.4612", "0.8806", "23.52%", "39.96%", "87.32%", "0.0063", "$495,999.13", "$305,803.72 (Cal) [$587,411.49 Raw]"],
        ["  4 - Joint Alternating BGNN (S4)", "0.4634", "0.4478", "0.8756", "19.47%", "39.53%", "84.10%", "0.0077", "$496,737.84", "$312,534.70 (Cal) [$576,749.67 Raw]"],
        ["Structural Control Ablation Arms:", "", "", "", "", "", "", "", "", ""],
        ["  3n - Res-GNN, No Graph (Adj = 0)", "0.4763", "0.4726", "0.8830", "22.88%", "39.60%", "86.27%", "0.0052", "$498,764.30", "$307,424.67 (Cal) [$576,331.41 Raw]"],
        ["  4n - Joint BGNN, No Boost (Gate)", "0.4811", "0.4764", "0.8807", "23.01%", "39.98%", "87.14%", "0.0091", "$497,229.62", "$306,866.66 (Cal) [$596,312.55 Raw]"],
    ],
}

TABLE_4_2 = {
    "caption": "TABLE 4.2. Formal Directional Hypothesis Testing Results (1,000 Block Resamples, Benjamini–Hochberg FDR Control, m = 9)",
    "headers": [
        "Hypothesis Identifier & Tested Contrast",
        "Contrast Arms",
        "Delta AP / Cost",
        "95% Bootstrap CI",
        "p_raw",
        "p_adj (FDR)",
        "Reject H0?",
        "Confirmatory Verdict",
    ],
    "col_widths": [36, 15, 13, 21, 10, 12, 12, 26],
    "rows": [
        ["Core Pre-Specified Confirmatory Contrasts:", "", "", "", "", "", "", ""],
        ["  H1: Relational Features vs. Tabular GBDT", "S2 vs S1", "+0.0438", "[+0.0377, +0.0504]", "0.0010", "0.0045", "True", "SUPPORTED [Confirmed]"],
        ["  H2: Learned Structure vs. Hand-Crafted GBDT", "S3 vs S2", "-0.0480", "[-0.0549, -0.0415]", "1.0000", "1.0000", "False", "NOT SUPPORTED (Directional Null)"],
        ["  H3: Joint Alternating BGNN vs. Res-GNN", "S4 vs S3", "-0.0138", "[-0.0191, -0.0087]", "1.0000", "1.0000", "False", "NOT SUPPORTED (Directional Null)"],
        ["  H4: Full Joint BGNN vs. Standalone Tabular", "S4 vs S1", "-0.0181", "[-0.0257, -0.0117]", "1.0000", "1.0000", "False", "NOT SUPPORTED (Directional Null)"],
        ["Structural & Ablation Contrasts:", "", "", "", "", "", "", ""],
        ["  H2a: Message Passing vs. Identical Network", "S3 vs S3n", "+0.0009", "[-0.0028, +0.0046]", "0.3100", "0.6975", "False", "NOT SUPPORTED"],
        ["  H2b: Message Passing on Connected Nodes Only", "S3 vs S3n(con)", "+0.0003", "[-0.0032, +0.0037]", "0.4220", "0.7596", "False", "NOT SUPPORTED"],
        ["  H3a: Functional Boosting Displacement Impact", "S4 vs S4n", "-0.0178", "[-0.0218, -0.0136]", "1.0000", "1.0000", "False", "NOT SUPPORTED"],
        ["  H3b: Monotonic Blending Gate vs. Frozen S3", "S4n vs S3", "+0.0039", "[+0.0006, +0.0072]", "0.0100", "0.0300", "True", "REAL BUT < THRESHOLD"],
        ["Operational Queue Loss Hypothesis:", "", "", "", "", "", "", ""],
        ["  H3c: S2 vs S1 Queue Loss at 1% Cap", "S2 vs S1 (QL)", "+$13,342.62", "[+$3,870.88, +$22,682.03]", "0.0010", "0.0045", "True", "SUPPORTED [Confirmed]"],
        ["Secondary / Exploratory Cohort Sub-Test:", "", "", "", "", "", "", ""],
        ["  R1: Relational vs Tabular on Seen Cards Only", "S2 vs S1(seen)", "+0.0587", "[+0.0501, +0.0668]", "0.0010", "--", "True", "SUPPORTED [Exploratory]"],
    ],
}

TABLE_4_3 = {
    "caption": "TABLE 4.3. Stepwise Architectural Ablation Trajectory and Performance Delta Analysis on Block 5 OOT Holdout",
    "headers": [
        "Step",
        "Architectural Stage",
        "Mechanism Evaluated",
        "OOT Raw AP",
        "Delta vs Base (S1)",
        "Delta vs Prior Step (FDR p_adj)",
        "Primary Technical Diagnosis",
    ],
    "col_widths": [9, 25, 28, 13, 18, 22, 30],
    "rows": [
        ["0", "1 - Tabular GBDT Baseline", "Raw 371 audit tabular features", "0.4814", "Reference (0.0000)", "--", "Strong non-relational tree split baseline"],
        ["1", "2 - Graph-Augmented GBDT", "+19 backward rolling counters", "0.5252", "+0.0438 (p_adj=0.0045)", "+0.0438 vs S1 (H1 CONFIRMED)", "Champion; lowest Queue ($490.3K) & Bayes ($283.9K) loss"],
        ["2", "3n - Res-GNN, No Graph", "Continuous Z_0 MLP projection (A=0)", "0.4763", "-0.0051", "-0.0489 vs S2", "Continuous PCA-15 & smoothing bottleneck"],
        ["3", "3 - Frozen Res-GNN", "Continuous Z_0 on G_<t (L=1 SELU)", "0.4772", "-0.0042", "+0.0009 vs S3n (H2a p_adj=0.6975)", "Zero significant message passing gain"],
        ["4", "4n - Joint BGNN, No Boost", "Z_0 on G_<t + Softplus gate alone", "0.4811", "-0.0003", "+0.0039 vs S3 (H3b p_adj=0.0300)", "Gate recovers tabular prior (<0.005 threshold)"],
        ["5", "4 - Joint Alternating BGNN", "GNN + Rank-1 Boost (M=4, J=2)", "0.4634", "-0.0181 (p_adj=1.0000)", "-0.0178 vs S4n (H3a p_adj=1.0000)", "Latent coordinate shift degrades AP"],
    ],
}

TABLE_4_4 = {
    "caption": "TABLE 4.4. Top-20 Feature Gain Importance from Stage 2 Graph-Augmented GBDT",
    "headers": ["Rank", "Feature Identifier", "Mean Split Gain", "Is Relational?", "Algorithmic Role & Domain Mechanism"],
    "col_widths": [12, 34, 21, 18, 60],
        "rows": [
        ["1", "V258", "30,362.39", "False", "Vesta engineered payment verification feature"],
        ["2", "C1", "23,408.28", "False", "Aggregated transaction count attribute"],
        ["3", "P_emaildomain_seen_before", "22,058.14", "True", "Expanding historical count of purchaser email domain"],
        ["4", "card2_seen_before", "20,854.73", "True", "Historical observation count of card issuer identification number"],
        ["5", "card1_seen_before", "19,691.68", "True", "Historical observation count of card issuing bank proxy (card1 BIN)"],
        ["6", "C13", "19,625.80", "False", "Aggregated transaction velocity counter"],
        ["7", "C14", "19,230.97", "False", "Aggregated transaction frequency counter"],
        ["8", "addr1_seen_before", "16,528.27", "True", "Historical observation count of billing region / address zip"],
        ["9", "D2", "16,110.37", "False", "Temporal delta feature tracking days since previous transaction"],
        ["10", "uid_device_seen_before", "14,814.72", "True", "Historical observation frequency of device fingerprint"],
        ["11", "uid_card_amt_running_mean", "13,047.04", "True", "Card running historical transaction amount baseline"],
        ["12", "R_emaildomain_seen_before", "12,824.96", "True", "Expanding historical count of recipient email domain"],
        ["13", "V317", "12,614.38", "False", "Tabular monetary transfer volume feature"],
        ["14", "uid_device_amt_running_mean", "12,583.77", "True", "Device running historical transaction amount baseline"],
        ["15", "C12", "12,256.97", "False", "High-cardinality transaction frequency counter"],
        ["16", "C8", "11,833.56", "False", "Aggregated payment channel velocity counter"],
        ["17", "V189", "10,369.12", "False", "Vesta engineered identity verification score"],
        ["18", "D15", "10,098.45", "False", "Long-range temporal delta feature tracking profile age"],
        ["19", "uid_device_amt_ratio", "10,062.31", "True", "Non-linear ratio of current amount to device historical mean"],
        ["20", "uid_device_amt_diff", "9,828.15", "True", "Raw monetary deviation from device historical mean"],
    ],
}

TABLE_4_5 = {
    "caption": "TABLE 4.5. Stratified Performance Breakdown Across Entity Overlap Cohorts and Longitudinal Time Slices (Block 5 OOT)",
    "headers": [
        "Model Variant",
        "Seen Cards (N=79,914)",
        "Cold-Start Cards (N=35,197)",
        "Slice t1 (N=28,778)",
        "Slice t2 (N=28,778)",
        "Slice t3 (N=28,777)",
        "Slice t4 (N=28,778)",
    ],
    "col_widths": [31, 21, 21, 18, 18, 18, 18],
    "rows": [
        ["0 - Random Floor", "0.0352", "0.0322", "0.0323", "0.0309", "0.0337", "0.0408"],
        ["0 - Amount Only", "0.0364", "0.0359", "0.0321", "0.0319", "0.0347", "0.0465"],
        ["0 - Relational Only", "0.3229", "0.2468", "0.3070", "0.2918", "0.2907", "0.3233"],
        ["1 - Tabular GBDT (S1)", "0.4410", "0.5805", "0.4252", "0.4953", "0.4460", "0.5462"],
        ["2 - Graph-Augmented GBDT (S2)", "0.4997 [Champ]", "0.5879 [Champ]", "0.5187 [Champ]", "0.5545 [Champ]", "0.4763 [Champ]", "0.5517 [Champ]"],
        ["3n - Res-GNN, No Graph", "0.4351", "0.5754", "0.4138", "0.4891", "0.4493", "0.5411"],
        ["3 - Frozen Res-GNN (S3)", "0.4411", "0.5658", "0.4128", "0.4946", "0.4511", "0.5364"],
        ["4n - Joint BGNN, No Boost", "0.4395", "0.5724", "0.4223", "0.5009", "0.4519", "0.5370"],
        ["4 - Joint BGNN (S4)", "0.4280", "0.5439", "0.4185", "0.5068", "0.4129", "0.5112"],
    ],
}

TABLE_4_6 = {
    "caption": "TABLE 4.6. Economic Loss Comparison: Fixed 1.0% Queue Cap vs. Corrected Unconstrained Bayes Loss (tau_i* = 20.0 / (A_i + 15.0))",
    "headers": [
        "Architectural Stage",
        "Regime 1: Queue Loss",
        "Regime 1: Frauds",
        "Regime 2: Raw Bayes",
        "Regime 2: Cal Bayes",
        "Raw Alerts (%)",
        "Cal Alerts (%)",
        "Cal. ECE",
    ],
    "col_widths": [28, 19, 14, 18, 18, 18, 18, 12],
    "rows": [
        ["0 - Random Prevalence Floor", "$601,295.41", "41 / 3,929", "$1,724,812.74", "$1,724,812.74", "74.18% (85,395)", "74.18% (85,395)", "0.4675"],
        ["0 - Amount-Only Baseline", "$545,460.15", "35 / 3,929", "$462,727.75", "$462,727.75", "7.72% (8,882)", "7.72% (8,882)", "0.0253"],
        ["0 - Relational Counters Only", "$512,543.02", "684 / 3,929", "$368,538.01", "$368,538.01", "5.84% (6,725)", "5.84% (6,725)", "0.0019"],
        ["1 - Standalone Tabular LightGBM", "$503,596.43", "1,017 / 3,929", "$307,201.37", "$301,621.83", "4.40% (5,067)", "6.07% (6,982)", "0.0063"],
        ["2 - Graph-Augmented GBDT (S2)", "$490,253.81 [Champ]", "1,026 / 3,929", "$282,936.56 [Champ]", "$283,904.49 [Champ]", "5.26% (6,057)", "5.91% (6,808)", "0.0034"],
        ["3 - Frozen Two-Stage Res-GNN (S3)", "$495,999.13", "1,005 / 3,929", "$587,411.49", "$305,803.72", "23.59% (27,149)", "6.83% (7,860)", "0.0063"],
        ["3n - Res-GNN, No Graph (Adj = 0)", "$498,764.30", "993 / 3,929", "$576,331.41", "$307,424.67", "23.06% (26,550)", "6.93% (7,979)", "0.0052"],
        ["4n - Joint BGNN, No Boost (Gate)", "$497,229.62", "1,003 / 3,929", "$596,312.55", "$306,866.66", "24.29% (27,955)", "7.53% (8,671)", "0.0091"],
        ["4 - Joint Alternating BGNN (S4)", "$496,737.84", "968 / 3,929", "$576,749.67", "$312,534.70", "23.15% (26,650)", "7.10% (8,178)", "0.0077"],
    ],
}

# ==============================================================================
# CHAPTER 5 TABLES (2 TABLES: 5.1 - 5.2)
# ==============================================================================
TABLE_5_1 = {
    "caption": "TABLE 5.1. Epistemological Contrast Between Reported GNN Literature and Causal Fraud Reality",
    "headers": ["Research Dimension", "Prevailing GNN Fraud Literature", "This Dissertation (Time-Causal Realism)"],
    "col_widths": [31, 57, 57],
    "rows": [
        ["Graph Adjacency", "Static, Transductive (Wang et al., 2019; Dou et al., 2020)", "Time-Causal Directed Multigraph G_<t"],
        ["Future Information Flow", "Look-ahead leakage (Van Belle et al., 2023)", "Strictly score-before-insert G_i"],
        ["Data Partitioning", "Random / Stratified Cross-Validation", "Chronological Block 5 Sealed Holdout"],
        ["Supervision Policy", "Instantaneous Oracle Labels (Dal Pozzolo et al., 2018)", "Practical Block-Delayed Embargo"],
        ["Entity Overlap", "High Transductive Node Overlap (Weber et al., 2019)", "Open-World 30.6% Entity Turnover"],
        ["Evaluation Metric", "ROC-AUC / Macro F1 (Artificially High)", "Non-interpolated AP & Ultra-low FPR"],
        ["Economic Grounding", "Unweighted Cross-Entropy (Lin et al., 2017)", "Scenario-Weighted Financial Loss L_fin"],
        ["Reported Outcome", "Massive SOTA Uplift (Liu et al., 2021)", "Scoped Null Result (Delta_4-1 = -0.0181)"],
    ],
}

TABLE_5_2 = {
    "caption": "TABLE 5.2. Operational and Hardware Latency Comparison Across Evaluated Architectural Paradigms",
    "headers": ["Evaluation Metric / Dimension", "Stage 1: Tabular LightGBM", "Stage 2: Graph-Augmented", "Stage 3: Res-GNN", "Stage 4: Joint BGNN"],
    "col_widths": [37, 27, 27, 27, 27],
    "rows": [
        ["Out-of-Time AP (Block 5)", "0.4814", "0.5252 (Cal.: 0.5088)", "0.4772 (Cal.: 0.4612)", "0.4634 (Cal.: 0.4478)"],
        ["ROC-AUC", "0.8731", "0.9053", "0.8806", "0.8756"],
        ["Recall @ 1.0% FPR", "39.83%", "43.90%", "39.96%", "39.53%"],
        ["Queue Loss (1% Cap, c_res=0.0)", "$503,596.43", "$490,253.81 [Champion]", "$495,999.13", "$496,737.84"],
        ["Corrected Bayes Loss (Calibrated)", "$301,621.83", "$283,904.49 [Champion]", "$305,803.72", "$312,534.70"],
        ["Expected Calibration Error (ECE)", "0.0061 (Cal: 0.0063)", "0.0055 (Cal: 0.0034)", "0.0928 (Cal: 0.0063)", "0.0892 (Cal: 0.0077)"],
        ["Measured Single-Transaction Latency (Batch = 1):", "", "", "", ""],
        ["  - Median Latency (ms)", "0.095 ms", "0.123 ms", "0.387 ms (End-to-End)", "0.559 ms (End-to-End)"],
        ["  - 95th Percentile Latency (p95)", "0.160 ms", "0.181 ms", "0.672 ms", "1.023 ms"],
        ["  - 99th Percentile Latency (p99)", "0.308 ms", "0.386 ms", "0.839 ms", "1.385 ms"],
        ["Operational Clearing SLA Budget", "< 100 ms (Met: 0.16% budget)", "< 100 ms (Met: 0.18% budget)", "< 100 ms (Met: 0.67% budget)", "< 100 ms (Met: 1.02% budget)"],
        ["Hardware Footprint & Pipeline", "Standard commercial CPU", "CPU + In-memory Redis cache", "CPU/GPU + 1-hop subg.", "CPU/GPU + 4-tree disp"],
        ["Cold-Start Entity Resilience", "Native split routing", "Default feature paths", "Zero-degree fallback", "Softplus blending gate"],
        ["Architectural Verdict", "Robust, zero infra overhead", "Production Champion", "Decoupled null result", "High-complexity null"],
    ],
}

print("Loaded all 14 tables into tables_data with synchronized ground truth.")
