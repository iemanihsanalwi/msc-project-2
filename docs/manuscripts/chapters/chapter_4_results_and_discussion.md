# Chapter 4: Empirical Results and Forensic Discussion

## 4.1 Benchmark Results on the Sealed Out-of-Time Holdout

To evaluate the predictive performance and cost-efficiency of graph-tabular architectures under production conditions, all models were evaluated on the sealed chronological holdout split (Block 5, $N = 118,108$ total transactions; removing 2,997 24-hour embargo boundary rows yields $N = 115,111$ evaluated holdout transactions with 3,929 frauds, 3.413% prevalence). As established in Section 3.2, Block 5 is strictly out-of-time (OOT), representing the final 41.88 days (~42 days) of the 182-day transaction stream ($t \in [12,192,900, 15,811,131]$ seconds). No transaction, label, or entity identifier from Block 5 was exposed during feature construction, out-of-fold logit generation, hyperparameter tuning, or model training. Furthermore, dynamic neighborhood construction on the directed multigraph $G_{<t}$ enforced score-before-insert semantics with a 1-day clearing embargo and a 30-day bounded lookback horizon.

Table 4.1 presents the master empirical benchmarking results across the Four-Stage Experimental Ladder, alongside reference baselines and structural ablation arms. Results are reported across both threshold-independent discrimination metrics (Average Precision and ROC-AUC), ultra-low false-positive operational operating points ($\text{Recall@0.1\%FPR}$ and $\text{Recall@1.0\%FPR}$), precision at capacity, calibration error metrics (Expected Calibration Error and Brier Score), and realized scenario-weighted financial loss ($L_{\text{fin}}$).


+––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––+
| Table 4.1: Master Evaluation Benchmark on Sealed Out-of-Time Block 5 Holdout (N = 118,108, Fraud Prevalence = 3.499%)                                                                |
+––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––+
| Architectural Stage / Model Variant   | Raw AP  | Cal. AP | ROC-AUC | Rec@0.1%FPR | Rec@1.0%FPR | Prec@1.0%Cap | ECE (Raw) | ECE (Cal.) | Realized L_fin ($)|
+–––––––––––––––––––-+––––-+––––-+––––-+––––––-+––––––-+–––––––+–––––-+––––––+––––––––––+
| Reference Baselines:                  |         |         |         |             |             |              |           |            |                    |
|   0 - Random Prevalence Floor         | 0.0342  | 0.0342  | 0.5016  | 0.05%       | 1.04%       | 3.56%        | 0.4675    | 0.4675     | $2,522,593.92      |
|   0 - Amount-Only Baseline            | 0.0359  | 0.0359  | 0.4788  | 0.05%       | 0.87%       | 3.04%        | 0.0253    | 0.0253     | $536,640.69        |
|   0 - Relational-Only (19 Counters)   | 0.3026  | 0.3026  | 0.8338  | 8.14%       | 25.66%      | 59.43%       | 0.0019    | 0.0019     | $425,227.51        |
|                                       |         |         |         |             |             |              |           |            |                    |
| Experimental Ladder:                  |         |         |         |             |             |              |           |            |                    |
|   1 - Standalone Tabular LightGBM (S1)| 0.4814  | 0.4614  | 0.8731  | 24.51%      | 39.83%      | 88.36%       | 0.0061    | 0.0063     | $365,728.99        |
|   2 - Graph-Augmented GBDT (S2)       | 0.5252  | 0.5088  | 0.9053  | 24.97%      | 43.90%      | 89.14%       | 0.0055    | 0.0034     | $345,402.31 [CHAMP]|
|   3 - Frozen Two-Stage Res-GNN (S3)   | 0.4772  | 0.4612  | 0.8806  | 23.52%      | 39.96%      | 87.32%       | 0.0928    | 0.0063     | $634,205.97        |
|   4 - Joint Alternating BGNN (S4)     | 0.4634  | 0.4478  | 0.8756  | 19.47%      | 39.53%      | 84.10%       | 0.0892    | 0.0077     | $629,411.78        |
|                                       |         |         |         |             |             |              |           |            |                    |
| Structural Ablation Arms:             |         |         |         |             |             |              |           |            |                    |
|   3n - Res-GNN, No Graph (Adjacency=0)| 0.4763  | 0.4763  | 0.8830  | 22.88%      | 39.60%      | 86.27%       | 0.0896    | 0.0896     | $624,270.40        |
|   4n - Joint BGNN, No Boost (GateOnly)| 0.4811  | 0.4811  | 0.8807  | 23.01%      | 39.98%      | 87.14%       | 0.0953    | 0.0953     | $651,120.81        |
+––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––+


### 4.1.1 Analysis of Primary Ranking Benchmarks

The empirical results in Table 4.1 reveal critical divergences between traditional machine learning assumptions and time-causal financial fraud realities:

1. Performance of Tree-Based Tabular Learning: The standalone tabular LightGBM baseline (Stage 1) achieves an uncalibrated Average Precision of $\text{AP} = 0.4814$ (calibrated $\text{AP} = 0.4614$) and $\text{ROC-AUC} = 0.8731$. Operating strictly on the 371 base transaction and identity features without graph information, Stage 1 detects 24.51% of all fraud attempts at a 0.1% false-positive rate, and captures 39.83% of all fraud at a 1.0% false-positive rate. This demonstrates that gradient-boosted decision trees, utilizing histogram binning, missing value branch routing, and non-linear interactions across card, device, and behavioral delta features ($D_1\text{–}D_{15}$, $C_1\text{–}C_{14}$), construct a strong statistical baseline.

2. Performance of Graph-Augmented GBDT (Stage 2): When the tabular feature matrix is augmented with 19 backward-rolling, time-causal relational features, Stage 2 achieves the highest performance across the benchmark, with an uncalibrated Average Precision of $\mathbf{0.5252}$ (calibrated $\mathbf{0.5088}$) and $\mathbf{0.9053}$ ROC-AUC. Stage 2 delivers a statistically significant uplift of $\Delta_{2-1} = +0.0438$ AP over Stage 1 ($p_{\text{adj}} = 0.00175$, confirming Hypothesis $H_1$). Furthermore, Stage 2 attains the lowest realized scenario-weighted financial loss across all evaluated systems at \$345,402.31, representing a financial savings of \$20,326.68 relative to Stage 1 (and \$348,690.01 post-calibration). Under post-hoc probability calibration, Stage 2 also achieves the highest precision at the 1.0% operational review capacity (89.14%) and Recall@1.0%FPR of 43.90%. This shows that explicit, backward-looking relational feature engineering captures fraud patterns without introducing optimization instabilities.

3. Decoupling in Frozen Res-GNN (Stage 3): Replacing explicit relational features with a continuous, two-stage residual Graph Neural Network on $G_{<t}$ results in a lower Average Precision of $\mathbf{0.4772}$ (calibrated $\text{AP} = 0.4612$, $\text{ROC-AUC} = 0.8806$), representing a decrease of $-0.0480$ AP compared to Stage 2 ($p = 0.001$, evaluating Hypothesis $H_2$). While its Recall@1.0%FPR remains at 39.96%, its uncalibrated financial loss increases to \$634,205.97 (calibrated: \$366,820.96) due to probability distortion under Focal Loss.

4. Functional Coupling in Joint BGNN (Stage 4): The alternating jointly optimized BGNN framework does not improve upon the frozen Res-GNN, achieving an Average Precision of $\mathbf{0.4634}$ (calibrated $\text{AP} = 0.4478$, $\Delta_{4-3} = -0.0138$ AP, $\text{ROC-AUC} = 0.8756$), which does not support Hypothesis $H_3$. Benchmarked against the macro system baseline, Stage 4 falls below standalone Tabular LightGBM ($\Delta_{4-1} = -0.0181$ AP, rejecting Hypothesis $H_4$). Under raw uncalibrated probabilities, Stage 4 produces a financial loss of \$629,411.78 (calibrated: \$373,410.83), driven by probability miscalibration under Focal Loss ($ECE = 0.0892$), which raises the model's raw alert rate to 13.46% and exceeds operational review capacity.

====================================================================================================
Figure 4.1: Empirical Precision-Recall Curves across Stages 1–4 and Reference Baselines on Sealed Block 5 Holdout
====================================================================================================



## 4.2 Pre-Specified Hypothesis Testing Verdicts

To prevent post-hoc rationalization, data snooping, and selective metric reporting, all architectural comparisons were bound to formal, pre-specified statistical hypotheses evaluated under a rigorous inferential protocol. Significance was determined using a **paired, chronological block bootstrap with 1,000 resamples** ($N_{\text{eval}} = 115,111$). Multiplicity across the 8 pre-specified structural hypotheses was controlled via the **Benjamini–Hochberg False Discovery Rate (FDR)** procedure at nominal level $\alpha = 0.05$. A hypothesis is deemed supported if and only if its lower confidence bound at 95% coverage exceeds the pre-specified materiality boundary: $\text{LCB}_{0.95}(\Delta\text{AP}) > 0$ with an observed effect size meeting the pre-specified threshold ($\Delta \ge +0.005$ AP).

Table 4.2 summarizes the formal hypothesis testing battery, reporting point estimates, empirical 95% bootstrap confidence intervals, raw asymptotic $p$-values, FDR-adjusted $p$-values ($p_{\text{adj}}$), and final confirmatory verdicts.


+––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––+
| Table 4.2: Formal Paired Bootstrap Hypothesis Testing Results (1,000 Block Resamples, Benjamini–Hochberg FDR Control)                               |
+–––––––––––––––––––––––-+–––––––-+–––––+––––––––––+––––-+––––-+––––––+–––––––––––+
| Hypothesis & Contrast Description             | Contrast Code | Delta AP | 95% Bootstrap CI   | p_raw   | p_adj   | FDR Reject | Confirmatory Verdict |
+–––––––––––––––––––––––-+–––––––-+–––––+––––––––––+––––-+––––-+––––––+–––––––––––+
| Core Pre-Specified Confirmatory Hypotheses:  |               |          |                    |         |         |            |                      |
|   H1: Relational Features vs. Tabular Baseline| S2 vs S1      | +0.0438  | [+0.0377, +0.0504] | 0.001   | 0.00175 | True       | SUPPORTED [Confirmed]|
|   H2: Learned Structure vs. Hand-Crafted GBDT | S3 vs S2      | -0.0480  | [-0.0549, -0.0415] | 0.001   | –      | False      | EXPLORATORY [Confound]
|   H3: Joint Alternating BGNN vs. Frozen ResGNN| S4 vs S3      | -0.0138  | [-0.0191, -0.0087] | 0.001   | 0.00175 | True       | NOT SUPPORTED (Null) |
|   H4: Macro Joint System vs. Tabular Baseline | S4 vs S1      | -0.0181  | [-0.0257, -0.0117] | 0.001   | 0.00175 | True       | NOT SUPPORTED (Null) |
|                                               |               |          |                    |         |         |            |                      |
| Fine-Grained Structural & Ablation Contrasts: |               |          |                    |         |         |            |                      |
|   H2a: Message Passing vs. Identical Network  | S3 vs S3n     | +0.0009  | [-0.0028, +0.0046] | 0.620   | 0.7233  | False      | NOT SUPPORTED (Null) |
|   H2b: Message Passing on Connected Nodes Only| S3 vs S3n(con)| +0.0003  | [-0.0032, +0.0037] | 0.844   | 0.8440  | False      | NOT SUPPORTED (Null) |
|   H3a: Functional Boosting Displacement Impact| S4 vs S4n     | -0.0178  | [-0.0218, -0.0136] | 0.001   | 0.00175 | True       | NOT SUPPORTED (Null) |
|   H3b: Monotonic Gating Alone vs. Frozen S3   | S4n vs S3     | +0.0039  | [+0.0006, +0.0072] | 0.020   | 0.0280  | True       | REAL BUT < THRESHOLD |
|                                               |               |          |                    |         |         |            |                      |
| Secondary / Exploratory Cohort Sub-Tests:     |               |          |                    |         |         |            |                      |
|   R1: Relational vs Tabular on Seen Cards Only| S2 vs S1(seen)| +0.0587  | [+0.0501, +0.0668] | 0.001   | –      | True       | SUPPORTED [Explor.]  |
+––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––+


====================================================================================================
Figure 4.2: Publication-Grade Forest Plot of Pre-Specified Hypotheses with 95% Bootstrap Confidence Intervals
====================================================================================================


### 4.2.1 Detailed Evaluation of Hypotheses

#### 1. Hypothesis 1 ($H_1$ / RQ1): Handcrafted Relational Augmentation
* **Formal Hypothesis:** Augmenting tabular GBDT with 19 backward-rolling causal graph features yields superior predictive discrimination over the tabular baseline: $\Delta_{2-1} = \text{AP}_2 - \text{AP}_1 \ge +0.005$.
* **Statistical Finding:** In uncalibrated raw ranking, $\Delta_{2-1} = \mathbf{+0.0438}$ AP with a 95% bootstrap confidence interval of $[+0.0377, +0.0504]$ ($one-sided p > 0.999, p_{\text{adj}} > 0.999$). Because the lower confidence bound ($+0.0377$) strictly exceeds the $+0.005$ materiality threshold, $H_1$ is SUPPORTED [Confirmed].
* **Critical Contextual Resolution:** This statistical verdict is further reinforced by operational and cohort metrics:
  1. Under post-hoc isotonic probability calibration, Stage 2 maintains its superiority ($\text{AP}_{\text{cal}} = 0.5088$ versus Stage 1's $\text{AP}_{\text{cal}} = 0.4614$, $\Delta = +0.0474$ AP).
  2. In financial terms, Stage 2 achieves the lowest dollar loss ($L_{\text{fin}} = \$345,402.31$), outperforming Stage 1 by **\$20,326.68** due to higher precision at the operational triage threshold ($89.14\%$ vs $88.36\%$) and higher Recall@1.0%FPR ($43.90\%$ vs $39.83\%$).
  3. When isolated to established card entities with historical transactions ($R_1$), relational features deliver an even larger uplift of **$+0.0587$ AP** ($95\%\text{ CI: } [+0.0501, +0.0668], p < 0.001$).

#### 2. Hypothesis 2 ($H_2$ / RQ2): Structural Decoupling Impact
* **Formal Hypothesis:** Learned structural graph representations via continuous message passing outperform handcrafted relational features: $\Delta_{3-2} = \text{AP}_3 - \text{AP}_2 \ge +0.005$.
* **Statistical Finding:** $\Delta_{3-2} = \mathbf{-0.0480}$ AP ($95\%\text{ CI: } [-0.0549, -0.0415], p = 0.001$). The hypothesis is **NOT SUPPORTED (Directional Null)**.
* **Methodological Implication:** Replacing engineered relational ratios with a continuous two-stage Res-GNN results in a 4.8 percentage point drop in Average Precision. Decoupling tabular feature extraction from spatial aggregation creates an information bottleneck, blunting sharp ratio features.

#### 3. Hypothesis 2a & 2b ($H_{2\text{a}}, H_{2\text{b}}$): Message-Passing Value Isolation
* **Formal Hypotheses:** $H_{2\text{a}}$ posits that multi-hop spatial message passing contributes positive predictive signal over an identical neural architecture with zeroed adjacency: $\Delta_{3-3\text{n}} \ge +0.005$. $H_{2\text{b}}$ restricts this test exclusively to connected nodes ($N_{\text{conn}} = 114,315$).
* **Statistical Finding:** 
  * Full OOT Holdout ($H_{2\text{a}}$): $\Delta_{3-3\text{n}} = \mathbf{+0.0009}$ AP (95% CI: [-0.0028, +0.0046], p = 0.620, p_adj = 0.7233).
  * Connected Nodes Only ($H_{2\text{b}}$): $\Delta_{3-3\text{n}} = \mathbf{+0.0003}$ AP (95% CI: [-0.0032, +0.0037], p = 0.844, p_adj = 0.8440).
* **Scientific Verdict:** Both hypotheses are **NOT SUPPORTED**. The empirical effect size is indistinguishable from zero noise. Spatial convolutions over $G_{<t}$ contribute zero statistically meaningful predictive information beyond the non-relational continuous MLP projection of $Z_0$.

#### 4. Hypothesis 3 ($H_3$ / RQ3): Primary Functional Coupling Uplift
* **Formal Hypothesis:** Joint alternating optimization via functional-gradient boosting updates significantly improves predictive performance over the frozen two-stage architecture: $\Delta_{4-3} = \text{AP}_4 - \text{AP}_3 \ge +0.005$ with $\text{LCB}_{0.95} > 0$.
* **Statistical Finding:** $\Delta_{4-3} = \mathbf{-0.0138}$ AP ($95\%\text{ CI: } [-0.0191, -0.0087], one-sided p > 0.999, p_{\text{adj}} > 0.999$).
* **Confirmatory Verdict:** **NOT SUPPORTED (Null Result)**. Shifting continuous coordinates via functional displacement updates alters the underlying latent manifold, producing negative transfer and reducing performance relative to the frozen Res-GNN.

#### 5. Hypothesis 4 ($H_4$ / RQ1): Macro System Deployment
* **Formal Hypothesis:** The end-to-end alternating joint graph–tabular system outperforms the standalone tabular GBDT baseline: $\Delta_{4-1} = \text{AP}_4 - \text{AP}_1 \ge +0.005$.
* **Statistical Finding:** $\Delta_{4-1} = \mathbf{-0.0181}$ AP ($95\%\text{ CI: } [-0.0257, -0.0117], one-sided p > 0.999, p_{\text{adj}} > 0.999$).
* **Confirmatory Verdict:** NOT SUPPORTED (Empirical Null Result). The macro deep graph system fails to outperform standalone tabular LightGBM, falling $0.0181$ AP points below the non-relational tree baseline.

## 4.3 Architectural Ablation Studies (RQ5)

To systematically deconstruct the performance trajectory across the experimental ladder, an end-to-end ablation study was executed. Table 4.3 maps the stepwise progression from Stage 2 through Stage 4, isolating the precise contribution of each architectural component: spatial convolutions, the monotonic blending gate, and functional coordinate displacement updates.


+––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––+
| Table 4.3: Stepwise Architectural Ablation Trajectory and Performance Delta Analysis on Block 5 OOT Holdout                                        |
+–––+–––––––––––––-+––––––––––––––––+––––-+––––––––––+––––––––––-+–––––––––––––––-+
| Step | Architectural Stage       | Mechanism Evaluated            | OOT AP  | Delta vs Base (S1) | Delta vs Prior Step | Primary Technical Diagnosis   |
+–––+–––––––––––––-+––––––––––––––––+––––-+––––––––––+––––––––––-+–––––––––––––––-+
| 0    | 1 - Tabular GBDT Baseline | Raw 371 audit tabular features | 0.4814  | Reference (0.0000) | –                  | Optimal tree split baseline   |
| 1    | 2 - Graph-Augmented GBDT  | +19 Backward rolling counters  | 0.5252  | +0.0438            | +0.0438             | Champion; lowest loss ($345k) |
| 2    | 3 - Frozen Res-GNN        | Continuous Z_0 on G_<t         | 0.4772  | -0.0042            | -0.0480 (vs S2)     | Continuous embedding bottle   |
| 3    | 3n - Res-GNN, No Graph    | Adjacency zeroed (MLP control) | 0.4763  | -0.0051            | -0.0009 (vs S3)     | Zero signal in message passing|
| 4    | 4n - Joint BGNN, No Boost | Monotonic blending gate alone  | 0.4811  | -0.0003            | +0.0039 (vs S3)     | Marginal tabular recovery     |
| 5    | 4 - Joint Alternating BGNN| GNN + Boosted Displacements    | 0.4634  | -0.0181            | -0.0177 (vs S4n)    | Manifold mutation degradation |
+–––+–––––––––––––-+––––––––––––––––+––––-+––––––––––+––––––––––-+–––––––––––––––-+


### 4.3.1 Forensic Failure Mode 1: The Multiplicative Ratio Advantage vs. Additive Linear Smoothing

The primary factor in the performance difference between Stage 2 and Stage 3 ($\Delta_{3-2} = -0.0480$ AP) lies in how fraud anomalies are represented:

In financial clearing systems, fraudulent transactions typically present as multiplicative deviations from an entity's established historical baseline. In Stage 2, domain knowledge was explicitly encoded as non-linear, scale-invariant interaction ratios and temporal deltas:
$$\mathrm{amt\_ratio}_i = \frac{x_{\text{amt}, i}}{\bar{x}_{\text{amt}, \text{hist}(i)} + 1.0}, \qquad \Delta t_i = t_i - t_{\text{prev}, i}$$
Because decision trees perform recursive, axis-aligned partition splits, a GBDT requires exactly one split decision (e.g., $\mathrm{amt\_ratio}_i > 4.5 \land \Delta t_i < 120\text{s}$) to isolate high-velocity credential stuffing or card-testing bursts.

In contrast, the Graph Neural Network in Stages 3 and 4 aggregates historical neighbor states via unweighted arithmetic mean pooling:
$$\bar{\mathbf{z}}_{\mathcal{N}(i)} = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \mathbf{z}_j$$
Arithmetic mean pooling acts as an **isotropic low-pass filter**, smoothing away the extreme point anomalies that define fraudulent activity. Furthermore, feed-forward neural layers operate via affine transformations:
$$\mathbf{h}_i = \text{Activation}(\mathbf{W}_1 \mathbf{z}_i + \mathbf{W}_2 \bar{\mathbf{z}}_{\mathcal{N}(i)} + \mathbf{b})$$
Linear layers cannot compute non-linear division, scale-invariant normalization, or multiplicative ratios without specialized product layers or unbounded polynomial parameterization. Consequently, spatial aggregation diluted the sharp anomaly signals that the tree models exploited.

This forensic deduction is directly corroborated by the empirical feature gain importance analysis from Stage 2. As detailed in Table 4.4, backward-rolling relational features captured **8 of the top 20 split gain positions (accounting for 21.84% of top-20 gain, and 29.79% across all 19 relational features in the full model)**, accounting for **$29.79\%$ of total model gain**. Replacing these explicit non-linear ratios with latent neural embeddings discarded the most potent predictive signal in the dataset.


+––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––+
| Table 4.4: Top-20 Feature Gain Importance from Stage 2 Graph-Augmented GBDT                                                                        |
+––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––+
| Rank | Feature Identifier               | Mean Split Gain | Is Relational? | Algorithmic Role & Domain Mechanism                               |
+–––+–––––––––––––––––+––––––––-+––––––––+–––––––––––––––––––––––––––––––––-+
| 1    | V258                             | 36,560.70       | False          | Vesta engineered payment verification feature                     |
| 2    | C1                               | 16,822.54       | False          | Aggregated transaction count feature                              |
| 3    | P_emaildomain_seen_before        | 12,245.07       | True           | Expanding historical count of purchaser email domain              |
| 4    | card2_seen_before                | 13,154.16       | True           | Historical frequency of issuing bank identification number        |
| 5    | card1_seen_before                | 10,501.06       | True           | Historical frequency of specific cardholder account number        |
| 6    | C13                              | 24,322.52       | False          | Aggregated transaction velocity counter                           |
| 7    | C14                              | 22,753.22       | False          | Aggregated transaction frequency counter                          |
| 8    | addr1_seen_before                | 10,001.29       | True           | Historical billing address observation count                      |
| 9    | D2                               | 16,254.22       | False          | Temporal delta feature tracking days since previous transaction   |
| 10   | uid_device_seen_before           | 7,990.12        | True           | Historical device fingerprint observation frequency               |
| 11   | uid_card_amt_running_mean        | 7,881.85        | True           | Card running historical transaction amount baseline               |
| 12   | R_emaildomain_seen_before        | 6,718.10        | True           | Historical recipient email domain observation count               |
| 13   | V317                             | 8,245.47        | False          | Tabular monetary transfer volume feature                          |
| 14   | uid_device_amt_running_mean      | 8,370.64        | True           | Device running historical transaction amount baseline             |
| 15   | C12                              | 12,196.17       | False          | High-cardinality count feature                                     |
| 16   | C8                               | 11,365.91       | False          | Aggregated counter feature                                        |
| 17   | V189                             | 11,717.83       | False          | Tabular identity verification score                               |
| 18   | D15                              | 13,076.42       | False          | Transaction timedelta feature                                     |
| 19   | uid_device_amt_ratio             | 6,952.07        | True           | Non-linear ratio of current amount to device historical mean      |
| 20   | uid_device_amt_diff              | 6,225.23        | True           | Raw monetary deviation from device historical mean                 |
+––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––+


### 4.3.2 Forensic Failure Mode 2: The Dimensional Compression Bottleneck

To construct a numerically tractable input matrix $\mathbf{Z}_0 \in \mathbb{R}^{N \times 49}$ for the PyTorch neural network, the high-dimensional tabular space had to be compressed. Specifically, the 339 raw continuous Vesta features ($V_1\text{–}V_{339}$) were projected down to 15 principal components via time-causal Block PCA fitted strictly on training data.

While PCA retained 69.5% of cumulative variance (approx. 70% across folds) across the $V$-columns, PCA is a purely linear projection that maximizes global variance across the entire transaction volume. In payment clearing streams, $96.5\%$ of all transactions are legitimate. Consequently, principal components align almost exclusively with the dominant variance axes of normal consumer shopping behavior. The subtle, non-linear subspace correlations that characterize fraudulent transactions ($<3.5\%$ prevalence) were projected into low-variance residual components and discarded. Decision trees in Stage 1 and Stage 2 split directly on raw, uncompressed $V$-features (such as $V_{258}$ and $V_{317}$ in Table 4.4), entirely bypassing this compression bottleneck.

### 4.3.3 Forensic Failure Mode 3: Latent Manifold Mutation vs. Pure Boosting

The ablation contrast between Stage 4 and Stage 4n ($H_{3\text{a}}$, Table 4.2) confirmed a substantial raw uplift of $\Delta = -0.0178$ AP resulting from functional displacement updates. However, analyzing the optimization dynamics reveals why Stage 4 failed to achieve parity with Stage 1:

In canonical Boost then Convolve (Ivanov & Prokhorenkova, 2021), functional gradient boosting trains decision trees directly on the multi-dimensional feature gradients of the GNN loss ($-\nabla_{\mathbf{z}_i} \mathcal{L}$), updating the continuous node representations $\mathbf{Z}_0$ that serve as GNN inputs. In our adapted streaming formulation, updating the input coordinates via a rank-1 empirical covariance projection:
$$\mathbf{Z}_0^{(m)} = \mathbf{Z}_0^{(m-1)} + \eta_{\text{boost}} \cdot \text{clip}_{\delta_R}\left( T_m(\mathbf{x}_i) \cdot \mathbf{d}, -\delta_R, \delta_R \right)$$
modifies the continuous coordinate manifold between outer boosting rounds. In subsequent inner rounds, AdamW resumes gradient optimization on an altered feature manifold. On heterogeneous financial tabular manifolds where decision boundaries are discontinuous and class imbalance is severe, mutating the underlying feature coordinates destabilizes neural optimization, inducing negative transfer.

Furthermore, empirical analysis of the trained monotonic blending gate:
$$\beta_i = \sigma\left( \text{softplus}(w_{\text{card}}) \cdot \mathrm{card\_mask}_i + \text{softplus}(w_{\text{dev}}) \cdot \mathrm{dev\_mask}_i + b_{\text{gate}} \right) \cdot \mathbb{I}(\text{connected}_i)$$
revealed that for connected transactions ($N = 114,315$, 99.31% of the holdout), the gate weights converged to an average value of:
$$\bar{\beta} = \mathbf{0.8726}$$
This reflects a learned preference that allocated 87.26% weight to the neural GNN logit and only 12.74% weight to the Stage 1 tabular logit on connected nodes. Because the neural representation was degraded by low-pass smoothing, heavily weighting the neural head diluted the high-fidelity tabular predictions.


## 4.4 Cohort Robustness, Entity Cold Start, and Temporal Drift (RQ4)

Real-world fraud detection pipelines operate in open-world environments characterized by severe entity turnover, asynchronous registration, and non-stationary behavioral drift. To evaluate robustness under these real-world pressures, performance was stratified across entity overlap cohorts and longitudinal time quartiles.

### 4.4.1 The 30.6% Open-World Cold-Start Dilemma

In the evaluated Block 5 holdout ($N = 115,111$), **35,197 transactions (30.6%) involved payment cards that had never appeared in the historical training continuum** (cold-start card prevalence: 3.15%), whereas established cards comprised 79,914 transactions (fraud prevalence: 3.53%). It is vital to distinguish these training-unseen cards from strictly isolated graph nodes: only 796 transactions (0.69%) were completely disconnected from both card and device proxy edges on $G_{<t}$ (exhibiting an elevated fraud rate of 6.66%), while 114,315 transactions (99.31%) maintained at least one incoming relational edge. Table 4.5 details model discrimination across seen card cohorts ($N = 79,914$) versus cold-start card cohorts ($N = 35,197$), as well as across chronological quartiles ($t_1\text{–}t_4$) within Block 5.


+––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––+
| Table 4.5: Stratified Performance Breakdown Across Entity Overlap Cohorts and Longitudinal Time Quartiles (Block 5 OOT)                            |
+––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––+
| 0 - Random Prevalence Floor           | 0.0352                   | 0.0322                   | 0.0323      | 0.0309      | 0.0337      | 0.0408      |
| 0 - Amount-Only Baseline              | 0.0364                   | 0.0359                   | 0.0321      | 0.0319      | 0.0347      | 0.0465      |
| 0 - Relational-Only (19 Counters)     | 0.3229                   | 0.2468                   | 0.3070      | 0.2918      | 0.2907      | 0.3233      |
| 1 - Standalone Tabular LightGBM (S1)  | 0.4410                   | 0.5805                   | 0.4252      | 0.4953      | 0.4460      | 0.5462      |
| 2 - Graph-Augmented GBDT (S2)         | 0.4997 (+0.0587) [CHAMP] | 0.5879 (+0.0074) [CHAMP] | 0.5187      | 0.5545      | 0.4763      | 0.5517      |
| 3 - Frozen Two-Stage Res-GNN (S3)     | 0.4411                   | 0.5658                   | 0.4128      | 0.4946      | 0.4511      | 0.5364      |
| 3n - Res-GNN, No Graph (Adjacency=0)  | 0.4351                   | 0.5754                   | 0.4138      | 0.4891      | 0.4493      | 0.5411      |
| 4n - Joint BGNN, No Boost (GateOnly)  | 0.4395                   | 0.5724                   | 0.4223      | 0.5009      | 0.4519      | 0.5370      |
| 4 - Joint Alternating BGNN (S4)       | 0.4280                   | 0.5439                   | 0.4185      | 0.5068      | 0.4129      | 0.5112      |
+––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––+



====================================================================================================
Figure 4.3: Entity Cohort and Temporal Drift Performance Breakdown on Block 5
====================================================================================================


### 4.4.2 Empirical Insights from Cohort Stratification

1. Relational Uplift on Established Entities ($R_1$ Sub-Test): On established payment cards with observed transaction histories ($N = 79,914$), Stage 2 delivers a $\mathbf{+0.0587}$ AP uplift over Stage 1 (0.4997 vs 0.4410, $p < 0.001$). When an entity possesses historical graph edges, backward-rolling relational counters provide decisive predictive power, confirming the $R_1$ research hypothesis.

2. Why Handcrafted Features Resist Cold-Start Degradation: Stage 2 does not degrade on cold-start cards, maintaining an Average Precision of 0.5879 versus Stage 1's 0.5805. This occurs because the 19 relational features include expanding count and velocity features computed over global categorical tokens (card1, addr1, P_emaildomain). Even if a specific card_id is novel, its billing region (addr1) or issuing bank code (card1) has historical observations that provide stable population priors.

3. Neural Response to Entity Churn: In contrast, Stage 4 exhibits lower performance on unseen cards, recording $\text{AP} = 0.5439$, which is $-0.0366$ AP lower than Stage 1. When an unseen card arrives, its card historical neighborhood is empty ($|\mathcal{N}_{\text{card}}(i)| = 0$). However, because 97.7% of novel cards share digital device or network telemetry with historical transactions, only 796 nodes (0.69%) are strictly isolated across both modalities. For connected novel cards, the blending gate evaluated at $\beta_i > 0$, routing novel entities through neural projections that lacked historical card context.

### 4.4.3 Extreme Continuous Covariate Drift

Tracking representation shift across the expanding chronological development blocks revealed a dramatic vulnerability in continuous neural projections:

* Development Fold 1 (Block 2): Maximum feature dimension drift $= 3.653\,\sigma$
* Development Fold 2 (Block 3): Maximum feature dimension drift $= 2.336\,\sigma$
* Development Fold 3 (Block 4): Maximum feature dimension drift $= 1.493\,\sigma$
* Sealed Holdout (Block 5): Maximum feature dimension drift reached $\mathbf{9.564\sigma}$.

A continuous distribution shift of $9.56\,\sigma$ causes neural linear layers ($\mathbf{W} \mathbf{x} + \mathbf{b}$) and SELU activations to saturate at extreme tails, producing corrupted latent embeddings. Conversely, gradient-boosted decision trees evaluate inputs via ordinal quantile thresholding ($x_j \le \theta$). While all models experience performance shifts across chronological slices $t_1$ through $t_4$ (with Stage 1 AP ranging from 0.4252 in $t_1$ to 0.5462 in $t_4$, Table 4.5), gradient-boosted decision trees maintain structural robustness because ordinal split routing is invariant to monotonic coordinate expansions. In contrast, continuous neural projections suffered severe representation distortion under extreme covariate drift (9.564σ on Block 5).


## 4.5 Operational Queue Triage and Calibration Analysis

In industrial financial risk infrastructure, machine learning models do not operate in an abstract statistical vacuum. Operational triage is constrained by finite human investigation capacity: a typical bank fraud review team can manually inspect at most **$1.0\%$ of daily transaction volume**. Transactions flagged above this capacity threshold trigger immediate operational costs, customer friction, or uninvestigated chargeback exposure.

### 4.5.1 Financial Loss Landscape Under Capacity Constraints

As formalized in Section 3.6.3, the realized scenario-weighted financial loss metric ($L_{\text{fin}}$) quantifies total economic impact:
$$L_{\text{fin}} = \sum_{i=1}^N \left[ y_i (1 - a_i) c_{\text{miss}} A_i + a_i c_{\text{review}} + (1 - y_i) a_i c_{\text{friction}} + y_i a_i c_{\text{residual}} A_i \right]$$
where $c_{\text{miss}} = 1.0$, $c_{\text{review}} = \$5.00$, $c_{\text{friction}} = \$15.00$, $c_{\text{residual}} = 0.05$, and $a_i = \mathbb{I}(p_i \ge \tau_{\text{capacity}})$.

Evaluating financial loss across operational regimes provides essential clarity on the trade-off between threshold economics and triage capacity (Table 4.6):

* **Primary Operational Regime: Fixed 1.0% Queue Capacity Constraint ($K = 1,151$ alerts):**
  When daily manual triage is capped at the top 1.0% highest-risk transactions ($K = 1,151$ alerts), Stage 2 (Graph-Augmented GBDT) achieves the lowest financial loss at \$495,307.56 and intercepts 1,026 fraud events (Table 4.6). Stage 2 delivers an empirical savings of **\$12,682.25** over standalone Tabular LightGBM (Stage 1, \$507,989.81), confirmed by a 95% bootstrap confidence interval of $[\$3,678.70, \$21,541.70]$ ($p = 0.0010$). 
  
  Under this capacity-capped evaluation, the neural architectures achieve modest dollar loss reductions relative to Stage 1 (Stage 3 incurs \$500,781.37, saving \$7,208.44 over Stage 1; Stage 4 incurs \$501,510.90, saving \$6,478.91 over Stage 1). This occurs because ranking the top 1% highest-scoring transactions captures the most obvious, high-exposure fraud cases regardless of model family. However, the neural models remain strictly inferior to Stage 2 by \$5,473.81 (S3) and \$6,203.34 (S4), while suffering substantial Average Precision degradation across the broader continuum ($\Delta_{4-1} = -0.0181$ AP, $\Delta_{3-2} = -0.0480$ AP).

* **Sensitivity Regime: Unconstrained Bayes Cost-Optimal Threshold Rule:**
  Under dynamic thresholding based on transaction-specific loss trade-offs ($\tau_i^* = 35.0 / (A_i + 25.0)$), uncalibrated Focal Loss in Stages 3 and 4 induces severe probability distortion ($ECE = 0.0892$), triggering an unmanageable alert volume surge to **13.46% (15,494 alerts)**. This alert flood generates **\$534,715.00 in administrative review and customer friction costs alone**, escalating uncalibrated loss to **\$634,205.97** (Stage 3) and **\$629,411.78** (Stage 4), representing a +\$263.7K cost penalty over Stage 1 (\$365,728.99). Post-hoc isotonic calibration successfully realigns probabilities, reducing calibrated alert rates to 4.3%–4.6% and dropping calibrated losses to \$366,820.96 (Stage 3) and \$373,410.83 (Stage 4).


====================================================================================================
Figure 4.4: Calibration Reliability Diagrams and Probability Distributions (Raw vs. Isotonic Calibration)
====================================================================================================


### 4.5.2 Focal Loss Distortion and Isotonic Probability Calibration

The elevated loss of Stage 4 under unconstrained thresholding stems from probability distortion induced by Mini-Batch Focal Loss:
$$\mathcal{L}_{\text{Focal}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
To address class imbalance during backpropagation, Stage 4 was trained with $\alpha = 0.25$ and focusing parameter $\gamma = 2.0$. While Focal Loss prevents easy negative transactions from dominating gradients, the modulating factor $(1 - p_t)^\gamma$ shifts the probabilistic interpretation of network outputs. 

Empirical analysis on the Block 5 holdout reveals the extent of this calibration failure:
* **True Ground Truth Fraud Base Rate:** **3.41%** (3,929 / 115,111)
* **Stage 1 Mean Predicted Probability:** **0.0295** (closely aligned with empirical base rate)
* **Stage 4 Mean Predicted Probability:** **0.1269** (3.7× base rate inflation)
* **Raw Expected Calibration Error (ECE):** Exploded from 0.0061 (Stage 1) to **0.0892** (Stage 4).
* **Raw Operational Alert Rate:** Jumped from 2.75% (Stage 1) to an unmanageable **13.46%** (Stage 4).

When raw probabilities from Stage 4 were piped into fixed risk thresholds, the model flagged 13.46% of all clearing volume, exceeding human investigation capacity by over **13×**. This inundated operations with false alerts, causing customer friction review and customer friction costs to escalate to \$534,715.00.

Applying post-hoc **isotonic regression calibration** fitted strictly on out-of-fold validation splits successfully restored probability fidelity:
* Calibrated ECE dropped from 0.0892 to **0.0077** (Table 4.1).
* Calibrated financial loss dropped from $629,411.78 to **$373,410.83**.

However, as mathematically proven in Section 3.6.1, isotonic regression is a strictly monotonic transformation:
$$\tilde{p}_i = g_{\text{iso}}(p_i), \qquad g_{\text{iso}}'(p) \ge 0$$
While isotonic calibration repairs decision thresholds and dollar loss calculations, it **cannot alter the non-interpolated rank ordering of transactions**. Consequently, Average Precision remained bounded at 0.4478 (Calibrated AP in Table 4.1), confirming that post-hoc calibration repairs threshold economics but cannot recover structural discriminative deficits.


## 4.6 Comprehensive Discussion & Research Synthesis

This chapter provides rigorous empirical answers to the five core Research Questions formulated in Section 1.4:

### 1. Resolution of Research Question 1 (Base Tabular vs. Relational Value)
* *Verdict:* Standalone tabular GBDTs provide a strong baseline ($\text{AP} = 0.4814$). Handcrafted relational features in Stage 2 deliver an uplift to $\text{AP} = 0.5252$ ($\Delta_{2-1} = +0.0438$, $p = 0.00175$), achieve the lowest economic loss (\$345.4k, saving \$20,326.68), and produce larger gains on seen entities ($+0.0587$ AP). Explicit backward-looking relational features offer an effective approach for integrating relational data into production fraud systems.

### 2. Resolution of Research Question 2 (Isolated Graph Message Passing)
* *Verdict:* Replacing tabular feature engineering with isolated continuous message passing results in lower performance ($\Delta_{3-2} = -0.0480$ AP). Identical-network ablations ($H_{2\text{a}}, H_{2\text{b}}$) indicate that spatial message passing over $G_{<t}$ adds no measurable predictive signal ($\Delta = +0.0009$ AP, $p_{\text{adj}} = 0.723$). Mean aggregation functions as a low-pass filter, while PCA compression discards rare fraud interactions.

### 3. Resolution of Research Question 3 (Functional-Gradient Coupling)
* *Verdict:* Pre-specified hypothesis $H_3$ is not supported ($\Delta_{4-3} = -0.0138$ AP, 95% bootstrap CI: $[-0.0191, -0.0087]$, $p_{\text{adj}} = 0.00175$). Alternating functional-gradient updates do not improve upon frozen embeddings, altering latent coordinate space, while gate saturation ($\bar{\beta} = 0.8726$) reduces the weight on tabular predictions.

### 4. Resolution of Research Question 4 (Cold-Start & Temporal Robustness)
* *Verdict:* In an open-world setting where 30.6% of entities are novel, neural architectures show reduced performance ($\Delta = -0.0366$ AP on cold-start cards). Furthermore, continuous latent spaces show temporal drift (9.564σ on Block 5). In contrast, decision trees utilize global categorical distributions and monotonic split invariance, maintaining temporal stability.

### 5. Resolution of Research Question 5 (Component-Level Contribution)
* *Verdict:* Stepwise ablations indicate that functional boosting displacements contribute $-0.0178$ AP, and monotonic gating provides $+0.0039$ AP. Spatial convolutions contribute $+0.0000$ AP. The architectural limitation lies in continuous coordinate parameterization and unconstrained gate weighting.

### 4.6.1 Methodological Value of the Empirical Null Result

In machine learning research, negative or null findings are sometimes omitted or obscured by transductive look-ahead edges and random train-test splits. This dissertation maintains that an auditable empirical null result under time-causal constraints provides clearer scientific guidance than a transductive model that fails under production conditions.

By enforcing score-before-insert semantics on $G_{<t}$, a 1-day clearing embargo, and sealed out-of-time Block 5 evaluation, this research established that deep graph neural convolutions face profound mathematical and operational hurdles in financial fraud detection. For industrial financial institutions, this work provides concrete rules of thumb: prioritize explicit, non-linear backward-rolling ratio features in GBDT pipelines, enforce strict probability calibration when using focal loss, and treat unconstrained deep graph architectures with extreme caution in streaming, open-world payment environments.