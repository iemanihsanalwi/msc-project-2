# Chapter 3: Research Methodology and System Formulation

## 3.1 Empirical Research Design

This empirical research design evaluates whether joint graph-tabular representation learning provides measurable, cost-effective improvements over tuned tabular models under streaming constraints. To support direct comparison across model families, the study applies a controlled Four-Stage Experimental Benchmarking Ladder.

Rather than comparing models with differing feature engineering, parameter counts, or training budgets, each stage in the ladder introduces one specific algorithmic mechanism (summarized formally in Table 3.1):


+–––––––––––––––––––––––––––––––––––––––––––––––––-+
|                        THE 4-STAGE INCREMENTAL BENCHMARKING LADDER                                |
+–––––––––––––––––––––––––––––––––––––––––––––––––-+
|                                                                                                   |
|  [STAGE 1: Standalone Tabular GBDT Baseline (S1)]                                                 |
|  - Feature Space: 371 Raw audit-hardened tabular attributes (Transaction & Identity).            |
|  - Model Architecture: Pure LightGBM with default gradient-based missingness paths.               |
|  - Parameter Budget: K_base = 500 trees, 256 leaves, learning rate eta = 0.03.                    |
|  - Research Role: Establishes the non-relational statistical upper bound.                         |
|                                         |                                                         |
|                                         v  (Introduce 19 Handcrafted Causal Relational Counters)  |
|  [STAGE 2: Graph-Augmented GBDT (S2)]                                                             |
|  - Feature Space: 371 Base Features + 19 Backward-Rolling Relational Features                     |
|    (SeenBefore counts, Inter-arrival DtDiff, Monetary Running Ratios & Differences).              |
|  - Model Architecture: Pure LightGBM with identical tree budget (K_base = 500 trees).             |
|  - Research Role: Isolates the marginal predictive uplift of manual feature engineering (Delta_2-1)|
|                                         |                                                         |
|                                         v  (Replace Handcrafted Features with Neural Convolutions)|
|  [STAGE 3: Frozen Two-Stage Res-GNN (S3)]                                                         |
|  - Feature Space: Continuous aligned coordinate matrix Z_0 in R^{N x 49} on G_<t.                 |
|  - Model Architecture: Decoupled 2-Stage PyTorch Res-GNN; message passing over causal neighbors;  |
|    concatenated residual skip connection; Stage 1 GBDT logits kept strictly frozen.               |
|  - Control Ablation (S3n): Identical neural network with graph propagation zeroed (Adjacency = 0).|
|  - Research Role: Isolates automated multi-hop message passing without tree feedback (Delta_3-2). |
|                                         |                                                         |
|                                         v  (Introduce Functional Coupling R_i^m & Monotonic Gate) |
|  [STAGE 4: Alternating Jointly Optimized BGNN (S4)]                                               |
|  - Feature Space: Dynamic coordinate matrix Z_0 updated via functional gradient boosting.         |
|  - Model Architecture: Monotonic Gated BGNN coupled with multi-output LightGBM displacement trees  |
|    via alternating residual fitting R_i^m; monotonic gate beta_i enforcing exact fallback.       |
|  - Control Ablation (S4n): Identical gated model with boosting updates disabled (eta_boost = 0).   |
|  - Research Role: Evaluates the primary coupling hypothesis Delta_4-3 and macro uplift Delta_4-1. |
|                                                                                                   |
+–––––––––––––––––––––––––––––––––––––––––––––––––-+


### 3.1.1 Mathematical Formalization of the Experimental Stages
Let $\mathcal{D} = \{ (\mathbf{x}_i, y_i, t_i) \}_{i=1}^N$ denote the chronological transaction stream, where $\mathbf{x}_i \in \mathcal{X}$ is the raw tabular feature vector, $y_i \in \{0, 1\}$ is the binary fraud indicator, and $t_i = \text{TransactionDT}(i)$ is the transaction timestamp.
1. **Stage 1 (Standalone Tabular GBDT):**
   $$\hat{y}_i^{(1)} = F_{\text{GBDT}}^{(S_1)}(\mathbf{x}_i) = \sum_{k=1}^{K_{\text{base}}} \eta f_k(\mathbf{x}_i)$$
   where $F_{\text{GBDT}}^{(S_1)}$ is trained solely on the 371 base tabular features.
2. **Stage 2 (Graph-Augmented GBDT):**
   $$\hat{y}_i^{(2)} = F_{\text{GBDT}}^{(S_2)}\left( [\mathbf{x}_i \,\|\, \mathbf{r}_i^{\text{hist}}] \right) = \sum_{k=1}^{K_{\text{base}}} \eta f_k\left( [\mathbf{x}_i \,\|\, \mathbf{r}_i^{\text{hist}}] \right)$$
   where $\mathbf{r}_i^{\text{hist}} \in \mathbb{R}^{19}$ denotes the vector of backward-looking, time-causal entity-history features extracted strictly from transactions occurring prior to $t_i$ (cataloged by group in Table 3.5).
3. **Stage 3 (Frozen Two-Stage Res-GNN):**
   $$\hat{y}_i^{(3)} = \text{GNN}\left( \mathbf{z}_i^{(0)}, G_{<t_i}; \mathbf{\Theta}_{\text{gnn}} \right) = \sigma\left( \mathbf{w}_{\text{head}}^T \left[ \mathbf{h}_{\text{graph}, i} \,\|\, \mathbf{z}_i^{(0)} \right] + b_{\text{head}} \right)$$
   where $\mathbf{z}_i^{(0)} \in \mathbb{R}^{49}$ is the aligned continuous coordinate embedding incorporating frozen Stage 1 predictions, and $G_{<t_i}$ is the strictly causal historical graph.
4. **Stage 4 (Alternating Jointly Optimized BGNN):**
   $$\hat{y}_i^{(4)} = \sigma\left( \beta_i \cdot \text{Logit}_{\text{GNN}, i}^{(m)}\left( \mathbf{z}_i^{(m)}, G_{<t_i} \right) + (1.0 - \beta_i) \cdot \text{Logit}_{\text{GBDT}, i}^{(S_1)} \right)$$
   where $\mathbf{z}_i^{(m)}$ is iteratively updated across $M$ outer rounds via multi-output functional boosting:
   $$\mathbf{z}_i^{(m)} = \mathbf{z}_i^{(m-1)} + \eta_{\text{boost}} \cdot \mathbf{R}_i^m$$
   and $\beta_i \in [0, 1]$ is the adaptive monotonic blending gate.

### 3.1.2 The Pre-Specified Confirmatory Hypothesis Framework
To enforce scientific rigor and protect against post-hoc $p$-hacking, all statistical claims are evaluated against a pre-specified confirmatory testing hierarchy:
* **Primary Coupling Uplift ($H_3$ / RQ3):** The integrated alternating BGNN (Stage 4) must demonstrate a statistically significant and materially meaningful ranking improvement over the decoupled Res-GNN pipeline (Stage 3). The pre-specified decision boundary requires that the lower bound of the 95% bootstrap confidence interval strictly exceeds zero, and the point estimate achieves at least a 0.5 percentage point absolute increase in out-of-time Average Precision:
  $$\Delta_{4-3} = \text{AP}_4 - \text{AP}_3 \ge +0.005, \quad \text{with } \text{LCB}_{0.95}(\Delta_{4-3}) > 0$$
* **Primary Operational Queue Loss Efficiency ($H_{3\text{c}}$):** Under the primary operational investigation capacity constraint ($\text{AlertRate} \le 1.0\%$, $K = 1,151$ alerts), Stage 2 (Graph-Augmented GBDT) must achieve a statistically significant reduction in realized financial loss relative to Stage 1:
  $$\Delta L_{\text{queue}}^{(2-1)} = L_{\text{fin}}^{(1)} - L_{\text{fin}}^{(2)} > 0, \quad \text{with } \text{LCB}_{0.95}(\Delta L) > 0$$
* **Macro System Uplift ($H_4$ / RQ1):** Stage 4 must outperform the primary industrial benchmark, Standalone Tabular LightGBM (Stage 1), in both Average Precision ($\Delta_{4-1} \ge +0.005, \ p_{\text{adj}} \le 0.05$) and realized financial loss ($L_{\text{fin}}^{(1)} - L_{\text{fin}}^{(4)} > 0$).
* **Structural Decoupling Impact ($H_2$ / RQ2):** Evaluates whether learned continuous graph message passing matches or exceeds manual feature engineering ($\Delta_{3-2} = \text{AP}_3 - \text{AP}_2 \ge 0$).
* **Multiplicity Correction:** Family-wise type-I error rate across all confirmatory tests is controlled at $\alpha = 0.05$ using the Benjamini–Hochberg False Discovery Rate (FDR) procedure.


## 3.2 Dataset, Temporal Partitioning, & Leakage Controls

### 3.2.1 IEEE-CIS Benchmark Dataset Specifications
The empirical investigation uses the benchmark IEEE-CIS Fraud Detection dataset, comprising real-world payment transactions collected by Vesta Corporation across an active 182-day continuum ($\text{TransactionDT} \in [86,400, 15,811,131]$ seconds). The unit of analysis is the individual transaction $v_i$. The ingestion pipeline ingests 432 raw attributes: 392 features from train_transaction (including ground-truth label isFraud, monetary value TransactionAmt, product code ProductCD, anonymized card proxies card1–card6, geographic coordinates addr1–addr2, distance proxies dist1–dist2, purchaser/recipient email domains, transaction velocity counts C1–C14, timedelta features D1–D15, match indicators M1–M9, and 339 proprietary behavioral features V1–V339) and 40 identity attributes from train_identity (DeviceType, DeviceInfo, and id_01–id_38, joining on N = 144,233 rows, i.e., 24.42% coverage and 75.58% dataset-wide row-level identity missingness; Table 3.2). For the tabular pipeline (Stages 1 and 2), retaining raw attributes yields 371 base tabular features (comprising the raw V-columns, D-deltas, C-counts, and categorical encodings). For the neural pipeline (Stages 3 and 4), continuous features are transformed into an aligned coordinate matrix $Z_0 \in \mathbb{R}^{N \times 49}$, consisting of 32 standardized continuous features, 15 orthogonal principal components from Block PCA compressing the 339 V-features (capturing 69.5% average cumulative variance across folds), 1 out-of-fold GBDT risk logit from Stage 1, and 1 logit availability indicator. Across the complete labeled dataset of $N = 590,540$ records, the global fraud prevalence is 3.4990% ($N = 20,663$).

### 3.2.2 Chronological Partitioning, Embargo, and Delayed Supervision
To reflect live production clearing pipelines and eliminate temporal look-ahead bias (quantified in Table 3.3):
1. **5-Block Chronological Slicing:** Transactions are ordered strictly by ascending TransactionDT and partitioned into five non-overlapping contiguous blocks of exactly 118,108 transactions each ($\approx 36.4$ days per block):
   * **Blocks 1 to 4 (Development Continuum):** $N = 472,432$ transactions (Days 0.00 to 140.12), dedicated to expanding chronological model fitting, internal hyperparameter selection, and probability calibration.
   * **Block 5 (Sealed Out-of-Time Test Set):** $N = 118,108$ transactions (Days 140.12 to 182.00 (spanning 41.88 days, approximately 42 days)), quarantined as a strictly sealed holdout evaluated exactly once.
2. **1-Day Clearing Embargo:** To prevent boundary leakage caused by multi-day transaction settlement and time-zone discrepancies, a strict **24-hour clearing embargo** ($\tau_{\text{embargo}} = 86,400\text{ seconds}$) is enforced between training blocks and evaluation slices. Transactions falling within the 24 hours immediately following the training boundary are completely excised from validation scoring.
3. **Conservative Block-Delayed Supervision Policy:** Because real-world cardholder fraud chargebacks exhibit a reporting latency of 30 to 90 days, labels from chronological block $B_k$ are treated as inaccessible until the complete termination of block $B_k$. During the execution of any active block, incoming transactions are treated as unlabeled, preventing models from learning on unverified recent feedback.


+–––––––––––––––––––––––––––––––––––––––––––––––––-+
|                        5-BLOCK EXPANDING CHRONOLOGICAL PARTITIONING TIMELINE                      |
+–––––––––––––––––––––––––––––––––––––––––––––––––-+
|                                                                                                   |
|  [Block 1: Days 0.00 - 25.74] –> Initial Training Continuum (N = 118,108, Fraud = 2.39%)          |
|  [Block 2: Days 25.74 - 63.73] –> Validation Fold 1 (N = 118,108, Fraud = 4.01%)                   |
|  [Block 3: Days 63.73 - 100.22]–> Validation Fold 2 (N = 118,108, Fraud = 3.75%)                  |
|  [Block 4: Days 100.22 - 140.12]-> Validation Fold 3 / Calibration Pool (N = 118,108, Fraud = 3.90%)|
|  ================================= 24-HOUR CLEARING EMBARGO (2,997 rows) ======================   |
|  [Block 5: Days 140.12 - 182.00]-> SEALED OUT-OF-TIME HOLDOUT (N = 115,111 eval, Fraud = 3.41%)    |
|                                                                                                   |
+–––––––––––––––––––––––––––––––––––––––––––––––––-+


### 3.2.3 Dual Feature Engineering Pipelines
To avoid cross-modality contamination and target leakage, tabular and neural feature processing pipelines are isolated:
* Tabular Feature Pipeline: Preserves the irregular structure of financial attributes. The 371 base features remain in native form. Continuous numerical features are not standardized; missing values remain as native NaN to allow LightGBM to compute default branch routing; categorical variables are mapped to integer indices using training dictionaries, routing unseen evaluation tokens to an UNKNOWN bin.
* Neural Feature Pipeline: Transforms tabular attributes into a standardized format for gradient descent:
  1. Logarithmic Transformation: Skewed monetary and distance features (TransactionAmt, dist1, dist2, C1, C2, C13) are log-transformed: $x \mapsto \log(1 + \max(0, x))$.
  2. Standard Scaling: Continuous variables are standardized to zero mean and unit variance using statistics fitted strictly on training folds ($\mu_{\text{train}}, \sigma_{\text{train}}$) without input outlier clipping (explaining why large positive temporal drift propagates in Section 4.4.3, whereas Stage 4 functional boosting displacements are clipped at $\delta_R = 0.20$).
  3. Block-Wise PCA Compression: The 339 behavioral $V$-columns exhibit substantial correlation. Rather than passing unnormalized $V$-features into the neural network, unsupervised Principal Component Analysis (PCA) fitted on training data compresses the 339 columns into 15 orthogonal principal components ($\mathbf{V}_{\text{PCA-15}}$), retaining $\approx 70\%$ of cumulative variance.

### 3.2.4 Chronological Out-of-Fold (OOF) GBDT Logit Generation for Z₀
To supply the neural network with native tabular predictions without target leakage, the initial coordinate representation $Z_0$ incorporates an Out-of-Fold GBDT logit. Within the development continuum (Blocks 1–4), an expanding chronological cross-validation protocol generates out-of-fold predictions:
* Block 1 is split into 4 chronological subfolds to bootstrap initial out-of-fold predictions.
* For each subsequent block $B_k$ ($k \in \{2, 3, 4\}$), an ensemble trained on blocks $B_1 \cup \dots \cup B_{k-1}$ predicts on block $B_k$.
* For the sealed Block 5 holdout, the ensemble trained on the full historical continuum (Blocks 1–4, $N = 472,432$) generates out-of-sample logits.

The resulting out-of-fold probability $\hat{p}_i$ is converted into an unbounded logit:
$$l_i = \log\left( \frac{\text{clip}(\hat{p}_i, \epsilon, 1 - \epsilon)}{1 - \text{clip}(\hat{p}_i, \epsilon, 1 - \epsilon)} \right), \quad \epsilon = 10^{-6}$$
and paired with an availability indicator $\mathbb{I}_{\text{avail}, i} \in \{0, 1\}$ to form the aligned representation.


## 3.3 Heterogeneous Temporal Graph Formulation

### 3.3.1 Mathematical Definition of the Causal Graph ($G_{<t}$)
The financial transaction stream is modeled as a time-causal, directed, heterogeneous multigraph:
$$G_{<t} = \left( \mathcal{V}_{<t}, \mathcal{E}_{<t}, \tau, \phi, \mathcal{T} \right)$$
where:
* $\mathcal{V}_{<t} = \mathcal{V}_{\text{tx}} \cup \mathcal{V}_{\text{card}} \cup \mathcal{V}_{\text{dev}}$ is the set of active vertices existing prior to timestamp $t$.
* $\mathcal{E}_{<t}$ is the set of directed, time-stamped relational edges.
* $\tau: \mathcal{V} \to \{ \text{Transaction}, \text{Card}, \text{Device} \}$ is the node-type mapping.
* $\phi: \mathcal{E} \to \{ \text{UsesCard}, \text{UsesDevice}, \text{TemporalPredecessor} \}$ is the relation-type mapping.
* $\mathcal{T}: \mathcal{V} \cup \mathcal{E} \to \mathbb{R}^+$ maps entities and events to their continuous chronological occurrence timestamp (TransactionDT). Table 3.4 summarizes node and edge counts across development and Block 5 partitions, highlighting the distinction between dataset-wide row-level identity absence (75.58%) and Block 5 missing-device supernode routing (83.80%).


+–––––––––––––––––––––––––––––––––––––––––––––––––-+
|                        TIME-CAUSAL HETEROGENEOUS TRANSACTION GRAPH SCHEMA                         |
+–––––––––––––––––––––––––––––––––––––––––––––––––-+
|                                                                                                   |
|     [Card Proxy Entity]                                           [Device Proxy Entity]           |
|     Key: card1-6 + addr1                                          Key: DeviceInfo + Browser + OS  |
|            ^       ^                                                     ^       ^                |
|            |       |                                                     |       |                |
|       (UsesCard) (UsesCard)                                         (UsesDev) (UsesDev)           |
|            |       |                                                     |       |                |
|            |       +––––––––––––––––––+                |       |                |
|            |                                            |                |       |                |
|     [Transaction v_{t-2}]                             [Transaction v_{t-1}]      |                |
|     Time: t - 14 days                                 Time: t - 2 hours          |                |
|            ^                                                    ^                |                |
|            |                                                    |                |                |
|            +====== (Backward Temporal Predecessor Edge) ========+                |                |
|                                 (t_{prev} < t)                                   |                |
|                                       ^                                          |                |
|                                       | (Backward Message Passing)               |                |
|                                       |                                          |                |
|                             [Query Transaction v_t] =============================+                |
|                             Timestamp: t (Current)                                                |
|                             Status: Score-Before-Insert (Ephemeral Subgraph G_t)                  |
|                                                                                                   |
+–––––––––––––––––––––––––––––––––––––––––––––––––-+


### 3.3.2 Entity Hierarchy and Resolution Rules
Because true cardholder account numbers (PANs) are masked for payment card industry (PCI-DSS) compliance, entity proxy nodes are synthesized using deterministic multi-attribute resolution rules:
1. **Card Proxy Nodes ($\mathcal{V}_{\text{card}}$):** Constructed via the composite string tuple:
   $$\text{Key}_{\text{card}} = \langle \text{card1}, \text{card2}, \text{card3}, \text{card4}, \text{card5}, \text{card6}, \text{addr1} \rangle$$
   This composite key disambiguates co-branded issuing cards while isolating individual household accounts sharing regional postal codes (addr1).
2. **Device Proxy Nodes ($\mathcal{V}_{\text{dev}}$):** Constructed via digital hardware and operating telemetry:
   $$\text{Key}_{\text{dev}} = \langle \text{DeviceInfo}, \text{id\_31}, \text{id\_30}, \text{DeviceType} \rangle$$
   capturing distinct hardware configurations executing transactions across the network.

### 3.3.3 Score-Before-Insert Ephemeral Query Graphs (Gᵢ)
To prevent intra-batch and forward leakage, graph evaluation follows a Score-Before-Insert lifecycle:
1. When query transaction $v_i$ arrives at time $t_i$, an ephemeral query graph $G_i$ is instantiated, consisting of $v_i$ and the historical graph $G_{<t_i}$ containing entities and transactions recorded at $t < t_i$.
2. Message passing aggregates representations along incoming edges directed backwards in time: $u \to v_i$, with $t_u < t_i$.
3. Transaction $v_i$ is scored and its output logit is generated.
4. After scoring completes, the structural edges connecting $v_i$ to its card and device proxies are committed to the persistent graph state.
5. 30-Day Bounded Lookback ($K_{\text{lookback}}$): Edges are restricted to a backward lookback horizon of 30 days (2,592,000 seconds). Transactions older than 30 days are aged out and disconnected from active neighborhoods, bounding graph degree expansion.

## 3.4 Adapted Boost then Convolve Architecture

### 3.4.1 Dimensional Accounting of the Aligned Embedding Z₀
Rather than initializing latent representations from scalar predictions, our adapted architecture constructs an aligned continuous coordinate embedding $Z_0 \in \mathbb{R}^{N \times d_{\text{emb}}}$ with dimension $d_{\text{emb}} = 49$:
$$Z_0 = \left[ \mathbf{X}_{\text{cont, scaled}} \,\|\, \mathbf{V}_{\text{PCA-15}} \,\|\, \mathbf{l}_{\text{GBDT}}^{(S_1)} \,\|\, \mathbb{I}_{\text{avail}} \right]$$
where:
* $\mathbf{X}_{\text{cont, scaled}} \in \mathbb{R}^{32}$: Standardized continuous transaction features.
* $\mathbf{V}_{\text{PCA-15}} \in \mathbb{R}^{15}$: Orthogonal principal components compressing the 339 raw $V$-features.
* $\mathbf{l}_{\text{GBDT}}^{(S_1)} \in \mathbb{R}^1$: Calibrated out-of-fold logit generated by the Stage 1 tabular LightGBM model.
* $\mathbb{I}_{\text{avail}} \in \{0, 1\}$: Binary availability flag indicating whether an upstream GBDT logit was present.

### 3.4.2 Relation-Aware Temporal Message Passing
For each active transaction node $v_i$, the Graph Neural Network aggregates incoming historical information across card and device relations using relation-specific projection matrices:
$$\mathbf{h}_{\text{self}, i} = \text{SELU}\left( \mathbf{W}_{\text{self}} \mathbf{z}_i \right)$$
$$\mathbf{h}_{\text{card}, i} = \text{SELU}\left( \mathbf{W}_{\text{card}} \bar{\mathbf{z}}_{\mathcal{N}_{\text{card}}(i)} \right) \cdot \text{mask}_{\text{card}, i}$$
$$\mathbf{h}_{\text{dev}, i} = \text{SELU}\left( \mathbf{W}_{\text{dev}} \bar{\mathbf{z}}_{\mathcal{N}_{\text{dev}}(i)} \right) \cdot \text{mask}_{\text{dev}, i}$$
where:
$$\bar{\mathbf{z}}_{\mathcal{N}(i)} = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \mathbf{z}_j$$
denotes the unweighted mean representation of causal historical predecessors, $\mathbf{W}_{\text{self}}, \mathbf{W}_{\text{card}}, \mathbf{W}_{\text{dev}} \in \mathbb{R}^{d_{\text{hidden}} \times d_{\text{emb}}}$ are learned linear projections ($d_{\text{hidden}} = 64$), and $\text{mask}_{\text{card}, i}, \text{mask}_{\text{dev}, i} \in \{0, 1\}$ are binary indicators signifying edge presence.

The projected representations are concatenated and fused through a non-linear projection layer with Scaled Exponential Linear Unit (SELU) activations and dropout:
$$\mathbf{h}_{\text{graph}, i} = \mathbf{W}_2 \cdot \text{SELU}\left( \text{Dropout}\left( \mathbf{W}_1 \left[ \mathbf{h}_{\text{self}, i} \,\|\, \mathbf{h}_{\text{card}, i} \,\|\, \mathbf{h}_{\text{dev}, i} \right] + \mathbf{b}_1 \right) \right) + \mathbf{b}_2$$
The fused structural representation is concatenated with the raw input coordinate $\mathbf{z}_i$ via a residual skip connection, feeding into a linear classification head:
$$\text{Logit}_{\text{GNN}, i} = \mathbf{w}_{\text{head}}^T \left[ \mathbf{h}_{\text{graph}, i} \,\|\, \mathbf{z}_i \right] + b_{\text{head}}$$

### 3.4.3 Normalized Mini-Batch Focal Loss
To counteract the extreme 96.5% negative class imbalance, neural optimization minimizes normalized **Focal Loss** ($\gamma = 2.0, \alpha = 0.25$):
$$\mathcal{L}_{\text{focal}}(p_i, y_i) = -\alpha_t (1 - p_{t, i})^\gamma \log(p_{t, i})$$
where:
$$p_{t, i} = y_i p_i + (1 - y_i)(1 - p_i), \quad \alpha_t = y_i \alpha + (1 - y_i)(1 - \alpha)$$
and $p_i = \sigma(\text{Logit}_{\text{blended}, i})$ is the predicted fraud probability. The focusing parameter $\gamma = 2.0$ dynamically down-weights easy negatives, concentrating gradient updates on hard-to-classify borderline transactions.

### 3.4.4 Alternating Functional-Gradient Optimization Mechanics
The alternating optimization framework couples the neural GNN and tabular GBDT through iterative functional residual updates in continuous representation space:


+–––––––––––––––––––––––––––––––––––––––––––––––––-+
|                        ALTERNATING FUNCTIONAL-GRADIENT OPTIMIZATION LOOP                          |
+–––––––––––––––––––––––––––––––––––––––––––––––––-+
|                                                                                                   |
|  [Outer Round m in {1, ..., M}]                                                                   |
|                                                                                                   |
|  +––––––––––––––––––––––––––––––––––––––––––––––-+  |
|  | Phase A: Inner Neural GNN Optimization (J epochs via AdamW)                                  |  |
|  | 1. Freeze GBDT displacement trees.                                                          |  |
|  | 2. Perform message passing over G_<t using current coordinate matrix Z_0^(m-1).             |  |
|  | 3. Compute blended logits: Logit_final = beta * Logit_GNN + (1 - beta) * Logit_GBDT.        |  |
|  | 4. Update neural weights Theta_GNN via backpropagation on Focal Loss.                       |  |
|  +––––––––––––––––––––––––––––––––––––––––––––––-+  |
|                                                |                                                  |
|                                                v                                                  |
|  +––––––––––––––––––––––––––––––––––––––––––––––-+  |
|  | Phase B: Bounded Functional Residual Calculation                                            |  |
|  | 1. Compute training predictions: p_i = sigma(Logit_final, i).                                |  |
|  | 2. Evaluate scalar residuals: r_i = y_i - p_i.                                               |  |
|  | 3. Fit LightGBM Regressor T_m on raw tabular features X predicting scalar residuals r_i.    |  |
|  | 4. Compute global projection direction: d = Normalize( Mean( Z_0 * r_i ) ).                 |  |
|  | 5. Calculate bounded coordinate displacement vector:                                        |  |
|  |    R_i^m = clip_{delta_R}( T_m(x_i) * d, [-0.2, 0.2] ).                                     |  |
|  +––––––––––––––––––––––––––––––––––––––––––––––-+  |
|                                                |                                                  |
|                                                v                                                  |
|  +––––––––––––––––––––––––––––––––––––––––––––––-+  |
|  | Phase C: Latent Manifold Update                                                             |  |
|  | 1. Additively displace continuous coordinates:                                              |  |
|  |    Z_0^(m) = Z_0^(m-1) + eta_boost * R_i^m   (eta_boost = 0.1).                             |  |
|  | 2. Re-aggregate causal neighborhood means over G_<t for next outer iteration.               |  |
|  +––––––––––––––––––––––––––––––––––––––––––––––-+  |
|                                                                                                   |
+–––––––––––––––––––––––––––––––––––––––––––––––––-+


1. Neural Parameter Optimization (Phase A): For $J_{\text{inner}} = 2$ epochs, GNN weights $\mathbf{\Theta}_{\text{gnn}}$ are trained using AdamW ($\eta = 0.003$) on mini-batches of size 4,096.
2. Bounded Coordinate Displacement (Phase B): The model generates training predictions $p_i = \sigma(\text{Logit}_{\text{blended}, i})$ and computes scalar residuals $r_i = y_i - p_i$. In canonical BGNN (Ivanov & Prokhorenkova, 2021), functional gradient boosting trains decision trees to predict multi-dimensional gradients with respect to node features ($-\nabla_{\mathbf{z}_i}\mathcal{L}$), requiring $d_{\text{emb}}$ separate trees per boosting round. Over the development continuum ($N = 472,432$ nodes), fitting 49 independent trees per outer round ($49 \times 4 = 196$ trees) adds significant computational overhead. To maintain tractability, the adapted Stage 4 architecture applies a Rank-1 Projected Functional Coordinate Update: a single scalar LightGBM regressor $T_m$ is trained on raw tabular attributes $\mathbf{x}_i$ to predict the scalar output residual $r_i = y_i - p_i$. Note that there is a loss target distinction: the tree regressor fits standard cross-entropy residuals $r_i = y_i - p_i$, whereas neural network parameters are optimized using Normalized Mini-Batch Focal Loss ($\alpha = 0.25, \gamma = 2.0$). The scalar tree prediction is projected into $\mathbb{R}^{d_{\text{emb}}}$ along the empirical covariance direction $\mathbf{d} = \text{Normalize}(\mathbb{E}[\mathbf{Z}_0 \cdot r_i])$ and clipped:
   $$\mathbf{R}_i^m = \text{clip}_{\delta_R}\left( T_m(\mathbf{x}_i) \cdot \mathbf{d}, \ [-\delta_R, \delta_R] \right), \quad \delta_R = 0.20$$
3. Latent Coordinate Update (Phase C): The continuous embedding matrix is updated additively:
   $$\mathbf{Z}_0^{(m)} = \mathbf{Z}_0^{(m-1)} + \eta_{\text{boost}} \cdot \mathbf{R}_i^m, \quad \eta_{\text{boost}} = 0.1$$
The optimization procedure runs for $M_{\text{outer}} = 4$ outer boosting rounds.

### 3.4.5 Algorithmic Pseudocode
Algorithm 3.1 formally specifies the complete alternating training protocol:


```text
Algorithm 3.1: Alternating Joint Optimization of Time-Causal BGNN Framework
------------------------------------------------------------------------------------------------------------------------
Input:  Chronological Training Stream X_train in R^(N x D), Targets Y_train in {0,1}^N, Timestamps T in R^N
        Lookback Cap tau_max = 2,592,000s (30 days), Dynamic Graph G_<t, Outer Rounds M = 4, Inner Epochs J = 2
Output: Optimized Base Ensembles F_base, Coordinate Updating Regressors {T_m}_(m=1)^(M-1), GNN Parameter Tensors Theta_gnn
------------------------------------------------------------------------------------------------------------------------
 1:  Initialize representation matrix Z_0 in R^(N x 49): [32 standard continuous || 15 PCA continuous || 1 OOF logit || 1 ID flag]
 2:  Initialize GNN parameter tensors Theta_gnn via Xavier normal initialization (d_hidden = 64, L = 1 aggregation layer, SELU)
 3:  Initialize fallback blending gate parameters: w_c, w_d, b_g
 4:  Z_current <- Z_0
 5:  For outer boosting round m = 1 to M (M = 4) do:
 6:      // Phase A: Neural Parameter Optimization (Inner GNN Loop across J = 2 epochs)
 7:      For inner epoch j = 1 to J do:
 8:          For each mini-batch B sampled chronologically (|B| = 4096) do:
 9:              Extract causal 1-hop neighbours from G_<t via causal mean-pooling (K_card <= 10, K_dev <= 5, 30-day window)
10:              Compute relation-specific message passing activations:
11:                  h_card <- SELU( W_card * mean_{u in N_card(v)} z_u )
12:                  h_dev  <- SELU( W_dev  * mean_{u in N_dev(v)}  z_u )
13:                  h_v    <- proj_self(z_v) + h_card + h_dev  // Residual skip-projection
14:              Compute connectivity indicator I(connected_i) = I(c_i + d_i > 0)
15:              Compute adaptive blending gate: beta_i <- I(connected_i) * sigmoid( Softplus(w_c)*c_i + Softplus(w_d)*d_i + b_g )
16:              Compute blended predictions: Logit_i <- beta_i * Head(h_i) + (1.0 - beta_i) * Z_0[i, OOF_index]
17:              Compute Normalized Mini-Batch Focal Loss: L_focal <- -1/|B| sum_(i in B) alpha_t (1 - p_i)^gamma log(p_i)
18:              Update neural weights: Theta_gnn <- Theta_gnn - lr_gnn * grad_(Theta)(L_focal)  (lr = 0.003, AdamW)
19:          End For
20:      End For
21:      // Phase B: Rank-1 Projected Functional Coordinate Update
22:      If m < M then:
23:          Compute cross-entropy residuals across training nodes: r_i <- y_i - sigmoid(Logit_i)
24:          Fit scalar regression tree T_m: X_train -> r_i to minimize squared error loss (31 leaves, max_depth=6)
25:          Compute empirical covariance projection vector: d <- Normalize( E_{i}[ Z_0[i] * r_i ] ) in R^49
26:          Compute rank-1 coordinate displacement: Delta_Z_i <- clip( T_m(x_i) * d, -delta_R, delta_R )  (delta_R = 0.20)
27:          Update continuous representation matrix: Z_current <- Z_current + eta_boost * Delta_Z  (eta_boost = 0.10)
28:      End If
29:  End For
30:  Fit post-hoc isotonic probability calibrator C* on validation split
31:  Return F_base, {T_m}_(m=1)^(M-1), Theta_gnn, gate parameters (w_c, w_d, b_g), calibrator C*
```



## 3.5 Robust Inference & Inductive Cold-Start Mechanics

### 3.5.1 Monotonic Decision Blending Gate Formulation
To regulate reliance between the neural graph representation and the standalone tabular GBDT, predictions are combined in logit space through the Monotonic Decision Blending Gate $\beta_i \in [0, 1]$, driven by an entity connectivity quality signal $q_i^{\text{hist}} = \mathbb{I}(\mathrm{card\_mask}_i + \mathrm{dev\_mask}_i > 0)$ reflecting whether entity history exists in $G_{<t}$ within the lookback window $\tau_{\max}$.
$$\text{Logit}_{\text{blended}, i} = \beta_i \cdot \text{Logit}_{\text{GNN}, i} + (1.0 - \beta_i) \cdot \text{Logit}_{\text{GBDT}, i}^{(S_1)}$$
where:
$$\beta_i = \sigma\left( \text{softplus}(w_{\text{card}}) \cdot \mathrm{card\_mask}_i + \text{softplus}(w_{\text{dev}}) \cdot \mathrm{dev\_mask}_i + b_{\text{gate}} \right) \cdot \mathbb{I}(\text{connected}_i)$$
Here, $\mathrm{card\_mask}_i \in \{0, 1\}$ and $\mathrm{dev\_mask}_i \in \{0, 1\}$ denote binary indicators of active historical graph edges for card and device entities, respectively. The softplus parameterization $\text{softplus}(w) = \log(1 + e^w) > 0$ strictly enforces non-negativity: having active graph connections can only monotonically increase or preserve reliance on the neural graph component.

### 3.5.2 Exact Tabular Fallback for Isolated Nodes
For open-world cold-start transactions where both card and device entities are completely novel ($q_i^{\text{hist}} = 0$):
$$\mathrm{card\_mask}_i + \mathrm{dev\_mask}_i = 0 \implies \mathbb{I}(\text{connected}_i) = 0 \implies \beta_i \equiv 0$$
When $\beta_i = 0$, the blended logit simplifies identically to:
$$\text{Logit}_{\text{blended}, i} \equiv \text{Logit}_{\text{GBDT}, i}^{(S_1)}$$
This fallback ensures that for isolated or cold-start nodes, the framework bypasses continuous neural convolutions and relies exclusively on native GBDT decision trees.


## 3.6 Evaluation Metrics & Uncertainty Quantification

### 3.6.1 Exact Trapezoid-Free Average Precision (AP) and Calibration Ties
Because payment fraud exhibits severe class imbalance (3.413% in the holdout), standard ROC-AUC is inflated by the large volume of true negatives. In this dissertation, **Raw (Uncalibrated) Average Precision serves as the primary threshold-independent metric** for assessing model discriminative rank-ordering ability across the entire precision-recall continuum. In contrast, **post-hoc calibrated probabilities are strictly reserved for operational thresholding, alert queue allocation, and dollar-denominated financial loss calculation ($L_{\text{fin}}$)** where probability fidelity directly governs operational economics.

The primary ranking metric is non-interpolated, trapezoid-free Average Precision (AP):
$$\text{AP} = \sum_{k=1}^K (R_k - R_{k-1}) P_k$$
evaluated over all sorted prediction thresholds, where $P_k$ and $R_k$ denote precision and recall at cutoff $k$.

Importantly, when evaluating post-hoc isotonic probability calibration ($\tilde{p}_i = g_{\text{iso}}(p_i)$), isotonic regression fits a piecewise constant, monotonically non-decreasing step function. By mapping continuous raw predictions into discrete step intervals, isotonic regression produces tied prediction values for transactions that fall within the same step bin. When computing non-interpolated trapezoid-free AP over tied score intervals, standard numerical integration algorithms break ties conservatively by evaluating the lower precision bound of the bin. Consequently, post-hoc calibrated AP is empirically slightly lower than raw AP across all models (Table 4.1), reflecting tie-induced quantization rather than loss of discriminative ordering.

### 3.6.2 Operational Recall at Strict Low False-Positive Rates
To measure detection efficacy within realistic clearing parameters, models are evaluated at strict operational thresholds:
$$\text{Recall@0.1\%FPR} = \frac{\text{TP}_{\tau_{0.1\%}}}{\text{TP}_{\tau_{0.1\%}} + \text{FN}_{\tau_{0.1\%}}}, \quad \text{where } \text{FPR}(\tau_{0.1\%}) = 0.001$$
$$\text{Recall@1.0\%FPR} = \frac{\text{TP}_{\tau_{1.0\%}}}{\text{TP}_{\tau_{1.0\%}} + \text{FN}_{\tau_{1.0\%}}}, \quad \text{where } \text{FPR}(\tau_{1.0\%}) = 0.010$$

### 3.6.3 Realized Scenario-Weighted Financial Loss ($L_{\text{fin}}$) and Operational Regimes
To evaluate bottom-line financial impact, the evaluation suite computes realized scenario-weighted financial loss:
$$L_{\text{fin}} = \sum_{i=1}^N \left[ y_i (1 - a_i) c_{\text{miss}} A_i + a_i c_{\text{review}} + (1 - y_i) a_i c_{\text{friction}} + y_i a_i c_{\text{residual}} A_i \right]$$
where $A_i$ is the raw transaction amount, $y_i \in \{0, 1\}$ is the ground-truth fraud label, and $a_i \in \{0, 1\}$ is the binary operational alert decision. Financial loss is evaluated under two distinct operational paradigms:

1. **Primary Operational Regime: Fixed 1.0% Queue Capacity Constraint ($K = 1,151$ alerts):**
   In commercial bank clearing, fraud triage teams operate under a fixed daily review capacity. The triage engine alerts strictly on the top 1.0% highest-scoring transactions ($K = 1,151$ alerts across $N = 115,111$ holdout transactions). Under this regime, operational costs are unified with Regime 2 and parameterized by $c_{\text{miss}} = 1.0$, $c_{\text{review}} = \$5.00$, $c_{\text{friction}} = \$15.00$, and $c_{\text{residual}} = 0.0$. Because total alert volume is held constant at exactly 1,151 alerts for all models, review costs are fixed (\$5,755.00), and model differentiation is driven entirely by precision at capacity and the monetary exposure of intercepted fraud.

2. **Sensitivity Regime: Unconstrained Bayes Cost-Optimal Threshold Rule:**
   Where operations permit dynamic queue sizing, transactions are alerted when the model's posterior fraud probability exceeds the cost-optimal Bayes threshold:
   $$\tau_i^* = \frac{c_{\text{review}} + c_{\text{friction}}}{A_i(c_{\text{miss}} - c_{\text{residual}}) + c_{\text{review}} + c_{\text{friction}}} = \frac{20.0}{A_i + 15.0}$$
   under primary operational cost parameters $c_{\text{review}} = \$5.00$, $c_{\text{friction}} = \$15.00$, $c_{\text{miss}} = 1.0$, and $c_{\text{residual}} = 0.0$. This unconstrained threshold rule evaluates model calibration sensitivity, verifying whether probability miscalibration under Focal Loss triggers unmanageable alert volume surges and excessive false-positive friction.

### 3.6.4 Paired Stationary Block Bootstrap and FDR Control
Statistical uncertainty is quantified using a paired, chronological 24-hour stationary block bootstrap (86,400-second blocks) with 1,000 resamples. Block bootstrapping preserves the temporal autocorrelation and entity clustering inherent in payment streams. Confidence intervals are reported at 95% coverage ($[\text{LCB}_{0.95}, \text{UCB}_{0.95}]$). Multiplicity across confirmatory hypotheses is controlled using the Benjamini-Hochberg False Discovery Rate procedure at nominal level $\alpha = 0.05$ (evaluating test-sample uncertainty conditional on trained models).

