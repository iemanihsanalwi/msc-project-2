# Chapter 1: Introduction

## 1.1 Background of Study

Payment card clearing networks provide the core operational infrastructure of modern digital commerce. Driven by card-not-present (CNP) e-commerce gateways, instant mobile payment rails, and open-banking APIs, total transaction clearing volume reached tens of trillions of US dollars in 2023. This transition toward faster financial settlement has coincided with an increase in the volume and technical sophistication of payment fraud. Worldwide retail payment card fraud losses exceeded USD 33.83 billion in 2023 and are projected to surpass USD 40.0 billion by 2028 (Nilson Report, 2024).

The operational reality of financial clearing imposes strict constraints on automated risk detection systems. First, card clearing exhibits severe statistical class asymmetry: unauthorized fraudulent transactions represent a small fraction of total volume, typically between 0.1% and 3.5% across commercial retail streams. In the benchmark IEEE-CIS payment continuum examined in this dissertation, the empirical fraud prevalence is 3.4990% ($N = 20,663$ positive cases out of 590,540 total labeled observations). 

Second, production fraud engines must generate risk evaluations under sub-100-millisecond latency budgets during in-flight authorization requests. Models must ingest transaction telemetry, execute inference, and return a scoring decision before the transaction times out at the point of sale terminal. 

Third, fraud detection operates under delayed ground-truth feedback. Unlike synthetic machine learning benchmarks where target labels $y_i \in \{0, 1\}$ are treated as immediately available, chargebacks and cardholder fraud reports materialize only after administrative reconciliation and dispute verification, which introduces an operational latency of 30 to 90 days (Dal Pozzolo et al., 2018). Streaming risk engines must therefore operate over rolling horizons where recent transaction records lack confirmed labels.


+–––––––––––––––––––––––––––––––––––––––––––––––––-+
|                        THE STRUCTURAL DUALITY OF TRANSACTION CLEARING DATA                        |
+–––––––––––––––––––––––––––––––––––––––––––––––––-+
|                                                                                                   |
|  [Tabular Feature Dimension]                               [Relational Graph Dimension]           |
|  - Continuous Measurements:                                - Multi-Entity Bipartite Topology:     |
|    TransactionAmt, Time-deltas (D-series),                   Card proxies (card1-6 + addr),       |
|    Distance proxies, C-series counts.                        Device digital telemetry (id_30-38). |
|  - Missingness Profile:                                    - Topological Interaction:             |
|    Median feature missingness of 47.29%;                     Shared payment instruments,          |
|    214 of 432 features exhibit >50% NaN.                     IP subnet coordination,              |
|  - Inductive Champion:                                       Card-testing burst velocity.         |
|    Gradient Boosted Decision Trees (GBDT)                  - Inductive Champion:                  |
|    - Monotonic invariance to scaling.                        Graph Neural Networks (GNN)          |
|    - Default missingness split routing.                      - Multi-hop message passing.         |
|    - Precise orthogonal boundary cuts.                       - Structural relational discovery.   |
|                                                                                                   |
+–––––––––––––––––––––––––––––––––––––––––––––––––-+


Payment records inherently present two distinct data structures:
1. The Irregular Tabular Attribute Space: Each transaction arriving at an authorization switch contains continuous measurements (such as transaction amount and temporal intervals since prior activity), discrete categorical identifiers (such as merchant category codes, issuing banks, and billing regions), and substantial missing values. In the IEEE-CIS continuum, the median missingness across features is 47.29%, with 214 of 432 raw attributes missing more than 50% of their entries. Gradient Boosted Decision Trees (GBDTs), such as LightGBM, XGBoost, and CatBoost, serve as the standard baseline for this space. GBDTs possess inductive biases suited to tabular manifolds: invariance to monotonic feature scaling, native missing value routing, and the capacity to isolate axis-aligned hyper-rectangular thresholds ($x_{\text{amt}} > \theta$) with minimal parameter complexity (Grinsztajn et al., 2022).
2. The Multi-Hop Relational Interaction Space: Concurrently, transactions do not occur as independent, identically distributed (i.i.d.) observations. Fraud syndicates conduct identity attacks, automated credential stuffing, and card-testing cascades across distributed accounts that share physical devices, IP subnets, or billing addresses. When transactions link to these shared identifiers, they form a heterogeneous bipartite graph. Graph Neural Networks (GNNs), such as Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), and Relational Graph Convolutional Networks (RGCN), can map these relational topologies by propagating risk representations across multi-hop neighborhoods (Hamilton et al., 2017; Wang et al., 2019).

Combining decision trees and graph neural networks into a unified, production-viable framework under streaming, latency-constrained, and non-transductive financial clearing conditions remains a challenging engineering problem.


## 1.2 Problem Statement & Research Gaps

The deployment of deep graph architectures in financial fraud detection faces four primary architectural, methodological, and economic challenges in the literature:


+–––––––––––––––––––––––––––––––––––––––––––––––––-+
|                            FOUR SYSTEMIC VULNERABILITIES IN GNN FRAUD SYSTEMS                     |
+–––––––––––––––––––––––––––––––––––––––––––––––––-+
|  1. THE ARCHITECTURAL OPTIMIZATION BLIND SPOT:                                                    |
|     - Decoupled Res-GNN: Static GBDT features passed to GNN; zero gradient feedback to trees.     |
|     - Differentiable Trees (NODE): Soft splits destroy sharp thresholds, exploding variance.      |
|     - Foundational BGNN: Formulated solely for static, transductive single-table benchmarks.      |
|                                                                                                   |
|  2. THE TEMPORAL-CAUSAL REALISM DEFICIT:                                                          |
|     - Transductive Look-Ahead Leakage: Random K-fold cross-validation uses future graph edges to  |
|       predict the past, reporting artificially inflated AP (>0.90) that collapses in production.  |
|     - Absence of Time-Causal Adjacency (G_{<t}) with strict backward-only lookback boundaries.    |
|                                                                                                   |
|  3. THE OPEN-WORLD COLD-START DILEMMA:                                                            |
|     - High Entity Churn: ~30.6% of streaming transactions feature completely novel cards/devices. |
|     - Empty Receptive Fields: Unseen nodes cause continuous multi-hop convolutions to collapse.   |
|     - Oversmoothing: Arithmetic mean aggregation acts as a low-pass filter on anomalous spikes.   |
|                                                                                                   |
|  4. LOSS & ECONOMIC COST MISALIGNMENT:                                                            |
|     - Standard Symmetric Loss: Unweighted cross-entropy ignores class imbalance and fraud scale.  |
|     - Production Asymmetry: Ignores human investigator triage capacity (<=1.0%) and the severe    |
|       monetary trade-off between false negatives (stolen exposure) and false positives (friction).|
+–––––––––––––––––––––––––––––––––––––––––––––––––-+


### 1.2.1 The Architectural Optimization Blind Spot
Prior hybrid architectures combining GBDTs and GNNs have predominantly relied on decoupled two-stage pipelines, termed Res-GNN models (Ivanov & Prokhorenkova, 2021). In these systems, a GBDT is trained independently on tabular features, and its fixed predictions or intermediate embeddings are concatenated into the node feature matrix of a downstream GNN. This design separates the two models: the decision tree ensemble receives no structural gradient feedback from the graph, keeping tabular representations fixed in non-relational space.

Conversely, differentiable decision tree approximations, including Neural Oblivious Decision Ensembles (NODE) and soft probabilistic routing trees, replace discrete step splits with continuous sigmoidal approximations (Popov et al., 2020; Hazimeh et al., 2020). Softening the split boundaries reduces the orthogonal partitioning precision that benefits GBDTs on tabular data, increasing model variance under severe class imbalance.

The Boost then Convolve (BGNN) framework by Ivanov and Prokhorenkova (2021) addressed this separation on static single-table benchmarks through alternating functional-gradient optimization. However, BGNN has not previously been adapted to or evaluated within dynamic multi-entity transaction streams that feature heterogeneous relations, continuous temporal drift, and explicit operational fallback requirements.

### 1.2.2 The Temporal-Causal Realism Deficit
A key methodological issue in the academic GNN fraud detection literature is the reliance on transductive, static, or randomly partitioned evaluation graphs (Akli et al., 2026; Van Belle et al., 2023). In production streams, time flows in one direction: an automated scoring engine operating at timestamp $t$ cannot observe transactions, entities, or topological connections that occur at $t' > t$.

Conventional academic benchmarks frequently construct undirected edges across the full transaction record and partition data using random $K$-fold cross-validation. This setup introduces temporal look-ahead leakage: early transactions aggregate representations from subsequent events on the same card, while feature scalers and dimensionality reductions are fitted over future data. As a result, published studies often report elevated performance metrics (such as Average Precision $> 0.90$ and ROC-AUC $> 0.95$) that degrade significantly when evaluated on chronologically quarantined data ($G_{<t}$).

### 1.2.3 The Open-World Cold-Start Dilemma
Standard graph convolutional formulations assume closed-world graphs where evaluation nodes reside in dense neighborhoods. Commercial payment clearing, however, is an open-world inductive stream characterized by continuous entity turnover: cardholders open new credit accounts, replace cards, switch devices, and transact at novel merchants.

In the sealed out-of-time evaluation split of the IEEE-CIS benchmark, 30.6% of evaluation transactions involve payment cards that never appeared in the historical training data. For an isolated or newly arrived node $v_{\text{new}}$, the historical neighborhood is empty:
$$\mathcal{N}(v_{\text{new}}) = \emptyset \implies \sum_{u \in \mathcal{N}(v_{\text{new}})} \mathbf{h}_u = \mathbf{0}$$
When multi-hop message passing is applied to these sparse topologies, continuous node projections provide little discriminative context. Furthermore, unweighted arithmetic mean aggregation across historical predecessors acts as a low-pass filter, smoothing away the localized monetary and velocity spikes that characterize fraud and lowering ranking performance relative to basic tabular models.

### 1.2.4 The Loss & Economic Cost Misalignment
In production fraud operations, models do not operate under symmetric cross-entropy loss objectives, nor are they evaluated solely on global receiver operating characteristic curves (ROC-AUC). First, standard log-loss optimization under a 96.5% negative class majority causes gradients to be dominated by routine transactions.

Second, financial institutions operate under investigation queue capacity constraints: manual review teams can typically inspect at most 1.0% of daily transaction volume. Models that achieve high theoretical recall only at loose false-positive rates (such as FPR $\ge 5.0\%$) cannot be deployed effectively.

Third, standard loss formulations treat all misclassifications equally, ignoring the monetary asymmetry between undetected fraud (which results in the direct loss of transaction amount $A_i$), administrative triage costs ($c_{\text{review}} = \$5.00$), and customer friction from false positives ($c_{\text{friction}} = \$15.00$). Existing hybrid graph-tabular frameworks rarely evaluate realized, scenario-weighted economic cost metrics ($L_{\text{fin}}$).


## 1.3 Research Objectives

To resolve these systemic deficiencies, this dissertation develops, formalizes, and forensically audits an adapted, time-causal Boost then Convolve architecture tailored to payment clearing streams. The overarching investigation is structured around five explicit Research Objectives (ROs):

* **RO1:** To design, implement, and audit a strictly time-causal, non-leaking dynamic heterogeneous graph formulation ($G_{<t}$) that captures multi-entity payment relationships (payment cards, digital devices, IP networks) without violating temporal event ordering or backward lookback boundaries.
* **RO2:** To adapt the Boost then Convolve (BGNN) alternating optimization framework for streaming fraud detection by formulating continuous multidimensional embeddings ($Z_0 \in \mathbb{R}^{d_{\text{emb}}}$), relation-aware temporal message passing, and multi-output functional-gradient coordinate updates ($R_i^m$).
* **RO3:** To formulate robust inductive inference mechanisms, specifically an exact tabular fallback and a monotonic decision blending gate ($\beta_i$) driven by localized connectivity quality signals ($q_i^{\text{hist}}$), that dynamically arbitrate between neural graph and tabular tree representations to protect cold-start nodes.
* **RO4:** To execute an audit-hardened, chronologically quarantined 4-stage experimental benchmarking ladder on the IEEE-CIS dataset, comparing Standalone Tabular GBDTs, Graph-Augmented GBDTs, Frozen Res-GNNs, and Jointly Optimized BGNNs across a sealed Out-of-Time (OOT) holdout.
* **RO5:** To conduct an exhaustive forensic failure mode analysis evaluating whether joint graph–tabular coupling maintains statistical and economic superiority under extreme operational constraints (FPR $\le 0.1\%$, review capacity $\le 1.0\%$, and realized financial loss $L_{\text{fin}}$), establishing the theoretical boundaries of deep graph learning in financial risk streams.


## 1.4 Research Questions and Pre-Specified Hypotheses

The experimental architecture is formulated around five core Research Questions (RQs) and a pre-specified confirmatory hypothesis testing hierarchy:

* **RQ1 (Tabular vs. Relational Base Value):** To what extent does the incorporation of time-aware relational graph structures improve Average Precision (AP), low false-positive rate recall (Recall@0.1%FPR), and realized financial loss ($L_{\text{fin}}$) over a tuned standalone tabular GBDT?
* **RQ2 (Value of Isolated Graph Propagation):** Does a decoupled, two-stage Res-GNN pipeline operating on fixed tabular embeddings outperform a feature-augmented GBDT when graph message passing is evaluated in isolation without joint optimization?
* **RQ3 (Value of Functional-Gradient Coupling):** Does an alternating, jointly trained BGNN model outperform the decoupled Res-GNN pipeline, demonstrating measurable ranking and economic gains attributable strictly to functional-gradient coordination ($R_i^m$)?
* **RQ4 (Cold-Start & Temporal Robustness):** Do performance deltas across the experimental ladder remain preserved when evaluated across open-world cold-start entity distributions (unseen cards/devices) and chronological time slices ($t_1\text{–}t_4$) under strict 1.0% alert capacity limits?
* **RQ5 (Component-Level Failure Attribution):** Which structural mechanisms, including inductive bias mismatch, unweighted aggregation smoothing, continuous covariate drift, and optimization interference, drive the empirical convergence or degradation of joint graph-tabular models under temporal distribution shifts?

### Pre-Specified Confirmatory Hypothesis Testing Protocol
To ensure academic rigor and preclude post-hoc cherry-picking, confirmatory hypotheses were pre-specified with defined decision boundaries evaluated on the sealed holdout:
1. **Relational Augmentation Hypothesis ($H_1$):** Stage 2 (Graph-Augmented GBDT) must demonstrate statistically significant superiority over Stage 1 (Standalone Tabular GBDT) by at least the pre-specified materiality boundary ($\Delta \ge +0.005$ AP) with a strictly positive lower confidence bound:
   $$\Delta_{2-1} = \text{AP}_2 - \text{AP}_1 \ge +0.005, \quad \text{with } \text{LCB}_{0.95}(\Delta_{2-1}) > 0$$
2. **Structural Decoupling Hypothesis ($H_2$):** Stage 3 (Frozen Res-GNN) must achieve superiority over Stage 2 (Graph-Augmented GBDT) to demonstrate the value of automated continuous message passing over explicit tabular feature engineering:
   $$\Delta_{3-2} = \text{AP}_3 - \text{AP}_2 \ge +0.005, \quad \text{with } p_{\text{adj}} \le 0.05$$
3. **Primary Rank-1 Coupling Hypothesis ($H_3$):** Stage 4 (Adapted Rank-1 Projected BGNN) must outperform Stage 3 (Frozen Res-GNN) in out-of-time Average Precision by at least the materiality threshold ($\Delta \ge +0.005$ AP):
   $$\Delta_{4-3} = \text{AP}_4 - \text{AP}_3 \ge +0.005, \quad \text{with } \text{LCB}_{0.95}(\Delta_{4-3}) > 0$$
4. **Macro System Uplift Hypothesis ($H_4$):** Stage 4 must outperform the primary industrial benchmark, Stage 1 (Standalone Tabular GBDT), in Average Precision:
   $$\Delta_{4-1} = \text{AP}_4 - \text{AP}_1 \ge +0.005, \quad \text{with } p_{\text{adj}} \le 0.05$$
5. **Operational Queue Loss Contrast ($H_{3\text{c}}$):** Under the primary 1.0% operational queue capacity constraint, Stage 2 must achieve a statistically significant reduction in realized financial loss relative to Stage 1:
   $$\Delta L_{\text{queue}}^{(2-1)} = L_{\text{fin}}^{(1)} - L_{\text{fin}}^{(2)} > 0, \quad \text{with } \text{LCB}_{0.95}(\Delta L) > 0$$
6. **Multiplicity Control:** All paired hypotheses are evaluated across 1,000 chronological block-bootstrap resamples under Benjamini–Hochberg False Discovery Rate (FDR) control at nominal level $\alpha = 0.05$ (evaluating test-sample uncertainty conditional on trained models).


## 1.5 Scope, Delimitations, & Computational Boundaries

To ensure empirical reproducibility and operational realism, the scope of this study is bounded by the following formal design criteria:


+–––––––––––––––––––––––––––––––––––––––––––––––––-+
|                        EXPERIMENTAL SCOPE AND COMPUTATIONAL BOUNDARIES                            |
+–––––––––––––––––––––––––––––––––––––––––––––––––-+
|  - DATASET HORIZON:                                                                               |
|    IEEE-CIS labeled dataset (train_transaction.csv and train_identity.csv).                       |
|    Total observations: N = 590,540; Time continuum: TransactionDT in [86,400, 15,811,131] (182d). |
|                                                                                                   |
|  - COMPLETE EXCLUSION OF KAGGLE TEST HOLDOUT:                                                     |
|    The unlabelled competition test set (test_transaction.csv, N = 506,691) is completely         |
|    banned because ground-truth chargeback labels are proprietary and unverified.                  |
|                                                                                                   |
|  - 5-BLOCK CHRONOLOGICAL PARTITIONING:                                                            |
|    Expanding development blocks B1-B4 (N = 472,432 rows, Days 0.00 to 140.12).                      |
|    Quarantined sealed Out-of-Time holdout Block B5 (N = 118,108 rows total, Days 140.12 to 182.00; 115,111 evaluated post-embargo rows).       |
|                                                                                                   |
|  - BOUNDED TEMPORAL LOOKBACK HORIZON:                                                             |
|    Edge insertion strictly bounded: t_prev < t_curr with lookback cap K_lookback = 30 days        |
|    (2,592,000 seconds). Interactions older than 30 days are aged out of active message passing.  |
|                                                                                                   |
|  - STANDARDIZED COMPUTATIONAL FAIRNESS:                                                           |
|    Executed on single-GPU (NVIDIA CUDA, PyTorch 2.6, LightGBM 4.6).                               |
|    Strictly matched tree budgets (256 leaves, eta = 0.03) and GNN budgets (dim = 64, lr = 0.003). |
+–––––––––––––––––––––––––––––––––––––––––––––––––-+


1. **Exclusion of the Unlabeled Competition Holdout:** The unlabeled Kaggle competition holdout (test_transaction.csv, $N = 506,691$) is excluded entirely. Because competition test labels are private and unverified, academic studies that submit predictions to Kaggle leaderboard proxies cannot compute calibration reliability diagrams, subgroup cohort breakdowns, or financial loss metrics. All evaluation in this dissertation is conducted strictly on labeled records with verified chargeback outcomes.
2. **Chronological 5-Block Design:** Data is partitioned into five non-overlapping contiguous chronological blocks of 118,108 transactions each ($\approx 36.4$ days per block). Blocks 1 through 4 serve as the expanding development and validation continuum. Block 5 (the final ~20% of the dataset, covering Days 140.12 to 182.00 (spanning 41.88 days, approximately 42 days)) represents the strictly sealed Out-of-Time (OOT) test set, quarantined until all preprocessing pipelines, graph extraction rules, hyperparameter spaces, and calibration parameters were frozen.
3. **Bounded Lookback Horizon ($K_{\text{lookback}}$):** To mirror production memory constraints and prevent runaway graph expansion, edge creation is strictly bounded: $t_u < t_i$ and $t_i - t_u \le 30 \text{ days} \ (2,592,000\text{ seconds})$. Historical interactions occurring beyond 30 days are purged from active message passing.
4. **Computational Fairness:** All deep graph architectures and tree ensembles are trained within standardized CUDA environments under strict resource equality, enforcing matched tree budgets (500 maximum rounds) and neural epochs across all experimental arms.


## 1.6 Academic & Practical Significance

The contributions of this dissertation bridge fundamental machine learning theory and applied quantitative risk management:

### 1.6.1 Methodological Value of Alternating Functional Coupling
From a machine learning perspective, this research delivers the first rigorous formulation of the Boost then Convolve (BGNN) paradigm adapted to dynamic streaming graphs with backward-causal edge constraints. It demonstrates how non-differentiable decision tree ensembles can be formally coupled with continuous graph neural networks via multi-output residual displacements ($R_i^m$), evaluating whether an adapted rank-1 projected coordinate update can transmit structural topological signal into continuous coordinate spaces ($Z_0$) without sacrificing native tree mechanics.

### 1.6.2 Auditable Inductive Safeguards for Financial Risk Systems
For enterprise risk engineering, the research introduces an auditable framework equipped with a monotonic decision blending gate ($\beta_i$). By tying the gate directly to localized graph connectivity telemetry ($q_i^{\text{hist}}$) and enforcing monotonic constraints, the architecture provides an exact mathematical fallback to tabular GBDTs ($\beta_i \to 0$) when transaction nodes arrive as isolated or cold-start entities. This eliminates the instability of standard deep GNNs that generate unconstrained predictions in sparse graph regimes.

### 1.6.3 Methodological Value of Negative Results
This dissertation provides an empirical evaluation of negative results in applied machine learning. By demonstrating that the alternating BGNN architecture fails to outperform feature-augmented GBDTs under out-of-time evaluation ($\Delta_{4-1} = -0.0181$ AP), this work examines the divergence between reported transductive benchmarks and temporal production reality. Documenting the specific failure modes of graph architectures under temporal distribution shift offers practical guidance for financial institutions evaluating graph neural network infrastructure.


## 1.7 Thesis Organization Roadmap

The remainder of this dissertation is structured into five comprehensive chapters:


+–––––––––––––––––––––––––––––––––––––––––––––––––-+
|                                  THESIS CHAPTER ROADMAP                                           |
+–––––––––––––––––––––––––––––––––––––––––––––––––-+
|                                                                                                   |
|  [Chapter 1: Introduction]                                                                        |
|  Background, Financial Clearing Duality, Problem Statement, Objectives, RQs, and Scope.           |
|                                         |                                                         |
|                                         v                                                         |
|  [Chapter 2: Literature Review and Theoretical Foundations]                                       |
|  Tabular vs. Graph Paradigms, Hybrid Architectures, Streaming Vulnerabilities (Leakage, Cold-     |
|  Start, Delayed Labels), Table 2.1 Conceptual Matrix, and Systematic Gap Synthesis.              |
|                                         |                                                         |
|                                         v                                                         |
|  [Chapter 3: Research Methodology and System Formulation]                                        |
|  Time-Causal Graph G_<t, Continuous Embeddings Z_0, Relation-Aware Message Passing,               |
|  Functional Coupling Mechanics R_i^m, Monotonic Gate beta_i, 4-Stage Ladder, L_fin Metric.       |
|                                         |                                                         |
|                                         v                                                         |
|  [Chapter 4: Empirical Results and Forensic Failure Analysis]                                     |
|  Sealed Block 5 Benchmark Results, Bootstrap Hypothesis Tests (H1-H4), Primary Coupling Uplift    |
|  Verification, and Exhaustive Forensic Audit of Four Failure Modes.                               |
|                                         |                                                         |
|                                         v                                                         |
|  [Chapter 5: Conclusion, Recommendations, and Defense Synthesis]                                  |
|  Theoretical Synthesis, Academic/Institutional Recommendations, Epistemological Defense of Null   |
|  Results, and Concrete Future Research Vectors.                                                   |
|                                                                                                   |
+–––––––––––––––––––––––––––––––––––––––––––––––––-+


* **Chapter 2 (Literature Review and Theoretical Foundations):** Examines the mathematical properties of tabular decision trees and graph neural networks, surveys existing hybrid models (Res-GNN, NODE, BGNN), details streaming vulnerabilities (temporal leakage, delayed chargeback feedback, cold-start turnover), and constructs the conceptual matrix justifying the experimental ladder.
* **Chapter 3 (Research Methodology and System Formulation):** Formalizes the complete mathematical framework, including the time-causal graph formulation $G_{<t}$, dimensional accounting of $Z_0$, relation-aware message passing, Focal Loss optimization, functional residual displacement updates, the monotonic blending gate $\beta_i$, algorithmic pseudocode, and the scenario-weighted financial loss metric $L_{\text{fin}}$.
* **Chapter 4 (Empirical Results and Forensic Failure Analysis):** Reports the complete empirical findings from the sealed Block 5 holdout ($N = 118,108$), evaluates pre-specified bootstrap hypothesis tests ($H_1\text{–}H_4$), validates the primary coupling uplift ($H_3$), and conducts an exhaustive forensic audit dissecting the failure modes of Stages 3 and 4 (ratio advantage, cold-start churn, continuous covariate drift, manifold displacement corruption, and probability miscalibration).
* **Chapter 5 (Conclusion, Recommendations, and Defense Synthesis):** Synthesizes the core theoretical insights, provides actionable recommendations for banking risk teams, articulates the epistemological defense of negative empirical results, and charts future research vectors for differentiable ratio operators and true logit-space functional boosting.
