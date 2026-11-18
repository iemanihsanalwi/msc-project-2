# Chapter 5: Discussion, Conclusion, & Recommendations

## 5.1 Methodological Defense of the Empirical Null Result

### 5.1.1 Reconciling Internal Coupling Uplift with Macro System Degradation
A central finding of this dissertation is the empirical divergence between explicit feature engineering and alternating functional coordinate updates:
1. Relational Feature Uplift ($H_1$ Confirmed) vs. Rank-1 Coupling Deficit ($H_3$ Not Supported): While explicit relational feature engineering in Stage 2 delivered an uplift over the tabular baseline ($\Delta_{2-1} = +0.0438$ AP, 95% bootstrap CI: $[+0.0377, +0.0504]$, $p_{\text{adj}} = 0.0080$, confirming $H_1$), the adapted rank-1 functional coordinate updates in Stage 4 did not improve upon the frozen Res-GNN ($\Delta_{4-3} = -0.0138$ AP, one-sided $p > 0.999$, rejecting $H_3$).
2. Macro System Uplift Hypothesis ($H_4$ / RQ1) in Ranking AP: In Average Precision, Stage 4 scored below standalone Tabular LightGBM ($\Delta_{4-1} = -0.0181$ AP, $p_{\text{adj}} > 0.999$). Under the primary 1.0% operational queue constraint, Stage 4 achieved a modest dollar loss reduction relative to Stage 1 (\$501,510.90 vs. \$507,989.81, saving \$6,478.91), but did not reach the performance of Stage 2 (\$495,307.56). Under unconstrained Bayes thresholding, raw Focal Loss alert inflation resulted in \$534,715.00 in review and friction costs, raising uncalibrated loss to \$629,411.78.

This outcome constitutes an informative empirical null result: adapting BGNN via rank-1 projected functional coordinate updates did not recover the performance gap relative to Stage 2 ($\Delta_{4-2} = -0.0618$ AP), showing that alternating feature-space perturbations face practical friction on streaming financial tabular manifolds.


+–––––––––––––––––––––––––––––––––––––––––––––––––-+
| Table 5.1: Epistemological Contrast Between Reported GNN Literature and Causal Fraud Reality       |
+–––––––––––––––––––––––––––––––––––––––––––––––––-+
| Research Dimension        | Prevailing GNN Fraud Literature       | This Dissertation (Time-Causal Realism)|
+–––––––––––––-+–––––––––––––––––––-+–––––––––––––––––––-+
| Graph Adjacency           | Static, Transductive (All V and E)    | Time-Causal Directed Multigraph G_<t  |
| Future Information Flow   | Look-ahead leakage from future edges  | Strictly score-before-insert G_i      |
| Data Partitioning         | Random / Stratified Cross-Validation  | Chronological Block 5 Sealed Holdout  |
| Supervision Policy        | Instantaneous Oracle Labels           | Conservative Block-Delayed Embargo    |
| Entity Overlap            | High Transductive Node Overlap        | Open-World 30.6% Entity Turnover      |
| Evaluation Metric         | ROC-AUC / Macro F1 (Artificially High)| Non-interpolated AP & Ultra-low FPR   |
| Economic Grounding        | Unweighted Symmetric Cross-Entropy    | Scenario-Weighted Financial Loss L_fin|
| Reported Outcome          | Massive SOTA Uplift (+10% to +30%)    | Macro Null Result (Delta_4-1 = -0.018) |
+–––––––––––––––––––––––––––––––––––––––––––––––––-+


### 5.1.2 Re-evaluating Performance Claims in Prevailing GNN Literature
Surveys of graph neural network literature for financial fraud detection document widespread reports of superior performance over tree-based baselines (such as Wang et al., 2019; Dou et al., 2020; Liu et al., 2021). However, as summarized in Table 5.1, these reported gains frequently stem from methodological designs that do not transfer to production conditions:

1. Transductive Look-Ahead Contamination: Many academic graph benchmarks construct the adjacency matrix across the full dataset prior to splitting: $\mathcal{V} = \mathcal{V}_{\text{train}} \cup \mathcal{V}_{\text{test}}$. During message passing, evaluation representations aggregate edge embeddings from subsequent transactions ($t > t_i$). When graph edges are restricted strictly to historical predecessors ($G_{<t}$), message-passing layers lose access to future transaction patterns.
2. Random Cross-Validation Leakage: Evaluating transaction streams with random or stratified $k$-fold cross-validation allows training folds to observe events occurring chronologically after evaluation events. Because card fraud occurs in rapid velocity bursts, random splitting enables models to interpolate between known fraud events on the same card rather than predicting future risk.
3. Unmasked Bidirectional Propagation: Modeling transaction networks as undirected graphs allows information from confirmed chargebacks to propagate across temporal boundaries. In production, chargeback reporting delays (30 to 90 days) prevent immediate label awareness, and legitimate cardholders sharing devices with suspicious users do not exhibit symmetric risk.
4. Baseline Feature Completeness: Published studies often benchmark complex GNN architectures against basic decision tree configurations without historical relational features. Furthermore, reporting global ROC-AUC under extreme class imbalance (roughly 3.5% fraud) can obscure low recall at operational false-positive rates ($\le 0.1\%$).

By enforcing score-before-insert semantics on $G_{<t}$, a 1-day clearing embargo, and sealed out-of-time Block 5 evaluation, this dissertation evaluates models under streaming constraints. This empirical null result establishes baseline limits for deep graph representations in non-transductive financial systems.

## 5.2 Theoretical Implications for Joint Graph–Tabular Learning

The empirical findings carry theoretical implications for research at the intersection of tabular and graph representation learning:

### 5.2.1 Functional Gradients vs. Spatial Neural Smoothing
The fundamental theoretical divergence between GBDTs and GNNs centers on how each model family constructs decision boundaries across irregular manifolds:
* **The GBDT Manifold:** Decision trees perform recursive, axis-aligned partitioning via greedy histogram splitting. In financial tabular data, fraud signatures reside in narrow, highly localized anomaly sub-spaces characterized by extreme, non-linear ratio deviations from historical entity baselines:
  $$\mathrm{amt\_ratio}_i = \frac{x_{\text{amt}, i}}{\bar{x}_{\text{amt}, \text{hist}(i)} + 1.0} > 4.5 \quad \land \quad \Delta t_i < 120\text{ s}$$
  Because tree splitting is invariant to monotonic transformations and handles missingness via dedicated gradient paths, GBDTs isolate sharp, discontinuous coordinate anomalies with high precision.
* **The Spatial Convolution Manifold:** In contrast, spatial graph convolutions rely on smooth, continuous message-passing operators:
  $$\bar{\mathbf{z}}_{\mathcal{N}(i)} = \sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{d_i d_j}} \mathbf{z}_j$$
  On financial transaction graphs that exhibit heterophily and class imbalance, spatial aggregation functions as a low-pass filter. Averaging latent embeddings across predecessor transactions dilutes the extreme monetary and velocity spikes that define fraudulent activity. Feed-forward affine projections ($\mathbf{W}\mathbf{x} + \mathbf{b}$) cannot natively evaluate division or scale-invariant multiplicative ratios without specialized product layers.

### 5.2.2 The Spatial Aggregation Regularizer Hypothesis
In homophilic domain benchmarks (such as citation networks or academic co-authorship graphs), neighbor smoothing acts as a beneficial inductive bias: connected nodes genuinely share categorical classes, and spatial averaging denoises measurement error.

In financial transaction clearing, spatial neural aggregation functions as a regularizer that smooths localized anomaly signals. This is supported by the identical-network ablation ($H_{2\text{a}}$ and $H_{2\text{b}}$): zeroing the adjacency matrix entirely ($S_3$ vs. $S_{3\text{n}}$) produced comparable Average Precision ($\Delta = +0.0009$ AP, $p_{\text{adj}} = 0.723$). Spatial message passing over $G_{<t}$ provided no measurable predictive gain beyond the continuous MLP projection of the tabular coordinates.

### 5.2.3 Continuous Coordinate Bottlenecks on Heterogeneous Tables
The failure of the joint BGNN to outperform standalone LightGBM highlights the inherent limitations of mapping heterogeneous tabular data into low-dimensional continuous spaces ($\mathbf{Z}_0 \in \mathbb{R}^{N \times 49}$). Compressing 339 Vesta features into 15 PCA components discarded rare, non-linear fraud interactions while retaining dominant legitimate variance. Furthermore, alternating functional boosting updates applied via coordinate displacements ($\mathbf{R}_i^m$) mutated the input space between outer rounds, inducing severe gradient oscillations during neural backpropagation. Canonical BGNN (Ivanov & Prokhorenkova, 2021) fits d_emb independent trees per round directly to feature-level gradients. Mutating the continuous coordinate manifold between outer rounds introduces non-stationary perturbations that destabilize neural gradient descent on heterogeneous tabular manifolds.


## 5.3 Practical and Economic Implications for Payment Clearing Systems

The empirical results provide decisive guidance for chief risk officers, quantitative fraud architects, and production machine learning engineers designing real-time payment clearing infrastructure.


+–––––––––––––––––––––––––––––––––––––––––––––––––-+
| Table 5.2: Operational and Economic Comparison Across Evaluated Architectural Paradigms           |
+–––––––––––––-+––––––––––––+–––––––––––-+–––––––––––-+
| Metric / Dimension        | Stage 1: Tabular Light | Stage 2: Graph GBDT   | Stage 4: Joint BGNN   |
+–––––––––––––-+––––––––––––+–––––––––––-+–––––––––––-+
| Out-of-Time AP (Block 5)  | 0.4814                 | 0.5252 (Cal.: 0.5088) | 0.4634 (Cal.: 0.4478) |
| ROC-AUC                   | 0.8731                 | 0.9053                | 0.8756                |
| Realized Dollar Loss L_fin| $365,728.99            | $345,402.31 [CHAMP]   | $629,411.78           |
| Calibrated Dollar Loss    | $367,011.13            | $348,690.01 [CHAMP]   | $373,410.83           |
| Recall @ 0.1% FPR         | 24.51%                 | 24.97%                | 19.47%                |
| Recall @ 1.0% FPR         | 39.83%                 | 43.90%                | 39.53%                |
| Inference Latency / Tx    | < 5 ms (CPU)           | < 8 ms (CPU Rolling)  | 68 ms (Alternating)   |
| Production Feasibility    | Industrial Standard    | Production Champion   | Negative ROI vs Base  |
+–––––––––––––-+––––––––––––+–––––––––––-+–––––––––––-+
| Out-of-Time AP (Block 5)  | 0.4814                 | 0.5252 (Cal.: 0.5088) | 0.4634 (Cal.: 0.4478)|
| Recall @ 0.1% FPR Cap     | 24.51%                 | 24.97%                | 19.47%               |
| Realized Dollar Loss L_fin| $365,728.99            | $345,402.31 [CHAMP]   | $629,411.78        |
| Inference Latency Cap     | < 2.5 ms (CPU)         | < 4.0 ms (In-Memory)  | > 45.0 ms (GPU/Subg) |
| Cold-Start Robustness     | Exceptional (Invar.)   | Highest (Global Prior)| Severe Degradation   |
| Regulatory Explainability | Exact TreeSHAP Audits  | Exact TreeSHAP Audits | Opaque Neural Latents|
| Infrastructure Overhead   | Lightweight C++ Binary | Redis / Feature Store | Distributed GPU Farm |
+–––––––––––––––––––––––––––––––––––––––––––––––––-+


### 5.3.1 Operational Viability of Graph-Augmented GBDT
From an operational and financial perspective, Stage 2 (Graph-Augmented GBDT) demonstrates the strongest overall profile:
1. Lowest Realized Financial Loss: Stage 2 achieves the lowest financial loss across all evaluated pipelines at \$345,402.31, delivering a savings of \$20,326.68 relative to Stage 1. Combining base tabular signals with 19 backward-rolling features maximizes fraud interception within the 1.0% review queue capacity ($\mathbf{\text{Precision@1.0\%Cap} = 89.14\%}$).
2. Inference Latency Efficiency: While deep GNN architectures require sub-graph extraction, neighbor sampling, and multi-layer neural forward passes (incurring latencies between 40 and 60 milliseconds), Stage 2 executes within standard decision tree runtimes in under 4 milliseconds. Predecessor counts and running averages can be maintained via in-memory streaming key-value stores.
3. Regulatory Auditability: In financial markets governed by model risk management guidelines (such as SR 11-7) and adverse action disclosure rules, fraud systems must provide explainable reasons for transaction declines. Tree ensembles support exact feature attribution via TreeSHAP, whereas deep graph representations that depend on non-linear neighbor embeddings present auditing challenges.

### 5.3.2 Financial Impact of Uncalibrated Probability Inflation
The $\mathbf{\$263K}$ loss increase observed in Stage 4 (\$629,411.78 vs. Stage 1's \$365,728.99) illustrates the operational risks of deploying uncalibrated models under class imbalance:
* Training with Focal Loss ($\alpha = 0.25, \gamma = 2.0$) alters output probability calibration, raising mean predicted risk to 0.1269 compared with the empirical base rate of 0.0341.
* When raw probabilities are evaluated under cost thresholds without fixed queue limits, this probability shift causes an alert rate of 13.46%, exceeding manual review capacity and creating high friction costs ($c_{\text{friction}} = \$15.00$ per false positive).
* Operational Principle: Neural models optimized with asymmetric loss functions require post-hoc isotonic probability calibration before deployment in capacity-constrained review workflows.

## 5.4 Limitations of the Study

To maintain transparency and delineate research boundaries, several structural delimitations and technical constraints must be acknowledged:

1. **Bounded 30-Day Lookback Horizon:**
   To guarantee bounded inference latency and prevent memory exhaustion across the 590,540-transaction stream, historical sub-graphs on $G_{<t}$ were truncated to a 30-day temporal window ($\Delta t \le 2,592,000$ seconds). While 30 days captures the vast majority of active card reuse and device velocity bursts, it obscures long-dormant fraud rings, seasonal sleeper accounts, and organized synthetic identities where initial account setup occurs months prior to cash-out attacks.
2. **Proxy Block-Delayed Supervision vs. Event-Level Chargeback Latency:**
   To prevent temporal label leakage, this dissertation enforced a conservative block-delayed supervision policy where models trained on Block $b$ were supervised strictly on labels confirmed prior to the start of Block $b$. In actual payment networks, chargebacks follow an asynchronous, long-tailed distribution (often modeled via Weibull or log-logistic survival curves) spanning 30 to 90 days. While the block-level embargo cleanly prevents look-ahead bias, it approximates the continuous, stochastic arrival of chargeback dispute notifications.
3. **Single-Gateway Transaction Clearing Topology:**
   The empirical evaluation was conducted on the IEEE-CIS benchmark dataset released under Kaggle academic research terms. While representing a massive, real-world commercial continuum (590,540 transactions over 182 days with high monetary exposure), the data originates from a single payment gateway (Vesta Corporation). Consequently, multi-bank fraud syndicates that distribute attacks across disparate acquiring gateways could not be observed without consortium-level clearing data.
4. **Static Linear Dimensionality Compression:**
   To establish a numerically tractable, fixed-dimensional node matrix $\mathbf{Z}_0 \in \mathbb{R}^{N \times 49}$ for the PyTorch network, the 339 continuous Vesta features were projected to 15 principal components via time-causal Block PCA. While PCA retained over $82\%$ of total variance, its linear formulation inevitably favored high-variance legitimate transactions over subtle, non-linear fraud interactions. Learned non-linear autoencoders or supervised entity embeddings may preserve richer tabular interactions, albeit at higher computational cost.


## 5.5 Actionable Recommendations

Based on the empirical findings, forensic failure analyses, and theoretical deductions of this dissertation, concrete, actionable recommendations are formulated for academic researchers and industrial financial institutions:

### 5.5.1 Actionable Recommendations for Academic Researchers
1. **Mandate Strict Time-Causal Graph Formulations ($G_{<t}$):**
   Academic venues (e.g., KDD, NeurIPS, ICLR, IEEE TKDE) should require all papers evaluating graph representation learning on financial, fraud, or temporal datasets to enforce strict score-before-insert semantics ($G_{<t}$). Any research that constructs static graphs across train and test splits or permits message passing from future timestamps must be recognized as fatally confounded by look-ahead leakage.
2. **Enforce Chronological, Expanding Out-of-Time Splitting:**
   Reviewers and benchmark creators must reject random $k$-fold cross-validation on time-stamped transaction streams. Models must be evaluated exclusively on forward, sealed chronological holdouts that reflect realistic open-world entity churn and covariate drift.
3. **Require Evaluation at Operational Low-FPR Operating Points:**
   Researchers should deprecate sole reliance on global ROC-AUC and macro F1 scores in extreme class-imbalanced settings ($\le 3.5\%$). Standard reporting must mandate non-interpolated Average Precision, Recall at $0.1\%$ FPR, Recall at $1.0\%$ FPR, and scenario-weighted dollar loss metrics ($L_{\text{fin}}$).
4. **Benchmark Against State-of-the-Art Tabular GBDTs with Causal Counters:**
   Novel graph-tabular architectures must be benchmarked directly against well-tuned gradient boosted decision trees (LightGBM/XGBoost) augmented with backward-rolling relational counters, rather than un-tuned vanilla MLPs.

### 5.5.2 Actionable Recommendations for Financial Institutions and Risk Teams
1. **Prioritize Explicit Relational Feature Engineering Over GNN Infrastructure:**
   Production risk teams should prioritize engineering 15 to 25 backward-looking entity counters, rolling monetary averages, and velocity deltas within a tabular GBDT framework. This configuration delivers state-of-the-art predictive discrimination, sub-4ms inference latency, exact TreeSHAP compliance, and the lowest financial dollar loss, without requiring GPU clusters or distributed graph databases.
2. **Implement Lexicographic Model Evaluation Frameworks:**
   Institutions should evaluate risk models using a tiered lexicographic protocol:
   $$\text{Select } M^* = \arg\min_M L_{\text{fin}}(M) \quad \text{subject to} \quad \text{Latency}(M) \le 10\text{ ms} \quad \land \quad \text{Auditable}(M) = \text{True}$$
3. **Enforce Hard Tabular Fallbacks for Cold-Start Entity Streams:**
   In production clearing streams where novel entities constitute 30% or more of transaction volume, hybrid models must enforce a mathematically hard tabular fallback ($\beta_i \to 0$ when $q_i^{\text{hist}} = 0$). GNN representations must never be evaluated on unobserved, disconnected nodes where lack of structural context causes uncontrolled variance.
4. **Mandate Post-Hoc Isotonic Probability Calibration:**
   Any model trained using Focal Loss, class-weighted cross-entropy, or under-sampling must be calibrated via out-of-fold isotonic regression prior to score publication, ensuring that operational thresholds reflect true posterior fraud probabilities and prevent triage queue overflow.


## 5.6 Future Research Directions

To advance the state of joint graph–tabular representation learning beyond the constraints identified in this dissertation, several promising avenues of future research are proposed:

### 1. Continuous-Time Dynamic Memory Networks (Temporal Graph Networks)
Rather than executing discrete, isotropic neighbor pooling across historical graphs, future work should explore continuous-time dynamic memory architectures, such as adapted **Temporal Graph Networks (TGNs)**. TGNs maintain an evolving, continuous-time memory state $\mathbf{s}_i(t)$ for each entity node, updating memory dynamically upon transaction arrival via recurrent neural units (e.g., GRUs) and temporal attention:
$$\mathbf{s}_i(t) = \text{GRU}\left( \mathbf{m}_i(t), \mathbf{s}_i(t^-) \right), \qquad \mathbf{m}_i(t) = \sum_{j \in \mathcal{N}_i(t)} \alpha_{ij}(t) \left[ \mathbf{s}_j(t^-) \,\|\, \mathbf{e}_{ij}(t) \,\|\, \Delta t \right]$$
Equipping TGNs with learned, cold-start fallback gates could enable continuous memory tracking of established fraudsters without degrading on unobserved cards.

### 2. Differentiable Multiplicative Aggregation and Product-Norm Layers
To bridge the fundamental representational gap between tree splits and neural layers, research must develop **non-linear, scale-invariant neural aggregation operators**. Instead of arithmetic mean pooling, neural layers should natively incorporate scale-invariant ratio transformations:
$$\mathbf{h}_i^{\text{ratio}} = \mathbf{z}_i \oslash \left( \bar{\mathbf{z}}_{\mathcal{N}(i)} + \boldsymbol{\epsilon} \right)$$
Integrating differentiable division and product layers into Graph Neural Networks could enable deep models to learn anomaly ratios directly from raw coordinates, overcoming the low-pass filter limitation.

### 3. True Logit-Space Alternating Functional Boosting
The adapted BGNN framework applied functional displacement updates to continuous input coordinates $\mathbf{Z}_0$, mutating the feature space between outer rounds. Future architectures should explore **true logit-space functional boosting**, where intermediate trees fit directly to the functional negative gradients of the joint objective:
$$F^{(m)}(\mathbf{x}_i, G_{<t}) = F^{(m-1)}(\mathbf{x}_i, G_{<t}) + \eta_m h_m(\mathbf{x}_i)$$
By holding the continuous coordinate manifold stationary and applying tree boosting strictly in logit space, the optimization loop can preserve neural stability while capturing residual tabular interactions.

### 4. Privacy-Preserving Multi-Bank Consortium Graphs via Federated Learning
Organized fraud syndicates systematically exploit visibility silos by executing low-velocity card testing across multiple independent issuing banks. Future research should investigate **privacy-preserving federated graph neural networks** combined with secure multi-party computation (SMPC) or differential privacy. This would allow financial consortia to learn joint relational representations of cross-bank fraud rings without exposing proprietary customer transaction records or violating data privacy regulations (e.g., GDPR).

## 5.7 Chapter Summary and Concluding Dissertation Statement

This dissertation evaluated whether joint graph-tabular representation learning using an adapted Boost then Convolve architecture provides measurable improvements in credit card fraud detection under time-causal streaming conditions.

Across 590,540 transactions and a sealed out-of-time holdout (Block 5, $N = 118,108$), explicit backward-looking relational feature engineering in Stage 2 achieved the highest ranking and cost performance ($H_1$: $\Delta_{2-1} = +0.0438$ AP, calibrated AP 0.5088, saving \$20,326.68 in financial loss over Stage 1). In contrast, alternating functional-gradient updates did not improve upon frozen embeddings ($H_3$: $\Delta_{4-3} = -0.0138$ AP), deep graph convolutions did not surpass standalone tabular LightGBM ($H_4$: $\Delta_{4-1} = -0.0181$ AP), and Focal Loss probability distortion raised uncalibrated financial losses to \$629,411.78. Graph-Augmented GBDT (Stage 2) recorded the lowest realized financial loss (\$345,402.31) and highest calibrated Average Precision ($\mathbf{\text{AP} = 0.5088}$), demonstrating that backward-looking relational feature engineering remains an effective, practical paradigm for payment clearing operations.

By replacing transductive configurations with strict score-before-insert causal graphs ($G_{<t}$) and pre-specified bootstrap hypothesis tests, this research establishes an empirical null result. These findings provide practical guidance for fraud risk engineering, delineating the operational boundaries of deep graph architectures in streaming payment environments.
