# Chapter 2: Literature Review and Theoretical Foundations

## 2.1 Tabular Representation Learning vs. Relational Graph Learning in Financial Systems
Financial transaction clearing represents a demanding operational environment for machine learning algorithms due to the dual structure of transaction data (Ivanov & Prokhorenkova, 2021). Worldwide card fraud losses exceed \$33 billion and are projected to surpass \$40 billion annually (Nilson Report, 2024). In high-throughput clearing architectures, transaction streams exhibit two distinct characteristics. On one hand, each authorization request arrives as a tabular record with heterogeneous numerical measurements, high-cardinality categorical identifiers, substantial missingness, and skewed monetary amounts (Ke et al., 2017). On the other hand, transactions connect through shared payment instruments, physical devices, merchant gateways, and recipient channels, forming an evolving relational network (Akli et al., 2026; Lahcen et al., 2026).

Fraud detection research has largely developed along two distinct tracks: tabular learning dominated by Gradient Boosted Decision Trees (GBDTs), and relational algorithms driven by Graph Neural Networks (GNNs). As surveys of credit card fraud systems indicate (Pourhabibi et al., 2020), these paradigms embody different inductive biases regarding coordinate continuity, data topology, and transformation invariance.

### 2.1.1 Gradient Boosted Decision Trees on Tabular Manifolds
Gradient Boosted Decision Trees serve as the standard baseline for tabular data across industrial benchmarks and competitive evaluations (Chen & Guestrin, 2016; Grinsztajn et al., 2022). Grounded in the greedy function approximation framework formulated by Friedman (2001), GBDTs construct an additive ensemble of $K$ decision trees in function space:
$$F_K(\mathbf{x}) = \sum_{k=1}^K f_k(\mathbf{x}), \quad f_k \in \mathcal{F}$$
where each base learner $f_k(\mathbf{x})$ recursively partitions the feature space into $J$ orthogonal, axis-aligned hyper-rectangular regions $\mathcal{R}_{j,k}$ and assigns a constant prediction weight $w_{j,k}$ to each partition:
$$f_k(\mathbf{x}) = \sum_{j=1}^J w_{j,k} \cdot \mathbb{I}(\mathbf{x} \in \mathcal{R}_{j,k})$$

At each boosting iteration $k$, the tree parameters and partition weights are determined by minimizing a second-order Taylor expansion of the empirical loss function $\mathcal{L}$:
$$\tilde{\mathcal{L}}^{(k)} \approx \sum_{i=1}^N \left[ g_i f_k(\mathbf{x}_i) + \frac{1}{2} h_i f_k^2(\mathbf{x}_i) \right] + \Omega(f_k)$$
where $g_i = \partial \mathcal{L}(y_i, \hat{y}_i^{(k-1)}) / \partial \hat{y}_i^{(k-1)}$ and $h_i = \partial^2 \mathcal{L}(y_i, \hat{y}_i^{(k-1)}) / \partial (\hat{y}_i^{(k-1)})^2$ represent the first- and second-order loss derivatives with respect to the current ensemble prediction, and $\Omega(f_k) = \gamma J + \frac{1}{2}\lambda \sum_{j=1}^J w_j^2$ enforces structural regularization.

The effectiveness of GBDTs over standard neural architectures on tabular data is governed by three inductive biases:
1. Invariance to Monotonic Coordinate Transformations: Because tree-splitting criteria evaluate univariate ordering statistics ($x_j > \theta$), decision trees are invariant to scaling, coordinate shifts, and skewed distributions commonly observed in transaction amounts.
2. Native Default-Direction Missingness Routing: Gradient boosting libraries, including LightGBM (Ke et al., 2017), XGBoost (Chen & Guestrin, 2016), and CatBoost (Prokhorenkova et al., 2018), assign a default split direction for missing values during node splitting based on maximum gradient reduction. This native handling treats data sparsity as a predictive signal rather than requiring synthetic imputation methods that can distort covariance structures.
3. Orthogonal Boundary Partitioning: Operational fraud thresholds frequently operate as step functions, such as transaction velocity rules exceeding Velocity Limit $L$ or authorization amounts crossing risk triggers. Tree ensembles isolate these discontinuous boundaries with low parameter complexity, whereas continuous feed-forward networks require greater depth to approximate step functions.

Despite developments in deep tabular architectures designed to emulate decision tree properties, such as TabNet with sparse attention masks (Arik & Pfister, 2021) and Neural Oblivious Decision Ensembles (NODE) using differentiable oblivious trees (Popov et al., 2020), empirical benchmarks indicate that optimized GBDT implementations regularly outperform neural alternatives on unnormalized tabular datasets (Grinsztajn et al., 2022).

However, tabular GBDTs evaluate observations under an independent and identically distributed (i.i.d.) assumption across transaction samples. A standalone GBDT evaluates each incoming transaction $\mathbf{x}_i$ in isolation, without mechanisms to capture the multi-hop relationships, fraud rings, and device-cycling behaviors that characterize organized payment fraud.

### 2.1.2 Graph Neural Networks on Relational Topologies
To exploit topological structures, Graph Neural Networks (GNNs) formulate fraud detection as node classification or edge classification over an evolving multigraph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, mapping relational connectivity and node attributes $\mathbf{X}$ into low-dimensional continuous vector spaces through recursive neighborhood aggregation (Kipf & Welling, 2017; Hamilton et al., 2017). Under the general Neural Message Passing framework (Gilmer et al., 2017), the node representation $\mathbf{h}_v^{(l)}$ at layer $l$ is computed iteratively through spatial message passing:
$$\mathbf{m}_v^{(l)} = \text{AGGREGATE}^{(l)}\left( \left\{ \text{MESSAGE}^{(l)}\left(\mathbf{h}_v^{(l-1)}, \mathbf{h}_u^{(l-1)}, \mathbf{e}_{uv}\right) : u \in \mathcal{N}(v) \right\} \right)$$
$$\mathbf{h}_v^{(l)} = \text{UPDATE}^{(l)}\left( \mathbf{h}_v^{(l-1)}, \mathbf{m}_v^{(l)} \right)$$

Foundational architectures instantiate these operators through diverse mechanisms: Graph Convolutional Networks (GCN) utilize symmetric normalized adjacency convolutions $\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}\mathbf{H}^{(l)}\mathbf{W}^{(l)}$ (Kipf & Welling, 2017); GraphSAGE introduces inductive uniform neighborhood sampling coupled with pooling or LSTM aggregators (Hamilton et al., 2017); and Graph Attention Networks (GAT) incorporate parameterized self-attention coefficients $\alpha_{uv}$ to weigh neighboring nodes dynamically based on feature similarity (Veličković et al., 2018).

The theoretical expressive power of standard message-passing GNNs is strictly bounded by the 1-Weisfeiler–Lehman (1-WL) graph isomorphism test (Xu et al., 2018; Morris et al., 2019). Under the 1-WL bound, standard GNNs cannot distinguish non-isomorphic graph structures possessing identical local degree distributions, restricting their ability to detect subtle multi-hop cycle anomalies without higher-order relational extensions (Morris et al., 2019).

In financial forensics, early network-based fraud detection relied on iterative label propagation and heuristic relational features, such as the GOTCHA! framework developed by Van Vlasselaer et al. (2017) for social security and payment fraud networks. The application of modern deep GNNs to financial crime was evaluated by Weber et al. (2019) on massive high-throughput transaction streams in Bitcoin forensic monitoring (the Elliptic dataset); notably, their benchmarks showed that Random Forests (0.79 F1 on illicit transactions) outperformed Graph Convolutional Networks (0.67 F1), establishing that well-tuned tree ensembles remain remarkably resilient against graph convolutions on tabular transaction benchmarks.

Nevertheless, when applied to high-throughput financial clearing streams, standard GNN architectures encounter severe operational failure modes:
1. Heterophily in Transaction Graphs: Standard message-passing formulations assume graph homophily, where connected nodes share identical class labels ($\mathbb{P}(y_u = y_v \mid (u,v) \in \mathcal{E}) \gg 0.5$). Financial fraud graphs, however, exhibit heterophily: illicit transactions are interleaved among thousands of legitimate cardholder events across common merchants and payment gateways (Zhu et al., 2020). Spatial convolutions over heterophilic graphs perform low-pass smoothing, diluting fraudulent node representations into the majority class. While camouflage-resistant architectures such as CARE-GNN (Dou et al., 2020) and PC-GNN (Liu et al., 2021) apply label-aware filtering, they rely on immediate ground-truth feedback that is unavailable in live clearing streams.
2. Smoothness and Differentiability Constraints: GNNs rely on continuous gradient descent operating across smooth Euclidean manifolds. When presented with unnormalized, skewed tabular attributes, continuous linear projection layers ($\mathbf{W}\mathbf{x} + \mathbf{b}$) struggle to resolve sharp step thresholds, which can destabilize optimization in high-cardinality categorical spaces (Ivanov & Prokhorenkova, 2021).
3. Heterogeneous Relation Complexity: Transaction graphs link multiple entity types, including cards, devices, IP addresses, and emails. Modeling these heterogeneous relations requires multi-relation architectures such as Heterogeneous Graph Attention Networks (Wang et al., 2019) or multi-relation anomaly detectors (Ma et al., 2021), increasing parameter counts and inference latencies.

## 2.2 Theoretical Hybrid Graph–Tabular Paradigms and the BGNN Core
Recognizing the complementary strengths of GBDTs (tabular partitioning) and GNNs (relational aggregation), researchers have proposed architectural paradigms to bridge this divide. These paradigms are synthesized conceptually in Table 2.1.

### 2.2.1 Decoupled Two-Stage Hybrid Pipelines (Res-GNN Models)
The most common industrial paradigm decouples the tabular and graph learning stages into an asymmetrical two-stage sequential pipeline. In this formulation, a standalone GBDT is trained on raw tabular features $\mathbf{X}$ to output baseline predictions or out-of-fold logit representations:
$$\mathbf{Z}_0 = F_{\text{GBDT}}(\mathbf{X})$$
These scalar logits or intermediate leaf embeddings are then projected into an initial node feature coordinate space $\mathbf{H}^{(0)} \in \mathbb{R}^{N \times d_{\text{emb}}}$, over which a frozen GNN is trained to predict the final transaction label (Ivanov & Prokhorenkova, 2021):
$$\hat{\mathbf{y}} = \text{GNN}(\mathbf{H}^{(0)}, \mathcal{G})$$

Recent hybrid architectures investigate variations of this decoupled scheme. Mao and Zhu (2026) demonstrated that combining ensemble tree outputs with graph neural representations improves detection in complex financial fraud rings. Similarly, Mohawesh et al. (2026) integrated GraphSAGE neighborhood representations with large-scale feature projections across payment fraud datasets, and Onwudebelu et al. (2026) introduced behavior-aware dynamic user profiling to counteract transactional drift. In decoupled relational settings, DISCO (Zhu et al., 2026) separates representation learning from downstream classification to mitigate label starvation, while DG-GCN (Wang et al., 2026) constructs dual similarity-guided graph convolutions to constrain relational message passing. In federated financial contexts, Tang and Liang (2024) explored multi-institution graph-tabular pipelines under privacy constraints.

While decoupled Res-GNN pipelines use pre-trained tabular representations, they exhibit gradient detachment. Once the base GBDT model is fitted, its decision boundaries remain fixed. The downstream GNN cannot backpropagate gradients into the tree ensemble, leaving the tabular feature selector incapable of adapting to errors identified during graph message passing.

### 2.2.2 Differentiable Tree Approximations
A second theoretical paradigm replaces standard greedy decision trees with fully differentiable tree approximations, enabling end-to-end backpropagation through both tree partitions and graph convolutions simultaneously. In models such as the Tree Ensemble Layer (TEL) formulated by Hazimeh et al. (2020), discrete indicator splitting functions $\mathbb{I}(x_j > \theta)$ are replaced by smooth continuous approximations:
$$\sigma_{\tau}(x_j - \theta) = \frac{1}{1 + \exp\left(-\frac{x_j - \theta}{\tau}\right)}$$
where $\tau > 0$ denotes a temperature hyperparameter governing split sharpness.

Differentiable tree relaxations, however, often show reduced accuracy on real-world tabular data. Soft continuous sigmoid splits round off the orthogonal partition boundaries that benefit greedy trees, while joint gradient optimization across large ensembles is susceptible to vanishing gradients and local minima (Grinsztajn et al., 2022). As a result, soft differentiable trees rarely match the predictive performance of greedy GBDTs on financial transaction datasets.

### 2.2.3 The Foundational Boost then Convolve (BGNN) Paradigm
To achieve true end-to-end coupling without sacrificing greedy discrete tree partitioning, Ivanov and Prokhorenkova (2021) formulated **Boost then Convolve (BGNN)**. BGNN resolves the gradient detachment dilemma through an alternating functional-gradient optimization scheme that alternates between:
1. Training continuous GNN weights $\mathbf{\Theta}_{\text{gnn}}$ via stochastic gradient descent while holding tree predictions constant; and
2. Updating the tabular tree ensemble $F(\mathbf{X})$ by fitting new decision trees directly to the functional coordinate gradients:
$$\mathbf{R}_i^m = -\left[ \frac{\partial \mathcal{L}}{\partial \mathbf{z}_i} \right]_{\mathbf{z}_i = \mathbf{Z}_0[i]}$$
where $\mathbf{z}_i$ represents the continuous tabular representation embedded in the graph.

By fitting greedy decision trees to the residual displacement vectors generated by the GNN loss, BGNN achieves mathematical coupling between tabular splits and graph convolutions. However, the original BGNN formulation was derived exclusively for static, homogeneous, transductive academic benchmarks (such as House, VK, and Avazu tabular graphs). When applied to real-time financial clearing environments, BGNN encounters streaming vulnerabilities not addressed in its original formulation.

## 2.3 Operational Vulnerabilities of Graph Architectures in Production Streams
Deploying relational graph architectures into industrial payment networks exposes four systemic operational failure modes that are widely ignored in static academic literature.

### 2.3.1 Temporal Look-Ahead Leakage and Rolling Non-Transductive Evaluation
Most academic GNN fraud benchmarks evaluate models under transductive random split protocols (such as 70% train, 15% validation, and 15% test selected uniformly across the dataset). In a relational graph, random splitting introduces leakage: an evaluation transaction occurring at Day 180 connects through shared card or device edges to training transactions from Day 10. Message passing across these bidirectional edges allows evaluation representations to aggregate features from future and past events simultaneously (Akli et al., 2026).

In production clearing switches, time flows in one direction. An incoming query transaction arriving at timestamp $t_i$ can connect only to transactions occurring in the strict historical past ($t < t_i$). Studies show that models reporting strong performance under transductive splits experience substantial performance drops when evaluated under chronological out-of-time (OOT) protocols (Van Belle et al., 2023). While continuous-time dynamic graph networks such as Temporal Graph Networks (TGN) maintain dynamic node memories (Rossi et al., 2020) and temporal-semantic models attempt to capture evolving transaction patterns (Zhang et al., 2026), standard academic benchmarks often overlook temporal leakage.

### 2.3.2 Delayed Operational Labels, Chargeback Latency, and Contaminated Feedback
Academic benchmarks frequently assume immediate label availability: as soon as transaction $v_i$ occurs, its ground-truth label $y_i \in \{0, 1\}$ is treated as known for model updating.

In operational banking reality, cardholders do not inspect authorization feeds continuously. Fraudulent transactions are discovered only when customers review billing statements, receive automated alerts, or dispute unauthorized charges. The resulting chargeback reporting latency ranges from 14 to 90 days (Dal Pozzolo et al., 2018). Consequently, throughout an active operational window $[t - \tau, t]$, unconfirmed fraudulent transactions appear as negative transactions ($y = 0$). Training continuous graph convolutions on contaminated labels introduces gradient noise and can destabilize optimization.

### 2.3.3 Inductive Inference, Open-World Subgraphs, and the Cold-Start Dilemma
Standard GNN architectures assume closed-world graphs where the full node set $\mathcal{V}$ is known in advance. Financial clearing, however, is an open-world inductive streaming environment: consumers open new accounts, replace cards, acquire new mobile devices, and transact at new merchants.

In the IEEE-CIS benchmark continuum, 30.6% of evaluation transactions involve payment card entities not observed in the historical training data. For an unseen, cold-start transaction $v_{\text{new}}$, the historical neighborhood is empty:
$$\mathcal{N}(v_{\text{new}}) = \emptyset \implies \sum_{u \in \mathcal{N}(v_{\text{new}})} \mathbf{h}_u = \mathbf{0}$$
When a graph model relies entirely on relational message passing, its receptive field collapses on isolated nodes, forcing predictions based on zero vectors or global bias terms.

## 2.4 Gap-Derived Design Requirements for the Proposed Framework
To address these operational failure modes, a production-grade joint graph-tabular learning framework should satisfy four requirements:
1. Strict Temporal Causality ($G_{<t}$): Graph construction must be strictly causal and backward-looking. For any query transaction $v_i$ arriving at timestamp $t_i$, edge generation is bounded: $\mathcal{E}_i \subseteq \{ (u, v_i) : t_u < t_i \text{ and } t_i - t_u \le K_{\text{lookback}} \}$. Future edges and unembargoed contemporaneous edges are masked.
2. Continuous Aligned Representation Space ($Z_0$): Tabular tree logit predictions should be projected into an aligned coordinate space $Z_0 \in \mathbb{R}^{N \times d_{\text{emb}}}$ that provides predictive capacity even when graph edges are absent.
3. Adaptive Monotonic Fallback Blending Gate ($\beta_i$): The prediction logit must dynamically arbitrate between neural graph and tabular GBDT heads based on local graph density telemetry ($c_i, d_i$). When a transaction node is isolated ($c_i = 0, d_i = 0$), the blending gate mathematically enforces exact tabular fallback ($\beta_i \to 0$), routing inference exclusively to the Stage 1 GBDT.
4. Asymmetric Operational Risk Optimization ($L_{\text{fin}}$): Optimization and evaluation must account for asymmetric financial costs. Normalized Mini-Batch Focal Loss addresses severe class imbalance (3.499% fraud prevalence), and models are evaluated against realized financial loss ($L_{\text{fin}}$) incorporating investigator review capacity ($c_{\text{review}}$ = $5.00) and customer friction costs ($c_{\text{friction}}$ = $15.00, $c_{\text{residual}} = 0.05$).

## 2.5 Synthesis Framework and Structural Justification of the Experimental Ladder
To evaluate these principles without confounding architectural variations with hyperparameter tuning, this dissertation establishes a Four-Stage Experimental Ladder (Section 3.1 and Table 3.1).

The ladder spans four distinct tiers:
* Stage 1 (Standalone Tabular GBDT): Establishes the tabular baseline using LightGBM on 371 tabular features under monotonic tree partitioning.
* Stage 2 (Graph-Augmented GBDT): Augments the tabular feature matrix with 19 time-causal, backward-rolling graph counters ($G_{<t}$), isolating the empirical value of relational features without neural convolutions.
* Stage 3 (Decoupled Res-GNN): Evaluates decoupled two-stage neural message passing by propagating frozen Stage 1 GBDT logits over dynamic bipartite graph neighborhoods.
* Stage 4 (Adapted Alternating BGNN): Evaluates the joint optimization framework, coupling GBDT tree fitting with GNN convolutions via the adaptive blending gate $\beta_i$.

Progressing through Stages 1 to 4 isolates the performance delta ($\Delta_{j-k}$) associated with each algorithmic mechanism, enabling a forensic evaluation of graph-tabular interactions.

## 2.6 Critical Synthesis of Research Gaps
In summary, this literature review identifies four research gaps in financial fraud detection:
* Gap 1 (Optimization Separation): The absence of an alternating functional-gradient framework that couples greedy decision tree ensembles with graph neural convolutions in payment networks.
* Gap 2 (Temporal Look-Ahead Leakage): Widespread reliance on transductive random splits and static graph topologies in academic benchmarks, which masks performance drops under production streaming conditions.
* Gap 3 (Cold-Start Inductive Performance): The lack of guaranteed tabular fallback mechanisms for isolated, open-world transaction nodes in streaming relational graphs.
* Gap 4 (Loss and Metric Misalignment): Limited evaluation of graph neural architectures against realized operational financial loss ($L_{\text{fin}}$) under strict false-positive rate and human triage capacity limits.

The methodology formalized in Chapter 3 addresses each of these four research gaps.
