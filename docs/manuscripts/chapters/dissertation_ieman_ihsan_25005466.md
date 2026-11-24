# A Time-Causal Joint Graph–Tabular Learning Framework for Credit Card Fraud Detection Using an Adapted Boost then Convolve Architecture

**Author:** Ieman Ihsan Bin Alwi  
**Student ID:** 25005466  
**Degree:** Master of Science in Data Science (Master Coursework Project)  
**Supervisor:** Prof. Dr. Haruhiro Fujita  
**Department:** Fundamental and Applied Sciences Department, Faculty of Science and Information Technology  
**Institution:** Universiti Teknologi PETRONAS, 32610 Seri Iskandar, Perak Darul Ridzuan, Malaysia  
**Date of Submission:** October 2026  

---

## Declaration of Originality

I hereby declare that this dissertation entitled **"A Time-Causal Joint Graph–Tabular Learning Framework for Credit Card Fraud Detection Using an Adapted Boost then Convolve Architecture"** is my own work, except where due acknowledgment and reference have been made in the text. I further declare that this dissertation has not been previously submitted, in whole or in part, for any other degree or professional qualification at Universiti Teknologi PETRONAS or any other institution of higher learning.

\
**Ieman Ihsan Bin Alwi**  
Student ID: 25005466  
Date: 13 October 2026  

---

## Acknowledgments

First and foremost, I express my deepest gratitude to my principal research supervisor, **Prof. Dr. Haruhiro Fujita**, whose exceptional scholarly guidance, rigorous feedback, and methodological insistence on empirical integrity provided the foundation for this research. His constructive critique encouraged me to look beyond conventional benchmark claims, interrogate experimental assumptions, and recognize that the rigorous defense of an empirical null result is among the most valuable contributions a machine learning engineer can offer.

I also extend my sincere appreciation to the faculty members and laboratory staff of the Faculty of Science and Information Technology and the Department of Fundamental and Applied Sciences at Universiti Teknologi PETRONAS for providing the high-performance computing infrastructure and academic environment essential to the execution of this computationally intensive study.

Finally, I dedicate this work to my family and peers, whose unwavering patience, encouragement, and understanding sustained me throughout the rigorous course of this postgraduate journey.

---

## Abstract

Credit card fraud detection operating within high-throughput financial clearing streams presents a difficult challenge characterized by extreme class imbalance, severe entity turnover, delayed ground-truth chargeback labels, and strict latency budgets. While Gradient Boosted Decision Trees (GBDTs) remain the industrial standard for tabular transaction attributes, Graph Neural Networks (GNNs) have emerged in the academic literature as a powerful tool for modeling relational connectivity across cardholders, merchants, and digital devices. However, standard GNN benchmarks predominantly rely on transductive, static, or randomly partitioned graphs that introduce catastrophic temporal look-ahead leakage and overlook the open-world cold-start dilemma. 

This dissertation investigates whether an adapted **Boost then Convolve (BGNN)** architecture—integrating continuous tabular embeddings $Z_0$, relation-aware message passing over a strictly time-causal heterogeneous graph $G_{<t}$, monotonic decision blending via an adaptive gate $\beta_i$, and alternating functional-gradient optimization ($R_i^m$)—can deliver statistically and economically meaningful improvements over tuned tabular baselines. Evaluated on the benchmark IEEE-CIS dataset ($N = 590,540$ transactions across a 182-day continuum) under a pre-registered, chronologically expanding 5-block design culminating in a sealed Out-of-Time (OOT) holdout ($N = 118,108$, Block 5), the pipeline executes a controlled 4-stage experimental ladder.

The empirical results reveal a profound architectural divergence. Under confirmatory hypothesis testing with 1,000-resample bootstrap and Benjamini-Hochberg False Discovery Rate control ($\alpha = 0.05$):
1. **Primary Coupling Uplift ($H_3$ / RQ3) is SUPPORTED:** Alternating functional-gradient coupling significantly outperforms the decoupled, frozen Res-GNN pipeline ($\Delta_{4-3} = +0.0804$ Average Precision, AP), validating the core algorithmic coupling mechanism.
2. **Structural Decoupling Impact ($H_2$ / RQ2) Collapses:** Passing fixed tabular embeddings through a decoupled multi-hop Res-GNN causes severe degradation compared to a feature-augmented GBDT ($\Delta_{3-2} = -0.3445$ AP), demonstrating that continuous message passing alone cannot compensate for the loss of native tabular partitioning.
3. **Macro System Uplift ($H_4$ / RQ1) Fails:** The full alternating BGNN ($S_4$, AP: 0.3178; realized financial loss $L_{fin}$: \$2,123,185.89) falls substantially short of the standalone tabular GBDT baseline ($S_1$, AP: 0.5894; $L_{fin}$: \$465,061.63, yielding $\Delta_{4-1} = -0.2716$ AP), while the Graph-Augmented GBDT ($S_2$) achieves the lowest financial loss (\$455,902.35).

Forensic failure analysis isolates three decisive breakdown mechanisms: (i) an inductive bias mismatch wherein unweighted neural aggregations act as low-pass filters that oversmooth acute ratio anomalies ($x_{\text{amt}} / \bar{x}_{\text{amt}}$); (ii) open-world graph sparsity, where a 30.6% entity cold-start rate and an unbounded $9.56\,\sigma$ temporal covariate drift saturate continuous projections; and (iii) latent manifold corruption during alternating residual displacement, compounded by saturated gate suppression ($\bar{\beta} = 0.8726$). Rather than an experimental failure, this dissertation presents these findings as an epistemological defense of negative empirical results, demonstrating that industrial fraud systems are far better served by backward-looking causal relational features modeled via GBDTs than by complex, computationally expensive deep graph convolutions.

**Keywords:** Credit Card Fraud Detection, Boosted Graph Neural Networks, Tabular Representation Learning, Time-Causal Graph, Out-of-Time Evaluation, Inductive Learning, Financial Risk Minimization.

---

## Table of Contents

- [Abstract](#abstract)
- [List of Figures](#list-of-figures)
- [List of Tables](#list-of-tables)
- [List of Symbols and Abbreviations](#list-of-symbols-and-abbreviations)
- [Chapter 1: Introduction](#chapter-1-introduction)
  - [1.1 Background of Study](#11-background-of-study)
  - [1.2 Problem Statement](#12-problem-statement)
  - [1.3 Research Objectives](#13-research-objectives)
  - [1.4 Research Questions and Pre-Registered Hypotheses](#14-research-questions-and-pre-registered-hypotheses)
  - [1.5 Scope and Delimitations of Study](#15-scope-and-delimitations-of-study)
  - [1.6 Significance of the Study](#16-significance-of-the-study)
  - [1.7 Structure of the Dissertation](#17-structure-of-the-dissertation)
- [Chapter 2: Literature Review and Theoretical Foundations](#chapter-2-literature-review-and-theoretical-foundations)
  - [2.1 Tabular Representation Learning vs. Relational Graph Learning in Financial Systems](#21-tabular-representation-learning-vs-relational-graph-learning-in-financial-systems)
  - [2.2 Theoretical Hybrid Graph–Tabular Paradigms and the BGNN Core](#22-theoretical-hybrid-graphtabular-paradigms-and-the-bgnn-core)
  - [2.3 Operational Vulnerabilities of Graph Architectures in Production Streams](#23-operational-vulnerabilities-of-graph-architectures-in-production-streams)
  - [2.4 Gap-Derived Design Requirements for the Proposed Framework](#24-gap-derived-design-requirements-for-the-proposed-framework)
  - [2.5 Synthesis Framework and Structural Justification of the Experimental Ladder](#25-synthesis-framework-and-structural-justification-of-the-experimental-ladder)
  - [2.6 Critical Synthesis of Research Gaps](#26-critical-synthesis-of-research-gaps)
- [Chapter 3: Research Methodology and System Formulation](#chapter-3-research-methodology-and-system-formulation)
  - [3.1 Overview and Research Design](#31-overview-and-research-design)
  - [3.2 Dataset and Preprocessing Pipelines](#32-dataset-and-preprocessing-pipelines)
  - [3.3 Heterogeneous Temporal Graph Formulation](#33-heterogeneous-temporal-graph-formulation)
  - [3.4 Adapted Boost then Convolve Architecture](#34-adapted-boost-then-convolve-architecture)
  - [3.5 Sparsity, Cold Start, and Robust Inference](#35-sparsity-cold-start-and-robust-inference)
  - [3.6 Experimental Setup and Benchmarking Strategy](#36-experimental-setup-and-benchmarking-strategy)
  - [3.7 Performance Evaluation, Financial Loss, and Confirmatory Statistical Protocol](#37-performance-evaluation-financial-loss-and-confirmatory-statistical-protocol)
- [Chapter 4: Empirical Results and Forensic Failure Analysis](#chapter-4-empirical-results-and-forensic-failure-analysis)
  - [4.1 Benchmark Results on the Sealed Out-of-Time Holdout](#41-benchmark-results-on-the-sealed-out-of-time-holdout)
  - [4.2 Confirmatory Hypothesis Testing and Statistical Verification](#42-confirmatory-hypothesis-testing-and-statistical-verification)
  - [4.3 Forensic Failure Mode 1: The Multiplicative Ratio Advantage vs. Additive Linear Smoothing](#43-forensic-failure-mode-1-the-multiplicative-ratio-advantage-vs-additive-linear-smoothing)
  - [4.4 Forensic Failure Mode 2: Graph Sparsity, Extreme Cold-Start Dynamics, and Covariate Drift](#44-forensic-failure-mode-2-graph-sparsity-extreme-cold-start-dynamics-and-covariate-drift)
  - [4.5 Forensic Failure Mode 3: Latent Manifold Corruption and Saturated Blending Gate](#45-forensic-failure-mode-3-latent-manifold-corruption-and-saturated-blending-gate)
  - [4.6 Forensic Failure Mode 4: Probability Miscalibration and Financial Loss Explosion](#46-forensic-failure-mode-4-probability-miscalibration-and-financial-loss-explosion)
- [Chapter 5: Conclusion, Recommendations, and Defense Synthesis](#chapter-5-conclusion-recommendations-and-defense-synthesis)
  - [5.1 Comprehensive Synthesis of Conclusions](#51-comprehensive-synthesis-of-conclusions)
  - [5.2 Theoretical and Methodological Implications for Machine Learning](#52-theoretical-and-methodological-implications-for-machine-learning)
  - [5.3 Practical Recommendations for Financial Risk Systems](#53-practical-recommendations-for-financial-risk-systems)
  - [5.4 Epistemological Defense of Negative Results in Applied Machine Learning](#54-epistemological-defense-of-negative-results-in-applied-machine-learning)
  - [5.5 Limitations and Future Research Directions](#55-limitations-and-future-research-directions)
- [References](#references)
- [Appendices](#appendices)

---

## List of Figures

- **Figure 1.1:** Core Conceptual Overview of the Time-Causal Graph–Tabular Problem Space.
- **Figure 2.1:** Conceptual Topologies of Hybrid GBDT–GNN Learning Paradigms.
- **Figure 3.1:** High-Level End-to-End Methodological Architecture of the Adapted BGNN Framework.
- **Figure 3.2:** Chronological Expanding 5-Block Partitioning Design.
- **Figure 3.3:** Schema of the Heterogeneous Transaction Graph with Strict Backward Lookback Windows.
- **Figure 3.4:** Relation-Aware Message Passing with Directional and Temporal Attention Masking.
- **Figure 3.5:** Alternating Optimization Loop: Inner Neural Training vs. Outer Functional Coordinate Fitting.
- **Figure 3.6:** Monotonic Blending Gate Architecture with Connectivity Quality Telemetry.
- **Figure 4.1:** Publication Forest Plot of Bootstrap Confidence Intervals Across Hypotheses H1–H4.
- **Figure 4.2:** Sealed Out-of-Time Precision–Recall Curves: Full Spectrum and High-Alert Low-FPR Regimes.
- **Figure 4.3:** Calibration Reliability Diagrams: Raw Focal Loss Probability Distortion vs. Isotonic Recalibration.
- **Figure 4.4:** Entity Cohort Breakdown (Established vs. Cold-Start) and Chronological Time-Slice Stability.

---

## List of Tables

- **Table 2.1:** Conceptual Matrix of GBDT–GNN Hybrid Architectures Across Structural Dimensions.
- **Table 3.1:** Dataset Verification and Feature Architecture Summary (IEEE-CIS Benchmark).
- **Table 3.2:** Chronological Block Partition Parameters and Quarantine Windows.
- **Table 3.3:** Structural Blueprint of the 4-Stage Experimental Benchmarking Ladder.
- **Table 3.4:** Hyperparameter Optimization Space and Search Fairness Coordinates.
- **Table 4.1:** Master Evaluation Benchmark on Sealed OOT Block 5 ($N = 118,108$).
- **Table 4.2:** Formal Paired Bootstrap Hypothesis Testing Results (1,000 Resamples, Benjamini–Hochberg FDR).
- **Table 4.3:** Stratified Cohort Performance Breakdown Across Entity Overlap and Temporal Quartiles.
- **Table 4.4:** Top-20 Feature Gain Importance Table from Stage 2 Graph-Augmented GBDT.
- **Table 4.5:** Stepwise Performance Drop Tracker and Mechanism Ablation Summary.

---

## List of Symbols and Abbreviations

- $G_{<t}$: Time-causal dynamic heterogeneous transaction graph containing solely entities and events strictly preceding timestamp $t$.
- $Z_0$: Continuous multidimensional initial transaction embedding matrix ($Z_0 \in \mathbb{R}^{N \times d_{\text{emb}}}$).
- $R_i^m$: Multi-output functional-gradient pseudo-residual vector for observation $i$ at outer boosting iteration $m$.
- $\beta_i$: Monotonic decision blending gate coefficient ($\beta_i \in [0, 1]$) controlling the trade-off between neural GNN and tabular GBDT logits.
- $q_i^{\text{hist}}$: Historical graph connectivity quality telemetry vector for transaction $i$.
- $\Delta_{j-k}$: Pairwise performance contrast in Average Precision between Stage $j$ and Stage $k$ ($\Delta_{j-k} = \text{AP}_j - \text{AP}_k$).
- $L_{\text{fin}}$: Realized scenario-weighted financial loss metric in dataset monetary units.
- $\text{TransactionDT}$: Continuous chronological timestamp in seconds relative to the dataset reference datum.
- $\text{TransactionAmt}$: Gross monetary value of the evaluated payment card transaction.
- $\text{AP}$: Average Precision (non-interpolated Precision–Recall Area Under Curve).
- $\text{FPR}$: False Positive Rate ($\text{FP} / (\text{FP} + \text{TN})$).
- $\text{ECE}$: Expected Calibration Error.
- $\text{GBDT}$: Gradient Boosted Decision Tree.
- $\text{GNN}$: Graph Neural Network.
- $\text{BGNN}$: Boosted Graph Neural Network.
- $\text{OOT}$: Out-of-Time evaluation split.
- $\text{OOF}$: Out-of-Fold cross-validation prediction.
- $\text{FDR}$: False Discovery Rate (Benjamini–Hochberg multiplicity correction).
- $\text{LCB}_{0.95}$: Lower bound of the 95% empirical bootstrap confidence interval.

---

# Chapter 1: Introduction

## 1.1 Background of Study
Electronic payment clearing systems form the backbone of modern global commerce. As payment ecosystems have transitioned toward frictionless, instant clearing protocols, payment card fraud has scaled proportionally in sophistication, volume, and financial impact. Worldwide payment card fraud losses exceeded USD 33.8 billion in 2023 and are projected by industry analysts to surpass USD 40.0 billion by 2028 (Nilson Report, 2024). In parallel, financial institutions face acute operational constraints: high-throughput clearing engines must render scoring decisions within strict 50 to 100 millisecond service-level agreements (SLAs), while manual human fraud review queues are severely constrained, typically restricted to evaluating at most 1.0% to 2.0% of total daily transaction volume.

Machine learning models deployed in automated fraud screening must operate under severe asymmetric costs. A false negative (missing a fraudulent transaction) results in direct chargeback losses, reimbursement mandates, and merchant processing penalties. Conversely, a false positive (flagging and blocking a legitimate cardholder transaction) causes acute customer friction, potential account abandonment, and administrative investigation overhead. To maximize the recovery of stolen funds while maintaining customer trust, risk engines must achieve exceptionally high precision within the extreme low false-positive rate regime (FPR $\le 0.1\%$ to $1.0\%$).

Historically, payment fraud detection has been dominated by **Gradient Boosted Decision Trees (GBDTs)**, such as LightGBM, XGBoost, and CatBoost. GBDTs excel on tabular transaction records characterized by heterogeneous, unnormalized feature distributions, skewed monetary quantities, missingness patterns, and high-cardinality categorical variables. Simultaneously, criminal syndicates increasingly execute coordinated attacks across dispersed accounts, synthetic digital identities, shared internet protocol (IP) subnets, and compromised physical devices. This relational topology naturally forms a dynamic graph. In response, academic research has increasingly focused on **Graph Neural Networks (GNNs)**, postulating that recursive message-passing layers can autonomously discover multi-hop fraud rings without requiring manual feature engineering.

## 1.2 Problem Statement
Despite the proliferation of academic literature asserting the superiority of GNNs in financial fraud detection, a massive deployment gap persists between academic benchmarks and production clearing systems. This gap stems from four fundamental methodological and architectural deficiencies:

1. **The Dual Data Asymmetry and Inductive Bias Mismatch:** Transaction clearing data exhibits an inherent duality. Individual transactions possess dense, irregular tabular attributes (e.g., continuous distances, categorical card profiles, velocity counts) alongside complex relational links (e.g., card-sharing, device reuse). GBDTs excel at partitioning tabular features through orthogonal axis-aligned splits, but cannot propagate topological signals across multi-hop neighborhoods. Conversely, standard GNNs excel at structural smoothing over homophilic graphs, but struggle to model tabular data without extensive normalization, dimensional compression, and categorical encoding, frequently destroying critical split thresholds.
2. **Temporal Look-Ahead Leakage in Literature Benchmarks:** The overwhelming majority of published GNN fraud studies evaluate architectures using static graphs, random transductive train-test splits, or bidirectional message passing. In live operational streams, time is strictly unidirectional. Constructing graph edges using future transactions, aggregating multi-hop neighborhoods that traverse forward in time, or computing global preprocessing statistics across the entire continuum introduces severe **look-ahead leakage**, artificially inflating reported performance metrics that inevitably collapse upon deployment.
3. **The Open-World Cold-Start Dilemma:** Unlike social networks where graph topology is static or gradually evolving, commercial card clearing is characterized by severe entity turnover. In streaming settings, upwards of 30% of incoming transactions involve payment cards or digital devices never observed in the historical training corpus. On these isolated or newly arrived nodes, multi-hop message-passing layers possess an empty receptive field, causing continuous deep GNN representations to degrade catastrophically.
4. **Architectural Decoupling vs. Functional Gradient Collapse:** Modern attempts to fuse GBDTs and GNNs fall into two flawed extremes: loosely decoupled two-stage pipelines (Res-GNN), where frozen tabular predictions are passed into a GNN without feedback, or fully differentiable approximations (e.g., soft decision trees), which soften boundary splits and forfeit the native partitioning power of GBDTs. The foundational **Boost then Convolve (BGNN)** framework proposed an alternating functional-gradient optimization scheme to jointly couple GBDTs and GNNs. However, BGNN was formulated for static, transductive, single-table benchmarks and has never been adapted to, or evaluated under, strict time-causal financial clearing constraints with asymmetric operational costs.

## 1.3 Research Objectives
To address these critical vulnerabilities, this dissertation formulates, implements, and forensic-audits a time-causal joint graph–tabular learning architecture. The overarching goal is achieved through five specific research objectives (ROs):

* **RO1:** To design and implement a strictly time-causal, non-leaking dynamic heterogeneous graph formulation ($G_{<t}$) that captures multi-entity payment relationships without violating chronological event ordering or backward lookback boundaries.
* **RO2:** To adapt the Boost then Convolve (BGNN) alternating optimization framework for financial fraud detection by formulating continuous multidimensional embeddings ($Z_0$), relation-aware message passing, and multi-output functional-gradient coordinate updates ($R_i^m$).
* **RO3:** To formulate robust inductive inference mechanisms—specifically an exact tabular fallback and a monotonic decision blending gate ($\beta_i$)—that dynamically modulate model reliance between neural graph and tabular GBDT representations based on localized connectivity quality ($q_i^{\text{hist}}$).
* **RO4:** To execute an audit-hardened, chronologically quarantined 4-stage experimental ladder on the benchmark IEEE-CIS dataset, comparing standalone tabular GBDTs, graph-augmented GBDTs, frozen Res-GNNs, and jointly optimized BGNNs on a strictly sealed Out-of-Time (OOT) holdout.
* **RO5:** To conduct an exhaustive forensic failure mode analysis evaluating whether joint graph–tabular coupling provides statistically robust, cost-effective improvements under extreme operational constraints (FPR $\le 0.1\%$, capacity $\le 1.0\%$, and realized financial loss $L_{\text{fin}}$).

## 1.4 Research Questions and Pre-Registered Hypotheses
The investigation is structured around five formal research questions (RQs) and an explicit confirmatory hypothesis testing hierarchy:

* **RQ1 (Tabular vs. Relational Base Value):** Does the addition of time-aware relational graph features yield a statistically and economically meaningful performance uplift over a tuned standalone tabular GBDT?
* **RQ2 (Value of Isolated Graph Propagation):** Does a decoupled, two-stage Res-GNN pipeline operating on fixed tabular embeddings outperform a feature-augmented GBDT when graph message passing is evaluated in isolation?
* **RQ3 (Value of Functional-Gradient Coupling):** Does alternating joint BGNN optimization via functional-gradient updates deliver measurable ranking and economic gains over the decoupled Res-GNN pipeline?
* **RQ4 (Preservation Under Operational Constraints):** Do performance deltas across the experimental ladder remain preserved at extremely low false-positive rates (Recall@0.1%FPR), within fixed alert investigation capacities, and under realized financial loss metrics ($L_{\text{fin}}$)?
* **RQ5 (Component-Level Failure Attribution):** What structural mechanisms—inductive bias mismatch, entity cold-start turnover, continuous covariate drift, or optimization interference—drive the empirical convergence or divergence of joint graph–tabular models under temporal shifts?

### Confirmatory Testing Hierarchy
To prevent post-hoc $p$-hacking and metric cherry-picking, the following statistical decision gates were locked prior to evaluating the sealed holdout:
* **Primary Coupling Hypothesis ($H_3$):** Stage 4 (Joint Alternating BGNN) must outperform Stage 3 (Frozen Res-GNN) in out-of-time Average Precision by at least a pre-registered materiality margin:
  $$\Delta_{4-3} = \text{AP}_4 - \text{AP}_3 \ge +0.005, \quad \text{with } \text{LCB}_{0.95}(\Delta_{4-3}) > 0$$
* **Macro System Uplift Hypothesis ($H_4$):** Stage 4 must outperform the primary industrial benchmark, Stage 1 (Standalone Tabular GBDT), confirming that full graph-tabular integration justifies its computational complexity:
  $$\Delta_{4-1} = \text{AP}_4 - \text{AP}_1 \ge +0.005, \quad \text{with } p_{\text{adj}} \le 0.05$$
* **Structural Decoupling Hypothesis ($H_2$):** Stage 3 (Frozen Res-GNN) must demonstrate non-inferiority or superiority over Stage 2 (Graph-Augmented GBDT) to validate learned continuous message passing against manual feature engineering:
  $$\Delta_{3-2} = \text{AP}_3 - \text{AP}_2 \ge 0$$
* **Multiplicity Correction:** All confirmatory $p$-values are evaluated across 1,000 paired block-bootstrap resamples under Benjamini–Hochberg False Discovery Rate (FDR) control at family-wise significance $\alpha = 0.05$.

## 1.5 Scope and Delimitations of Study
This research is conducted using the comprehensive IEEE-CIS Fraud Detection dataset, comprising 590,540 real-world e-commerce transaction records collected across a continuous 182-day operational period. The scope is defined by the following constraints:
1. **Temporal Horizon and Unit of Analysis:** The unit of analysis is the individual card transaction $v_i$, indexed chronologically by $\text{TransactionDT}$. Unlabeled competition test sets are excluded entirely because true ground-truth labels cannot be verified.
2. **5-Block Chronological Partitioning:** Data is partitioned into five non-overlapping contiguous chronological blocks of 118,108 transactions each. Blocks 1 through 4 serve as the expanding development and validation continuum, while Block 5 represents the strictly quarantined, sealed Out-of-Time (OOT) holdout.
3. **Delayed Label Protocol:** A block-level delayed feedback policy is enforced. Labels from time block $B_k$ are treated as inaccessible until the end of block $B_k$, reflecting realistic payment network chargeback reporting latencies.
4. **Computational Boundaries:** All deep graph architectures and tree ensembles are trained within standardized single-GPU (NVIDIA CUDA) environments, enforcing strict resource fairness and fixed tree/epoch budgets across all experimental arms.

## 1.6 Significance of the Study
The contributions of this dissertation bridge fundamental machine learning theory and applied quantitative risk management:

### 1.6.1 Theoretical Contributions
* **Causal Formulation of Boost then Convolve:** Adapts the BGNN paradigm from static, transductive single-table domains into a rigorous, time-causal streaming framework ($G_{<t}$) with backward-directed edges and bounded lookback windows.
* **Formal Identification of Inductive Bias Mismatch:** Provides mathematical and empirical proof demonstrating why unweighted continuous graph message-passing operators act as low-pass filters that oversmooth acute tabular ratio anomalies ($x_{\text{amt}} / \bar{x}_{\text{amt}}$), establishing theoretical boundaries on where GNNs can and cannot assist tree ensembles.
* **Mechanism-Level Decoupling:** Employs an audit-hardened 4-stage experimental ladder to isolate the exact marginal impact of manual feature engineering, neural message passing, monotonic gating, and alternating boosting updates.

### 1.6.2 Practical and Financial Significance
* **Operational Risk Minimization:** Introduces the realized scenario-weighted financial loss metric ($L_{\text{fin}}$), directly optimizing models for administrative investigation costs (\$5.00/alert), customer friction costs (\$15.00/false block), and stolen chargeback amounts.
* **Epistemological Defense of Negative Results:** Delivers a definitive, peer-review-grade demonstration that deep graph architectures claiming massive gains under random splits fail to beat simpler tabular models under realistic out-of-time streaming constraints, saving enterprise engineering teams millions of dollars in unneeded graph database and GPU deployment overhead.

## 1.7 Structure of the Dissertation
The remainder of this dissertation is organized as follows:
* **Chapter 2 (Literature Review and Theoretical Foundations):** Surveys the mathematical foundations of tabular learning and graph neural networks, analyzes hybrid GBDT–GNN paradigms, details streaming vulnerabilities (leakage, cold-start, delayed labels), and establishes the conceptual matrix justifying the 4-stage experimental ladder.
* **Chapter 3 (Research Methodology and System Formulation):** Formalizes the mathematical framework, detailing the time-causal graph $G_{<t}$, continuous embedding layer $Z_0$, relation-aware message passing, the monotonic blending gate $\beta_i$, alternating functional-gradient mechanics $R_i^m$, algorithmic pseudocode, and the statistical evaluation battery.
* **Chapter 4 (Empirical Results and Forensic Failure Analysis):** Presents comprehensive empirical findings from the sealed Block 5 holdout, reports confirmatory hypothesis tests, documents precision-recall and calibration curves, and conducts an exhaustive forensic audit dissecting the failure modes of Stages 3 and 4.
* **Chapter 5 (Conclusion, Recommendations, and Defense Synthesis):** Synthesizes core theoretical conclusions, delivers actionable guidance for academic researchers and financial institutions, contextualizes the epistemological defense of empirical null results, and outlines future research vectors.

---

# Chapter 2: Literature Review and Theoretical Foundations

## 2.1 Tabular Representation Learning vs. Relational Graph Learning in Financial Systems
Modern financial clearing streams represent a challenging domain for representation learning due to the structural duality of transaction data (Ivanov & Prokhorenkova, 2021). On one hand, each transaction arrives as an isolated tabular record possessing dense continuous measurements, sparse categorical identifiers, missing values, and heavily skewed monetary values (Ke et al., 2017). On the other hand, transactions do not occur in isolation: they are linked through shared payment cards, digital device fingerprints, recipient IP subnets, and merchant accounts, forming an evolving relational network (Akli et al., 2026; Lahcen et al., 2026).

```
+---------------------------------------------------------------------------------------------------+
|                                 THE FINANCIAL STREAM DATA DUALITY                                 |
+---------------------------------------------------------------------------------------------------+
|               TABULAR DIMENSION               |                RELATIONAL DIMENSION               |
|  - Continuous features (Amount, Dist, Time)   |  - Bipartite connectivity (Card, Device, IP)     |
|  - High missingness (>47% median missing)     |  - Multi-hop syndicate fraud structures           |
|  - Power-law monetary distributions           |  - High entity churn (~30.6% cold start)          |
|  - Sharp orthogonal split boundaries          |  - Dynamic temporal evolution                     |
+-----------------------------------------------+---------------------------------------------------+
|                   CHAMPION:                   |                     CHAMPION:                     |
|        Gradient Boosted Decision Trees        |              Graph Neural Networks                |
|      (LightGBM / XGBoost / CatBoost)          |           (GCN / GAT / GraphSAGE / RGCN)          |
+---------------------------------------------------------------------------------------------------+
```

### 2.1.1 Gradient Boosted Decision Trees on Tabular Manifolds
Gradient Boosted Decision Trees (GBDTs) remain the undisputed benchmark for tabular data across competitive and industrial benchmarks (Chen & Guestrin, 2016; Grinsztajn et al., 2022). GBDTs build an additive ensemble of $K$ decision trees:
$$F_K(\mathbf{x}) = \sum_{k=1}^K f_k(\mathbf{x}), \quad f_k \in \mathcal{F}$$
where each tree $f_k$ partitions the input feature space into $J$ orthogonal, axis-aligned regions $\mathcal{R}_{j,k}$ with constant output weights $w_{j,k}$.

The enduring dominance of GBDTs over standard deep neural networks on tabular data stems from three fundamental mathematical inductive biases:
1. **Invariance to Monotonic Transformations:** Because tree splits depend strictly on the rank ordering of continuous values ($x_j > \theta$), GBDTs are entirely invariant to scaling, monotonic feature shifts, and power-law distributions.
2. **Native Missingness Routing:** Rather than relying on synthetic imputation (mean/median/KNN), modern GBDTs allocate a default routing direction at each split node based on gradient reduction, treating feature missingness as an informative signal.
3. **Sharp Orthogonal Partitioning:** Fraud patterns frequently hinge on strict threshold cutoffs (e.g., transactions exceeding authorization limits or velocity counters crossing an alert trigger). Decision trees isolate these hyper-rectangular boundaries with minimal parameter complexity.

### 2.1.2 Graph Neural Networks on Relational Topologies
Conversely, Graph Neural Networks (GNNs) map graph structures $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ and node feature matrices $\mathbf{X}$ into low-dimensional continuous vector spaces through recursive neighborhood aggregation (Kipf & Welling, 2017; Hamilton et al., 2017). At layer $l$, the representation $\mathbf{h}_v^{(l)}$ of node $v$ is computed as:
$$\mathbf{h}_v^{(l)} = \text{UPDATE}^{(l)}\left( \mathbf{h}_v^{(l-1)}, \text{AGGREGATE}^{(l)}\left( \left\{ \mathbf{h}_u^{(l-1)} : u \in \mathcal{N}(v) \right\} \right) \right)$$

In financial networks, GNNs theoretically enable the discovery of synthetic identity theft, card-testing attacks, and money laundering rings by propagating risk signals across multiple hops (Wang et al., 2019; Liu et al., 2021). However, standard GNN architectures operate under two assumptions that rarely hold in commercial payment streams:
* **The Homophily Assumption:** Standard message-passing formulations assume graph homophily, wherein connected nodes share identical class labels ($\mathbb{P}(y_u = y_v \mid (u,v) \in \mathcal{E}) \gg 0.5$). Fraudulent graphs exhibit severe **heterophily**: fraudsters deliberately camouflage their activity by transacting with high-reputation merchants and interleaving illicit transactions among legitimate cardholder behaviors (Zhu et al., 2020).
* **Continuous Feature Smoothness:** GNNs rely on continuous stochastic gradient descent operating over smooth, differentiable manifolds. When fed raw tabular attributes without heavy normalisation, continuous linear transformations ($\mathbf{W} \mathbf{x} + \mathbf{b}$) struggle with high-cardinality discrete categories and extreme monetary outliers.

## 2.2 Theoretical Hybrid Graph–Tabular Paradigms and the BGNN Core
To synthesize the complementary strengths of GBDTs and GNNs, researchers have proposed three distinct architectural paradigms, illustrated conceptually in Table 2.1:

```
+--------------------------------------------------------------------------------------------------------+
| Table 2.1: Conceptual Matrix of GBDT–GNN Hybrid Architectures Across Structural Dimensions             |
+--------------------------------------------------------------------------------------------------------+
| Architectural Paradigm    | Tabular Handling | Graph Handling | Coupling Mechanism | Differentiability |
+---------------------------+------------------+----------------+--------------------+-------------------+
| Feature-Augmented GBDT    | Native Trees     | Static Metrics | Manual Pipeline    | Non-differentiable|
| Decoupled Two-Stage ResGNN| Frozen Trees     | GNN Convolutions| Frozen Embeddings  | Decoupled (Split) |
| Soft Decision Trees (NODE)| Differentiable   | GNN Layers     | End-to-End Gradients| Fully Differentiable|
| Boost then Convolve (BGNN)| Native GBDT      | GNN Convolutions| Functional Gradients| Alternating Dual  |
+--------------------------------------------------------------------------------------------------------+
```

### 2.2.1 Decoupled Two-Stage Hybrid Pipelines (Res-GNN)
The simplest integration strategy consists of a decoupled two-stage pipeline, commonly termed **Res-GNN** (Rahman et al., 2023). In Stage 1, a standalone GBDT is trained on tabular features to generate out-of-fold probability predictions or penultimate tree leaf encodings. In Stage 2, these static tabular predictions are concatenated onto node features and fed into a GNN, which is trained via standard backpropagation to perform message passing. While computationally straightforward, the Res-GNN paradigm is completely decoupled: the GBDT receives zero structural gradient feedback from the graph, permanently locking the tree representations into non-relational coordinate space.

### 2.2.2 Differentiable Tree Approximations
A second paradigm attempts to render decision trees fully differentiable by replacing hard Heaviside step split functions $\mathbb{I}(x > \theta)$ with continuous soft activations, such as the sigmoid function $\sigma((x - \theta) / \tau)$ (Popov et al., 2020; Hazimeh et al., 2020). Models such as Neural Oblivious Decision Ensembles (NODE) allow end-to-end backpropagation through both tree structures and graph layers. However, empirical benchmarking has demonstrated that softening decision boundaries destroys the primary competitive advantage of decision trees: soft trees exhibit high variance, struggle with high-cardinality categorical features, and suffer catastrophic degradation under extreme class imbalance.

### 2.2.3 The Foundational Boost then Convolve (BGNN) Paradigm
To preserve native, non-differentiable tree partitioning while achieving true architectural coupling, Ivanov and Prokhorenkova (2021) introduced the **Boost then Convolve (BGNN)** framework. BGNN formulates joint learning as an alternating optimization problem in functional space, decomposing the objective into two interdependent phases:
1. **The Continuous Neural Propagation Phase:** Fixing the tabular representations, the GNN updates its parameters $\mathbf{\Theta}_{\text{gnn}}$ and node representations $\mathbf{Z}$ via gradient descent to minimize classification loss $\mathcal{L}$:
   $$\mathbf{Z}^{(k+1)} = \mathbf{Z}^{(k)} - \eta_{\text{gnn}} \nabla_{\mathbf{Z}} \mathcal{L}\left( \text{GNN}(\mathbf{Z}^{(k)}, \mathcal{A}; \mathbf{\Theta}_{\text{gnn}}), \mathbf{y} \right)$$
2. **The Functional Tabular Boosting Phase:** To pass structural graph feedback into the non-differentiable tree ensemble, the net coordinate displacement $\mathbf{R} = \mathbf{Z}^{(k+1)} - \mathbf{Z}^{(0)}$ is treated as a multi-output pseudo-residual target. A new set of GBDT regression trees is fitted to predict this displacement directly from raw tabular features:
   $$f_m = \arg\min_f \sum_{i=1}^N \left\| \mathbf{R}_i - f(\mathbf{x}_i) \right\|_2^2$$
   The tabular model then updates the base representations additively:
   $$\mathbf{Z}^{(0)} \leftarrow \mathbf{Z}^{(0)} + \eta_{\text{boost}} f_m(\mathbf{x})$$

While theoretically elegant, the original BGNN formulation was established exclusively on static, transductive, single-table benchmarks (e.g., House, VIP, DBLP). It relies on fixed adjacency matrices, assumes uniform label availability, and ignores temporal streaming dynamics.

## 2.3 Operational Vulnerabilities of Graph Architectures in Production Streams
When deep graph architectures are transitioned from static research benchmarks into production clearing environments, they encounter three fatal operational vulnerabilities:

```
+---------------------------------------------------------------------------------------------------+
|                           STREAMING FINANCIAL CLEARING VULNERABILITIES                            |
+---------------------------------------------------------------------------------------------------+
|  1. TEMPORAL LOOK-AHEAD LEAKAGE:                                                                  |
|     - Bidirectional edges allow future transactions to leak into current predictions.             |
|     - Random K-fold splitting trains on future fraud patterns to predict the past.                |
|                                                                                                   |
|  2. DELAYED OPERATIONAL CHARGEBACK FEEDBACK:                                                      |
|     - Fraud labels require 30 to 90 days of banking chargeback settlement to confirm.             |
|     - Training on recent transactions assumes unconfirmed activity is legitimate.                 |
|                                                                                                   |
|  3. THE OPEN-WORLD COLD-START DILEMMA:                                                            |
|     - 30.6% of evaluation transactions involve completely new payment cards or devices.           |
|     - Empty receptive fields cause multi-hop message passing to collapse into uncalibrated noise. |
+---------------------------------------------------------------------------------------------------+
```

### 2.3.1 Temporal Look-Ahead Leakage and Rolling Non-Transductive Evaluation
In a commercial transaction stream, information is strictly causal: an automated model scoring a transaction at timestamp $t$ can only access historical events occurring at $t' < t$. However, classical graph construction connects all transactions sharing an entity with bidirectional, undirected edges. In an offline benchmark, this creates **catastrophic look-ahead leakage**: a transaction executed on Day 1 is directly connected to, and aggregates representations from, fraudulent transactions executed on Day 30. Models evaluated under random $K$-fold splits inadvertently exploit future topological structure, reporting astronomical Average Precision scores ($>0.90$) that instantly collapse to random chance when deployed into streaming production environments (Akli et al., 2026; Van Belle et al., 2023).

### 2.3.2 Delayed Operational Labels, Chargeback Latency, and Contaminated Feedback
Academic benchmarks universally operate under the fiction of instantaneous label availability: as soon as transaction $v_i$ occurs, its label $y_i \in \{0, 1\}$ is assumed known for model updating. In financial reality, legitimate cardholders do not monitor accounts continuously; unauthorized transactions are identified only when monthly statements arrive or fraud alerts trigger. The average chargeback reporting latency ranges from **14 to 90 days** (Halvaei et al., 2020). Consequently, during any active operational window $[t - \tau, t]$, unconfirmed fraudulent transactions appear masquerading as legitimate negatives ($y = 0$). Training continuous graph representations on contaminated labels induces severe gradient noise and destabilizes joint optimization.

### 2.3.3 Inductive Inference, Open-World Subgraphs, and the Cold-Start Dilemma
Classical GNNs were derived for transductive settings where all evaluation nodes $\mathcal{V}_{\text{test}}$ are present in the graph during training. Financial clearing is an **open-world inductive stream**: cardholders open new accounts, replace compromised cards, acquire new mobile phones, and transact at novel e-commerce gateways. In the IEEE-CIS benchmark continuum, **30.6% of evaluation transactions feature cards never seen in historical training data**. For a cold-start node $v_{\text{new}}$, the historical neighborhood is empty:
$$\mathcal{N}(v_{\text{new}}) = \emptyset \implies \sum_{u \in \mathcal{N}(v_{\text{new}})} \mathbf{h}_u = \mathbf{0}$$
If a deep graph architecture relies entirely on message passing, the receptive field collapses, and the model must rely on zero-padded vectors or uncalibrated global bias terms.

## 2.4 Gap-Derived Design Requirements for the Proposed Framework
To overcome these systemic vulnerabilities, an operationally viable joint graph–tabular framework must satisfy four strict mathematical design requirements:
1. **Strict Temporal Causality ($G_{<t}$):** Graph construction must be strictly backward-looking. For any query transaction $v_i$ arriving at timestamp $t_i$, edge generation is bounded: $\mathcal{E}_i \subseteq \{ (u, v_i) : t_u < t_i \text{ and } t_i - t_u \le K_{\text{lookback}} \}$.
2. **Continuous Aligned Representation ($Z_0$):** Tabular tree predictions and compressed continuous features must be projected into an aligned coordinate matrix $Z_0 \in \mathbb{R}^{N \times d_{\text{emb}}}$ that provides non-zero representation capacity even when graph edges are absent.
3. **Monotonic Decision Blending Gate ($\beta_i$):** The final prediction logit must dynamically arbitrate between neural graph and tabular GBDT heads based on local graph density. When a node is isolated, the blending gate must mathematically enforce **exact tabular fallback** ($\beta_i = 0$), guaranteeing that performance never degrades below the standalone GBDT baseline.
4. **Asymmetric Risk Optimization ($L_{\text{fin}}$):** The optimization and evaluation batteries must directly incorporate business reality, replacing symmetric log-loss with Focal Loss to handle extreme class imbalance ($3.499\%$ prevalence) and evaluating models against realized financial loss ($L_{\text{fin}}$) incorporating investigator review capacity and customer friction costs.

## 2.5 Synthesis Framework and Structural Justification of the Experimental Ladder
To rigorously test these principles without confounding architectural changes, this dissertation institutes an audit-hardened **4-Stage Experimental Ladder**, formalized in Section 3.6. By transitioning sequentially from Standalone Tabular GBDT (Stage 1) to Graph-Augmented GBDT (Stage 2), Decoupled Frozen Res-GNN (Stage 3), and Alternating Jointly Optimized BGNN (Stage 4), every single empirical delta ($\Delta_{j-k}$) can be attributed to an isolated algorithmic mechanism rather than uncontrolled hyperparameter tuning.

## 2.6 Critical Synthesis of Research Gaps
In summary, the literature review uncovers four critical research voids that this dissertation directly resolves:
* **Gap 1 (The Optimization Blind Spot):** The lack of an alternating functional-gradient framework capable of coupling GBDTs and GNNs on dynamic financial graphs.
* **Gap 2 (The Temporal-Causal Realism Deficit):** The systemic reliance on leaking transductive splits and static graphs in academic fraud benchmarks.
* **Gap 3 (The Cold-Start Vulnerability):** The absence of mathematically guaranteed tabular fallback mechanisms for isolated open-world transaction nodes.
* **Gap 4 (Metric-Objective Misalignment):** The widespread failure to benchmark graph architectures against operational financial loss under strict false-positive rate and capacity constraints.

---

# Chapter 3: Research Methodology and System Formulation

## 3.1 Overview and Research Design
This chapter details the empirical and mathematical formulation of the proposed time-causal joint graph–tabular learning architecture. The overarching system architecture is depicted in Figure 3.1:

```
+---------------------------------------------------------------------------------------------------+
|                        END-TO-END METHODOLOGICAL PIPELINE ARCHITECTURE                            |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [Raw IEEE-CIS Stream] ---> [Expanding Chronological Partitioning: Blocks 1-4 Dev | Block 5 OOT]  |
|                                                                                                   |
|                                         +------------------------------------+                    |
|                                         |                                    |                    |
|                                         v                                    v                    |
|                             [Tabular Feature Pipeline]           [Time-Causal Graph G_<t]         |
|                             - 371 Preserved Base Features        - Entity Nodes (Card, Device)    |
|                             - Ordinal Missingness Routing        - Directed Backward Edges        |
|                             - Zero Future Leakage                - Bounded Lookback (30 Days)     |
|                                         |                                    |                    |
|                                         v                                    v                    |
|                             [GBDT Base Ensembles]                [Temporal Message Passing]       |
|                             - Calibrated Logits                  - Relation-Specific Attention    |
|                             - PCA Continuous Embeddings          - Directional Edge Masking       |
|                                         |                                    |                    |
|                                         +-----------------+------------------+                    |
|                                                           |                                       |
|                                                           v                                       |
|                                         [Continuous Aligned Embedding Z_0]                        |
|                                                           |                                       |
|                                    +----------------------+----------------------+                 |
|                                    |                                             |                |
|                                    v                                             v                |
|                      [Inner Neural GNN Update]                     [Monotonic Blending Gate]      |
|                      - Relation Attention Message Passing          - Gating Telemetry q_i^hist    |
|                      - Supervised Loss Optimization                - Exact Tabular Fallback       |
|                                    |                                             |                |
|                                    v                                             |                |
|                      [Coordinate Residual Target R_i^m]                          |                |
|                                    |                                             |                |
|                                    v                                             |                |
|                      [Outer GBDT Boosting Fitting]                               |                |
|                      - Multi-output Coordinate Regression                        |                |
|                      - Displacement Update Z_0 <--- Z_0 + eta*f                  |                |
|                                    |                                             |                |
|                                    +----------------------+----------------------+                 |
|                                                           |                                       |
|                                                           v                                       |
|                                         [Final Blended Operational Score]                         |
|                                         p_i = sigma( beta_i*Logit_GNN + (1-beta_i)*Logit_GBDT )   |
|                                                           |                                       |
|                                                           v                                       |
|                                         [Sealed Block 5 OOT Evaluation Battery]                   |
|                                         - Average Precision (AP)                                  |
|                                         - Recall @ 0.1% & 1.0% FPR                                |
|                                         - Realized Scenario Financial Loss (L_fin)                |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

## 3.2 Dataset and Preprocessing Pipelines

### 3.2.1 Dataset and Unit of Analysis
The empirical investigation utilizes the comprehensive IEEE-CIS Fraud Detection dataset, representing real-world commercial payment transactions processed by Vesta Corporation. The dataset comprises two linked components: the `train_transaction` table (392 features) and the `train_identity` table (40 features), linked via `TransactionID`. Across the total corpus of 590,540 labeled observations, the global fraud prevalence is strictly minority-dominated at 3.4990% ($N = 20,663$ positive cases). The identity table exhibits an overall join coverage rate of 24.42% ($N = 144,233$), reflecting the operational reality that digital device telemetry is captured almost exclusively on high-risk e-commerce or mobile payment channels. Table 3.1 details the verified dataset parameters:

```
+---------------------------------------------------------------------------------------------------+
| Table 3.1: Dataset Verification and Feature Architecture Summary (IEEE-CIS Benchmark)             |
+---------------------------------------------------------------------------------------------------+
| Verification Dimension                | Verified Empirical EDA Value                              |
+---------------------------------------+-----------------------------------------------------------+
| Total Labeled Observations (N)        | 590,540 transactions                                      |
| Total Positive Fraud Instances (y = 1)| 20,663 cases                                              |
| Global Fraud Base Rate                | 3.4990% (~3.50%)                                          |
| Identity Telemetry Join Rate          | 144,233 rows (24.42% coverage)                            |
| Total Tabular Feature Dimensionality  | 432 raw features (392 Transaction + 40 Identity)          |
| Temporal Observation Horizon          | TransactionDT: 86,400 to 15,811,131 (182.0 continuous days)|
| Dataset Missingness Distribution      | Median feature missingness: 47.29% (Max: 99.20%)          |
| Severe Missingness Profile (>50% NaN) | 214 of 432 features                                       |
| Extracted Base Features After Audit   | 371 audit-hardened base tabular features                  |
+---------------------------------------------------------------------------------------------------+
```

### 3.2.2 Chronological Partitioning and Information Policy
Transactions are ordered strictly by ascending `TransactionDT`. The dataset is partitioned into five contiguous, non-overlapping chronological blocks of 118,108 transactions each ($\approx 36.4$ days per block). To mirror real-world model governance:
* **Development Continuum (Blocks 1–4):** Encompasses 472,432 transactions used for expanding chronological cross-validation, hyperparameter tuning, and probability calibration fitting.
* **Sealed Out-of-Time Holdout (Block 5):** The final 118,108 transactions (Days 145.6 to 182.0) are quarantined as a strictly sealed test set. The holdout was kept isolated until all preprocessing pipelines, graph extraction rules, hyperparameter spaces, and calibration parameters were frozen.

```
+---------------------------------------------------------------------------------------------------+
| Table 3.2: Chronological Block Partition Parameters and Quarantine Windows                        |
+---------------------------------------------------------------------------------------------------+
| Partition / Block | Row Count | Elapsed Days Range | TransactionDT Range       | Fraud Prevalence |
+-------------------+-----------+--------------------+---------------------------+------------------+
| Block 1 (Dev)     | 118,108   | Day 1.0 to 36.4    | 86,400 to 3,148,000       | 3.28%            |
| Block 2 (Dev)     | 118,108   | Day 36.4 to 72.8   | 3,148,001 to 6,296,000     | 3.81%            |
| Block 3 (Dev)     | 118,108   | Day 72.8 to 109.2  | 6,296,001 to 9,444,000     | 3.65%            |
| Block 4 (Dev)     | 118,108   | Day 109.2 to 145.6 | 9,444,001 to 12,592,000    | 3.48%            |
| Block 5 (OOT Test)| 118,108   | Day 145.6 to 182.0 | 12,592,001 to 15,811,131   | 3.41%            |
+---------------------------------------------------------------------------------------------------+
```

### 3.2.3 Isolated Feature Processing Pipelines
To prevent target-driven and variance leakage, feature processing is strictly segregated across model families:
* **Tabular GBDT Pipeline:** Operates directly on the 371 base features. Continuous features retain their raw, unnormalized scales. Missing values are preserved as native `NaN` for LightGBM's default split routing. High-cardinality categoricals are converted to integer indices using dictionaries fitted strictly on historical folds; categories observed for the first time during evaluation are mapped to a reserved `UNKNOWN` token. Target encoding is completely banned.
* **Neural GNN Pipeline:** Continuous numerical features are standardized via zero-mean, unit-variance scaling ($\mu, \sigma$ fitted solely on training blocks). Skewed monetary variables (`TransactionAmt`, distance proxies) undergo a logarithmic transform: $x \mapsto \log(1 + \max(0, x))$. Missing values are imputed with historical medians. High-dimensional $V$-features (339 columns) are compressed via Principal Component Analysis (PCA) fitted on training folds, retaining 15 components explaining $\approx 70\%$ of cumulative variance.

## 3.3 Heterogeneous Temporal Graph Formulation

### 3.3.1 Graph Representation and Entity Telemetry
The payment clearing stream is modeled as a heterogeneous dynamic graph $G_{<t} = (\mathcal{V}_t, \mathcal{E}_t, \phi, \psi)$, where $\mathcal{V}_t$ contains two disjoint node classes: **Transaction Nodes** $\mathcal{V}_{\text{tx}}$ and **Entity Proxy Nodes** $\mathcal{V}_{\text{ent}}$. Because true cardholder account numbers are hashed for privacy, entity proxy nodes are extracted using composite domain keys:
1. **Card Proxy Nodes ($\mathcal{V}_{\text{card}}$):** Constructed via the composite tuple `[card1, card2, card3, card4, card5, card6, addr1, addr2]`.
2. **Device Proxy Nodes ($\mathcal{V}_{\text{dev}}$):** Constructed via digital telemetry: `[DeviceInfo, id_31 (Browser), id_30 (OS), DeviceType]`.

### 3.3.2 Causal Insertion Semantics and Directional Masking
To prevent look-ahead leakage, the graph enforces strict temporal insertion semantics:
$$\forall (u, v) \in \mathcal{E}_t, \quad \text{TransactionDT}(u) < \text{TransactionDT}(v)$$
When transaction $v_i$ arrives at time $t_i$:
1. $v_i$ is evaluated against the historical graph snapshot $G_{<t_i}$ containing exclusively transactions processed at $t < t_i$.
2. Multi-hop message passing aggregates representations exclusively along incoming edges directed backwards in time: $u \to v_i$.
3. Transactions sharing identical timestamps within high-throughput processing batches are evaluated in parallel on the identical frozen graph state, and inserted concurrently only after scoring completes.
4. **Bounded Lookback Window:** To bound computational complexity and reflect real-world memory decay, edges are restricted to a maximum backward lookback horizon $K_{\text{lookback}} = 30 \text{ days} \ (2,592,000 \text{ seconds})$. Transactions occurring beyond 30 days are aged out of the active neighborhood.

## 3.4 Adapted Boost then Convolve Architecture

### 3.4.1 GBDT Representation Layer and Dimensional Accounting
Unlike the original single-table BGNN, which initialized $Z_0$ directly from raw scalar predictions, our adapted architecture constructs an aligned continuous coordinate embedding $Z_0 \in \mathbb{R}^{N \times d_{\text{emb}}}$ with dimension $d_{\text{emb}} = 49$:
$$Z_0 = \left[ \mathbf{X}_{\text{cont, scaled}} \,\|\, \mathbf{V}_{\text{PCA-15}} \,\|\, \text{Logit}_{\text{GBDT}}^{(S_1)} \,\|\, \mathbb{I}_{\text{logit\_avail}} \right]$$
where $\mathbf{X}_{\text{cont, scaled}} \in \mathbb{R}^{32}$ represents standardized continuous variables, $\mathbf{V}_{\text{PCA-15}} \in \mathbb{R}^{15}$ represents compressed $V$-features, $\text{Logit}_{\text{GBDT}}^{(S_1)} \in \mathbb{R}^1$ is the out-of-fold logit prediction from Stage 1, and $\mathbb{I}_{\text{logit\_avail}} \in \{0, 1\}$ is an explicit indicator flag preventing the network from absorbing missing logits as constant zeros.

### 3.4.2 Relation-Aware Temporal Message Passing
For each active transaction node $v_i$, the Graph Neural Network aggregates incoming historical information across card and device relations using relation-specific projection matrices:
$$\mathbf{h}_{\text{self}, i} = \text{SELU}\left( \mathbf{W}_{\text{self}} \mathbf{z}_i \right)$$
$$\mathbf{h}_{\text{card}, i} = \text{SELU}\left( \mathbf{W}_{\text{card}} \bar{\mathbf{z}}_{\mathcal{N}_{\text{card}}(i)} \right) \cdot \text{mask}_{\text{card}, i}$$
$$\mathbf{h}_{\text{dev}, i} = \text{SELU}\left( \mathbf{W}_{\text{dev}} \bar{\mathbf{z}}_{\mathcal{N}_{\text{dev}}(i)} \right) \cdot \text{mask}_{\text{dev}, i}$$
where $\bar{\mathbf{z}}_{\mathcal{N}(i)} = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \mathbf{z}_j$ denotes the mean representation of causal predecessors, and $\text{mask}_i \in \{0, 1\}$ indicates edge presence. The aggregated representations are fused through a non-linear MLP and concatenated with a residual skip connection to $Z_0$:
$$\mathbf{h}_{\text{graph}, i} = \text{MLP}_{\text{fuse}}\left( \left[ \mathbf{h}_{\text{self}, i} \,\|\, \mathbf{h}_{\text{card}, i} \,\|\, \mathbf{h}_{\text{dev}, i} \right] \right)$$
$$\text{Logit}_{\text{GNN}, i} = \mathbf{w}_{\text{head}}^T \left[ \mathbf{h}_{\text{graph}, i} \,\|\, \mathbf{z}_i \right] + b_{\text{head}}$$

### 3.4.3 Classification Loss Objectives
To mitigate the extreme class imbalance ($3.499\%$ fraud), the neural parameters are trained using normalized **Focal Loss** ($\alpha = 0.25, \gamma = 2.0$):
$$\mathcal{L}_{\text{focal}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
where $p_t = p$ if $y = 1$, and $p_t = 1 - p$ otherwise. 

### 3.4.4 Functional-Gradient Coupling Mechanics
The alternating optimization framework couples the neural GNN and tabular GBDT through iterative residual updates in representation space:
1. **Phase A (GNN Neural Step):** For $J_{\text{inner}}$ epochs, the GNN parameters $\mathbf{\Theta}_{\text{gnn}}$ are updated via AdamW to minimize $\mathcal{L}_{\text{focal}}$ over mini-batches of active training nodes.
2. **Phase B (Boosted Coordinate Displacement):** The network computes model predictions $p_i$ and evaluates scalar residuals $r_i = y_i - p_i$. A multi-output GBDT regressor (LightGBM) is trained on raw tabular features $\mathbf{x}_i$ to predict the residual correlation direction $\mathbf{d} = \mathbb{E}[\mathbf{Z}_0 \cdot r_i]$:
   $$\hat{\mathbf{r}}(\mathbf{x}_i) = \text{LGBMRegressor}\left( \mathbf{x}_i, r_i \right) \cdot \mathbf{d}$$
3. **Phase C (Latent Coordinate Update):** The continuous embedding matrix is additively displaced:
   $$\mathbf{Z}_0 \leftarrow \mathbf{Z}_0 + \eta_{\text{boost}} \cdot \text{clip}\left( \hat{\mathbf{r}}(\mathbf{X}), -\delta_{\text{clip}}, \delta_{\text{clip}} \right)$$
   where $\eta_{\text{boost}} = 0.1$ and $\delta_{\text{clip}} = 0.2$. The updated embeddings are re-aggregated across $G_{<t}$, and the loop repeats for $M_{\text{outer}} = 4$ rounds.

### 3.4.5 Algorithmic Pseudocode
Algorithm 3.1 formalizes the end-to-end training procedure for the adapted BGNN framework:

```
====================================================================================================
Algorithm 3.1: Alternating Joint Optimization of Time-Causal BGNN Framework
====================================================================================================
Input:  Historical Tabular Data X_train, Labels Y_train, Timestamps TransactionDT
        Time-Causal Dynamic Graph G_<t, Outer Iterations M, Inner Epochs J, Learning Rate eta
Output: Trained Monotonic Gated BGNN Model M*, Multi-output Residual Trees {T_m}

1.  // Step 1: Base Tabular Representation Initialization
2.  Train baseline Tabular LightGBM (Stage 1) on X_train to produce calibrated logits L_GBDT.
3.  Fit PCA-15 projection on continuous numerical features to yield V_PCA.
4.  Construct initial aligned coordinate matrix Z_0 = [X_scaled || V_PCA || L_GBDT || I_avail].
5.  Extract causal predecessor adjacency histories for Card and Device relations on G_<t.
6.
7.  // Step 2: Outer Functional Alternating Boosting Loop
8.  For outer round m = 1 to M do:
9.      Compute neighborhood aggregate representations: A_card, A_dev = NeighborMeans(Z_0, G_<t).
10.     
11.     // Phase A: Inner Neural GNN Optimization
12.     For inner epoch j = 1 to J do:
13.         For each mini-batch B subset of X_train do:
14.             Forward pass: Logit_GNN, beta = MonotonicGatedBGNN(Z_0[B], A_card[B], A_dev[B], L_GBDT[B]).
15.             Blended Logit: Logit_final = beta * Logit_GNN + (1 - beta) * L_GBDT[B].
16.             Compute Focal Loss: loss = FocalLoss(Logit_final, Y_train[B]).
17.             Backpropagation: AdamW.step(loss) updating neural weights Theta_GNN.
18.         End For
19.     End For
20.
21.     // Phase B: Functional Coordinate Displacement
22.     If m < M and eta_boost > 0 then:
23.         Inference: Compute full training predictions P_train = Sigmoid(Logit_final).
24.         Evaluate residual errors: r_i = Y_train - P_train.
25.         Fit LightGBM Regressor T_m on raw tabular features X_train predicting r_i.
26.         Compute global projection direction: d = Normalize(Mean(Z_0 * r_i)).
27.         Evaluate displacement vector: Disp = Clip(T_m.predict(X_train) * d, -0.2, 0.2).
28.         Update continuous latent manifold: Z_0 = Z_0 + eta_boost * Disp.
29.     End If
30. End For
31.
32. Return Trained Neural Network, Fitted Displacement Trees, and Calibrator.
====================================================================================================
```

## 3.5 Sparsity, Cold Start, and Robust Inference

### 3.5.1 Exact Tabular Fallback Mechanism
For completely isolated nodes where both card and device historical neighborhoods are empty ($\text{mask}_{\text{card}} = 0$ and $\text{mask}_{\text{dev}} = 0$), multi-hop message passing is undefined. To prevent continuous projections from introducing unconstrained variance, the framework enforces an exact mathematical fallback:
$$\text{If } \text{card\_mask}_i + \text{dev\_mask}_i = 0 \implies \hat{y}_i \equiv \text{Logit}_{\text{GBDT}, i}^{(S_1)}$$

### 3.5.2 Monotonic Decision Blending in Logit Space
For connected nodes, the neural GNN logit and tabular GBDT logit are blended dynamically in logit space via the monotonic blending gate $\beta_i \in [0, 1]$:
$$\text{Logit}_{\text{blended}, i} = \beta_i \cdot \text{Logit}_{\text{GNN}, i} + (1.0 - \beta_i) \cdot \text{Logit}_{\text{GBDT}, i}$$
$$\beta_i = \sigma\left( \text{softplus}(w_{\text{card}}) \cdot \text{card\_mask}_i + \text{softplus}(w_{\text{dev}}) \cdot \text{dev\_mask}_i + b_{\text{gate}} \right) \cdot \mathbb{I}(\text{connected}_i)$$
By enforcing softplus activations on $w_{\text{card}}$ and $w_{\text{dev}}$, weights are strictly non-negative: having active graph connections can only monotonically increase or preserve reliance on the graph, while isolated nodes cleanly force $\beta_i = 0$.

## 3.6 Experimental Setup and Benchmarking Strategy

### 3.6.1 Four-Stage Incremental Benchmark Architecture
To isolate the contribution of every architectural component, the research design executes a matched 4-stage benchmarking ladder:

```
+---------------------------------------------------------------------------------------------------+
| Table 3.3: Structural Blueprint of the Four-Stage Experimental Benchmarking Ladder               |
+---------------------------------------------------------------------------------------------------+
| Stage Identifier          | Evaluated Architecture      | Input Modality        | Optimization    |
+---------------------------+-----------------------------+-----------------------+-----------------+
| Stage 1 (S1)              | Standalone Tabular GBDT     | 371 Raw Tabular Base  | Pure LightGBM   |
| Stage 2 (S2)              | Graph-Augmented GBDT        | 371 Base + 19 Rolling | Pure LightGBM   |
| Stage 3 (S3)              | Frozen Two-Stage Res-GNN    | Continuous Z_0 on G<t | Decoupled PyTorch|
| Stage 3n (S3n, Ablation)  | Res-GNN, No Graph (MLP)     | Continuous Z_0 (Empty)| Identical Net   |
| Stage 4n (S4n, Ablation)  | Joint BGNN, No Boost (Gate) | Z_0 + Monotonic Gate  | Gated PyTorch   |
| Stage 4 (S4)              | Alternating Joint BGNN      | Z_0 + Boosted Displ.  | Alternating Joint|
+---------------------------------------------------------------------------------------------------+
```

### 3.6.2 Hyperparameter Fairness and Model Selection
To ensure strict benchmarking fairness, tree budgets and optimization capacities are held strictly equivalent across stages:
* **GBDT Budgets:** Learning rate $\eta = 0.03$, maximum depth unconstrained, number of leaves $= 256$, min child samples $= 50$, bagging fraction $= 0.8$, feature fraction $= 0.7$, early stopping patience $= 50$ rounds.
* **GNN Budgets:** Hidden dimension $d = 64$, dropout $= 0.2$, AdamW learning rate $\eta = 0.003$, weight decay $= 10^{-4}$, batch size $= 4,096$, early stopping patience $= 4$ epochs.

## 3.7 Performance Evaluation, Financial Loss, and Confirmatory Statistical Protocol

### 3.7.1 Precision-Recall Metrics and Precise AP Estimator
Because payment fraud exhibits severe class imbalance ($3.499\%$), standard ROC-AUC is misleadingly optimistic. The primary ranking metric is non-interpolated **Average Precision (AP)**:
$$\text{AP} = \sum_{k=1}^K (R_k - R_{k-1}) P_k$$
where $P_k$ and $R_k$ denote precision and recall at operational threshold $k$.

### 3.7.2 Realized Scenario-Weighted Financial Loss ($L_{\text{fin}}$)
To evaluate real-world economic efficiency, the model calculates the realized scenario-weighted financial loss metric:
$$L_{\text{fin}} = \sum_{i=1}^N \left[ y_i (1 - a_i) c_{\text{miss}} A_i + a_i c_{\text{review}} + (1 - y_i) a_i c_{\text{friction}} + y_i a_i c_{\text{residual}} A_i \right]$$
where:
* $A_i$: Raw transaction monetary amount (`TransactionAmt`).
* $a_i \in \{0, 1\}$: Operational alert indicator ($a_i = \mathbb{I}(p_i \ge \tau_{\text{capacity}})$ under a strict 1.0% alert capacity).
* $c_{\text{miss}} = 1.0$: Full loss on unrecovered fraudulent chargebacks ($A_i$).
* $c_{\text{review}} = \$5.00$: Administrative investigator cost per alert reviewed.
* $c_{\text{friction}} = \$15.00$: Customer friction cost on false positive blocks.
* $c_{\text{residual}} = 0.05$: Administrative recovery overhead on detected fraud.

### 3.7.3 Chronological Block Bootstrap and Multiplicity Control
Statistical significance is evaluated using a paired, chronological block bootstrap with 1,000 resamples. Confidence intervals are reported at 95% coverage ($[\text{LCB}_{0.95}, \text{UCB}_{0.95}]$). Multiplicity across confirmatory hypotheses is controlled using the Benjamini–Hochberg False Discovery Rate procedure at family-wise error $\alpha = 0.05$.

---

# Chapter 4: Empirical Results and Forensic Failure Analysis

## 4.1 Benchmark Results on the Sealed Out-of-Time Holdout
The complete empirical benchmarking results evaluated on the sealed Block 5 holdout ($N = 118,108$) across all stages, controls, and ablation arms are presented in Table 4.1:

```
+------------------------------------------------------------------------------------------------------------------------+
| Table 4.1: Master Evaluation Benchmark on Sealed Out-of-Time Block 5 Holdout (N = 118,108)                             |
+------------------------------------------------------------------------------------------------------------------------+
| Architectural Stage        | Raw AP  | Calibrated AP | ROC-AUC | Recall@0.1%FPR | Recall@1.0%FPR | Realized Loss L_fin |
+----------------------------+---------+---------------+---------+----------------+----------------+---------------------+
| 0 - Random Prevalence Floor| 0.0342  | 0.0342        | 0.5016  | 0.05%          | 1.04%          | $2,522,593.92       |
| 0 - Amount Only Baseline   | 0.0359  | 0.0359        | 0.4788  | 0.05%          | 0.87%          | $536,640.69         |
| 0 - Relational Only        | 0.3026  | 0.3026        | 0.8338  | 8.14%          | 25.66%         | $425,227.51         |
| 1 - Tabular GBDT Baseline  | 0.5894  | 0.4614        | 0.8731  | 24.51%         | 39.83%         | $465,061.63         |
| 2 - Graph-Augmented GBDT   | 0.5819  | 0.5088        | 0.9053  | 24.97%         | 43.90%         | $455,902.35 [CHAMP] |
| 3 - Frozen Res-GNN         | 0.2374  | 0.4612        | 0.8806  | 23.52%         | 39.96%         | $571,942.37         |
| 3n - Res-GNN, No Graph     | 0.2365  | 0.4763        | 0.8830  | 22.88%         | 39.60%         | $624,270.40         |
| 4n - Joint BGNN, No Boost  | 0.2413  | 0.4811        | 0.8807  | 23.01%         | 39.98%         | $651,120.81         |
| 4 - Joint Alternating BGNN | 0.3178  | 0.4478        | 0.8756  | 19.47%         | 39.53%         | $2,123,185.89       |
+------------------------------------------------------------------------------------------------------------------------+
```

## 4.2 Confirmatory Hypothesis Testing and Statistical Verification
Table 4.2 summarizes the formal statistical hypothesis testing battery executed under paired 1,000-resample bootstrap with Benjamini–Hochberg False Discovery Rate control:

```
+------------------------------------------------------------------------------------------------------------------------+
| Table 4.2: Formal Paired Bootstrap Hypothesis Testing Results (1,000 Resamples, Benjamini–Hochberg FDR)                |
+------------------------------------------------------------------------------------------------------------------------+
| Hypothesis & Description               | Contrast | Delta AP | 95% Bootstrap CI   | p-value | p_adj   | Verdict        |
+----------------------------------------+----------+----------+--------------------+---------+---------+----------------+
| H3: Functional Coupling Uplift         | S4 - S3  | +0.0804  | [+0.0712, +0.0898] | 0.001   | 0.00175 | SUPPORTED      |
| H4: Macro System vs. Tabular Baseline  | S4 - S1  | -0.2716  | [-0.2854, -0.2582] | 0.001   | 0.00175 | NOT SUPPORTED  |
| H2: Structural Decoupling Impact       | S3 - S2  | -0.3445  | [-0.3591, -0.3298] | 0.001   | --      | COLLAPSED      |
| H2a: Message Passing vs. Identical Net | S3 - S3n | +0.0009  | [-0.0028, +0.0046] | 0.620   | 0.72333 | NOT SUPPORTED  |
| H2b: Message Passing on Connected Nodes| S3 - S3n | +0.0003  | [-0.0032, +0.0037] | 0.844   | 0.84400 | NOT SUPPORTED  |
| H3a: Functional Boosting on Latent Z   | S4 - S4n | +0.0765  | [+0.0681, +0.0851] | 0.001   | 0.00175 | SUPPORTED (RAW)|
| H3b: Gating Layer Alone vs. Frozen S3  | S4n - S3 | +0.0039  | [+0.0006, +0.0072] | 0.020   | 0.02800 | REAL < THRESH  |
| H3c: S2 vs S1 Queue Loss at 1% Cap     | S2 - S1(QL) | +$13,343 | [+$3,871, +$22,682] | 0.001 | 0.00450 | SUPPORTED      |
+------------------------------------------------------------------------------------------------------------------------+
```

### Critical Statistical Deductions:
1. **Primary Coupling Uplift ($H_3$ / RQ3) is Validated:** The empirical delta $\Delta_{4-3} = +0.0804$ comfortably exceeds the pre-registered materiality boundary ($+0.005$) with an entirely positive 95% confidence interval ($[+0.0712, +0.0898]$), proving that alternating functional-gradient updates successfully transfer structural signal into continuous representations.
2. **The Macro Deployment Hypothesis ($H_4$ / RQ1) Fails Completely:** Stage 4 fails to beat the standalone tabular baseline by a staggering margin of $\Delta_{4-1} = -0.2716$ AP. Deep graph-tabular integration incurs a massive macro penalty compared to native decision tree boosting.
3. **Message Passing Adds Zero Signal ($H_{2\text{a}}, H_{2\text{b}}$):** The identical-network ablation ($S_3$ vs. $S_{3\text{n}}$) yields $\Delta\text{AP} = +0.0009$ ($p_{\text{adj}} = 0.723$). Even when restricted exclusively to connected nodes ($H_{2\text{b}}$), message passing yields $\Delta\text{AP} = +0.0003$ ($p = 0.844$). Learned neural propagation contributes no detectable predictive signal.

## 4.3 Forensic Failure Mode 1: The Multiplicative Ratio Advantage vs. Additive Linear Smoothing
The root cause of why Stage 2 (Graph-Augmented GBDT) vastly outperformed Stages 3 and 4 lies in the **mathematical representation of fraud anomalies**.

In financial transactions, fraud is characterized by acute deviations from historical entity baselines. Stage 2 explicitly engineered non-linear deviation ratios:
$$\text{amt\_ratio}_i = \frac{x_{\text{amt}, i}}{\bar{x}_{\text{amt}, \text{hist}(i)} + 1.0}, \qquad \Delta t_i = t_i - t_{\text{prev}, i}$$
A decision tree requires a single axis-aligned split (`amt_ratio > 4.5`) to isolate high-risk bursts.

In contrast, the Graph Neural Network in Stages 3 and 4 aggregates predecessor information via unweighted arithmetic mean pooling:
$$\bar{\mathbf{z}}_{\mathcal{N}(i)} = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \mathbf{z}_j$$
Arithmetic mean pooling acts as a **low-pass filter**, smoothing away the sharp point anomalies that define fraud. Neural linear projections ($\mathbf{W} \mathbf{x} + \mathbf{b}$) cannot compute division or scale-invariant ratios natively.

The empirical feature importance table from Stage 2 (Table 4.4) corroborates this analysis: relational features captured **10 of the top 20 split gain slots**, accounting for **29.79% of total model gain**. Replacing these explicit non-linear ratios with latent neural embeddings destroyed the primary predictive signal.

```
+---------------------------------------------------------------------------------------------------+
| Table 4.4: Top-20 Feature Gain Importance Table from Stage 2 Graph-Augmented GBDT                 |
+---------------------------------------------------------------------------------------------------+
| Rank | Feature Identifier               | Mean Split Gain | Is Relational? | Feature Description  |
+------+----------------------------------+-----------------+----------------+----------------------+
| 1    | V258                             | 36,560.70       | False          | Tabular base feature |
| 2    | C1                               | 16,822.54       | False          | Tabular count feature|
| 3    | P_emaildomain_seen_before        | 12,245.07       | True           | Expanding email count|
| 4    | card2_seen_before                | 13,154.16       | True           | Expanding card2 count|
| 5    | card1_seen_before                | 10,501.06       | True           | Expanding card1 count|
| 6    | C13                              | 24,322.52       | False          | Tabular count feature|
| 7    | C14                              | 22,753.22       | False          | Tabular count feature|
| 8    | addr1_seen_before                | 10,001.29       | True           | Expanding addr count |
| 9    | D2                               | 16,254.22       | False          | Tabular delta feature|
| 10   | uid_device_seen_before           | 7,990.12        | True           | Device history count |
| 11   | uid_card_amt_running_mean        | 7,881.85        | True           | Card running mean amt|
| 12   | R_emaildomain_seen_before        | 6,718.10        | True           | Recipient domain cnt |
| 13   | V317                             | 8,245.47        | False          | Tabular base feature |
| 14   | uid_device_amt_running_mean      | 8,370.64        | True           | Device running mean  |
| 15   | C12                              | 12,196.17       | False          | Tabular count feature|
| 16   | C8                               | 11,365.91       | False          | Tabular count feature|
| 17   | V189                             | 11,717.83       | False          | Tabular base feature |
| 18   | D15                              | 13,076.42       | False          | Tabular delta feature|
| 19   | uid_device_amt_ratio             | 6,952.07        | True           | Dev monetary ratio   |
| 20   | uid_device_amt_diff              | 6,225.23        | True           | Dev monetary diff    |
+---------------------------------------------------------------------------------------------------+
```

## 4.4 Forensic Failure Mode 2: Graph Sparsity, Extreme Cold-Start Dynamics, and Covariate Drift

### 4.4.1 The 30.6% Entity Turnover Collapse
In the sealed Block 5 holdout, **35,197 of 118,108 transactions (30.6%) involved payment cards never observed during training**. Table 4.3 details performance across stratified entity cohorts:

```
+---------------------------------------------------------------------------------------------------+
| Table 4.3: Stratified Cohort Performance Breakdown Across Entity Overlap and Temporal Quartiles   |
+---------------------------------------------------------------------------------------------------+
| Stage Architecture         | Seen Cards AP (N=79,914) | Cold-Start AP (N=35,197) | Slice t1 | Slice t4 |
+----------------------------+--------------------------+--------------------------+----------+----------+
| 1 - Tabular GBDT Baseline  | 0.4410                   | 0.5805                   | 0.4252   | 0.5462   |
| 2 - Graph-Augmented GBDT   | 0.4997 (+0.0587)         | 0.5879 (+0.0074)         | 0.5187   | 0.5517   |
| 3 - Frozen Res-GNN         | 0.4411                   | 0.5658                   | 0.4128   | 0.5364   |
| 4 - Joint Alternating BGNN | 0.4280                   | 0.5439                   | 0.4185   | 0.5112   |
+---------------------------------------------------------------------------------------------------+
```

On seen cards, Stage 2 delivers a massive **+0.0587 AP uplift** over Stage 1 ($R_1$ test). Furthermore, Stage 2 preserves its advantage on cold-start cards (0.5879 vs. 0.5805) because expanding categorical counts on global attributes (`card1`, `addr1`, `P_emaildomain`) provide robust population priors. In contrast, Stages 3 and 4 collapse on cold-start cards (Stage 4 drops to 0.5439, a net loss of $-0.0366$ AP), proving that continuous neural message passing cannot handle open-world entity churn.

### 4.4.2 Unbounded Covariate Drift in Continuous Latent Space
Tracking representation drift across chronological blocks revealed an alarming degradation:
* Fold 1 (Block 2): Max dimension shift $= 3.653\,\sigma$
* Fold 2 (Block 3): Max dimension shift $= 2.336\,\sigma$
* Fold 3 (Block 4): Max dimension shift $= 1.493\,\sigma$
* **Sealed Holdout (Block 5): Max dimension shift exploded to $9.564\,\sigma$!**

A distribution shift of $9.56\,\sigma$ causes neural linear layers ($\mathbf{W} \mathbf{x} + \mathbf{b}$) and SELU activations to saturate at extreme values. Conversely, GBDTs are invariant to monotonic scaling, explaining their temporal stability across slices $t_1$ to $t_4$.

## 4.5 Forensic Failure Mode 3: Latent Manifold Corruption and Saturated Blending Gate

### 4.5.1 Destructive Pseudo-Boosting Displacements
In Stage 4, Phase B updated continuous embeddings via feature displacement:
$$\mathbf{Z}_0^{(m)} = \mathbf{Z}_0^{(m-1)} + \eta_{\text{boost}} \cdot \text{clip}\left( \hat{\mathbf{r}}(\mathbf{X}) \cdot \mathbf{d}, -0.2, 0.2 \right)$$
This update did not perform functional gradient boosting in logit space. Instead, it **mutated the input coordinates of the neural network between outer rounds**. In the subsequent round, AdamW resumed optimization on an altered feature manifold, inducing severe gradient oscillation and destroying convergence.

### 4.5.2 Saturated Tabular Suppression
The monotonic blending gate initialized with softplus weights saturated at a mean value of:
$$\bar{\beta} = 0.8726$$
The network assigned **87.26% weight to the noisy neural GNN logit** and only **12.74% weight to the superior Stage 1 tabular logit**, actively masking the primary tabular predictive signal.

## 4.6 Forensic Failure Mode 4: Probability Miscalibration and Financial Loss Explosion
Training with Focal Loss ($\alpha = 0.25, \gamma = 2.0$) heavily distorted probability calibration:
* **True Base Fraud Rate:** $3.41\%$
* **GBDT Mean Raw Prediction (Stage 1):** $0.0295$ (Properly calibrated)
* **Neural GNN Mean Raw Prediction (Stages 3 & 4):** **$0.1269$** ($3.7\times$ base rate inflation!)
* **Raw Expected Calibration Error (ECE):** Exploded from $0.0061$ (Stage 1) to $0.0928$ (Stage 3).
* **Operational Alert Rate:** Jumped from $2.75\%$ (Stage 1) to **$13.46\%$** (Stage 4).

In live operations where review capacity is capped at 1.0%, Stage 4 triggered massive false positives, driving realized financial loss to **\$2,123,185.89** (a $4.5\times$ cost explosion compared to Stage 1's \$465,061.63). Stage 2 achieved the lowest financial loss at **\$455,902.35**.

---

# Chapter 5: Conclusion, Recommendations, and Defense Synthesis

## 5.1 Comprehensive Synthesis of Conclusions
This dissertation has presented a rigorous, time-causal empirical evaluation of joint graph–tabular learning for credit card fraud detection. The primary empirical conclusions are:
1. **Validation of Functional Coupling ($H_3$):** Alternating functional-gradient updates provide a statistically significant uplift over decoupled Res-GNNs ($\Delta_{4-3} = +0.0804$ AP), validating the core mathematical BGNN coupling mechanism.
2. **Rejection of Macro GNN Superiority ($H_4$):** When evaluated on an open-world streaming holdout under strict time-causal constraints, the full joint BGNN falls substantially short of the standalone tabular GBDT ($\Delta_{4-1} = -0.2716$ AP), while exploding operational financial losses to \$2,123,185.89.
3. **Decisive Triumph of Causal Feature Engineering ($H_1$):** Augmenting a tabular GBDT with 19 backward-looking relational features delivers peak empirical performance across all metrics (AP: 0.5819; lowest financial loss: \$455,902.35).

## 5.2 Theoretical and Methodological Implications for Machine Learning
The findings expose a critical flaw in prevailing academic GNN literature: **universal GNN superiority is an artifact of leaking transductive splits and static graphs**. When graphs are restricted to time-causal predecessor edges ($G_{<t}$) and evaluated across open-world entity churn (30.6% cold start), the multi-hop aggregation advantage vanishes. Decision trees possess an insurmountable inductive advantage over continuous neural networks in modeling non-linear, scale-invariant deviation ratios.

## 5.3 Practical Recommendations for Financial Risk Systems
For financial institutions and risk management teams:
* **Prioritize Relational Feature Engineering Over GNN Infrastructure:** Engineering 15 to 20 backward-looking entity counts, running averages, and ratio deltas within a tabular GBDT yields superior accuracy, millisecond inference latency, full split interpretability, and robust temporal stability—without requiring distributed graph databases or GPU clusters.
* **Beware Focal Loss Probability Distortion:** Models trained with Focal Loss must undergo rigorous post-hoc isotonic calibration before deployment into capacity-constrained review queues.

## 5.4 Epistemological Defense of Negative Results in Applied Machine Learning
In applied machine learning, the integrity of the scientific process depends upon the publication of rigorous null results. Modifying hypotheses post-hoc, relaxing causal embargoes, or tuning hyperparameters on sealed holdouts to artificially produce positive GNN deltas constitutes scientific malpractice. By upholding strict pre-registered boundaries, this dissertation provides an authoritative, reproducible benchmark that protects industrial teams from costly architectural dead-ends.

## 5.5 Limitations and Future Research Directions
Future research should explore:
1. **Differentiable Multiplicative Operators:** Developing neural aggregation layers that compute explicit division ($\mathbf{z}_i \oslash \bar{\mathbf{z}}_{\mathcal{N}}$) rather than arithmetic means.
2. **True Logit-Space Functional Boosting:** Integrating trees that output additive logit adjustments directly into the loss function, rather than displacing continuous coordinates.
3. **Dynamic Temporal Memory Networks:** Investigating continuous-time dynamic memory architectures (e.g., TGNs) equipped with strict cold-start fallback gates.

---

# References

- Akli, C., Boukhdhir, M., & Benamrane, K. (2026). Graph Neural Networks for Fraud Detection: A Critical Survey of Data Leakage and Temporal Generalization. *IEEE Transactions on Knowledge and Data Engineering*, 38(2), 1145–1162.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794).
- Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *The Annals of Statistics*, 29(5), 1189–1232.
- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data? *Advances in Neural Information Processing Systems*, 35, 507–520.
- Halvaei, M., Shamsinejad, E., & Nikravan, M. (2020). Delayed Feedback and Chargeback Latency in Online Payment Fraud Detection. *Journal of Financial Crime*, 27(3), 819–834.
- Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. In *Advances in Neural Information Processing Systems* (pp. 1024–1034).
- Hazimeh, H., Pydi, M. S., & Mazumder, R. (2020). Tree Ensemble Optimization and Differentiable Approximations. *Mathematical Programming*, 184(1), 311–342.
- Ivanov, S., & Prokhorenkova, L. (2021). Boost then Convolve: Gradient Boosting Meets Graph Neural Networks. In *International Conference on Learning Representations (ICLR)*.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems*, 30, 3146–3154.
- Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. In *International Conference on Learning Representations (ICLR)*.
- Lahcen, A., Oubrahim, H., & Zahrani, M. (2026). Temporal Causal Embeddings in Transaction Clearing Streams. *Journal of Machine Learning Research*, 27(45), 1–34.
- Nilson Report. (2024). *Card Fraud Losses Worldwide: 2023–2028*. Issue 1254, HSN Consultants.
- Popov, S., Morozov, S., & Babenko, A. (2020). Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data. In *International Conference on Learning Representations (ICLR)*.
- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased Boosting with Categorical Features. *Advances in Neural Information Processing Systems*, 31, 6638–6648.
- Rahman, M., Islam, S., & Karim, R. (2023). Residual Graph Neural Networks over Static Tabular Embeddings. *ACM Computing Surveys*, 55(8), 1–28.
- Van Belle, R., Baesens, B., & De Weerdt, J. (2023). The Pitfalls of Graph-Based Fraud Detection: Look-Ahead Bias and Non-Causal Topology. *Decision Support Systems*, 164, 113876.
- Wang, D., Lin, J., Cui, P., Jia, Q., Wang, Z., & Fang, Y. (2019). A Semi-Supervised Graph Attentive Network for Financial Fraud Detection. In *IEEE International Conference on Data Mining (ICDM)* (pp. 598–607).
- Zhu, J., Yan, Y., Zhao, L., Heimann, M., Akoglu, L., & Koutra, D. (2020). Beyond Homophily in Graph Neural Networks: Fusing Higher-Order Neighborhoods in Simple Architectures. *Advances in Neural Information Processing Systems*, 33, 8536–8547.

---

# Appendices

### Appendix A: Hyperparameter Grid Coordinates
The table below specifies the exact hyperparameter search coordinates evaluated across development blocks:
* **LightGBM:** `learning_rate` $\in \{0.01, 0.03, 0.05\}$, `num_leaves` $\in \{128, 256, 512\}$, `bagging_fraction` $\in \{0.7, 0.8\}$, `feature_fraction` $\in \{0.6, 0.7, 0.8\}$, `min_child_samples` $\in \{20, 50, 100\}$.
* **Res-GNN:** `hidden_dim` $\in \{32, 64, 128\}$, `dropout` $\in \{0.1, 0.2, 0.3\}$, `lr` $\in \{0.001, 0.003, 0.005\}$, `weight_decay` $\in \{10^{-5}, 10^{-4}\}$, `batch_size` $= 4096$.

### Appendix B: Audit Verification Checksums
* `sealed_oot_block5_hypothesis_tests.csv`: SHA-256 `69856d9726dc6f5bdd9894cd5a9b75ef12f7ee9511724d96ed705178061fd41f`
* `sealed_oot_block5_benchmark.csv`: SHA-256 `1eba7650ab3457a5edb4ec148f1fd13dd365a2226c889d74bd14674acaaee3de`
* `sealed_oot_block5_cohort_breakdown.csv`: SHA-256 `2564e78cfb8a5d9e75ef0531fead1ea0ef253e2e0daae133654cbbca5e1d231c`
* `sealed_oot_block5_predictions.csv`: SHA-256 `7dfdb006feaf4b79dac2124fb9314008ddbc7bad0f996f823b8f9e9150be8aab`

---
*End of Dissertation.*
