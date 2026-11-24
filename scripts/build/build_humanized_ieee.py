import re
# -*- coding: utf-8 -*-
"""
build_humanized_ieee.py
Builds IEEE_Technical_Paper_Ieman_Ihsan_25005466_humanized.docx with humanized text.
"""

import os
import docx
from docx.shared import Mm, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

import build_ieee_technical_paper as bip

# Map of index in DOCUMENT_FLOW to humanized text
HUMANIZED_P = {
    2: (
        "Electronic clearing networks process billions of card transactions daily. Over the same period, "
        "card fraud has expanded, with annual worldwide losses exceeding $33.8 billion {REF:NILSON}. "
        "In production clearing systems, automated fraud screening engines operate under strict latency limits, "
        "requiring incoming authorizations to be evaluated in flight within sub-100-millisecond budgets. "
        "In addition, ground-truth labels are delayed: fraudulent transactions are confirmed through chargeback "
        "disputes only 30 to 90 days after authorization {REF:DAL_POZZOLO}. Models running in streaming payment "
        "environments therefore cannot rely on immediate ground truth or future transaction data."
    ),
    4: (
        "Payment transaction streams exhibit two distinct data structures. First, transactions arrive as tabular "
        "records containing transaction amounts, velocity counts, geographic codes, and high-cardinality categorical "
        "identifiers where missingness frequently exceeds 40%. For these tabular features, Gradient Boosted Decision "
        "Trees (GBDTs), such as LightGBM {REF:LIGHTGBM}, serve as the standard baseline due to their invariance to "
        "monotonic transformations, learned default routing for missing values, and orthogonal axis-aligned "
        "partitioning {REF:GRINSZTAJN}."
    ),
    5: (
        "Second, transactions are linked through shared payment entities, including cards, terminals, devices, "
        "and IP addresses. Graph Neural Networks (GNNs) provide an established approach for modeling these "
        "relational structures {REF:VAN_BELLE}, {REF:POURHABIBI}, {REF:MA}. Most academic GNN benchmarks, however, "
        "construct static, transductive graphs that connect nodes bidirectionally across time, introducing "
        "temporal look-ahead leakage {REF:VAN_BELLE}."
    ),
    7: (
        "Deploying graph architectures into production fraud detection highlights four practical vulnerabilities:"
    ),
    8: (
        "1) Architectural Optimization Separation: Two-stage hybrid pipelines, such as Residual Graph Neural Networks "
        "(Res-GNN) {REF:IVANOV}, feed fixed tabular representations into GNN layers without backpropagating structural "
        "gradients into the tree ensemble. Conversely, differentiable decision trees, including Neural Oblivious Decision "
        "Ensembles {REF:POPOV}, TabNet {REF:ARIK}, and Tree Ensemble Layers {REF:HAZIMEH}, approximate step splits with "
        "smooth sigmoidal activations, which degrades partition accuracy on tabular boundaries."
    ),
    9: (
        "2) Temporal Look-Ahead Leakage: Academic GNN benchmarks frequently evaluate static graphs with random cross-validation. "
        "In this setting, early transactions aggregate relational embeddings from future transactions associated with the same "
        "cardholder, inflating reported metrics {REF:AKLI}, {REF:ROSSI}."
    ),
    10: (
        "3) Open-World Cold-Start Conditions: High entity turnover is common in transaction streams. In production payment "
        "clearing, over 30% of transactions involve cards or devices not present in historical training data. When relational "
        "neighborhoods are empty, multi-hop message passing has no historical edges to aggregate, leaving neural representations "
        "uninformative {REF:AKLI}."
    ),
    11: (
        "4) Objective and Cost Misalignment: Standard cross-entropy objectives treat misclassification symmetrically, "
        "overlooking extreme class imbalance (~3.5% fraud). They also neglect operational queue limits, where fraud triage "
        "teams can manually review at most 1.0% of transaction volume."
    ),
    13: (
        "To evaluate these issues under production constraints, this study formulates an adapted Boost then Convolve (BGNN) "
        "framework {REF:IVANOV} for streaming transaction data. The research addresses five specific objectives: "
        "(1) specify a strictly time-causal dynamic heterogeneous graph $G_{<t}$ that eliminates future look-ahead edges; "
        "(2) establish the performance baseline of tuned tabular GBDTs (Stage 1); "
        "(3) measure the empirical gain from historical graph feature engineering within GBDTs (Stage 2); "
        "(4) evaluate decoupled neural message passing over fixed tabular embeddings (Stage 3); and "
        "(5) test whether alternating functional-gradient optimization between decision trees and graph convolutions "
        "with an adaptive fallback blending gate $\\beta_i$ (Stage 4) improves upon tabular baselines under Out-of-Time (OOT) "
        "evaluation on the IEEE-CIS benchmark ($N = 590,540$ transactions)."
    ),
    16: (
        "GBDTs build on the greedy function approximation framework of Friedman {REF:FRIEDMAN}, implemented in libraries "
        "such as XGBoost {REF:XGBOOST}, LightGBM {REF:LIGHTGBM}, and CatBoost {REF:CATBOOST}. They construct an additive "
        "ensemble of $K$ decision trees, $F_K(\\mathbf{x}) = F_0(\\mathbf{x}) + \\sum_{k=1}^K \\eta f_k(\\mathbf{x})$, "
        "where each tree partitions the input space into orthogonal hyper-rectangles. Grinsztajn et al. {REF:GRINSZTAJN} showed "
        "that GBDTs remain competitive with deep neural models on tabular datasets due to their invariance to monotonic "
        "feature transformations, native handling of missing values via learned default split directions, and capability "
        "to represent sharp threshold discontinuities. However, standard GBDTs treat transactions as independent and identically "
        "distributed (i.i.d.) observations, without mechanisms to aggregate relational dependencies across accounts "
        "{REF:POURHABIBI}, {REF:MA}."
    ),
    17: (
        "Conversely, Message Passing Neural Networks (MPNNs) {REF:GILMER}, including Graph Convolutional Networks (GCN) {REF:KIPF}, "
        "GraphSAGE {REF:HAMILTON}, and Graph Attention Networks (GAT) {REF:VELICKOVIC}, aggregate representations across "
        "node neighborhoods: $\\mathbf{h}_v^{(l)} = \\text{UPDATE}(\\mathbf{h}_v^{(l-1)}, \\text{AGG}(\\{\\mathbf{h}_u^{(l-1)} : u \\in \\mathcal{N}(v)\\}))$. "
        "The expressive capacity of standard MPNNs is bounded by the 1-Weisfeiler-Lehman (1-WL) graph isomorphism test "
        "{REF:MORRIS}, {REF:XU}. In transaction networks, MPNNs often encounter low homophily {REF:ZHU_HOMOPHILY}. Because fraudulent "
        "transactions are interleaved among normal cardholder activity across merchants and devices, financial transaction graphs "
        "exhibit heterophily. In these structures, spatial averaging functions as a low-pass filter, smoothing anomalous fraud "
        "signals into the surrounding legitimate transactions {REF:ZHU_HOMOPHILY}."
    ),
    19: (
        "Several hybrid designs attempt to combine tabular and graph learning. Two-stage pipelines (Res-GNN) {REF:IVANOV} feed "
        "pre-trained tabular embeddings into downstream GNN layers, but prevent tree splits from adjusting to errors identified "
        "during graph propagation. Differentiable tree models, including Neural Oblivious Decision Ensembles (NODE) {REF:POPOV}, "
        "TabNet {REF:ARIK}, and Tree Ensemble Layers {REF:HAZIMEH}, enable gradient backpropagation through soft continuous "
        "routing, but frequently encounter training instabilities on high-cardinality categorical data."
    ),
    20: (
        "The Boost then Convolve (BGNN) framework by Ivanov and Prokhorenkova {REF:IVANOV} introduced an alternating scheme "
        "between GBDT tree fitting and GNN message passing on static homogeneous graphs. In each outer iteration, the GNN trains "
        "neural weights and evaluates functional gradients with respect to node representation coordinates: "
        "$\\mathbf{R}_i^m = -\\nabla_{\\mathbf{z}_i}\\mathcal{L}$. Decision trees then fit these residual vectors to update "
        "coordinates additively: $\\mathbf{z}_i^m = \\mathbf{z}_i^{m-1} + \\eta_{\\text{boost}} F_{\\text{tree}}(\\mathbf{x}_i)$ {REF:IVANOV}."
    ),
    22: (
        "Recent surveys {REF:POURHABIBI}, {REF:MA} and empirical evaluations {REF:DOU}, {REF:WANG_GAT}, {REF:VAN_VLASSELAER}, "
        "{REF:WEBER}, {REF:LAHCEN}, {REF:TANG}, {REF:MAO} document relational methods for fraud detection. Several benchmark "
        "designs, however, diverge from production conditions. Akli et al. {REF:AKLI} and Van Belle et al. {REF:VAN_BELLE} showed "
        "that static graph partitioning introduces future look-ahead information into training sets. While dynamic formulations "
        "like Temporal Graph Networks (TGN) {REF:ROSSI} maintain causal ordering, their memory state requirements can challenge "
        "sub-100-millisecond clearing latency limits."
    ),
    24: (
        "To adapt BGNN to streaming transaction constraints, Table I summarizes the design differences between canonical BGNN "
        "{REF:IVANOV} and the time-causal formulation evaluated in this study."
    ),
    28: (
        "The empirical study uses the IEEE-CIS fraud detection benchmark, comprising 590,540 verified transactions over 182 days "
        "(TransactionDT in [86,400, 15,811,131]) with a 3.499% overall fraud prevalence. The data is partitioned into five "
        "chronological blocks of 118,108 transactions each. Blocks 1 through 4 serve as the expanding training and validation sets. "
        "Block 5 (covering Days 140.12 to 182.00, with 115,111 post-embargo transactions and 3,929 frauds) is held out as a sealed "
        "Out-of-Time (OOT) test set, evaluated only once. A 1-day embargo window and a 30-day chargeback latency buffer prevent "
        "temporal leakage across boundaries {REF:DAL_POZZOLO}."
    ),
    30: (
        "At query timestamp $t_i$, the historical graph $G_{<t_i} = (\\mathcal{V}_{<t_i}, \\mathcal{E}_{<t_i})$ contains only "
        "entities and edges recorded at timestamps $t < t_i$. Entities are resolved using deterministic composite keys: card "
        "entities $\\text{CardID} = \\langle\\text{card1}, \\dots, \\text{card6}, \\text{addr1}\\rangle$ and device entities "
        "$\\text{DeviceID} = \\langle\\text{DeviceInfo}, \\text{id\\_31}, \\text{id\\_30}, \\text{DeviceType}\\rangle$. Each "
        "incoming transaction $v_i$ is evaluated within an ephemeral query graph $G_i$ that draws directed edges from historical "
        "nodes within a 30-day lookback window ($K_{\\text{lookback}} = 2{,}592{,}000\\,\\text{s}$). After inference completes, "
        "$v_i$ is committed to the persistent graph."
    ),
    32: (
        "The model initializes an aligned continuous representation matrix $\\mathbf{Z}_0 \\in \\mathbb{R}^{N \\times 49}$ "
        "consisting of 32 standardized continuous attributes, 15 principal components capturing 69.5% of variance from the "
        "behavioral features, one out-of-fold cross-validated LightGBM risk logit, and a binary identity availability indicator. "
        "Message passing computes representations through relation-specific projections:"
    ),
    34: (
        "To address class imbalance, neural parameters are optimized using Normalized Mini-Batch Focal Loss {REF:LIN_FOCAL}:"
    ),
    36: (
        "where $\\alpha_t = 0.25$ and $\\gamma = 2.0$. In canonical BGNN {REF:IVANOV}, node coordinates $\\mathbf{Z}_0$ are updated "
        "by fitting trees directly to feature-level loss gradients $-\\nabla_{\\mathbf{z}_i}\\mathcal{L}$. To maintain computational "
        "tractability over streaming data, this implementation uses a Rank-1 Projected Functional Coordinate Update. In each outer "
        "boosting round $m \\in \\{1, \\dots, M\\}$ ($M = 4$), a scalar regression tree $T_m$ fits the cross-entropy residuals "
        "$r_i = y_i - \\sigma(\\text{logit}_i)$. The predicted residual is projected along the empirical covariance direction "
        "$\\mathbf{d} = \\text{Normalize}(\\mathbb{E}[\\mathbf{Z}_0 \\cdot r_i]) \\in \\mathbb{R}^{49}$ and bounded to "
        "[$-\\delta_R, \\delta_R$] with $\\delta_R = 0.20$ and $\\eta_{\\text{boost}} = 0.10$:"
    ),
    39: (
        "To protect cold-start entities from missing structural context, a monotonic decision blending gate $\\beta_i$ arbitrates "
        "between GNN predictions and the base GBDT logits:"
    ),
    41: (
        "where $c_i, d_i \\in \\{0, 1\\}$ denote historical card and device connectivity flags, and "
        "$\\mathbb{I}(\\text{connected}_i) = \\mathbb{I}(c_i + d_i > 0)$. For the 796 strictly isolated holdout transactions (0.69%), "
        "$\\mathbb{I}(\\text{connected}_i) = 0 \\implies \\beta_i \\equiv 0$, providing an exact mathematical fallback to Stage 1. "
        "On connected nodes (99.31%), the gate converges to an average weighting of $\\bar{\\beta} = 0.8726$ toward the neural head."
    ),
    43: (
        "Models are assessed using non-interpolated trapezoid-free Average Precision (AP), Recall at fixed operational false-positive "
        "rates (Recall@0.1%FPR, Recall@1.0%FPR), and realized scenario-weighted financial loss ($L_{\\text{fin}}$):"
    ),
    45: (
        "where $A_i$ is transaction amount, $a_i = \\mathbb{I}(p_i \\ge \\theta)$ is the alert decision, and financial parameters "
        "are evaluated under two operational regimes. The Primary Regime enforces a fixed 1.0% queue capacity constraint "
        "($K = 1{,}151$ alerts) with $c_{\\text{miss}} = 1.0$, $c_{\\text{review}} = \\$5.00$, $c_{\\text{friction}} = \\$15.00$, "
        "and $c_{\\text{residual}} = 0.05$. The Sensitivity Regime applies an unconstrained Bayes-optimal threshold rule with "
        "$c_{\\text{review}} = \\$10.00$ and $c_{\\text{friction}} = \\$25.00$. Statistical significance is evaluated using 1,000 "
        "paired 24-hour stationary block bootstrap resamples under Benjamini-Hochberg False Discovery Rate (FDR) control at $\\alpha = 0.05$."
    ),
    48: (
        "Table II summarizes benchmark performance on the sealed Block 5 holdout ($N = 115,111$ evaluated post-embargo transactions, "
        "including 3,929 frauds). Under the primary 1.0% operational queue constraint ($K = 1{,}151$ alerts), Stage 1 incurs a realized "
        "loss of \\$507,989.81, intercepting 1,017 frauds, while achieving an Average Precision (AP) of 0.4814 (calibrated AP 0.4614) "
        "and ROC-AUC of 0.8731."
    ),
    49: (
        "Stage 2 (Graph-Augmented GBDT) achieves the strongest overall performance across the evaluated models. Adding 19 backward-looking "
        "relational features lowers Queue Loss to \\$495,307.56 (intercepting 1,026 frauds), saving \\$12,682.25 compared with Stage 1 "
        "(95% bootstrap CI: [\\$3,678.70, \\$21,541.70], $p = 0.0010$). Under unconstrained Bayes-optimal thresholding, Stage 2 also "
        "records the lowest loss at \\$345,402.31, saving \\$20,326.68 over Stage 1. Stage 2 produces the highest calibrated AP at "
        "0.5088 (raw AP 0.5252, ROC-AUC 0.9053, Rec@1.0% 43.90%), confirming Hypothesis $H_1$ ($p_{\\text{adj}} = 0.0080$)."
    ),
    50: (
        "In comparison, the neural models obtain modest dollar savings within the top 1% queue (\\$500,781.37 for Stage 3 and "
        "\\$501,510.90 for Stage 4, versus \\$507,989.81 for Stage 1) by prioritizing high-exposure fraud cases. Neither neural "
        "model matches the queue loss of Stage 2, and both exhibit lower AP (0.4772 for Stage 3; 0.4634 for Stage 4). Under "
        "unconstrained Bayes-optimal thresholding, uncalibrated Focal Loss scores produce an alert rate of 13.46% (15,494 alerts). "
        "In this unconstrained regime, manual review and customer friction costs reach \\$534,715.00, raising operational losses "
        "to \\$634,205.97 (Stage 3) and \\$629,411.78 (Stage 4), an increase of +\\$263,682.79 relative to Stage 1."
    ),
    53: (
        "Table III presents confirmatory hypothesis test results across 1,000 paired 24-hour stationary block bootstrap resamples "
        "under Benjamini-Hochberg False Discovery Rate (FDR) control at $\\alpha = 0.05$. One-sided p-values reflect the directional "
        "alternative hypotheses:"
    ),
    54: (
        "1) Relational Feature Augmentation ($H_1$ Supported): Adding 19 time-causal backward relational features to the tabular GBDT "
        "yields an improvement of $\\Delta_{2-1} = +0.0438$ AP (95% bootstrap CI: $[+0.0377, +0.0504]$, $p = 0.0010$, $p_{\\text{adj}} = 0.0080$). "
        "On cards with prior history ($R_1$), the gain increases to $+0.0587$ AP ($p = 0.0010$), reflecting the utility of historical entity baselines."
    ),
    55: (
        "2) Learned Structure vs. Engineered Features ($H_2$ Not Supported): Replacing explicit relational features with the continuous "
        "two-stage Res-GNN results in $\\Delta_{3-2} = -0.0480$ AP (95% CI: $[-0.0549, -0.0415]$, directional $p > 0.999$, "
        "$p_{\\text{adj}} > 0.999$). The directional null cannot be rejected."
    ),
    56: (
        "3) Functional Gradient Coupling ($H_3$ Not Supported): Alternating coordinate updates produce $\\Delta_{4-3} = -0.0138$ AP "
        "(95% bootstrap CI: $[-0.0191, -0.0087]$, directional $p > 0.999$, $p_{\\text{adj}} > 0.999$). The alternating updates do not "
        "improve upon the frozen two-stage pipeline."
    ),
    57: (
        "4) Macro System Deployment ($H_4$ Not Supported): The alternating BGNN achieves lower Average Precision than the standalone "
        "tabular LightGBM baseline ($\\Delta_{4-1} = -0.0181$ AP, 95% bootstrap CI: $[-0.0257, -0.0117]$, directional $p > 0.999$, "
        "$p_{\\text{adj}} > 0.999$). The joint architecture does not surpass the primary tabular baseline."
    ),
    66: (
        "To account for the macro null result ($H_4$ rejection), forensic analysis identifies four breakdown mechanisms:"
    ),
    68: (
        "Gradient Boosted Decision Trees isolate fraud anomalies through discrete, non-linear split boundaries on ratio and velocity "
        "features. Feature gain importance analysis (Table IV) indicates that the highest split gains come from payment verification "
        "metrics and historical entity counters: V258 (gain 30,362.39), C1 (gain 23,408.28), P_emaildomain_seen_before (gain 22,058.14), "
        "and card1_seen_before (gain 20,404.75). Exactly 10 relational features rank in the top 20, accounting for 47.86% of top-20 "
        "gain and 23.51% of total model split gain, with all 19 relational features comprising 29.79% of total gain. A decision tree "
        "isolates an anomalous deviation with a single split on an amount ratio or velocity counter. In contrast, standard message-passing "
        "layers compute linear additive combinations, $\\mathbf{h}_v = \\sum_{u \\in \\mathcal{N}(v)} \\mathbf{W}\\mathbf{h}_u$. "
        "Because financial fraud networks exhibit heterophily {REF:ZHU_HOMOPHILY}, where illicit transactions appear alongside normal activity "
        "at popular merchants, spatial averaging acts as a low-pass filter that dampens localized transaction anomalies."
    ),
    71: (
        "Streaming financial networks experience continuous entity turnover. In the sealed Block 5 holdout ($N = 118{,}108$), "
        "35,197 transactions (30.6%) involve payment cards not observed during training (Fig. 3). For these cold-start entities, "
        "historical neighborhoods are empty ($\\mathcal{N}(v) = \\emptyset$). GBDTs handle unobserved categorical levels through "
        "learned default routing and global distribution splits. In contrast, continuous GNN layers receive zero-vector inputs "
        "for unobserved neighbors, providing little discriminative signal to the classification head {REF:AKLI}. In addition, "
        "continuous feature distributions show temporal covariate drift across the 182-day period, reaching a median absolute "
        "shift of $9.564\\sigma$ in Block 5. While monotonic tree splits are invariant to uniform coordinate scaling, large "
        "distribution shifts can cause neural activation layers to saturate."
    ),
    73: (
        "In canonical BGNN {REF:IVANOV}, outer trees fit continuous displacement targets $\\mathbf{R}_i^m = -\\nabla_{\\mathbf{z}_i}\\mathcal{L}$. "
        "For tabular transaction data, displacing $\\mathbf{z}_i$ in continuous latent space alters the coordinate manifold between "
        "iterations. Decision trees in native feature space construct orthogonal boundaries aligned with physical attributes. "
        "Displacing latent coordinates across boosting rounds shifts these reference points, requiring subsequent trees to adapt "
        "to coordinate changes rather than isolating residual risk. This dynamic explains why removing coordinate displacements "
        "(Stage 4n) yields higher Average Precision than full alternating updates (Stage 4)."
    ),
    75: (
        "Although Stage 4 achieves a raw AP of 0.4634, its raw predicted probabilities exhibit substantial calibration error (Fig. 4). "
        "The Focal Loss parameterization ($\\alpha = 0.25, \\gamma = 2.0$) focuses training on hard negative examples, shifting "
        "predicted scores upward. As a result, Stage 4 shows an Expected Calibration Error (ECE) of 0.0892 and a mean predicted "
        "probability of 12.69%, approximately 3.7 times the true holdout prevalence of 3.499%. When evaluated under cost-based "
        "thresholding without a fixed queue cap, these uncalibrated scores produce a 13.46% alert rate, overwhelming review queues "
        "and driving realized financial losses to \\$629,411.78 (Table II). While post-hoc isotonic calibration reduces ECE to 0.0077 "
        "and calibrated loss to \\$373,410.83, monotonic rescaling cannot improve rank-order Average Precision."
    ),
    77: (
        "These empirical results contrast with reports in the graph learning literature where GNNs demonstrate large performance "
        "margins over tree baselines {REF:DOU}, {REF:WANG_GAT}, {REF:MAO}, {REF:TANG}. Three methodological factors account "
        "for this discrepancy:"
    ),
    78: (
        "1) Transductive Look-Ahead Leakage: Many benchmark studies construct static, bidirectional transaction graphs across "
        "full datasets. Transactions early in the sequence aggregate embeddings from subsequent transactions on the same "
        "cardholder account. When evaluated on dynamic graphs ($G_{<t}$) restricted to historical edges, this look-ahead advantage "
        "is eliminated {REF:VAN_BELLE}, {REF:AKLI}."
    ),
    79: (
        "2) Random Partitioning Bias: Cross-validation on time-stamped transactions places events from the same account into both "
        "training and evaluation folds. This configuration allows models to interpolate between known events rather than predict "
        "future risk. Under chronological Out-of-Time evaluation with 30.6% cold-start entity turnover, neural message passing "
        "cannot rely on seen historical neighborhoods."
    ),
    80: (
        "3) Baseline Feature Completeness: Published evaluations often compare GNNs against basic decision tree configurations "
        "without historical relational features. When GBDTs incorporate backward-looking relational counters (Stage 2), they "
        "yield the lowest financial loss ($345,402.31) and highest calibrated AP (0.5088), outperforming the neural graph "
        "configurations while maintaining lower inference latency."
    ),
    83: (
        "This study investigated joint graph-tabular learning for credit card fraud detection under strict streaming constraints. "
        "Explicit relational feature engineering in Stage 2 achieved the strongest overall performance ($H_1$ confirmed, "
        "$\\Delta_{2-1} = +0.0438$, calibrated AP 0.5088, saving \\$12,682.25 under a fixed 1.0% operational queue constraint, "
        "95% CI: [\\$3,678.70, \\$21,541.70], $p = 0.0010$). However, the joint BGNN architecture did not outperform the standalone "
        "tabular LightGBM baseline ($H_4$ directional null not rejected, $\\Delta_{4-1} = -0.0181$), and alternating functional "
        "updates did not improve upon the frozen Res-GNN ($H_3$ directional null not rejected, $\\Delta_{4-3} = -0.0138$). "
        "Under production streaming conditions, spatial neural convolutions did not match the partitioning effectiveness of "
        "tree ensembles on tabular features."
    ),
    85: (
        "1) Mandate Time-Causal Graph Construction ($G_{<t}$): Benchmarks for transaction fraud should prohibit static, "
        "undirected graphs across splits. Graph construction must restrict edge induction to historical events strictly "
        "preceding the query timestamp {REF:VAN_BELLE}, {REF:AKLI}."
    ),
    86: (
        "2) Require Chronological Out-of-Time Holdouts: Financial risk models should be evaluated on forward chronological "
        "holdouts rather than random cross-validation, preserving realistic entity turnover and distribution shift."
    ),
    87: (
        "3) Standardize Operational Loss Metrics: Evaluations should supplement ROC-AUC with Average Precision, Recall at fixed "
        "operational false-positive rates (0.1% and 1.0% FPR), and dollar loss under realistic review capacity limits."
    ),
    89: (
        "1) Prioritize Relational Feature Engineering in GBDTs: Production risk teams achieve stronger detection performance "
        "and lower operational complexity by computing rolling relational counts and ratio features for GBDT models, rather "
        "than deploying deep GNN pipelines."
    ),
    90: (
        "2) Implement Cold-Start Fallback Routing: When hybrid models are deployed, scoring systems should route isolated or "
        "novel entities directly to tabular trees ($\\beta_i \\to 0$), bypassing unanchored neural layers."
    ),
    91: (
        "3) Apply Isotonic Probability Calibration: Models optimized with asymmetric objectives like Focal Loss require "
        "post-hoc isotonic calibration before score-based queue triage, preventing queue overflows caused by probability inflation."
    ),
    93: (
        "Future research vectors include: (1) integrating continuous-time memory networks (Temporal Graph Networks {REF:ROSSI}) "
        "with low-latency caching; (2) formulating differentiable ratio and product-norm layers in GNNs to model scale-invariant "
        "deviations without smoothing; and (3) exploring logit-space functional boosting to avoid continuous coordinate displacements."
    ),
}

# Update DOCUMENT_FLOW in memory
for idx, new_text in HUMANIZED_P.items():
    bip.DOCUMENT_FLOW[idx] = ("p", new_text)

# Also update the abstract inside build_ieee_body_section
# We can create a patched build_ieee_body_section
orig_build_body = bip.build_ieee_body_section

def humanized_build_ieee_body_section(doc):
    sec2 = doc.add_section(WD_SECTION.CONTINUOUS)
    sec2.page_width = Mm(210)
    sec2.page_height = Mm(297)
    sec2.top_margin = Mm(19)
    sec2.bottom_margin = Mm(25.4)
    sec2.left_margin = Mm(16)
    sec2.right_margin = Mm(16)
    
    sectPr = sec2._sectPr
    cols = sectPr.find(qn('w:cols'))
    if cols is None:
        cols = OxmlElement('w:cols')
        sectPr.append(cols)
    cols.set(qn('w:num'), '2')
    cols.set(qn('w:space'), '360')

    sec2.header.is_linked_to_previous = False
    sec2.footer.is_linked_to_previous = False
    for p in sec2.header.paragraphs:
        p.text = ""
    for p in sec2.footer.paragraphs:
        p.text = ""

    # Abstract
    p_abs = doc.add_paragraph()
    p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_abs.paragraph_format.line_spacing = 1.0
    p_abs.paragraph_format.space_before = Pt(4)
    p_abs.paragraph_format.space_after = Pt(4)
    
    r_abs_lead = p_abs.add_run("Abstract—")
    r_abs_lead.font.name = "Times New Roman"
    r_abs_lead.font.size = Pt(9)
    r_abs_lead.bold = True
    r_abs_lead.italic = True
    
    abstract_text = (
        "Credit card fraud detection in payment clearing networks requires reliable risk "
        "discrimination under sub-100-millisecond latency budgets and delayed chargeback feedback. While Gradient "
        "Boosted Decision Trees remain the primary operational choice for tabular transaction data, Graph Neural Networks "
        "are widely studied for modeling relational connections across cards and devices. However, many published "
        "graph benchmarks use static, transductive partitions that introduce temporal look-ahead "
        "leakage and ignore the open-world cold-start conditions of production streams. This paper evaluates an adapted Boost then "
        "Convolve architecture that combines continuous tabular embeddings, relation-aware message passing on a strictly "
        "time-causal dynamic graph, monotonic decision blending, and alternating functional-gradient optimization. "
        "Across the IEEE-CIS benchmark dataset (590,540 transactions over 182 days) and a sealed Out-of-Time holdout "
        "(115,111 evaluated transactions, Block 5), results show a clear separation between tree-based and neural methods. "
        "Under a fixed one percent operational review queue, handcrafted relational feature engineering in Stage 2 achieves the lowest "
        "financial loss, saving USD 12,682.25 compared with the tabular baseline "
        "(95 percent CI: [USD 3,678.70, USD 21,541.70], p = 0.0010) with an Average Precision of 0.5252. In contrast, neural "
        "message passing degrades ranking performance (delta = -0.0181), and uncalibrated Focal Loss predictions increase "
        "unconstrained operational losses by USD 263,682.79 through excessive alerts. Analysis identifies three primary breakdown mechanisms: "
        "spatial smoothing of ratio anomalies, an inductive cold-start bottleneck from unseen entities, and probability miscalibration."
    )
    r_abs_body = p_abs.add_run(abstract_text)
    r_abs_body.font.name = "Times New Roman"
    r_abs_body.font.size = Pt(9)
    r_abs_body.bold = True

    # Keywords
    p_key = doc.add_paragraph()
    p_key.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_key.paragraph_format.line_spacing = 1.0
    p_key.paragraph_format.space_before = Pt(2)
    p_key.paragraph_format.space_after = Pt(10)
    
    r_key_lead = p_key.add_run("Keywords—")
    r_key_lead.font.name = "Times New Roman"
    r_key_lead.font.size = Pt(9)
    r_key_lead.bold = True
    r_key_lead.italic = True
    
    r_key_body = p_key.add_run("credit card fraud detection, boosted graph neural networks, tabular representation learning, time-causal graph, out-of-time evaluation, financial risk minimization.")
    r_key_body.font.name = "Times New Roman"
    r_key_body.font.size = Pt(9)
    r_key_body.bold = True
    r_key_body.italic = True

    # Build rest using bip helper
    cite_map = {}
    ordered_refs = []
    
    def resolve_ref(match):
        key = match.group(1)
        if key not in bip.REFERENCE_DB:
            raise KeyError(f"Reference key '{key}' not found in REFERENCE_DB catalog!")
        if key not in cite_map:
            num = len(cite_map) + 1
            cite_map[key] = num
            ordered_refs.append((num, key, bip.REFERENCE_DB[key]))
        return f"[{cite_map[key]}]"

    # Render Document Elements
    for elem_type, payload in bip.DOCUMENT_FLOW:
        if elem_type == "h1":
            resolved_h1 = re.sub(r'\{REF:([A-Za-z0-9_]+)\}', resolve_ref, payload)
            bip.add_heading_1(doc, resolved_h1)
        elif elem_type == "h2":
            resolved_h2 = re.sub(r'\{REF:([A-Za-z0-9_]+)\}', resolve_ref, payload)
            bip.add_heading_2(doc, resolved_h2)
        elif elem_type == "h3":
            resolved_h3 = re.sub(r'\{REF:([A-Za-z0-9_]+)\}', resolve_ref, payload)
            bip.add_heading_3(doc, resolved_h3)
        elif elem_type == "p":
            resolved_p = re.sub(r'\{REF:([A-Za-z0-9_]+)\}', resolve_ref, payload)
            bip.add_body_p(doc, resolved_p)
        elif elem_type == "eq":
            latex_code, eq_num = payload
            bip.add_ieee_display_eq(doc, latex_code, eq_num)
        elif elem_type == "figure":
            img_path, caption, width_mm = payload
            resolved_caption = re.sub(r'\{REF:([A-Za-z0-9_]+)\}', resolve_ref, caption)
            bip.add_ieee_figure(doc, img_path, resolved_caption, width_mm)
        elif elem_type == "table":
            cap_str, headers, rows, col_widths = payload
            resolved_cap = re.sub(r'\{REF:([A-Za-z0-9_]+)\}', resolve_ref, cap_str)
            resolved_headers = [re.sub(r'\{REF:([A-Za-z0-9_]+)\}', resolve_ref, h) for h in headers]
            resolved_rows = [
                [re.sub(r'\{REF:([A-Za-z0-9_]+)\}', resolve_ref, str(c)) for c in row]
                for row in rows
            ]
            bip.add_ieee_table(doc, resolved_cap, resolved_headers, resolved_rows, col_widths)

        # References
    bip.add_heading_1(doc, "REFERENCES")
    for num, key, text in ordered_refs:
        p_ref = doc.add_paragraph()
        p_ref.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_ref.paragraph_format.line_spacing = 1.0
        p_ref.paragraph_format.space_before = Pt(0)
        p_ref.paragraph_format.space_after = Pt(2)
        p_ref.paragraph_format.left_indent = Mm(4.5)
        p_ref.paragraph_format.first_line_indent = Mm(-4.5)
        
        r_num = p_ref.add_run(f"[{num}]\t")
        r_num.font.name = "Times New Roman"
        r_num.font.size = Pt(8)
        
        r_txt = p_ref.add_run(text)
        r_txt.font.name = "Times New Roman"
        r_txt.font.size = Pt(8)

# Run build
doc = docx.Document()
doc.core_properties.author = "Ieman Ihsan Bin Alwi"
doc.core_properties.title = "A Time-Causal Joint Graph-Tabular Learning Framework for Credit Card Fraud Detection Using an Adapted Boost then Convolve Architecture"
doc.core_properties.subject = "IEEE Technical Paper, Universiti Teknologi PETRONAS"

bip.build_ieee_header(doc)
humanized_build_ieee_body_section(doc)

out_file = "IEEE_Technical_Paper_Ieman_Ihsan_25005466_humanized.docx"
doc.save(out_file)
print(f"Generated {out_file} successfully! Size: {os.path.getsize(out_file)/(1024*1024):.2f} MB")
