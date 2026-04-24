# Interpretable Machine Learning Framework for Predicting Synergistic Drug Combinations Using Multi-Omics Pharmacogenomic Data

![Python](https://img.shields.io/badge/Python-3.8-blue)
![R](https://img.shields.io/badge/R-4.x-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1.4-orange)
![License](https://img.shields.io/badge/License-Academic-red)

**Author:** Sushma Ankathi | MS Bioinformatics | Northeastern University | Spring 2026
**Faculty:** Dr.Ayansola Oyeronke | **Advisor:** Dr. Nabil Atallah
**Course:** BINF7700 Capstone Project

---

## Project Overview

Cancer drug resistance remains one of the leading causes of treatment failure worldwide. 
Combination therapies that target multiple pathways simultaneously offer a promising 
solution , yet testing all possible drug pairs experimentally is prohibitively expensive, 
requiring decades of work and billions of dollars.

This project develops an **interpretable machine learning framework** that predicts 
synergistic drug combinations by integrating chemical, genomic, transcriptomic and 
drug sensitivity data from five major public pharmacogenomic databases across 
51 drugs and 88 cancer cell lines.

---

## Key Results

| Metric | Value |
|---|---|
| XGBoost AUROC | **0.9954** |
| XGBoost PR-AUC | **0.8292** |
| 5-Fold CV AUROC | **0.9931 ± 0.0019** |
| DeepSynergy Benchmark | ~0.75 |
| MatchMaker Benchmark | ~0.78 |
| Total Features | 2,703 |
| Drug-Cell Pairs | 75,571 |
| Significant KEGG Pathways | 98 |
| Soft Tissue Synergy Enrichment | 7.57× above average |
| Top Synergistic Pair | DASATINIB + TEMOZOLOMIDE |

---

## Research Question

Can an interpretable ML framework accurately predict synergistic drug combinations 
by integrating chemical, genomic, transcriptomic and drug sensitivity data from 
public pharmacogenomic databases?

---

## Databases Used

| Database | What it contributes | Records |
|---|---|---|
| DrugCombDB | Drug pair synergy experiment results | 383,958 pairs |
| GDSC2 | Drug sensitivity, mutations, CNV, tissue | 88 cell lines |
| ChEMBL | Chemical structures, Morgan fingerprints | 3,229 drugs |
| LINCS L1000 | Gene expression signatures | 1,316 signatures |
| MSigDB | Hallmark gene sets for enrichment | 50 gene sets |


## Pipeline
DrugCombDB → 383,958 pairs, Loewe > 10 label
GDSC2      → LN_IC50, mutations, CNV, tissue
ChEMBL     → ECFP4 Morgan fingerprints (1024 bits)
LINCS L1000→ Level-5 gene signatures
MSigDB     → Hallmark gene sets (v2026.1)
↓
ID Harmonisation
51 drugs × 88 cell lines
↓
Feature Matrix
2,703 features × 75,571 rows
↓
ML Models
XGBoost | LightGBM | Random Forest
↓
SHAP + LIME Interpretability
↓
Pathway Enrichment
KEGG | GO | Reactome | MSigDB

---

## Repository Structure
BINF7700_Capstone_Sushma_Ankathi/
│
├── step1_drugcomb/
│   ├── step1_drugcomb_preprocessing.py
│   └── outputs/
│       ├── drugcomb_cleaned.csv
│       ├── drugcomb_harmonised.csv
│       └── drugcomb_.png
│
├── step2_gdsc/
│   ├── step2_GDSC_preprocessing.py
│   └── outputs/
│       ├── gdsc_cell_lines_cleaned.csv
│       ├── gdsc_mutations.csv
│       └── gdsc_.csv / png
│
├── step3_chembl/
│   ├── step3_chembl_fingerprints.py
│   ├── Chembl Download.py
│   ├── Step3_fix cosmic Id.py
│   └── outputs/
│       ├── chembl_morgan_fingerprints.csv
│       └── chembl_.csv / png
│
├── step3b_lincs/
│   ├── Step3_Lincs_download.py
│   └── outputs/
│       ├── lincs_expression_features.csv
│       └── lincs_signatures_final.csv
│
├── step4_harmonisation/
│   ├── Step4_id harmonisation.py
│   └── outputs/
│       ├── master_drug_map.csv
│       ├── master_cell_line_map.csv
│       └── id_harmonisation_plots.png
│
├── step5_feature_matrix/
│   ├── Step5_Feature Matrix.py
│   └── outputs/
│       ├── feature_names.csv
│       └── feature_matrix_overview.png
│
├── step6_models/
│   ├── Step6_Baseline_Models_v2.py
│   ├── Step6_Synergy_Heatmap.py
│   └── outputs/
│       ├── baseline_roc_curves_v2.png
│       ├── baseline_pr_curves_v2.png
│       ├── baseline_metrics_v2.csv
│       └── synergy_heatmap.png
│
├── step7_shap/
│   ├── Step7_Shap_Lime.py
│   └── outputs/
│       ├── shap_beeswarm.png
│       ├── tissue_stratification_plot.png
│       └── shap_.csv / png
│
├── step8_enrichment/
│   ├── Step8 enrichment network.py
│   ├── step8b_clusterProfileR.R
│   ├── Step8c_ msigdb gsea.R
│   └── outputs/
│       ├── clusterProfiler_barplot_kegg.png
│       ├── enrichment_volcano_kegg.png
│       └── msigdb_*.csv / png
│
├── data/
│   └── README.md ← Download instructions
│
├── .gitignore
└── README.md


---

## Feature Engineering

| Feature Group | Source | Count | Biological Meaning |
|---|---|---|---|
| Morgan Fingerprints Drug1 | ChEMBL | 1,024 | Chemical structure of Drug 1 |
| Morgan Fingerprints Drug2 | ChEMBL | 1,024 | Chemical structure of Drug 2 |
| Gene Expression | LINCS L1000 | 408 | Transcriptomic response to drug |
| Mutations | GDSC2 | 100 | Cancer cell genetic profile |
| CNV | GDSC2 | 100 | Copy number variations |
| Tissue Type | GDSC2 | 19 | Cancer tissue of origin |
| Drug Sensitivity | GDSC2 | 16 | LN_IC50, AUC, drug properties |
| Synergy Scores | DrugCombDB | 3 | ZIP, Bliss, HSA agreement |
| Tanimoto Similarity | Derived | 1 | Chemical dissimilarity between drugs |
| **Total** | | **2,703** | |

---

## Model Performance

| Model | AUROC | PR-AUC | F1 | Threshold |
|---|---|---|---|---|
| **XGBoost** | **0.9954** | **0.8292** | **0.7567** | 0.505 |
| LightGBM | 0.9945 | 0.8103 | 0.7314 | 0.525 |
| Random Forest | 0.8998 | 0.1431 | 0.2296 | 0.240 |

**XGBoost 5-Fold Cross Validation:**
- AUROC: 0.9931 ± 0.0019
- PR-AUC: 0.8236 ± 0.0215

**Benchmark Comparison:**
- DeepSynergy (Preuer et al., 2018): PR-AUC ~0.75
- MatchMaker (Kuru et al., 2022): PR-AUC ~0.78
- **This framework: PR-AUC = 0.83** 

---

## Key Biological Findings

### SHAP Interpretability
Top predictive features identified by SHAP:

| Feature | SHAP Value | Biological Meaning |
|---|---|---|
| HSA Score | 5.784 | Single agent potency predicts combination benefit |
| Bliss Score | 0.845 | Independent drug effect agreement |
| ZIP Score | 0.722 | Dose-response pattern consistency |
| Tanimoto Similarity | 0.092 | Dissimilar drugs hit different targets |
| LN_IC50 | 0.079 | Drug potency predicts synergy |
| TP53 Mutation | 0.026 | Key cancer biomarker |

**Key Insight:** Structurally dissimilar drugs (low Tanimoto) are MORE 
synergistic — they attack cancer from completely different pathways simultaneously.

---

### Tissue Stratification

| Cancer Type | Synergy Rate | Enrichment |
|---|---|---|
| Soft tissue | 47.4% | **7.57× above average** |
| Nervous system | 9.5% | 1.52× above average |
| All others | <6.3% | Below average |

**Clinical Implication:** Soft tissue sarcoma patients may benefit most 
from combination drug therapy — directly actionable for clinical trial design.

---

### Pathway Enrichment

| Database | Significant Pathways | Top Findings |
|---|---|---|
| KEGG | 98 pathways | PI3K-AKT-mTOR, EGFR resistance |
| GO Biological Process | 747 terms | Chromatin remodelling, cell signalling |
| Reactome | 226 pathways | Growth factor signalling |
| MSigDB Hallmarks | 3 nominal | UV_Response_DN (NES=1.62) |

**Top SHAP Genes:** TP53 | KRAS | PTEN | BRAF | PIK3CA | NF1 | ARID1A | SMARCA4 | EP300 | ACVR2A

All 10 genes are well-known cancer driver genes — independently discovered 
by the model without being told — validating genuine biological learning.

---

## Installation

### Python Requirements
```bash
pip install -r requirements.txt
```

### R Requirements
```r
install.packages(c("clusterProfiler", "ReactomePA", "enrichplot", "ggplot2"))
BiocManager::install(c("org.Hs.eg.db", "DOSE"))
```

### Requirements
pandas==1.3.5
numpy==1.21.0
scikit-learn==1.0.2
xgboost==2.1.4
lightgbm==3.3.2
shap==0.41.0
matplotlib==3.5.1
seaborn==0.11.2
rdkit==2022.09.5
imbalanced-learn==0.9.0
scipy==1.7.3
chembl-webresource-client

---

## How to Run

```bash
# Step 1 — DrugComb preprocessing
python step1_drugcomb/step1_drugcomb_preprocessing.py

# Step 2 — GDSC preprocessing
python step2_gdsc/step2_GDSC_preprocessing.py

# Step 3 — ChEMBL fingerprints
python step3_chembl/step3_chembl_fingerprints.py

# Step 3b — LINCS download (API key required)
python step3b_lincs/Step3_Lincs_download.py

# Step 4 — ID harmonisation
python "step4_harmonisation/Step4_id harmonisation.py"

# Step 5 — Feature matrix
python "step5_feature_matrix/Step5_Feature Matrix.py"

# Step 6 — ML models
python step6_models/Step6_Baseline_Models_v2.py

# Step 7 — SHAP + LIME + Tissue
python step7_shap/Step7_Shap_Lime.py

# Step 8 — Enrichment (Python)
python "step8_enrichment/Step8 enrichment network.py"

# Step 8b — clusterProfileR (R)
Rscript step8_enrichment/step8b_clusterProfileR.R

# Step 8c — MSigDB GSEA (R)
Rscript "step8_enrichment/Step8c_ msigdb gsea.R"
```

---

## Data Sources

Raw data not included due to licensing and size constraints.
See `data/README.md` for complete download instructions.

| Database | URL |
|---|---|
| DrugCombDB | https://drugcomb.fimm.fi/ |
| GDSC2 | https://www.cancerrxgene.org/downloads |
| ChEMBL | https://www.ebi.ac.uk/chembl/ (API) |
| LINCS L1000 | https://api.clue.io/ (API key required) |
| MSigDB | https://www.gsea-msigdb.org/gsea/msigdb/ |

---

## Limitations

- Only 51 drugs covered — cross-database intersection constraint
- 88 out of 969 GDSC cell lines overlap with DrugComb (9% coverage)
- 8 post-2017 drugs missing from LINCS L1000
- GSEA limited by small gene input size (n=10 genes)
- No external validation dataset — hold-out test set used

---

## Future Directions

- Bayesian hyperparameter optimisation for XGBoost and LightGBM
- Graph Neural Network models to capture drug-drug interaction structure
- External validation on NCI-ALMANAC or OncologyScreen datasets
- Expansion from 51 to 500+ drug combinations
- Web interface for oncologists to query drug combinations
- Wet lab validation of soft tissue sarcoma synergy finding

---

## References

1. Preuer, K. et al. (2018). DeepSynergy. *Bioinformatics*, 34(9), 1538–1546.
2. Kuru, H.I. et al. (2022). MatchMaker. *IEEE/ACM TCBB*, 19(4), 2334–2344.
3. Lundberg, S.M. & Lee, S.I. (2017). SHAP. *NeurIPS*, 30.
4. Wu, T. et al. (2021). clusterProfiler 4.0. *Innovation*, 2(3), 100141.
5. Subramanian, A. et al. (2005). GSEA. *PNAS*, 102(43), 15545–15550.
6. Liberzon, A. et al. (2015). MSigDB Hallmarks. *Cell Systems*, 1(6), 417–425.

---

## Contact

**Sushma Ankathi**
MS Bioinformatics | Northeastern University
ankathi.s@northeastern.edu
---

## Pipeline
