# Data Sources

Raw data is not included in this repository due to licensing 
restrictions and file size constraints.
Please download or access each database using the instructions below.

---

## 1. DrugCombDB
**URL:** https://drugcomb.fimm.fi/

**Steps:**
1. Go to https://drugcomb.fimm.fi/
2. Click Download
3. Download drug_combinations.csv
4. Place in: data/raw/drugcomb_raw.csv

**File used:** drug_combinations.csv
**Size:** ~500MB
**License:** Academic use only

---

## 2. GDSC2 (Genomics of Drug Sensitivity in Cancer)
**URL:** https://www.cancerrxgene.org/downloads

**Steps:**
1. Go to https://www.cancerrxgene.org/downloads
2. Download the following four files:

| File | Section on website |
|---|---|
| GDSC2_fitted_dose_response.xlsx | Dose Response Data |
| Cell_Lines_Details.xlsx | Cell Line Data |
| PANCANCER_IC50.csv | Drug Sensitivity Data |
| PANCANCER_Genetic_feature.csv | Genomic Data |

3. Place all four files in: data/raw/

**License:** Academic use only

---

## 3. ChEMBL
**URL:** https://www.ebi.ac.uk/chembl/

**Access via Python API — no direct file download needed**

**Steps:**
1. Install ChEMBL client:
pip install chembl-webresource-client

2. Query is handled automatically in:
step3_chembl/step3_chembl_fingerprints.py

**What is downloaded:**
- Approved small molecule drugs
- Valid SMILES strings
- Molecular properties
- Drug target relationships

**License:** Creative Commons Attribution-ShareAlike 3.0

---

## 4. LINCS L1000
**URL:** https://api.clue.io/

**Access via clue.io API — no direct file download needed**

**Steps:**
1. Register free at https://clue.io/ to get your API key
2. Open step3b_lincs/step3b_lincs_l1000.py
3. Replace API_KEY with your own key:
   API_KEY = "your_api_key_here"
4. Run the script — data downloads automatically

**What is downloaded:**
- Level 5 gold standard consensus signatures
- pert_type = trt_cp
- 43 out of 51 drugs matched
- 408 landmark gene features
- 1,316 signatures

**Note:** 8 post-2017 drugs are not available in LINCS
as the dataset was created in 2017.

**License:** NIH public data — free for academic use

---

## 5. MSigDB Hallmark Gene Sets
**URL:** https://www.gsea-msigdb.org/gsea/msigdb/

**Steps:**
1. Go to https://www.gsea-msigdb.org/gsea/msigdb/
2. Click Collections
3. Select H — Hallmark gene sets
4. Select Homo sapiens
5. Select Entrez Gene IDs format
6. Download: h.all.v2026.1.Hs.entrez.gmt
7. Place in: step8_enrichment/

**File used:** h.all.v2026.1.Hs.entrez.gmt
**Version:** v2026.1
**Gene ID format:** Entrez IDs
**Number of gene sets:** 50 Hallmark sets
**Total gene entries:** 7,322

**License:** MSigDB license — free for academic use
Register free at msigdb.org to download

---

## File Placement Summary

After downloading place files as follows:

data/
└── raw/
    ├── drug_combinations.csv          ← DrugCombDB
    ├── GDSC2_fitted_dose_response.xlsx ← GDSC2
    ├── Cell_Lines_Details.xlsx         ← GDSC2
    ├── PANCANCER_IC50.csv              ← GDSC2
    └── PANCANCER_Genetic_feature.csv   ← GDSC2

step3b_lincs/
└── (downloaded automatically via API)

step8_enrichment/
└── h.all.v2026.1.Hs.entrez.gmt       ← MSigDB

---

## Notes

- ChEMBL and LINCS data are downloaded programmatically
  via API calls inside the scripts
- No manual file placement needed for ChEMBL and LINCS
- All scripts handle missing files with clear error messages
- Total raw data size approximately 2GB+
- GitHub file size limit is 100MB — raw data excluded

---

## Citation

If you use this data please cite the original sources:

1. DrugCombDB: Tang et al. (2020). Nucleic Acids Research.
2. GDSC: Yang et al. (2012). Nucleic Acids Research.
3. ChEMBL: Mendez et al. (2019). Nucleic Acids Research.
4. LINCS L1000: Subramanian et al. (2017). Cell.
5. MSigDB: Liberzon et al. (2015). Cell Systems.
