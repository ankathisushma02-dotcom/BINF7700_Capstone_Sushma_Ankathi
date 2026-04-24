import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from difflib import get_close_matches
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
PROJECT_DIR = r"C:\Users\sush\PYCHARM\Capstone project"

FILES = {
    "drugcomb"        : os.path.join(PROJECT_DIR, "drugcomb_cleaned.csv"),
    "gdsc_ic50"       : os.path.join(PROJECT_DIR, "gdsc_ic50_cleaned.csv"),
    "gdsc_dose_resp"  : os.path.join(PROJECT_DIR, "gdsc_dose_response_cleaned.csv"),
    "gdsc_cells"      : os.path.join(PROJECT_DIR, "gdsc_cell_lines_cleaned.csv"),
    "gdsc_mutations"  : os.path.join(PROJECT_DIR, "gdsc_mutations.csv"),
    "gdsc_cnv"        : os.path.join(PROJECT_DIR, "gdsc_cnv.csv"),
    "gdsc_tissue"     : os.path.join(PROJECT_DIR, "gdsc_tissue_map.csv"),
    "chembl_fp"       : os.path.join(PROJECT_DIR, "chembl_morgan_fingerprints.csv"),
    "chembl_info"     : os.path.join(PROJECT_DIR, "chembl_drug_info_clean.csv"),
    "chembl_targets"  : os.path.join(PROJECT_DIR, "chembl_targets_clean.csv"),
    "overlap_cells"   : os.path.join(PROJECT_DIR, "gdsc_drugcomb_overlap_cells.csv"),
    "fuzzy_map"       : os.path.join(PROJECT_DIR, "gdsc_drugcomb_fuzzy_name_map.csv"),
    "overlap_drugs"   : os.path.join(PROJECT_DIR, "chembl_drugcomb_gdsc_overlap_drugs.csv"),
}
def section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def std_name(s):
    return str(s).upper().strip()

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load all cleaned files
# ══════════════════════════════════════════════════════════════════════════
section("STEP 1: Loading All Cleaned Files")

drugcomb       = pd.read_csv(FILES["drugcomb"])
gdsc_ic50      = pd.read_csv(FILES["gdsc_ic50"])
gdsc_dose      = pd.read_csv(FILES["gdsc_dose_resp"])
gdsc_cells     = pd.read_csv(FILES["gdsc_cells"])
gdsc_mutations = pd.read_csv(FILES["gdsc_mutations"])
gdsc_cnv       = pd.read_csv(FILES["gdsc_cnv"])
gdsc_tissue    = pd.read_csv(FILES["gdsc_tissue"])
chembl_fp      = pd.read_csv(FILES["chembl_fp"])
chembl_info    = pd.read_csv(FILES["chembl_info"])
chembl_targets = pd.read_csv(FILES["chembl_targets"])
overlap_cells  = pd.read_csv(FILES["overlap_cells"])
fuzzy_map      = pd.read_csv(FILES["fuzzy_map"])
overlap_drugs  = pd.read_csv(FILES["overlap_drugs"])

print(f"DrugComb          : {drugcomb.shape[0]:,} rows")
print(f"GDSC IC50         : {gdsc_ic50.shape[0]:,} rows")
print(f"GDSC Dose Resp    : {gdsc_dose.shape[0]:,} rows")
print(f"GDSC Cell Lines   : {gdsc_cells.shape[0]:,} rows")
print(f"GDSC Mutations    : {gdsc_mutations.shape[0]:,} rows")
print(f"GDSC CNV          : {gdsc_cnv.shape[0]:,} rows")
print(f"GDSC Tissue Map   : {gdsc_tissue.shape[0]:,} rows")
print(f"ChEMBL FP         : {chembl_fp.shape[0]:,} rows")
print(f"ChEMBL Info       : {chembl_info.shape[0]:,} rows")
print(f"ChEMBL Targets    : {chembl_targets.shape[0]:,} rows")
print(f"Overlap Cells     : {overlap_cells.shape[0]:,} rows")
print(f"Fuzzy Map         : {fuzzy_map.shape[0]:,} rows")
print(f"Overlap Drugs     : {overlap_drugs.shape[0]:,} rows")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Standardise all names
# ══════════════════════════════════════════════════════════════════════════
section("STEP 2: Standardising All Names")

# DrugComb
drugcomb.columns = drugcomb.columns.str.upper().str.strip()
for col in ["DRUG1", "DRUG2"]:
    if col in drugcomb.columns:
        drugcomb[col] = drugcomb[col].apply(std_name)
cell_col_dc = next((c for c in drugcomb.columns if "CELL" in c), None)
if cell_col_dc:
    drugcomb[cell_col_dc] = drugcomb[cell_col_dc].apply(std_name)

# GDSC IC50
gdsc_ic50.columns = gdsc_ic50.columns.str.upper().str.strip()
if "DRUG_NAME" in gdsc_ic50.columns:
    gdsc_ic50["DRUG_NAME"] = gdsc_ic50["DRUG_NAME"].apply(std_name)
if "CELL_LINE_NAME" in gdsc_ic50.columns:
    gdsc_ic50["CELL_LINE_NAME"] = gdsc_ic50["CELL_LINE_NAME"].apply(std_name)

# GDSC Dose Response
gdsc_dose.columns = gdsc_dose.columns.str.upper().str.strip()
if "DRUG_NAME" in gdsc_dose.columns:
    gdsc_dose["DRUG_NAME"] = gdsc_dose["DRUG_NAME"].apply(std_name)
if "CELL_LINE_NAME" in gdsc_dose.columns:
    gdsc_dose["CELL_LINE_NAME"] = gdsc_dose["CELL_LINE_NAME"].apply(std_name)

# ChEMBL
chembl_info.columns = chembl_info.columns.str.upper().str.strip()
chembl_fp.columns   = chembl_fp.columns.str.upper().str.strip()
if "DRUG_NAME" in chembl_info.columns:
    chembl_info["DRUG_NAME"] = chembl_info["DRUG_NAME"].apply(std_name)
if "DRUG_NAME" in chembl_fp.columns:
    chembl_fp["DRUG_NAME"] = chembl_fp["DRUG_NAME"].apply(std_name)

# Tissue map
gdsc_tissue.columns = gdsc_tissue.columns.str.upper().str.strip()
if "CELL_LINE_NAME" in gdsc_tissue.columns:
    gdsc_tissue["CELL_LINE_NAME"] = gdsc_tissue["CELL_LINE_NAME"].apply(std_name)

print("All names standardised to UPPERCASE across all files.")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Build Master Cell Line Map
#           COSMIC ID is the stable identifier (from GDSC)
# ══════════════════════════════════════════════════════════════════════════
section("STEP 3: Building Master Cell Line Map")

# Load fuzzy name map for DrugComb -> GDSC name resolution
fuzzy_map.columns = fuzzy_map.columns.str.upper().str.strip()
fuzzy_dc_to_gdsc = dict(zip(
    fuzzy_map["DRUGCOMB_NAME"].apply(std_name),
    fuzzy_map["GDSC_NAME"].apply(std_name)
))
print(f"Fuzzy name map loaded: {len(fuzzy_dc_to_gdsc):,} mappings")

# Build GDSC cell line -> COSMIC ID map
gdsc_cells.columns = gdsc_cells.columns.str.upper().str.strip()
cell_name_col = next((c for c in gdsc_cells.columns
                      if "SAMPLE" in c or ("CELL" in c and "NAME" in c)), None)
cosmic_col    = next((c for c in gdsc_cells.columns if "COSMIC" in c), None)

print(f"\nGDSC cell name column : '{cell_name_col}'")
print(f"GDSC COSMIC column    : '{cosmic_col}'")

if cell_name_col and cosmic_col:
    gdsc_cells[cell_name_col] = gdsc_cells[cell_name_col].apply(std_name)
    gdsc_cosmic_map = gdsc_cells[[cell_name_col, cosmic_col]].dropna()
    gdsc_cosmic_map.columns = ["GDSC_CELL_NAME", "COSMIC_ID"]
    gdsc_cosmic_map["COSMIC_ID"] = pd.to_numeric(
        gdsc_cosmic_map["COSMIC_ID"], errors="coerce")
    gdsc_cosmic_map.dropna(subset=["COSMIC_ID"], inplace=True)
    gdsc_cosmic_map["COSMIC_ID"] = gdsc_cosmic_map["COSMIC_ID"].astype(int)
    print(f"GDSC -> COSMIC map built: {len(gdsc_cosmic_map):,} cell lines")

# Build master cell line map
# For each overlapping cell line, find COSMIC ID
overlap_cells.columns = overlap_cells.columns.str.upper().str.strip()
overlap_list = overlap_cells["CELL_LINE_NAME"].apply(std_name).tolist()

master_cell_rows = []
for cell in overlap_list:
    # Try direct GDSC match
    gdsc_name = cell
    if cell in fuzzy_dc_to_gdsc:
        gdsc_name = fuzzy_dc_to_gdsc[cell]

    # Get COSMIC ID
    cosmic_match = gdsc_cosmic_map[
        gdsc_cosmic_map["GDSC_CELL_NAME"] == gdsc_name]
    cosmic_id = int(cosmic_match["COSMIC_ID"].values[0]) \
                if len(cosmic_match) > 0 else None

    # Get tissue type
    tissue_match = gdsc_tissue[
        gdsc_tissue["CELL_LINE_NAME"] == gdsc_name] \
        if "CELL_LINE_NAME" in gdsc_tissue.columns else pd.DataFrame()
    tissue = tissue_match["TISSUE_TYPE"].values[0] \
             if len(tissue_match) > 0 else "Unknown"

    master_cell_rows.append({
        "DRUGCOMB_CELL_NAME" : cell,
        "GDSC_CELL_NAME"     : gdsc_name,
        "COSMIC_ID"          : cosmic_id,
        "TISSUE_TYPE"        : tissue,
        "FUZZY_MATCHED"      : cell in fuzzy_dc_to_gdsc
    })

master_cell_map = pd.DataFrame(master_cell_rows)
master_cell_map.dropna(subset=["COSMIC_ID"], inplace=True)

print(f"\nMaster cell line map built:")
print(f"  Total overlapping cell lines : {len(overlap_list):,}")
print(f"  Successfully mapped          : {len(master_cell_map):,}")
print(f"  With COSMIC ID               : {master_cell_map['COSMIC_ID'].notna().sum():,}")
print(f"  Fuzzy matched                : {master_cell_map['FUZZY_MATCHED'].sum():,}")
print(f"\nTissue distribution:\n{master_cell_map['TISSUE_TYPE'].value_counts().head(10)}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Build Master Drug Map
#           ChEMBL ID is the stable drug identifier
# ══════════════════════════════════════════════════════════════════════════
section("STEP 4: Building Master Drug Map")

# Load three-way overlap drugs
overlap_drugs.columns = overlap_drugs.columns.str.upper().str.strip()
overlap_drug_list = overlap_drugs["DRUG_NAME"].apply(std_name).tolist()
print(f"Three-way overlap drugs: {len(overlap_drug_list):,}")

# Build ChEMBL drug name -> CHEMBL_ID map
chembl_name_to_id = dict(zip(
    chembl_info["DRUG_NAME"],
    chembl_info["CHEMBL_ID"]
)) if "DRUG_NAME" in chembl_info.columns and "CHEMBL_ID" in chembl_info.columns else {}

# Build ChEMBL synonym map for better matching
chembl_synonym_map = {}
if "SYNONYMS" in chembl_info.columns:
    for _, row in chembl_info.iterrows():
        syns = str(row.get("SYNONYMS", ""))
        if syns and syns != "nan":
            for syn in syns.split("|"):
                chembl_synonym_map[std_name(syn)] = row["DRUG_NAME"]

# Build GDSC drug name -> DRUG_ID map
gdsc_drug_id_map = {}
if "DRUG_NAME" in gdsc_ic50.columns and "DRUG_ID" in gdsc_ic50.columns:
    gdsc_drug_id_map = dict(zip(
        gdsc_ic50["DRUG_NAME"],
        gdsc_ic50["DRUG_ID"]
    ))

# Build master drug map
master_drug_rows = []
gdsc_drug_names = set(gdsc_ic50["DRUG_NAME"].unique()) \
                  if "DRUG_NAME" in gdsc_ic50.columns else set()
dc_drug_names   = set(drugcomb["DRUG1"].unique()) | \
                  set(drugcomb["DRUG2"].unique()) \
                  if "DRUG1" in drugcomb.columns else set()

for drug in overlap_drug_list:
    # Get ChEMBL ID — direct or via synonym
    chembl_id = chembl_name_to_id.get(drug)
    if not chembl_id and drug in chembl_synonym_map:
        chembl_id = chembl_name_to_id.get(chembl_synonym_map[drug])

    # Get GDSC Drug ID
    gdsc_id = gdsc_drug_id_map.get(drug)

    # Find best GDSC name match
    gdsc_name = drug if drug in gdsc_drug_names else None
    if not gdsc_name:
        matches = get_close_matches(drug, list(gdsc_drug_names), n=1, cutoff=0.85)
        gdsc_name = matches[0] if matches else None
        if gdsc_name:
            gdsc_id = gdsc_drug_id_map.get(gdsc_name)

    # Find best DrugComb name match
    dc_name = drug if drug in dc_drug_names else None
    if not dc_name:
        matches = get_close_matches(drug, list(dc_drug_names), n=1, cutoff=0.85)
        dc_name = matches[0] if matches else None

    master_drug_rows.append({
        "MASTER_DRUG_NAME"  : drug,
        "CHEMBL_ID"         : chembl_id,
        "GDSC_DRUG_NAME"    : gdsc_name,
        "GDSC_DRUG_ID"      : gdsc_id,
        "DRUGCOMB_DRUG_NAME": dc_name,
        "IN_CHEMBL"         : chembl_id is not None,
        "IN_GDSC"           : gdsc_name is not None,
        "IN_DRUGCOMB"       : dc_name is not None,
    })

master_drug_map = pd.DataFrame(master_drug_rows)
fully_mapped = master_drug_map[
    master_drug_map["IN_CHEMBL"] &
    master_drug_map["IN_GDSC"] &
    master_drug_map["IN_DRUGCOMB"]
]

print(f"\nMaster drug map built:")
print(f"  Total overlap drugs          : {len(overlap_drug_list):,}")
print(f"  Successfully mapped          : {len(master_drug_map):,}")
print(f"  With ChEMBL ID               : {master_drug_map['IN_CHEMBL'].sum():,}")
print(f"  With GDSC ID                 : {master_drug_map['IN_GDSC'].sum():,}")
print(f"  With DrugComb name           : {master_drug_map['IN_DRUGCOMB'].sum():,}")
print(f"  Fully mapped (all three)     : {len(fully_mapped):,}  <- ML-ready drugs")
print(f"\nSample master drug map:")
print(master_drug_map[["MASTER_DRUG_NAME","CHEMBL_ID","GDSC_DRUG_ID"]].head(10).to_string())

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — Validate Joins
#           Test that COSMIC IDs and drug IDs join correctly
# ══════════════════════════════════════════════════════════════════════════
section("STEP 5: Validating ID Joins")

# Test 1 — COSMIC ID join: cell map -> GDSC IC50
cosmic_ids = set(master_cell_map["COSMIC_ID"].dropna().astype(int).unique())
gdsc_cosmic = set(pd.to_numeric(
    gdsc_ic50["COSMIC_ID"], errors="coerce").dropna().astype(int).unique()) \
    if "COSMIC_ID" in gdsc_ic50.columns else set()
cosmic_join_ok = cosmic_ids & gdsc_cosmic
print(f"COSMIC ID join test:")
print(f"  Master cell map COSMIC IDs   : {len(cosmic_ids):,}")
print(f"  GDSC IC50 COSMIC IDs         : {len(gdsc_cosmic):,}")
print(f"  Successful joins             : {len(cosmic_join_ok):,} ")

# Test 2 — Drug name join: drug map -> GDSC IC50
gdsc_drug_join = set(master_drug_map["GDSC_DRUG_NAME"].dropna().unique())
gdsc_drugs_all = set(gdsc_ic50["DRUG_NAME"].unique()) \
                 if "DRUG_NAME" in gdsc_ic50.columns else set()
drug_join_ok = gdsc_drug_join & gdsc_drugs_all
print(f"\nDrug name join test (master -> GDSC):")
print(f"  Master drug map GDSC names   : {len(gdsc_drug_join):,}")
print(f"  GDSC IC50 drug names         : {len(gdsc_drugs_all):,}")
print(f"  Successful joins             : {len(drug_join_ok):,} ")

# Test 3 — ChEMBL ID join: drug map -> ChEMBL fingerprints
chembl_ids_map = set(master_drug_map["CHEMBL_ID"].dropna().unique())
chembl_ids_fp  = set(chembl_fp["CHEMBL_ID"].unique()) \
                 if "CHEMBL_ID" in chembl_fp.columns else set()
chembl_join_ok = chembl_ids_map & chembl_ids_fp
print(f"\nChEMBL ID join test (master -> fingerprints):")
print(f"  Master drug map ChEMBL IDs   : {len(chembl_ids_map):,}")
print(f"  ChEMBL FP ChEMBL IDs         : {len(chembl_ids_fp):,}")
print(f"  Successful joins             : {len(chembl_join_ok):,} ")

# Test 4 — DrugComb join: drug map -> DrugComb
dc_drugs_map = set(master_drug_map["DRUGCOMB_DRUG_NAME"].dropna().unique())
dc_drugs_all = set(drugcomb["DRUG1"].unique()) | \
               set(drugcomb["DRUG2"].unique()) \
               if "DRUG1" in drugcomb.columns else set()
dc_join_ok   = dc_drugs_map & dc_drugs_all
print(f"\nDrugComb join test (master -> DrugComb):")
print(f"  Master drug map DC names     : {len(dc_drugs_map):,}")
print(f"  DrugComb drug names          : {len(dc_drugs_all):,}")
print(f"  Successful joins             : {len(dc_join_ok):,} ")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — Build Harmonised DrugComb subset
#           Filter DrugComb to only ML-ready drugs + cell lines
# ══════════════════════════════════════════════════════════════════════════
section("STEP 6: Building Harmonised DrugComb Subset")

# Map DrugComb cell names to GDSC names using fuzzy map
cell_dc_to_gdsc = dict(zip(
    master_cell_map["DRUGCOMB_CELL_NAME"],
    master_cell_map["GDSC_CELL_NAME"]
))
cell_dc_to_cosmic = dict(zip(
    master_cell_map["DRUGCOMB_CELL_NAME"],
    master_cell_map["COSMIC_ID"]
))

# ML-ready drugs and cell lines
ml_drugs = set(fully_mapped["MASTER_DRUG_NAME"].unique())
ml_cells = set(master_cell_map["DRUGCOMB_CELL_NAME"].unique())

print(f"ML-ready drugs     : {len(ml_drugs):,}")
print(f"ML-ready cell lines: {len(ml_cells):,}")

# Filter DrugComb
dc_filtered = drugcomb[
    drugcomb["DRUG1"].isin(ml_drugs) |
    drugcomb["DRUG2"].isin(ml_drugs)
].copy()

dc_filtered = dc_filtered[
    dc_filtered[cell_col_dc].isin(ml_cells)
].copy() if cell_col_dc else dc_filtered

# Add COSMIC ID and GDSC cell name to DrugComb
if cell_col_dc:
    dc_filtered["GDSC_CELL_NAME"] = dc_filtered[cell_col_dc].map(cell_dc_to_gdsc)
    dc_filtered["COSMIC_ID"]      = dc_filtered[cell_col_dc].map(cell_dc_to_cosmic)

print(f"\nHarmonised DrugComb subset:")
print(f"  Original DrugComb rows       : {len(drugcomb):,}")
print(f"  After ML drug/cell filter    : {len(dc_filtered):,}")
print(f"  Synergistic pairs            : {dc_filtered['SYNERGY_LABEL'].sum() if 'SYNERGY_LABEL' in dc_filtered.columns else 'N/A'}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 7 — Save all harmonised outputs
# ══════════════════════════════════════════════════════════════════════════
section("STEP 7: Saving All Harmonised Files")

# Master maps
master_cell_map.to_csv(
    os.path.join(PROJECT_DIR, "master_cell_line_map.csv"), index=False)
print("Saved -> master_cell_line_map.csv")

master_drug_map.to_csv(
    os.path.join(PROJECT_DIR, "master_drug_map.csv"), index=False)
print("Saved -> master_drug_map.csv")

fully_mapped.to_csv(
    os.path.join(PROJECT_DIR, "master_drug_map_fully_mapped.csv"), index=False)
print("Saved -> master_drug_map_fully_mapped.csv")

# Harmonised DrugComb subset
dc_filtered.to_csv(
    os.path.join(PROJECT_DIR, "drugcomb_harmonised.csv"), index=False)
print("Saved -> drugcomb_harmonised.csv")

# Save as Parquet too
try:
    master_cell_map.to_parquet(
        os.path.join(PROJECT_DIR, "master_cell_line_map.parquet"), index=False)
    master_drug_map.to_parquet(
        os.path.join(PROJECT_DIR, "master_drug_map.parquet"), index=False)
    dc_filtered.to_parquet(
        os.path.join(PROJECT_DIR, "drugcomb_harmonised.parquet"), index=False)
    print("Parquet files saved successfully!")
except Exception as e:
    print(f"Parquet save skipped: {e}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 8 — Visualisations
# ══════════════════════════════════════════════════════════════════════════
section("STEP 8: Visualisations")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("ID Harmonisation Overview", fontsize=16, fontweight="bold")

# Plot 1 — Cell line tissue distribution
if "TISSUE_TYPE" in master_cell_map.columns:
    tissue_counts = master_cell_map["TISSUE_TYPE"].value_counts()
    tissue_counts.plot(kind="barh", ax=axes[0,0], color="steelblue")
    axes[0,0].set_title("ML-Ready Cell Lines by Tissue Type")
    axes[0,0].set_xlabel("Number of Cell Lines")

# Plot 2 — Drug mapping status
mapping_status = {
    "In ChEMBL"   : master_drug_map["IN_CHEMBL"].sum(),
    "In GDSC"     : master_drug_map["IN_GDSC"].sum(),
    "In DrugComb" : master_drug_map["IN_DRUGCOMB"].sum(),
    "All Three"   : len(fully_mapped)
}
colors = ["#66b3ff", "#99ff99", "#ffcc99", "#ff9999"]
axes[0,1].bar(mapping_status.keys(), mapping_status.values(), color=colors)
axes[0,1].set_title("Drug Mapping Status Across Databases")
axes[0,1].set_ylabel("Number of Drugs")
for i, (k, v) in enumerate(mapping_status.items()):
    axes[0,1].text(i, v + 0.3, str(v), ha="center", fontweight="bold")

# Plot 3 — Join success rates
join_tests = {
    "COSMIC\n(Cells)"     : len(cosmic_join_ok),
    "Drug\n(GDSC)"        : len(drug_join_ok),
    "ChEMBL ID\n(FP)"     : len(chembl_join_ok),
    "Drug\n(DrugComb)"    : len(dc_join_ok)
}
axes[1,0].bar(join_tests.keys(), join_tests.values(), color="mediumseagreen")
axes[1,0].set_title("ID Join Validation Results")
axes[1,0].set_ylabel("Successful Joins")
for i, (k, v) in enumerate(join_tests.items()):
    axes[1,0].text(i, v + 0.3, str(v), ha="center", fontweight="bold")

# Plot 4 — DrugComb harmonised subset
dc_summary = {
    "Original\nDrugComb"   : len(drugcomb),
    "Harmonised\nSubset"   : len(dc_filtered),
    "ML-Ready\nDrugs"      : len(ml_drugs),
    "ML-Ready\nCell Lines" : len(ml_cells)
}
colors2 = ["#aec6cf", "#77dd77", "#ffb347", "#ff6961"]
axes[1,1].bar(dc_summary.keys(), dc_summary.values(), color=colors2)
axes[1,1].set_title("Dataset Size After Harmonisation")
axes[1,1].set_ylabel("Count")
for i, (k, v) in enumerate(dc_summary.items()):
    axes[1,1].text(i, v + 100, str(v), ha="center", fontweight="bold")

plt.tight_layout()
out = os.path.join(PROJECT_DIR, "id_harmonisation_plots.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"\nPlot saved -> id_harmonisation_plots.png")

# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY + FILE CHECKLIST
# ══════════════════════════════════════════════════════════════════════════
section("FINAL SUMMARY")

print(f"  Master cell line map         : {len(master_cell_map):,} cell lines")
print(f"  Master drug map              : {len(master_drug_map):,} drugs")
print(f"  Fully mapped drugs           : {len(fully_mapped):,} <- ML-ready")
print(f"  Harmonised DrugComb rows     : {len(dc_filtered):,}")
print(f"  COSMIC ID join success       : {len(cosmic_join_ok):,}")
print(f"  ChEMBL ID join success       : {len(chembl_join_ok):,}")

print("\n  --- File Checklist ---")
expected_files = [
    "master_cell_line_map.csv",
    "master_cell_line_map.parquet",
    "master_drug_map.csv",
    "master_drug_map.parquet",
    "master_drug_map_fully_mapped.csv",
    "drugcomb_harmonised.csv",
    "drugcomb_harmonised.parquet",
    "id_harmonisation_plots.png",
]
for f in expected_files:
    path = os.path.join(PROJECT_DIR, f)
    status = "[OK]" if os.path.exists(path) else "[MISSING]"
    print(f"    {status}  {f}")

print("\n STEP 4 — ID Harmonisation 100% COMPLETE!")
print(" Next -> Step 5: Feature Matrix Construction\n")