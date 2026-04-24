import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from difflib import get_close_matches

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
PROJECT_DIR = r"C:\Users\sush\PYCHARM\Capstone project"

FILES = {
    "ic50"      : os.path.join(PROJECT_DIR, "PANCANCER_IC_Sat Feb 28 03_40_45 2026.csv"),
    "genetic"   : os.path.join(PROJECT_DIR, "PANCANCER_Genetic_features_Sat Feb 28 03_39_55 2026.csv"),
    "cells"     : os.path.join(PROJECT_DIR, "Cell_Lines_Details.xlsx"),
    "dose_resp" : os.path.join(PROJECT_DIR, "GDSC2_fitted_dose_response_27Oct23.xlsx"),
    "drugcomb"  : os.path.join(PROJECT_DIR, "drugcomb_cleaned.csv"),
}

# Quality filter thresholds
RMSE_THRESHOLD   = 0.3    # Drop rows with RMSE > 0.3 (poor sigmoid curve fit)
ZSCORE_THRESHOLD = 3.0    # Drop rows with |Z_SCORE| > 3 (extreme outliers)

def section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def std_cols(df):
    df.columns = (df.columns.str.strip()
                             .str.upper()
                             .str.replace("\n", "_", regex=False)
                             .str.replace(" ", "_")
                             .str.replace("-", "_"))
    return df

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load all files
# ══════════════════════════════════════════════════════════════════════════
section("STEP 1: Loading All Files")

ic50      = pd.read_csv(FILES["ic50"], low_memory=False)
genetic   = pd.read_csv(FILES["genetic"], low_memory=False)
cells     = pd.read_excel(FILES["cells"])
dose_resp = pd.read_excel(FILES["dose_resp"])
drugcomb  = pd.read_csv(FILES["drugcomb"])

for name, df in [("PANCANCER_IC50", ic50), ("GENETIC_FEATURES", genetic),
                 ("CELL_LINES_DETAILS", cells), ("GDSC2_DOSE_RESPONSE", dose_resp),
                 ("DRUGCOMB_CLEANED", drugcomb)]:
    print(f"\n{name}: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"  Columns: {list(df.columns[:8])} ...")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Standardise ALL column names
# ══════════════════════════════════════════════════════════════════════════
section("STEP 2: Standardising Column Names")

ic50      = std_cols(ic50)
genetic   = std_cols(genetic)
cells     = std_cols(cells)
dose_resp = std_cols(dose_resp)
drugcomb  = std_cols(drugcomb)

print("All column names standardised to UPPERCASE.")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Clean PANCANCER_IC50
# ══════════════════════════════════════════════════════════════════════════
section("STEP 3: Cleaning PANCANCER_IC50")

print("\nAll columns:", list(ic50.columns))
print("\nMissing values:\n", ic50.isnull().sum())

# Detect IC50 / LN_IC50 columns
ln_col   = [c for c in ic50.columns if "LN_IC50" in c]
ic50_col = [c for c in ic50.columns if c == "IC50" or "IC_50" in c]
target_cols = ln_col if ln_col else ic50_col
print(f"\nLN_IC50 columns: {ln_col}")
print(f"IC50 columns   : {ic50_col}")

# Drop rows missing IC50
before = len(ic50)
ic50.dropna(subset=target_cols, inplace=True)
print(f"Rows dropped (missing IC50): {before - len(ic50):,}")

# Standardise drug & cell line names
if "DRUG_NAME" in ic50.columns:
    ic50["DRUG_NAME"] = ic50["DRUG_NAME"].astype(str).str.upper().str.strip()
    print(f"Unique drugs     : {ic50['DRUG_NAME'].nunique():,}")

cell_col_ic50 = next((c for c in ic50.columns if "CELL" in c and "NAME" in c), None)
if cell_col_ic50:
    ic50[cell_col_ic50] = ic50[cell_col_ic50].astype(str).str.upper().str.strip()
    print(f"Unique cell lines: {ic50[cell_col_ic50].nunique():,}")

# Drug coverage stats
drug_cov = None
if "DRUG_NAME" in ic50.columns and cell_col_ic50:
    drug_cov = ic50.groupby("DRUG_NAME")[cell_col_ic50].nunique().sort_values(ascending=False)
    print(f"\nTop 10 drugs by cell line coverage:\n{drug_cov.head(10)}")
    print(f"Drugs tested in <10 cell lines: {(drug_cov < 10).sum():,}")
    # Save drug coverage stats for report
    drug_cov.reset_index().rename(columns={cell_col_ic50: "NUM_CELL_LINES"})\
            .to_csv(os.path.join(PROJECT_DIR, "gdsc_drug_coverage_stats.csv"), index=False)
    print("Drug coverage stats saved -> gdsc_drug_coverage_stats.csv")

print(f"\nIC50 shape after cleaning: {ic50.shape}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Clean GDSC2 Dose Response + extract LN_IC50
#          NEW: RMSE quality filter + Z-score outlier filter
# ══════════════════════════════════════════════════════════════════════════
section("STEP 4: Cleaning GDSC2 Dose Response & Extracting LN_IC50")

print("\nAll columns:", list(dose_resp.columns))
print("\nMissing values:\n", dose_resp.isnull().sum())

# Filter to GDSC2 only
if "DATASET" in dose_resp.columns:
    before = len(dose_resp)
    dose_resp = dose_resp[dose_resp["DATASET"] == "GDSC2"].copy()
    print(f"\nFiltered to GDSC2: {len(dose_resp):,} rows (was {before:,})")

# Standardise names
if "DRUG_NAME" in dose_resp.columns:
    dose_resp["DRUG_NAME"] = dose_resp["DRUG_NAME"].astype(str).str.upper().str.strip()
cell_col_dr = next((c for c in dose_resp.columns if "CELL" in c and "NAME" in c), None)
if cell_col_dr:
    dose_resp[cell_col_dr] = dose_resp[cell_col_dr].astype(str).str.upper().str.strip()

# Extract & verify LN_IC50
ln_cols_dr = [c for c in dose_resp.columns if "LN_IC50" in c]
print(f"\nLN_IC50 columns: {ln_cols_dr}")
if ln_cols_dr:
    print(f"LN_IC50 stats:\n{dose_resp[ln_cols_dr[0]].describe()}")
    before = len(dose_resp)
    dose_resp.dropna(subset=ln_cols_dr, inplace=True)
    print(f"Rows dropped (missing LN_IC50): {before - len(dose_resp):,}")

# ── NEW FIX 1: RMSE Quality Filter ────────────────────────────────────────
# Rows with high RMSE = sigmoid curve fit poorly = unreliable IC50 value
if "RMSE" in dose_resp.columns:
    before = len(dose_resp)
    rmse_stats = dose_resp["RMSE"].describe()
    print(f"\nRMSE stats before filter:\n{rmse_stats}")
    dose_resp = dose_resp[dose_resp["RMSE"] <= RMSE_THRESHOLD].copy()
    removed = before - len(dose_resp)
    pct = removed / before * 100
    print(f"RMSE filter (>{RMSE_THRESHOLD}): removed {removed:,} rows ({pct:.1f}%)")
    print(f"Rows remaining after RMSE filter: {len(dose_resp):,}")
else:
    print("\nRMSE column not found — skipping RMSE filter")

# ── NEW FIX 2: Z-score Outlier Filter ─────────────────────────────────────
# Extreme Z-scores indicate measurements far outside the expected range
if "Z_SCORE" in dose_resp.columns:
    before = len(dose_resp)
    zscore_stats = dose_resp["Z_SCORE"].describe()
    print(f"\nZ_SCORE stats before filter:\n{zscore_stats}")
    dose_resp = dose_resp[dose_resp["Z_SCORE"].abs() <= ZSCORE_THRESHOLD].copy()
    removed = before - len(dose_resp)
    pct = removed / before * 100
    print(f"Z-score filter (|Z|>{ZSCORE_THRESHOLD}): removed {removed:,} rows ({pct:.1f}%)")
    print(f"Rows remaining after Z-score filter: {len(dose_resp):,}")
else:
    print("\nZ_SCORE column not found — skipping Z-score filter")

# ── NEW FIX 3: COSMIC ID Harmonisation ────────────────────────────────────
# COSMIC ID is the stable cell line identifier — needed for Step 4 joins
cosmic_col_dr = next((c for c in dose_resp.columns if "COSMIC" in c), None)
if cosmic_col_dr:
    dose_resp[cosmic_col_dr] = pd.to_numeric(dose_resp[cosmic_col_dr], errors="coerce")
    n_missing_cosmic = dose_resp[cosmic_col_dr].isna().sum()
    print(f"\nCOSMIC ID column: '{cosmic_col_dr}'")
    print(f"Missing COSMIC IDs: {n_missing_cosmic:,}")
    print(f"Unique COSMIC IDs: {dose_resp[cosmic_col_dr].nunique():,}")
else:
    print("\nCOSMIC ID column not found in dose response")

# Also harmonise COSMIC ID in IC50
cosmic_col_ic50 = next((c for c in ic50.columns if "COSMIC" in c), None)
if cosmic_col_ic50:
    ic50[cosmic_col_ic50] = pd.to_numeric(ic50[cosmic_col_ic50], errors="coerce")
    print(f"COSMIC ID harmonised in IC50: '{cosmic_col_ic50}' — {ic50[cosmic_col_ic50].nunique():,} unique IDs")

print(f"\nDose response shape after all filters: {dose_resp.shape}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — Clean Cell Lines + extract tissue labels
# ══════════════════════════════════════════════════════════════════════════
section("STEP 5: Cleaning Cell Lines + Extracting Tissue Labels")

print("\nAll columns:", list(cells.columns))
print("\nMissing values:\n", cells.isnull().sum())

# Find cell line name column
cell_name_col = next((c for c in cells.columns
                      if ("SAMPLE" in c) or ("CELL" in c and "NAME" in c)), None)
if cell_name_col:
    cells[cell_name_col] = cells[cell_name_col].astype(str).str.upper().str.strip()
    print(f"\nCell line column : '{cell_name_col}'")
    print(f"Unique cell lines: {cells[cell_name_col].nunique():,}")

# Harmonise COSMIC ID in cell lines too
cosmic_col_cells = next((c for c in cells.columns if "COSMIC" in c), None)
if cosmic_col_cells:
    cells[cosmic_col_cells] = pd.to_numeric(cells[cosmic_col_cells], errors="coerce")
    print(f"COSMIC ID harmonised in cell lines: '{cosmic_col_cells}'")

# Find tissue column
tissue_col = None
for c in cells.columns:
    if any(kw in c for kw in ["TISSUE", "CANCER_TYPE", "SITE"]):
        tissue_col = c
        print(f"Tissue column found: '{c}'")
        break

if tissue_col and cell_name_col:
    tissue_map = cells[[cell_name_col, tissue_col]].dropna()
    tissue_map.columns = ["CELL_LINE_NAME", "TISSUE_TYPE"]
    tissue_map.to_csv(os.path.join(PROJECT_DIR, "gdsc_tissue_map.csv"), index=False)
    print(f"\nTissue map saved -> gdsc_tissue_map.csv ({len(tissue_map):,} rows)")
    print(f"\nTop 15 tissue types:\n{tissue_map['TISSUE_TYPE'].value_counts().head(15)}")
else:
    print("Tissue column not found — all columns:")
    for c in cells.columns:
        print(f"   {c}")

print(f"\nCell lines shape: {cells.shape}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — Clean Genetic Features: separate Mutation vs CNV
# ══════════════════════════════════════════════════════════════════════════
section("STEP 6: Cleaning Genetic Features — Separating Mutation & CNV")

print(f"\nShape: {genetic.shape[0]:,} rows x {genetic.shape[1]} cols")
print("\nColumns:", list(genetic.columns))
print("\nMissing values:\n", genetic.isnull().sum())

# Drop columns >50% missing
before_cols = genetic.shape[1]
thresh = int(0.5 * len(genetic))
genetic.dropna(axis=1, thresh=thresh, inplace=True)
print(f"\nColumns dropped (>50% missing): {before_cols - genetic.shape[1]}")

# Standardise cell line col
cell_col_gen = next((c for c in genetic.columns if "CELL" in c), None)
if cell_col_gen:
    genetic[cell_col_gen] = genetic[cell_col_gen].astype(str).str.upper().str.strip()
    print(f"Unique cell lines in genetic: {genetic[cell_col_gen].nunique():,}")

# Harmonise COSMIC ID in genetic features
cosmic_col_gen = next((c for c in genetic.columns if "COSMIC" in c), None)
if cosmic_col_gen:
    genetic[cosmic_col_gen] = pd.to_numeric(genetic[cosmic_col_gen], errors="coerce")
    print(f"COSMIC ID harmonised in genetic: '{cosmic_col_gen}'")

# Separate mutation vs CNV
mut_cols = [c for c in genetic.columns if any(k in c for k in ["MUT", "MUTATION", "IS_MUTATED"])]
cnv_cols = [c for c in genetic.columns if any(k in c for k in ["GAIN", "LOSS", "CNV", "CNA", "RECURRENT"])]

print(f"\nMutation columns: {mut_cols}")
print(f"CNV columns     : {cnv_cols}")

# Save mutation file
if cell_col_gen and mut_cols:
    mut_df = genetic[[cell_col_gen] + mut_cols].copy()
    mut_df.to_csv(os.path.join(PROJECT_DIR, "gdsc_mutations.csv"), index=False)
    print(f"\nMutation features saved -> gdsc_mutations.csv {mut_df.shape}")

# Save CNV file
if cell_col_gen and cnv_cols:
    cnv_df = genetic[genetic[cnv_cols[0]].notna()].copy()
    keep = [cell_col_gen, "GENETIC_FEATURE"] + cnv_cols \
           if "GENETIC_FEATURE" in genetic.columns else [cell_col_gen] + cnv_cols
    cnv_df = cnv_df[[c for c in keep if c in genetic.columns]]
    cnv_df.to_csv(os.path.join(PROJECT_DIR, "gdsc_cnv.csv"), index=False)
    print(f"CNV features saved -> gdsc_cnv.csv {cnv_df.shape}")
elif "GENETIC_FEATURE" in genetic.columns:
    cnv_df = genetic[genetic["GENETIC_FEATURE"].str.contains(
                     "gain|loss|cnv|cna", case=False, na=False)]
    if len(cnv_df) > 0:
        cnv_df.to_csv(os.path.join(PROJECT_DIR, "gdsc_cnv.csv"), index=False)
        print(f"CNV saved via fallback -> gdsc_cnv.csv {cnv_df.shape}")
    else:
        print("CNV data not found — may be in a separate file")

# ══════════════════════════════════════════════════════════════════════════
# STEP 7 — Overlap check with fuzzy matching
# ══════════════════════════════════════════════════════════════════════════
section("STEP 7: Cell Line Overlap — GDSC vs DrugComb (with Fuzzy Matching)")

gdsc_cells = list(ic50[cell_col_ic50].dropna().unique()) if cell_col_ic50 else []

dc_cell_col = next((c for c in drugcomb.columns if "CELL" in c), None)
if dc_cell_col:
    drugcomb[dc_cell_col] = drugcomb[dc_cell_col].astype(str).str.upper().str.strip()
dc_cells = list(drugcomb[dc_cell_col].unique()) if dc_cell_col else []

# Exact match
exact_overlap = set(gdsc_cells) & set(dc_cells)
print(f"\nGDSC cell lines    : {len(gdsc_cells):,}")
print(f"DrugComb cell lines: {len(dc_cells):,}")
print(f"Exact matches      : {len(exact_overlap):,}")

# Fuzzy match for unmatched DrugComb cells
unmatched_dc = [c for c in dc_cells if c not in exact_overlap]
print(f"Trying fuzzy match on {len(unmatched_dc):,} unmatched DrugComb cells...")

fuzzy_map = {}
for dc_cell in unmatched_dc:
    matches = get_close_matches(dc_cell, gdsc_cells, n=1, cutoff=0.85)
    if matches:
        fuzzy_map[dc_cell] = matches[0]

print(f"Fuzzy matches found: {len(fuzzy_map):,}")
if fuzzy_map:
    print("Sample fuzzy matches:")
    for k, v in list(fuzzy_map.items())[:10]:
        print(f"   DrugComb: '{k}' -> GDSC: '{v}'")

all_overlap = exact_overlap | set(fuzzy_map.values())
print(f"\nTotal overlap after fuzzy: {len(all_overlap):,} cell lines")

# Save overlap + fuzzy map
pd.DataFrame(sorted(all_overlap), columns=["CELL_LINE_NAME"])\
  .to_csv(os.path.join(PROJECT_DIR, "gdsc_drugcomb_overlap_cells.csv"), index=False)
print("Overlap saved -> gdsc_drugcomb_overlap_cells.csv")

if fuzzy_map:
    pd.DataFrame(list(fuzzy_map.items()), columns=["DRUGCOMB_NAME", "GDSC_NAME"])\
      .to_csv(os.path.join(PROJECT_DIR, "gdsc_drugcomb_fuzzy_name_map.csv"), index=False)
    print("Fuzzy name map saved -> gdsc_drugcomb_fuzzy_name_map.csv")

# Drug overlap
drug_overlap = set()
if "DRUG_NAME" in ic50.columns:
    gdsc_drugs = set(ic50["DRUG_NAME"].dropna().unique())
    dc_drug1   = set(drugcomb["DRUG1"].str.upper().str.strip()) if "DRUG1" in drugcomb.columns else set()
    dc_drug2   = set(drugcomb["DRUG2"].str.upper().str.strip()) if "DRUG2" in drugcomb.columns else set()
    drug_overlap = gdsc_drugs & (dc_drug1 | dc_drug2)
    print(f"\nGDSC drugs       : {len(gdsc_drugs):,}")
    print(f"DrugComb drugs   : {len(dc_drug1 | dc_drug2):,}")
    print(f"Drug overlap     : {len(drug_overlap):,}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 8 — Save all cleaned files (CSV + Parquet)
# ══════════════════════════════════════════════════════════════════════════
section("STEP 8: Saving All Cleaned Files (CSV + Parquet)")

# Save as CSV
ic50.to_csv(os.path.join(PROJECT_DIR, "gdsc_ic50_cleaned.csv"), index=False)
dose_resp.to_csv(os.path.join(PROJECT_DIR, "gdsc_dose_response_cleaned.csv"), index=False)
cells.to_csv(os.path.join(PROJECT_DIR, "gdsc_cell_lines_cleaned.csv"), index=False)
genetic.to_csv(os.path.join(PROJECT_DIR, "gdsc_genetic_features_cleaned.csv"), index=False)

print("CSV files saved:")
print("  gdsc_ic50_cleaned.csv")
print("  gdsc_dose_response_cleaned.csv")
print("  gdsc_cell_lines_cleaned.csv")
print("  gdsc_genetic_features_cleaned.csv")

# ── NEW FIX 4: Save big files as Parquet for fast IO in Steps 4/5 ─────────
print("\nSaving Parquet files for fast IO in Steps 4/5...")
try:
    ic50.to_parquet(os.path.join(PROJECT_DIR, "gdsc_ic50_cleaned.parquet"), index=False)
    dose_resp.to_parquet(os.path.join(PROJECT_DIR, "gdsc_dose_response_cleaned.parquet"), index=False)
    genetic.to_parquet(os.path.join(PROJECT_DIR, "gdsc_genetic_features_cleaned.parquet"), index=False)
    print("  gdsc_ic50_cleaned.parquet")
    print("  gdsc_dose_response_cleaned.parquet")
    print("  gdsc_genetic_features_cleaned.parquet")
    print("Parquet files saved successfully!")
except ImportError:
    print("pyarrow not installed — run: pip install pyarrow")
    print("Parquet saving skipped, CSV files are sufficient for now.")

# ══════════════════════════════════════════════════════════════════════════
# STEP 9 — Visualisations (added RMSE distribution plot)
# ══════════════════════════════════════════════════════════════════════════
section("STEP 9: Visualisations")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("GDSC Preprocessing Overview", fontsize=16, fontweight="bold")

# Plot 1 — LN_IC50 distribution
ln_src = ln_cols_dr if ln_cols_dr else target_cols
if ln_src:
    src_df = dose_resp if ln_cols_dr else ic50
    src_df[ln_src[0]].dropna().hist(bins=60, ax=axes[0,0], color="steelblue", edgecolor="white")
    axes[0,0].set_title("LN_IC50 Distribution (GDSC2 — after quality filters)")
    axes[0,0].set_xlabel("LN_IC50"); axes[0,0].set_ylabel("Count")

# Plot 2 — Top tissue types
if tissue_col and cell_name_col:
    top_t = cells[tissue_col].value_counts().head(15)
    top_t.plot(kind="barh", ax=axes[0,1], color="coral")
    axes[0,1].set_title("Top 15 Tissue Types")
    axes[0,1].set_xlabel("Number of Cell Lines")
else:
    axes[0,1].text(0.5, 0.5, "Tissue data\nnot available",
                   ha="center", va="center", transform=axes[0,1].transAxes)
    axes[0,1].set_title("Tissue Types")

# Plot 3 — Drug coverage
if drug_cov is not None and cell_col_ic50:
    drug_cov.head(20).plot(kind="bar", ax=axes[1,0], color="mediumseagreen")
    axes[1,0].set_title("Top 20 Drugs by Cell Line Coverage")
    axes[1,0].set_xlabel("Drug"); axes[1,0].set_ylabel("# Cell Lines")
    axes[1,0].tick_params(axis="x", rotation=90)

# Plot 4 — Overlap bar chart
overlap_data = {
    "GDSC only"    : len(set(gdsc_cells) - all_overlap),
    "Overlap"      : len(all_overlap),
    "DrugComb only": len(set(dc_cells) - all_overlap)
}
colors = ["#ff9999", "#66b3ff", "#ffcc99"]
axes[1,1].bar(overlap_data.keys(), overlap_data.values(), color=colors, edgecolor="white")
axes[1,1].set_title("Cell Line Overlap: GDSC vs DrugComb")
axes[1,1].set_ylabel("Number of Cell Lines")
for i, (k, v) in enumerate(overlap_data.items()):
    axes[1,1].text(i, v + 1, str(v), ha="center", fontweight="bold")

plt.tight_layout()
out = os.path.join(PROJECT_DIR, "gdsc_overview_plots.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"\nPlot saved -> gdsc_overview_plots.png")

# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY + FILE CHECKLIST
# ══════════════════════════════════════════════════════════════════════════
section("FINAL SUMMARY")

print(f"  IC50 cleaned rows                : {len(ic50):,}")
print(f"  Dose response rows (after filter): {len(dose_resp):,}")
print(f"  Cell lines                       : {len(cells):,}")
print(f"  Genetic features shape           : {genetic.shape[0]:,} rows x {genetic.shape[1]} cols")
print(f"  Cell line overlap                : {len(all_overlap):,}")
print(f"  Drug overlap                     : {len(drug_overlap):,}")

print("\n  --- File Checklist ---")
expected_files = [
    "drugcomb_cleaned.csv",
    "gdsc_ic50_cleaned.csv",
    "gdsc_ic50_cleaned.parquet",
    "gdsc_dose_response_cleaned.csv",
    "gdsc_dose_response_cleaned.parquet",
    "gdsc_genetic_features_cleaned.csv",
    "gdsc_genetic_features_cleaned.parquet",
    "gdsc_cell_lines_cleaned.csv",
    "gdsc_mutations.csv",
    "gdsc_cnv.csv",
    "gdsc_tissue_map.csv",
    "gdsc_drugcomb_overlap_cells.csv",
    "gdsc_drugcomb_fuzzy_name_map.csv",
    "gdsc_drug_coverage_stats.csv",
]
for f in expected_files:
    path = os.path.join(PROJECT_DIR, f)
    status = "[OK]" if os.path.exists(path) else "[MISSING]"
    print(f"    {status}  {f}")

print("\n STEP 2 — GDSC Preprocessing  COMPLETE!")
print(" Next -> Step 3: ChEMBL Fingerprints\n")