import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")  # Suppress RDKit warnings

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
PROJECT_DIR = r"C:\Users\sush\PYCHARM\Capstone project"

FILES = {
    "chembl_smiles"  : os.path.join(PROJECT_DIR, "chembl_drugs_smiles.csv"),
    "chembl_targets" : os.path.join(PROJECT_DIR, "chembl_drug_targets.csv"),
    "drugcomb"       : os.path.join(PROJECT_DIR, "drugcomb_cleaned.csv"),
    "gdsc_ic50"      : os.path.join(PROJECT_DIR, "gdsc_ic50_cleaned.csv"),
}

# Morgan fingerprint settings (ECFP4 = radius 2, 1024 bits)
MORGAN_RADIUS   = 2
MORGAN_NBITS    = 1024

def section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def std_name(s):
    """Standardise drug name to UPPERCASE stripped string."""
    return str(s).upper().strip()

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load all files
# ══════════════════════════════════════════════════════════════════════════
section("STEP 1: Loading All Files")

chembl  = pd.read_csv(FILES["chembl_smiles"])
targets = pd.read_csv(FILES["chembl_targets"])
drugcomb = pd.read_csv(FILES["drugcomb"])
gdsc    = pd.read_csv(FILES["gdsc_ic50"])

print(f"ChEMBL SMILES   : {chembl.shape[0]:,} rows x {chembl.shape[1]} cols")
print(f"ChEMBL Targets  : {targets.shape[0]:,} rows x {targets.shape[1]} cols")
print(f"DrugComb        : {drugcomb.shape[0]:,} rows x {drugcomb.shape[1]} cols")
print(f"GDSC IC50       : {gdsc.shape[0]:,} rows x {gdsc.shape[1]} cols")

print(f"\nChEMBL columns  : {list(chembl.columns)}")
print(f"Targets columns : {list(targets.columns)}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Standardise drug names across all datasets
# ══════════════════════════════════════════════════════════════════════════
section("STEP 2: Standardising Drug Names")

# ChEMBL
chembl["DRUG_NAME"] = chembl["pref_name"].apply(std_name)
chembl["CHEMBL_ID"] = chembl["molecule_chembl_id"].astype(str).str.upper().str.strip()

# Also expand synonyms for better matching
def get_synonyms(syn_str):
    """Parse synonym string into list of uppercase names."""
    if pd.isna(syn_str) or syn_str == "":
        return []
    return [std_name(s) for s in str(syn_str).split("|")]

chembl["SYNONYM_LIST"] = chembl["synonyms"].apply(get_synonyms)

print(f"ChEMBL unique drugs (pref_name) : {chembl['DRUG_NAME'].nunique():,}")
print(f"Drugs with synonyms             : {chembl['SYNONYM_LIST'].apply(len).gt(0).sum():,}")

# DrugComb
drugcomb.columns = drugcomb.columns.str.upper().str.strip()
dc_drug_col1 = next((c for c in drugcomb.columns if "DRUG1" in c or "DRUG_1" in c), None)
dc_drug_col2 = next((c for c in drugcomb.columns if "DRUG2" in c or "DRUG_2" in c), None)
if dc_drug_col1:
    drugcomb[dc_drug_col1] = drugcomb[dc_drug_col1].apply(std_name)
if dc_drug_col2:
    drugcomb[dc_drug_col2] = drugcomb[dc_drug_col2].apply(std_name)
dc_drugs = set()
if dc_drug_col1: dc_drugs |= set(drugcomb[dc_drug_col1].dropna().unique())
if dc_drug_col2: dc_drugs |= set(drugcomb[dc_drug_col2].dropna().unique())
print(f"DrugComb unique drugs           : {len(dc_drugs):,}")

# GDSC
gdsc.columns = gdsc.columns.str.upper().str.strip()
gdsc_drug_col = next((c for c in gdsc.columns if "DRUG_NAME" in c or "DRUG" in c), None)
if gdsc_drug_col:
    gdsc[gdsc_drug_col] = gdsc[gdsc_drug_col].apply(std_name)
gdsc_drugs = set(gdsc[gdsc_drug_col].dropna().unique()) if gdsc_drug_col else set()
print(f"GDSC unique drugs               : {len(gdsc_drugs):,}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Validate SMILES & generate Morgan (ECFP4) fingerprints
# ══════════════════════════════════════════════════════════════════════════
section("STEP 3: Generating Morgan Fingerprints (ECFP4)")

print(f"Settings: radius={MORGAN_RADIUS}, nBits={MORGAN_NBITS}")
print(f"Processing {len(chembl):,} drugs...")

valid_mols   = []
invalid_smiles = []
fingerprints = []

for idx, row in chembl.iterrows():
    smiles = row.get("canonical_smiles", "")
    mol = None
    if pd.notna(smiles) and smiles != "":
        mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        invalid_smiles.append(row["DRUG_NAME"])
        continue

    # Generate Morgan fingerprint (ECFP4)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS)
    fp_array = np.array(fp)
    fingerprints.append(fp_array)
    valid_mols.append(row)

print(f"\nValid SMILES     : {len(valid_mols):,}")
print(f"Invalid SMILES   : {len(invalid_smiles):,}")
print(f"Fingerprint shape: ({len(fingerprints)}, {MORGAN_NBITS})")

# Build fingerprint DataFrame
chembl_valid = pd.DataFrame(valid_mols).reset_index(drop=True)
fp_cols = [f"FP_{i}" for i in range(MORGAN_NBITS)]
fp_df   = pd.DataFrame(fingerprints, columns=fp_cols)

# Combine drug info + fingerprints
chembl_fp = pd.concat([
    chembl_valid[["CHEMBL_ID", "DRUG_NAME", "canonical_smiles",
                  "molecular_weight", "alogp", "hbd", "hba",
                  "psa", "num_rings"]].reset_index(drop=True),
    fp_df
], axis=1)

print(f"\nFinal fingerprint table shape: {chembl_fp.shape}")
print(f"Columns: ID + name + SMILES + 7 props + {MORGAN_NBITS} FP bits")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Generate Murcko Scaffolds (for chemotype enrichment)
# ══════════════════════════════════════════════════════════════════════════
section("STEP 4: Generating Murcko Scaffolds")

scaffolds     = []
scaffold_smiles = []

for idx, row in chembl_valid.iterrows():
    smiles = row.get("canonical_smiles", "")
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smi = Chem.MolToSmiles(scaffold)
            scaffolds.append(scaffold_smi)
        else:
            scaffolds.append(None)
    except Exception:
        scaffolds.append(None)
    scaffold_smiles.append(scaffolds[-1])

chembl_fp["MURCKO_SCAFFOLD"] = scaffolds

n_unique_scaffolds = pd.Series(scaffolds).dropna().nunique()
print(f"Scaffolds generated     : {len([s for s in scaffolds if s])}")
print(f"Unique Murcko scaffolds : {n_unique_scaffolds:,}")
print(f"Drugs with no scaffold  : {scaffolds.count(None):,}")

# Top scaffolds
top_scaffolds = pd.Series([s for s in scaffolds if s]).value_counts().head(10)
print(f"\nTop 10 most common scaffolds (by drug count):")
for smi, cnt in top_scaffolds.items():
    print(f"  {cnt} drugs — {smi[:60]}...")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — Drug name overlap: ChEMBL vs DrugComb vs GDSC
# ══════════════════════════════════════════════════════════════════════════
section("STEP 5: Drug Overlap — ChEMBL vs DrugComb vs GDSC")

chembl_names = set(chembl_fp["DRUG_NAME"].dropna().unique())

# Also build synonym lookup for better matching
synonym_to_chembl = {}
for _, row in chembl.iterrows():
    for syn in row["SYNONYM_LIST"]:
        synonym_to_chembl[syn] = row["DRUG_NAME"]

# Direct name overlap
chembl_dc_overlap    = chembl_names & dc_drugs
chembl_gdsc_overlap  = chembl_names & gdsc_drugs
all_three_overlap    = chembl_names & dc_drugs & gdsc_drugs

# Synonym-expanded overlap
dc_via_synonym   = set()
gdsc_via_synonym = set()

for drug in dc_drugs:
    if drug in synonym_to_chembl:
        dc_via_synonym.add(drug)

for drug in gdsc_drugs:
    if drug in synonym_to_chembl:
        gdsc_via_synonym.add(drug)

total_dc_overlap   = chembl_dc_overlap   | dc_via_synonym
total_gdsc_overlap = chembl_gdsc_overlap | gdsc_via_synonym

print(f"\nChEMBL drugs              : {len(chembl_names):,}")
print(f"DrugComb drugs            : {len(dc_drugs):,}")
print(f"GDSC drugs                : {len(gdsc_drugs):,}")
print(f"\nDirect name matches:")
print(f"  ChEMBL ∩ DrugComb       : {len(chembl_dc_overlap):,}")
print(f"  ChEMBL ∩ GDSC           : {len(chembl_gdsc_overlap):,}")
print(f"  All three overlap       : {len(all_three_overlap):,}")
print(f"\nAfter synonym expansion:")
print(f"  ChEMBL ∩ DrugComb       : {len(total_dc_overlap):,}")
print(f"  ChEMBL ∩ GDSC           : {len(total_gdsc_overlap):,}")

# Flag overlapping drugs in fingerprint table
chembl_fp["IN_DRUGCOMB"] = chembl_fp["DRUG_NAME"].isin(total_dc_overlap)
chembl_fp["IN_GDSC"]     = chembl_fp["DRUG_NAME"].isin(total_gdsc_overlap)
chembl_fp["IN_BOTH"]     = chembl_fp["IN_DRUGCOMB"] & chembl_fp["IN_GDSC"]

print(f"\nDrugs flagged in fingerprint table:")
print(f"  IN_DRUGCOMB : {chembl_fp['IN_DRUGCOMB'].sum():,}")
print(f"  IN_GDSC     : {chembl_fp['IN_GDSC'].sum():,}")
print(f"  IN_BOTH     : {chembl_fp['IN_BOTH'].sum():,}  ← usable for ML feature matrix")

# Save overlap lists
pd.DataFrame(sorted(all_three_overlap), columns=["DRUG_NAME"])\
  .to_csv(os.path.join(PROJECT_DIR, "chembl_drugcomb_gdsc_overlap_drugs.csv"), index=False)
print(f"\nThree-way drug overlap saved -> chembl_drugcomb_gdsc_overlap_drugs.csv")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — Save all outputs (CSV + Parquet)
# ══════════════════════════════════════════════════════════════════════════
section("STEP 6: Saving All Outputs (CSV + Parquet)")

# Main fingerprint table
fp_out_csv = os.path.join(PROJECT_DIR, "chembl_morgan_fingerprints.csv")
fp_out_pq  = os.path.join(PROJECT_DIR, "chembl_morgan_fingerprints.parquet")
chembl_fp.to_csv(fp_out_csv, index=False)
print(f"Morgan fingerprints saved -> chembl_morgan_fingerprints.csv  {chembl_fp.shape}")

try:
    chembl_fp.to_parquet(fp_out_pq, index=False)
    print(f"Morgan fingerprints saved -> chembl_morgan_fingerprints.parquet")
except ImportError:
    print("pyarrow not found — skipping parquet. Run: pip install pyarrow")

# Drug info only (without FP bits) — useful for reports
drug_info_cols = ["CHEMBL_ID", "DRUG_NAME", "canonical_smiles",
                  "molecular_weight", "alogp", "hbd", "hba",
                  "psa", "num_rings", "MURCKO_SCAFFOLD",
                  "IN_DRUGCOMB", "IN_GDSC", "IN_BOTH"]
chembl_fp[drug_info_cols].to_csv(
    os.path.join(PROJECT_DIR, "chembl_drug_info_clean.csv"), index=False)
print(f"Drug info (no FP bits) saved -> chembl_drug_info_clean.csv")

# Targets file — standardise and save
targets.columns = targets.columns.str.upper().str.strip()
if "MOLECULE_CHEMBL_ID" in targets.columns:
    targets["MOLECULE_CHEMBL_ID"] = targets["MOLECULE_CHEMBL_ID"].astype(str).str.upper().str.strip()
targets.to_csv(os.path.join(PROJECT_DIR, "chembl_targets_clean.csv"), index=False)
print(f"Targets cleaned saved -> chembl_targets_clean.csv  {targets.shape}")

# Scaffold summary
scaffold_summary = chembl_fp[["DRUG_NAME", "MURCKO_SCAFFOLD", "IN_DRUGCOMB", "IN_GDSC", "IN_BOTH"]]
scaffold_summary.to_csv(os.path.join(PROJECT_DIR, "chembl_murcko_scaffolds.csv"), index=False)
print(f"Murcko scaffolds saved -> chembl_murcko_scaffolds.csv")

# ══════════════════════════════════════════════════════════════════════════
# STEP 7 — Visualisations
# ══════════════════════════════════════════════════════════════════════════
section("STEP 7: Visualisations")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("ChEMBL Fingerprint Overview", fontsize=16, fontweight="bold")

# Plot 1 — Molecular weight distribution
if "molecular_weight" in chembl_fp.columns:
    mw = pd.to_numeric(chembl_fp["molecular_weight"], errors="coerce").dropna()
    mw[mw < 1500].hist(bins=60, ax=axes[0,0], color="steelblue", edgecolor="white")
    axes[0,0].axvline(500, color="red", linestyle="--", label="Lipinski MW=500")
    axes[0,0].set_title("Molecular Weight Distribution")
    axes[0,0].set_xlabel("Molecular Weight (Da)")
    axes[0,0].set_ylabel("Count")
    axes[0,0].legend()

# Plot 2 — ALogP distribution
if "alogp" in chembl_fp.columns:
    logp = pd.to_numeric(chembl_fp["alogp"], errors="coerce").dropna()
    logp.hist(bins=60, ax=axes[0,1], color="coral", edgecolor="white")
    axes[0,1].axvline(5, color="red", linestyle="--", label="Lipinski LogP=5")
    axes[0,1].set_title("ALogP Distribution")
    axes[0,1].set_xlabel("ALogP")
    axes[0,1].set_ylabel("Count")
    axes[0,1].legend()

# Plot 3 — Drug overlap Venn-style bar chart
overlap_data = {
    "ChEMBL only"        : len(chembl_names) - len(total_dc_overlap) - len(total_gdsc_overlap),
    "ChEMBL+DrugComb"    : len(total_dc_overlap),
    "ChEMBL+GDSC"        : len(total_gdsc_overlap),
    "All Three"          : len(all_three_overlap),
}
colors = ["#aec6cf", "#77dd77", "#ffb347", "#ff6961"]
axes[1,0].bar(overlap_data.keys(), overlap_data.values(), color=colors, edgecolor="white")
axes[1,0].set_title("Drug Overlap: ChEMBL vs DrugComb vs GDSC")
axes[1,0].set_ylabel("Number of Drugs")
axes[1,0].tick_params(axis="x", rotation=15)
for i, (k, v) in enumerate(overlap_data.items()):
    axes[1,0].text(i, v + 5, str(v), ha="center", fontweight="bold")

# Plot 4 — Fingerprint bit density (mean ON bits per position)
fp_matrix = chembl_fp[fp_cols].values.astype(float)
bit_density = fp_matrix.mean(axis=0)
axes[1,1].plot(bit_density, color="purple", linewidth=0.5, alpha=0.7)
axes[1,1].set_title(f"Morgan FP Bit Density (ECFP4, {MORGAN_NBITS} bits)")
axes[1,1].set_xlabel("Bit Position")
axes[1,1].set_ylabel("Mean Activation")
axes[1,1].set_ylim(0, bit_density.max() * 1.2)

plt.tight_layout()
out = os.path.join(PROJECT_DIR, "chembl_overview_plots.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"\nPlot saved -> chembl_overview_plots.png")

# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY + FILE CHECKLIST
# ══════════════════════════════════════════════════════════════════════════
section("FINAL SUMMARY")

print(f"  ChEMBL drugs loaded           : {len(chembl):,}")
print(f"  Valid SMILES                  : {len(valid_mols):,}")
print(f"  Invalid SMILES (dropped)      : {len(invalid_smiles):,}")
print(f"  Morgan FP shape               : {chembl_fp.shape}")
print(f"  Unique Murcko scaffolds       : {n_unique_scaffolds:,}")
print(f"  Drug-target relationships     : {len(targets):,}")
print(f"  Drugs in DrugComb             : {len(total_dc_overlap):,}")
print(f"  Drugs in GDSC                 : {len(total_gdsc_overlap):,}")
print(f"  Drugs in ALL THREE datasets   : {len(all_three_overlap):,}  <- ML-ready drugs")

print("\n  --- File Checklist ---")
expected_files = [
    "chembl_drugs_smiles.csv",
    "chembl_drug_targets.csv",
    "chembl_morgan_fingerprints.csv",
    "chembl_morgan_fingerprints.parquet",
    "chembl_drug_info_clean.csv",
    "chembl_targets_clean.csv",
    "chembl_murcko_scaffolds.csv",
    "chembl_drugcomb_gdsc_overlap_drugs.csv",
    "chembl_overview_plots.png",
]
for f in expected_files:
    path = os.path.join(PROJECT_DIR, f)
    status = "[OK]" if os.path.exists(path) else "[MISSING]"
    print(f"    {status}  {f}")

print("\n STEP 3 — ChEMBL Fingerprints COMPLETE!")
print(" Next -> Step 4: ID Harmonisation\n")