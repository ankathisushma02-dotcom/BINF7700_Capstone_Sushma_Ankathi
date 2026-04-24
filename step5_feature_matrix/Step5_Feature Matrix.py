import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
PROJECT_DIR = r"C:\Users\sush\PYCHARM\Capstone project"

FILES = {
    "drugcomb_harmonised" : os.path.join(PROJECT_DIR, "drugcomb_harmonised.csv"),
    "master_drug_map"     : os.path.join(PROJECT_DIR, "master_drug_map.csv"),
    "master_cell_map"     : os.path.join(PROJECT_DIR, "master_cell_line_map.csv"),
    "chembl_fp"           : os.path.join(PROJECT_DIR, "chembl_morgan_fingerprints.csv"),
    "chembl_drug_info"    : os.path.join(PROJECT_DIR, "chembl_drug_info_clean.csv"),
    "chembl_murcko"       : os.path.join(PROJECT_DIR, "chembl_murcko_scaffolds.csv"),
    "chembl_targets"      : os.path.join(PROJECT_DIR, "chembl_targets_clean.csv"),
    "gdsc_dose_resp"      : os.path.join(PROJECT_DIR, "gdsc_dose_response_cleaned.csv"),
    "gdsc_ic50"           : os.path.join(PROJECT_DIR, "gdsc_ic50_cleaned.csv"),
    "gdsc_mutations"      : os.path.join(PROJECT_DIR, "gdsc_mutations.csv"),
    "gdsc_cnv"            : os.path.join(PROJECT_DIR, "gdsc_cnv.csv"),
    "gdsc_tissue"         : os.path.join(PROJECT_DIR, "gdsc_tissue_map.csv"),
    "gdsc_cell_lines"     : os.path.join(PROJECT_DIR, "gdsc_cell_lines_cleaned.csv"),
    "gdsc_gen_features"   : os.path.join(PROJECT_DIR, "gdsc_genetic_features_cleaned.csv"),
    "lincs"               : os.path.join(PROJECT_DIR, "lincs_expression_features.csv"),
}

MAX_MUT_FEATURES = 100
MAX_CNV_FEATURES = 100
RANDOM_STATE     = 42
TEST_SIZE        = 0.2

def section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def std(s):
    return str(s).upper().strip()

def strip_cell(s):
    return str(s).upper().strip()\
        .replace("-","").replace(" ","").replace(".","")

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load all files
# ══════════════════════════════════════════════════════════════════════════
section("STEP 1: Loading All Files")

drugcomb   = pd.read_csv(FILES["drugcomb_harmonised"])
drug_map   = pd.read_csv(FILES["master_drug_map"])
cell_map   = pd.read_csv(FILES["master_cell_map"])
chembl_fp  = pd.read_csv(FILES["chembl_fp"])
drug_info  = pd.read_csv(FILES["chembl_drug_info"])
murcko     = pd.read_csv(FILES["chembl_murcko"])
targets    = pd.read_csv(FILES["chembl_targets"])
gdsc_dose  = pd.read_csv(FILES["gdsc_dose_resp"])
gdsc_ic50  = pd.read_csv(FILES["gdsc_ic50"])
mutations  = pd.read_csv(FILES["gdsc_mutations"])
cnv        = pd.read_csv(FILES["gdsc_cnv"])
tissue     = pd.read_csv(FILES["gdsc_tissue"])
cell_lines = pd.read_csv(FILES["gdsc_cell_lines"])
gen_feat   = pd.read_csv(FILES["gdsc_gen_features"], low_memory=False)
lincs      = pd.read_csv(FILES["lincs"])

for df_obj in [drugcomb, drug_map, cell_map, chembl_fp, drug_info,
               murcko, targets, gdsc_dose, gdsc_ic50, mutations,
               cnv, tissue, cell_lines, gen_feat, lincs]:
    df_obj.columns = df_obj.columns.str.upper().str.strip()

print(f"DrugComb harmonised : {drugcomb.shape}")
print(f"ChEMBL fingerprints : {chembl_fp.shape}")
print(f"GDSC dose response  : {gdsc_dose.shape}")
print(f"GDSC mutations      : {mutations.shape}")
print(f"GDSC CNV            : {cnv.shape}")
print(f"LINCS expression    : {lincs.shape}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Standardise names + build ID maps
# ══════════════════════════════════════════════════════════════════════════
section("STEP 2: Standardising All Names")

dc_drug1_col = next((c for c in drugcomb.columns if "DRUG1" in c), None)
dc_drug2_col = next((c for c in drugcomb.columns if "DRUG2" in c), None)
dc_cell_col  = next((c for c in drugcomb.columns if "CELL" in c), None)
dc_label_col = next((c for c in drugcomb.columns
                     if "SYNERGY" in c or "LABEL" in c), None)
dc_loewe_col = next((c for c in drugcomb.columns if "LOEWE" in c), None)
dc_zip_col   = next((c for c in drugcomb.columns if c == "ZIP"), None)
dc_bliss_col = next((c for c in drugcomb.columns if c == "BLISS"), None)
dc_hsa_col   = next((c for c in drugcomb.columns if c == "HSA"), None)

print(f"Drug1={dc_drug1_col}, Drug2={dc_drug2_col}, "
      f"Cell={dc_cell_col}, Label={dc_label_col}")
print(f"ZIP={dc_zip_col}, Bliss={dc_bliss_col}, HSA={dc_hsa_col}")

if dc_drug1_col: drugcomb[dc_drug1_col] = drugcomb[dc_drug1_col].apply(std)
if dc_drug2_col: drugcomb[dc_drug2_col] = drugcomb[dc_drug2_col].apply(std)
if dc_cell_col:  drugcomb[dc_cell_col]  = drugcomb[dc_cell_col].apply(std)

drug_map["MASTER_DRUG_NAME"]   = drug_map["MASTER_DRUG_NAME"].apply(std)
cell_map["GDSC_CELL_NAME"]     = cell_map["GDSC_CELL_NAME"].apply(std)
cell_map["DRUGCOMB_CELL_NAME"] = cell_map["DRUGCOMB_CELL_NAME"].apply(std)

cell_to_cosmic = {}
for _, row in cell_map.iterrows():
    cid = row["COSMIC_ID"]
    if pd.notna(cid):
        for n in [row["GDSC_CELL_NAME"], row["DRUGCOMB_CELL_NAME"]]:
            cell_to_cosmic[std(n)]        = int(cid)
            cell_to_cosmic[strip_cell(n)] = int(cid)

drug_to_chembl  = dict(zip(drug_map["MASTER_DRUG_NAME"],
                           drug_map["CHEMBL_ID"]))
drug_to_gdsc_id = dict(zip(drug_map["MASTER_DRUG_NAME"],
                           drug_map["GDSC_DRUG_ID"]))
chembl_to_drug  = {v: k for k, v in drug_to_chembl.items()
                   if pd.notna(v)}

print(f"COSMIC map     : {len(cell_to_cosmic)} entries")
print(f"Drug-ChEMBL map: {len(drug_to_chembl)} entries")
print("All names standardised ")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Drug Feature Table
# ══════════════════════════════════════════════════════════════════════════
section("STEP 3: Building Drug Feature Table")

# Morgan fingerprints
fp_cols = [c for c in chembl_fp.columns if c.startswith("FP_")]
chembl_fp["DRUG_NAME"] = chembl_fp["DRUG_NAME"].apply(std) \
    if "DRUG_NAME" in chembl_fp.columns \
    else chembl_fp.iloc[:,1].apply(std)
fp_table = chembl_fp[["DRUG_NAME"] + fp_cols]\
    .drop_duplicates(subset=["DRUG_NAME"])
print(f"FP table          : {fp_table.shape}")

# Drug properties
drug_info["DRUG_NAME"] = drug_info["DRUG_NAME"].apply(std) \
    if "DRUG_NAME" in drug_info.columns \
    else drug_info.iloc[:,1].apply(std)
prop_cols = ["DRUG_NAME"] + [
    c for c in ["MOLECULAR_WEIGHT","ALOGP","HBD",
                "HBA","PSA","NUM_RINGS"]
    if c in drug_info.columns]
prop_table = drug_info[prop_cols].drop_duplicates(subset=["DRUG_NAME"])
for c in prop_cols[1:]:
    prop_table[c] = pd.to_numeric(prop_table[c], errors="coerce")
print(f"Properties table  : {prop_table.shape}")

# Murcko scaffold
murcko["DRUG_NAME"] = murcko["DRUG_NAME"].apply(std) \
    if "DRUG_NAME" in murcko.columns else murcko.iloc[:,0].apply(std)
scaffold_col = next((c for c in murcko.columns
                     if "SCAFFOLD" in c or "MURCKO" in c), None)
if scaffold_col:
    murcko["SCAFFOLD_ID"] = pd.Categorical(murcko[scaffold_col]).codes
    scaffold_table = murcko[["DRUG_NAME","SCAFFOLD_ID"]]\
        .drop_duplicates(subset=["DRUG_NAME"])
else:
    scaffold_table = pd.DataFrame(
        columns=["DRUG_NAME","SCAFFOLD_ID"])
print(f"Scaffold table    : {scaffold_table.shape}")

# FIX: Target count for ALL drugs, fill 0 for missing
targets.columns = targets.columns.str.upper().str.strip()
target_mol_col  = next((c for c in targets.columns
                        if "MOLECULE" in c or "CHEMBL_ID" in c), None)
target_tgt_col  = next((c for c in targets.columns
                        if "TARGET" in c), None)
if target_mol_col and target_tgt_col:
    tgt_cnt = targets.groupby(target_mol_col)[target_tgt_col]\
        .nunique().reset_index()
    tgt_cnt.columns = ["CHEMBL_ID","N_TARGETS"]
    tgt_cnt["DRUG_NAME"] = tgt_cnt["CHEMBL_ID"].map(chembl_to_drug)
    # Fill 0 for drugs not in targets file
    all_drugs = fp_table["DRUG_NAME"].tolist()
    found     = set(tgt_cnt["DRUG_NAME"].dropna())
    missing   = [d for d in all_drugs if d not in found]
    if missing:
        fill_df = pd.DataFrame({
            "CHEMBL_ID" : [drug_to_chembl.get(d,np.nan)
                           for d in missing],
            "N_TARGETS" : [0]*len(missing),
            "DRUG_NAME" : missing
        })
        tgt_cnt = pd.concat([tgt_cnt, fill_df], ignore_index=True)
    target_count = tgt_cnt[["DRUG_NAME","N_TARGETS"]]\
        .dropna(subset=["DRUG_NAME"])
    print(f"Target count table: {target_count.shape}")
else:
    target_count = pd.DataFrame(columns=["DRUG_NAME","N_TARGETS"])

# Merge all drug features
drug_features = fp_table.copy()
drug_features = drug_features.merge(prop_table,    on="DRUG_NAME", how="left")
drug_features = drug_features.merge(scaffold_table,on="DRUG_NAME", how="left")
drug_features = drug_features.merge(target_count,  on="DRUG_NAME", how="left")
print(f"Final drug features: {drug_features.shape}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Cell Line Feature Table
# ══════════════════════════════════════════════════════════════════════════
section("STEP 4: Building Cell Line Feature Table")

# LN_IC50
gdsc_dose["DRUG_NAME"]      = gdsc_dose["DRUG_NAME"].apply(std) \
    if "DRUG_NAME" in gdsc_dose.columns else gdsc_dose.iloc[:,8].apply(std)
gdsc_dose["CELL_LINE_NAME"] = gdsc_dose["CELL_LINE_NAME"].apply(std) \
    if "CELL_LINE_NAME" in gdsc_dose.columns \
    else gdsc_dose.iloc[:,4].apply(std)
ln_ic50_col = next((c for c in gdsc_dose.columns
                    if "LN_IC50" in c), None)
if ln_ic50_col:
    ic50_pivot = gdsc_dose.groupby(
        ["DRUG_NAME","CELL_LINE_NAME"])[ln_ic50_col]\
        .mean().reset_index()
    ic50_pivot.columns = ["DRUG_NAME","CELL_LINE_NAME","LN_IC50"]
    print(f"LN_IC50 table : {ic50_pivot.shape}")
else:
    ic50_pivot = pd.DataFrame(
        columns=["DRUG_NAME","CELL_LINE_NAME","LN_IC50"])

# AUC
gdsc_ic50["DRUG_NAME"]      = gdsc_ic50["DRUG_NAME"].apply(std) \
    if "DRUG_NAME" in gdsc_ic50.columns else gdsc_ic50.iloc[:,0].apply(std)
gdsc_ic50["CELL_LINE_NAME"] = gdsc_ic50["CELL_LINE_NAME"].apply(std) \
    if "CELL_LINE_NAME" in gdsc_ic50.columns \
    else gdsc_ic50.iloc[:,2].apply(std)
auc_col = next((c for c in gdsc_ic50.columns if "AUC" in c), None)
if auc_col:
    auc_pivot = gdsc_ic50.groupby(
        ["DRUG_NAME","CELL_LINE_NAME"])[auc_col]\
        .mean().reset_index()
    auc_pivot.columns = ["DRUG_NAME","CELL_LINE_NAME","AUC"]
    print(f"AUC table     : {auc_pivot.shape}")
else:
    auc_pivot = pd.DataFrame(
        columns=["DRUG_NAME","CELL_LINE_NAME","AUC"])

# Mutations
gen_feat["CELL_LINE_NAME"] = gen_feat["CELL_LINE_NAME"].apply(std) \
    if "CELL_LINE_NAME" in gen_feat.columns \
    else gen_feat.iloc[:,0].apply(std)
gene_col    = next((c for c in gen_feat.columns
                    if "GENETIC_FEATURE" in c or "GENE" in c), None)
ml_cells    = set(cell_map["GDSC_CELL_NAME"].unique())
gen_feat_ml = gen_feat[gen_feat["CELL_LINE_NAME"].isin(ml_cells)]

if gene_col and "IS_MUTATED" in gen_feat.columns:
    mut_df   = gen_feat_ml[gen_feat_ml["IS_MUTATED"].notna()].copy()
    gene_var = mut_df.groupby(gene_col)["IS_MUTATED"]\
        .var().sort_values(ascending=False)
    top_mut  = gene_var.head(MAX_MUT_FEATURES).index.tolist()
    mut_pivot = mut_df[mut_df[gene_col].isin(top_mut)].pivot_table(
        index="CELL_LINE_NAME", columns=gene_col,
        values="IS_MUTATED", aggfunc="mean"
    ).reset_index()
    mut_pivot.columns = ["CELL_LINE_NAME"] + \
        [f"MUT_{c}" for c in mut_pivot.columns[1:]]
    print(f"Mutation pivot: {mut_pivot.shape}")
else:
    mut_pivot = pd.DataFrame(columns=["CELL_LINE_NAME"])

# CNV — FIX: encode gain/loss to numeric
cnv_cell_col = next((c for c in cnv.columns if "CELL" in c), None)
cnv_feat_col = next((c for c in cnv.columns
                     if "GENE" in c or "FEATURE" in c), None)
cnv_val_col  = next((c for c in cnv.columns
                     if "GAIN" in c or "LOSS" in c
                     or "RECURRENT" in c), None)
if cnv_cell_col and cnv_feat_col and cnv_val_col:
    cnv[cnv_cell_col] = cnv[cnv_cell_col].apply(std)
    cnv_ml = cnv[cnv[cnv_cell_col].isin(ml_cells)].copy()
    cnv_ml[cnv_val_col] = cnv_ml[cnv_val_col]\
        .astype(str).str.lower()
    cnv_ml[cnv_val_col] = cnv_ml[cnv_val_col].map({
        "gain":1,"loss":-1,"neutral":0,
        "recurrent gain":1,"recurrent loss":-1,"nan":np.nan
    })
    cnv_ml[cnv_val_col] = pd.to_numeric(
        cnv_ml[cnv_val_col], errors="coerce")
    if len(cnv_ml) > 0:
        cnv_var = cnv_ml.groupby(cnv_feat_col)[cnv_val_col]\
            .var().sort_values(ascending=False)
        top_cnv = cnv_var.head(MAX_CNV_FEATURES).index.tolist()
        cnv_pivot = cnv_ml[cnv_ml[cnv_feat_col].isin(top_cnv)]\
            .pivot_table(
                index=cnv_cell_col, columns=cnv_feat_col,
                values=cnv_val_col, aggfunc="mean"
            ).reset_index()
        cnv_pivot.columns = ["CELL_LINE_NAME"] + \
            [f"CNV_{c}" for c in cnv_pivot.columns[1:]]
        print(f"CNV pivot     : {cnv_pivot.shape}")
    else:
        cnv_pivot = pd.DataFrame(columns=["CELL_LINE_NAME"])
else:
    cnv_pivot = pd.DataFrame(columns=["CELL_LINE_NAME"])

# Tissue one-hot
tissue.columns = tissue.columns.str.upper().str.strip()
tissue_cell_col = next((c for c in tissue.columns if "CELL" in c), None)
tissue_type_col = next((c for c in tissue.columns
                        if "TISSUE" in c), None)
if tissue_cell_col and tissue_type_col:
    tissue[tissue_cell_col] = tissue[tissue_cell_col].apply(std)
    tissue_dummies = pd.get_dummies(
        tissue[[tissue_cell_col, tissue_type_col]],
        columns=[tissue_type_col], prefix="TISSUE"
    ).rename(columns={tissue_cell_col:"CELL_LINE_NAME"})\
     .drop_duplicates(subset=["CELL_LINE_NAME"])
    print(f"Tissue one-hot: {tissue_dummies.shape}")
else:
    tissue_dummies = pd.DataFrame(columns=["CELL_LINE_NAME"])

# MSI + Growth
cell_lines.columns = cell_lines.columns.str.upper().str.strip()
cell_name_col2 = next((c for c in cell_lines.columns
                       if "SAMPLE" in c or
                       ("CELL" in c and "NAME" in c)), None)
msi_col    = next((c for c in cell_lines.columns
                   if "MSI" in c or "MICROSATELLITE" in c), None)
growth_col = next((c for c in cell_lines.columns
                   if "GROWTH" in c), None)
cell_extra_cols = ["CELL_LINE_NAME"]
if cell_name_col2:
    cell_lines[cell_name_col2] = cell_lines[cell_name_col2].apply(std)
    cell_extra = cell_lines.rename(
        columns={cell_name_col2:"CELL_LINE_NAME"}).copy()
    if msi_col:
        cell_extra["MSI_STATUS"] = (
            cell_extra[msi_col].astype(str).str.upper()=="MSI-H"
        ).astype(int)
        cell_extra_cols.append("MSI_STATUS")
    if growth_col:
        g_dummies = pd.get_dummies(
            cell_extra[["CELL_LINE_NAME",growth_col]],
            columns=[growth_col], prefix="GROWTH")
        cell_extra = cell_extra.merge(
            g_dummies, on="CELL_LINE_NAME", how="left")
        cell_extra_cols += [c for c in g_dummies.columns
                            if c!="CELL_LINE_NAME"]
    cell_extra_table = cell_extra[cell_extra_cols]\
        .drop_duplicates(subset=["CELL_LINE_NAME"])
    print(f"Cell extra    : {cell_extra_table.shape}")
else:
    cell_extra_table = pd.DataFrame(columns=["CELL_LINE_NAME"])

# FIX: LINCS — strip cell names for better matching
lincs["DRUG_NAME"]     = lincs["DRUG_NAME"].apply(std) \
    if "DRUG_NAME" in lincs.columns else lincs.iloc[:,0].apply(std)
lincs["CELL_ID"]       = lincs["CELL_ID"].apply(std) \
    if "CELL_ID" in lincs.columns else lincs.iloc[:,1].apply(std)
lincs["CELL_STRIPPED"] = lincs["CELL_ID"].apply(strip_cell)
lincs_meta     = ["DRUG_NAME","CELL_ID","CELL_STRIPPED","COSMIC_ID",
                  "CHEMBL_ID","GDSC_DRUG_ID"]
lincs_feat_cols= [c for c in lincs.columns if c not in lincs_meta]
lincs_table    = lincs[["DRUG_NAME","CELL_STRIPPED"]+lincs_feat_cols]
print(f"LINCS table   : {lincs_table.shape}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — Build Feature Matrix
# ══════════════════════════════════════════════════════════════════════════
section("STEP 5: Building Feature Matrix")

df = drugcomb.rename(columns={
    dc_drug1_col:"DRUG1", dc_drug2_col:"DRUG2",
    dc_cell_col:"CELL_LINE"
}).copy()
if dc_label_col: df = df.rename(columns={dc_label_col:"SYNERGY_LABEL"})
if dc_loewe_col: df = df.rename(columns={dc_loewe_col:"LOEWE_SCORE"})

df["COSMIC_ID"]          = df["CELL_LINE"].map(cell_to_cosmic)
df["CELL_LINE_STRIPPED"] = df["CELL_LINE"].apply(strip_cell)

print(f"Base rows      : {len(df)}")
print(f"With COSMIC_ID : {df['COSMIC_ID'].notna().sum()}")



# Drug1 features
fp_d1 = drug_features.copy()
fp_d1.columns = ["DRUG1"]+[f"D1_{c}" for c in fp_d1.columns[1:]]
df = df.merge(fp_d1, on="DRUG1", how="left")
print(f"After Drug1 join  : {df.shape}")
# Keep ZIP, BLISS, HSA as features
score_feature_cols = []
for col_var, col_name in [(dc_zip_col,"ZIP_SCORE"),
                           (dc_bliss_col,"BLISS_SCORE"),
                           (dc_hsa_col,"HSA_SCORE")]:
    if col_var and col_var in df.columns:
        df = df.rename(columns={col_var:col_name})
        score_feature_cols.append(col_name)
print(f"Score features added: {score_feature_cols}")
# Drug2 features
fp_d2 = drug_features.copy()
fp_d2.columns = ["DRUG2"]+[f"D2_{c}" for c in fp_d2.columns[1:]]
df = df.merge(fp_d2, on="DRUG2", how="left")
print(f"After Drug2 join  : {df.shape}")

# LN_IC50
ic50_d1 = ic50_pivot.copy()
ic50_d1.columns = ["DRUG1","CELL_LINE","D1_LN_IC50"]
df = df.merge(ic50_d1, on=["DRUG1","CELL_LINE"], how="left")
ic50_d2 = ic50_pivot.copy()
ic50_d2.columns = ["DRUG2","CELL_LINE","D2_LN_IC50"]
df = df.merge(ic50_d2, on=["DRUG2","CELL_LINE"], how="left")

# AUC
if len(auc_pivot) > 0:
    auc_d1 = auc_pivot.copy()
    auc_d1.columns = ["DRUG1","CELL_LINE","D1_AUC"]
    df = df.merge(auc_d1, on=["DRUG1","CELL_LINE"], how="left")
    auc_d2 = auc_pivot.copy()
    auc_d2.columns = ["DRUG2","CELL_LINE","D2_AUC"]
    df = df.merge(auc_d2, on=["DRUG2","CELL_LINE"], how="left")

# FIX: Drug pair interaction features
df["PAIR_LN_IC50_SUM"]   = df["D1_LN_IC50"] + df["D2_LN_IC50"]
# Additional pair features
df["PAIR_AUC_SUM"]          = df["D1_AUC"] + df["D2_AUC"]
df["PAIR_AUC_RATIO"]        = df["D1_AUC"] / df["D2_AUC"].replace(0, np.nan)
df["PAIR_TARGET_OVERLAP"]   = df["D1_N_TARGETS"] + df["D2_N_TARGETS"]
df["PAIR_LN_IC50_RATIO"] = df["D1_LN_IC50"] / \
    df["D2_LN_IC50"].replace(0, np.nan)
df["PAIR_LN_IC50_DIFF"]  = df["D1_LN_IC50"] - df["D2_LN_IC50"]

# Tanimoto similarity
fp_d1_vals = df[[f"D1_{c}" for c in fp_cols]].fillna(0).values
fp_d2_vals = df[[f"D2_{c}" for c in fp_cols]].fillna(0).values
intersect  = (fp_d1_vals * fp_d2_vals).sum(axis=1)
union_     = ((fp_d1_vals + fp_d2_vals) > 0).sum(axis=1)
df["PAIR_TANIMOTO_SIM"] = np.where(
    union_ > 0, intersect / union_, 0)
print(f"Pair features added — Tanimoto mean: "
      f"{df['PAIR_TANIMOTO_SIM'].mean():.3f} ✅")
print(f"After pair features: {df.shape}")

# Mutations
if len(mut_pivot.columns) > 1:
    df = df.merge(mut_pivot, left_on="CELL_LINE",
                  right_on="CELL_LINE_NAME", how="left")
    df.drop(columns=["CELL_LINE_NAME"], errors="ignore", inplace=True)
    print(f"After mutation join: {df.shape}")

# CNV
if len(cnv_pivot.columns) > 1:
    df = df.merge(cnv_pivot, left_on="CELL_LINE",
                  right_on="CELL_LINE_NAME", how="left")
    df.drop(columns=["CELL_LINE_NAME"], errors="ignore", inplace=True)
    print(f"After CNV join     : {df.shape}")

# Tissue
if len(tissue_dummies.columns) > 1:
    df = df.merge(tissue_dummies, left_on="CELL_LINE",
                  right_on="CELL_LINE_NAME", how="left")
    df.drop(columns=["CELL_LINE_NAME"], errors="ignore", inplace=True)
    print(f"After tissue join  : {df.shape}")

# Cell extra
if len(cell_extra_table.columns) > 1:
    df = df.merge(cell_extra_table, left_on="CELL_LINE",
                  right_on="CELL_LINE_NAME", how="left")
    df.drop(columns=["CELL_LINE_NAME"], errors="ignore", inplace=True)
    print(f"After cell join    : {df.shape}")

# FIX: LINCS D1 — join using stripped cell name
lincs_d1 = lincs_table.copy()
lincs_d1.columns = (["DRUG1","CELL_LINE_STRIPPED"] +
                    [f"D1_LINCS_{c}" for c in lincs_feat_cols])
df = df.merge(lincs_d1, on=["DRUG1","CELL_LINE_STRIPPED"], how="left")
d1_match = df[[c for c in df.columns if "D1_LINCS" in c][0]]\
    .notna().sum() if any("D1_LINCS" in c for c in df.columns) else 0
print(f"After LINCS D1     : {df.shape} ({d1_match} rows matched)")

# FIX: LINCS D2
lincs_d2 = lincs_table.copy()
lincs_d2.columns = (["DRUG2","CELL_LINE_STRIPPED"] +
                    [f"D2_LINCS_{c}" for c in lincs_feat_cols])
df = df.merge(lincs_d2, on=["DRUG2","CELL_LINE_STRIPPED"], how="left")
d2_match = df[[c for c in df.columns if "D2_LINCS" in c][0]]\
    .notna().sum() if any("D2_LINCS" in c for c in df.columns) else 0
print(f"After LINCS D2     : {df.shape} ({d2_match} rows matched)")

df.drop(columns=["CELL_LINE_STRIPPED"],
        errors="ignore", inplace=True)

# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — Handle Missing Values
#           FIX: Separate threshold LINCS(80%) vs others(50%)
# ══════════════════════════════════════════════════════════════════════════
section("STEP 6: Handling Missing Values")

id_cols     = ["DRUG1","DRUG2","CELL_LINE","COSMIC_ID"]
target_cols = ["SYNERGY_LABEL"]
if "LOEWE_SCORE" in df.columns:
    target_cols.append("LOEWE_SCORE")
meta_cols    = id_cols + target_cols
feature_cols = [c for c in df.columns if c not in meta_cols]

print(f"Total columns   : {df.shape[1]}")
print(f"Feature columns : {len(feature_cols)}")

missing_pct     = df[feature_cols].isnull().mean()
lincs_feats_all = [c for c in feature_cols if "LINCS" in c]
other_feats_all = [c for c in feature_cols if "LINCS" not in c]

drop_other = missing_pct[
    (missing_pct > 0.5) &
    (missing_pct.index.isin(other_feats_all))
].index.tolist()
# Don't drop LINCS — fill with 0 instead (no expression = 0)
drop_lincs = []
cols_to_drop = drop_other + drop_lincs

print(f"Non-LINCS >50% : {len(drop_other)} dropped")
print(f"LINCS >80%     : {len(drop_lincs)} dropped")
print(f"Total dropped  : {len(cols_to_drop)}")

if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    feature_cols = [c for c in feature_cols
                    if c not in cols_to_drop]

lincs_kept = [c for c in feature_cols if "LINCS" in c]
print(f"LINCS retained : {len(lincs_kept)}")

# Impute
num_cols = [c for c in feature_cols
            if df[c].dtype in [np.float64, np.int64,
                                np.float32, np.int32]]
cat_cols = [c for c in feature_cols if c not in num_cols]
# LINCS columns — fill missing with 0 (no expression signal)
lincs_num_cols  = [c for c in feature_cols if "LINCS" in c]
other_num_cols  = [c for c in feature_cols
                   if "LINCS" not in c and
                   df[c].dtype in [np.float64, np.int64,
                                   np.float32, np.int32]]
cat_cols = [c for c in feature_cols
            if c not in lincs_num_cols + other_num_cols]

if other_num_cols:
    imputer = SimpleImputer(strategy="median")
    df[other_num_cols] = imputer.fit_transform(df[other_num_cols])
if lincs_num_cols:
    df[lincs_num_cols] = df[lincs_num_cols].fillna(0)
if cat_cols:
    df[cat_cols] = df[cat_cols].fillna(0)
print(f"LINCS missing filled with 0 ")

df.dropna(subset=["SYNERGY_LABEL"], inplace=True)
print(f"Final missing  : {df[feature_cols].isnull().sum().sum()}")
print(f"Final rows     : {len(df)}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 7 — Feature Groups Summary
# ══════════════════════════════════════════════════════════════════════════
section("STEP 7: Feature Groups Summary")

fp_feat    = [c for c in feature_cols if "FP_" in c]
prop_feat  = [c for c in feature_cols
              if any(x in c for x in
                     ["MOLECULAR_WEIGHT","ALOGP","HBD","HBA",
                      "PSA","NUM_RINGS","SCAFFOLD","N_TARGET"])]
ic50_feat  = [c for c in feature_cols
              if "LN_IC50" in c or
              ("AUC" in c and "LINCS" not in c)]
pair_feat  = [c for c in feature_cols
              if "PAIR_" in c or "TANIMOTO" in c]
score_feat = [c for c in feature_cols
              if any(x in c for x in
                     ["ZIP_SCORE","BLISS_SCORE","HSA_SCORE"])]
mut_feat   = [c for c in feature_cols if "MUT_" in c]
cnv_feat   = [c for c in feature_cols if "CNV_" in c]
tis_feat   = [c for c in feature_cols if "TISSUE_" in c]
cell_feat  = [c for c in feature_cols
              if "MSI" in c or "GROWTH" in c]
lincs_feat = [c for c in feature_cols if "LINCS" in c]

print(f"Feature Group Breakdown:")
print(f"  Morgan FP (Drug1+2)     : {len(fp_feat)}")
print(f"  Drug properties         : {len(prop_feat)}")
print(f"  IC50/AUC (Drug1+2)      : {len(ic50_feat)}")
print(f"  Drug pair interactions  : {len(pair_feat)}")
print(f"  Synergy score features  : {len(score_feat)}")
print(f"  Mutation features       : {len(mut_feat)}")
print(f"  CNV features            : {len(cnv_feat)}")
print(f"  Tissue type             : {len(tis_feat)}")
print(f"  Cell line (MSI/Growth)  : {len(cell_feat)}")
print(f"  LINCS gene expression   : {len(lincs_feat)}")
print(f"  ─────────────────────────────────")
print(f"  TOTAL FEATURES          : {len(feature_cols)}")
print(f"  TOTAL ROWS              : {len(df)}")
print(f"  SYNERGISTIC (1)         : "
      f"{(df['SYNERGY_LABEL']==1).sum()} "
      f"({(df['SYNERGY_LABEL']==1).mean()*100:.1f}%)")
print(f"  NON-SYNERGISTIC (0)     : "
      f"{(df['SYNERGY_LABEL']==0).sum()} "
      f"({(df['SYNERGY_LABEL']==0).mean()*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════
# STEP 8 — Train/Test Split + Feature Scaling
# ══════════════════════════════════════════════════════════════════════════
section("STEP 8: Train/Test Split + Feature Scaling")

# Remove any non-numeric columns that slipped through
non_numeric = [c for c in feature_cols if df[c].dtype == object]
if non_numeric:
    print(f"Dropping non-numeric: {non_numeric}")
    df.drop(columns=non_numeric, inplace=True)
    feature_cols = [c for c in feature_cols
                    if c not in non_numeric]
X = df[feature_cols].values
y = df["SYNERGY_LABEL"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE,
    random_state=RANDOM_STATE, stratify=y)

print(f"Train set : {X_train.shape} | "
      f"Synergistic: {y_train.sum()} "
      f"({y_train.mean()*100:.1f}%)")
print(f"Test set  : {X_test.shape}  | "
      f"Synergistic: {y_test.sum()} "
      f"({y_test.mean()*100:.1f}%)")

# Remove any non-numeric columns that slipped through
non_numeric = [c for c in feature_cols
               if df[c].dtype == object]
if non_numeric:
    print(f"Dropping non-numeric columns: {non_numeric}")
    df.drop(columns=non_numeric, inplace=True)
    feature_cols = [c for c in feature_cols
                    if c not in non_numeric]

X = df[feature_cols].values
y = df["SYNERGY_LABEL"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE,
    random_state=RANDOM_STATE, stratify=y)

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print(f"Scaling done ")

# ══════════════════════════════════════════════════════════════════════════
# STEP 9 — SMOTE
# ══════════════════════════════════════════════════════════════════════════
section("STEP 9: SMOTE — Class Imbalance Handling")

print(f"Before SMOTE — synergistic: {y_train.sum()} "
      f"({y_train.mean()*100:.1f}%)")
try:
    smote = SMOTE(random_state=RANDOM_STATE,
                  k_neighbors=min(5, int(y_train.sum())-1))
    X_train_smote, y_train_smote = smote.fit_resample(
        X_train_scaled, y_train)
    print(f"After SMOTE  — rows: {len(X_train_smote)} | "
          f"synergistic: {y_train_smote.sum()} "
          f"({y_train_smote.mean()*100:.1f}%)")
    smote_ok = True
except Exception as e:
    print(f"SMOTE failed: {e} — using original")
    X_train_smote = X_train_scaled
    y_train_smote = y_train
    smote_ok      = False

# ══════════════════════════════════════════════════════════════════════════
# STEP 10 — Save All Outputs
# ══════════════════════════════════════════════════════════════════════════
section("STEP 10: Saving All Outputs")

# Full matrix
df.to_csv(os.path.join(PROJECT_DIR,
                       "feature_matrix_full.csv"), index=False)
try:
    df.to_parquet(os.path.join(PROJECT_DIR,
                               "feature_matrix_full.parquet"),
                  index=False)
except Exception:
    pass
print("feature_matrix_full.csv ")

# Unscaled train/test (for tree models)
train_df = pd.DataFrame(X_train, columns=feature_cols)
train_df["SYNERGY_LABEL"] = y_train
test_df  = pd.DataFrame(X_test,  columns=feature_cols)
test_df["SYNERGY_LABEL"]  = y_test
train_df.to_csv(os.path.join(PROJECT_DIR,"train_set.csv"),
                index=False)
test_df.to_csv(os.path.join(PROJECT_DIR,"test_set.csv"),
               index=False)
print("train_set.csv + test_set.csv ")

# Scaled train/test (for DNN)
pd.DataFrame(X_train_scaled, columns=feature_cols)\
    .assign(SYNERGY_LABEL=y_train)\
    .to_csv(os.path.join(PROJECT_DIR,"train_set_scaled.csv"),
            index=False)
pd.DataFrame(X_test_scaled, columns=feature_cols)\
    .assign(SYNERGY_LABEL=y_test)\
    .to_csv(os.path.join(PROJECT_DIR,"test_set_scaled.csv"),
            index=False)
print("train_set_scaled.csv + test_set_scaled.csv ")

# SMOTE train set
pd.DataFrame(X_train_smote, columns=feature_cols)\
    .assign(SYNERGY_LABEL=y_train_smote)\
    .to_csv(os.path.join(PROJECT_DIR,"train_set_smote.csv"),
            index=False)
print("train_set_smote.csv ")

# Feature names
pd.DataFrame({
    "FEATURE_NAME" : feature_cols,
    "FEATURE_GROUP": [
        "FP"        if "FP_" in c else
        "DRUG_PROP" if any(x in c for x in
                           ["MOLECULAR","ALOGP","HBD","HBA",
                            "PSA","RING","SCAFFOLD","TARGET"]) else
        "IC50_AUC"  if any(x in c for x in
                           ["LN_IC50","AUC"]) else
        "PAIR"      if "PAIR_" in c or "TANIMOTO" in c else
        "SCORE"     if any(x in c for x in
                           ["ZIP","BLISS","HSA"]) else
        "MUTATION"  if "MUT_" in c else
        "CNV"       if "CNV_" in c else
        "TISSUE"    if "TISSUE_" in c else
        "CELL"      if any(x in c for x in
                           ["MSI","GROWTH"]) else
        "LINCS"     if "LINCS" in c else "OTHER"
        for c in feature_cols
    ]
}).to_csv(os.path.join(PROJECT_DIR,"feature_names.csv"),
          index=False)
print("feature_names.csv ")

# ══════════════════════════════════════════════════════════════════════════
# STEP 11 — Visualisations
# ══════════════════════════════════════════════════════════════════════════
section("STEP 11: Visualisations")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Feature Matrix Overview",
             fontsize=16, fontweight="bold")

# Class balance before/after SMOTE
if smote_ok:
    cats = ["Train\n(original)","Train\n(SMOTE)","Test"]
    syns = [y_train.sum(), y_train_smote.sum(), y_test.sum()]
    nons = [(y_train==0).sum(),
            (y_train_smote==0).sum(), (y_test==0).sum()]
    x = np.arange(3); w = 0.35
    axes[0,0].bar(x-w/2, nons, w,
                  label="Non-Syn", color="#66b3ff")
    axes[0,0].bar(x+w/2, syns, w,
                  label="Syn", color="#ff9999")
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(cats)
    axes[0,0].set_title("Class Balance Before/After SMOTE")
    axes[0,0].legend()
else:
    counts = pd.Series(y).value_counts()
    axes[0,0].bar(["Non-Syn","Syn"],
                  [counts.get(0,0), counts.get(1,0)],
                  color=["#66b3ff","#ff9999"])
    axes[0,0].set_title("Class Balance")

# Feature groups
grp = pd.DataFrame({
    "GROUP": ["FP","DRUG_PROP","IC50_AUC","PAIR","SCORE",
              "MUTATION","CNV","TISSUE","CELL","LINCS"],
    "COUNT": [len(fp_feat),len(prop_feat),len(ic50_feat),
              len(pair_feat),len(score_feat),len(mut_feat),
              len(cnv_feat),len(tis_feat),len(cell_feat),
              len(lincs_feat)]
})
axes[0,1].bar(grp["GROUP"], grp["COUNT"], color="steelblue")
axes[0,1].set_title("Features by Group")
axes[0,1].tick_params(axis="x", rotation=45)
for i, v in enumerate(grp["COUNT"]):
    axes[0,1].text(i, v+1, str(v), ha="center",
                   fontsize=8, fontweight="bold")

# Tanimoto similarity
if "PAIR_TANIMOTO_SIM" in df.columns:
    df["PAIR_TANIMOTO_SIM"].hist(
        bins=50, ax=axes[1,0],
        color="mediumseagreen", edgecolor="white")
    axes[1,0].set_title("Drug Pair Tanimoto Similarity")
    axes[1,0].set_xlabel("Tanimoto")

# Loewe by class
if "LOEWE_SCORE" in df.columns:
    syn = df[df["SYNERGY_LABEL"]==1]["LOEWE_SCORE"].dropna()
    non = df[df["SYNERGY_LABEL"]==0]["LOEWE_SCORE"].dropna()
    axes[1,1].hist(non.clip(-50,50), bins=50,
                   alpha=0.6, label="Non-Syn",
                   color="#66b3ff")
    axes[1,1].hist(syn.clip(-50,50), bins=50,
                   alpha=0.6, label="Syn",
                   color="#ff9999")
    axes[1,1].set_title("Loewe Score by Class")
    axes[1,1].legend()

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR,
                         "feature_matrix_overview.png"), dpi=150)
plt.show()
print("Plot saved ")

# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
section("FINAL SUMMARY")

print(f"  Feature matrix shape     : {df.shape}")
print(f"  Total features           : {len(feature_cols)}")
print(f"  Total samples            : {len(df)}")
print(f"  Train / Test             : {len(X_train)} / {len(X_test)}")
print(f"  SMOTE train samples      : {len(X_train_smote)}")
print(f"\n  Feature breakdown:")
print(f"    Morgan FP              : {len(fp_feat)}")
print(f"    Drug properties        : {len(prop_feat)}")
print(f"    IC50/AUC               : {len(ic50_feat)}")
print(f"    Drug pair interactions : {len(pair_feat)}")
print(f"    Synergy score features : {len(score_feat)}")
print(f"    Mutations              : {len(mut_feat)}")
print(f"    CNV                    : {len(cnv_feat)}")
print(f"    Tissue                 : {len(tis_feat)}")
print(f"    Cell (MSI/Growth)      : {len(cell_feat)}")
print(f"    LINCS expression       : {len(lincs_feat)}")

print("\n  --- File Checklist ---")
for f in ["feature_matrix_full.csv",
          "feature_matrix_full.parquet",
          "train_set.csv","test_set.csv",
          "train_set_scaled.csv","test_set_scaled.csv",
          "train_set_smote.csv",
          "feature_names.csv",
          "feature_matrix_overview.png"]:
    path   = os.path.join(PROJECT_DIR, f)
    status = "[OK]" if os.path.exists(path) else "[MISSING]"
    print(f"    {status}  {f}")

print("\n STEP 5 — Feature Matrix COMPLETE!")
print(" Next -> Step 6: Baseline ML Models\n")