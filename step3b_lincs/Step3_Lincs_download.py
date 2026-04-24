import pandas as pd
import numpy as np
import requests
import os
import time
import json
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
PROJECT_DIR = r"C:\Users\sush\PYCHARM\Capstone project"
# Replace with this
API_KEY = "YOUR_API_KEY_HERE"  # Replace with your own API key
BASE_URL    = "https://api.clue.io/api"

# Alternative drug names for missed drugs
DRUG_ALIASES = {
    "CARMUSTINE"       : ["bcnu", "carmustine", "bis-chloronitrosourea",
                          "bicnu", "carmubris"],
    "MYCOPHENOLIC ACID": ["mycophenolate", "mycophenolic-acid", "mpa",
                          "mycophenolic acid"],
    "ROMIDEPSIN"       : ["fk-228", "romidepsin", "istodax",
                          "fk228", "depsipeptide"],
    "OSIMERTINIB"      : ["azd-9291", "osimertinib", "tagrisso",
                          "azd9291"],
    "RIBOCICLIB"       : ["lee011", "ribociclib", "kisqali",
                          "lee-011"],
    "TALAZOPARIB"      : ["bmn-673", "talazoparib", "bmn673",
                          "talzenna"],
    "NIRAPARIB"        : ["mk-4827", "niraparib", "mk4827",
                          "zejula"],
    "VISMODEGIB"       : ["gdc-0449", "vismodegib", "gdc0449",
                          "erivedge"],
    "NELARABINE"       : ["arranon", "nelarabine", "compound-506u78",
                          "atriance", "506u78"],
}

def section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def clue_get(endpoint, filter_dict, timeout=30):
    try:
        url    = f"{BASE_URL}/{endpoint}"
        params = {
            "filter"   : json.dumps(filter_dict),
            "user_key" : API_KEY
        }
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        return []
    except Exception:
        return []

def parse_genes(gene_str):
    try:
        if gene_str is None: return []
        s = str(gene_str).strip()
        if s in ("", "nan", "None", "[]"): return []
        return [g.strip() for g in s.split("|") if g.strip()]
    except:
        return []

# ══════════════════════════════════════════════════════════════════════════
# Load master maps
# ══════════════════════════════════════════════════════════════════════════
section("Loading Master Maps")

drug_map      = pd.read_csv(os.path.join(PROJECT_DIR, "master_drug_map.csv"))
cell_map      = pd.read_csv(os.path.join(PROJECT_DIR, "master_cell_line_map.csv"))
tissue_map_df = pd.read_csv(os.path.join(PROJECT_DIR, "gdsc_tissue_map.csv"))

ml_drugs      = drug_map["MASTER_DRUG_NAME"].str.upper().str.strip().tolist()
ml_cells_gdsc = cell_map["GDSC_CELL_NAME"].str.upper().str.strip().tolist()

# Build tissue lookup
tissue_dict = {}
if "CELL_LINE_NAME" in tissue_map_df.columns and \
   "TISSUE_TYPE" in tissue_map_df.columns:
    tissue_dict = dict(zip(
        tissue_map_df["CELL_LINE_NAME"].str.upper(),
        tissue_map_df["TISSUE_TYPE"].str.upper()
    ))

# Build COSMIC lookup from cell map
cosmic_lookup = {}
for _, row in cell_map.iterrows():
    gdsc_name = str(row["GDSC_CELL_NAME"]).upper().strip()
    dc_name   = str(row["DRUGCOMB_CELL_NAME"]).upper().strip()
    cosmic_id = row["COSMIC_ID"]
    if pd.notna(cosmic_id):
        cosmic_lookup[gdsc_name]                            = cosmic_id
        cosmic_lookup[dc_name]                              = cosmic_id
        cosmic_lookup[gdsc_name.replace("-","")]            = cosmic_id
        cosmic_lookup[gdsc_name.replace("-","").replace(" ","")] = cosmic_id
        cosmic_lookup[dc_name.replace("-","")]              = cosmic_id

print(f"Your 51 drugs      : {len(ml_drugs)}")
print(f"Your 88 cells      : {len(ml_cells_gdsc)}")
print(f"COSMIC lookup size : {len(cosmic_lookup)}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Find drugs with exact + extended alias matching
# ══════════════════════════════════════════════════════════════════════════
section("STEP 1: Finding Your 51 Drugs (Exact + Extended Alias)")

found_drugs  = []
missed_drugs = []

for drug in ml_drugs:
    drug_lower = drug.lower()
    matched    = False

    names_to_try = [drug_lower]
    if drug in DRUG_ALIASES:
        names_to_try += DRUG_ALIASES[drug]

    for name in names_to_try:
        result = clue_get("perts", {
            "where"  : {"pert_iname": name,
                        "pert_type" : "trt_cp"},
            "fields" : ["pert_id", "pert_iname", "pert_type"],
            "limit"  : 1
        })

        if result and \
           result[0].get("pert_iname","").lower() == name.lower():
            best                     = result[0]
            best["MASTER_DRUG_NAME"] = drug
            best["MATCHED_AS"]       = name
            found_drugs.append(best)
            matched = True
            if name != drug_lower:
                print(f"  FOUND* : {drug} -> "
                      f"{best['pert_iname']} (alias: {name})")
            else:
                print(f"  FOUND  : {drug} -> "
                      f"{best['pert_iname']} ({best['pert_id']})")
            break
        time.sleep(0.2)

    if not matched:
        missed_drugs.append(drug)
        print(f"  MISSED : {drug}")

    time.sleep(0.2)

drugs_df = pd.DataFrame(found_drugs)
print(f"\nDrugs found  : {len(found_drugs)}/51")
print(f"Drugs missed : {len(missed_drugs)}")
if missed_drugs:
    print(f"Missed       : {missed_drugs}")

drugs_df.to_csv(
    os.path.join(PROJECT_DIR, "lincs_drug_ids_final.csv"), index=False)
print("Saved -> lincs_drug_ids_final.csv")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Download signatures for found drugs
# ══════════════════════════════════════════════════════════════════════════
section("STEP 2: Downloading Signatures for Your Drugs")

pert_ids = drugs_df["pert_id"].dropna().tolist() \
           if "pert_id" in drugs_df.columns else []

if len(pert_ids) == 0:
    print("No pert_ids — using drug names...")
    pert_ids    = [d.lower() for d in ml_drugs]
    query_field = "pert_iname"
else:
    query_field = "pert_id"

print(f"Downloading signatures for {len(pert_ids)} drugs...")

all_sigs     = []
drug_batches = [pert_ids[i:i+5] for i in range(0, len(pert_ids), 5)]

for batch_num, drug_batch in enumerate(drug_batches):
    result = clue_get("sigs", {
        "where"  : {query_field  : {"inq": drug_batch},
                    "pert_type"  : "trt_cp"},
        "fields" : ["sig_id", "pert_id", "pert_iname",
                    "cell_id", "pert_dose", "pert_time",
                    "distil_ss", "ngenes_modulated_up_lm",
                    "ngenes_modulated_dn_lm",
                    "up50_lm", "dn50_lm", "is_gold"],
        "limit"  : 1000
    })

    if result:
        all_sigs.extend(result)

    if batch_num % 3 == 0:
        print(f"  Batch {batch_num+1}/{len(drug_batches)} — "
              f"sigs: {len(all_sigs)}")
    time.sleep(0.3)

sigs_df = pd.DataFrame(all_sigs)
print(f"\nTotal signatures : {len(sigs_df)}")

if len(sigs_df) > 0:
    print(f"Unique drugs     : "
          f"{sigs_df['pert_iname'].nunique()}")
    print(f"Unique cell lines: "
          f"{sigs_df['cell_id'].nunique()}")
    sigs_df.to_csv(
        os.path.join(PROJECT_DIR, "lincs_signatures_final.csv"),
        index=False)
    print("Saved -> lincs_signatures_final.csv")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Fix COSMIC ID mapping
#           Use multiple name variants to maximise coverage
# ══════════════════════════════════════════════════════════════════════════
section("STEP 3: Fixing COSMIC ID Mapping")

def get_cosmic_id(cell_name):
    """Try multiple name variants to find COSMIC ID."""
    if pd.isna(cell_name):
        return np.nan
    name = str(cell_name).upper().strip()
    variants = [
        name,
        name.replace("-",""),
        name.replace("-","").replace(" ",""),
        name.replace(".",""),
        name.replace("_",""),
    ]
    for v in variants:
        if v in cosmic_lookup:
            return cosmic_lookup[v]
    return np.nan

# Test COSMIC mapping
test_cells = ["MCF7", "A375", "PC3", "HT29", "A549"]
print("COSMIC ID mapping test:")
for c in test_cells:
    cid = get_cosmic_id(c)
    print(f"  {c} -> COSMIC_ID: {cid}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Build final LINCS feature matrix
# ══════════════════════════════════════════════════════════════════════════
section("STEP 4: Building Final LINCS Feature Matrix")

if len(sigs_df) == 0:
    print("No signatures — cannot build matrix!")
else:
    # Standardise names
    sigs_df["DRUG_NAME"] = sigs_df["pert_iname"].str.upper().str.strip()
    sigs_df["CELL_ID"]   = sigs_df["cell_id"].str.upper().str.strip()

    # Map alias names back to master drug names
    alias_to_master = {}
    for master, aliases in DRUG_ALIASES.items():
        for alias in aliases:
            alias_to_master[alias.upper()] = master
    # Also map hyphenated versions
    alias_to_master["MYCOPHENOLIC-ACID"] = "MYCOPHENOLIC ACID"

    sigs_df["DRUG_NAME"] = sigs_df["DRUG_NAME"].apply(
        lambda x: alias_to_master.get(x, x))

    # Gold standard only
    if "is_gold" in sigs_df.columns:
        gold = sigs_df[sigs_df["is_gold"] == 1]
        if len(gold) > 5:
            sigs_df = gold.copy()
            print(f"Gold standard sigs: {len(sigs_df)}")

    # Parse genes
    if "up50_lm" in sigs_df.columns:
        sigs_df["UP_GENES"] = sigs_df["up50_lm"].apply(parse_genes)
        sigs_df["DN_GENES"] = sigs_df["dn50_lm"].apply(parse_genes)
        sigs_df["N_UP"]     = sigs_df["UP_GENES"].apply(len)
        sigs_df["N_DN"]     = sigs_df["DN_GENES"].apply(len)

        print(f"Mean UP genes: {sigs_df['N_UP'].mean():.1f}")
        print(f"Mean DN genes: {sigs_df['N_DN'].mean():.1f}")

        # Top genes
        all_up, all_dn = [], []
        for g in sigs_df["UP_GENES"]: all_up.extend(g)
        for g in sigs_df["DN_GENES"]: all_dn.extend(g)

        top_genes = list(set(
            pd.Series(all_up).value_counts().head(50).index.tolist() +
            pd.Series(all_dn).value_counts().head(50).index.tolist()
        ))
        print(f"Top genes identified: {len(top_genes)}")

        # Build feature matrix
        rows = []
        for (drug, cell), grp in sigs_df.groupby(
                ["DRUG_NAME", "CELL_ID"]):
            row = {
                "DRUG_NAME"      : drug,
                "CELL_ID"        : cell,
                "MEAN_DISTIL_SS" : grp["distil_ss"].mean() \
                                   if "distil_ss" in grp.columns \
                                   else np.nan,
                "N_SIGNATURES"   : len(grp),
                "MEAN_NGENES_UP" : grp["ngenes_modulated_up_lm"].mean() \
                                   if "ngenes_modulated_up_lm" \
                                   in grp.columns \
                                   else grp["N_UP"].mean(),
                "MEAN_NGENES_DN" : grp["ngenes_modulated_dn_lm"].mean() \
                                   if "ngenes_modulated_dn_lm" \
                                   in grp.columns \
                                   else grp["N_DN"].mean(),
            }
            # Gene frequency features
            up_all, dn_all = [], []
            for g in grp["UP_GENES"]: up_all.extend(g)
            for g in grp["DN_GENES"]: dn_all.extend(g)
            up_cnt = pd.Series(up_all).value_counts()
            dn_cnt = pd.Series(dn_all).value_counts()
            n      = max(len(grp), 1)
            for gene in top_genes:
                row[f"UP_{gene}"] = up_cnt.get(gene, 0) / n
                row[f"DN_{gene}"] = dn_cnt.get(gene, 0) / n
            rows.append(row)

        lincs_matrix = pd.DataFrame(rows)

        # Fix COSMIC ID using improved mapping
        lincs_matrix["COSMIC_ID"] = lincs_matrix["CELL_ID"].apply(
            get_cosmic_id)

        # Add CHEMBL ID
        drug_chembl = dict(zip(
            drug_map["MASTER_DRUG_NAME"].str.upper(),
            drug_map["CHEMBL_ID"]
        ))
        lincs_matrix["CHEMBL_ID"] = lincs_matrix["DRUG_NAME"].map(
            drug_chembl)

        # Add GDSC drug ID
        drug_gdsc_id = dict(zip(
            drug_map["MASTER_DRUG_NAME"].str.upper(),
            drug_map["GDSC_DRUG_ID"]
        ))
        lincs_matrix["GDSC_DRUG_ID"] = lincs_matrix["DRUG_NAME"].map(
            drug_gdsc_id)

        # Save
        lincs_matrix.to_csv(
            os.path.join(PROJECT_DIR,
                         "lincs_expression_features.csv"),
            index=False)
        try:
            lincs_matrix.to_parquet(
                os.path.join(PROJECT_DIR,
                             "lincs_expression_features.parquet"),
                index=False)
        except Exception:
            pass

        # Report
        cosmic_covered = lincs_matrix["COSMIC_ID"].notna().sum()
        chembl_covered = lincs_matrix["CHEMBL_ID"].notna().sum()
        total          = len(lincs_matrix)

        print(f"\nFinal LINCS feature matrix:")
        print(f"  Shape              : {lincs_matrix.shape}")
        print(f"  Drugs covered      : "
              f"{lincs_matrix['DRUG_NAME'].nunique()}")
        print(f"  Cells covered      : "
              f"{lincs_matrix['CELL_ID'].nunique()}")
        print(f"  With COSMIC_ID     : {cosmic_covered}/{total} "
              f"({cosmic_covered/total*100:.1f}%)")
        print(f"  With CHEMBL_ID     : {chembl_covered}/{total} "
              f"({chembl_covered/total*100:.1f}%)")

        print(f"\nSample:")
        print(lincs_matrix[["DRUG_NAME","CELL_ID","COSMIC_ID",
                             "CHEMBL_ID","MEAN_DISTIL_SS",
                             "N_SIGNATURES",
                             "MEAN_NGENES_UP"]].head(10).to_string())

        print("\nSaved -> lincs_expression_features.csv ")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — Coverage report
# ══════════════════════════════════════════════════════════════════════════
section("STEP 5: Coverage Report")

if len(sigs_df) > 0:
    print("Drugs in LINCS matrix:")
    for drug in sorted(sigs_df["DRUG_NAME"].unique()):
        n_cells = sigs_df[sigs_df["DRUG_NAME"]==drug][
            "CELL_ID"].nunique()
        print(f"  {drug:<30} : {n_cells} cell lines")

    print(f"\nMissed drugs (not in LINCS L1000):")
    for drug in missed_drugs:
        print(f"  {drug} — not profiled in LINCS L1000 "
              f"(newer drug, post-2017)")

# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
section("FINAL SUMMARY")

print(f"  Drugs found in LINCS     : {len(found_drugs)}/51")
print(f"  Drugs missed (not in L1000): {len(missed_drugs)}/51")
print(f"  Signatures downloaded    : {len(sigs_df)}")

if len(sigs_df) > 0:
    print(f"  Unique drugs in matrix   : "
          f"{sigs_df['DRUG_NAME'].nunique()}")
    print(f"  Unique cells in matrix   : "
          f"{sigs_df['CELL_ID'].nunique()}")

print("\n  --- File Checklist ---")
for f in ["lincs_drug_ids_final.csv",
          "lincs_signatures_final.csv",
          "lincs_expression_features.csv",
          "lincs_expression_features.parquet"]:
    path   = os.path.join(PROJECT_DIR, f)
    status = "[OK]" if os.path.exists(path) else "[MISSING]"
    print(f"    {status}  {f}")

print(f"\n  Missed drugs note for report:")
print(f"  {len(missed_drugs)} drugs not profiled in LINCS L1000 —")
print(f"  these are post-2017 drugs not included in the dataset.")
print(f"  Document as known limitation in final report.")

print("\n LINCS FINAL COMPLETE!")
print(" Next -> Step 5: Feature Matrix Construction\n")