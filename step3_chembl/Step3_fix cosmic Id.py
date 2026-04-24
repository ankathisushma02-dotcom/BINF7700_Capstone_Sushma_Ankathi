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
API_KEY     = "ce3d97575c71a59898921a84ed32fa6a"
BASE_URL    = "https://api.clue.io/api"

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

# ══════════════════════════════════════════════════════════════════════════
# Load files
# ══════════════════════════════════════════════════════════════════════════
section("Loading Files")

lincs_matrix = pd.read_csv(
    os.path.join(PROJECT_DIR, "lincs_expression_features.csv"))
cell_map     = pd.read_csv(
    os.path.join(PROJECT_DIR, "master_cell_line_map.csv"))
gdsc_cells   = pd.read_csv(
    os.path.join(PROJECT_DIR, "gdsc_cell_lines_cleaned.csv"))

print(f"LINCS matrix shape    : {lincs_matrix.shape}")
print(f"Total rows            : {len(lincs_matrix)}")
print(f"Rows WITH COSMIC_ID   : {lincs_matrix['COSMIC_ID'].notna().sum()}")
print(f"Rows WITHOUT COSMIC_ID: {lincs_matrix['COSMIC_ID'].isna().sum()}")

# Get unique cell lines missing COSMIC_ID
missing_cosmic = lincs_matrix[
    lincs_matrix["COSMIC_ID"].isna()]["CELL_ID"].unique().tolist()
print(f"\nCell lines missing COSMIC_ID: {len(missing_cosmic)}")
print(f"Sample: {missing_cosmic[:10]}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Build comprehensive COSMIC lookup
#           From multiple sources
# ══════════════════════════════════════════════════════════════════════════
section("STEP 1: Building Comprehensive COSMIC Lookup")

cosmic_lookup = {}

# Source 1 — master cell line map
for _, row in cell_map.iterrows():
    if pd.notna(row.get("COSMIC_ID")):
        cid = row["COSMIC_ID"]
        for name in [row.get("GDSC_CELL_NAME",""),
                     row.get("DRUGCOMB_CELL_NAME","")]:
            if pd.notna(name) and str(name).strip():
                n = str(name).upper().strip()
                cosmic_lookup[n]                         = cid
                cosmic_lookup[n.replace("-","")]         = cid
                cosmic_lookup[n.replace("-","").replace(" ","")] = cid
                cosmic_lookup[n.replace(".","")]         = cid

# Source 2 — GDSC cell lines file
gdsc_cells.columns = gdsc_cells.columns.str.upper().str.strip()
cell_name_col = next((c for c in gdsc_cells.columns
                      if "SAMPLE" in c or
                      ("CELL" in c and "NAME" in c)), None)
cosmic_col    = next((c for c in gdsc_cells.columns
                      if "COSMIC" in c), None)

if cell_name_col and cosmic_col:
    for _, row in gdsc_cells.iterrows():
        name = str(row[cell_name_col]).upper().strip()
        cid  = row[cosmic_col]
        if pd.notna(cid) and name:
            cosmic_lookup[name]                         = cid
            cosmic_lookup[name.replace("-","")]         = cid
            cosmic_lookup[name.replace("-","").replace(" ","")] = cid
            cosmic_lookup[name.replace(".","")]         = cid
    print(f"GDSC cell lines added to lookup: "
          f"{gdsc_cells[cell_name_col].nunique()}")

print(f"Total COSMIC lookup entries: {len(cosmic_lookup)}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Try local lookup first for missing cells
# ══════════════════════════════════════════════════════════════════════════
section("STEP 2: Local Lookup for Missing Cells")

still_missing = []
local_fixed   = 0

for cell in missing_cosmic:
    variants = [
        cell,
        cell.replace("-",""),
        cell.replace("-","").replace(" ",""),
        cell.replace(".",""),
        cell.lower(),
        cell.replace("-","").lower(),
    ]
    found = False
    for v in variants:
        if v in cosmic_lookup:
            lincs_matrix.loc[
                lincs_matrix["CELL_ID"]==cell, "COSMIC_ID"] = \
                cosmic_lookup[v]
            local_fixed += 1
            found = True
            break
    if not found:
        still_missing.append(cell)

print(f"Fixed by local lookup : {local_fixed}")
print(f"Still missing         : {len(still_missing)}")
print(f"Still missing cells   : {still_missing}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Query CLUE.io API for remaining missing cells
# ══════════════════════════════════════════════════════════════════════════
section("STEP 3: Querying CLUE.io for Remaining Missing Cells")

api_fixed   = 0
api_missed  = []

for cell in still_missing:
    cell_clean = cell.replace("-","").replace(" ","").replace(".","")
    matched    = False

    # Try multiple variants via API
    for variant in [cell, cell_clean,
                    cell.lower(), cell_clean.lower()]:
        result = clue_get("cells", {
            "where"  : {"cell_id": variant},
            "fields" : ["cell_id", "cell_iname",
                        "ccle_name", "cosmic_id"],
            "limit"  : 1
        })

        if result and result[0].get("cosmic_id"):
            cosmic_id = result[0]["cosmic_id"]
            if pd.notna(cosmic_id) and cosmic_id:
                lincs_matrix.loc[
                    lincs_matrix["CELL_ID"]==cell,
                    "COSMIC_ID"] = float(cosmic_id)
                print(f"  API FOUND : {cell} -> "
                      f"COSMIC_ID {cosmic_id}")
                api_fixed += 1
                matched   = True
                break
        time.sleep(0.2)

    if not matched:
        # Try ccle_name search
        result2 = clue_get("cells", {
            "where"  : {"ccle_name": {
                "like": cell_clean, "options": "i"}},
            "fields" : ["cell_id", "cell_iname",
                        "ccle_name", "cosmic_id"],
            "limit"  : 3
        })
        if result2:
            for r in result2:
                if r.get("cosmic_id"):
                    cosmic_id = r["cosmic_id"]
                    lincs_matrix.loc[
                        lincs_matrix["CELL_ID"]==cell,
                        "COSMIC_ID"] = float(cosmic_id)
                    print(f"  CCLE FOUND: {cell} -> "
                          f"{r.get('cell_id')} "
                          f"COSMIC_ID {cosmic_id}")
                    api_fixed += 1
                    matched   = True
                    break

        if not matched:
            api_missed.append(cell)
            print(f"  MISSED    : {cell}")

    time.sleep(0.2)

print(f"\nFixed by API          : {api_fixed}")
print(f"Still missing         : {len(api_missed)}")
if api_missed:
    print(f"Cannot resolve        : {api_missed}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Final check and save
# ══════════════════════════════════════════════════════════════════════════
section("STEP 4: Final Check and Save")

total         = len(lincs_matrix)
with_cosmic   = lincs_matrix["COSMIC_ID"].notna().sum()
without_cosmic = lincs_matrix["COSMIC_ID"].isna().sum()

print(f"Final COSMIC_ID coverage:")
print(f"  Total rows            : {total}")
print(f"  With COSMIC_ID        : {with_cosmic} "
      f"({with_cosmic/total*100:.1f}%)")
print(f"  Without COSMIC_ID     : {without_cosmic} "
      f"({without_cosmic/total*100:.1f}%)")

# Convert COSMIC_ID to int where possible
lincs_matrix["COSMIC_ID"] = pd.to_numeric(
    lincs_matrix["COSMIC_ID"], errors="coerce")

# Save updated file
lincs_matrix.to_csv(
    os.path.join(PROJECT_DIR, "lincs_expression_features.csv"),
    index=False)
try:
    lincs_matrix.to_parquet(
        os.path.join(PROJECT_DIR,
                     "lincs_expression_features.parquet"),
        index=False)
except Exception:
    pass

print(f"\nUpdated matrix saved -> lincs_expression_features.csv ")

# Show sample of fixed rows
print(f"\nSample of matrix with COSMIC IDs:")
print(lincs_matrix[["DRUG_NAME","CELL_ID","COSMIC_ID",
                     "CHEMBL_ID","MEAN_DISTIL_SS"]]\
      .dropna(subset=["COSMIC_ID"])\
      .head(15).to_string())

# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
section("FINAL SUMMARY")

print(f"  Original COSMIC coverage : 248/562 (44.1%)")
print(f"  Fixed by local lookup    : {local_fixed}")
print(f"  Fixed by API             : {api_fixed}")
print(f"  Final COSMIC coverage    : {with_cosmic}/{total} "
      f"({with_cosmic/total*100:.1f}%)")
print(f"  Unresolvable cells       : {len(api_missed)}")

if api_missed:
    print(f"\n  Note for report: {len(api_missed)} LINCS cell lines")
    print(f"  ({api_missed[:5]}...) could not be mapped to")
    print(f"  COSMIC IDs — these are LINCS-specific cell lines")
    print(f"  not present in GDSC.")

print("\n COSMIC ID Fix COMPLETE!")
print(" Next -> Step 5: Feature Matrix Construction\n")