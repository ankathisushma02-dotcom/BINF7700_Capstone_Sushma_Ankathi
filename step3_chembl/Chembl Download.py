#!/usr/bin/env python3
"""
ChEMBL Data Download Script
Capstone Project - Drug Synergy Prediction
Author: Sushma
"""

# ============================================================
# FIRST: Install required libraries (run once in terminal)
# pip install chembl_webresource_client pandas
# ============================================================

from chembl_webresource_client.new_client import new_client
import pandas as pd
import time


# ============================================================
# SCRIPT 1: Download Drug SMILES + Properties
# ============================================================

def download_drug_smiles():
    print("Starting ChEMBL drug SMILES download...")

    molecule = new_client.molecule

    # Get all approved drugs (Phase 4 = FDA approved)
    print("Querying ChEMBL for approved drugs...")
    approved_drugs = molecule.filter(
        max_phase=4
    ).only([
        'molecule_chembl_id',  # ChEMBL unique ID
        'pref_name',  # Preferred drug name
        'molecule_structures',  # Contains SMILES
        'molecule_properties',  # Molecular weight, LogP etc
        'molecule_synonyms',  # Alternative drug names
    ])

    # Convert to list (this triggers the API call)
    print("Fetching data from ChEMBL API... (this may take a few minutes)")
    drug_list = list(approved_drugs)
    print(f"Retrieved {len(drug_list)} approved drugs!")

    # Convert to DataFrame
    df = pd.DataFrame(drug_list)

    # Extract SMILES from nested structure
    print("Extracting SMILES strings...")
    df['canonical_smiles'] = df['molecule_structures'].apply(
        lambda x: x['canonical_smiles'] if x and 'canonical_smiles' in x else None
    )

    # Extract molecular properties from nested structure
    print("Extracting molecular properties...")
    df['molecular_weight'] = df['molecule_properties'].apply(
        lambda x: x['full_mwt'] if x and 'full_mwt' in x else None
    )
    df['alogp'] = df['molecule_properties'].apply(
        lambda x: x['alogp'] if x and 'alogp' in x else None
    )
    df['hbd'] = df['molecule_properties'].apply(
        lambda x: x['hbd'] if x and 'hbd' in x else None
    )
    df['hba'] = df['molecule_properties'].apply(
        lambda x: x['hba'] if x and 'hba' in x else None
    )
    df['psa'] = df['molecule_properties'].apply(
        lambda x: x['psa'] if x and 'psa' in x else None
    )
    df['num_rings'] = df['molecule_properties'].apply(
        lambda x: x['rtb'] if x and 'rtb' in x else None
    )

    # Extract synonyms
    df['synonyms'] = df['molecule_synonyms'].apply(
        lambda x: '|'.join([s['molecule_synonym'] for s in x]) if x else None
    )

    # Keep only useful columns
    df_final = df[[
        'molecule_chembl_id',
        'pref_name',
        'canonical_smiles',
        'molecular_weight',
        'alogp',
        'hbd',
        'hba',
        'psa',
        'num_rings',
        'synonyms'
    ]].copy()

    # Drop rows with no SMILES (not useful for RDKit)
    df_final = df_final.dropna(subset=['canonical_smiles'])
    print(f"Drugs with valid SMILES: {len(df_final)}")

    # Save to CSV
    output_file = 'chembl_drugs_smiles.csv'
    df_final.to_csv(output_file, index=False)
    print(f" Saved to {output_file}")
    print(f"   Rows: {len(df_final)}")
    print(f"   Columns: {list(df_final.columns)}")

    return df_final


# ============================================================
# SCRIPT 2: Download Drug Targets
# ============================================================

def download_drug_targets():
    print("\nStarting ChEMBL drug targets download...")

    mechanism = new_client.mechanism

    # Get drug-target mechanisms for approved drugs
    print("Querying ChEMBL for drug targets...")
    drug_mechanisms = mechanism.filter(
        max_phase=4  # Approved drugs only
    ).only([
        'molecule_chembl_id',  # Drug ChEMBL ID
        'target_chembl_id',  # Target ChEMBL ID
        'mechanism_of_action',  # e.g. "Inhibitor", "Agonist"
        'action_type',  # e.g. "INHIBITOR", "ACTIVATOR"
        'molecule_name',  # Drug name
        'target_name',  # Target protein name
    ])

    # Convert to list
    print("Fetching drug target data... (this may take a few minutes)")
    mechanism_list = list(drug_mechanisms)
    print(f"Retrieved {len(mechanism_list)} drug-target relationships!")

    # Convert to DataFrame
    df_targets = pd.DataFrame(mechanism_list)

    # Save to CSV
    output_file = 'chembl_drug_targets.csv'
    df_targets.to_csv(output_file, index=False)
    print(f" Saved to {output_file}")
    print(f"   Rows: {len(df_targets)}")
    print(f"   Columns: {list(df_targets.columns)}")

    return df_targets


# ============================================================
# MAIN - Run both scripts
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ChEMBL Data Download for Drug Synergy Capstone Project")
    print("=" * 60)

    # Download SMILES + Properties
    df_smiles = download_drug_smiles()

    # Small pause between API calls
    print("\nPausing 5 seconds before next query...")
    time.sleep(5)

    # Download Drug Targets
    df_targets = download_drug_targets()

    print("\n" + "=" * 60)
    print(" ALL DONE! Files saved:")
    print("   1. chembl_drugs_smiles.csv")
    print("   2. chembl_drug_targets.csv")
    print("=" * 60)
    print("\nNext step: Use chembl_drugs_smiles.csv with RDKit")
    print("to generate Morgan fingerprints for your ML model!")