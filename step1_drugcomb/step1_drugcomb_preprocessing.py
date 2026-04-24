# ============================================================
# CAPSTONE PROJECT - STEP 1: DrugCombDB Preprocessing
# Interpretable ML Framework for Drug Synergy Prediction
# Student: Sushma |Instructor: Ayansola Oyeronke |Mentor: Dr. Nabil Atallah | Spring 2026
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── 1. LOAD DATA ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading DrugCombDB scores.csv...")
print("=" * 60)

file_path = r"C:\Users\sush\PYCHARM\Capstone project\drugcombs_scored.csv"
df = pd.read_csv(file_path)

print(f" File loaded successfully!")
print(f"   Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\n Columns: {list(df.columns)}")
print(f"\n First 5 rows:")
print(df.head())

# ── 2. BASIC INSPECTION ───────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Basic Inspection...")
print("=" * 60)

print(f"\n Data Types:")
print(df.dtypes)

print(f"\n Missing Values per Column:")
print(df.isnull().sum())

print(f"\n Basic Statistics BEFORE cleaning (Synergy Scores):")
print(df[['ZIP', 'Bliss', 'Loewe', 'HSA']].describe())

# ── 3. CLEAN MISSING VALUES ───────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Removing rows with missing synergy scores...")
print("=" * 60)

before = len(df)
df = df.dropna(subset=['ZIP', 'Bliss', 'Loewe', 'HSA', 'Drug1', 'Drug2'])
after = len(df)

print(f"   Rows before cleaning : {before}")
print(f"   Rows after cleaning  : {after}")
print(f"   Rows removed         : {before - after}")

# ── 4. STANDARDIZE DRUG & CELL LINE NAMES ────────────────────
print("\n" + "=" * 60)
print("STEP 4: Standardizing Drug and Cell Line names...")
print("=" * 60)

df['Drug1']     = df['Drug1'].str.strip().str.upper()
df['Drug2']     = df['Drug2'].str.strip().str.upper()
df['Cell line'] = df['Cell line'].str.strip().str.upper()

print(f" Drug1, Drug2, Cell line names standardized to UPPERCASE")
print(f"\n   Unique Drug1 count : {df['Drug1'].nunique()}")
print(f"   Unique Drug2 count : {df['Drug2'].nunique()}")
print(f"   Unique Cell lines  : {df['Cell line'].nunique()}")

# ── 5. REMOVE DUPLICATES ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Removing Duplicates...")
print("=" * 60)

before = len(df)
df = df.drop_duplicates(subset=['Drug1', 'Drug2', 'Cell line'])
after = len(df)

print(f"   Rows before dedup  : {before}")
print(f"   Rows after dedup   : {after}")
print(f"   Duplicates removed : {before - after}")

# ── 6. REMOVE OUTLIERS (IQR method) ──────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Removing Outliers using IQR method...")
print("=" * 60)

print("\n Score ranges BEFORE outlier removal:")
for col in ['ZIP', 'Bliss', 'Loewe', 'HSA']:
    print(f"   {col:6s} → min: {df[col].min():15.2f}  max: {df[col].max():15.2f}")

before = len(df)

for col in ['ZIP', 'Bliss', 'Loewe', 'HSA']:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR
    df = df[df[col].between(lower, upper)]

after = len(df)

print(f"\n   Rows before outlier removal : {before}")
print(f"   Rows after outlier removal  : {after}")
print(f"   Outlier rows removed        : {before - after}")

print("\n Score ranges AFTER outlier removal:")
for col in ['ZIP', 'Bliss', 'Loewe', 'HSA']:
    print(f"   {col:6s} → min: {df[col].min():10.2f}  max: {df[col].max():10.2f}")

# ── 7. CREATE BINARY SYNERGY LABEL (using Loewe) ─────────────
print("\n" + "=" * 60)
print("STEP 7: Creating Binary Synergy Label from Loewe score...")
print("=" * 60)

df['synergy_label'] = (df['Loewe'] > 10).astype(int)

synergistic     = df['synergy_label'].sum()
non_synergistic = len(df) - synergistic
total           = len(df)

print(f"   Threshold used        : Loewe > 10 = Synergistic")
print(f"   Synergistic (1)       : {synergistic} ({100*synergistic/total:.1f}%)")
print(f"   Non-synergistic (0)   : {non_synergistic} ({100*non_synergistic/total:.1f}%)")
print(f"\n  Class imbalance will be handled in Week 4 using SMOTE during ML training.")
print(f"  Confirm Loewe threshold with Dr. Atallah before ML training!")

# ── 8. FINAL STATISTICS ───────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Final Statistics after all cleaning...")
print("=" * 60)

print(f"\n Basic Statistics AFTER cleaning (Synergy Scores):")
print(df[['ZIP', 'Bliss', 'Loewe', 'HSA']].describe())

# ── 9. SAVE CLEAN FILE ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Saving cleaned file...")
print("=" * 60)

output_path = r"C:\Users\sush\PYCHARM\Capstone project\drugcomb_cleaned.csv"
df.to_csv(output_path, index=False)

print(f" Cleaned file saved to:")
print(f"   {output_path}")
print(f"\n Final cleaned dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\n Final columns: {list(df.columns)}")

# ── 10. VISUALIZATIONS ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10: Creating Visualizations...")
print("=" * 60)

# Score distributions AFTER outlier removal
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('DrugCombDB Synergy Score Distributions (After Outlier Removal)',
             fontsize=15, fontweight='bold')

scores = ['ZIP', 'Bliss', 'Loewe', 'HSA']
colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple']

for i, (score, color) in enumerate(zip(scores, colors)):
    ax = axes[i // 2][i % 2]
    ax.hist(df[score], bins=100, color=color, alpha=0.7, edgecolor='none')
    ax.axvline(x=10, color='red', linestyle='--', linewidth=1.5, label='Threshold = 10')
    ax.set_title(f'{score} Score Distribution', fontweight='bold')
    ax.set_xlabel(f'{score} Score')
    ax.set_ylabel('Count')
    ax.legend()

plt.tight_layout()
plot_path = r"C:\Users\sush\PYCHARM\Capstone project\drugcomb_score_distributions_clean.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f" Clean distribution plot saved!")

# Class balance pie chart
fig2, ax2 = plt.subplots(figsize=(7, 7))
ax2.pie([synergistic, non_synergistic],
        labels=['Synergistic (1)', 'Non-synergistic (0)'],
        colors=['mediumseagreen', 'coral'],
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 13})
ax2.set_title('Class Balance (Loewe > 10 threshold)\n Will be handled with SMOTE in Week 4',
              fontsize=13, fontweight='bold')
pie_path = r"C:\Users\sush\PYCHARM\Capstone project\drugcomb_class_balance.png"
plt.savefig(pie_path, dpi=150, bbox_inches='tight')
print(f" Class balance pie chart saved!")

# Boxplots to confirm outliers removed
fig3, axes3 = plt.subplots(1, 4, figsize=(16, 5))
fig3.suptitle('Synergy Score Boxplots (After Outlier Removal)', fontsize=14, fontweight='bold')

for i, (score, color) in enumerate(zip(scores, colors)):
    axes3[i].boxplot(df[score], patch_artist=True,
                     boxprops=dict(facecolor=color, alpha=0.6))
    axes3[i].set_title(score, fontweight='bold')
    axes3[i].set_ylabel('Score')

plt.tight_layout()
box_path = r"C:\Users\sush\PYCHARM\Capstone project\drugcomb_boxplots_clean.png"
plt.savefig(box_path, dpi=150, bbox_inches='tight')
print(f" Boxplot saved!")

# ── 11. FINAL SUMMARY ─────────────────────────────────────────
print("\n" + "=" * 60)
print(" STEP 1 COMPLETE — DrugCombDB Preprocessing v2 Done!")
print("=" * 60)
print(f"""
 Output Files Created:
   1. drugcomb_cleaned.csv                    ← clean data with labels
   2. drugcomb_score_distributions_clean.png  ← clean score distributions
   3. drugcomb_class_balance.png              ← class balance chart
   4. drugcomb_boxplots_clean.png             ← boxplots after outlier removal

 Final Dataset Summary:
   Total combinations  : {len(df)}
   Unique Drug1        : {df['Drug1'].nunique()}
   Unique Drug2        : {df['Drug2'].nunique()}
   Unique cell lines   : {df['Cell line'].nunique()}
   Synergistic (1)     : {synergistic} ({100*synergistic/total:.1f}%)
   Non-synergistic (0) : {non_synergistic} ({100*non_synergistic/total:.1f}%)



  Next Step: STEP 2 — GDSC Preprocessing
""")