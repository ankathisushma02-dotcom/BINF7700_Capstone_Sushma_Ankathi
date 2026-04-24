"""
============================================================
  STEP 7: SHAP + LIME + TISSUE STRATIFICATION
  Capstone Project — Nabil Atallah, Ph.D.
  Northeastern University — MS Bioinformatics Spring 2026
============================================================
  Output files:
    shap_feature_importance.csv
    shap_top_genes_for_enrichment.csv
    shap_gene_ranking_for_gsea.csv
    shap_summary_bar.png
    shap_beeswarm.png
    shap_group_contribution.png
    shap_waterfall_synergistic.png
    shap_dependence_top_feature.png
    lime_results.csv
    lime_synergistic_explanation.png
    shap_lime_agreement.csv
    tissue_stratification_results.csv
    tissue_stratification_plot.png
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import shap
from lime import lime_tabular
import xgboost as xgb

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
PROJECT_DIR = r"C:\Users\sush\PYCHARM\Capstone project"


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ── Feature group assignment from prefix ──────────────────────────────────
# feature_names.csv has only one column (the feature names).
# Groups are inferred from name patterns.
def assign_group(f):
    if "LINCS" in f:                          return "LINCS"
    if f.startswith("MUT_"):                  return "MUTATION"
    if f.startswith("CNV_"):                  return "CNV"
    if f.startswith("TISSUE_"):               return "TISSUE"
    if "_FP_" in f:                           return "FP"
    if f in ["ZIP_SCORE","BLISS_SCORE","HSA_SCORE"]: return "SCORE"
    if "LN_IC50" in f or "_AUC" in f:        return "IC50_AUC"
    if "TANIMOTO" in f:                       return "PAIR"
    if any(x in f for x in ["ALOGP","PSA","HBD","HBA","MW",
                             "MOLECULAR_WEIGHT","NUM_RINGS",
                             "SCAFFOLD","TARGET_COUNT"]): return "DRUG_PROP"
    if f.startswith("MSI") or "GROWTH" in f: return "CELL"
    return "FP"   # remaining unmapped are almost certainly fingerprint bits


# colour map shared across all plots
GROUP_COLORS = {
    "FP"       : "#0891B2",
    "MUTATION" : "#7C3AED",
    "CNV"      : "#DB2777",
    "LINCS"    : "#16A34A",
    "SCORE"    : "#D97706",
    "IC50_AUC" : "#DC2626",
    "PAIR"     : "#EA580C",
    "TISSUE"   : "#0D9488",
    "DRUG_PROP": "#6366F1",
    "CELL"     : "#84CC16",
    "OTHER"    : "#64748B",
}


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load Model, Scaler, Data
# ══════════════════════════════════════════════════════════════════════════
section("STEP 1: Loading Model, Scaler and Data")

# Load calibrated XGBoost (best model from Step 6 v2)
with open(os.path.join(PROJECT_DIR, "xgboost_calibrated_v2.pkl"), "rb") as f:
    xgb_calibrated = pickle.load(f)
print("  xgboost_calibrated_v2.pkl loaded")

# Load scaler
with open(os.path.join(PROJECT_DIR, "scaler_v2.pkl"), "rb") as f:
    scaler = pickle.load(f)
print("  scaler_v2.pkl loaded")

# ── Feature names ────────────────────────────────────────────────────────
# feature_names.csv has ONE column — the feature name.
# We read it safely regardless of what the header says.
feat_df = pd.read_csv(os.path.join(PROJECT_DIR, "feature_names.csv"))
feature_cols = feat_df.iloc[:, 0].tolist()

# Build group lookup from name patterns
feature_grps = {f: assign_group(f) for f in feature_cols}
print(f"  Features loaded  : {len(feature_cols)}")
print(f"  Group counts     : {pd.Series(list(feature_grps.values())).value_counts().to_dict()}")

# ── Test set ──────────────────────────────────────────────────────────────
test_df = pd.read_csv(os.path.join(PROJECT_DIR, "test_set.csv"))

# Keep only feature columns that exist in test_df
common_features = [f for f in feature_cols if f in test_df.columns]
print(f"  Features in test : {len(common_features)} / {len(feature_cols)}")

X_test_raw    = test_df[common_features].values.astype(np.float32)
X_test_raw    = np.nan_to_num(X_test_raw, nan=0.0, posinf=0.0, neginf=0.0)
y_test        = test_df["SYNERGY_LABEL"].values.astype(int)
X_test_scaled = scaler.transform(X_test_raw)

print(f"  Test set shape   : {X_test_scaled.shape}")
print(f"  Synergistic      : {y_test.sum()} ({y_test.mean()*100:.1f}%)")

# ── Train set (for LIME background only) ─────────────────────────────────
train_df = pd.read_csv(os.path.join(PROJECT_DIR, "train_set.csv"))
X_train_raw    = train_df[common_features].values.astype(np.float32)
X_train_raw    = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
X_train_scaled = scaler.transform(X_train_raw)
print(f"  Train set shape  : {X_train_scaled.shape}")

# Sync feature_cols to only common features (avoids index mismatches later)
feature_cols = common_features
feature_grps = {f: assign_group(f) for f in feature_cols}


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Extract base XGBoost + SHAP Analysis
# ══════════════════════════════════════════════════════════════════════════
section("STEP 2: SHAP Analysis — XGBoost Calibrated")

# ── FIX 2: handle both sklearn CalibratedClassifierCV versions ───────────
try:
    base_xgb = xgb_calibrated.estimator          # sklearn < 1.2
    print("  base_xgb via .estimator")
except AttributeError:
    try:
        base_xgb = xgb_calibrated.calibrated_classifiers_[0].estimator
        print("  base_xgb via .calibrated_classifiers_[0].estimator")
    except Exception as e:
        raise RuntimeError(
            f"Could not extract base XGBoost from calibrated model: {e}"
        )

print("  Initialising SHAP TreeExplainer...")
explainer = shap.TreeExplainer(base_xgb)

# 1 000-sample subset of test set
N_SHAP = min(1000, len(X_test_scaled))
np.random.seed(42)
shap_idx    = np.random.choice(len(X_test_scaled), N_SHAP, replace=False)
X_shap      = X_test_scaled[shap_idx]
y_shap      = y_test[shap_idx]

print(f"  Computing SHAP values for {N_SHAP} samples...")
shap_values = explainer.shap_values(X_shap)

# For binary classifiers shap_values may be a list [neg_class, pos_class]
if isinstance(shap_values, list):
    shap_values = shap_values[1]   # positive class = synergistic
print(f"  SHAP values shape: {shap_values.shape}")

# Mean absolute SHAP per feature
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({
    "FEATURE_NAME"  : feature_cols,
    "MEAN_ABS_SHAP" : mean_abs_shap,
    "FEATURE_GROUP" : [feature_grps.get(f, "OTHER") for f in feature_cols],
}).sort_values("MEAN_ABS_SHAP", ascending=False).reset_index(drop=True)

shap_df.to_csv(
    os.path.join(PROJECT_DIR, "shap_feature_importance.csv"), index=False)
print("  Saved -> shap_feature_importance.csv")

print("\n  Top 20 features by SHAP:")
print(shap_df.head(20)[["FEATURE_NAME", "FEATURE_GROUP",
                          "MEAN_ABS_SHAP"]].to_string(index=False))

# ── Gene lists for Step 8 enrichment ──────────────────────────────────────
top_mut_feats   = shap_df[shap_df["FEATURE_NAME"].str.startswith("MUT_")].head(50)
top_lincs_feats = shap_df[shap_df["FEATURE_NAME"].str.contains("LINCS", na=False)].head(50)

top_mut_genes = (
    top_mut_feats["FEATURE_NAME"]
    .str.replace("MUT_", "", regex=False)
    .tolist()
)
top_lincs_genes = (
    top_lincs_feats["FEATURE_NAME"]
    .str.replace("D1_LINCS_UP_", "", regex=False)
    .str.replace("D1_LINCS_DN_", "", regex=False)
    .str.replace("D2_LINCS_UP_", "", regex=False)
    .str.replace("D2_LINCS_DN_", "", regex=False)
    .tolist()
)

all_top_genes = list(set(top_mut_genes + top_lincs_genes))
pd.DataFrame({"GENE": all_top_genes}).to_csv(
    os.path.join(PROJECT_DIR, "shap_top_genes_for_enrichment.csv"), index=False)

# Ranked gene list for GSEA (mutation genes with SHAP score)
shap_gene_ranked = shap_df[
    shap_df["FEATURE_NAME"].str.startswith("MUT_")
][["FEATURE_NAME", "MEAN_ABS_SHAP"]].copy()
shap_gene_ranked["GENE"] = (
    shap_gene_ranked["FEATURE_NAME"].str.replace("MUT_", "", regex=False)
)
shap_gene_ranked = (
    shap_gene_ranked[["GENE", "MEAN_ABS_SHAP"]]
    .sort_values("MEAN_ABS_SHAP", ascending=False)
)
shap_gene_ranked.to_csv(
    os.path.join(PROJECT_DIR, "shap_gene_ranking_for_gsea.csv"), index=False)

print(f"\n  Top mutation genes   : {len(top_mut_genes)}")
print(f"  Top LINCS genes      : {len(top_lincs_genes)}")
print(f"  Combined unique genes: {len(all_top_genes)}")
print("  Saved -> shap_top_genes_for_enrichment.csv")
print("  Saved -> shap_gene_ranking_for_gsea.csv")


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — SHAP Plots
# ══════════════════════════════════════════════════════════════════════════
section("STEP 3: SHAP Visualisations")

# Plot 1 — SHAP Summary Bar (Top 30) ──────────────────────────────────────
print("  Generating SHAP summary bar plot...")
top30  = shap_df.head(30)
colors = [GROUP_COLORS.get(g, "#64748B") for g in top30["FEATURE_GROUP"]]

fig, ax = plt.subplots(figsize=(12, 10))
ax.barh(range(len(top30)),
        top30["MEAN_ABS_SHAP"].values[::-1],
        color=colors[::-1])
ax.set_yticks(range(len(top30)))
ax.set_yticklabels(top30["FEATURE_NAME"].values[::-1], fontsize=9)
ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
ax.set_title("Top 30 Features by SHAP Importance\n"
             "(XGBoost Calibrated — Test Set)", fontsize=14, fontweight="bold")

present_groups = top30["FEATURE_GROUP"].unique()
patches = [mpatches.Patch(color=GROUP_COLORS.get(g, "#64748B"), label=g)
           for g in present_groups]
ax.legend(handles=patches, loc="lower right", fontsize=8, title="Feature Group")
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, "shap_summary_bar.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> shap_summary_bar.png")

# Plot 2 — SHAP Beeswarm (Top 20) ─────────────────────────────────────────
print("  Generating SHAP beeswarm plot...")
top20_names = shap_df.head(20)["FEATURE_NAME"].tolist()
top20_idx   = [feature_cols.index(f) for f in top20_names if f in feature_cols]

shap.summary_plot(
    shap_values[:, top20_idx],
    X_shap[:, top20_idx],
    feature_names=[feature_cols[i] for i in top20_idx],
    show=False,
    max_display=20,
    plot_size=(12, 8),
)
plt.title("SHAP Beeswarm — Top 20 Features",
          fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, "shap_beeswarm.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> shap_beeswarm.png")

# Plot 3 — SHAP by Feature Group ──────────────────────────────────────────
print("  Generating SHAP group contribution plot...")
grp_shap = (shap_df.groupby("FEATURE_GROUP")["MEAN_ABS_SHAP"]
            .sum().sort_values(ascending=False))

fig, ax = plt.subplots(figsize=(10, 6))
bar_colors = [GROUP_COLORS.get(g, "#64748B") for g in grp_shap.index]
bars = ax.bar(grp_shap.index, grp_shap.values, color=bar_colors)
ax.set_xlabel("Feature Group", fontsize=12)
ax.set_ylabel("Total |SHAP Value|", fontsize=12)
ax.set_title("SHAP Contribution by Feature Group\n(Multi-Omics Attribution)",
             fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right")
for bar, val in zip(bars, grp_shap.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, "shap_group_contribution.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> shap_group_contribution.png")

# Plot 4 — SHAP Waterfall for one synergistic sample ──────────────────────
print("  Generating SHAP waterfall plot...")
syn_idx_shap = np.where(y_shap == 1)[0]
if len(syn_idx_shap) > 0:
    sample_idx = syn_idx_shap[0]

    # FIX 3: handle expected_value as list or scalar
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[1])   # positive class
    else:
        base_val = float(base_val)

    shap.waterfall_plot(
        shap.Explanation(
            values        = shap_values[sample_idx],
            base_values   = base_val,
            data          = X_shap[sample_idx],
            feature_names = feature_cols,
        ),
        show=False,
        max_display=15,
    )
    plt.title("SHAP Waterfall — Single Synergistic Prediction",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, "shap_waterfall_synergistic.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved -> shap_waterfall_synergistic.png")
else:
    print("  No synergistic samples in SHAP subset — waterfall skipped")

# Plot 5 — SHAP Dependence (top feature) ──────────────────────────────────
top_feat     = shap_df.iloc[0]["FEATURE_NAME"]
top_feat_idx = feature_cols.index(top_feat)
print(f"  Generating SHAP dependence plot for: {top_feat}")
shap.dependence_plot(
    top_feat_idx,
    shap_values,
    X_shap,
    feature_names=feature_cols,
    show=False,
)
plt.title(f"SHAP Dependence — {top_feat}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, "shap_dependence_top_feature.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> shap_dependence_top_feature.png")


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — LIME Analysis
# ══════════════════════════════════════════════════════════════════════════
section("STEP 4: LIME Analysis")

# FIX 4 & 6: use a 500-row background sample — much faster than full train
print("  Building LIME background (500 rows from train set)...")
np.random.seed(42)
lime_bg_idx    = np.random.choice(len(X_train_scaled), 500, replace=False)
lime_background = X_train_scaled[lime_bg_idx]

# FIX 5: no stray backslash on init line
print("  Initialising LIME explainer...")
lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data        = lime_background,          # FIX 6: use background
    feature_names        = feature_cols,
    class_names          = ["Non-Synergistic", "Synergistic"],
    mode                 = "classification",
    discretize_continuous= True,
    random_state         = 42,
)

# Predict function — calibrated model
def predict_fn(X):
    return xgb_calibrated.predict_proba(X)

# Run LIME on 5 synergistic and 5 non-synergistic samples
syn_idx_test = np.where(y_test == 1)[0][:5]
non_idx_test = np.where(y_test == 0)[0][:5]
lime_results = []

print(f"  Running LIME on {len(syn_idx_test)} synergistic samples...")
for i, idx in enumerate(syn_idx_test):
    exp = lime_explainer.explain_instance(
        data_row  = X_test_scaled[idx],
        predict_fn= predict_fn,
        num_features = 20,
        num_samples  = 200,          # FIX 4: reduced from 500 for speed
    )
    for feat, weight in exp.as_list():
        lime_results.append({
            "SAMPLE_IDX" : int(idx),
            "TRUE_LABEL" : 1,
            "SAMPLE_TYPE": "Synergistic",
            "FEATURE"    : feat,
            "LIME_WEIGHT": weight,
        })
    print(f"    LIME synergistic sample {i+1}/5 done")

print(f"  Running LIME on {len(non_idx_test)} non-synergistic samples...")
for i, idx in enumerate(non_idx_test):
    exp = lime_explainer.explain_instance(
        data_row  = X_test_scaled[idx],
        predict_fn= predict_fn,
        num_features = 20,
        num_samples  = 200,
    )
    for feat, weight in exp.as_list():
        lime_results.append({
            "SAMPLE_IDX" : int(idx),
            "TRUE_LABEL" : 0,
            "SAMPLE_TYPE": "Non-Synergistic",
            "FEATURE"    : feat,
            "LIME_WEIGHT": weight,
        })
    print(f"    LIME non-synergistic sample {i+1}/5 done")

lime_df = pd.DataFrame(lime_results)
lime_df.to_csv(os.path.join(PROJECT_DIR, "lime_results.csv"), index=False)
print("  Saved -> lime_results.csv")


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — LIME Plots
# ══════════════════════════════════════════════════════════════════════════
section("STEP 5: LIME Visualisations")

# Plot 1 — Average LIME contributions for synergistic samples ─────────────
print("  Generating LIME bar plot for synergistic samples...")
syn_lime = (
    lime_df[lime_df["SAMPLE_TYPE"] == "Synergistic"]
    .groupby("FEATURE")["LIME_WEIGHT"]
    .mean()
    .sort_values(key=abs, ascending=False)
    .head(15)
)

bar_colors = ["#16A34A" if v > 0 else "#DC2626" for v in syn_lime.values]
fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(range(len(syn_lime)), syn_lime.values[::-1], color=bar_colors[::-1])
ax.set_yticks(range(len(syn_lime)))
ax.set_yticklabels(syn_lime.index[::-1], fontsize=9)
ax.set_xlabel("LIME Weight", fontsize=12)
ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_title("LIME Feature Contributions — Synergistic Predictions\n"
             "(Average over 5 samples, Top 15 features)",
             fontsize=13, fontweight="bold")
green_p = mpatches.Patch(color="#16A34A", label="Pushes towards Synergistic")
red_p   = mpatches.Patch(color="#DC2626", label="Pushes towards Non-Synergistic")
ax.legend(handles=[green_p, red_p], fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, "lime_synergistic_explanation.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> lime_synergistic_explanation.png")

# Plot 2 — SHAP vs LIME agreement ─────────────────────────────────────────
print("  Computing SHAP-LIME agreement...")
shap_top_set = set(shap_df.head(15)["FEATURE_NAME"].tolist())
lime_top_set = set(
    lime_df.groupby("FEATURE")["LIME_WEIGHT"]
    .apply(lambda x: np.abs(x).mean())
    .sort_values(ascending=False)
    .head(15)
    .index
)
overlap      = shap_top_set & lime_top_set
agree_pct    = len(overlap) / 15 * 100

print(f"\n  SHAP-LIME Feature Agreement:")
print(f"    Top 15 SHAP features : {len(shap_top_set)}")
print(f"    Top 15 LIME features : {len(lime_top_set)}")
print(f"    Overlap (agreement)  : {len(overlap)}")
print(f"    Agreement rate       : {agree_pct:.1f}%")

pd.DataFrame({
    "SHAP_TOP15"   : list(shap_top_set),
    "IN_LIME_TOP15": [f in lime_top_set for f in shap_top_set],
}).to_csv(os.path.join(PROJECT_DIR, "shap_lime_agreement.csv"), index=False)
print("  Saved -> shap_lime_agreement.csv")


# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — Tissue Stratification Analysis
# ══════════════════════════════════════════════════════════════════════════
section("STEP 6: Tissue Stratification Analysis")

full_df     = pd.read_csv(os.path.join(PROJECT_DIR, "feature_matrix_full.csv"))
tissue_cols = [c for c in full_df.columns if c.startswith("TISSUE_")]
print(f"  Tissue columns found: {len(tissue_cols)}")

tissue_stats = []
for tc in tissue_cols:
    tissue_name = tc.replace("TISSUE_", "")
    mask        = full_df[tc] == 1
    subset      = full_df[mask]
    if len(subset) > 10:
        tissue_stats.append({
            "TISSUE"       : tissue_name,
            "N_TOTAL"      : int(len(subset)),
            "N_SYNERGISTIC": int(subset["SYNERGY_LABEL"].sum()),
            "SYNERGY_RATE" : round(float(subset["SYNERGY_LABEL"].mean()), 4),
        })

tissue_stats_df = (
    pd.DataFrame(tissue_stats)
    .sort_values("SYNERGY_RATE", ascending=False)
    .reset_index(drop=True)
)
tissue_stats_df.to_csv(
    os.path.join(PROJECT_DIR, "tissue_stratification_results.csv"), index=False)

print(f"\n  Tissue Synergy Rates:")
print(tissue_stats_df.to_string(index=False))
print("  Saved -> tissue_stratification_results.csv")

# Plot — Tissue synergy rates ──────────────────────────────────────────────
mean_rate  = tissue_stats_df["SYNERGY_RATE"].mean()
bar_colors = [
    "#DC2626" if r > mean_rate else "#0891B2"
    for r in tissue_stats_df["SYNERGY_RATE"]
]
fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(tissue_stats_df["TISSUE"], tissue_stats_df["SYNERGY_RATE"],
        color=bar_colors)
ax.axvline(x=mean_rate, color="black", linestyle="--", linewidth=1.5,
           label=f"Mean = {mean_rate:.3f}")
ax.set_xlabel("Synergy Rate", fontsize=12)
ax.set_title("Drug Synergy Rate by Tissue Type\n(Context Stratification Analysis)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, "tissue_stratification_plot.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> tissue_stratification_plot.png")


# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
section("FINAL SUMMARY")

print(f"""
  SHAP samples analysed  : {N_SHAP}
  Top mutation genes     : {len(top_mut_genes)}
  Top LINCS genes        : {len(top_lincs_genes)}
  Combined unique genes  : {len(all_top_genes)}
  SHAP-LIME agreement    : {agree_pct:.1f}%
  Tissues analysed       : {len(tissue_stats_df)}
""")

files_checklist = [
    ("shap_feature_importance.csv",      "SHAP scores — all features"),
    ("shap_top_genes_for_enrichment.csv","Gene list for Step 8 gseapy ORA"),
    ("shap_gene_ranking_for_gsea.csv",   "Ranked gene list for GSEA"),
    ("shap_summary_bar.png",             "Top 30 SHAP bar chart"),
    ("shap_beeswarm.png",                "SHAP beeswarm top 20"),
    ("shap_group_contribution.png",      "SHAP by omics group"),
    ("shap_waterfall_synergistic.png",   "Waterfall — single prediction"),
    ("shap_dependence_top_feature.png",  "Dependence — top feature"),
    ("lime_results.csv",                 "LIME raw weights"),
    ("lime_synergistic_explanation.png", "LIME bar — synergistic avg"),
    ("shap_lime_agreement.csv",          "SHAP-LIME overlap table"),
    ("tissue_stratification_results.csv","Tissue synergy rates"),
    ("tissue_stratification_plot.png",   "Tissue stratification plot"),
]

print("  --- File Checklist ---")
for fname, desc in files_checklist:
    status = "[OK]" if os.path.exists(
        os.path.join(PROJECT_DIR, fname)) else "[MISSING]"
    print(f"    {status}  {fname:<42} {desc}")

print("""
  Key outputs for Step 8:
    shap_top_genes_for_enrichment.csv  -> gseapy ORA / clusterProfiler input
    shap_gene_ranking_for_gsea.csv     -> preranked GSEA input
    tissue_stratification_results.csv  -> context enrichment input
""")

print("=" * 60)
print("  STEP 7 — SHAP + LIME + Tissue Stratification COMPLETE!")
print("  Next -> Step 8: GSEA + Pathway Enrichment + Chemotype")
print("=" * 60)