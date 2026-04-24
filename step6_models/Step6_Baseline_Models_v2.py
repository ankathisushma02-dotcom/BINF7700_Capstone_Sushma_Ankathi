"""
============================================================
  STEP 6 v2 FINAL: BASELINE ML MODELS
  Capstone — Dr. Nabil Atallah | Northeastern University
  MS Bioinformatics Spring 2026
============================================================
  Models     : XGBoost | LightGBM | RandomForest
  Train data : train_set.csv        (60,456 — ORIGINAL)
               train_set_smote.csv  (118,308 — RF only)
  Test data  : test_set.csv         (15,115 — original 2.2%)
  Features   : 2,703 features (synergy scores included)
  Metrics    : AUROC · PRAUC · F1 · Precision · Recall · CM
============================================================
"""

import os, time, warnings, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, train_test_split
)
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
PROJECT_DIR  = r"C:\Users\sush\PYCHARM\Capstone project"
TRAIN_ORIG   = os.path.join(PROJECT_DIR, "train_set.csv")
TRAIN_SMOTE  = os.path.join(PROJECT_DIR, "train_set_smote.csv")
TEST_SET     = os.path.join(PROJECT_DIR, "test_set.csv")
FEAT_NAMES   = os.path.join(PROJECT_DIR, "feature_names.csv")
LABEL_COL    = "SYNERGY_LABEL"
RANDOM_STATE = 42

def section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def get_features(df, feature_names):
    cols = [f for f in feature_names if f in df.columns]
    X    = df[cols].values.astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), cols

def evaluate_model(name, model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    auroc  = roc_auc_score(y_test, y_prob)
    prauc  = average_precision_score(y_test, y_prob)
    thresholds  = np.arange(0.01, 0.99, 0.005)
    f1_scores   = [f1_score(y_test, (y_prob >= t).astype(int),
                            zero_division=0) for t in thresholds]
    best_thresh = float(thresholds[np.argmax(f1_scores)])
    y_pred      = (y_prob >= best_thresh).astype(int)
    metrics = {
        "model"          : name,
        "auroc"          : round(auroc, 4),
        "prauc"          : round(prauc, 4),
        "best_threshold" : round(best_thresh, 3),
        "f1_best"        : round(f1_score(y_test, y_pred,
                                          zero_division=0), 4),
        "precision_best" : round(precision_score(y_test, y_pred,
                                                 zero_division=0), 4),
        "recall_best"    : round(recall_score(y_test, y_pred,
                                              zero_division=0), 4),
    }
    cm = confusion_matrix(y_test, y_pred)
    metrics.update({
        "tn": int(cm[0,0]), "fp": int(cm[0,1]),
        "fn": int(cm[1,0]), "tp": int(cm[1,1])
    })
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    p, r, _     = precision_recall_curve(y_test, y_prob)
    curves = {
        "roc" : (fpr, tpr, auroc),
        "pr"  : (r, p, prauc),
        "cm"  : cm,
        "prob": y_prob
    }
    return metrics, curves

# ══════════════════════════════════════════════════════════════
# STEP 1 — Load Data
# ══════════════════════════════════════════════════════════════
section("STEP 1: Loading Data")

train_orig  = pd.read_csv(TRAIN_ORIG)
train_smote = pd.read_csv(TRAIN_SMOTE)
test_df     = pd.read_csv(TEST_SET)
feat_df     = pd.read_csv(FEAT_NAMES)
feature_names = feat_df.iloc[:, 0].tolist()
print(f"  Feature names: {len(feature_names)}")

X_train_orig,  common_features = get_features(train_orig,  feature_names)
X_train_smote, _               = get_features(train_smote, feature_names)
X_test,        _               = get_features(test_df,     feature_names)

y_train_orig  = train_orig[LABEL_COL].values.astype(int)
y_train_smote = train_smote[LABEL_COL].values.astype(int)
y_test        = test_df[LABEL_COL].values.astype(int)

print(f"  Original train : {X_train_orig.shape} | "
      f"Synergistic: {y_train_orig.sum()} "
      f"({y_train_orig.mean()*100:.1f}%)")
print(f"  SMOTE train    : {X_train_smote.shape} | "
      f"Synergistic: {y_train_smote.sum()} "
      f"({y_train_smote.mean()*100:.1f}%)")
print(f"  Test           : {X_test.shape} | "
      f"Synergistic: {y_test.sum()} "
      f"({y_test.mean()*100:.1f}%)")

# Scaler — fit on original train
scaler = StandardScaler()
X_train_orig_sc  = scaler.fit_transform(X_train_orig)
X_train_smote_sc = scaler.transform(X_train_smote)
X_test_sc        = scaler.transform(X_test)
with open(os.path.join(PROJECT_DIR, "scaler_v2.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print("  Scaler saved -> scaler_v2.pkl")

neg              = int((y_train_orig == 0).sum())
pos              = int((y_train_orig == 1).sum())
scale_pos_weight = neg / pos
print(f"  Original train — neg: {neg}  pos: {pos}")
print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

# Validation split for early stopping
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_orig_sc, y_train_orig,
    test_size=0.1, random_state=RANDOM_STATE,
    stratify=y_train_orig
)

# ══════════════════════════════════════════════════════════════
# STEP 2 — XGBoost on ORIGINAL train (no SMOTE)
# ══════════════════════════════════════════════════════════════
section("STEP 2: Training XGBoost on ORIGINAL train set")
print("  (No SMOTE — scale_pos_weight handles imbalance internally)")
print(f"  XGBoost version: {xgb.__version__}")

xgb_params = {
    "n_estimators"        : 600,
    "max_depth"           : 6,
    "learning_rate"       : 0.05,
    "subsample"           : 0.8,
    "colsample_bytree"    : 0.8,
    "min_child_weight"    : 3,
    "gamma"               : 0.5,
    "scale_pos_weight"    : scale_pos_weight,
    "objective"           : "binary:logistic",
    "eval_metric"         : "aucpr",
    "tree_method"         : "hist",
    "random_state"        : RANDOM_STATE,
    "n_jobs"              : -1,
    "early_stopping_rounds": 40,
}

print("\n  Parameters:")
for k, v in xgb_params.items():
    print(f"    {k}: {v}")

t0        = time.time()
xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_model.fit(
    X_tr, y_tr,
    eval_set = [(X_val, y_val)],
    verbose  = 50
)
xgb_time = time.time() - t0
print(f"\n  Training time : {xgb_time:.1f}s")
print(f"  Best iteration: "
      f"{getattr(xgb_model, 'best_iteration', 'N/A')}")

xgb_model.save_model(
    os.path.join(PROJECT_DIR, "xgboost_model_v2.json"))
print("  Saved -> xgboost_model_v2.json")

print("\n  Applying Platt scaling calibration to XGBoost...")
xgb_cal = CalibratedClassifierCV(
    xgb_model, method="sigmoid", cv="prefit")
xgb_cal.fit(X_val, y_val)
with open(os.path.join(PROJECT_DIR,
                       "xgboost_calibrated_v2.pkl"), "wb") as f:
    pickle.dump(xgb_cal, f)
print("  Saved -> xgboost_calibrated_v2.pkl")

# ══════════════════════════════════════════════════════════════
# STEP 3 — LightGBM on ORIGINAL train (no SMOTE)
# ══════════════════════════════════════════════════════════════
section("STEP 3: Training LightGBM on ORIGINAL train set")
print("  (No SMOTE — is_unbalance=True handles imbalance internally)")

lgb_params = {
    "n_estimators"     : 600,
    "max_depth"        : 6,
    "learning_rate"    : 0.05,
    "num_leaves"       : 63,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "min_child_samples": 10,
    "is_unbalance"     : True,
    "objective"        : "binary",
    "metric"           : "average_precision",
    "boosting_type"    : "gbdt",
    "random_state"     : RANDOM_STATE,
    "n_jobs"           : -1,
    "verbose"          : -1,
}

print("\n  Parameters:")
for k, v in lgb_params.items():
    print(f"    {k}: {v}")

X_tr_lgb, X_val_lgb, y_tr_lgb, y_val_lgb = train_test_split(
    X_train_orig_sc, y_train_orig,
    test_size=0.1, random_state=RANDOM_STATE,
    stratify=y_train_orig
)

t0        = time.time()
lgb_model = lgb.LGBMClassifier(**lgb_params)
lgb_model.fit(
    X_tr_lgb, y_tr_lgb,
    eval_set  = [(X_val_lgb, y_val_lgb)],
    callbacks = [
        lgb.early_stopping(stopping_rounds=40, verbose=False),
        lgb.log_evaluation(period=50),
    ]
)
lgb_time = time.time() - t0
print(f"\n  Training time: {lgb_time:.1f}s")

with open(os.path.join(PROJECT_DIR,
                       "lightgbm_model_v2.pkl"), "wb") as f:
    pickle.dump(lgb_model, f)
print("  Saved -> lightgbm_model_v2.pkl")

print("\n  Applying Platt scaling calibration to LightGBM...")
lgb_cal = CalibratedClassifierCV(
    lgb_model, method="sigmoid", cv="prefit")
lgb_cal.fit(X_val_lgb, y_val_lgb)
with open(os.path.join(PROJECT_DIR,
                       "lightgbm_calibrated_v2.pkl"), "wb") as f:
    pickle.dump(lgb_cal, f)
print("  Saved -> lightgbm_calibrated_v2.pkl")

# ══════════════════════════════════════════════════════════════
# STEP 4 — RandomForest on SMOTE train
# ══════════════════════════════════════════════════════════════
section("STEP 4: Training RandomForest on SMOTE Train Set")
print("  (RF has no scale_pos_weight — SMOTE required)")
print("  (Isotonic calibration on original dist after training)")

rf_params = {
    "n_estimators"    : 300,
    "max_depth"       : 20,
    "min_samples_leaf": 5,
    "max_features"    : "sqrt",
    "class_weight"    : "balanced",
    "random_state"    : RANDOM_STATE,
    "n_jobs"          : -1,
    "oob_score"       : True,
}

print("\n  Parameters:")
for k, v in rf_params.items():
    print(f"    {k}: {v}")

t0       = time.time()
rf_model = RandomForestClassifier(**rf_params)
rf_model.fit(X_train_smote_sc, y_train_smote)
rf_time  = time.time() - t0
print(f"\n  Training time: {rf_time:.1f}s")
print(f"  OOB Score    : {rf_model.oob_score_:.4f}")

with open(os.path.join(PROJECT_DIR,
                       "randomforest_model_v2.pkl"), "wb") as f:
    pickle.dump(rf_model, f)
print("  Saved -> randomforest_model_v2.pkl")

print("\n  Applying isotonic calibration on original distribution...")
rf_cal = CalibratedClassifierCV(
    rf_model, method="isotonic", cv="prefit")
rf_cal.fit(X_train_orig_sc, y_train_orig)
with open(os.path.join(PROJECT_DIR,
                       "randomforest_calibrated_v2.pkl"), "wb") as f:
    pickle.dump(rf_cal, f)
print("  Saved -> randomforest_calibrated_v2.pkl")

# ══════════════════════════════════════════════════════════════
# STEP 5 — Evaluate Raw vs Calibrated
# ══════════════════════════════════════════════════════════════
section("STEP 5: Evaluating All Models — Raw vs Calibrated")

raw_models  = {"XGBoost": xgb_model,
               "LightGBM": lgb_model,
               "RandomForest": rf_model}
cal_models  = {"XGBoost": xgb_cal,
               "LightGBM": lgb_cal,
               "RandomForest": rf_cal}
train_times = {"XGBoost": xgb_time,
               "LightGBM": lgb_time,
               "RandomForest": rf_time}

all_raw_metrics = []
all_cal_metrics = []
raw_curves      = {}
cal_curves      = {}

for name in ["XGBoost", "LightGBM", "RandomForest"]:
    print(f"\n  --- {name} ---")

    raw_m, raw_c = evaluate_model(
        name, raw_models[name], X_test_sc, y_test)
    raw_m["train_time_s"] = round(train_times[name], 1)
    all_raw_metrics.append(raw_m)
    raw_curves[name] = raw_c

    cal_m, cal_c = evaluate_model(
        name, cal_models[name], X_test_sc, y_test)
    cal_m["train_time_s"] = round(train_times[name], 1)
    all_cal_metrics.append(cal_m)
    cal_curves[name] = cal_c

    print(f"    {'Metric':<20} {'Raw':>10} "
          f"{'Calibrated':>12} {'Change':>10}")
    print(f"    {'-'*54}")
    for metric in ["auroc","prauc","f1_best","best_threshold"]:
        rv   = raw_m[metric]
        cv_  = cal_m[metric]
        diff = cv_ - rv
        sign = "+" if diff >= 0 else ""
        print(f"    {metric:<20} {rv:>10.4f} {cv_:>12.4f} "
              f"{sign+f'{diff:.4f}':>10}")

    cm = cal_c["cm"]
    print(f"\n    Confusion matrix "
          f"(calibrated, threshold={cal_m['best_threshold']:.3f}):")
    print(f"      TN={cm[0,0]:>6}  FP={cm[0,1]:>6}")
    print(f"      FN={cm[1,0]:>6}  TP={cm[1,1]:>6}")

    y_pred = (cal_c["prob"] >= cal_m["best_threshold"]).astype(int)
    print(f"\n    Classification report (calibrated):")
    print(classification_report(
        y_test, y_pred,
        target_names=["Non-synergistic","Synergistic"],
        zero_division=0
    ))

# ══════════════════════════════════════════════════════════════
# STEP 6 — 5-Fold Cross-Validation
# ══════════════════════════════════════════════════════════════
section("STEP 6: 5-Fold Stratified CV — XGBoost")

cv = StratifiedKFold(n_splits=5, shuffle=True,
                     random_state=RANDOM_STATE)
xgb_cv = xgb.XGBClassifier(
    n_estimators     = 400,
    max_depth        = 6,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 3,
    gamma            = 0.5,
    scale_pos_weight = scale_pos_weight,
    objective        = "binary:logistic",
    eval_metric      = "aucpr",
    tree_method      = "hist",
    random_state     = RANDOM_STATE,
    n_jobs           = -1,
)
print("  Running 5-fold CV (takes a few minutes)...")
print("  Using original train set for honest CV...")
cv_results = cross_validate(
    xgb_cv, X_train_orig_sc, y_train_orig,
    cv                 = cv,
    scoring            = {"auroc": "roc_auc",
                          "prauc": "average_precision",
                          "f1"   : "f1"},
    return_train_score = True,
    n_jobs             = 1,
    verbose            = 1
)

print(f"\n  XGBoost 5-Fold CV Results:")
print(f"  {'Metric':<8} {'Test mean':>12} {'Test std':>10} "
      f"{'Train mean':>12} {'Train std':>10} {'Gap':>8}")
print(f"  {'-'*60}")
for metric in ["auroc","prauc","f1"]:
    tm  = cv_results[f"test_{metric}"].mean()
    ts  = cv_results[f"test_{metric}"].std()
    trm = cv_results[f"train_{metric}"].mean()
    trs = cv_results[f"train_{metric}"].std()
    gap = trm - tm
    print(f"  {metric.upper():<8} {tm:>12.4f} {ts:>10.4f} "
          f"{trm:>12.4f} {trs:>10.4f} {gap:>8.4f}")

cv_df = pd.DataFrame({
    "fold"       : range(1, 6),
    "test_auroc" : cv_results["test_auroc"],
    "test_prauc" : cv_results["test_prauc"],
    "test_f1"    : cv_results["test_f1"],
    "train_auroc": cv_results["train_auroc"],
    "train_prauc": cv_results["train_prauc"],
    "train_f1"   : cv_results["train_f1"],
})
cv_df.to_csv(
    os.path.join(PROJECT_DIR, "baseline_cv_results_v2.csv"),
    index=False)
print("\n  CV results saved -> baseline_cv_results_v2.csv")

# ══════════════════════════════════════════════════════════════
# STEP 7 — Save Metrics
# ══════════════════════════════════════════════════════════════
section("STEP 7: Saving Metrics")

pd.DataFrame(all_raw_metrics).to_csv(
    os.path.join(PROJECT_DIR, "baseline_metrics_v2.csv"),
    index=False)

before_after = []
for rm, cm_ in zip(all_raw_metrics, all_cal_metrics):
    before_after.append({
        "model"               : rm["model"],
        "auroc_raw"           : rm["auroc"],
        "auroc_calibrated"    : cm_["auroc"],
        "prauc_raw"           : rm["prauc"],
        "prauc_calibrated"    : cm_["prauc"],
        "prauc_improvement"   : round(cm_["prauc"]-rm["prauc"], 4),
        "f1_raw"              : rm["f1_best"],
        "f1_calibrated"       : cm_["f1_best"],
        "threshold_raw"       : rm["best_threshold"],
        "threshold_calibrated": cm_["best_threshold"],
        "train_time_s"        : rm["train_time_s"],
    })
pd.DataFrame(before_after).to_csv(
    os.path.join(PROJECT_DIR,
                 "baseline_metrics_before_after_calibration.csv"),
    index=False)
print("  Saved -> baseline_metrics_v2.csv")
print("  Saved -> baseline_metrics_before_after_calibration.csv")

print(f"\n  -- CALIBRATED MODEL COMPARISON --")
print(f"  {'Model':<16} {'AUROC':>8} {'PRAUC':>8} "
      f"{'F1':>8} {'Threshold':>10} {'Train(s)':>9}")
print(f"  {'-'*65}")
for m in all_cal_metrics:
    print(f"  {m['model']:<16} {m['auroc']:>8.4f} "
          f"{m['prauc']:>8.4f} {m['f1_best']:>8.4f} "
          f"{m['best_threshold']:>10.3f} "
          f"{m['train_time_s']:>9.1f}s")

best_model = max(all_cal_metrics, key=lambda x: x["prauc"])
print(f"\n  Best model by PRAUC (calibrated): {best_model['model']}")

# ══════════════════════════════════════════════════════════════
# STEP 8 — Plots
# ══════════════════════════════════════════════════════════════
section("STEP 8: Generating Plots")

COLORS = {
    "XGBoost"      : "#185FA5",
    "LightGBM"     : "#1D9E75",
    "RandomForest" : "#BA7517",
}
synergy_prev = y_test.mean()

# Plot 1 — ROC curves
fig, ax = plt.subplots(figsize=(7,6))
ax.plot([0,1],[0,1],"k--",lw=0.8,label="Random (AUROC=0.50)")
for name, curves in cal_curves.items():
    fpr, tpr, auroc = curves["roc"]
    ax.plot(fpr, tpr, color=COLORS[name], lw=2,
            label=f"{name} (AUROC={auroc:.3f})")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — Calibrated Models\n"
             "Test set: original 2.2% synergistic", fontsize=11)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR,
                         "baseline_roc_curves_v2.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> baseline_roc_curves_v2.png")

# Plot 2 — PR curves
fig, ax = plt.subplots(figsize=(7,6))
ax.axhline(y=synergy_prev, color="k", ls="--", lw=0.8,
           label=f"Random baseline ({synergy_prev:.3f})")
for name, curves in cal_curves.items():
    r, p, prauc = curves["pr"]
    ax.plot(r, p, color=COLORS[name], lw=2,
            label=f"{name} (PRAUC={prauc:.3f})")
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curves — Calibrated Models\n"
             "Primary metric: PRAUC", fontsize=11)
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR,
                         "baseline_pr_curves_v2.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> baseline_pr_curves_v2.png")

# Plot 3 — Calibration improvement
fig, axes = plt.subplots(1,2,figsize=(11,5))
model_names = [m["model"] for m in all_raw_metrics]
x = np.arange(len(model_names)); w = 0.35
for ax, metric, ylabel, title in zip(
    axes,
    ["prauc","f1_best"],
    ["PRAUC","F1 (best threshold)"],
    ["PRAUC before vs after calibration",
     "F1 before vs after calibration"]
):
    raw_vals = [m[metric] for m in all_raw_metrics]
    cal_vals = [m[metric] for m in all_cal_metrics]
    b1 = ax.bar(x-w/2, raw_vals, w,
                label="Raw", color="#B4B2A9", alpha=0.85)
    b2 = ax.bar(x+w/2, cal_vals, w,
                label="Calibrated", color="#185FA5", alpha=0.85)
    for bar, v in list(zip(b1,raw_vals))+list(zip(b2,cal_vals)):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim([0, max(max(raw_vals),max(cal_vals))*1.2])
fig.suptitle("Impact of probability calibration", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR,
                         "baseline_calibration_improvement.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> baseline_calibration_improvement.png")

# Plot 4 — Confusion matrices
fig, axes = plt.subplots(1,3,figsize=(14,4))
for ax, (name,curves), cal_m in zip(
        axes, cal_curves.items(), all_cal_metrics):
    cm = curves["cm"]
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(
        f"{name}\n(threshold={cal_m['best_threshold']:.3f})",
        fontsize=10)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Non-syn","Syn"], fontsize=9)
    ax.set_yticklabels(["Non-syn","Syn"], fontsize=9)
    ax.set_ylabel("True label", fontsize=9)
    ax.set_xlabel("Predicted label", fontsize=9)
    thresh_v = cm.max()/2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i,j], "d"),
                    ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if cm[i,j]>thresh_v else "black")
fig.suptitle("Confusion matrices — calibrated models",
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR,
                         "baseline_confusion_matrices_v2.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> baseline_confusion_matrices_v2.png")

# Plot 5 — Feature importance XGBoost top 30
importances = xgb_model.feature_importances_
indices     = np.argsort(importances)[::-1][:30]
top_names   = [common_features[i] for i in indices]
top_vals    = importances[indices]
fig, ax = plt.subplots(figsize=(9,7))
ax.barh(range(30), top_vals[::-1],
        color="#185FA5", alpha=0.85)
ax.set_yticks(range(30))
ax.set_yticklabels(top_names[::-1], fontsize=8)
ax.set_xlabel("Feature Importance (gain)", fontsize=11)
ax.set_title("XGBoost — Top 30 Feature Importances",
             fontsize=11)
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR,
                         "baseline_feature_importance_v2.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> baseline_feature_importance_v2.png")

# Plot 6 — Threshold analysis
fig, axes = plt.subplots(1,3,figsize=(14,4))
thresholds = np.arange(0.01, 0.99, 0.005)
for ax, (name,curves), cal_m in zip(
        axes, cal_curves.items(), all_cal_metrics):
    y_prob = curves["prob"]
    f1s  = [f1_score(y_test,(y_prob>=t).astype(int),
                     zero_division=0) for t in thresholds]
    prcs = [precision_score(y_test,(y_prob>=t).astype(int),
                            zero_division=0) for t in thresholds]
    recs = [recall_score(y_test,(y_prob>=t).astype(int),
                         zero_division=0) for t in thresholds]
    ax.plot(thresholds, f1s,  color="#185FA5", lw=2,
            label="F1")
    ax.plot(thresholds, prcs, color="#1D9E75", lw=1.5,
            label="Precision", ls="--")
    ax.plot(thresholds, recs, color="#BA7517", lw=1.5,
            label="Recall", ls=":")
    ax.axvline(x=cal_m["best_threshold"], color="red", lw=1,
               ls="--",
               label=f"Best ({cal_m['best_threshold']:.3f})")
    ax.set_xlabel("Threshold", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(f"{name}", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
fig.suptitle(
    "F1/Precision/Recall vs threshold — calibrated models",
    fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR,
                         "baseline_threshold_analysis_v2.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> baseline_threshold_analysis_v2.png")

# Plot 7 — Metrics comparison
metric_cols   = ["auroc","prauc","f1_best"]
metric_labels = ["AUROC","PRAUC (primary)","F1 (best thr)"]
x = np.arange(len(metric_cols)); w = 0.25
fig, ax = plt.subplots(figsize=(9,5))
for i, (name, color) in enumerate(COLORS.items()):
    vals = [all_cal_metrics[i][m] for m in metric_cols]
    bars = ax.bar(x+i*w, vals, w,
                  label=name, color=color, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.005,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=8)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Calibrated model comparison", fontsize=12)
ax.set_xticks(x+w)
ax.set_xticklabels(metric_labels, fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim([0,1.05])
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR,
                         "baseline_metrics_comparison_v2.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> baseline_metrics_comparison_v2.png")

# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
section("FINAL SUMMARY")

print(f"  Training setup:")
print(f"    XGBoost     : original train "
      f"({X_train_orig.shape[0]} rows) + "
      f"scale_pos_weight={scale_pos_weight:.1f}")
print(f"    LightGBM    : original train "
      f"({X_train_orig.shape[0]} rows) + is_unbalance=True")
print(f"    RandomForest: SMOTE train "
      f"({X_train_smote.shape[0]} rows) + class_weight=balanced")
print(f"    Calibration : Platt (XGB/LGBM) | "
      f"Isotonic (RF) via CalibratedClassifierCV")

print(f"\n  -- Calibrated test set performance --")
print(f"  {'Model':<16} {'AUROC':>8} {'PRAUC':>8} "
      f"{'F1':>8} {'Threshold':>10} {'TP':>6} {'FP':>6}")
print(f"  {'-'*65}")
for m, c in zip(all_cal_metrics, cal_curves.values()):
    cm_ = c["cm"]
    print(f"  {m['model']:<16} {m['auroc']:>8.4f} "
          f"{m['prauc']:>8.4f} {m['f1_best']:>8.4f} "
          f"{m['best_threshold']:>10.3f} "
          f"{cm_[1,1]:>6} {cm_[0,1]:>6}")

print(f"\n  Best model by PRAUC: {best_model['model']} "
      f"(PRAUC={best_model['prauc']:.4f})")

print(f"\n  -- v1 vs v2 PRAUC improvement --")
v1_prauc = {"XGBoost":0.0759,
            "LightGBM":0.0875,
            "RandomForest":0.1226}
for m in all_cal_metrics:
    v1 = v1_prauc[m["model"]]
    v2 = m["prauc"]
    print(f"  {m['model']:<16}  v1={v1:.4f}  "
          f"v2={v2:.4f}  change={v2-v1:+.4f}")

print(f"\n  -- File checklist --")
files = [
    "xgboost_model_v2.json",
    "lightgbm_model_v2.pkl",
    "randomforest_model_v2.pkl",
    "xgboost_calibrated_v2.pkl",
    "lightgbm_calibrated_v2.pkl",
    "randomforest_calibrated_v2.pkl",
    "scaler_v2.pkl",
    "baseline_metrics_v2.csv",
    "baseline_metrics_before_after_calibration.csv",
    "baseline_cv_results_v2.csv",
    "baseline_roc_curves_v2.png",
    "baseline_pr_curves_v2.png",
    "baseline_feature_importance_v2.png",
    "baseline_confusion_matrices_v2.png",
    "baseline_metrics_comparison_v2.png",
    "baseline_calibration_improvement.png",
    "baseline_threshold_analysis_v2.png",
]
for fname in files:
    path   = os.path.join(PROJECT_DIR, fname)
    status = "[OK]" if os.path.exists(path) else "[MISSING]"
    print(f"    {status}  {fname}")

print(f"""
  Notes for Dr. Atallah:
  1. v1 used SMOTE + scale_pos_weight — conflicting signals
     caused miscalibrated probabilities (test PRAUC 0.076)
  2. v2 fix: XGB/LGBM trained on ORIGINAL distribution with
     scale_pos_weight/is_unbalance — no SMOTE conflict
  3. RF kept SMOTE (no built-in imbalance param) but calibrated
     with isotonic regression on original distribution data
  4. CalibratedClassifierCV satisfies PDF requirement:
     "Class imbalance handling — threshold moving"
  5. CV PRAUC reported alongside test PRAUC for honest estimate
  6. All v2 models saved — use *_calibrated_v2.pkl in SHAP
""")

print("="*60)
print("  STEP 6 v2 FINAL — COMPLETE!")
print("  Next -> Step 7: SHAP + LIME + Tissue Stratification")
print("="*60)

