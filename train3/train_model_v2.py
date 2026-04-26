"""
Improved TF-IDF + Logistic Regression classifier — VERSION 2
HASOC 2019 + 2020 English + Hindi hate-speech detection (task_1: HOF / NOT)

CHANGES FROM train_model.py (v1):

  Fix 1 (CRITICAL): Removed strip_accents="unicode" from both TfidfVectorizers.
    strip_accents="unicode" calls unicodedata.normalize('NFKD') which decomposes
    Devanagari script into base+diacritic pairs, then strips the diacritics entirely.
    Result: Hindi words silently became garbage tokens that never matched between
    train and test. This killed recall on Hindi HOF samples.

  Fix 2: Word n-gram range (1,3) → (1,2)
    Trigrams on 17K samples are mostly hapax-like (appear 1-2 times → min_df drops them).
    They inflate the feature space without providing generalizable signal.
    Bigrams already capture most phrase-level patterns ("hate speech", "kill them").

  Fix 3: Expanded C grid [0.1 → 50.0] with finer resolution
    v1 best was often C=5 or C=10 — the grid stopped too early.
    New grid: [0.1, 0.3, 1.0, 3.0, 10.0, 30.0] covers the full range.

  Fix 4: Increased char_wb max_features 40K → 60K
    Character n-grams are the primary signal for Hinglish (handles spelling variation).
    More features → better coverage of phonetic variants (chutiya/chootiya/chutia).

  Fix 5: Increased word max_features 60K → 80K
    Combined vocab (EN+HI+Hinglish) is large; 60K truncated many Hindi word forms.

  No change: solver=saga, max_iter=2000, class_weight='balanced', random_state=42

Outputs:
    model3/lr_v2/model.pkl
    model3/lr_v2/vectorizer.pkl
    model3/lr_v2/report.json
    model3/plots/lr_v2_*.png

Run:
    conda activate test2
    python train3/train_model_v2.py
"""

import os
import re
import sys
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.prep2 import preprocess_lr

# ── paths ───────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EN_TRAIN_2019 = os.path.join(BASE_DIR, "english_dataset", "english_dataset", "english_dataset.tsv")
EN_TEST_2019  = os.path.join(BASE_DIR, "english_dataset", "english_dataset", "hasoc2019_en_test-2919.tsv")
HI_TRAIN_2019 = os.path.join(BASE_DIR, "hindi_dataset",  "hindi_dataset",  "hindi_dataset.tsv")
HI_TEST_2019  = os.path.join(BASE_DIR, "hindi_dataset",  "hindi_dataset",  "hasoc2019_hi_test_gold_2919.tsv")

EN_TRAIN_2020 = os.path.join(BASE_DIR, "english_dataset_2020", "hasoc_2020_en_train.xlsx")
EN_TEST_2020  = os.path.join(BASE_DIR, "english_dataset_2020", "english_test_1509.csv")
HI_TRAIN_2020 = os.path.join(BASE_DIR, "hindi_dataset_2020",  "hasoc_2020_hi_train.xlsx")
HI_TEST_2020  = os.path.join(BASE_DIR, "hindi_dataset_2020",  "hindi_test_1509.csv")

MODEL_DIR       = os.path.join(BASE_DIR, "model3")
LR_DIR          = os.path.join(MODEL_DIR, "lr_v2")
MODEL_PATH      = os.path.join(LR_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(LR_DIR, "vectorizer.pkl")
PLOTS_DIR       = os.path.join(MODEL_DIR, "plots")
REPORT_PATH     = os.path.join(LR_DIR, "report.json")

os.makedirs(LR_DIR,    exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── preprocessing ───────────────────────────────────────────────────────────────
def clean(text: str) -> str:
    return preprocess_lr(str(text))

# ── loaders ─────────────────────────────────────────────────────────────────────
def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    if "task1" in df.columns and "task_1" not in df.columns:
        df = df.rename(columns={"task1": "task_1"})
    df = df[["text", "task_1"]].dropna()
    df["task_1"] = df["task_1"].str.strip().str.upper().replace("NNOT", "NOT")
    df = df[df["task_1"].isin(["HOF", "NOT"])]
    df["text"] = df["text"].apply(clean)
    return df

def load_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return _normalise(pd.read_excel(path, dtype=str))
    elif ext == ".csv":
        return _normalise(pd.read_csv(path, dtype=str))
    else:
        return _normalise(pd.read_csv(path, sep="\t", dtype=str))

# ── load data ────────────────────────────────────────────────────────────────────
print("Loading datasets (HASOC 2019 + 2020)...")
train_frames, test_frames = [], []
dataset_stats = []

for path, bucket in [
    (EN_TRAIN_2019, train_frames), (HI_TRAIN_2019, train_frames),
    (EN_TRAIN_2020, train_frames), (HI_TRAIN_2020, train_frames),
    (EN_TEST_2019,  test_frames),  (HI_TEST_2019,  test_frames),
    (EN_TEST_2020,  test_frames),  (HI_TEST_2020,  test_frames),
]:
    if os.path.exists(path):
        df = load_any(path)
        bucket.append(df)
        hof  = int((df["task_1"] == "HOF").sum())
        not_ = int((df["task_1"] == "NOT").sum())
        print(f"  {os.path.basename(path)}: {len(df)} rows  HOF={hof}  NOT={not_}")
        dataset_stats.append({
            "file": os.path.basename(path), "total": len(df),
            "HOF": hof, "NOT": not_,
            "split": "train" if bucket is train_frames else "test",
        })

if not train_frames:
    sys.exit("No training data found.")

train_df = pd.concat(train_frames, ignore_index=True)
test_df  = pd.concat(test_frames,  ignore_index=True) if test_frames else None

print(f"\nTrain: {len(train_df)}  HOF={(train_df['task_1']=='HOF').sum()}  NOT={(train_df['task_1']=='NOT').sum()}")
if test_df is not None:
    print(f"Test : {len(test_df)}   HOF={(test_df['task_1']=='HOF').sum()}  NOT={(test_df['task_1']=='NOT').sum()}")

X_train = train_df["text"].tolist()
y_train = train_df["task_1"].tolist()
X_test  = test_df["text"].tolist()  if test_df is not None else X_train
y_test  = test_df["task_1"].tolist() if test_df is not None else y_train

# ── build pipeline ───────────────────────────────────────────────────────────────
print("\nBuilding feature pipeline...")

# FIX 1: strip_accents REMOVED — it destroys Devanagari script.
# FIX 2: word ngram_range (1,3) → (1,2) — trigrams on 17K samples are mostly noise.
# FIX 5: word max_features 60K → 80K — larger EN+HI+Hinglish combined vocab.
word_vec = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=80000,
    sublinear_tf=True,
    min_df=2,
    # strip_accents intentionally omitted — would destroy Devanagari
)

# FIX 1: strip_accents REMOVED here too.
# FIX 4: char max_features 40K → 60K — char n-grams are main signal for Hinglish.
char_vec = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5),
    max_features=60000,
    sublinear_tf=True,
    min_df=3,
    # strip_accents intentionally omitted
)

features = FeatureUnion([("word", word_vec), ("char", char_vec)])

pipeline = Pipeline([
    ("features", features),
    ("clf", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="saga",
        random_state=42,
    )),
])

# FIX 3: Expanded C grid — v1's grid stopped at 10 but optimum was often at the edge.
param_grid = {"clf__C": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print("Running GridSearchCV (3-fold, f1_macro)...")
gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1_macro",
                  n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)

best_pipeline = gs.best_estimator_
print(f"\nBest C      : {gs.best_params_['clf__C']}")
print(f"CV f1_macro : {gs.best_score_:.4f}")

cv_df = pd.DataFrame(gs.cv_results_)[["param_clf__C", "mean_test_score", "std_test_score"]]
cv_df.columns = ["C", "f1_macro_mean", "f1_macro_std"]
print("\nCV results:")
print(cv_df.to_string(index=False))

# ── evaluate ─────────────────────────────────────────────────────────────────────
y_pred  = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)
acc     = accuracy_score(y_test, y_pred)
f1mac   = f1_score(y_test, y_pred, average="macro")
cr      = classification_report(y_test, y_pred, target_names=["HOF", "NOT"], output_dict=True)

print(f"\nTest Accuracy : {acc:.4f}")
print(f"F1-macro      : {f1mac:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["HOF", "NOT"]))

# ── plots ─────────────────────────────────────────────────────────────────────────
def savefig(name: str):
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved plot -> {path}")

print("\nGenerating plots...")

# GridSearchCV C vs F1
fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(cv_df["C"].astype(float), cv_df["f1_macro_mean"],
            yerr=cv_df["f1_macro_std"], marker="o", color="#5b9bd5", linewidth=2, capsize=4)
ax.axvline(gs.best_params_["clf__C"], color="#e05c5c", linestyle="--",
           label=f"Best C={gs.best_params_['clf__C']}")
ax.set_xscale("log"); ax.set_xlabel("C (log scale)"); ax.set_ylabel("CV F1-macro")
ax.set_title("GridSearchCV: C vs F1-macro (LR v2)", fontsize=12)
ax.legend(); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); savefig("lr_v2_gridsearch_cv.png")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["HOF", "NOT"])
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["HOF","NOT"],
            yticklabels=["HOF","NOT"], ax=ax, linewidths=0.5)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix (LR v2)", fontsize=12)
plt.tight_layout(); savefig("lr_v2_confusion_matrix.png")

# ROC curve
y_bin   = (pd.Series(y_test) == "HOF").astype(int).values
hof_idx = list(best_pipeline.classes_).index("HOF")
fpr, tpr, _ = roc_curve(y_bin, y_proba[:, hof_idx])
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#5b9bd5", lw=2, label=f"AUC = {roc_auc:.3f}")
ax.plot([0,1],[0,1],"k--",lw=1); ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate"); ax.set_title("ROC Curve – HOF (LR v2)", fontsize=12)
ax.legend(loc="lower right"); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); savefig("lr_v2_roc_curve.png")

# Precision-recall
prec, rec, _ = precision_recall_curve(y_bin, y_proba[:, hof_idx])
pr_auc = auc(rec, prec)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec, prec, color="#e05c5c", lw=2, label=f"AUC = {pr_auc:.3f}")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall – HOF (LR v2)", fontsize=12)
ax.legend(loc="upper right"); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); savefig("lr_v2_precision_recall.png")

# ── save report ──────────────────────────────────────────────────────────────────
report = {
    "version": "lr_v2",
    "changes_from_v1": [
        "strip_accents removed (was destroying Devanagari)",
        "word ngram_range (1,3) -> (1,2)",
        "word max_features 60K -> 80K",
        "char max_features 40K -> 60K",
        "C grid extended to [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]",
    ],
    "dataset": dataset_stats,
    "train_total": len(train_df),
    "test_total": len(y_test),
    "best_C": gs.best_params_["clf__C"],
    "cv_f1_macro": round(gs.best_score_, 4),
    "test_accuracy": round(acc, 4),
    "test_f1_macro": round(f1mac, 4),
    "roc_auc_HOF": round(roc_auc, 4),
    "pr_auc_HOF":  round(pr_auc, 4),
    "classification_report": cr,
    "confusion_matrix": {
        "HOF_as_HOF": int(cm[0,0]), "HOF_as_NOT": int(cm[0,1]),
        "NOT_as_HOF": int(cm[1,0]), "NOT_as_NOT": int(cm[1,1]),
    },
    "cv_results": cv_df.to_dict(orient="records"),
}
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# ── save artifacts ────────────────────────────────────────────────────────────────
with open(MODEL_PATH, "wb") as f:
    pickle.dump(best_pipeline, f, protocol=4)
with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(best_pipeline.named_steps["features"], f, protocol=4)

print(f"\n  Saved model  -> {MODEL_PATH}")
print(f"  Saved vec    -> {VECTORIZER_PATH}")
print(f"  Saved report -> {REPORT_PATH}")
print("\nDone! LR v2 trained on HASOC 2019 + 2020 combined data.")
