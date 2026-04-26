"""
TF-IDF + Logistic Regression — VERSION 3
HASOC 2019 + 2020 English + Hindi (task_1: HOF / NOT)

CHANGE FROM v2 → v3: One targeted fix for Hindi performance.

  Analysis of why v2 didn't improve Hindi over v1:
    strip_accents="unicode" strips Devanagari vowel matras but KEEPS consonant
    base characters, so "बेवकूफ" → "बवकफ" — different token, but CONSISTENT
    between train and test. The model learned the consonant-skeleton pattern.
    So strip_accents removal gave near-zero gain on Hindi specifically.

  What actually limits LR on Hindi:
    The combined EN+HI corpus has ~9560 English and ~7628 Hindi train samples.
    Hindi-specific HOF words are rare relative to English. With min_df=2,
    many Hindi abuse words appearing only once get dropped. They then compete
    with English for the same 80K word-feature slots, and English n-grams win
    by frequency.

  Fix (v3): Add a dedicated Devanagari word vectorizer (3rd branch):
    - Extracts ONLY Devanagari tokens (regex pre-filter before TF-IDF)
    - min_df=1 — no Devanagari word is dropped regardless of rarity
    - ngram_range=(1,2) — word + bigram for Hindi phrase patterns
    - max_features=30K — dedicated budget, doesn't compete with English
    This gives rare Hindi HOF words their own feature space with guaranteed
    representation.

Outputs:
    model3/lr_v3/model.pkl
    model3/lr_v3/vectorizer.pkl
    model3/lr_v3/report.json
    model3/plots/lr_v3_*.png

Run:
    conda activate test2
    python train3/train_model_v3.py
"""

import os, re, sys, pickle, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

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
LR_DIR          = os.path.join(MODEL_DIR, "lr_v3")
MODEL_PATH      = os.path.join(LR_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(LR_DIR, "vectorizer.pkl")
PLOTS_DIR       = os.path.join(MODEL_DIR, "plots")
REPORT_PATH     = os.path.join(LR_DIR, "report.json")
os.makedirs(LR_DIR, exist_ok=True); os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Devanagari extractor transformer ───────────────────────────────────────────
_DEVA_RE = re.compile(r'[\u0900-\u097F]+')

class DevanagariExtractor(BaseEstimator, TransformerMixin):
    """Extracts only Devanagari tokens from text, rejoins as string.
    Passed to a dedicated TfidfVectorizer so Hindi words get their own
    feature space without competing with English for vocabulary slots."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return [' '.join(_DEVA_RE.findall(t)) or ' ' for t in X]

# ── preprocessing ────────────────────────────────────────────────────────────────
def clean(text: str) -> str:
    return preprocess_lr(str(text))

# ── loaders ──────────────────────────────────────────────────────────────────────
def _normalise(df):
    df.columns = df.columns.str.strip()
    if "task1" in df.columns and "task_1" not in df.columns:
        df = df.rename(columns={"task1": "task_1"})
    df = df[["text", "task_1"]].dropna()
    df["task_1"] = df["task_1"].str.strip().str.upper().replace("NNOT", "NOT")
    df = df[df["task_1"].isin(["HOF", "NOT"])]
    df["text"] = df["text"].apply(clean)
    return df

def load_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"): return _normalise(pd.read_excel(path, dtype=str))
    elif ext == ".csv":          return _normalise(pd.read_csv(path, dtype=str))
    else:                        return _normalise(pd.read_csv(path, sep="\t", dtype=str))

# ── load data ─────────────────────────────────────────────────────────────────────
print("Loading datasets (HASOC 2019 + 2020)...")
train_frames, test_frames, dataset_stats = [], [], []
for path, bucket in [
    (EN_TRAIN_2019, train_frames), (HI_TRAIN_2019, train_frames),
    (EN_TRAIN_2020, train_frames), (HI_TRAIN_2020, train_frames),
    (EN_TEST_2019,  test_frames),  (HI_TEST_2019,  test_frames),
    (EN_TEST_2020,  test_frames),  (HI_TEST_2020,  test_frames),
]:
    if os.path.exists(path):
        df = load_any(path)
        bucket.append(df)
        hof, not_ = int((df["task_1"]=="HOF").sum()), int((df["task_1"]=="NOT").sum())
        print(f"  {os.path.basename(path)}: {len(df)} rows  HOF={hof}  NOT={not_}")
        dataset_stats.append({"file": os.path.basename(path), "total": len(df),
                               "HOF": hof, "NOT": not_,
                               "split": "train" if bucket is train_frames else "test"})

if not train_frames: sys.exit("No training data found.")

train_df = pd.concat(train_frames, ignore_index=True)
test_df  = pd.concat(test_frames,  ignore_index=True) if test_frames else None
print(f"\nTrain: {len(train_df)}  HOF={(train_df['task_1']=='HOF').sum()}  NOT={(train_df['task_1']=='NOT').sum()}")
if test_df is not None:
    print(f"Test : {len(test_df)}   HOF={(test_df['task_1']=='HOF').sum()}  NOT={(test_df['task_1']=='NOT').sum()}")

X_train = train_df["text"].tolist()
y_train = train_df["task_1"].tolist()
X_test  = test_df["text"].tolist()  if test_df is not None else X_train
y_test  = test_df["task_1"].tolist() if test_df is not None else y_train

# ── build pipeline ─────────────────────────────────────────────────────────────
print("\nBuilding feature pipeline (word + char + devanagari)...")

# Branch 1: mixed EN+HI+Hinglish word n-grams (from v2)
word_vec = TfidfVectorizer(
    analyzer="word", ngram_range=(1, 2), max_features=80000,
    sublinear_tf=True, min_df=2,
)

# Branch 2: character n-grams for phonetic/Hinglish variants (from v2)
char_vec = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(2, 5), max_features=60000,
    sublinear_tf=True, min_df=3,
)

# Branch 3 (NEW): Devanagari-only word n-grams, min_df=1
# No rare Hindi word is discarded. Dedicated 30K slots so Hindi doesn't
# compete with English for vocabulary budget.
deva_vec = Pipeline([
    ("extract", DevanagariExtractor()),
    ("tfidf",   TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), max_features=30000,
        sublinear_tf=True, min_df=1,
    )),
])

features = FeatureUnion([("word", word_vec), ("char", char_vec), ("deva", deva_vec)])

pipeline = Pipeline([
    ("features", features),
    ("clf", LogisticRegression(
        class_weight="balanced", max_iter=2000,
        solver="saga", random_state=42,
    )),
])

param_grid = {"clf__C": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print("Running GridSearchCV (3-fold, f1_macro)...")
gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)

best_pipeline = gs.best_estimator_
print(f"\nBest C      : {gs.best_params_['clf__C']}")
print(f"CV f1_macro : {gs.best_score_:.4f}")

cv_df = pd.DataFrame(gs.cv_results_)[["param_clf__C", "mean_test_score", "std_test_score"]]
cv_df.columns = ["C", "f1_macro_mean", "f1_macro_std"]
print("\nCV results:")
print(cv_df.to_string(index=False))

# ── evaluate ──────────────────────────────────────────────────────────────────
y_pred  = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)
acc     = accuracy_score(y_test, y_pred)
f1mac   = f1_score(y_test, y_pred, average="macro")
cr      = classification_report(y_test, y_pred, target_names=["HOF", "NOT"], output_dict=True)

print(f"\nTest Accuracy : {acc:.4f}")
print(f"F1-macro      : {f1mac:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["HOF", "NOT"]))

# ── plots ──────────────────────────────────────────────────────────────────────
def savefig(name):
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved plot -> {path}")

print("\nGenerating plots...")
fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(cv_df["C"].astype(float), cv_df["f1_macro_mean"],
            yerr=cv_df["f1_macro_std"], marker="o", color="#5b9bd5", linewidth=2, capsize=4)
ax.axvline(gs.best_params_["clf__C"], color="#e05c5c", linestyle="--",
           label=f"Best C={gs.best_params_['clf__C']}")
ax.set_xscale("log"); ax.set_xlabel("C (log scale)"); ax.set_ylabel("CV F1-macro")
ax.set_title("GridSearchCV: C vs F1-macro (LR v3)", fontsize=12)
ax.legend(); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); savefig("lr_v3_gridsearch_cv.png")

cm = confusion_matrix(y_test, y_pred, labels=["HOF", "NOT"])
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["HOF","NOT"],
            yticklabels=["HOF","NOT"], ax=ax, linewidths=0.5)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix (LR v3)", fontsize=12)
plt.tight_layout(); savefig("lr_v3_confusion_matrix.png")

y_bin   = (pd.Series(y_test) == "HOF").astype(int).values
hof_idx = list(best_pipeline.classes_).index("HOF")
fpr, tpr, _ = roc_curve(y_bin, y_proba[:, hof_idx])
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#5b9bd5", lw=2, label=f"AUC = {roc_auc:.3f}")
ax.plot([0,1],[0,1],"k--",lw=1); ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate"); ax.set_title("ROC Curve – HOF (LR v3)", fontsize=12)
ax.legend(loc="lower right"); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); savefig("lr_v3_roc_curve.png")

# ── save ───────────────────────────────────────────────────────────────────────
report = {
    "version": "lr_v3",
    "changes_from_v2": ["Added DevanagariExtractor + dedicated TF-IDF branch (min_df=1, 30K features)"],
    "dataset": dataset_stats,
    "train_total": len(train_df), "test_total": len(y_test),
    "best_C": gs.best_params_["clf__C"],
    "cv_f1_macro": round(gs.best_score_, 4),
    "test_accuracy": round(acc, 4), "test_f1_macro": round(f1mac, 4),
    "roc_auc_HOF": round(roc_auc, 4),
    "classification_report": cr,
    "confusion_matrix": {"HOF_as_HOF": int(cm[0,0]), "HOF_as_NOT": int(cm[0,1]),
                         "NOT_as_HOF": int(cm[1,0]), "NOT_as_NOT": int(cm[1,1])},
    "cv_results": cv_df.to_dict(orient="records"),
}
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(best_pipeline, f, protocol=4)
with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(best_pipeline.named_steps["features"], f, protocol=4)

print(f"\n  Saved model  -> {MODEL_PATH}")
print(f"  Saved vec    -> {VECTORIZER_PATH}")
print(f"  Saved report -> {REPORT_PATH}")
print("\nDone! LR v3 trained on HASOC 2019 + 2020 combined data.")
