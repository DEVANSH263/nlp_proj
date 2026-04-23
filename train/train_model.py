"""
Improved TF-IDF (word + char n-grams) + Logistic Regression classifier.
HASOC 2019 English + Hindi hate-speech detection (task_1: HOF / NOT).

Improvements over baseline:
  - Word n-grams (1-3) + character n-grams (2-5) via FeatureUnion
  - Repeated-character normalisation in text cleaning
  - GridSearchCV over C values (3-fold, scoring=f1_macro)
  - Full sklearn Pipeline saved as model.pkl

Run:
    python train/train_model.py

Outputs:
    model/model.pkl        (full Pipeline: FeatureUnion → LogisticRegression)
    model/vectorizer.pkl   (FeatureUnion only, kept for backward compatibility)
"""

import os
import re
import sys
import pickle
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # no display needed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
    precision_recall_curve
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EN_TRAIN = os.path.join(BASE_DIR, "english_dataset", "english_dataset", "english_dataset.tsv")
EN_TEST  = os.path.join(BASE_DIR, "english_dataset", "english_dataset", "hasoc2019_en_test-2919.tsv")
HI_TRAIN = os.path.join(BASE_DIR, "hindi_dataset", "hindi_dataset", "hindi_dataset.tsv")
HI_TEST  = os.path.join(BASE_DIR, "hindi_dataset", "hindi_dataset", "hasoc2019_hi_test_gold_2919.tsv")

MODEL_DIR       = os.path.join(BASE_DIR, "model")
LR_DIR          = os.path.join(MODEL_DIR, "lr")
MODEL_PATH      = os.path.join(LR_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(LR_DIR, "vectorizer.pkl")
PLOTS_DIR       = os.path.join(MODEL_DIR, "plots")
REPORT_PATH     = os.path.join(LR_DIR, "report.json")

os.makedirs(LR_DIR,   exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── enhanced text cleaning ─────────────────────────────────────────────────────
_REPEAT_RE = re.compile(r'(.)\1{2,}')  # collapse 3+ repeated chars → 2

def clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)           # remove URLs
    text = re.sub(r"@\w+", "", text)                      # remove @mentions
    text = re.sub(r"#(\w+)", r"\1", text)                 # strip # keep word
    # keep Devanagari (U+0900–U+097F) and standard word chars + spaces
    text = re.sub(r"[^\w\s\u0900-\u097F]", " ", text)
    text = _REPEAT_RE.sub(r"\1\1", text)                  # haaate → haate
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── load one TSV ───────────────────────────────────────────────────────────────
def load_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = df.columns.str.strip()
    df = df[["text", "task_1"]].dropna()
    df["task_1"] = df["task_1"].str.strip().str.upper()
    df["task_1"] = df["task_1"].replace("NNOT", "NOT")
    df = df[df["task_1"].isin(["HOF", "NOT"])]
    df["text"] = df["text"].apply(clean)
    return df

# ── load all splits ────────────────────────────────────────────────────────────
print("Loading datasets...")

train_frames, test_frames = [], []
dataset_stats = []   # for report

for path, split_list in [
    (EN_TRAIN, train_frames), (HI_TRAIN, train_frames),
    (EN_TEST,  test_frames),  (HI_TEST,  test_frames),
]:
    if os.path.exists(path):
        df = load_tsv(path)
        split_list.append(df)
        hof = int((df["task_1"] == "HOF").sum())
        not_ = int((df["task_1"] == "NOT").sum())
        print(f"  {os.path.basename(path)}: {len(df)} rows  HOF={hof}  NOT={not_}")
        dataset_stats.append({
            "file": os.path.basename(path),
            "total": len(df),
            "HOF": hof,
            "NOT": not_,
            "split": "train" if split_list is train_frames else "test",
        })
    else:
        print(f"  [SKIP] {path} not found")

if not train_frames:
    sys.exit("No training data found.")

train_df = pd.concat(train_frames, ignore_index=True)
print(f"\nCombined train: {len(train_df)} rows  "
      f"HOF={(train_df['task_1']=='HOF').sum()}  NOT={(train_df['task_1']=='NOT').sum()}")

if test_frames:
    test_df = pd.concat(test_frames, ignore_index=True)
    print(f"Combined test : {len(test_df)} rows")
    X_train = train_df["text"].tolist()
    y_train = train_df["task_1"].tolist()
    X_test  = test_df["text"].tolist()
    y_test  = test_df["task_1"].tolist()
else:
    print("No test files found — using 80/20 split")
    X_train, X_test, y_train, y_test = train_test_split(
        train_df["text"].tolist(), train_df["task_1"].tolist(),
        test_size=0.2, random_state=42, stratify=train_df["task_1"]
    )

# ── build pipeline: word n-grams + char n-grams ────────────────────────────────
print("\nBuilding feature pipeline...")

word_vec = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=60000,
    sublinear_tf=True,
    min_df=2,
    strip_accents="unicode",
)

char_vec = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5),
    max_features=40000,
    sublinear_tf=True,
    min_df=3,
    strip_accents="unicode",
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

# ── hyperparameter search ──────────────────────────────────────────────────────
param_grid = {"clf__C": [0.1, 0.5, 1.0, 5.0, 10.0]}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print("Running GridSearchCV (3-fold, f1_macro)...")
gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1_macro",
                  n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)

best_pipeline = gs.best_estimator_
print(f"\nBest C : {gs.best_params_['clf__C']}")
print(f"CV f1_macro : {gs.best_score_:.4f}")

# print CV results table
cv_results = pd.DataFrame(gs.cv_results_)[["param_clf__C", "mean_test_score", "std_test_score"]]
cv_results.columns = ["C", "f1_macro_mean", "f1_macro_std"]
print("\nCV results:")
print(cv_results.to_string(index=False))

# ── evaluate on held-out test set ─────────────────────────────────────────────
y_pred      = best_pipeline.predict(X_test)
y_proba     = best_pipeline.predict_proba(X_test)
acc         = accuracy_score(y_test, y_pred)
f1mac       = f1_score(y_test, y_pred, average="macro")
class_rep   = classification_report(y_test, y_pred, target_names=["HOF", "NOT"], output_dict=True)

print(f"\nTest accuracy : {acc:.4f}")
print(f"F1-macro      : {f1mac:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["HOF", "NOT"]))

# ── helper: save figure ────────────────────────────────────────────────────────
def savefig(name: str):
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot -> {path}")

# ── plot 1: dataset class distribution ────────────────────────────────────────
print("\nGenerating plots...")
labels_order = ["HOF", "NOT"]
colors = {"HOF": "#e05c5c", "NOT": "#5b9bd5"}

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (title, df_) in zip(
    axes,
    [("Train split", train_df), ("Test split", test_df if test_frames else pd.DataFrame({"task_1": y_test}))],
):
    counts = df_["task_1"].value_counts().reindex(labels_order, fill_value=0)
    bars = ax.bar(labels_order, counts.values, color=[colors[l] for l in labels_order], edgecolor="white", width=0.5)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(v), ha="center", va="bottom", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(counts.values) * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Class Distribution", fontsize=14, fontweight="bold")
plt.tight_layout()
savefig("lr_class_distribution.png")

# ── plot 2: text length distributions ─────────────────────────────────────────
train_df["text_len"] = train_df["text"].str.split().str.len()
fig, ax = plt.subplots(figsize=(8, 4))
for lbl, col in colors.items():
    data = train_df[train_df["task_1"] == lbl]["text_len"]
    ax.hist(data, bins=40, alpha=0.6, label=lbl, color=col)
ax.set_xlabel("Token count")
ax.set_ylabel("Frequency")
ax.set_title("Token Length Distribution (train)", fontsize=12)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
savefig("lr_text_length_distribution.png")

# ── plot 3: GridSearchCV – C vs F1-macro ──────────────────────────────────────
cv_df = pd.DataFrame(gs.cv_results_)[["param_clf__C", "mean_test_score", "std_test_score"]]
cv_df.columns = ["C", "f1_macro_mean", "f1_macro_std"]
fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(cv_df["C"].astype(float), cv_df["f1_macro_mean"],
            yerr=cv_df["f1_macro_std"], marker="o", color="#5b9bd5",
            linewidth=2, capsize=4)
ax.axvline(gs.best_params_["clf__C"], color="#e05c5c", linestyle="--",
           label=f"Best C={gs.best_params_['clf__C']}")
ax.set_xscale("log")
ax.set_xlabel("C (log scale)")
ax.set_ylabel("CV F1-macro")
ax.set_title("GridSearchCV: C vs F1-macro", fontsize=12)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
savefig("lr_gridsearch_cv.png")

# ── plot 4: confusion matrix ───────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred, labels=["HOF", "NOT"])
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["HOF", "NOT"],
            yticklabels=["HOF", "NOT"], ax=ax, linewidths=0.5)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix", fontsize=12)
plt.tight_layout()
savefig("lr_confusion_matrix.png")

# ── plot 5: per-class precision / recall / f1 bar chart ───────────────────────
metrics_df = pd.DataFrame({
    "Precision": [class_rep["HOF"]["precision"], class_rep["NOT"]["precision"]],
    "Recall":    [class_rep["HOF"]["recall"],    class_rep["NOT"]["recall"]],
    "F1-score":  [class_rep["HOF"]["f1-score"],  class_rep["NOT"]["f1-score"]],
}, index=["HOF", "NOT"])
fig, ax = plt.subplots(figsize=(7, 4))
metrics_df.plot(kind="bar", ax=ax, color=["#5b9bd5", "#f0a842", "#70b87e"],
                edgecolor="white", width=0.6)
ax.set_xticklabels(["HOF", "NOT"], rotation=0)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Per-class Metrics", fontsize=12)
ax.legend(loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
savefig("lr_per_class_metrics.png")

# ── plot 6: ROC curve ─────────────────────────────────────────────────────────
# y_bin: 1 = HOF (positive class), 0 = NOT
y_bin = (pd.Series(y_test) == "HOF").astype(int).values
hof_idx = list(best_pipeline.classes_).index("HOF")
fpr, tpr, _ = roc_curve(y_bin, y_proba[:, hof_idx])
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#5b9bd5", lw=2, label=f"AUC = {roc_auc:.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve (HOF class)", fontsize=12)
ax.legend(loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
savefig("lr_roc_curve.png")

# ── plot 7: precision-recall curve ────────────────────────────────────────────
prec, rec, _ = precision_recall_curve(y_bin, y_proba[:, hof_idx])
pr_auc = auc(rec, prec)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec, prec, color="#e05c5c", lw=2, label=f"AUC = {pr_auc:.3f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve (HOF class)", fontsize=12)
ax.legend(loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
savefig("lr_precision_recall_curve.png")

# ── save JSON report ──────────────────────────────────────────────────────────
report = {
    "dataset": dataset_stats,
    "train_total": len(train_df),
    "test_total": len(y_test),
    "best_C": gs.best_params_["clf__C"],
    "cv_f1_macro": round(gs.best_score_, 4),
    "test_accuracy": round(acc, 4),
    "test_f1_macro": round(f1mac, 4),
    "roc_auc_HOF": round(roc_auc, 4),
    "pr_auc_HOF": round(pr_auc, 4),
    "classification_report": class_rep,
    "confusion_matrix": {"HOF_as_HOF": int(cm[0,0]), "HOF_as_NOT": int(cm[0,1]),
                         "NOT_as_HOF": int(cm[1,0]), "NOT_as_NOT": int(cm[1,1])},
    "cv_results": cv_df.to_dict(orient="records"),
}
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print(f"\n  Saved report -> {REPORT_PATH}")

# ── save artifacts ────────────────────────────────────────────────────────────
with open(MODEL_PATH, "wb") as f:
    pickle.dump(best_pipeline, f, protocol=4)

with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(best_pipeline.named_steps["features"], f, protocol=4)

print(f"\nSaved -> {MODEL_PATH}")
print(f"Saved -> {VECTORIZER_PATH}")
print("\nDone! Flask app will now use the improved model automatically.")
