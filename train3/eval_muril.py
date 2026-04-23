"""
Evaluate the saved MuRIL checkpoint (model3/muril/) and regenerate:
  - model3/muril/report.json
  - model3/plots/muril_confusion_matrix.png
  - model3/plots/muril_roc_curve.png

Run:
    conda activate test2
    python train3/eval_muril.py
"""

import os, re, json, sys
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    confusion_matrix, roc_curve, auc,
)

# Import prep2 preprocessing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.prep2 import preprocess_muril

# ── config ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EN_TEST_2019  = os.path.join(BASE_DIR, "english_dataset", "english_dataset", "hasoc2019_en_test-2919.tsv")
HI_TEST_2019  = os.path.join(BASE_DIR, "hindi_dataset",  "hindi_dataset",  "hasoc2019_hi_test_gold_2919.tsv")
EN_TEST_2020  = os.path.join(BASE_DIR, "english_dataset_2020", "english_test_1509.csv")
HI_TEST_2020  = os.path.join(BASE_DIR, "hindi_dataset_2020",  "hindi_test_1509.csv")

MURIL_DIR   = os.path.join(BASE_DIR, "model3", "muril")
PLOTS_DIR   = os.path.join(BASE_DIR, "model3", "plots")
REPORT_PATH = os.path.join(MURIL_DIR, "report.json")

os.makedirs(PLOTS_DIR, exist_ok=True)

MAX_LEN    = 128
BATCH_SIZE = 16
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL2IDX = {"HOF": 1, "NOT": 0}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}

# ── cleaning using prep2.preprocess_muril() ────────────────────────────────────
def clean(text: str) -> str:
    """Use prep2.preprocess_muril() for minimal preprocessing."""
    return preprocess_muril(text)

def _normalise(df):
    df.columns = df.columns.str.strip()
    if "task1" in df.columns and "task_1" not in df.columns:
        df = df.rename(columns={"task1": "task_1"})
    df = df[['text', 'task_1']].dropna()
    df['task_1'] = df['task_1'].str.strip().str.upper().replace('NNOT', 'NOT')
    df = df[df['task_1'].isin(['HOF', 'NOT'])]
    df['text'] = df['text'].apply(clean)
    return df

def load_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xls'):
        return _normalise(pd.read_excel(path, dtype=str))
    elif ext == '.csv':
        return _normalise(pd.read_csv(path, dtype=str))
    else:
        return _normalise(pd.read_csv(path, sep='\t', dtype=str))

# ── load test data ─────────────────────────────────────────────────────────────
print("Loading test datasets...")
test_paths = [EN_TEST_2019, HI_TEST_2019, EN_TEST_2020, HI_TEST_2020]
test_df = pd.concat([load_any(p) for p in test_paths if os.path.exists(p)], ignore_index=True)
print(f"  Test: {len(test_df)}  HOF={(test_df['task_1']=='HOF').sum()}  NOT={(test_df['task_1']=='NOT').sum()}")

# ── dataset ────────────────────────────────────────────────────────────────────
print(f"\nLoading tokenizer from {MURIL_DIR} ...")
tokenizer = AutoTokenizer.from_pretrained(MURIL_DIR)

class HateDataset(Dataset):
    def __init__(self, df):
        self.texts  = df['text'].tolist()
        self.labels = [LABEL2IDX[l] for l in df['task_1']]

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        enc = tokenizer(
            self.texts[i],
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids'     : enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels'        : torch.tensor(self.labels[i], dtype=torch.long),
        }

test_loader = DataLoader(HateDataset(test_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── load model ─────────────────────────────────────────────────────────────────
print(f"Loading model from {MURIL_DIR} ...")
model = AutoModelForSequenceClassification.from_pretrained(MURIL_DIR).to(DEVICE)
model.eval()
print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ── evaluate ───────────────────────────────────────────────────────────────────
print("\nRunning evaluation...")
all_preds, all_labels, all_proba = [], [], []
with torch.no_grad():
    for batch in test_loader:
        ids  = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        lbls = batch['labels'].to(DEVICE)
        outputs = model(input_ids=ids, attention_mask=mask)
        preds  = outputs.logits.argmax(dim=1).cpu().tolist()
        proba  = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(lbls.cpu().tolist())
        all_proba.extend(proba)

acc   = accuracy_score(all_labels, all_preds)
f1mac = f1_score(all_labels, all_preds, average='macro')
class_rep = classification_report(
    [IDX2LABEL[l] for l in all_labels],
    [IDX2LABEL[p] for p in all_preds],
    target_names=["HOF", "NOT"], output_dict=True
)

print(f"\nTest accuracy : {acc:.4f}")
print(f"F1-macro      : {f1mac:.4f}")
print("\nClassification Report:")
print(classification_report(
    [IDX2LABEL[l] for l in all_labels],
    [IDX2LABEL[p] for p in all_preds],
    target_names=["HOF", "NOT"]
))

# ── plots ──────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["NOT","HOF"], yticklabels=["NOT","HOF"],
                ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (MuRIL)", fontsize=12)
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "muril_confusion_matrix.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved plot -> {p}")

    # ROC
    y_bin = np.array(all_labels)
    fpr, tpr, _ = roc_curve(y_bin, all_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#5b9bd5", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve – MuRIL (HOF class)", fontsize=12)
    ax.legend(loc="lower right"); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "muril_roc_curve.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved plot -> {p}")

except Exception as e:
    print(f"Plot skipped: {e}")
    roc_auc = None

# ── save report ────────────────────────────────────────────────────────────────
report = {
    "model": "google/muril-base-cased",
    "dataset": "HASOC 2019 + 2020",
    "device": str(DEVICE),
    "max_len": MAX_LEN,
    "batch_size": BATCH_SIZE,
    "test_accuracy": round(acc, 4),
    "test_f1_macro": round(f1mac, 4),
    "roc_auc_HOF": round(roc_auc, 4) if roc_auc is not None else None,
    "classification_report": class_rep,
}
with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nSaved report -> {REPORT_PATH}")
print("\nDone!")
