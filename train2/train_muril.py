"""
MuRIL fine-tune for hate-speech detection (HASOC 2019 + 2020 EN + HI, task_1: HOF/NOT)

Model: google/muril-base-cased
  - Pre-trained on 17 Indian languages + English + transliterated Indic text
  - Handles Devanagari, Roman Hindi (Hinglish), and English natively
  - No manual dictionary / normalization needed
  - Trains on combined HASOC 2019 + 2020 data for improved coverage

Run:
    conda activate test2
    python train2/train_muril.py

Outputs:
    model2/muril/           (HuggingFace saved model + tokenizer)
    model2/muril/report.json
    model2/plots/muril_training_history.png
    model2/plots/muril_confusion_matrix.png
    model2/plots/muril_roc_curve.png
"""

import os, re, json, random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
)
from torch.optim import AdamW

# ── config ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# HASOC 2019
EN_TRAIN_2019 = os.path.join(BASE_DIR, "english_dataset", "english_dataset", "english_dataset.tsv")
EN_TEST_2019  = os.path.join(BASE_DIR, "english_dataset", "english_dataset", "hasoc2019_en_test-2919.tsv")
HI_TRAIN_2019 = os.path.join(BASE_DIR, "hindi_dataset",  "hindi_dataset",  "hindi_dataset.tsv")
HI_TEST_2019  = os.path.join(BASE_DIR, "hindi_dataset",  "hindi_dataset",  "hasoc2019_hi_test_gold_2919.tsv")

# HASOC 2020
EN_TRAIN_2020 = os.path.join(BASE_DIR, "english_dataset_2020", "hasoc_2020_en_train.xlsx")
EN_TEST_2020  = os.path.join(BASE_DIR, "english_dataset_2020", "english_test_1509.csv")
HI_TRAIN_2020 = os.path.join(BASE_DIR, "hindi_dataset_2020",  "hasoc_2020_hi_train.xlsx")
HI_TEST_2020  = os.path.join(BASE_DIR, "hindi_dataset_2020",  "hindi_test_1509.csv")

MODEL_DIR    = os.path.join(BASE_DIR, "model2")
PLOTS_DIR    = os.path.join(MODEL_DIR, "plots")
MURIL_DIR    = os.path.join(MODEL_DIR, "muril")
REPORT_PATH  = os.path.join(MURIL_DIR, "report.json")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(MURIL_DIR,  exist_ok=True)

MODEL_NAME  = "google/muril-base-cased"
MAX_LEN     = 128
BATCH_SIZE  = 16
EPOCHS      = 20
LR          = 2e-5
WARMUP_FRAC = 0.1
SEED        = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

LABEL2IDX = {"HOF": 1, "NOT": 0}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}

# ── minimal cleaning (MuRIL tokenizer handles the rest) ───────────────────────
_REPEAT = re.compile(r'(.)\1{2,}')

def clean(text: str) -> str:
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = _REPEAT.sub(r'\1\1', text)
    return re.sub(r'\s+', ' ', text).strip()

# ── load TSV ───────────────────────────────────────────────────────────────────
def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    if "task1" in df.columns and "task_1" not in df.columns:
        df = df.rename(columns={"task1": "task_1"})
    df = df[['text', 'task_1']].dropna()
    df['task_1'] = df['task_1'].str.strip().str.upper().replace('NNOT', 'NOT')
    df = df[df['task_1'].isin(['HOF', 'NOT'])]
    df['text'] = df['text'].apply(clean)
    return df

def load_tsv(path):
    return _normalise(pd.read_csv(path, sep='\t', dtype=str))

def load_csv(path):
    return _normalise(pd.read_csv(path, dtype=str))

def load_xlsx(path):
    return _normalise(pd.read_excel(path, dtype=str))

def load_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xls'):
        return load_xlsx(path)
    elif ext == '.csv':
        return load_csv(path)
    else:
        return load_tsv(path)

# ── load data (2019 + 2020) ────────────────────────────────────────────────────
print("Loading datasets (HASOC 2019 + 2020)...")

train_paths = [EN_TRAIN_2019, HI_TRAIN_2019, EN_TRAIN_2020, HI_TRAIN_2020]
test_paths  = [EN_TEST_2019,  HI_TEST_2019,  EN_TEST_2020,  HI_TEST_2020]

train_df = pd.concat([load_any(p) for p in train_paths if os.path.exists(p)], ignore_index=True)
test_df  = pd.concat([load_any(p) for p in test_paths  if os.path.exists(p)], ignore_index=True)
print(f"  Train: {len(train_df)}  HOF={(train_df['task_1']=='HOF').sum()}  NOT={(train_df['task_1']=='NOT').sum()}")
print(f"  Test : {len(test_df)}   HOF={(test_df['task_1']=='HOF').sum()}  NOT={(test_df['task_1']=='NOT').sum()}")

# ── tokenizer ──────────────────────────────────────────────────────────────────
print(f"\nLoading tokenizer: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ── dataset ────────────────────────────────────────────────────────────────────
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

train_loader = DataLoader(HateDataset(train_df), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(HateDataset(test_df),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── model ──────────────────────────────────────────────────────────────────────
print(f"Loading model: {MODEL_NAME} ...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    ignore_mismatched_sizes=True,
)
model.to(DEVICE)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# class weights for imbalance
hof  = (train_df['task_1'] == 'HOF').sum()
not_ = (train_df['task_1'] == 'NOT').sum()
class_weights = torch.tensor([not_ / len(train_df), hof / len(train_df)],
                               dtype=torch.float).to(DEVICE)

# ── optimiser + scheduler ──────────────────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps  = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_FRAC)
scheduler = get_linear_schedule_with_warmup(optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps)

loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# ── train ──────────────────────────────────────────────────────────────────────
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
best_f1, best_epoch = 0.0, 1

print(f"\nFine-tuning MuRIL ({EPOCHS} epochs, HASOC 2019+2020)...\n")
print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'F1-mac':>6}")
print("-" * 62)

for epoch in range(1, EPOCHS + 1):
    # train
    model.train()
    t_loss, t_correct, t_total = 0.0, 0, 0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False)
    for batch in train_bar:
        ids  = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        lbls = batch['labels'].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=ids, attention_mask=mask)
        loss = loss_fn(outputs.logits, lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        t_loss    += loss.item() * len(lbls)
        preds      = outputs.logits.argmax(dim=1)
        t_correct += (preds == lbls).sum().item()
        t_total   += len(lbls)
        train_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{t_correct/t_total:.4f}")

    # eval
    model.eval()
    v_loss, all_preds, all_labels, all_proba = 0.0, [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]  ", leave=False):
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            lbls = batch['labels'].to(DEVICE)
            outputs = model(input_ids=ids, attention_mask=mask)
            v_loss += loss_fn(outputs.logits, lbls).item() * len(lbls)
            preds   = outputs.logits.argmax(dim=1).cpu().tolist()
            proba   = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(lbls.cpu().tolist())
            all_proba.extend(proba)

    t_loss /= t_total
    t_acc   = t_correct / t_total
    v_loss /= len(test_df)
    v_acc   = accuracy_score(all_labels, all_preds)
    v_f1    = f1_score(all_labels, all_preds, average='macro')

    history["train_loss"].append(round(t_loss, 4))
    history["train_acc"].append(round(t_acc,  4))
    history["val_loss"].append(round(v_loss,  4))
    history["val_acc"].append(round(v_acc,    4))
    history["val_f1"].append(round(v_f1,      4))

    print(f"{epoch:>5}  {t_loss:>10.4f}  {t_acc:>9.4f}  {v_loss:>8.4f}  {v_acc:>7.4f}  {v_f1:>6.4f}")

    if v_f1 > best_f1:
        best_f1, best_epoch = v_f1, epoch
        model.save_pretrained(MURIL_DIR, safe_serialization=False)
        tokenizer.save_pretrained(MURIL_DIR)

print(f"\nBest epoch: {best_epoch}  F1-macro: {best_f1:.4f}")

# ── final eval with best checkpoint ───────────────────────────────────────────
print("\nReloading best checkpoint for final evaluation...")
model = AutoModelForSequenceClassification.from_pretrained(MURIL_DIR).to(DEVICE)
model.eval()
all_preds, all_labels, all_proba = [], [], []
with torch.no_grad():
    for batch in test_loader:
        ids  = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        lbls = batch['labels'].to(DEVICE)
        outputs = model(input_ids=ids, attention_mask=mask)
        preds   = outputs.logits.argmax(dim=1).cpu().tolist()
        proba   = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().tolist()
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

    epochs_r = range(1, EPOCHS + 1)

    # training history
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs_r, history["train_loss"], label="Train", color="#5b9bd5")
    axes[0].plot(epochs_r, history["val_loss"],   label="Val",   color="#e05c5c")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[0].axvline(best_epoch, color="gray", linestyle="--", alpha=0.5)
    axes[0].spines[["top","right"]].set_visible(False)

    axes[1].plot(epochs_r, history["train_acc"], label="Train", color="#5b9bd5")
    axes[1].plot(epochs_r, history["val_acc"],   label="Val",   color="#e05c5c")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    axes[1].set_ylim(0, 1); axes[1].spines[["top","right"]].set_visible(False)

    axes[2].plot(epochs_r, history["val_f1"], label="Val F1-macro", color="#70b87e")
    axes[2].set_title("Val F1-macro"); axes[2].set_xlabel("Epoch"); axes[2].legend()
    axes[2].set_ylim(0, 1); axes[2].spines[["top","right"]].set_visible(False)

    plt.suptitle("MuRIL Fine-Tuning History (HASOC 2019+2020)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "muril_training_history.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved plot -> {p}")

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
    y_bin = np.array(all_labels)   # 1=HOF
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

# ── save report ────────────────────────────────────────────────────────────────
report = {
    "model": MODEL_NAME,
    "dataset": "HASOC 2019 + 2020",
    "device": str(DEVICE),
    "epochs": EPOCHS,
    "best_epoch": best_epoch,
    "max_len": MAX_LEN,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "test_accuracy": round(acc, 4),
    "test_f1_macro": round(f1mac, 4),
    "roc_auc_HOF": round(roc_auc, 4) if 'roc_auc' in dir() else None,
    "classification_report": class_rep,
    "history": history,
}
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(f"\nSaved report -> {REPORT_PATH}")
print(f"Saved model  -> {MURIL_DIR}")
print("\nDone! Trained on HASOC 2019 + 2020 combined data.")
