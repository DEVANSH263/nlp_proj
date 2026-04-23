"""
compare_models.py — generates model comparison plots from saved reports
Saves: model/plots/comparison_*.png
"""
import os, json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PLOTS_DIR = os.path.join(BASE_DIR, "model", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── load reports ───────────────────────────────────────────────────────────────
with open(os.path.join(BASE_DIR, "model", "lr",    "report.json")) as f: lr    = json.load(f)
with open(os.path.join(BASE_DIR, "model", "lstm",  "report.json")) as f: lstm  = json.load(f)
with open(os.path.join(BASE_DIR, "model", "muril", "report.json")) as f: muril = json.load(f)

MODELS  = ["LR", "BiLSTM", "MuRIL"]
COLORS  = ["#5b9bd5", "#f0a842", "#70b87e"]

# ── helper ─────────────────────────────────────────────────────────────────────
def savefig(name):
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")

# ── plot 1: overall accuracy & F1-macro bar chart ─────────────────────────────
acc = [lr["test_accuracy"], lstm["test_accuracy"], muril["test_accuracy"]]
f1  = [lr["test_f1_macro"], lstm["test_f1_macro"],  muril["test_f1_macro"]]

x = np.arange(len(MODELS))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, acc, width, label="Accuracy", color=[c + "cc" for c in COLORS],
               edgecolor="white")
bars2 = ax.bar(x + width/2, f1,  width, label="F1-macro", color=COLORS, edgecolor="white")

for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(MODELS, fontsize=12)
ax.set_ylim(0.5, 0.92)
ax.set_ylabel("Score")
ax.set_title("Model Comparison: Accuracy & F1-macro", fontsize=13, fontweight="bold")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
savefig("comparison_accuracy_f1.png")

# ── plot 2: per-class precision / recall / F1 ─────────────────────────────────
cr_lr    = lr["classification_report"]
cr_lstm  = lstm["classification_report"]
cr_muril = muril["classification_report"]

metrics  = ["precision", "recall", "f1-score"]
classes_list  = ["HOF", "NOT"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, cls in zip(axes, classes_list):
    vals_lr    = [cr_lr[cls][m]    for m in metrics]
    vals_lstm  = [cr_lstm[cls][m]  for m in metrics]
    vals_muril = [cr_muril[cls][m] for m in metrics]

    xp = np.arange(len(metrics))
    w  = 0.25
    b1 = ax.bar(xp - w,   vals_lr,    w, label="LR",     color=COLORS[0], edgecolor="white")
    b2 = ax.bar(xp,       vals_lstm,  w, label="BiLSTM", color=COLORS[1], edgecolor="white")
    b3 = ax.bar(xp + w,   vals_muril, w, label="MuRIL",  color=COLORS[2], edgecolor="white")

    for bar in list(b1) + list(b2) + list(b3):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(xp)
    ax.set_xticklabels(["Precision", "Recall", "F1"], fontsize=11)
    ax.set_ylim(0.4, 1.0)
    ax.set_title(f"Class: {cls}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Per-class Metrics by Model", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("comparison_per_class.png")

# ── plot 3: radar chart ────────────────────────────────────────────────────────
labels_radar = ["Accuracy", "F1-macro", "HOF-F1", "NOT-F1", "HOF-Prec", "HOF-Rec"]
data = {
    "LR":     [lr["test_accuracy"],    lr["test_f1_macro"],
               cr_lr["HOF"]["f1-score"],    cr_lr["NOT"]["f1-score"],
               cr_lr["HOF"]["precision"],   cr_lr["HOF"]["recall"]],
    "BiLSTM": [lstm["test_accuracy"],  lstm["test_f1_macro"],
               cr_lstm["HOF"]["f1-score"],  cr_lstm["NOT"]["f1-score"],
               cr_lstm["HOF"]["precision"], cr_lstm["HOF"]["recall"]],
    "MuRIL":  [muril["test_accuracy"], muril["test_f1_macro"],
               cr_muril["HOF"]["f1-score"], cr_muril["NOT"]["f1-score"],
               cr_muril["HOF"]["precision"],cr_muril["HOF"]["recall"]],
}

N = len(labels_radar)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for (name, vals), color in zip(data.items(), COLORS):
    vals_plot = vals + vals[:1]
    ax.plot(angles, vals_plot, "o-", linewidth=2, color=color, label=name)
    ax.fill(angles, vals_plot, alpha=0.12, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels_radar, fontsize=10)
ax.set_ylim(0.4, 1.0)
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
ax.set_yticklabels(["0.5","0.6","0.7","0.8","0.9"], fontsize=8)
ax.set_title("Model Comparison Radar", fontsize=13, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
savefig("comparison_radar.png")

# ── plot 4: summary table as figure ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis("off")

col_labels = ["Model", "Accuracy", "F1-macro", "HOF F1", "NOT F1", "HOF Prec", "HOF Rec"]
rows = [
    ["LR",
     f"{lr['test_accuracy']:.4f}",   f"{lr['test_f1_macro']:.4f}",
     f"{cr_lr['HOF']['f1-score']:.4f}", f"{cr_lr['NOT']['f1-score']:.4f}",
     f"{cr_lr['HOF']['precision']:.4f}", f"{cr_lr['HOF']['recall']:.4f}"],
    ["BiLSTM",
     f"{lstm['test_accuracy']:.4f}", f"{lstm['test_f1_macro']:.4f}",
     f"{cr_lstm['HOF']['f1-score']:.4f}", f"{cr_lstm['NOT']['f1-score']:.4f}",
     f"{cr_lstm['HOF']['precision']:.4f}", f"{cr_lstm['HOF']['recall']:.4f}"],
    ["MuRIL",
     f"{muril['test_accuracy']:.4f}", f"{muril['test_f1_macro']:.4f}",
     f"{cr_muril['HOF']['f1-score']:.4f}", f"{cr_muril['NOT']['f1-score']:.4f}",
     f"{cr_muril['HOF']['precision']:.4f}", f"{cr_muril['HOF']['recall']:.4f}"],
]

table = ax.table(cellText=rows, colLabels=col_labels,
                 cellLoc="center", loc="center",
                 bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(11)

# header style
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")

# row colors + highlight best per column
row_colors = [COLORS[0] + "33", COLORS[1] + "33", COLORS[2] + "33"]
for i, row in enumerate(rows):
    for j in range(len(col_labels)):
        table[i+1, j].set_facecolor(row_colors[i])

# bold best value in each numeric column
for j in range(1, len(col_labels)):
    vals_col = [float(rows[i][j]) for i in range(len(rows))]
    best_i   = int(np.argmax(vals_col))
    table[best_i+1, j].set_text_props(fontweight="bold")
    table[best_i+1, j].set_facecolor(COLORS[best_i] + "99")

ax.set_title("Model Comparison Summary", fontsize=13, fontweight="bold", pad=10)
plt.tight_layout()
savefig("comparison_table.png")

print("\nAll comparison plots saved to model/plots/")
