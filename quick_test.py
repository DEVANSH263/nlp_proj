import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"c:\Documents\Projects\nlp_proj")

from app import create_app
app = create_app()

# Real examples from the HASOC dataset (train + test)
EXAMPLES = [
    # (text,                                                      expected, source)
    ("You are a disgusting worthless piece of trash!",           "HOF", "EN-TEST"),
    ("Have a wonderful day everyone, stay safe!",                "NOT", "EN-TRAIN"),
    ("teri maa ki aankh saale kamine, sharam kar!",              "HOF", "HI-TRAIN"),
    ("yaar tu sach mein bohot achha hai, keep it up!",           "NOT", "HI-TRAIN"),
    ("Fuck you. Go back to the dark ages you cow.",              "HOF", "EN-TEST"),
    ("The cricket match was brilliant today! #TeamIndia",        "NOT", "EN-TRAIN"),
]

MODELS = ["lr", "lstm", "muril"]
MODEL_LABEL = {"lr": "Logistic Regression", "lstm": "BiLSTM", "muril": "MuRIL"}

with app.app_context():
    from utils.predict import predict

    print()
    print("=" * 95)
    print("  HateShield – Quick Example Test  (Train + Test set samples, all 3 models)")
    print("=" * 95)

    totals = {m: {"ok": 0, "miss": 0} for m in MODELS}

    for text, gt, src in EXAMPLES:
        print(f"\n  [{src}]  GT={gt}")
        print(f"  Text: {text}")
        print(f"  {'─'*80}")
        for mt in MODELS:
            r = predict(text, model_type=mt)
            pred = r["prediction"]
            conf = r["confidence"] * 100
            match = "✓ OK  " if pred == gt else "✗ MISS"
            totals[mt]["ok" if pred == gt else "miss"] += 1
            print(f"    [{match}]  {MODEL_LABEL[mt]:<22}  → {pred}   conf={conf:5.1f}%")

    print()
    print("=" * 95)
    print("  ACCURACY SUMMARY")
    print("=" * 95)
    n = len(EXAMPLES)
    for mt in MODELS:
        ok = totals[mt]["ok"]
        acc = ok / n * 100
        print(f"  {MODEL_LABEL[mt]:<22}  {ok}/{n}  ({acc:.1f}%)")
    print()
