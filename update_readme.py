"""Rewrite the 'Current scope' + 'Results' part of README.md from reports/metrics.json (run after train.py)."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
m = json.loads((root / "reports" / "metrics.json").read_text())
t, d, names = m["test"], m["dataset"], m["classes"]
rep, cm = t["classification_report"], t["confusion_matrix"]
lo, hi = t["accuracy_ci95"]
roc = t.get("roc") or {}
auc = roc.get("auc", roc.get("auc_ovr_macro"))
sp = d["split_sizes"]

scope = (f"| | |\n|---|---|\n| **Task** | {len(names)}-class classification: {', '.join(names)} |\n"
         f"| **Dataset** | {d['raw_images']} images -> {d['unique_images']} unique ({d['duplicates_removed']} byte-identical duplicates removed) |\n"
         f"| **Split** | stratified {sp['train']} / {sp['val']} / {sp['test']} (train / val / test), seed 42 |\n"
         "| **Retrain** | `python train.py --dataset <folder>` (class folders -> classes; add `--exclude notumor` to drop a class) |\n")
out = ["## Current scope\n", scope, f"\n## Results (held-out test set, n = {t['n']})\n",
       "| Metric | Value |\n|---|---|",
       f"| Accuracy | **{t['accuracy']*100:.1f} %** (95 % CI {lo*100:.0f} - {hi*100:.0f} %) |",
       f"| Precision / Recall / F1 (macro) | {t['precision_macro']*100:.1f} % / {t['recall_macro']*100:.1f} % / {t['f1_macro']*100:.1f} % |"]
if auc is not None:
    out.append(f"| ROC-AUC | {auc:.3f} |")
out += ["\n| Class | Precision | Recall | F1 | Support |\n|---|---|---|---|---|"]
out += [f"| {c} | {rep[c]['precision']:.3f} | {rep[c]['recall']:.3f} | {rep[c]['f1-score']:.3f} | {int(rep[c]['support'])} |" for c in names]
out += ["\nConfusion matrix (rows = actual, columns = predicted):\n",
        "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
out += [f"| **{names[i]}** | " + " | ".join(str(v) for v in cm[i]) + " |" for i in range(len(names))]
out.append("\nFull numbers: `reports/metrics.json`; plots: `reports/confusion_matrix.png`, `reports/training_curves.png`. "
           "Results come from one training run on one random split; they are not clinically validated.\n")

readme = root / "README.md"
txt = readme.read_text(encoding="utf-8")
a = txt.index("## Current scope")
b = txt.index("## Features")
readme.write_text(txt[:a] + "\n".join(out) + "\n" + txt[b:], encoding="utf-8")
print("README.md updated from reports/metrics.json")
