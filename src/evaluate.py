"""مقاييس التقييم."""
from typing import Dict

from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    out = {"accuracy": float(accuracy_score(y_true, y_pred))}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        out.update({f"precision_{avg}": float(p), f"recall_{avg}": float(r), f"f1_{avg}": float(f)})
    return out


def report_text(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, zero_division=0, digits=3)


def confusion(y_true, y_pred, labels):
    return confusion_matrix(y_true, y_pred, labels=labels)


def top_confusions(y_true, y_pred, k: int = 5):
    pairs = {}
    for t, p in zip(y_true, y_pred):
        if t != p:
            pairs[(t, p)] = pairs.get((t, p), 0) + 1
    return sorted(pairs.items(), key=lambda x: -x[1])[:k]
