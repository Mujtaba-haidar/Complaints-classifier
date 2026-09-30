"""تدريب ومقارنة النماذج وحفظ الأفضل.

الاستخدام:
    python -m src.train --data data/complaints.csv --out artifacts
اختيار النموذج يتم بالتحقق المتقاطع على بيانات التدريب فقط،
وبيانات الاختبار تُستخدم مرة واحدة للتقييم النهائي.
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .data import load_and_clean
from .evaluate import compute_metrics, confusion, report_text, top_confusions


def _word():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def _char():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)


def build_models(seed: int = 42):
    return {
        "Baseline (الفئة الأكثر تكرارًا)": DummyClassifier(strategy="most_frequent"),
        "TF-IDF(word) + NaiveBayes": Pipeline([("f", _word()), ("c", MultinomialNB(alpha=0.3))]),
        "TF-IDF(word) + LogReg": Pipeline([("f", _word()),
                                           ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "TF-IDF(word+char) + LogReg": Pipeline([("f", FeatureUnion([("w", _word()), ("c", _char())])),
                                                ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "TF-IDF(word+char) + LinearSVC": Pipeline([("f", FeatureUnion([("w", _word()), ("c", _char())])),
                                                   ("c", LinearSVC(class_weight="balanced", random_state=seed))]),
    }


def run(data_path: str, out_dir: str, test_size: float = 0.2, seed: int = 42, cv: int = 5) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df, data_report = load_and_clean(data_path)
    X_tr, X_te, y_tr, y_te, raw_tr, raw_te = train_test_split(
        df["clean"], df["label"], df["text"], test_size=test_size, stratify=df["label"], random_state=seed)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    rows, fitted = [], {}
    for name, model in build_models(seed).items():
        cv_f1 = cross_val_score(model, X_tr, y_tr, cv=skf, scoring="f1_macro")
        model.fit(X_tr, y_tr)
        m = compute_metrics(y_te, model.predict(X_te))
        rows.append({"model": name, "cv_f1_macro_mean": cv_f1.mean(), "cv_f1_macro_std": cv_f1.std(), **m})
        fitted[name] = model
        print(f"{name:40s} CV-F1={cv_f1.mean():.3f}±{cv_f1.std():.3f}  test-F1={m['f1_macro']:.3f}")

    comp = pd.DataFrame(rows).round(4)
    comp.to_csv(out / "model_comparison.csv", index=False, encoding="utf-8-sig")

    best_name = comp.loc[comp["cv_f1_macro_mean"].idxmax(), "model"]  # الاختيار بالـ CV فقط
    best = fitted[best_name]
    pred = best.predict(X_te)
    labels = sorted(df["label"].unique())

    (out / "classification_report.txt").write_text(report_text(y_te, pred), encoding="utf-8")
    pd.DataFrame(confusion(y_te, pred, labels), index=labels, columns=labels)\
        .to_csv(out / "confusion_matrix.csv", encoding="utf-8-sig")
    err = pd.DataFrame({"text": raw_te.values, "clean": X_te.values, "true": y_te.values, "pred": pred})
    err[err.true != err.pred].to_csv(out / "errors.csv", index=False, encoding="utf-8-sig")

    joblib.dump({"model": best, "labels": labels, "name": best_name}, out / "model.joblib")
    summary = {"best_model": best_name, "data_report": data_report,
               "test_metrics": compute_metrics(y_te, pred),
               "top_confusions": [[t, p, c] for (t, p), c in top_confusions(y_te, pred)],
               "n_train": len(X_tr), "n_test": len(X_te)}
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nأفضل نموذج: {best_name}\n{report_text(y_te, pred)}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="تدريب مصنّف الشكاوى العربية")
    ap.add_argument("--data", default="data/complaints.csv")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run(a.data, a.out, a.test_size, a.seed)
