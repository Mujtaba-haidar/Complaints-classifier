"""تصنيف شكاوى جديدة باستخدام النموذج المحفوظ.

    python -m src.predict --text "الطلب تأخر كثيرا"
    python -m src.predict --input new.csv --output predictions.csv
"""
import argparse
from typing import List

import joblib
import pandas as pd

from .preprocess import clean_text


def load_model(path: str = "artifacts/model.joblib"):
    return joblib.load(path)


def predict_texts(texts: List[str], bundle) -> List[str]:
    return list(bundle["model"].predict([clean_text(t) for t in texts]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/model.joblib")
    ap.add_argument("--text")
    ap.add_argument("--input")
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--text-col", default="text")
    a = ap.parse_args()
    bundle = load_model(a.model)
    if a.text:
        print(predict_texts([a.text], bundle)[0])
    elif a.input:
        df = pd.read_csv(a.input, encoding="utf-8-sig")
        df["predicted_label"] = predict_texts(df[a.text_col].fillna("").tolist(), bundle)
        df.to_csv(a.output, index=False, encoding="utf-8-sig")
        print(f"تم حفظ التنبؤات في {a.output}")
    else:
        ap.error("حدد --text أو --input")
