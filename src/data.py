"""تحميل البيانات وتنظيف الجدول (قيم فارغة، تكرار، تسميات)."""
from typing import Dict, Tuple

import pandas as pd

from .preprocess import clean_text


def load_and_clean(path: str, text_col: str = "text", label_col: str = "label",
                   min_class_count: int = 10) -> Tuple[pd.DataFrame, Dict]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    report: Dict = {"rows_raw": len(df)}

    df = df.rename(columns={text_col: "text", label_col: "label"})[["text", "label"]]
    df["label"] = df["label"].astype("string").str.strip()

    before = len(df)
    df = df.dropna(subset=["text", "label"])
    report["dropped_null"] = before - len(df)

    df["clean"] = df["text"].map(clean_text)
    before = len(df)
    df = df[df["clean"].str.len() > 0]
    report["dropped_empty_after_cleaning"] = before - len(df)

    # إزالة التكرار قبل التقسيم لتجنب تسرب البيانات بين train و test
    before = len(df)
    df = df.drop_duplicates(subset=["clean", "label"])
    report["dropped_duplicates"] = before - len(df)

    # نصوص متطابقة بتسميات مختلفة = تعارض في التسمية
    conflict = df.groupby("clean")["label"].nunique()
    conflict = conflict[conflict > 1].index
    report["conflicting_label_texts"] = int(len(conflict))
    df = df[~df["clean"].isin(conflict)]

    counts = df["label"].value_counts()
    rare = counts[counts < min_class_count].index.tolist()
    report["dropped_rare_classes"] = rare
    df = df[~df["label"].isin(rare)]

    report["rows_final"] = len(df)
    report["class_distribution"] = df["label"].value_counts().to_dict()
    return df.reset_index(drop=True), report
