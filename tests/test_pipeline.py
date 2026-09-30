from src.data import load_and_clean
from src.generate_sample_data import generate
from src.predict import load_model, predict_texts
from src.train import run


def test_data_cleaning_report(tmp_path):
    p = tmp_path / "d.csv"
    generate(n=400, seed=1).to_csv(p, index=False, encoding="utf-8-sig")
    out, rep = load_and_clean(str(p))
    assert rep["rows_final"] == len(out) > 0
    assert out["label"].str.strip().eq(out["label"]).all()
    assert not out.duplicated(subset=["clean", "label"]).any()
    assert (out["clean"].str.len() > 0).all()


def test_train_and_predict_end_to_end(tmp_path):
    data = tmp_path / "d.csv"
    generate(n=800, seed=3).to_csv(data, index=False, encoding="utf-8-sig")
    summary = run(str(data), str(tmp_path / "art"), seed=3, cv=3)
    assert 0.0 <= summary["test_metrics"]["f1_macro"] <= 1.0
    assert summary["test_metrics"]["accuracy"] > 0.5
    bundle = load_model(str(tmp_path / "art" / "model.joblib"))
    preds = predict_texts(["الطلب تأخر عن الموعد ولم يصل", "لا أستطيع تسجيل الدخول للتطبيق"], bundle)
    assert len(preds) == 2 and all(p in bundle["labels"] for p in preds)
    assert (tmp_path / "art" / "metrics.json").exists()
