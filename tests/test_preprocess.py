from src.preprocess import clean_text, normalize_arabic


def test_removes_diacritics():
    assert normalize_arabic("الطَّلَبُ") == "الطلب"


def test_removes_tatweel_and_repeats():
    assert normalize_arabic("سـريـع") == "سريع"
    assert normalize_arabic("مشكلللللة") == "مشكله"


def test_normalizes_letters():
    assert normalize_arabic("إلى أحمد آمن") == "الي احمد امن"
    assert normalize_arabic("مستشفى") == "مستشفي"


def test_removes_symbols_digits_latin_emoji_urls():
    out = normalize_arabic("طلب 123 order! 😡 https://x.com/a ؟؟")
    assert out == "طلب"


def test_stopwords_removed_but_negation_kept():
    out = clean_text("الطلب لم يصل الى المنزل في الموعد")
    assert "لم" in out.split()
    assert "الي" not in out.split() and "في" not in out.split()


def test_handles_non_string():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""
    assert clean_text("") == ""
