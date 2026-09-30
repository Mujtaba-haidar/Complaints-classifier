"""تنظيف وتطبيع النصوص العربية.

الخطوات: إزالة الروابط -> إزالة التشكيل والتطويل -> توحيد الحروف
-> حذف الرموز/الأرقام/الحروف اللاتينية/الإيموجي -> ضغط الحروف المكررة
-> حذف كلمات التوقف (مع الإبقاء على أدوات النفي عمدًا).
"""
import re
import unicodedata
from typing import Iterable, List

_URL = re.compile(r"(https?://\S+|www\.\S+|\S+@\S+\.\S+)")
_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL = "\u0640"
_ALEF = re.compile("[إأآٱ]")
_NON_ARABIC = re.compile(r"[^\u0621-\u064A\s]")  # يحذف الأرقام والرموز واللاتيني والإيموجي
_REPEAT = re.compile(r"(.)\1{2,}")  # ســـــريع / مشكلللة -> حرف واحد
_SPACES = re.compile(r"\s+")

# ملاحظة: أدوات النفي (لا، لم، لن، ما، ليس، غير) غير مدرجة عمدًا
# لأنها تحمل معنى أساسيًا في الشكاوى ("لم يصل" مختلفة عن "وصل").
_RAW_STOPWORDS = """
في من الى إلى على عن مع هذا هذه هذي ذلك تلك هو هي هم هن انا أنا نحن انت أنت انتم
كان كانت يكون تكون ان أن إن انه أنه انها أنها التي الذي الذين اللي او أو ثم قد كل
بعد قبل عند حتى هنا هناك ايضا أيضا جدا فقط لكن بل اذا إذا حيث بين كما مثل
لدي لدى عندي معي له لها لهم به بها بهم فيه فيها منه منها عنه عنها اي أي
و ف ب ل ك يا ايه اية
"""


def normalize_arabic(text: str) -> str:
    """تطبيع الحروف دون حذف الكلمات."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _URL.sub(" ", text)
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    text = _ALEF.sub("ا", text)
    text = (text.replace("ى", "ي").replace("ة", "ه")
                .replace("ؤ", "و").replace("ئ", "ي"))
    text = _NON_ARABIC.sub(" ", text)
    text = _REPEAT.sub(r"\1", text)
    return _SPACES.sub(" ", text).strip()


STOPWORDS = frozenset(normalize_arabic(w) for w in _RAW_STOPWORDS.split())


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """التنظيف الكامل لنص واحد."""
    text = normalize_arabic(text)
    if remove_stopwords:
        text = " ".join(t for t in text.split() if t not in STOPWORDS)
    return text


def clean_many(texts: Iterable[str], remove_stopwords: bool = True) -> List[str]:
    return [clean_text(t, remove_stopwords) for t in texts]
