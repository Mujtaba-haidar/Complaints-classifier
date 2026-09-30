"""مولّد بيانات اصطناعية لشكاوى العملاء (للتجربة عند غياب بيانات حقيقية).

يحاكي مشاكل واقعية: تشكيل، إيموجي، روابط، حروف مكررة، كلمات إنجليزية،
لهجات، تكرار، قيم فارغة، تسميات بمسافات زائدة، شكاوى متداخلة بين فئتين،
وأخطاء في التسمية (label noise).

تنبيه: النتائج على هذه البيانات لا تعكس الأداء على بيانات حقيقية.
"""
import argparse
import random
from pathlib import Path

import pandas as pd

CORES = {
    "التوصيل": [
        "الطلب تأخر عن الموعد المحدد بأكثر من أسبوع",
        "لم تصلني الشحنة حتى الآن وتتبع الطلب لا يتحدث",
        "مندوب التوصيل لا يرد على الاتصال ويرجع الطلب",
        "الطرد وصل مفتوح وممزق من الخارج",
        "وصلني الطلب لعنوان خاطئ رغم أني كتبت العنوان صح",
        "شركة الشحن تقول سلمنا الطلب وأنا ما استلمت شي",
        "وش سبب تأخير التوصيل كل مرة",
    ],
    "الفوترة والدفع": [
        "تم خصم المبلغ مرتين من بطاقتي",
        "الفاتورة تحتوي على رسوم لم أوافق عليها",
        "كود الخصم لم يعمل عند الدفع",
        "السعر في الفاتورة يختلف عن السعر المعلن",
        "تم سحب اشتراك شهري بدون علمي",
        "عملية الدفع فشلت والمبلغ انسحب من حسابي",
        "ما وصلتني فاتورة ضريبية رغم الطلب",
    ],
    "جودة المنتج": [
        "المنتج وصل معطوب ولا يعمل",
        "الجهاز توقف عن العمل بعد يومين فقط",
        "الخامة رديئة جدا ولا تشبه الصور",
        "المنتج لا يطابق الوصف في الموقع",
        "المقاس غير مطابق والجودة سيئة",
        "فيه خدوش واضحة والمنتج مستعمل",
        "البطارية تنفد بسرعة والمنتج ضعيف",
    ],
    "خدمة العملاء": [
        "موظف خدمة العملاء كان غير محترم في الرد",
        "اتصلت أكثر من خمس مرات ولم يرد أحد",
        "انتظرت على الخط نصف ساعة بدون فائدة",
        "لم يحل أحد مشكلتي وكل مرة يحولوني لقسم ثاني",
        "الدعم الفني وعدني بمكالمة ولم يتصل",
        "رد الدعم آلي ولا يفهم مشكلتي",
        "تذكرتي مفتوحة من أسبوعين وما أحد رد",
    ],
    "الاسترجاع والاستبدال": [
        "طلبت استبدال المنتج ولم يتم حتى الآن",
        "سياسة الإرجاع غير واضحة ورفضوا طلبي",
        "رفضوا استرجاع المنتج رغم أنه ضمن المدة",
        "المبلغ المسترد لم يصل إلى حسابي",
        "أرسلت المنتج المرتجع ولم أحصل على أي تأكيد",
        "طلب الاستبدال ينتظر الموافقة منذ أيام",
        "يطلبون مني دفع رسوم شحن للإرجاع",
    ],
    "الحساب والتطبيق": [
        "لا أستطيع تسجيل الدخول إلى حسابي",
        "التطبيق يتوقف باستمرار ويغلق لوحده",
        "نسيت كلمة المرور ولم تصلني رسالة التحقق",
        "تم إغلاق حسابي بدون أي سبب",
        "الموقع بطيء جدا ولا يفتح صفحة الدفع",
        "بعد التحديث الأخير اختفت بيانات حسابي",
        "لا أستطيع تغيير رقم الجوال في الملف الشخصي",
    ],
}
WEIGHTS = {"التوصيل": .25, "الفوترة والدفع": .18, "جودة المنتج": .20,
           "خدمة العملاء": .15, "الاسترجاع والاستبدال": .12, "الحساب والتطبيق": .10}
OPENERS = ["السلام عليكم،", "مرحبا", "للأسف", "أريد أن أشتكي:", "", "", "بعد إذنكم", "يا جماعة"]
CLOSERS = ["أرجو الحل بسرعة", "غير راضي عن الخدمة", "هذا غير مقبول", "شكرا", "", "", "أنتظر ردكم"]
ALSO = ["كمان", "وأيضا", "وفوق هذا", "ومن جهة ثانية"]
FATHA_KASRA = "\u064E\u0650\u064F\u0651"


def _noise(text: str, rng: random.Random) -> str:
    if rng.random() < 0.12:  # تشكيل عشوائي
        text = "".join(c + (rng.choice(FATHA_KASRA) if c.isalpha() and rng.random() < .3 else "") for c in text)
    if rng.random() < 0.10:
        text += " " + rng.choice(["😡", "😤", "!!!", "؟؟؟", "https://t.co/xyz", "order #" + str(rng.randint(1000, 9999))])
    if rng.random() < 0.08:  # تمطيط
        words = text.split()
        i = rng.randrange(len(words))
        words[i] = words[i] + "ـ" * 3 if rng.random() < .5 else words[i].replace("ا", "ااا", 1)
        text = " ".join(words)
    if rng.random() < 0.10:
        text = f"طلب رقم {rng.randint(10000, 99999)} " + text
    return text


def generate(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    cats, weights = list(WEIGHTS), list(WEIGHTS.values())
    rows = []
    for _ in range(n):
        cat = rng.choices(cats, weights)[0]
        parts = [rng.choice(OPENERS), rng.choice(CORES[cat])]
        if rng.random() < 0.28:  # شكوى متداخلة: جزء من فئة أخرى
            other = rng.choice([c for c in cats if c != cat])
            parts += [rng.choice(ALSO), rng.choice(CORES[other])]
        parts.append(rng.choice(CLOSERS))
        text = _noise(" ".join(p for p in parts if p), rng)
        label = cat
        if rng.random() < 0.03:  # خطأ تسمية
            label = rng.choice([c for c in cats if c != cat])
        rows.append({"text": text, "label": label})
    df = pd.DataFrame(rows)
    dup = df.sample(frac=0.03, random_state=seed)
    df = pd.concat([df, dup], ignore_index=True)
    idx = df.sample(frac=0.01, random_state=seed + 1).index
    df.loc[idx, "text"] = rng.choice(["", "   ", None])
    idx = df.sample(frac=0.02, random_state=seed + 2).index
    df.loc[idx, "label"] = df.loc[idx, "label"] + " "
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/complaints.csv")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    generate(a.n, a.seed).to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"تم حفظ البيانات في {a.out}")
