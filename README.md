# تصنيف شكاوى العملاء العربية (Arabic Customer Complaints Classification)

نظام تعلّم آلي يصنّف نص الشكوى العربية إلى إحدى الفئات الموجودة في عمود `category`، باستخدام TF-IDF وعدة نماذج نصية مع تنظيف مخصص للعربية.

## هيكلية المشروع
```
├── README.md            دليل التشغيل
├── REPORT.md            التقرير الفني
├── AI_USAGE.md          الإفصاح عن أدوات الذكاء الاصطناعي
├── requirements.txt
├── pytest.ini
├── data/                البيانات (complaints.csv)
├── artifacts/           مخرجات التدريب (النموذج، المقاييس، الأخطاء)
├── src/
│   ├── preprocess.py    تنظيف النص العربي (تشكيل، رموز، كلمات توقف، توحيد الحروف)
│   ├── data.py          تحميل الجدول وتنظيفه (فارغ، تكرار، تعارض التسميات)
│   ├── train.py         تدريب ومقارنة النماذج واختيار الأفضل
│   ├── evaluate.py      Accuracy / Precision / Recall / F1 + مصفوفة الالتباس
│   ├── predict.py       تصنيف شكاوى جديدة
│   └── generate_sample_data.py   مولّد بيانات اصطناعية للتجربة
├── notebooks/           استكشاف اختياري
└── tests/               اختبارات pytest
```

## 1) متطلبات البيئة
- Python **3.9 أو أحدث** (تم الاختبار على 3.12)
- لا حاجة إلى GPU

## 2) التثبيت (من الصفر)
```bash
# انسخ المشروع أو فك ضغط ملف ZIP ثم ادخل المجلد
cd arabic-complaints-classifier

python -m venv .venv
source .venv/bin/activate          # على Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## 3) تجهيز البيانات
**البيانات الحالية:** الملف `data/complaints.csv` يحتوي 5,000 شكوى بالأعمدة `id`, `text`, `channel`, `created_at`, `category`, `sentiment`. يستخدم مصنف الفئة عمود `text` للتنبؤ بعمود `category`؛ يقبل أيضًا الملفات القديمة التي تستخدم `label` بدل `category`. الأعمدة الأخرى محفوظة في ملف CSV لكنها لا تدخل في تدريب مصنف الفئة، ولا يُدرّب النظام حاليًا مصنفًا للمشاعر.

يتعامل التحميل مع النصوص المفقودة أو الفارغة والتكرارات والتعارضات بعد تطبيع النص. القيم التي لا يمكن حسم تسميتها آليًا تُستبعد عند وجود النص نفسه بتصنيفات مختلفة. راجع [REPORT.md](REPORT.md) لتفاصيل التنظيف والنتائج. البيانات المرفقة شبيهة بالواقع وقد تحتوي على تسميات غير صحيحة؛ لا تعد نتيجة الاختبار ضمانًا للأداء على بيانات تشغيلية جديدة.

**الخيار ب – بيانات اصطناعية للتجربة:**
```bash
python -m src.generate_sample_data --n 2000 --out data/complaints.csv
```
> النتائج على البيانات الاصطناعية لا تمثل الأداء الحقيقي (انظر REPORT.md).

## 4) التدريب والتقييم
```bash
python -m src.train --data data/complaints.csv --out artifacts
```
يطبع مقارنة النماذج ويحفظ في `artifacts/`:
`model.joblib`, `metrics.json`, `model_comparison.csv`, `classification_report.txt`, `confusion_matrix.csv`, `errors.csv`.

## 5) التصنيف (الاستدعاء)
```bash
# نص واحد
python -m src.predict --text "الطلب تأخر عن الموعد ولم يصل حتى الآن"

# ملف كامل (يضيف عمود predicted_label)
python -m src.predict --input new_complaints.csv --output predictions.csv
```
أو من داخل بايثون:
```python
from src.predict import load_model, predict_texts
bundle = load_model("artifacts/model.joblib")
print(predict_texts(["لا أستطيع تسجيل الدخول"], bundle))
```

## 6) الاختبارات
```bash
python -m pytest -q
```

## 7) التسليم
```bash
# ZIP (استبعد .venv)
zip -r complaints-classifier.zip . -x ".venv/*" "__pycache__/*" "*/__pycache__/*"
```

### الرفع إلى GitHub
أنشئ مستودعًا فارغًا على GitHub، ثم نفّذ الأوامر التالية من مجلد المشروع في PowerShell:
```powershell
git init
git branch -M main
git add .
git status
git commit -m "Initial project setup"
git remote add origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin main
```
استبدل `USERNAME/REPOSITORY` بمسار المستودع الذي أنشأته. لا يستبعد `.gitignore` أي ملفات من المشروع؛ لذلك ستُرفع الملفات المؤقتة والمحلية أيضًا إن وُجدت.

## ملاحظات
- إذا ظهرت الحروف العربية مقطّعة في الطرفية، فهذا عرض فقط؛ الملفات محفوظة بـ UTF-8.
- للتبديل إلى AraBERT: استبدل خط الأنابيب في `build_models` بنموذج `transformers` (انظر "الخطوات القادمة" في REPORT.md).
