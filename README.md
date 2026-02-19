# AI Engine

مشروع بسيط لتصنيف الصور والبحث عن أقرب فئة باستخدام:
- **Torch / torchvision** لاستخراج السمات والتصنيف.
- **FAISS** لبناء فهرس متجهات والبحث السريع.
- **Gradio** لواجهة تنبؤ تفاعلية.

## المتطلبات

- Python 3.10+
- تثبيت الحزم:

```bash
pip install -r requirements.txt
```

## هيكل المشروع

- `main.py`: واجهة أوامر CLI.
- `app.py`: تطبيق Gradio للتصنيف.
- `core/build_index.py`: بناء FAISS index من الصور.
- `core/search.py`: البحث عن أقرب فئة لصورة جديدة.
- `services/predict.py`: تغليف منطق التنبؤ عبر CLI.
- `utils/config.py`: إعدادات المسارات والثوابت.

## طريقة الاستخدام

### 1) تدريب الموديل (إنشاء `best_model.pth`)

```bash
python core/train.py --data-dir data --epochs 5
```

### 2) بناء الفهرس

```bash
python main.py --build
```

### 3) التنبؤ لصورة

```bash
python main.py --predict /path/to/image.jpg
```

### 4) تشغيل واجهة Gradio

```bash
python app.py
```

## ملاحظات مهمة

- تأكد من وجود بيانات الصور داخل مجلد `data/` حسب الفئات.
- إذا لم يكن `best_model.pth` موجودًا، شغّل `python core/train.py` أولًا لإنشائه.
- في حال عدم توفر `faiss` أو `gradio` ستفشل أوامر مرتبطة بها وقت التشغيل.
