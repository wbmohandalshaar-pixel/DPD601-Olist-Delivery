# إدارة نسخ مشروع DPD601 على GitHub

المستودع الفعلي: https://github.com/wbmohandalshaar-pixel/DPD601-Olist-Delivery

يستخدم المشروع فرع `main`. بحسب سجل الأمر الذي قدمه صاحب المشروع، رفع Commit `cb85496` إلى هذا الفرع بنجاح. لا ينبغي إعادة كتابة سجل Git التاريخي أو إنشاء Commits مؤرخة بأثر رجعي.

لرفع الوثائق والشفرة المعدلة بعد مراجعتها، من داخل `C:\Projects\DPD601_git`:

```cmd
cd /d C:\Projects\DPD601_git
git remote -v
git status
git add README.md PERSONAL_CHECKLIST_AR.md docs
git status
git commit -m "Finalize report, team responsibilities and submission documentation"
git push github main
```

**مهم:** أمر `git add docs` يضيف أي ملف داخل `docs` تغيّر محلياً؛ افحص `git status` قبل التسجيل. إذا احتاج التقرير إلى تحويل PDF بعد تعديل Word، ضع النسختين النهائيّتين قبل الـCommit. ملفات البيانات الكبيرة والنموذج والمخرجات ليست ضمن Git لأن `.gitignore` يستثنيها، لكنها مضمنة في ZIP التسليم. لا تستخدم `--force`.

لا يثبت سجل Commit واحد مراجعات الطلاب الأربعة. لكل عضو مراجعة فعلية موثقة، مع عنوان الملف وملخص الفحص ودليل GitHub إن أمكن. مستند `docs/PEER_REVIEW_TASKS_AR.md` يساعد على تنفيذها.
