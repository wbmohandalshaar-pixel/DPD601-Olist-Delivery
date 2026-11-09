"""Readable Arabic/English typography and confirmed team credits for the generated report.
Text revisions are keyed to the original generated paragraphs and only applied on exact matches;
no metric is recalculated here. Re-run run_pipeline.py before build_report.py when data change.
"""
from pathlib import Path
import json,copy,re
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
HERE=Path(__file__).resolve().parent
TEAM=[
 ('مهند الشعار (Mohanad Alshaar)','mohanad_118315','تكامل المشروع وتشغيله على Windows، إدارة Git وتجهيز بيئة التشغيل.'),
 ('بيان علي إيتي (Bayan Ali Ite)','bayan_117286','تجهيز البيانات وتنظيفها والتحليل الاستكشافي وإعداد الرسوم.'),
 ('مازن اليافي (Mazen Alyafi)','mazen_216443','بناء النماذج ومقارنتها زمنياً وتقييم أدائها وتحليل الأخطاء.'),
 ('محمد القصير (Muhammad Al-Qasir)','Muhammad_172653','إعداد التقرير والتوثيق ومراجعة الاختبارات وتجهيز المناقشة.')]
def bidi(p):
 pr=p._p.get_or_add_pPr();e=pr.find(qn('w:bidi'))
 if e is None:e=OxmlElement('w:bidi');pr.append(e)
 e.set(qn('w:val'),'1')
def set_font(run,size=None,bold=None):
 run.font.name='Liberation Sans'
 if size is not None:run.font.size=Pt(size)
 if bold is not None:run.bold=bold
 f=run._element.get_or_add_rPr().rFonts
 if f is None:
  f=OxmlElement('w:rFonts');run._element.get_or_add_rPr().insert(0,f)
 f.set(qn('w:ascii'),'Liberation Sans');f.set(qn('w:hAnsi'),'Liberation Sans');f.set(qn('w:cs'),'Noto Naskh Arabic');f.set(qn('w:eastAsia'),'Liberation Sans')
def finalize(path):
 path=Path(path)
 d=Document(path)
 alterations=json.loads((HERE/'report_text_revisions.json').read_text(encoding='utf-8'))
 for key,item in alterations.items():
  n=int(key)
  if n>=len(d.paragraphs):continue
  p=d.paragraphs[n]
  if p.text==item['source'] and p.runs:
   p.runs[0].text=item['revised']
   for r in p.runs[1:]:r.text=''
 # Team credits on first page: actual team IDs, confirmed roles, without invented review history.
 table=d.tables[0]
 for row,member in zip(table.rows[1:],TEAM):
  row.cells[0].text=member[0];row.cells[1].text=member[1]
 # Explanatory role paragraph near the end, direct and consistent.
 for p in d.paragraphs:
  if p.text.startswith('يوضح الجدول التالي توزيع العمل') or p.text.startswith('أفاد الفريق بأن العمل'):
   p.text=('توزّع العمل بين أعضاء الفريق بحسب ما أكده الفريق: تولّى مهند التكامل والتشغيل وإدارة Git، '
     'وبيان تجهيز البيانات وتنظيفها والتحليل الاستكشافي، ومازن بناء النماذج وتقييم نتائجها، '
     'ومحمد التقرير والتوثيق ومراجعة الاختبارات. وترد المسؤوليات في TEAM_CONTRIBUTIONS.csv. '
     'أما مراجعات كود الزملاء فتحتاج إلى أدلتها الفردية وفق التكليف.')
 # Add a readable roles table below the team-contribution paragraph, before the references.
 exists=any('الطالب والرقم الجامعي' in c.text for tab in d.tables for row in tab.rows[:1] for c in row.cells)
 if not exists:
  T=d.add_table(rows=1,cols=2)
  T.style='Table Grid'
  T.rows[0].cells[0].text='الطالب والرقم الجامعي';T.rows[0].cells[1].text='مسؤوليات العمل'
  for name,sid,role in TEAM:
   cs=T.add_row().cells;cs[0].text=name+'\n'+sid;cs[1].text=role
  p=next((p for p in d.paragraphs if p.text.startswith('توزّع العمل بين أعضاء الفريق')),None)
  d.tables[0]._tbl.addnext(T._tbl)
 # Clear a redundant empty paragraph directly following the cover team table.
 if len(d.paragraphs)>8 and not d.paragraphs[7].text.strip() and d.paragraphs[8].style.name.startswith('Heading'):
  el=d.paragraphs[7]._element;el.getparent().remove(el)
 s=d.sections[0];s.page_width=Inches(8.5);s.page_height=Inches(11)
 s.left_margin=Inches(.72);s.right_margin=Inches(.72)
 s.top_margin=Inches(.61);s.bottom_margin=Inches(.65)
 styles=d.styles
 for nm,size,color in [('Normal',10.5,(25,36,46)),('Title',19,(17,42,57)),('Subtitle',12,(55,68,76)),('Heading 1',14,(25,67,88)),('Heading 2',12,(25,67,88))]:
  st=styles[nm];st.font.size=Pt(size);st.font.name='Noto Naskh Arabic';st.font.color.rgb=RGBColor(*color)
  st.paragraph_format.space_after=Pt(5 if nm=='Normal' else 9)
  st.paragraph_format.line_spacing=1.12
 for p in d.paragraphs:
  if not p.text.strip():continue
  p.alignment=WD_ALIGN_PARAGRAPH.RIGHT;bidi(p)
  if p.style.name.startswith('Heading') or p.style.name in ('Title','Subtitle'):
   p.paragraph_format.keep_with_next=True
   if p.style.name=='Heading 1':
    p.paragraph_format.page_break_before=False
  for r in p.runs:
   set_font(r)
   if p.style.name=='Title':r.font.size=Pt(19)
   elif p.style.name=='Subtitle':r.font.size=Pt(12)
   elif p.style.name=='Heading 1':r.font.size=Pt(14);r.bold=True
   elif p.style.name=='Heading 2':r.font.size=Pt(12);r.bold=True
 for tab in d.tables:
  tab.autofit=False
  for ri,row in enumerate(tab.rows):
   trpr=row._tr.get_or_add_trPr()
   if trpr.find(qn('w:cantSplit')) is None: trpr.append(OxmlElement('w:cantSplit'))
   if ri==0 and trpr.find(qn('w:tblHeader')) is None: trpr.append(OxmlElement('w:tblHeader'))
   for cell in row.cells:
    for p in cell.paragraphs:
     p.alignment=WD_ALIGN_PARAGRAPH.CENTER
     if re.search('[\u0600-\u06FF]',p.text):bidi(p)
     p.paragraph_format.space_after=Pt(2);p.paragraph_format.space_before=Pt(2)
     for r in p.runs:set_font(r,9 if ri else 9.2,bold=(ri==0))
 cp=d.core_properties;cp.title='تحليل التجارة الإلكترونية والتنبؤ بتأخر تسليم الطلبات';cp.subject='مشروع برمجة خاصة بعلم البيانات DPD601';cp.author='فريق المشروع - الشعبة C1';cp.keywords='DPD601, Olist, delivery analysis';cp.comments=''
 d.save(path)
 return path
if __name__=='__main__':finalize(HERE/'Report_AR.docx')
