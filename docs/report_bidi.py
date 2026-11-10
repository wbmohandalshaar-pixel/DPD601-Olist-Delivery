"""Normalize Arabic RTL with isolated Latin identifiers/numerals for Word + PDF.

This is a typography-only pass. Numbers and scientific results are not recomputed.
Calls are safe to repeat: older direction marks are stripped before rebuilding runs.
"""
from copy import deepcopy
from re import compile
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
ARABIC=compile('[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]')
LATIN=compile(r'(?:[A-Za-z][A-Za-z0-9_./:+%\-]*[A-Za-z0-9_%]|[A-Za-z])|(?:\d(?:[\d,./:+\-]*\d)?%?)')
STRIP_MARKS=dict.fromkeys(map(ord,'\u200e\u200f\u2066\u2067\u2068\u2069\u202a\u202b\u202c'),None)

def strip_marks(text):
    return text.translate(STRIP_MARKS)

def _set_bidi(paragraph, enabled):
    ppr=paragraph._p.get_or_add_pPr()
    element=ppr.find(qn('w:bidi'))
    if element is None:element=OxmlElement('w:bidi');ppr.append(element)
    element.set(qn('w:val'),'1' if enabled else '0')

def _style_run(run, is_arabic):
    run.font.rtl=is_arabic
    run.font.name='Arial'
    rpr=run._r.get_or_add_rPr()
    rf=rpr.find(qn('w:rFonts'))
    if rf is None:rf=OxmlElement('w:rFonts');rpr.insert(0,rf)
    for tag in ('ascii','hAnsi','cs','eastAsia'):rf.set(qn('w:'+tag),'Arial')
    lang=rpr.find(qn('w:lang'))
    if lang is None:lang=OxmlElement('w:lang');rpr.append(lang)
    lang.set(qn('w:val'),'ar-SY' if is_arabic else 'en-US')

def normalize_report_direction(doc):
    from docx.text.run import Run
    from docx.shared import Pt
    # Avoid prepositions sticking to numbers, and make the summary's language natural.
    for p in doc.paragraphs:
        text=strip_marks(p.text)
        fixed=text
        if text.startswith('تضم البيانات الأصلية سبعة جداول مرتبطة وفيها 99,441 طلباً.'):
            fixed=text.replace(
                'تضم البيانات الأصلية سبعة جداول مرتبطة وفيها 99,441 طلباً. وبعد تدقيق الحالات والتواريخ أمكن تحديد نتيجة التسليم لـ96,394 طلباً، تأخر منها 6,532 طلباً (6.78%).',
                'تضم البيانات الأصلية سبعة جداول مرتبطة، تحتوي على 99,441 طلبًا. وبعد تدقيق الحالات والتواريخ أمكن تحديد نتيجة التسليم لعدد 96,394 طلبًا، تأخر منها 6,532 طلبًا، أي بنسبة 6.78%.')
        elif text.startswith('اعتماداً على متوسط الأداء في فترات التحقق الزمنية اخترنا LogisticLog_C1_plain.'):
            fixed=text.replace(
                'اعتماداً على متوسط الأداء في فترات التحقق الزمنية اخترنا LogisticLog_C1_plain. على فترة التقييم الأحدث وصل الاستدعاء إلى 53.83% والإحكام إلى 9.44%، وبلغ F1 نحو 0.1606.',
                'اعتمادًا على متوسط الأداء في فترات التحقق الزمنية، اخترنا النموذج LogisticLog_C1_plain. وفي فترة التقييم الأحدث بلغت قيمة الاستدعاء 53.83%، والإحكام 9.44%، وبلغ مقياس F1 نحو 0.1606.')
        if fixed!=text and p.runs and not any(r._r.xpath('.//w:drawing|.//w:object') for r in p.runs):
            p.runs[0].text=fixed
            for run in p.runs[1:]:run.text=''

    paragraphs=[(p,False) for p in doc.paragraphs]
    paragraphs.extend((p,True) for tbl in doc.tables for row in tbl.rows
                      for cell in row.cells for p in cell.paragraphs)
    for section in doc.sections:
        paragraphs.extend((p,False) for part in (section.header,section.footer) for p in part.paragraphs)
    for p,in_table in paragraphs:
        original=strip_marks(p.text)
        if not original.strip():continue
        arabic=bool(ARABIC.search(original))
        _set_bidi(p,arabic)
        if p.alignment!=WD_ALIGN_PARAGRAPH.CENTER:
            if not in_table or p.alignment is None:
                p.alignment=WD_ALIGN_PARAGRAPH.RIGHT if arabic else WD_ALIGN_PARAGRAPH.LEFT
        for run in list(p.runs):
            if not run.text:continue
            if any(el.tag not in (qn('w:rPr'),qn('w:t')) for el in run._r):continue
            source=strip_marks(run.text)
            pieces=[];pos=0
            for match in LATIN.finditer(source):
                if match.start()>pos:
                    chunk=source[pos:match.start()]
                    pieces.append((chunk,bool(ARABIC.search(chunk))))
                pieces.append((match.group(0),False));pos=match.end()
            if pos<len(source):
                chunk=source[pos:]
                pieces.append((chunk,bool(ARABIC.search(chunk))))
            if not pieces:pieces=[(source,arabic)]
            for fragment,is_ar in pieces:
                if not fragment:continue
                xml=OxmlElement('w:r')
                if run._r.rPr is not None:xml.append(deepcopy(run._r.rPr))
                wtext=OxmlElement('w:t')
                wtext.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
                wtext.text=('\u2066'+fragment+'\u2069' if arabic and not is_ar and LATIN.fullmatch(fragment) else fragment)
                xml.append(wtext);run._r.addprevious(xml)
                _style_run(Run(xml,p),is_ar)
            run._r.getparent().remove(run._r)
    for tbl in doc.tables:
        prop=tbl._tbl.tblPr
        direction=prop.find(qn('w:bidiVisual'))
        if direction is None:direction=OxmlElement('w:bidiVisual');prop.append(direction)
        direction.set(qn('w:val'),'1')
        for ix,row in enumerate(tbl.rows):
            rp=row._tr.get_or_add_trPr()
            if rp.find(qn('w:cantSplit')) is None:rp.append(OxmlElement('w:cantSplit'))
            if ix==0 and rp.find(qn('w:tblHeader')) is None:rp.append(OxmlElement('w:tblHeader'))
    for name in ('Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3'):
        if name in doc.styles:doc.styles[name].font.name='Arial'
    # Arabic text uses the complex-script size, not only w:sz. Keep the title on one line.
    if doc.paragraphs and doc.paragraphs[0].style.name=='Title':
        first=doc.paragraphs[0]
        first.style.font.size=Pt(16)
        for rr in first.runs:rr.font.size=Pt(16)
        for element in [first.style._element.rPr]+[r._r.get_or_add_rPr() for r in first.runs]:
            size=element.find(qn('w:szCs'))
            if size is None:size=OxmlElement('w:szCs');element.append(size)
            size.set(qn('w:val'),'32')
    for p in doc.paragraphs:
        if strip_marks(p.text).strip()=='الأسئلة التي بدأنا منها':
            p.paragraph_format.page_break_before=True
        if p.style.name.startswith('Heading'):
            p.paragraph_format.keep_with_next=True
    return doc
