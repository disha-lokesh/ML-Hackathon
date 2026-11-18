"""Build report/PathSnap_Run_Guide.pdf from report/RUN_GUIDE.md with ReportLab.

Supports the small markdown subset used in the guide: #/##/### headings, paragraphs,
"- " bullets, "1. " numbered lists, **bold**, `inline code`, a line that is only
`a command` (drawn as a shaded code box) and pipe tables.

python report/build_run_guide.py
"""
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HERE = Path(__file__).resolve().parent
FONTS = Path("C:/Windows/Fonts")

# Unicode TTF fonts when available (Windows), otherwise the ReportLab built-ins
if (FONTS / "arial.ttf").exists():
    pdfmetrics.registerFont(TTFont("Body", str(FONTS / "arial.ttf")))
    pdfmetrics.registerFont(TTFont("Body-Bold", str(FONTS / "arialbd.ttf")))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body", boldItalic="Body-Bold")
    BODY, BOLD = "Body", "Body-Bold"
else:
    BODY, BOLD = "Helvetica", "Helvetica-Bold"
if (FONTS / "consola.ttf").exists():
    pdfmetrics.registerFont(TTFont("Mono", str(FONTS / "consola.ttf")))
    MONO = "Mono"
else:
    MONO = "Courier"

ACCENT = colors.HexColor("#1f4e79")
ss = getSampleStyleSheet()
S = {
    "h1": ParagraphStyle("h1", parent=ss["Title"], fontName=BOLD, fontSize=20, leading=24, textColor=ACCENT,
                         alignment=TA_LEFT, spaceAfter=6),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName=BOLD, fontSize=14, leading=18, textColor=ACCENT,
                         spaceBefore=10, spaceAfter=4),
    "h3": ParagraphStyle("h3", parent=ss["Heading3"], fontName=BOLD, fontSize=11.5, leading=15, textColor=ACCENT,
                         spaceBefore=6, spaceAfter=3),
    "p": ParagraphStyle("p", parent=ss["BodyText"], fontName=BODY, fontSize=10, leading=14, spaceAfter=4),
    "code": ParagraphStyle("code", fontName=MONO, fontSize=9.5, leading=12.5),
    "cell": ParagraphStyle("cell", fontName=BODY, fontSize=9, leading=12),
    "cellh": ParagraphStyle("cellh", fontName=BOLD, fontSize=9, leading=12, textColor=colors.white),
}


def inline(text):
    """markdown inline -> ReportLab paragraph markup."""
    parts = re.split(r"(`[^`]+`)", text)
    out = []
    for part in parts:
        if part.startswith("`") and part.endswith("`"):
            out.append(f'<font name="{MONO}" color="#8b1a1a">{html.escape(part[1:-1])}</font>')
        else:
            t = html.escape(part)
            t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
            out.append(t)
    return "".join(out)


def code_box(cmd):
    t = Table([[Paragraph(html.escape(cmd), S["code"])]], colWidths=[170 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f3f5")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#c8ccd0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def make_list(items, ordered):
    flow = [ListItem(Paragraph(inline(it), S["p"]), leftIndent=14) for it in items]
    kw = dict(bulletType="1", bulletFontName=BODY, bulletFontSize=10) if ordered else \
        dict(bulletType="bullet", start="\u2022", bulletFontName=BODY, bulletFontSize=10)
    return ListFlowable(flow, leftIndent=14, **kw)


def make_table(rows):
    data = [[Paragraph(inline(c), S["cellh"]) for c in rows[0]]]
    data += [[Paragraph(inline(c), S["cell"]) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=[60 * mm, 110 * mm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fa")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8ccd0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def parse(md):
    story, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), S["h3"])); i += 1
        elif line.startswith("## "):
            # keep a heading on the same page as what follows it
            story.append(KeepTogether([Paragraph(inline(line[3:]), S["h2"])])); i += 1
        elif line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), S["h1"])); i += 1
        elif re.fullmatch(r"`[^`]+`", line):
            box = code_box(line[1:-1])
            if story and isinstance(story[-1], Paragraph):
                box = KeepTogether([story.pop(), box])   # keep "Terminal 2 starts the UI:" with its command
            story += [box, Spacer(1, 5)]; i += 1
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            story += [make_table(rows), Spacer(1, 6)]
        elif re.match(r"(- |\d+\. )", line):
            ordered = bool(re.match(r"\d+\. ", line))
            items = []
            while i < len(lines) and re.match(r"(- |\d+\. )", lines[i]):
                items.append(re.sub(r"^(- |\d+\. )", "", lines[i].rstrip())); i += 1
            story += [make_list(items, ordered), Spacer(1, 4)]
        else:
            para = [line]; i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"(#|- |\d+\. |\||`[^`]+`$)", lines[i]):
                para.append(lines[i].rstrip()); i += 1
            story.append(Paragraph(inline(" ".join(para)), S["p"]))
    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(BODY, 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(20 * mm, 10 * mm, "PathSnap - Run Guide")
    canvas.drawRightString(190 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


if __name__ == "__main__":
    out = HERE / "PathSnap_Run_Guide.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm, title="PathSnap Run Guide")
    doc.build(parse((HERE / "RUN_GUIDE.md").read_text(encoding="utf-8")), onFirstPage=footer, onLaterPages=footer)
    print("wrote", out)
