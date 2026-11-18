"""Build a PDF from a markdown file in report/ (markdown -> HTML -> fpdf2).

python report/build_report.py                                   # REPORT.md -> PathSnap_Report.pdf
python report/build_report.py RUN_GUIDE.md PathSnap_Run_Guide.pdf
"""
import re
import sys
from pathlib import Path

import markdown
from fpdf import FPDF

HERE = Path(__file__).resolve().parent
SRC = sys.argv[1] if len(sys.argv) > 1 else "REPORT.md"
DST = sys.argv[2] if len(sys.argv) > 2 else "PathSnap_Report.pdf"
md = (HERE / SRC).read_text(encoding="utf-8")
# resolve image paths relative to the report folder
md = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", lambda m: f'<img src="{(HERE / m.group(2)).resolve().as_posix()}" width="520">', md)
# python-markdown needs a blank line before a list that follows a paragraph line
md = re.sub(r"(?m)^([^\n\-\d >|][^\n]*)\n(- |\d+\. )", r"\1\n\n\2", md)
html = markdown.markdown(md, extensions=["tables"])
html = html.replace("<table>", '<table border="1" cellpadding="2" width="100%" style="font-size:7pt">')
html = html.replace("<blockquote>", "<i>").replace("</blockquote>", "</i>")
html = html.replace("<hr />", "")
# fpdf2 does not allow inline tags inside table cells
html = re.sub(r"(<t[dh][^>]*>)(.*?)(</t[dh]>)",
              lambda m: m.group(1) + re.sub(r"</?(strong|em|code|b|i)>", "", m.group(2)) + m.group(3), html, flags=re.S)

pdf = FPDF(format="A4")
fonts = Path("C:/Windows/Fonts")
reg = next((fonts / f for f in ["arial.ttf", "segoeui.ttf"] if (fonts / f).exists()), None)
if reg:
    pdf.add_font("Body", "", str(reg))
    pdf.add_font("Body", "B", str(fonts / "arialbd.ttf") if (fonts / "arialbd.ttf").exists() else str(reg))
    pdf.add_font("Body", "I", str(fonts / "ariali.ttf") if (fonts / "ariali.ttf").exists() else str(reg))
    pdf.add_font("Body", "BI", str(fonts / "arialbi.ttf") if (fonts / "arialbi.ttf").exists() else str(reg))
    sym = fonts / "seguisym.ttf"
    if sym.exists():
        pdf.add_font("Sym", "", str(sym))
        pdf.set_fallback_fonts(["Sym"])
    family = "Body"
    mono = fonts / "consola.ttf"
    for style in ["", "B", "I", "BI"]:
        pdf.add_font("Mono", style, str(mono if mono.exists() else reg))
else:
    family = "helvetica"
pdf.set_auto_page_break(True, margin=15)
pdf.add_page()
pdf.set_font(family, size=9)
pdf.write_html(html, font_family=family, table_line_separators=True, pre_code_font="Mono" if reg else "courier")
out = HERE / DST
pdf.output(str(out))
print("wrote", out)
