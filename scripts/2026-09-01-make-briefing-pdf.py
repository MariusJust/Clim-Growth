"""One-page briefing PDF: projection findings, uncertainty bands, and the
implausible country-level projections implied by Burke's regression models.

Outputs: paper/Briefings/2026-09-01-projections-briefing.pdf
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate,
                               Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/Briefings"
OUT.mkdir(parents=True, exist_ok=True)
PDF = OUT / "2026-09-01-projections-briefing.pdf"

NAVY = colors.HexColor("#123f5f")
RED = colors.HexColor("#b23a48")
GREY = colors.HexColor("#5b6570")
LINE = colors.HexColor("#d5dae0")

ss = getSampleStyleSheet()
H = ParagraphStyle("H", parent=ss["Title"], fontSize=15, leading=18,
                   textColor=NAVY, alignment=0, spaceAfter=1)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontSize=8.5, textColor=GREY,
                     spaceAfter=9)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=10.5, leading=13,
                    textColor=NAVY, spaceBefore=9, spaceAfter=4)
BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontSize=8.8, leading=11.6)
NOTE = ParagraphStyle("NOTE", parent=ss["Normal"], fontSize=7.8, leading=10,
                      textColor=GREY)


def table(data, widths, align_right_from=1, size=8.2, head=True):
    t = Table(data, colWidths=widths, hAlign="LEFT")
    st = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), size),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
        ("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, LINE),
    ]
    if head:
        st += [("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
               ("TEXTCOLOR", (0, 0), (-1, 0), NAVY),
               ("LINEBELOW", (0, 0), (-1, 0), 0.7, NAVY)]
    t.setStyle(TableStyle(st))
    return t


def main():
    doc = SimpleDocTemplate(str(PDF), pagesize=A4,
                            leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=13 * mm, bottomMargin=13 * mm,
                            title="Projections briefing", author="Marius Leisgaard Just")
    S = []
    S.append(Paragraph("Out-of-sample projections: findings and uncertainty", H))
    S.append(Paragraph(
        "RCP 8.5 / SSP5, Burke (2015) harness, corrected climate data. Response capped at "
        "f(min(T,30&deg;C)). Bands are 95% country-cluster bootstrap, 1000 draws, "
        "clustered on 195 countries. 162 countries in the projection.", SUB))

    # ---------------- headline ------------------------------------------
    S.append(Paragraph("1. Headline: 2099 change in global GDP per capita", H2))
    S.append(table([
        ["Model", "Point", "95% band", "Median country", "Pop. harmed"],
        ["Neural network", "+13.2%", "[-50.9, +316.6]", "-66.7%", "64%"],
        ["Burke quadratic, our data", "-7.8%", "[-56.5, +165.6]", "-72.1%", "87%"],
        ["Leirvik, our data", "-9.0%", "[-55.2, +170.7]", "-68.7%", "87%"],
        ["Burke published (validation)", "-22.8%", "not bootstrapped", "-76.8%", "88%"],
    ], [52 * mm, 20 * mm, 32 * mm, 30 * mm, 22 * mm]))
    S.append(Spacer(1, 5))
    S.append(Paragraph(
        "<b>The 2099 aggregate is uninformative.</b> A band spanning -51% to +317% cannot "
        "discriminate between any two models. The arms are close at short horizons and "
        "diverge only through compounding: they differ by 2.3 pp in 2040, 6.4 pp in 2060, "
        "and 21 pp in 2099.", BODY))
    S.append(Spacer(1, 3))
    S.append(Paragraph(
        "<b>The aggregate also hides the incidence.</b> In every model the median country "
        "loses roughly 70% of GDP per capita while the population-weighted mean stays near "
        "zero, because a few cold countries compound large gains.", BODY))

    # ---------------- Burke's implausible projections --------------------
    S.append(Paragraph("2. Implausible country-level projections implied by the "
                       "Burke regression model", H2))
    S.append(Paragraph(
        "Burke's published coefficients, run through his own projection code, imply the "
        "following 2099 outcomes relative to a no-climate-change baseline. These are not "
        "our results; they are what his specification produces.", BODY))
    S.append(Spacer(1, 4))
    S.append(table([
        ["Country", "Baseline T", "2099 change", "GDP/cap multiple", "Burke, our data"],
        ["Mongolia", "-0.7 C", "+1333%", "x14.3", "+1801%"],
        ["Finland", "+3.7 C", "+495%", "x5.9", "+638%"],
        ["Iceland", "+1.5 C", "+490%", "x5.9", "+420%"],
        ["Russia", "+4.3 C", "+403%", "x5.0", "+531%"],
        ["Estonia", "+5.6 C", "+250%", "x3.5", "+336%"],
        ["Norway", "+4.8 C", "+241%", "x3.4", "+359%"],
    ], [40 * mm, 24 * mm, 26 * mm, 32 * mm, 34 * mm]))
    S.append(Spacer(1, 5))
    S.append(Paragraph(
        "Mongolia's GDP per capita reaches <b>USD 369,310</b> in 2099 under climate change "
        "against USD 25,770 without it &mdash; i.e. warming makes Mongolia richer than any "
        "country today, purely as a by-product of a quadratic fitted on 1960-2010 growth "
        "rates. At the other end, Saudi Arabia -95.5%, Kuwait -95.3%, Oman -94.0%, "
        "UAE -93.6%.", BODY))
    S.append(Spacer(1, 3))
    S.append(Paragraph(
        "<b>Why this happens.</b> Temperature enters as a permanent growth effect, so a "
        "country below the estimated optimum compounds a positive growth premium for 89 "
        "consecutive years. Nothing bounds it. The published headline of -23% is a "
        "population-weighted mean of exactly these numbers: 126 of 165 countries are hurt, "
        "88% of world population is hurt, and the median country is at -76.8%.", BODY))

    # ---------------- what moves the number -----------------------------
    S.append(Paragraph("3. Fragility: what moves the 2099 number", H2))
    S.append(table([
        ["Change (each individually defensible)", "Effect on 2099"],
        ["WDI growth-data vintage", "+16.4 pp"],
        ["Corrected population-weighted climate data", "+29.0 pp"],
        ["Extending the sample to 2024", "-18.6 pp"],
        ["Evaluating at own baseline temperatures rather than UDel", "-11.2 pp"],
        ["Removing the 30 C cap", "-0.3 to -0.7 pp"],
    ], [110 * mm, 30 * mm]))
    S.append(Spacer(1, 4))
    S.append(Paragraph(
        "Individually defensible choices move the century-scale headline by 16-29 pp each. "
        "The estimator itself is validated: refitting on Burke's own data recovers "
        "b<sub>T</sub> = +0.012680 and b<sub>T2</sub> = -0.0004942 against his published "
        "+0.0127184 and -0.0004871, and the harness reproduces -22.77%.", BODY))

    # ---------------- response function ---------------------------------
    S.append(Paragraph("4. Response function and its band", H2))
    S.append(table([
        ["Quantity", "Value"],
        ["Point-estimate optimum", "17.87 C"],
        ["Bootstrap median optimum", "17.53 C"],
        ["95% interval for the optimum", "[10.10, 27.98] C"],
        ["Draws with an interior optimum", "96.5%"],
        ["Burke quadratic optimum, same data", "14.55 C"],
    ], [110 * mm, 30 * mm]))
    S.append(Spacer(1, 4))
    S.append(Paragraph(
        "The concave shape is robust (96.5% of draws) and the bootstrap median sits on the "
        "point estimate, but the optimum is only weakly pinned down. <b>Both Burke "
        "quadratics lie inside the network's 95% band across the entire 0-30 C range</b>, "
        "so the network's response is not statistically distinguishable from a quadratic. "
        "On identical draws the quadratic's key contrasts are significant and the network's "
        "are not &mdash; f(30)-f(20) = -10.34 [-16.48, -3.66] for Burke versus "
        "-9.84 [-17.48, +0.43] for the network &mdash; i.e. flexibility costs power.", BODY))

    S.append(Spacer(1, 7))
    S.append(Paragraph(
        "Sources: runs/bootstrap/2026-08-29_20-56-38global_IC (1000/1000 draws, node (2,)); "
        "runs/estimation/2026-08-27_11-01-35_global_IC_country_trends; "
        "scripts/2026-08-30-projections-bands.py; scripts/2026-08-30-bootstrap-bands.py. "
        "Burke published arm reproduces -22.77% against his -23%.", NOTE))

    doc.build(S)
    print(f"wrote {PDF}")


if __name__ == "__main__":
    main()
