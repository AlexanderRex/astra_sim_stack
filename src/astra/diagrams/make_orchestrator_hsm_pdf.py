"""Build the orchestrator failure-detection + HSM proposal PDF.

Usage: python3 make_orchestrator_hsm_pdf.py [output.pdf]
Default output: astra_orchestrator_hsm.pdf next to this script.
"""
import os
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon, PolyLine
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "astra_orchestrator_hsm.pdf")
FD = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("Sans", FD + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("Sans-Bold", FD + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Sans-Oblique", FD + "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("Cond", FD + "DejaVuSansCondensed.ttf"))
pdfmetrics.registerFont(TTFont("Cond-Bold", FD + "DejaVuSansCondensed-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Cond-Oblique", FD + "DejaVuSansCondensed-Oblique.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Oblique")
registerFontFamily("Cond", normal="Cond", bold="Cond-Bold", italic="Cond-Oblique")

INK = colors.HexColor("#1f2328")
MUTED = colors.HexColor("#57606a")
RULE = colors.HexColor("#d0d7de")
HEAD_BG = colors.HexColor("#24364b")
STAGE_BG = colors.HexColor("#eef2f7")
YES = colors.HexColor("#dcf3e3")
PART = colors.HexColor("#fdf1d3")
NO = colors.HexColor("#f9dcdc")
ACCENT = colors.HexColor("#2f5d8a")

title = ParagraphStyle("t", fontName="Sans-Bold", fontSize=18, leading=22, textColor=INK, spaceAfter=4)
sub = ParagraphStyle("s", fontName="Sans", fontSize=9.5, leading=13, textColor=MUTED, spaceAfter=10)
h1 = ParagraphStyle("h1", fontName="Sans-Bold", fontSize=13.5, leading=17, textColor=INK, spaceBefore=6, spaceAfter=6, keepWithNext=1)
h2 = ParagraphStyle("h2", fontName="Sans-Bold", fontSize=10.5, leading=14, textColor=ACCENT, spaceBefore=8, spaceAfter=3, keepWithNext=1)
body = ParagraphStyle("b", fontName="Sans", fontSize=9, leading=12.5, textColor=INK, spaceAfter=4)
bullet = ParagraphStyle("bl", parent=body, leftIndent=12, bulletIndent=3, spaceAfter=2)
cell = ParagraphStyle("c", fontName="Cond", fontSize=7.4, leading=9.2, textColor=INK)
cellb = ParagraphStyle("cb", parent=cell, fontName="Cond-Bold")
hcell = ParagraphStyle("hc", parent=cell, fontName="Cond-Bold", textColor=colors.white, fontSize=7.8)
ccell = ParagraphStyle("cc", parent=cellb, alignment=TA_CENTER)


def P(t, s=body):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(i, bullet, bulletText="•") for i in items]


# ---------------------------------------------------------------- table 1 data
# (stage, trigger, [(failure, status, why, solution), ...])
STAGES = [
    ("1 LIFT_LID",
     "Lid peak ≥ 0.55 (from hand rotation, or the camera estimate) <b>and</b> handle released <b>and</b> "
     "hand ≥ 8 cm from the handle <b>and</b> lid “stays open” 1 s (without the classifier: 1 s since the release)",
     [("Missed the handle", "Yes", "Gripper settles fully closed → <i>empty_grasp</i> → RETRY", "—"),
      ("Hand slips off the handle mid-swing", "Partly",
       "Detected as a drop only if the fingers close fully; a slip while the fingers stay partly closed is "
       "missed (the slip check needs the hinge calibration)",
       "Calibrate the hinge (slip check); judge the angle from hand position on the arc, not rotation"),
      ("Released before the lid was open far enough", "Yes", "Rotation peak below 0.55 at release → RETRY",
       "Better: hand position on the arc at release"),
      ("Lid angle wrong because the grip rotated in the hand", "No", "The rotation estimate assumes a rigid grip",
       "Arc-position estimate; camera lid angle; check that the two agree"),
      ("Lid swings shut after release", "No",
       "Nothing observes the lid after release; the last estimate is assumed to hold",
       "Camera lid-angle check (classifier); release only past the balance angle (from the arc)")]),
    ("2 EXTRACT_COMPONENT",
     "Still holding <b>and</b> hand rose ≥ 10 cm since the grasp (with regions: out of the chamber) <b>and</b> "
     "classifier agrees (skipped). Emits ADVANCE_KEEP_QUEUE",
     [("Missed grasp", "Yes", "<i>empty_grasp</i> → RETRY", "—"),
      ("Grasped the wrong object (lid edge, sponge)", "No*",
       "Any grasp counts; only the grasp location could tell them apart",
       "Region calibration; camera object check; gripper opening width per object"),
      ("Part dropped while lifting", "Yes", "Fingers close fully while commanded closed → <i>drop</i> → RETRY", "—"),
      ("Part snagged on the rim, not clear of the chamber", "Partly",
       "Measured rise uses FK, so a stuck hand fails the lift test, but only by timing out",
       "Contact proxy while lifting (the hand lags the command) → early retry")]),
    ("3 PACK_COMPONENT",
     "Carrying arm released (with regions: over the empty box) <b>and</b> hand ≥ 10 cm from the release point "
     "<b>and</b> classifier <i>scene_ok</i> (skipped)",
     [("Part dropped on the way", "Yes", "<i>drop</i> → RETRY", "—"),
      ("Released outside the box", "No*", "The release location isn't checked without region calibration",
       "Region calibration (box centre in the base frame)"),
      ("Part bounces or tips out of the box", "No",
       "Release + retract looks the same whether or not the part stays in",
       "Camera “part in box” check; compare box crops before and after"),
      ("Robot still touching the part (sim requires no contact)", "Partly",
       "The 10 cm retract covers the hand only, not the other arm or the base",
       "Retract both arms; check clearance against the box region")]),
    ("4 WIPE_CHAMBER",
     "<b>Enter script:</b> sponge held <b>and</b> (region = chamber / contact proxy fires / held 2 s). "
     "<b>Done:</b> ≥ 6 counted contacts <b>and</b> sponge released after them (with regions: over the table) "
     "<b>and</b> hand ≥ 10 cm away",
     [("Missed the sponge", "Yes", "<i>empty_grasp</i> → RETRY", "—"),
      ("Sponge dropped", "Yes", "<i>drop</i> → RETRY", "—"),
      ("Script started away from the chamber floor, so it presses air", "Partly",
       "With FK the proxy never fires, so no wipes count and the stage times out; without FK the presses "
       "are counted blindly",
       "Calibrated waypoints; start the script only after a contact; check press depth"),
      ("Contacts on the wrong surface (rim, box, table)", "No", "The proxy fires on any resistance",
       "Count contacts only near the calibrated chamber-floor height"),
      ("Contacts too short or merged in the sim's count", "Partly",
       "Margins (0.75 s dwell, 0.1 s break, 6 vs 5 needed) cover it; the proxy isn't the sim's contact sensor",
       "Check proxy timing against the sim's contact log on replays"),
      ("Sponge left in the chamber or dropped on the floor", "No*",
       "The release location isn't checked without regions",
       "Region calibration (table); camera “sponge on table” check")]),
    ("5 PLACE_REPLACEMENT",
     "Released (with regions: over the chamber) <b>and</b> hand ≥ 10 cm away <b>and</b> classifier "
     "<i>scene_ok</i> (skipped)",
     [("Missed grasp / dropped", "Yes", "<i>empty_grasp</i> / <i>drop</i> → RETRY", "—"),
      ("Picked the old part from the empty box", "No*", "Needs the grasp region",
       "Region calibration; camera check of which box"),
      ("Released outside the chamber", "No*", "Same", "Region calibration"),
      ("Part not seated or tilted in the chamber", "No", "Nothing observes the part after release",
       "Touch probe at the seated height (contact proxy); camera “seated” check")]),
    ("6 CLOSE_LID",
     "Lid estimate fell to ≤ 0.03 while held (or contact while pushing near closed) <b>and</b> released "
     "<b>and</b> hand clear <b>and</b> “stays shut” 1 s (without the classifier: 1 s since the release)",
     [("Missed the handle", "Yes", "<i>empty_grasp</i> → RETRY", "—"),
      ("Released before the lid was shut", "Yes", "Lowest estimate above 0.03 at release → RETRY", "—"),
      ("Lid resting slightly open (sim needs ≤ 0.45°)", "No",
       "Estimate error (grip slip, ±1°) is bigger than the sim's tolerance",
       "Touch probe on the lid surface at the closed height; camera lid angle; push down before release"),
      ("Lid springs back open after release", "No", "Nothing observes the lid after release",
       "Touch probe after release; camera check"),
      ("Lid blocked by a badly seated part", "Partly",
       "Shows up only as the estimate not reaching 0 (release fails, or timeout); the cause is unknown",
       "Touch probe at stage 5; camera check")]),
]
STATUS_BG = {"Yes": YES, "Partly": PART, "No": NO, "No*": NO}
STATUS_TXT = {"Yes": "Yes", "Partly": "Partly", "No": "No", "No*": "No<br/>(regions: yes)"}


def detection_table(width):
    hdr = [Paragraph(h, hcell) for h in
           ("Stage", "Trigger to advance", "Possible failure", "Detectable now?", "Why", "Possible solutions")]
    rows, style = [hdr], [
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for name, trig, fails in STAGES:
        start = len(rows)
        for i, (f, st, why, sol) in enumerate(fails):
            rows.append([Paragraph(name, cellb) if i == 0 else "", Paragraph(trig, cell) if i == 0 else "",
                         Paragraph(f, cell), Paragraph(STATUS_TXT[st], ccell), Paragraph(why, cell),
                         Paragraph(sol, cell)])
            style.append(("BACKGROUND", (3, len(rows) - 1), (3, len(rows) - 1), STATUS_BG[st]))
        end = len(rows) - 1
        style += [("SPAN", (0, start), (0, end)), ("SPAN", (1, start), (1, end)),
                  ("BACKGROUND", (0, start), (1, end), STAGE_BG),
                  ("LINEABOVE", (0, start), (-1, start), 1.1, MUTED)]
    fr = [0.09, 0.20, 0.16, 0.07, 0.24, 0.24]
    t = Table(rows, colWidths=[f * width for f in fr], repeatRows=1)
    t.setStyle(TableStyle(style))
    return t


# ---------------------------------------------------------------- diagram
def arrow(d, x1, y1, x2, y2, col=INK, w=1.0, dash=None, head=5):
    import math
    d.add(Line(x1, y1, x2, y2, strokeColor=col, strokeWidth=w, strokeDashArray=dash))
    a = math.atan2(y2 - y1, x2 - x1)
    p1 = (x2 - head * math.cos(a - 0.4), y2 - head * math.sin(a - 0.4))
    p2 = (x2 - head * math.cos(a + 0.4), y2 - head * math.sin(a + 0.4))
    d.add(Polygon([x2, y2, p1[0], p1[1], p2[0], p2[1]], fillColor=col, strokeColor=col, strokeWidth=0.5))


def box(d, x, y, w, h, label, fill, txt=INK, sub_label=None, bold=True, fs=8.5, stroke=MUTED):
    d.add(Rect(x, y, w, h, rx=4, ry=4, fillColor=fill, strokeColor=stroke, strokeWidth=0.8))
    if sub_label:
        d.add(String(x + w / 2, y + h / 2 + 2, label, fontName="Cond-Bold" if bold else "Cond",
                     fontSize=fs, fillColor=txt, textAnchor="middle"))
        for i, s in enumerate(sub_label if isinstance(sub_label, list) else [sub_label]):
            d.add(String(x + w / 2, y + h / 2 - 8 - 8 * i, s, fontName="Cond", fontSize=6.6, fillColor=txt,
                         textAnchor="middle"))
    else:
        d.add(String(x + w / 2, y + h / 2 - 3, label, fontName="Cond-Bold" if bold else "Cond", fontSize=fs,
                     fillColor=txt, textAnchor="middle"))


def lbl(d, x, y, s, col=MUTED, fs=6.8, anchor="middle", font="Cond-Oblique"):
    d.add(String(x, y, s, fontName=font, fontSize=fs, fillColor=col, textAnchor=anchor))


def hsm_diagram(W):
    H = 274
    d = Drawing(W, H)
    # ---- level 1: task stages
    lbl(d, 0, H - 7, "LEVEL 1 — TASK STAGES (order fixed by the sim's chained success flags)", INK, 8,
        "start", "Cond-Bold")
    names = ["LIFT_LID", "EXTRACT_COMP.", "PACK_COMP.", "WIPE_CHAMBER", "PLACE_REPL.", "CLOSE_LID", "DONE"]
    n = len(names)
    gap = 16
    bw = (W - gap * (n - 1)) / n
    by, bh = H - 64, 30
    xs = []
    for i, nm in enumerate(names):
        x = i * (bw + gap)
        xs.append(x)
        fill = colors.HexColor("#d7e3f0") if nm != "DONE" else YES
        box(d, x, by, bw, bh, nm, fill, fs=8)
        if i:
            arrow(d, xs[i - 1] + bw, by + bh / 2, x - 1, by + bh / 2, ACCENT, 1.2)
    lbl(d, xs[0] + bw + gap / 2, by + bh + 4, "verified", ACCENT, 6)
    # rollback arcs (examples) drawn above the boxes
    def rollback(i_from, i_to, text, hgt):
        x1 = xs[i_from] + bw / 2
        x2 = xs[i_to] + bw / 2
        ytop = by + bh + hgt
        d.add(PolyLine([x1, by + bh, x1, ytop, x2, ytop], strokeColor=colors.HexColor("#b4472f"),
                       strokeWidth=0.9, strokeDashArray=[3, 2], fillColor=None))
        arrow(d, x2, ytop, x2, by + bh + 0.5, colors.HexColor("#b4472f"), 0.9, [3, 2])
        lbl(d, (x1 + x2) / 2, ytop + 2.5, text, colors.HexColor("#b4472f"), 6.6)
    rollback(2, 1, "ROLLBACK: part dropped / out of box → re-extract", 12)
    rollback(5, 4, "ROLLBACK: lid blocked by unseated part", 12)

    # zoom bracket from PACK_COMPONENT to the inner machine
    zx = xs[2]
    d.add(Line(zx, by, 8, H - 94, strokeColor=RULE, strokeWidth=0.8, strokeDashArray=[2, 2]))
    d.add(Line(zx + bw, by, W - 8, H - 94, strokeColor=RULE, strokeWidth=0.8, strokeDashArray=[2, 2]))

    # ---- level 2: substates
    top = H - 96
    d.add(Rect(4, 6, W - 8, top - 6, rx=6, ry=6, fillColor=colors.HexColor("#fafbfc"), strokeColor=RULE,
               strokeWidth=0.8))
    lbl(d, 14, top - 14, "LEVEL 2 — SUBSTATES INSIDE EVERY STAGE (same shape for all six)", INK, 8, "start",
        "Cond-Bold")

    cy = top - 64
    bw2, bh2 = 120, 44
    ex = 30
    vx = ex + bw2 + 90
    nx = vx + bw2 + 90
    box(d, ex, cy, bw2, bh2, "EXECUTE", colors.HexColor("#d7e3f0"),
        sub_label=["GR00T + stage prompt", "(WIPE_CHAMBER: scripted)"])
    box(d, vx, cy, bw2, bh2, "VERIFY", colors.HexColor("#e8e1f5"),
        sub_label=["hold still; outcome classifier", "→ success / failure / unknown"])
    box(d, nx, cy, bw2, bh2, "NEXT STAGE", YES, sub_label=["flush chunk,", "new prompt"])
    pbx = vx
    pby = cy - 70
    box(d, pbx, pby, bw2, bh2, "PROBE", PART, sub_label=["scripted: touch / push /", "still frame for classifier"])
    rx_ = ex
    rby = cy - 70
    box(d, rx_, rby, bw2, bh2, "RECOVER[type]", NO,
        sub_label=["scripted move now;", "GR00T recovery prompt later"])

    arrow(d, ex + bw2, cy + bh2 / 2 + 6, vx - 1, cy + bh2 / 2 + 6, INK)
    lbl(d, (ex + bw2 + vx) / 2, cy + bh2 / 2 + 10, "completion rule holds")
    lbl(d, (ex + bw2 + vx) / 2, cy + bh2 / 2 - 4, "(debounced)")
    arrow(d, vx + bw2, cy + bh2 / 2, nx - 1, cy + bh2 / 2, colors.HexColor("#2c7a4b"), 1.3)
    lbl(d, (vx + bw2 + nx) / 2, cy + bh2 / 2 + 4, "success", colors.HexColor("#2c7a4b"), 7, font="Cond-Bold")
    # verify -> probe (unknown) and back
    arrow(d, vx + bw2 / 2 - 14, cy, vx + bw2 / 2 - 14, pby + bh2 + 1, colors.HexColor("#9a6b00"))
    lbl(d, vx + bw2 / 2 - 18, (cy + pby + bh2) / 2, "unknown", colors.HexColor("#9a6b00"), 7, "end", "Cond-Bold")
    arrow(d, vx + bw2 / 2 + 14, pby + bh2, vx + bw2 / 2 + 14, cy - 1, colors.HexColor("#9a6b00"))
    lbl(d, vx + bw2 / 2 + 18, (cy + pby + bh2) / 2, "re-check", colors.HexColor("#9a6b00"), 7, "start")
    # verify -> recover (failure)
    red = colors.HexColor("#b4472f")
    arrow(d, vx + 8, cy + 4, rx_ + bw2 - 1, rby + bh2 - 4, red, 1.1)
    lbl(d, (vx + rx_ + bw2) / 2 + 22, (cy + rby + bh2) / 2 + 6, "failure (typed)", red, 7, font="Cond-Bold")
    # execute -> recover (failure during execute)
    arrow(d, ex + 22, cy, ex + 22, rby + bh2 + 1, red, 1.1)
    lbl(d, ex + 18, (cy + rby + bh2) / 2, "failure", red, 7, "end", "Cond-Bold")
    lbl(d, ex + 18, (cy + rby + bh2) / 2 - 9, "event", red, 7, "end", "Cond-Bold")
    # recover -> execute (retry)
    arrow(d, ex + bw2 - 22, rby + bh2, ex + bw2 - 22, cy - 1, INK)
    lbl(d, ex + bw2 - 18, (cy + rby + bh2) / 2, "retry", INK, 7, "start")
    # recover -> rollback (to earlier stage)
    arrow(d, rx_ + bw2 / 2, rby, rx_ + bw2 / 2, 18, red, 1.0, [3, 2])
    lbl(d, rx_ + bw2 / 2 + 5, 21, "ROLLBACK → latest stage whose precondition holds", red, 6.8, "start")

    # escalation ladder on the right
    lx = nx + bw2 + 40
    ly = cy + bh2
    lbl(d, lx, ly, "ESCALATION (per failure type)", INK, 7.5, "start", "Cond-Bold")
    steps = ["1. recovery move, retry stage", "2. restart stage from its pre-pose",
             "3. rollback to an earlier stage", "4. hold and stop the episode"]
    for i, s in enumerate(steps):
        box(d, lx, ly - 30 - 30 * i, W - lx - 18, 22, s, [YES, PART, NO, colors.HexColor("#e5e7ea")][i],
            bold=False, fs=7.2)
        if i:
            arrow(d, lx + 14, ly - 30 - 30 * (i - 1), lx + 14, ly - 30 - 30 * i + 23, MUTED, 0.8, head=4)
    lbl(d, lx, ly - 30 - 30 * 3 - 14, "Each step has its own retry budget;", MUTED, 6.6, "start")
    lbl(d, lx, ly - 30 - 30 * 3 - 23, "all of it stays inside the 900 s episode.", MUTED, 6.6, "start")
    return d


# ---------------------------------------------------------------- recovery table
RECOVERY = [
    ("Missed grasp (<i>empty_grasp</i>)", "any", "Gripper event", "Open gripper, back off 5 cm, move to the stage's pre-grasp pose",
     "Same stage, EXECUTE"),
    ("Object dropped while carried (<i>drop</i>)", "EXTRACT, PACK", "Gripper event",
     "Open gripper, lift clear, look (still frame)", "Rollback: EXTRACT_COMPONENT (re-pick wherever the part is)"),
    ("Sponge / replacement dropped", "WIPE_CHAMBER, PLACE", "Gripper event", "Open, lift clear, back to pre-grasp pose",
     "Same stage, EXECUTE"),
    ("Wrong object grasped", "EXTRACT, PLACE", "Grasp region; camera", "Release in place, back off", "Same stage, EXECUTE"),
    ("Lid released too low / swung shut", "LIFT_LID", "Rotation or arc estimate; camera lid angle",
     "Retract from the lid, go to the handle pre-pose", "Same stage, EXECUTE"),
    ("Part outside the box / bounced out", "PACK_COMPONENT", "Region at release; camera “part in box”",
     "Retract, still frame to locate the part", "Rollback: EXTRACT_COMPONENT"),
    ("Robot still touching the packed part", "PACK_COMPONENT", "Clearance check (both arms)", "Retract both arms",
     "VERIFY again"),
    ("Wipe presses air / wrong surface", "WIPE_CHAMBER", "Contact count vs. height", "Scripted downward contact search, re-anchor the wipe grid",
     "WIPE_CHAMBER script restart"),
    ("Sponge not on the table", "WIPE_CHAMBER", "Region at release; camera", "Re-grasp sponge, place on table",
     "Same stage (release substep)"),
    ("Replacement part not seated", "PLACE_REPL.", "Touch probe at seated height", "Scripted press-down; if still high, re-grasp and re-place",
     "VERIFY again"),
    ("Lid not fully shut (sim: ≤ 0.45°)", "CLOSE_LID", "Touch probe on lid surface; camera",
     "Scripted push-to-contact on the lid, hold, release", "VERIFY again"),
    ("Lid blocked by unseated part", "CLOSE_LID", "Lid estimate stuck above 0; probe at stage 5",
     "Open lid again", "Rollback: PLACE_REPLACEMENT"),
    ("Stall / over budget", "any", "Timer; classifier progress", "Retract to the stage pre-pose",
     "Same stage, then escalate"),
]


def recovery_table(width):
    hdr = [Paragraph(h, hcell) for h in ("Failure type", "Stages", "Detected by", "Recovery maneuver", "Resume at")]
    rows = [hdr] + [[Paragraph(a, cellb), Paragraph(b, cell), Paragraph(c, cell), Paragraph(e, cell),
                     Paragraph(f, cell)] for a, b, c, e, f in RECOVERY]
    t = Table(rows, colWidths=[w * width for w in (0.22, 0.11, 0.21, 0.27, 0.19)], repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), HEAD_BG), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("GRID", (0, 0), (-1, -1), 0.4, RULE),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    for r in range(1, len(rows)):
        if "Rollback" in RECOVERY[r - 1][4]:
            st.append(("BACKGROUND", (4, r), (4, r), NO))
        if r % 2 == 0:
            st.append(("BACKGROUND", (0, r), (3, r), colors.HexColor("#f6f8fa")))
    t.setStyle(TableStyle(st))
    return t


SOURCES = [
    ("EXECUTE", "GR00T, stage prompt (WIPE_CHAMBER: scripted wipe; CLOSE_LID final seating: scripted push)",
     "Fixed prompt per stage, exactly as labelled in the fine-tuning data"),
    ("VERIFY", "Hold (current pose)", "— (no model call; gives the classifier a still frame)"),
    ("PROBE", "Script: touch probe, push, or look", "—"),
    ("RECOVER", "Script now; GR00T once recovery demos exist",
     "Recovery prompt per failure type (e.g. “re-grasp the part”), only after it is trained"),
]


def source_table(width):
    hdr = [Paragraph(h, hcell) for h in ("Substate", "Action source", "Prompt to GR00T")]
    rows = [hdr] + [[Paragraph(a, cellb), Paragraph(b, cell), Paragraph(c, cell)] for a, b, c in SOURCES]
    t = Table(rows, colWidths=[w * width for w in (0.14, 0.46, 0.40)])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), HEAD_BG), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("GRID", (0, 0), (-1, -1), 0.4, RULE),
                           ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t


# ---------------------------------------------------------------- classifier table
# (stage, outcome VERIFY must confirm, classifier head(s), camera input & timing,
#  training labels from sim state, decision rule, without it)
CLASSIFIERS = [
    ("1 LIFT_LID",
     "Lid open past its balance point and still open after the hand lets go",
     "<b>lid_angle</b> (regression, 0 closed … 1 open; the existing <i>lid_open</i> output)",
     "Head camera; still frame 1 s after release (wrist camera while the handle is held)",
     "Lid joint value per frame",
     "Success: ≥ 0.55 on every check over 1 s. Failure: below 0.40. In between: unknown → PROBE",
     "Hinge-arc angle at release; assume it stays open"),
    ("2 EXTRACT_COMPONENT",
     "The <i>part</i> (not the lid or sponge) is in the gripper and clear of the chamber",
     "<b>object_in_hand</b> (classes: part / sponge / replacement / lid / none) and <b>part_clear</b> (binary)",
     "Wrist camera of the carrying arm while lifting; head camera for clearance",
     "Gripper–object contact pairs; part height above the chamber rim",
     "Success: “part” and clear, both above threshold. “none” or another object: failure",
     "Lift ≥ 10 cm; gripper opening width per object"),
    ("3 PACK_COMPONENT",
     "Part resting inside the empty box, robot not touching it",
     "<b>part_in_box</b> (binary)",
     "Head camera; still frame after both arms retract",
     "Part pose inside the box bounds and the sim's PACK_COMPONENT flag; contact sensors for “no robot contact”",
     "Success: p ≥ 0.9. Failure: p ≤ 0.2. Else unknown → look again from a second pose",
     "Region at release only (cannot see bounce-outs)"),
    ("4 WIPE_CHAMBER (enter)",
     "Sponge held and on the chamber floor before the script starts",
     "<b>sponge_on_floor</b> (binary)",
     "Wrist camera of the wiping arm",
     "Sponge–chamber-floor contact from the sim contact sensor",
     "Enter the script when p ≥ 0.8 (or the contact proxy fires)",
     "Region = chamber, contact proxy, or held 2 s"),
    ("4 WIPE_CHAMBER (exit)",
     "Enough wipes counted and the sponge back on the table",
     "<b>sponge_on_table</b> (binary). The wipe count stays with the contact proxy: the sim's contact counter is not visible",
     "Head camera; still frame after release",
     "Sponge pose on the table surface; sim contact log to calibrate the proxy",
     "Success: wipe count ≥ 6 (proxy) and p ≥ 0.9",
     "Region at release"),
    ("5 PLACE_REPLACEMENT",
     "Replacement part seated flat in the chamber",
     "<b>part_seated</b> (binary) plus optional <b>tilt</b> regression",
     "Head camera and wrist camera; still frame after retract",
     "Part pose vs. its seated pose (position error, tilt angle) and the sim flag",
     "Success: p ≥ 0.9 <b>and</b> touch probe at the seated height agrees",
     "Touch probe alone"),
    ("6 CLOSE_LID",
     "Lid shut within the sim's 0.45° tolerance and staying shut",
     "<b>lid_fully_closed</b> (binary), trained with hard negatives at 1–3° open, on a high-resolution crop of the lid seam",
     "Wrist camera close-up of the seam before release; head camera 1 s after release",
     "Lid joint ≤ 0.45° (positive) vs. 0.45–5° (hard negatives)",
     "Success: p ≥ 0.95 <b>and</b> touch probe on the lid surface agrees; otherwise scripted push and re-check",
     "Touch probe + kinematic estimate (too coarse alone)"),
    ("All stages",
     "Which stage the scene is in; whether progress is being made",
     "<b>stage</b> and <b>progress</b> (the existing <i>stage_logits</i> / <i>progress</i> outputs)",
     "All three cameras, short clip (current classifier input, T = 15)",
     "Stage boundaries from the sim's success flags",
     "Stall: no progress rise for 8 s → RETRY. Disagrees with the orchestrator's stage → PROBE",
     "Timer only"),
]


def classifier_table(width):
    hdr = [Paragraph(h, hcell) for h in ("Stage", "Outcome VERIFY must confirm", "Classifier head(s)",
                                         "Camera input and timing", "Training labels (from sim state)",
                                         "Decision rule", "Without it")]
    rows = [hdr] + [[Paragraph(r[0], cellb)] + [Paragraph(x, cell) for x in r[1:]] for r in CLASSIFIERS]
    t = Table(rows, colWidths=[w * width for w in (0.09, 0.14, 0.18, 0.15, 0.15, 0.17, 0.12)], repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), HEAD_BG), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("GRID", (0, 0), (-1, -1), 0.4, RULE), ("BACKGROUND", (0, 1), (0, -1), STAGE_BG),
          ("BACKGROUND", (6, 1), (6, -1), colors.HexColor("#f6f8fa")),
          ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    t.setStyle(TableStyle(st))
    return t


def legend():
    items = [(YES, "Yes: caught today"), (PART, "Partly: caught in some cases, or only by timeout"),
             (NO, "No: not caught today; “regions: yes” = caught once region calibration is loaded")]
    cells = []
    for c, t in items:
        cells += ["", Paragraph(t, cell)]
    t = Table([cells], colWidths=[8, 95, 8, 175, 8, 330])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), YES), ("BACKGROUND", (2, 0), (2, 0), PART),
                           ("BACKGROUND", (4, 0), (4, 0), NO), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 1),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    t.hAlign = "LEFT"
    return t


def on_page(c, doc):
    c.saveState()
    c.setFont("Sans", 7)
    c.setFillColor(MUTED)
    c.drawString(doc.leftMargin, 9 * mm, "Astra runtime: orchestrator failure detection and proposed hierarchical state machine")
    c.drawRightString(doc.pagesize[0] - doc.rightMargin, 9 * mm, f"Page {doc.page}")
    c.restoreState()


def main():
    page = landscape(A4)
    doc = SimpleDocTemplate(OUT, pagesize=page, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=13 * mm,
                            bottomMargin=15 * mm, title="Astra orchestrator: failure detection and HSM proposal",
                            author="Astra runtime")
    W = page[0] - doc.leftMargin - doc.rightMargin
    s = []
    s.append(P("Astra orchestrator: failure detection and a hierarchical state machine", title))
    s.append(P("Status of the current runtime on <b>feature/runtime-policy-impl</b> (implemented, not yet run "
               "against the sim) and a proposal for the next orchestrator design. No code has been changed for "
               "the proposal.", sub))

    s.append(P("1. Where the current orchestrator stands", h1))
    s.append(P("The orchestrator moves forward through the six <i>ChamberServiceTask</i> stages on gripper events "
               "(grasp, release, drop, empty grasp), the hand poses from FK and a kinematic lid estimate. The "
               "policy observation has no lid angle and no object poses, so these signals confirm that the "
               "<b>robot did what it meant to do</b>, not that the <b>world changed as intended</b>. The stage "
               "classifier is the part of the design meant to read outcomes from the cameras; it does not exist "
               "yet."))
    s.append(P("“Detectable now?” assumes the current default config: no classifier, no region or hinge "
               "calibration, FK available. Every advance also needs its rule to hold for 12 ticks in a row "
               "(about 0.25 s) and the stage's minimum time.", body))
    s.append(legend())
    s.append(Spacer(1, 5))
    s.append(detection_table(W))
    s.append(Spacer(1, 8))
    s.extend([
        P("Across all stages", h2),
        *bullets([
            "Every detected failure gets the same recovery: flush the action queue and reset the policy, from "
            "wherever the arm is. <i>pre_grasp_pose</i> is not filled yet, so a retry does not return to a known pose.",
            "The most damaging error is a <b>wrong advance</b>, and none of the “No” rows can prevent one. The sim "
            "chains its stage flags, so one false “done” forfeits every later stage.",
            "Checks whose signal is not configured pass automatically (regions, classifier), which keeps the stack "
            "moving but removes failure detection when no other signal is available."]),
        P("Solutions that cover the most “No” rows", h2),
        *bullets([
            "<b>Camera outcome checks</b>: classifier heads for lid angle and object placement, labelled from the "
            "sim states in the demos or replays. Covers rows in all six stages.",
            "<b>Touch probes with the contact proxy</b>: stages 5 and 6, which have the tightest tolerances.",
            "<b>Region and height calibration</b> in the base frame: the wrong-object, wrong-place and "
            "wrong-surface rows in stages 2–5.",
            "<b>Hinge-arc geometry</b>: the lid-angle rows in stages 1 and 6."]),
    ])

    s.append(PageBreak())
    s.append(P("2. Proposed hierarchical state machine", h1))
    s.append(P("Two levels. The top level is the fixed sequence of task stages. Inside each stage the same small "
               "set of substates runs: GR00T executes with the stage's prompt, a VERIFY substate checks the "
               "outcome, an unknown result triggers an active PROBE instead of an advance, and each typed "
               "failure has its own recovery maneuver: retry the stage, or roll back to the latest earlier stage "
               "whose precondition still holds."))
    s.append(Spacer(1, 4))
    s.append(hsm_diagram(W))
    s.append(Spacer(1, 6))

    s.extend([
        P("Design rules", h2),
        *bullets([
            "<b>Success is a checked outcome.</b> A stage ends on VERIFY confirming the world state (part in box, "
            "lid angle), not on “released and retracted”. Detectors return success, failure or <b>unknown</b>; "
            "unknown leads to PROBE, never to an advance. Every transition flushes the action chunk.",
            "<b>Recovery may cross stages.</b> Each stage has a precondition (“part in hand”, “lid open”). A "
            "failure maps to a maneuver <i>and</i> a resume point: the latest stage whose precondition holds.",
            "<b>A prompt only works if GR00T was trained on it.</b> Each prompt must match a label in the "
            "fine-tuning data exactly. Recovery prompts need recovery demos (induced failures in the sim, or "
            "DAgger-style corrections). Until then recovery is a scripted move followed by the normal stage prompt "
            "from a cleaner start. Explicit per-substate prompts also make up for GR00T having no memory (T = 1).",
            "<b>Not every state uses GR00T.</b> WIPE_CHAMBER is scripted; the final lid seating (within 0.45°) is a "
            "scripted push-to-contact with hinge geometry, since GR00T cannot see that gap at its input "
            "resolution; holds and safety stops stay in the runtime.",
        ]),
    ])

    s.append(PageBreak())
    s.append(P("3. Classifiers needed to advance each stage (proposal)", h1))
    s.append(P("VERIFY needs one outcome check per stage. The table lists the classifier head that would confirm "
               "each stage's outcome from the cameras, what it is trained on, and the rule that turns its output "
               "into success, failure or unknown. All labels come for free offline: the demos and sim replays "
               "record the lid joint, object poses, contacts and the sim's own stage flags, even though the runtime "
               "never sees them. Thresholds are starting points to tune on replays."))
    s.append(classifier_table(W))
    s.append(Spacer(1, 6))
    s.extend([
        P("How to build them", h2),
        *bullets([
            "<b>One model, many heads.</b> A shared image backbone with one small head per row. The existing "
            "classifier contract already has <i>stage_logits</i>, <i>progress</i> and <i>lid_open</i>; the single "
            "<i>scene_ok</i> output splits into the per-stage heads above.",
            "<b>Reuse GR00T's backbone</b> as the feature extractor if possible, instead of training a second vision "
            "network. Inference time is not a constraint: the sim waits for the policy.",
            "<b>Two thresholds per head.</b> Above the high one is success, below the low one is failure, the band "
            "between is <i>unknown</i> and leads to PROBE. Checks run on still frames during VERIFY, so motion blur "
            "and occlusion by the moving arm are avoided.",
            "<b>Hard negatives decide the quality</b>: near-misses such as a lid 1–3° open, a part on the box rim, a "
            "sponge on the chamber edge. Generate them by inducing failures in the sim. For stages 5 and 6 the "
            "classifier alone is not enough: advance only when it and the touch probe agree.",
        ]),
    ])

    s.append(Spacer(1, 6))
    s.append(P("4. Failure types and recovery maneuvers (proposal)", h1))
    s.append(P("One row per failure type. “Detected by” names the signal that has to exist for the branch to run; "
               "most of them are the solutions from section 1 and the classifiers from section 3. Rows marked red in “Resume at” roll back to an "
               "earlier stage."))
    s.append(recovery_table(W))
    s.append(Spacer(1, 8))
    s.append(KeepTogether([P("Action source and prompt per substate", h2), source_table(W * 0.85)]))
    s.append(Spacer(1, 8))
    s.extend([
        P("5. Mapping to the current code", h1),
        *bullets([
            "<b>orchestrator/stage_specs.py</b>: the failure predicates already return a reason string. Turn it "
            "into a failure type, and move the recovery table above into <b>config/runtime.yaml</b>.",
            "<b>orchestrator/task_orchestrator.py</b>: today a flat stage list with RETRY and ENTER/EXIT_SCRIPT. "
            "Add the substate layer (EXECUTE / VERIFY / PROBE / RECOVER), stage preconditions, rollback "
            "transitions and the escalation counters.",
            "<b>orchestrator/stage_classifier.py</b>: extend the TorchScript output contract with the per-stage "
            "outcome heads from section 3 and return success / failure / unknown per head.",
            "<b>policies/router.py</b>: today maps stage → source. Map (stage, substate) → (source, prompt).",
            "<b>policies/</b>: add scripted primitives for back-off, pre-pose, touch probe and lid push, alongside "
            "the existing scripted wipe."]),
        P("6. Measure before building", h1),
        P("The state machine is only as good as its detectors: a recovery branch whose failure is never detected "
          "never runs. Start with offline replay scoring: run the orchestrator over recorded episodes and sim "
          "runs, and log each advance and retry next to the true sim state (stage flags, lid joint, object poses). "
          "That gives the wrong-advance and missed-failure rate per stage and shows which detectors and recovery "
          "branches to build first. The lid stages and the release-and-place stages are expected to dominate."),
    ])
    doc.build(s, onFirstPage=on_page, onLaterPages=on_page)


main()
