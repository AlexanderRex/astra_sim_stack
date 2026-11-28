"""Build the proposed event-detection vision model PDF (architecture diagram + layer and head tables).

Usage: python3 make_event_vision_model_pdf.py [output.pdf]
Default output: astra_event_vision_model.pdf next to this script (gitignored).

The model is a proposal, not implemented. Parameter counts and timings are estimates.
"""
import os
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily, stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "astra_event_vision_model.pdf")
FD = "/usr/share/fonts/truetype/dejavu/"
for name, f in [("Sans", "DejaVuSans"), ("Sans-Bold", "DejaVuSans-Bold"), ("Sans-Oblique", "DejaVuSans-Oblique"),
                ("Cond", "DejaVuSansCondensed"), ("Cond-Bold", "DejaVuSansCondensed-Bold"),
                ("Cond-Oblique", "DejaVuSansCondensed-Oblique")]:
    pdfmetrics.registerFont(TTFont(name, FD + f + ".ttf"))
registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Oblique")
registerFontFamily("Cond", normal="Cond", bold="Cond-Bold", italic="Cond-Oblique")

INK = colors.HexColor("#1f2328")
MUTED = colors.HexColor("#57606a")
RULE = colors.HexColor("#d0d7de")
HEAD_BG = colors.HexColor("#24364b")
ACCENT = colors.HexColor("#2f5d8a")
KINDS = {  # fill, stroke, legend text
    "runtime": (colors.HexColor("#fdf1d3"), colors.HexColor("#b9975b"), "Existing runtime (input or consumer)"),
    "fixed": (colors.HexColor("#ffffff"), colors.HexColor("#8c959f"), "Fixed processing, no learning"),
    "frozen": (colors.HexColor("#e3ebf5"), colors.HexColor("#6f8db3"), "Pretrained, frozen"),
    "train": (colors.HexColor("#dcf3e3"), colors.HexColor("#5f9b6f"), "Trained (sim labels, then real fine-tune)"),
}

title = ParagraphStyle("t", fontName="Sans-Bold", fontSize=19, leading=23, textColor=INK, spaceAfter=3)
sub = ParagraphStyle("s", fontName="Sans", fontSize=9.5, leading=13, textColor=MUTED, spaceAfter=6)
h1 = ParagraphStyle("h1", fontName="Sans-Bold", fontSize=13.5, leading=17, textColor=INK, spaceBefore=6,
                    spaceAfter=6, keepWithNext=1)
h2 = ParagraphStyle("h2", fontName="Sans-Bold", fontSize=10.5, leading=14, textColor=ACCENT, spaceBefore=8,
                    spaceAfter=3, keepWithNext=1)
body = ParagraphStyle("b", fontName="Sans", fontSize=10, leading=14, textColor=INK, spaceAfter=4)
bullet = ParagraphStyle("bl", parent=body, leftIndent=12, bulletIndent=3, spaceAfter=2)
cell = ParagraphStyle("c", fontName="Cond", fontSize=9.2, leading=11.6, textColor=INK)
cellb = ParagraphStyle("cb", parent=cell, fontName="Cond-Bold")
hcell = ParagraphStyle("hc", parent=cell, fontName="Cond-Bold", textColor=colors.white, fontSize=9.6)


def P(t, s=body):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(i, bullet, bulletText="•") for i in items]


# ------------------------------------------------------------------------------------ diagram
# Columns left to right: (heading, width, [(key, kind, title, [lines])]). Lines starting with
# "in:" / "out:" get a coloured prefix.
COLUMNS = [
    ("INPUTS", 104, [
        ("in_head", "runtime", "Head camera RGB",
         ["first_person_camera_rgb", "out: 480×640×3 uint8", "sampled at 5 Hz (every 10th 50 Hz tick)"]),
        ("in_lw", "runtime", "Left wrist RGB", ["left_hand_camera_rgb", "out: 224×224×3 uint8, 5 Hz"]),
        ("in_rw", "runtime", "Right wrist RGB", ["right_hand_camera_rgb", "out: 224×224×3 uint8, 5 Hz"]),
        ("in_cam", "runtime", "Camera calibration",
         ["galbot_calibration.json + FK of the head and arm joints",
          "out: base_from_optical 3 × (4×4), K 3 × (3×3)"]),
        ("in_state", "runtime", "Robot state (CanonicalObs)",
         ["out: 34-D: arm q 7 + 7, gripper 2, EEF world pos + rot6d 9 + 9"]),
        ("in_ev", "runtime", "Rule-based events (EventState)",
         ["out: 16-D: holding, grasp, release, drop, empty_grasp (×2 arms), in_contact, lid_tracking, "
          "lid_angle_kin, lid_slip, EEF tracking error ×2"]),
        ("in_act", "runtime", "Last sim action", ["out: 5-D: gripper cmd ×2, base vel vx, vy, wz"]),
    ]),
    ("L1  PRE-PROCESS", 104, [
        ("p_img", "fixed", "Image pre-processing",
         ["in: 3 RGB frames", "resize keeping the aspect ratio: head → 224×308, wrists → 224×224 "
          "(multiples of 14)", "ImageNet mean/std, bf16",
          "out: head (3, 224, 308), wrists 2 × (3, 224, 224)", "raw uint8 frames are passed on too"]),
        ("p_voc", "fixed", "Object vocabulary",
         ["8 fixed text queries: chamber, lid handle, old part, empty box, full box, sponge, "
          "replacement part, table", "out: 8 strings"]),
        ("p_geo", "fixed", "Geometry setup",
         ["in: camera poses, K", "support planes: table z = 0.72 m, floor z = 0, enclosure roof "
          "z = table + 0.414 m", "out: pixel → plane intersection in base_link, per camera"]),
        ("p_prop", "fixed", "Proprio vector",
         ["in: 34 + 16 + 5", "concatenate, min-max normalise with training statistics", "out: (55,) float"]),
    ]),
    ("L2  FROZEN ENCODERS", 110, [
        ("b_dino", "frozen", "DINOv2 ViT-B/14",
         ["in: 3 normalised images", "12 layers, d = 768, 86 M params, not fine-tuned",
          "out per camera: CLS (768) + patch grid", "head 16×22×768, wrists 2 × 16×16×768",
          "≈ 6 ms for 3 images in bf16 (estimate)"]),
        ("b_gd", "frozen", "Grounding DINO-T",
         ["in: raw frames + 8 text queries", "runs at 1 Hz, asynchronously, to (re)seed the objects",
          "out: per camera boxes (8, 4) + scores (8)"]),
        ("b_sam", "frozen", "SAM 2 small",
         ["in: boxes as prompts, frames at 5 Hz", "video mode: propagates the masks between seeds",
          "out: masks (3, 8, H, W) bool, mask scores (3, 8)"]),
    ]),
    ("L3  TOKENIZERS", 116, [
        ("t_cam", "train", "Camera tokens: attention pooling",
         ["in: patch grid per camera", "8 learned queries, 1 cross-attention layer, 768 → 256, "
          "+ camera-ID embedding", "out: 3 × 8 = 24 tokens × 256"]),
        ("t_obj", "train", "Object tokens: mask pooling",
         ["in: patch grid + masks resized to the grid", "mean patch feature inside each mask, from "
          "the camera that sees most of it; MLP 768 → 256, + object-ID embedding",
          "out: 8 tokens × 256, visible (8)"]),
        ("t_geo", "train", "Object geometry encoding",
         ["in: mask bottom-centre pixel, mask area, plane intersection",
          "ray ∩ support plane → coarse xyz in base_link",
          "Fourier features (3 × 16 sin/cos) → MLP → 256, added to the object token",
          "out: xyz_coarse (8, 3), geometry emb (8, 256)"]),
        ("t_prop", "train", "Proprio token", ["in: (55,)", "MLP 55 → 256 → 256, GELU", "out: 1 token × 256"]),
    ]),
    ("L4  FRAME FUSION", 104, [
        ("f_fuse", "train", "Frame fusion transformer",
         ["in: [CLS] + 24 camera + 8 object + 1 proprio = 34 tokens × 256, + token-type embedding",
          "4 layers, 8 heads, FFN 1024, pre-LN, ≈ 3.2 M params",
          "out: frame token z_t (256), object tokens (8, 256)"]),
        ("f_cache", "fixed", "Feature cache (ring buffer)",
         ["keeps the fused tokens of the last T = 16 frames (3.2 s at 5 Hz)",
          "only the newest frame is computed each tick", "out: (16, 1 + 8, 256) + frame ages (16)"]),
    ]),
    ("L5  TEMPORAL", 102, [
        ("s_glob", "train", "Temporal transformer: scene",
         ["in: frame tokens (16, 256) + age embedding", "2 layers, causal, 8 heads",
          "sees motion: lid swinging after release, “stays shut 1 s”, start vs stage 6",
          "out: h_t (256)"]),
        ("s_obj", "train", "Temporal transformer: objects",
         ["in: object tokens (16, 8, 256)", "same design, weights shared over the 8 objects",
          "out: o_t (8, 256)"]),
    ]),
    ("L6  OUTPUT HEADS  (MLP 256 → 128 → k)", 142, [
        ("h_task", "train", "Task heads (from h_t)",
         ["stage: 7 logits (stages 1–6 + DONE)", "phase: 5 logits", "progress: 1, sigmoid",
          "outcome_ok: 6 sigmoids, one per stage goal: lid open, part out, part in box, chamber wiped, "
          "replacement seated, lid shut"]),
        ("h_lid", "train", "Lid heads (h_t ⊕ o_t[chamber])",
         ["lid_angle: 1 value (0 closed … 1 = 90°) and its log σ²", "lid_shut: 1 sigmoid (angle ≤ 0.005)"]),
        ("h_obj", "train", "Object state heads (from o_t)",
         ["old part: in chamber / in hand / in box / elsewhere (4)",
          "replacement: in box / in hand / seated / elsewhere (4)",
          "sponge: on table / in hand / in chamber / elsewhere (4)",
          "xyz refinement Δ (8, 3) + visible (8)"]),
        ("h_int", "train", "Interaction heads (h_t ⊕ o_t)",
         ["grasped object per hand: 2 × 9 logits (none + 8 objects)",
          "contact: sponge ↔ chamber floor (1), hand ↔ lid handle × 2 (2)"]),
    ]),
    ("L7  DECISION", 106, [
        ("d_cal", "fixed", "Calibration",
         ["in: raw head outputs", "temperature scaling per head, fitted on held-out episodes",
          "out: calibrated probabilities, σ for lid_angle"]),
        ("d_3way", "fixed", "Three-way decision per check",
         ["p ≥ τ_hi → success, p ≤ τ_lo → failure, otherwise unknown → PROBE",
          "debounce: N agreeing frames", "out: success / failure / unknown per check"]),
        ("d_fuse", "fixed", "Fusion with rule-based detectors",
         ["lid: kinematic estimate while held, vision after release; disagreement > 0.2 for 1 s → "
          "trust vision", "grasp: gripper joint AND grasped-object head",
          "regions: object xyz replaces the fixed calibration"]),
    ]),
    ("OUTPUTS", 98, [
        ("o_sp", "runtime", "StagePrediction (extended)",
         ["stage_probs (7), phase_probs (5), progress", "lid_open + σ, lid_shut",
          "outcome_ok (6) replaces scene_ok"]),
        ("o_ve", "runtime", "VisionEvents (new type)",
         ["object_state 3 × 4, object_xyz (8, 3) + visible", "grasped_object (2), contacts (3)",
          "success / failure / unknown per check"]),
        ("o_orch", "runtime", "TaskOrchestrator",
         ["VERIFY and PROBE substates (HSM proposal)", "region tests from object_xyz"]),
    ]),
]

# (src, dst, mode, label). mode: "h" between columns, "v" down within a column, "bus" routed below.
EDGES = [
    ("in_head", "p_img", "h", ""), ("in_lw", "p_img", "h", ""), ("in_rw", "p_img", "h", ""),
    ("in_cam", "p_geo", "h", ""),
    ("in_state", "p_prop", "h", ""), ("in_ev", "p_prop", "h", ""), ("in_act", "p_prop", "h", ""),
    ("p_img", "b_dino", "h", "normalised"), ("p_img", "b_gd", "h", "raw"), ("p_img", "b_sam", "h", "raw"),
    ("p_voc", "b_gd", "h", "queries"),
    ("b_gd", "b_sam", "v", "boxes"),
    ("b_dino", "t_cam", "h", "patches"), ("b_dino", "t_obj", "h", "patches"),
    ("b_sam", "t_obj", "h", "masks"), ("b_sam", "t_geo", "h", "masks"),
    ("p_geo", "t_geo", "h", ""), ("p_prop", "t_prop", "h", ""),
    ("t_cam", "f_fuse", "h", ""), ("t_obj", "f_fuse", "h", ""), ("t_geo", "f_fuse", "h", ""),
    ("t_prop", "f_fuse", "h", ""),
    ("f_fuse", "f_cache", "v", "push"),
    ("f_cache", "s_glob", "h", ""), ("f_cache", "s_obj", "h", ""),
    ("s_glob", "h_task", "h", ""), ("s_glob", "h_lid", "h", ""), ("s_glob", "h_int", "h", ""),
    ("s_obj", "h_lid", "h", ""), ("s_obj", "h_obj", "h", ""), ("s_obj", "h_int", "h", ""),
    ("h_task", "d_cal", "h", ""), ("h_lid", "d_cal", "h", ""), ("h_obj", "d_cal", "h", ""),
    ("h_int", "d_cal", "h", ""),
    ("d_cal", "d_3way", "v", ""), ("d_3way", "d_fuse", "v", ""),
    ("d_fuse", "o_sp", "h", ""), ("d_fuse", "o_ve", "h", ""),
    ("o_ve", "o_orch", "v", ""),
    ("in_ev", "d_fuse", "bus", "EventState (rule-based detectors)"),
]

LINE_FS, TITLE_FS, PAD = 7.9, 9.0, 5
LINE_H, TITLE_H = 9.7, 11.0


def wrap(text, font, fs, width):
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if stringWidth(trial, font, fs) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def layout_box(w, title, lines):
    rows = []  # (text, prefix or None)
    for ln in lines:
        prefix = None
        for p in ("in:", "out:"):
            if ln.startswith(p):
                prefix, ln = p, ln[len(p):].strip()
        indent = stringWidth(prefix + " ", "Cond-Bold", LINE_FS) if prefix else 0
        parts = wrap(ln, "Cond", LINE_FS, w - 2 * PAD - indent)
        for i, part in enumerate(parts):
            rows.append((part, prefix if i == 0 else None, indent))
    title_rows = wrap(title, "Cond-Bold", TITLE_FS, w - 2 * PAD)
    h = PAD + TITLE_H * len(title_rows) + 1 + LINE_H * len(rows) + PAD
    return title_rows, rows, h


def draw_box(d, b):
    fill, stroke, _ = KINDS[b["kind"]]
    d.add(Rect(b["x"], b["y"], b["w"], b["h"], rx=3.5, ry=3.5, fillColor=fill, strokeColor=stroke,
               strokeWidth=0.9))
    y = b["y"] + b["h"] - PAD - TITLE_FS * 0.85
    for tr in b["title_rows"]:
        d.add(String(b["x"] + PAD, y, tr, fontName="Cond-Bold", fontSize=TITLE_FS, fillColor=INK))
        y -= TITLE_H
    y -= 0.5
    for text, prefix, indent in b["rows"]:
        x = b["x"] + PAD
        if prefix:
            d.add(String(x, y, prefix, fontName="Cond-Bold", fontSize=LINE_FS, fillColor=ACCENT))
        d.add(String(x + indent, y, text, fontName="Cond", fontSize=LINE_FS, fillColor=INK))
        y -= LINE_H


def arrow_poly(d, pts, col=INK, w=0.8, dash=None, head=4.2):
    import math
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        d.add(Line(x1, y1, x2, y2, strokeColor=col, strokeWidth=w, strokeDashArray=dash))
    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    a = math.atan2(y2 - y1, x2 - x1)
    p1 = (x2 - head * math.cos(a - 0.42), y2 - head * math.sin(a - 0.42))
    p2 = (x2 - head * math.cos(a + 0.42), y2 - head * math.sin(a + 0.42))
    d.add(Polygon([x2, y2, p1[0], p1[1], p2[0], p2[1]], fillColor=col, strokeColor=col, strokeWidth=0.4))


BOTTOM, MAX_VGAP = 34, 46


def architecture_diagram(W, H):
    d = Drawing(W, H)
    gap = (W - sum(c[1] for c in COLUMNS)) / (len(COLUMNS) - 1)
    boxes, col_x, x = {}, [], 0.0
    top = H - 22
    for ci, (heading, cw, items) in enumerate(COLUMNS):
        col_x.append((x, x + cw))
        d.add(String(x + cw / 2, H - 9, heading, fontName="Cond-Bold", fontSize=8.6, fillColor=HEAD_BG,
                     textAnchor="middle"))
        d.add(Line(x, H - 13, x + cw, H - 13, strokeColor=RULE, strokeWidth=0.8))
        laid = [(key, kind) + layout_box(cw, ttl, lines) for key, kind, ttl, lines in items]
        avail = top - BOTTOM
        vgap = min(MAX_VGAP, (avail - sum(item[4] for item in laid)) / max(len(laid) - 1, 1))
        y = top
        for key, kind, title_rows, rows, h in laid:
            boxes[key] = dict(x=x, y=y - h, w=cw, h=h, kind=kind, col=ci, title_rows=title_rows, rows=rows)
            y -= h + vgap
        x += cw + gap
    for b in boxes.values():
        draw_box(d, b)

    # Spread the ports of edges that share a box side, ordered by the far end's height.
    outs, ins = {}, {}
    for e in EDGES:
        if e[2] == "h":
            outs.setdefault(e[0], []).append(e)
            ins.setdefault(e[1], []).append(e)

    def cy(k):
        return boxes[k]["y"] + boxes[k]["h"] / 2

    def port(key, edge, table, other):
        lst = sorted(table[key], key=lambda e: -cy(e[other]))
        n, i, b = len(lst), lst.index(edge), boxes[key]
        lo, hi = b["y"] + b["h"] * 0.2, b["y"] + b["h"] * 0.8
        return (lo + hi) / 2 if n == 1 else hi - (hi - lo) * i / (n - 1)

    def collides(xa, xb, y, skip):
        xa, xb = min(xa, xb), max(xa, xb)
        return any(k not in skip and b["x"] < xb and b["x"] + b["w"] > xa and b["y"] - 2 < y < b["y"] + b["h"] + 2
                   for k, b in boxes.items())

    # One vertical lane per trunk in each gap: a source that fans out shares one lane (and one exit
    # point); otherwise edges that collect at the same target share one lane (and one entry point).
    def trunk(e):
        return ("src", e[0]) if len(outs[e[0]]) > 1 else ("dst", e[1])

    gap_keys = {}
    for e in EDGES:
        if e[2] == "h":
            keys = gap_keys.setdefault(boxes[e[1]]["col"], [])
            if trunk(e) not in keys:
                keys.append(trunk(e))
    for col, keys in gap_keys.items():  # order lanes by trunk height to limit crossings
        keys.sort(key=lambda k: -cy(k[1]))
    bus_y = min(b["y"] for b in boxes.values()) - 14
    for src, dst, mode, label in EDGES:
        s, t = boxes[src], boxes[dst]
        if mode == "v":
            xs = s["x"] + s["w"] / 2
            arrow_poly(d, [(xs, s["y"]), (xs, t["y"] + t["h"])])
            if label:
                d.add(String(xs + 3, (s["y"] + t["y"] + t["h"]) / 2 - 2, label, fontName="Cond-Oblique",
                             fontSize=7.4, fillColor=MUTED))
            continue
        if mode == "bus":
            x1, y1 = s["x"] + s["w"], s["y"] + s["h"] * 0.12
            xg = x1 + gap * 0.12
            x2, y2 = t["x"], t["y"] + t["h"] * 0.15
            xe = x2 - gap * 0.12
            arrow_poly(d, [(x1, y1), (xg, y1), (xg, bus_y), (xe, bus_y), (xe, y2), (x2, y2)], col=ACCENT,
                       dash=[3, 2])
            d.add(String((xg + xe) / 2, bus_y - 10, label, fontName="Cond-Oblique", fontSize=7.8,
                         fillColor=ACCENT, textAnchor="middle"))
            continue
        e = (src, dst, mode, label)
        kind, _ = key = trunk(e)
        x1 = s["x"] + s["w"]
        x2 = t["x"]
        y1 = cy(src) if kind == "src" else port(src, e, outs, 1)
        y2 = cy(dst) if kind == "dst" else port(dst, e, ins, 0)
        keys = gap_keys[t["col"]]
        g_lo = col_x[t["col"] - 1][1]
        frac = 0.5 if len(keys) == 1 else 0.28 + 0.44 * keys.index(key) / (len(keys) - 1)
        xm = g_lo + gap * frac
        if abs(y1 - y2) < 0.5 and not collides(x1, x2, y1, {src, dst}):
            pts = [(x1, y1), (x2, y2)]
        elif not collides(x1, xm, y1, {src, dst}):
            pts = [(x1, y1), (xm, y1), (xm, y2), (x2, y2)]
        else:  # horizontal at the source would cross a box: drop in the first gap, then go across
            g1 = col_x[s["col"]][1] + gap * 0.5
            cands = [t["y"] + t["h"] * f for f in (0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.12, 0.88)]
            y2 = next((y for y in cands if not collides(g1, x2, y, {src, dst})), y2)
            pts = [(x1, y1), (g1, y1), (g1, y2), (x2, y2)]
        arrow_poly(d, pts)
    return d


def legend(W):
    d = Drawing(W, 14)
    x = 0
    for kind, (fill, stroke, text) in KINDS.items():
        d.add(Rect(x, 2, 18, 9, rx=2, ry=2, fillColor=fill, strokeColor=stroke, strokeWidth=0.9))
        d.add(String(x + 23, 4, text, fontName="Cond", fontSize=7.6, fillColor=INK))
        x += 31 + stringWidth(text, "Cond", 7.6)
    d.add(Line(x, 6.5, x + 22, 6.5, strokeColor=ACCENT, strokeWidth=0.8, strokeDashArray=[3, 2]))
    d.add(String(x + 27, 4, "Rule-based evidence joining at the decision layer", fontName="Cond", fontSize=7.6,
                 fillColor=INK))
    return d


# ------------------------------------------------------------------------------------ tables
LAYERS = [
    ("L1", "Image pre-processing", "3 RGB frames, uint8", "head (3, 224, 308), wrists 2 × (3, 224, 224), bf16; "
     "raw frames", "—", "5 Hz, < 1 ms"),
    ("L1", "Proprio vector", "state 34, EventState 16, last action 5", "(55,) normalised", "—", "5 Hz"),
    ("L1", "Geometry setup", "camera poses + K from FK and calibration; support-plane heights",
     "pixel → base_link plane intersection per camera", "—", "5 Hz"),
    ("L2", "DINOv2 ViT-B/14", "normalised images", "CLS 3 × 768; patches 16×22×768 (head), 2 × 16×16×768",
     "86 M frozen", "5 Hz, ≈ 6 ms"),
    ("L2", "Grounding DINO-T", "raw frames + 8 text queries", "boxes (3, 8, 4), scores (3, 8)", "≈ 172 M frozen",
     "1 Hz async, ≈ 60 ms"),
    ("L2", "SAM 2 small", "frames + box prompts", "masks (3, 8, H, W), scores (3, 8)", "≈ 46 M frozen",
     "5 Hz, ≈ 15 ms per camera"),
    ("L3", "Camera tokens (attention pooling)", "patch grids", "(24, 256)", "≈ 2.6 M trainable", "5 Hz"),
    ("L3", "Object tokens (mask pooling)", "patch grids + masks", "(8, 256), visible (8)", "≈ 0.3 M", "5 Hz"),
    ("L3", "Object geometry encoding", "mask pixels, plane intersection", "xyz_coarse (8, 3), emb (8, 256)",
     "≈ 0.1 M", "5 Hz"),
    ("L3", "Proprio token", "(55,)", "(1, 256)", "≈ 0.08 M", "5 Hz"),
    ("L4", "Frame fusion transformer", "34 tokens × 256", "z_t (256), object tokens (8, 256)", "≈ 3.2 M",
     "5 Hz, < 1 ms"),
    ("L4", "Feature cache", "fused tokens of the newest frame", "(16, 9, 256) + ages (16)", "—", "5 Hz"),
    ("L5", "Temporal transformer: scene", "(16, 256)", "h_t (256)", "≈ 1.6 M", "5 Hz, < 1 ms"),
    ("L5", "Temporal transformer: objects", "(16, 8, 256)", "o_t (8, 256)", "≈ 1.6 M", "5 Hz, < 1 ms"),
    ("L6", "Output heads (4 groups)", "h_t, o_t", "see table 2", "≈ 0.4 M", "5 Hz"),
    ("L7", "Calibration, three-way decision, fusion", "head outputs + EventState",
     "StagePrediction (extended), VisionEvents", "≈ 20 fitted scalars", "5 Hz"),
]

HEADS = [
    ("stage", "7 logits", "cross-entropy", "Stage boundaries: the sim's stage flags, and the rule-based "
     "detectors run over the demos", "Stage tracking, start vs stage 6", "Annotate stage boundary timestamps per "
     "episode (cheap)"),
    ("phase", "5 logits", "cross-entropy", "Phase from the gripper events in the demos", "HSM substates",
     "Same rule over the real episodes"),
    ("progress", "1, sigmoid", "L1", "Time fraction inside the stage", "Stall detection", "Same"),
    ("outcome_ok", "6 sigmoids", "BCE", "The sim's own success flag for each stage goal, per frame",
     "VERIFY at each stage end", "Per-episode outcome plus spot labels at the stage ends"),
    ("lid_angle", "1 + log σ²", "Gaussian NLL", "Lid001_joint value / 90°", "Lid after the hand releases it "
     "(stages 1 and 6)", "Kinematic angle while the hand holds the lid as a weak label; a fiducial on the lid "
     "while recording"),
    ("lid_shut", "1, sigmoid", "BCE", "Lid001_joint ≤ 0.005", "Stage 6 final check, with the touch probe",
     "Manual label; touch probe agreement"),
    ("object state", "3 × 4 logits", "cross-entropy", "Object rigid-body poses against the chamber and box "
     "volumes; “in hand” = touching the finger links and lifted", "Stages 2, 3 and 5 outcomes, wrong-object "
     "checks", "Key-frame annotation"),
    ("xyz refinement", "(8, 3) + visible (8)", "L1 on visible objects", "Object poses from the sim",
     "Region tests (replace the fixed calibration)", "Measured layout or fiducials, sparse"),
    ("grasped object", "2 × 9 logits", "cross-entropy", "Which body the finger links touch while holding",
     "Grasp check: right object in hand", "Gripper event + annotation"),
    ("contact", "3 sigmoids", "BCE", "Sim contact sensors (the sponge already has a contact report API)",
     "WIPE_CHAMBER contact count, touch probe", "Joint current or wrist force if the robot has them; otherwise annotation"),
]


def table(headers, rows, widths):
    data = [[P(h, hcell) for h in headers]] + [[P(str(c), cellb if j == 0 else cell) for j, c in enumerate(r)]
                                               for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), HEAD_BG), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("GRID", (0, 0), (-1, -1), 0.4, RULE), ("LEFTPADDING", (0, 0), (-1, -1), 4),
             ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f6f8fa")))
    t.setStyle(TableStyle(style))
    t.hAlign = "LEFT"
    return t


def on_page(c, doc):
    c.saveState()
    c.setFont("Sans", 7)
    c.setFillColor(MUTED)
    c.drawString(doc.leftMargin, 8 * mm, "Astra runtime: proposed event-detection vision model (not implemented; "
                 "parameter counts and timings are estimates)")
    c.drawRightString(doc.pagesize[0] - doc.rightMargin, 8 * mm, f"Page {doc.page}")
    c.restoreState()


def main():
    page = landscape(A3)
    doc = SimpleDocTemplate(OUT, pagesize=page, leftMargin=12 * mm, rightMargin=12 * mm, topMargin=10 * mm,
                            bottomMargin=13 * mm, title="Astra: proposed event-detection vision model",
                            author="Astra runtime")
    W = page[0] - doc.leftMargin - doc.rightMargin
    s = [
        P("Event-detection vision model: proposed architecture", title),
        P("One frozen pretrained vision backbone with small trained heads, one per object state. It replaces the "
          "hand-tuned colour masks and fixed region calibration, and fills the outcome checks (VERIFY) of the "
          "orchestrator HSM proposal. The rule-based detectors that use the robot's own sensing stay, and join at "
          "the decision layer. Shapes are per 5 Hz inference tick with one environment.", sub),
        legend(W), Spacer(1, 4),
        architecture_diagram(W, 676),
        PageBreak(),
        P("1. Layers: inputs, outputs, size and rate", h1),
        P("Timings are estimates for an RTX 5090 in bf16 and need measuring. The control loop runs at 50 Hz; the "
          "model runs on every 10th tick and the orchestrator reuses the last result in between, as the current "
          "stage classifier contract already does."),
        table(["Layer", "Module", "Input", "Output", "Parameters", "Rate, time"], LAYERS,
              [W * f for f in (0.04, 0.17, 0.22, 0.27, 0.13, 0.17)]),
        Spacer(1, 4),
        P("<b>Total:</b> ≈ 10 M trainable parameters on ≈ 300 M frozen ones. Per tick ≈ 55 ms on the GPU "
          "(DINOv2 6 ms, SAM 2 3 × 15 ms, the rest under 5 ms), plus Grounding DINO once per second off the "
          "critical path. That fits the 200 ms budget of a 5 Hz tick. The sim waits for the policy, so timing only "
          "matters on the real robot."),
        P("Why this split", h2),
        *bullets([
            "<b>Frozen backbones carry the sim-to-real transfer.</b> Features from large pretrained models hold up "
            "much better between rendered and real images than a network trained on sim images. Only about 10 M "
            "parameters learn, so a few dozen real episodes are enough to re-fit them.",
            "<b>Object tokens make the heads object-centric.</b> “Is the part in the box” is answered from the "
            "part's own mask features and its position in base_link, not from the whole image, which makes it "
            "robust to clutter and to the arm passing through the view.",
            "<b>Geometry is computed, not learned.</b> The plane intersection is the same method the RGB workbench "
            "localizer uses (camera pose from FK, known support heights). The learned head only refines it.",
            "<b>Temporal context is needed for several checks:</b> the lid swinging after release, “stays shut for "
            "1 s”, and telling the start from stage 6, which look the same in a single frame.",
            "<b>The decision layer stays rule-based and inspectable.</b> Thresholds are tuned on replays, unknown "
            "leads to PROBE and never to an advance, and the robot's own sensing can veto a vision result.",
        ]),
        PageBreak(),
        P("2. Output heads: training labels and real-robot plan", h1),
        P("Sim labels come for free offline: the demos and sim replays record the lid joint, object poses, contact "
          "sensors and the sim's stage flags, which the runtime never sees. On the real robot each head needs its "
          "own label source; the right column lists the cheapest one we know of."),
        table(["Head", "Output", "Loss", "Sim label (privileged, offline only)", "Used by", "Real robot"],
              HEADS, [W * f for f in (0.08, 0.09, 0.08, 0.27, 0.2, 0.28)]),
        P("3. Training", h1),
        *bullets([
            "<b>Data:</b> the teleoperation demos, plus rollouts of the runtime, plus induced failures (lid released "
            "early, part dropped on the rim, sponge left in the chamber). The induced failures are the hard "
            "negatives that decide the quality of the outcome heads. Sample balanced over stages and outcomes.",
            "<b>Loss:</b> weighted sum of the head losses in table 2; heads whose label is missing in a frame (an "
            "object not visible) are masked.",
            "<b>Domain randomisation in the sim:</b> lighting, textures and colours of the table, floor and objects, "
            "camera pose jitter of a few degrees, image noise, blur and exposure. Without it the heads will lean on "
            "render-specific cues.",
            "<b>Real robot:</b> keep L2 frozen, fine-tune L3–L6 on 30–50 labelled real episodes, then refit the L7 "
            "temperatures and thresholds on held-out real episodes.",
        ]),
        P("4. Will it work?", h1),
        *bullets([
            "<b>In the sim: yes.</b> Every head has an exact label, and the scene is fixed (scene04_1.usd).",
            "<b>On the real robot: likely for the object-state, grasped-object, stage and coarse lid heads, after "
            "the real fine-tune.</b> Not without real data: sim-only training will leave a gap however good the "
            "backbone is.",
            "<b>Not from vision alone:</b> the lid fully shut (the sim needs ≤ 0.45°) and contact force. Use the "
            "kinematic lid estimate while the hand holds the lid, a touch probe at the end, and force or joint "
            "current sensing if the real robot has it.",
            "<b>To check first:</b> which sensors the real robot provides (depth, force, joint current); the real "
            "camera resolutions and calibration; and the measured latency of SAM 2 on three cameras. Depth, if "
            "available, replaces the plane intersection in L3.",
        ]),
    ]
    doc.build(s, onFirstPage=on_page, onLaterPages=on_page)
    print(OUT)


main()
