"""Render the ACT / GR00T switchable-system diagram to astra_policy_diagram.pdf.

Usage: python3 make_diagram.py [OUTPUT_DIR]   (default: this script's directory)
"""

import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

VERSION = "v1.0"
OUT_NAME = "astra_policy_diagram"
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / f"{OUT_NAME}.pdf"

W, H = 30.0, 20.5
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

FS = 10.0  # body font size
LH = FS * 1.38 / 72  # line height in inches
PAD = 0.12

C = {
    "panel": ("#f4f4f0", "#c9c9c0"),
    "comp": ("#ffffff", "#8c8c8c"),
    "in": ("#dce8f7", "#7fa3d6"),
    "out": ("#ddefe1", "#7fbf8e"),
    "pol": ("#fbe2cd", "#e3a274"),
    "new": ("#fff4c2", "#d9b23a"),
}


def rbox(x, y, w, h, kind, lw=1.0, r=0.08):
    fc, ec = C[kind]
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw)
    )


def wrap(lines, w, fs=FS):
    """lines: list of str or (str, bold). Returns list of (text, bold, indent)."""
    chars = max(10, int((w - 2 * PAD) * 72 / (fs * 0.54)))
    out = []
    for item in lines:
        text, bold = (item, False) if isinstance(item, str) else item
        if text == "":
            out.append(("", False, 0))
            continue
        bullet = text.startswith("• ")
        sub = textwrap.wrap(text, int(chars * 0.88) if bold else chars, subsequent_indent="   " if bullet else "")
        for s in sub:
            out.append((s, bold, 0))
    return out


def text_block(x, y_top, w, lines, fs=FS):
    """Draw wrapped lines starting at y_top; return height used."""
    wl = wrap(lines, w, fs)
    lh = fs * 1.38 / 72
    for i, (s, bold, _) in enumerate(wl):
        ax.text(
            x + PAD,
            y_top - PAD - (i + 0.8) * lh,
            s,
            fontsize=fs,
            fontweight="bold" if bold else "normal",
            va="baseline",
            ha="left",
            family="DejaVu Sans",
        )
    return len(wl) * lh + 2 * PAD


def measured(w, lines, fs=FS):
    return len(wrap(lines, w, fs)) * fs * 1.38 / 72 + 2 * PAD


def card(x, y_top, w, kind, lines, fs=FS, h=None):
    hh = h if h is not None else measured(w, lines, fs)
    rbox(x, y_top - hh, w, hh, kind)
    text_block(x, y_top, w, lines, fs)
    return y_top - hh


def arrow(p0, p1, style="-|>", ls="-", color="#333", lw=1.1, conn="arc3"):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle=style, mutation_scale=11, lw=lw, color=color, linestyle=ls,
            connectionstyle=conn, shrinkA=0, shrinkB=0,
        )
    )


def panel(x, y, w, h, title):
    rbox(x, y, w, h, "panel", r=0.14)
    ax.text(x + 0.02, y + h + 0.1, title, fontsize=9.2, fontweight="bold", color="#555", va="bottom")


# ---------------------------------------------------------------- header
ax.text(0.4, 20.0, f"ACT / GR00T N1.7 switchable system  ·  {VERSION}, matched to the task repo sim and GR00T N1.7",
        fontsize=21, va="center")
ax.text(0.4, 19.5, "One schema in, one 19-D Galbot-Mobile action out  ·  "
        "source: task repo (galbot.py MobileActionsCfg, ChamberServiceTask)",
        fontsize=11.5, va="center", color="#333")
lx = 18.2
for label, kind in [("Inputs", "in"), ("Outputs", "out"), ("Swappable policy", "pol"),
                    ("Component", "comp"), ("Code we write / configure", "new")]:
    wbox = 0.25 + len(label) * 0.085
    rbox(lx, 19.33, wbox, 0.34, kind, r=0.15)
    ax.text(lx + wbox / 2, 19.5, label, fontsize=8.2, ha="center", va="center")
    lx += wbox + 0.2

# ---------------------------------------------------------------- action contract band
LX0, LX1 = 0.4, 17.2
ax.text(LX0, 19.0, "Action contract (fixed by the sim; order = ActionsCfg field order)",
        fontsize=10, fontweight="bold", color="#555", va="center")
segs = [
    (7, "a[0:7]  left EEF pose target\nx, y, z + quaternion · base_link frame · absolute", "out"),
    (7, "a[7:14]  right EEF pose target\nx, y, z + quaternion · base_link frame · absolute", "out"),
    (1, "a[14]\nL grip", "in"),
    (1, "a[15]\nR grip", "in"),
    (4, "a[16:19]  base vx, vy, wz\nbody frame · m/s, rad/s", "pol"),
]
bx = LX0
bw_unit = (LX1 - LX0) / 20
for n, label, kind in segs:
    rbox(bx, 17.95, n * bw_unit - 0.04, 0.82, kind, r=0.05)
    ax.text(bx + n * bw_unit / 2, 18.36, label, fontsize=8.4, ha="center", va="center")
    bx += n * bw_unit
ax.text(LX0, 17.72,
        "Arms: Differential IK (dls, command_type=pose, use_relative_mode=False) → arm joint targets.   "
        "Grippers: BinaryJointPosition, open 0.0 / close 1.703 rad.   Base: body velocity vx, vy, wz (smoothed in the env).   "
        "No torso / head / base-twist action (leg_joint*, head_joint* held at default).",
        fontsize=8.4, va="center", color="#333")
ax.text(LX0, 17.42,
        "Model action space (23, shared): left_eef 9 (xyz+rot6d) · right_eef 9 · gripper 2 · base_vel 3   "
        "→ converted to a ∈ ℝ¹⁹ in layer 5.   Chunks in data time (50 Hz, one row per control step); control 50 Hz (dt 1/200, decimation 4).",
        fontsize=8.4, va="center", color="#7a5a00", fontweight="bold")

ax.text(LX0, 17.0, "Runtime loop, every control tick", fontsize=11.5, va="center")

# ---------------------------------------------------------------- runtime panels
PW = (LX1 - LX0 - 2 * 0.45) / 3
cols = [LX0 + i * (PW + 0.45) for i in range(3)]
R1_TOP, R1_BOT = 16.45, 10.95
R2_TOP, R2_BOT = 10.15, 4.55
IW = PW - 0.3


def fill_panel(x, top, blocks, gap=0.13):
    y = top - 0.15
    for kind, lines in blocks:
        y = card(x + 0.15, y, IW, kind, lines) - gap
    return y


# 1 · Simulator
panel(cols[0], R1_BOT, PW, R1_TOP - R1_BOT, "1 · Env server: Galbot-Mobile (outside submission)")
fill_panel(cols[0], R1_TOP, [
    ("comp", [("Isaac Lab ChamberServiceTask", True),
              "Physics 200 Hz, decimation 4 → 50 Hz control (read usr_args['decimation']). "
              "Episode limit 900 s. Talks to the policy container over 127.0.0.1:50000."]),
    ("in", [("In (from 5, via task_env.step)", True),
            "• a[0:14] L / R EEF pose targets (pos + quat)",
            "• a[14:16] gripper L, R (binary)",
            "• a[16:19] base velocity vx, vy, wz",
            "• Action terms (DiffIK, binary gripper, base vel) = layer 6, run inside the sim"]),
    ("out", [("Out (policy obs)", True),
             "• RGB: head 480×640, L / R wrist 224×224",
             "• joint_pos, joint_vel (all joints, absolute after the group merge), eef_pos, eef_quat, gripper_pos, last_action",
             "• No base odometry in obs",
             "• Stage flags: scoring only, not visible to the policy"]),
])

# 2 · Observation adapter
panel(cols[1], R1_BOT, PW, R1_TOP - R1_BOT, "2 · Observation adapter")
fill_panel(cols[1], R1_TOP, [
    ("comp", [("Policy obs → canonical schema → model views", True),
              "schema.yaml is the single source of truth: state slices, camera keys, rotation format, rates."]),
    ("in", [("In", True),
            "• Policy obs dict (from 1)",
            "• schema.yaml · decimation from eval metadata",
            "• No normalization here: each checkpoint carries its own processor"]),
    ("new", [("Out", True),
             "• State S = 34: left_arm_q 7, right_arm_q 7, gripper 2 (joint angles), "
             "left_eef_world 9, right_eef_world 9 (xyz + rot6d from eef_pos / eef_quat)",
             "• ACT view: 3 × (3,224,224) + state (34) → LeRobot preprocessor",
             "• GR00T view (Policy API): video.{head, left_wrist, right_wrist} (B,1,H,W,3) uint8 native size; "
             "state.<group> (B,1,D) float32; language.task [[prompt]]",
             "• Time base (50 Hz), stale flag, timestamp"]),
])

# 3 · Task orchestrator
panel(cols[2], R1_BOT, PW, R1_TOP - R1_BOT, "3 · Task orchestrator")
fill_panel(cols[2], R1_TOP, [
    ("comp", [("6-stage state machine", True),
              "Lift lid → extract part → pack in box → wipe chamber → place replacement → close. "
              "Stage progress from a learned classifier, not from sim flags."]),
    ("in", [("In", True),
            "• Canonical observation (from 2)",
            "• Subtask classifier p(stage | images, state)",
            "• Elapsed time per stage"]),
    ("out", [("Out", True),
             "• Subtask ID (0–5)",
             "• Fixed prompt per stage, e.g. \"wipe the chamber interior\"",
             "• Advance / retry / skip (no reset: interventions score 0)",
             "• Time budget per stage, Σ ≤ 900 s"]),
])

# 4 · Router + policies
panel(cols[0], R2_BOT, PW, R2_TOP - R2_BOT, "4 · Router + policies")
y = R2_TOP - 0.15
y = card(cols[0] + 0.15, y, IW, "new", [
    ("Policy router", True),
    "act | groot | per-subtask map. Switch only at stage boundaries; flush queue on switch. "
    "Latency watchdog → hold current pose (no mid-stage handoff)."]) - 0.13
half = (IW - 0.15) / 2
act_lines = [("Custom ACT", True), "In: ACT view + subtask one-hot",
             "Out: chunk (k, 23), k = 50–100, un-normalized by LeRobot postprocessor"]
gr_lines = [("GR00T N1.7 (Gr00tPolicy)", True), "In: GR00T view, NEW_EMBODIMENT",
            "Out: get_action() dict action.<group> (B, 40, D), absolute, physical units"]
hh = max(measured(half, act_lines), measured(half, gr_lines))
card(cols[0] + 0.15, y, half, "pol", act_lines, h=hh)
card(cols[0] + 0.30 + half, y, half, "pol", gr_lines, h=hh)
y = y - hh - 0.13
y = card(cols[0] + 0.15, y, IW, "in", [("In", True), "• Subtask ID + prompt (from 3)",
                                        "• ACT view / GR00T view (from 2)"]) - 0.13
card(cols[0] + 0.15, y, IW, "out", [("Out", True),
                                     "• Chunk (T, 23) = [left_eef 9 | right_eef 9 | gripper 2 | base_vel 3], "
                                     "absolute (GR00T groups concatenated in this order)",
                                     "• Source tag: act | groot"])

# 5 · Action post-processor
panel(cols[1], R2_BOT, PW, R2_TOP - R2_BOT, "5 · Action post-processor")
fill_panel(cols[1], R2_TOP, [
    ("comp", [("Chunk executor + safety", True),
              "Execute first k steps (ACT ≈ 25, GR00T execution horizon ≈ 20 = 0.4 s), chunk time mapped to control ticks; "
              "smooth chunk joins on continuous dims only."]),
    ("in", [("In", True),
            "• Chunk + source tag (from 4)",
            "• Current EEF pose, gripper state (from 2)",
            "• Workspace box, max EEF step per tick, base velocity limits"]),
    ("new", [("Out: one a ∈ ℝ¹⁹ per tick", True),
             "• EEF pos clamped to workspace + per-tick step limit",
             "• rot6d → unit quaternion in the sim's order, hemisphere-consistent",
             "• Gripper: ACT logit → sigmoid; GR00T continuous [0,1]; threshold + hysteresis → binary",
             "• Wheel velocities clamped, deadband",
             "• Packed [L pose 7 | R pose 7 | gL | gR | vx vy wz]"]),
])

# 6 · Low-level controllers
panel(cols[2], R2_BOT, PW, R2_TOP - R2_BOT, "6 · Low-level controllers (Isaac Lab action terms, in sim)")
fill_panel(cols[2], R2_TOP, [
    ("comp", [("Applied on the same env step", True),
              "Not part of the submission: provided by the env server's ActionsCfg."]),
    ("in", [("In", True), "• a ∈ ℝ¹⁹ (from 5) via task_env.step()"]),
    ("out", [("Out", True),
             "• L / R arms: DiffIK (dls) → 7 + 7 joint targets",
             "• Grippers: 0.0 / 1.703 rad",
             "• Base: wheel velocity targets + root velocity",
             "• Torso, head: fixed at default pose",
             "• ↻ Back to 1 on the next tick"]),
])

# runtime arrows
mid1 = (R1_TOP + R1_BOT) / 2 + 0.6
mid2 = (R2_TOP + R2_BOT) / 2 + 0.6
arrow((cols[0] + PW, mid1), (cols[1], mid1))
arrow((cols[1] + PW, mid1), (cols[2], mid1))
arrow((cols[0] + PW, mid2), (cols[1], mid2))
arrow((cols[1] + PW, mid2), (cols[2], mid2))
# 3 -> 4 elbow
yb = (R1_BOT + R2_TOP) / 2
xa, xb = cols[2] + PW / 2, cols[0] + PW / 2
ax.plot([xa, xa, xb], [R1_BOT, yb, yb], color="#333", lw=1.1)
arrow((xb, yb), (xb, R2_TOP))

# ---------------------------------------------------------------- offline pipeline
OF_TOP, OF_BOT = 4.0, 1.15
panel(LX0, OF_BOT, LX1 - LX0, OF_TOP - OF_BOT, "Offline training pipeline (feeds 2, 3 and 4)")
off = [
    ("comp", [("Data collection", True),
              "task dataset (50 Hz, 77-D joint_pos state, 20-D or 19-D action), "
              "scripted expert, teleop, recovery demos"]),
    ("new", [("Labeler (sim replay)", True),
             "Replay episodes → stage flags → per-frame subtask labels; FK → EEF pose state; "
             "drop failed episodes"]),
    ("new", [("Dataset builder", True),
             "state 34 / action 23 (quat → xyz+rot6d, grip → {0,1}, wheels → vx vy wz); stage prompt column. "
             "ACT: LeRobot v3. GR00T: convert_v3_to_v2 + meta/modality.json + stats.py"]),
    ("comp", [("Trainers", True),
              "ACT (L1 + BCE grip + β·KL); subtask classifier (→ 3); GR00T launch_finetune.py "
              "from GR00T-N1.7-3B, galbot_config.py (→ 4)"]),
    ("comp", [("Evaluator", True),
              "Env server, fixed seeds → per-milestone success per model → subtask→model map (→ 4)"]),
    ("comp", [("Packager", True),
              "Layers 2–5, ACT + GR00T checkpoints, Cosmos-Reason2-2B, ResNet caches; transformers 4.57.3, py3.12, CUDA 12.8; HF_HUB_OFFLINE=1; "
              "test with --network none"]),
]
n = len(off)
ow = (LX1 - LX0 - 0.3 - (n - 1) * 0.3) / n
ox = LX0 + 0.15
oh = OF_TOP - OF_BOT - 0.3
for i, (kind, lines) in enumerate(off):
    card(ox, OF_TOP - 0.15, ow, kind, lines, h=oh)
    if i < n - 1:
        arrow((ox + ow, OF_TOP - 0.15 - oh / 2), (ox + ow + 0.3, OF_TOP - 0.15 - oh / 2))
    if lines[0][0] == "Trainers":
        tx = ox + ow / 2
    ox += ow + 0.3
arrow((tx, OF_TOP - 0.15), (cols[0] + PW * 0.7, R2_BOT), ls="--",
      conn="angle,angleA=90,angleB=180,rad=0")

# footer
rbox(LX0, 0.3, LX1 - LX0, 0.55, "comp", r=0.1)
ax.text((LX0 + LX1) / 2, 0.575,
        "Submission image = policy container with layers 2–5 + all weights + backbone · env server (layers 1, 6) "
        "is provided · no network at evaluation · same simulator config for every team",
        fontsize=9, ha="center", va="center", fontweight="bold")

# ---------------------------------------------------------------- policy internals
RX0 = 17.85
ax.text(RX0, 19.0, "Policy internals: layer 4 in detail", fontsize=11.5, va="center")
CW = (W - 0.4 - RX0 - 0.35) / 2
ccols = [RX0, RX0 + CW + 0.35]
IN_TOP, IN_BOT = 18.4, 0.3
panel(ccols[0], IN_BOT, CW, IN_TOP - IN_BOT, "4a · Custom ACT (inside layer 4)")
panel(ccols[1], IN_BOT, CW, IN_TOP - IN_BOT, "4b · GR00T N1.7 (inside layer 4)")

act_stack = [
    ("in", [("Inputs (ACT view from layer 2)", True),
            "• Images: 3 cameras, (3, 3, 224, 224), ImageNet-normalized",
            "• State S = 34 (EEF xyz+rot6d 9+9, arm q 7+7, grip 2), z-scored",
            "• Subtask ID 0–5 · training only: target chunk (k, 23)"]),
    ("pol", [("1 · Vision backbone  ResNet-18 (ImageNet weights baked into image)", True),
             "In: one (3, 224, 224) image per camera",
             "Out: (512, 7, 7) → 49 tokens × 512 per camera"]),
    ("pol", [("2 · State + task embedding  linear layers", True),
             "In: state (34), subtask one-hot (6)",
             "Out: 1 proprio token + 1 task token, 512 each"]),
    ("pol", [("3 · CVAE style encoder (training only)", True),
             "In: [CLS] + state + target chunk (k, 23)",
             "Out: latent z (32) → 1 token · at inference z = 0"]),
    ("pol", [("4 · Transformer encoder  4 layers, d = 512, 8 heads", True),
             "In: 147 image tokens + proprio + task + z, 2-D sine position embeddings",
             "Out: memory, 150 × 512"]),
    ("pol", [("5 · Transformer decoder  DETR-style (LeRobot default 1 layer)", True),
             "In: k learned queries (k = 50–100) + memory",
             "Out: k × 512 action embeddings"]),
    ("new", [("6 · Action heads", True),
             "Linear 512 → 21: left_eef 9, right_eef 9 (xyz+rot6d, absolute), base_vel 3",
             "Linear 512 → 2: gripper logits L, R"]),
    ("new", [("Outputs (to layer 5)", True),
             "• Chunk (k, 23) = [left_eef 9 | right_eef 9 | gripper 2 | base_vel 3]",
             "• Un-normalized by LeRobot policy_postprocessor (saved with checkpoint)",
             "• Temporal ensembling on continuous dims only",
             "• Loss: L1 (eef, base) + BCE (grip) + β·KL (β ≈ 10)"]),
]
gr_stack = [
    ("in", [("Inputs (GR00T view from layer 2, Policy API)", True),
            "• video.head (B,1,480,640,3), video.left_wrist / right_wrist (B,1,224,224,3) uint8",
            "• state.left_arm_q, state.right_arm_q (7 each), state.gripper (2), "
            "state.left_eef_world, state.right_eef_world (9 each): (B,1,D) float32, raw units",
            "• language.task [[stage prompt]] · EmbodimentTag.NEW_EMBODIMENT",
            "• Training only: action chunk delta_indices 0–39 (40 steps at 50 Hz = 0.8 s)"]),
    ("new", [("galbot_config.py (register_modality_config, saved in checkpoint)", True),
              "action groups + ActionConfig:",
              "• left_eef, right_eef: ABSOLUTE, NON_EEF, DEFAULT (root-frame targets)",
              "• gripper: ABSOLUTE, NON_EEF, DEFAULT",
              "• base_vel: ABSOLUTE, NON_EEF, DEFAULT",
              "state: optional sin_cos_embedding_keys = arm_q; meta/modality.json maps start / end slices"]),
    ("pol", [("1 · Processor  Gr00tN1d7Processor (framework, configured)", True),
             "In: obs dict · Images: native aspect ratio, shortest edge 256 (crop 0.95 in training)",
             "Out: state min-max normalized (stats.json), concatenated, padded to 132; "
             "actions min-max normalized; prompt tokens"]),
    ("pol", [("2 · VLM backbone (System 2)  Cosmos-Reason2-2B (Qwen3-VL)", True),
             "In: images + prompt tokens · select_layer 12",
             "Default fine-tune: LLM and vision frozen; projector + DiT trained"]),
    ("pol", [("3 · State + action encoders  embodiment-specific MLPs", True),
             "In: state (132) · noisy action chunk (40 × 132) + flow time τ",
             "Out: 1 state token + 40 action tokens"]),
    ("pol", [("4 · Diffusion transformer (System 1)  flow-matching DiT, 16 layers", True),
             "In: state + action tokens, cross-attending to VLM embeddings",
             "Out: predicted velocity per action token"]),
    ("pol", [("5 · Action decoder  embodiment-specific MLP", True),
             "Out: denoised chunk (40, 132) · steps 3–5 repeat 4× (num_inference_timesteps)"]),
    ("pol", [("6 · Un-processor  decode_action / unapply (framework, configured)", True),
             "Slice 23 used dims by group → un-normalize (all groups absolute)"]),
    ("new", [("Outputs: get_action() → (action, info), to layer 5", True),
             "• action.left_eef / right_eef (B,40,9), action.gripper (B,40,2), "
             "action.base_vel (B,40,3): absolute, physical units",
             "• Gripper is continuous (flow matching, no logits)",
             "• Execution horizon 8–16 steps, then re-plan (RTC optional; 40 ≥ 32)",
             "• Loss: flow matching on used dims, padding masked"]),
]

for cx, stack in zip(ccols, [act_stack, gr_stack]):
    bw = CW - 0.4
    hs = [measured(bw, lines) for _, lines in stack]
    avail = IN_TOP - IN_BOT - 0.3
    gap = (avail - sum(hs)) / (len(stack) - 1)
    y = IN_TOP - 0.15
    for i, ((kind, lines), h) in enumerate(zip(stack, hs)):
        card(cx + 0.2, y, bw, kind, lines, h=h)
        if i < len(stack) - 1:
            arrow((cx + CW / 2, y - h), (cx + CW / 2, y - h - gap))
        y -= h + gap


fig.savefig(OUT, format="pdf")
fig.savefig(OUT.with_suffix(".png"), dpi=60)
print(f"wrote {OUT}")
