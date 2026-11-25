# /// script
# requires-python = ">=3.10"
# dependencies = ["h5py", "numpy", "pyarrow", "av"]
# ///
"""Stage flags of every recorded case, computed from its HDF5 without the simulator.

    uv run scripts/compute_stage_flags.py INPUT_DIR WORK_DIR [--workers 4]
    uv run scripts/compute_stage_flags.py --self-test

INPUT_DIR  the folder with batch1/ batch2/ ... (as for the converter)
WORK_DIR   the converter's work folder (OUTPUT_DIR.work); flags go to WORK_DIR/flags/<batch>/<case>.npz,
           where the converter's merge looks for them. Then: dataset_conversion.py ... --merge-only

Every step is checked the way ChamberServiceTask._check_success does it (13 scoring points, stages in
order, points latched, hold timers), on the recorded states:
    lid          states/articulation/Chamber001_01/joint_position[:, LID_JOINT]
    old part     states/rigid_object/Chamber002/root_pose
    new part     states/rigid_object/Chamber002_01/root_pose
    sponge       states/rigid_object/spongebody/root_pose
One part cannot be recomputed: the sponge-chamber contact sensor is not recorded. A wipe face counts
as touched while the sponge centre is inside the cavity and within --wipe-contact-dist of that face
(the task's own distance test, plus this stand-in for the contact). The stage-4 label boundary is
set by the sponge being put down after the fifth face, so the stand-in hardly moves it.

Written per case:
    flags       (T, 6) bool  the label flags (LabelFlags of the converter: flag 2 = part out of the chamber)
    task_flags  (T, 6) bool  the task's six stage flags, as get_success_stage_flags() would return them
    points      (T, 13) bool the 13 scoring points (POINT_NAMES)
and <case>.json with the first step of every point and flag. A summary of all cases is printed and
written to WORK_DIR/flags/summary.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import sys
import tarfile
import tempfile
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

FPS = 50
DT = 1.0 / FPS
CASE_RE = re.compile(r"ChamberServiceTask_\d+")
LID_JOINT = 0          # Chamber001_01 joint of the lid (single 90 deg hinge); confirm with lid_joint_names
LID_SPAN = np.pi / 2   # lid joint range (rad): normalised opening = |q - q_closed| / LID_SPAN

# ---- ChamberServiceTask constants (robofinals_tasks ... chamber_service_task.py) -------------------
LID_LENGTH_M = 0.698
LID_LIFT_HEIGHT_M = 0.01
LID_OPEN_THRESHOLD = 0.4
LID_CLOSED_THRESHOLD = 0.005
LID_STABLE_ANG_VEL = 0.1
STABLE_DURATION_S = 1.0
STABLE_LIN_VEL = 0.05
PART_LIFT_HEIGHT_M = 0.01
SPONGE_LIFT_HEIGHT_M = 0.01
REFERENCE_SETTLE_S = 0.5
BOX_CENTER_XY = (0.6979, -0.4223)
BOX_HALF_SIZE_XY = 0.175
BOX_TOP_Z = 0.8049
NEW_PART_PLACE_DISTANCE_M = 0.05
CHAMBER_INNER_HALF_X = 0.344
CHAMBER_INNER_HALF_Y = 0.347
CHAMBER_FLOOR_BELOW_PART_M = 0.183
CHAMBER_INNER_HEIGHT_M = 0.30
SPONGE_CHAMBER_CONTACT_DURATION_S = 0.5
WIPE_FACE_MAX_DISTANCE_M = 0.08
WIPE_FACE_OUTSIDE_TOLERANCE_M = 0.01
WIPE_FACE_NAMES = ("wall_x_neg", "wall_x_pos", "wall_y_neg", "wall_y_pos", "floor")
WIPE_CONTACT_DIST = 0.08   # stand-in for the contact sensor: the task's own face distance (WIPE_FACE_MAX_DISTANCE_M)
POINT_NAMES = (
    "1a_lid_lifted_1cm", "1b_lid_stable_open", "2_part_lifted_1cm", "3_part_in_box_stable",
    "4a_sponge_picked_up", "4b_wipe_wall_x_neg", "4b_wipe_wall_x_pos", "4b_wipe_wall_y_neg",
    "4b_wipe_wall_y_pos", "4b_wipe_floor", "5a_new_part_lifted_1cm", "5b_new_part_placed_stable",
    "6_lid_closed_stable",
)
STAGE_POINT_SLICES = ((0, 2), (2, 3), (3, 4), (4, 10), (10, 12), (12, 13))
STATE_KEYS = {
    "lid": "states/articulation/Chamber001_01/joint_position",
    "old": "states/rigid_object/Chamber002/root_pose",
    "new": "states/rigid_object/Chamber002_01/root_pose",
    "sponge": "states/rigid_object/spongebody/root_pose",
}


# =========================================================================== the task's checks
def task_points(lid_q, old, new, sponge, dt=DT, lid_span=LID_SPAN, contact_dist=WIPE_CONTACT_DIST):
    """NumPy copy of ChamberServiceTask._check_success over a whole recording (one env).
    lid_q (T,) lid joint; old, new, sponge (T, 3) env-local positions. Returns points (T, 13) and
    stage flags (T, 6), both as they stand after each step."""
    lid_q = np.asarray(lid_q, dtype=np.float64)
    old, new, sponge = (np.asarray(x, dtype=np.float32)[:, :3] for x in (old, new, sponge))
    T = len(lid_q)
    f32 = np.float32
    dt32 = f32(dt)
    lid_norm = (np.abs(lid_q - lid_q[0]) / lid_span).astype(np.float32)   # 0 = closed (at reset), 1 = open
    lid_angle = (np.clip(lid_norm, 0.0, 1.0) * lid_span).astype(np.float32)

    P = np.zeros(13, dtype=bool)
    hold = np.zeros(4, dtype=np.float32)
    face_time = np.zeros(5, dtype=np.float32)
    ref = np.zeros(3, dtype=np.float32)
    ref_ok = False
    new_min, sponge_min = f32(np.inf), f32(np.inf)
    settle = max(1, int(round(REFERENCE_SETTLE_S / dt)))
    points = np.zeros((T, 13), dtype=bool)
    stages = np.zeros((T, 6), dtype=bool)

    def held(i, cond):
        hold[i] = hold[i] + dt32 if cond else f32(0.0)
        return hold[i]

    for t in range(T):
        steps = t + 1
        if t == 0:
            old_speed = new_speed = lid_rate = np.inf
        else:
            old_speed = float(np.linalg.norm(old[t] - old[t - 1])) / dt
            new_speed = float(np.linalg.norm(new[t] - new[t - 1])) / dt
            lid_rate = abs(float(lid_angle[t] - lid_angle[t - 1])) / dt
        new_min = min(new_min, new[t, 2])
        sponge_min = min(sponge_min, sponge[t, 2])
        if not ref_ok and steps >= settle:
            ref, ref_ok = old[t].copy(), True

        # stage 1: lift off the chamber top
        P[0] |= LID_LENGTH_M * np.sin(lid_angle[t]) >= LID_LIFT_HEIGHT_M
        still_open = lid_norm[t] >= LID_OPEN_THRESHOLD and lid_rate <= LID_STABLE_ANG_VEL
        P[1] |= P[0] and held(0, P[0] and still_open) >= STABLE_DURATION_S
        s1 = P[0] and P[1]
        # stage 2: extract the large part
        P[2] |= s1 and ref_ok and old[t, 2] >= ref[2] + PART_LIFT_HEIGHT_M
        s2 = s1 and P[2]
        # stage 3: pack the part into the box
        in_box = (abs(old[t, 0] - BOX_CENTER_XY[0]) <= BOX_HALF_SIZE_XY and abs(old[t, 1] - BOX_CENTER_XY[1]) <= BOX_HALF_SIZE_XY
                  and old[t, 2] < BOX_TOP_Z)
        P[3] |= s2 and held(1, s2 and in_box and old_speed <= STABLE_LIN_VEL) >= STABLE_DURATION_S
        s3 = s2 and P[3]
        # stage 4: wipe the chamber
        P[4] |= s3 and sponge[t, 2] >= sponge_min + SPONGE_LIFT_HEIGHT_M
        active = s3 and P[4] and ref_ok
        rel = sponge[t] - ref
        height = rel[2] + CHAMBER_FLOOR_BELOW_PART_M
        dist = np.array([rel[0] + CHAMBER_INNER_HALF_X, CHAMBER_INNER_HALF_X - rel[0],
                         rel[1] + CHAMBER_INNER_HALF_Y, CHAMBER_INNER_HALF_Y - rel[1], height], dtype=np.float32)
        face = int(np.argmin(dist))
        in_cavity = dist[face] >= -WIPE_FACE_OUTSIDE_TOLERANCE_M and height <= CHAMBER_INNER_HEIGHT_M
        contact = active and dist[face] <= contact_dist                     # stand-in for the contact sensor
        touching = contact and in_cavity and dist[face] <= WIPE_FACE_MAX_DISTANCE_M
        mask = np.zeros(5, dtype=bool)
        mask[face] = touching
        face_time = np.where(mask, face_time + dt32, f32(0.0)).astype(np.float32)
        P[5:10] |= mask & (face_time >= SPONGE_CHAMBER_CONTACT_DURATION_S)
        s4 = s3 and bool(P[4:10].all())
        # stage 5: unbox and place the replacement
        P[10] |= s4 and new[t, 2] >= new_min + PART_LIFT_HEIGHT_M
        at_target = float(np.linalg.norm(new[t] - ref)) <= NEW_PART_PLACE_DISTANCE_M
        placed_still = ref_ok and at_target and new_speed <= STABLE_LIN_VEL
        P[11] |= s4 and P[10] and held(2, s4 and P[10] and placed_still) >= STABLE_DURATION_S
        s5 = s4 and P[10] and P[11]
        # stage 6: close up the chamber
        closed_still = lid_norm[t] <= LID_CLOSED_THRESHOLD and lid_rate <= LID_STABLE_ANG_VEL
        P[12] |= s5 and held(3, s5 and closed_still) >= STABLE_DURATION_S
        s6 = s5 and P[12]

        points[t] = P
        stages[t] = (s1, s2, s3, s4, s5, s6)
    return points, stages, ref


def label_flags(stages, old, label_flags_cls):
    """Task flags -> label flags with the converter's LabelFlags (flag 2 = part out of the chamber)."""
    lf = label_flags_cls()
    ref = None
    out = np.zeros_like(stages)
    settle = max(1, int(round(REFERENCE_SETTLE_S / DT)))
    for t in range(len(stages)):
        if t + 1 >= settle and ref is None:
            ref = np.asarray(old[t, :3], dtype=np.float64)
        out[t] = lf(stages[t], old[t, :3], ref if ref is not None else old[0, :3])
    return out


# =========================================================================== reading a case
def find_cases(root: Path, batches):
    out = []
    for b in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if batches and b.name not in batches:
            continue
        seen = set()
        for p in sorted(b.iterdir()):
            m = CASE_RE.search(p.name)
            if m and m.group(0) not in seen and (p.is_dir() or p.name.endswith((".tar", ".tar.gz", ".tgz"))):
                seen.add(m.group(0))
                out.append((b.name, m.group(0), p))
    return out


def hdf5_of_case(path: Path, tmp: Path) -> Path:
    if path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix in (".hdf5", ".h5"):
                return p
        raise FileNotFoundError("no HDF5 in case folder")
    with tarfile.open(path, "r|*") as tf:
        for m in tf:
            if m.isfile() and m.name.endswith((".hdf5", ".h5")):
                dest = tmp / Path(m.name).name
                with tf.extractfile(m) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst, 16 * 2**20)
                return dest
    raise FileNotFoundError("no HDF5 in case archive")


def load_states(h5: Path, lid_joint: int) -> dict:
    import h5py

    with h5py.File(h5, "r") as f:
        data = f["data"]
        demos = [k for k in data if "actions" in data[k]]
        if not demos:
            raise ValueError("no demo with actions")
        g = data[max(demos, key=lambda k: data[k]["actions"].shape[0])]   # the converter's choice
        T = g["actions"].shape[0]
        out = {"T": T}
        for k, key in STATE_KEYS.items():
            if key not in g:
                raise KeyError(f"missing {key}")
            out[k] = g[key][()]
            if len(out[k]) != T:
                raise ValueError(f"{key} has {len(out[k])} rows, actions {T}")
        if out["lid"].shape[1] <= lid_joint:
            raise ValueError(f"chamber has {out['lid'].shape[1]} joints, lid joint {lid_joint} requested")
        out["lid"] = out["lid"][:, lid_joint]
    return out


def first_steps(x):
    return [int(np.argmax(x[:, k])) if x[:, k].any() else None for k in range(x.shape[1])]


def label_case(batch, cid, path, flags_dir, opts_d, converter_path):
    opts = argparse.Namespace(**opts_d)
    t0 = time.time()
    rec = {"batch": batch, "case": cid, "status": "failed"}
    tmp = Path(tempfile.mkdtemp(prefix=f"{cid}_", dir=opts.tmp))
    try:
        conv = load_converter(converter_path)
        s = load_states(hdf5_of_case(Path(path), tmp), opts.lid_joint)
        points, stages, ref = task_points(s["lid"], s["old"], s["new"], s["sponge"],
                                          lid_span=opts.lid_span, contact_dist=opts.wipe_contact_dist)
        flags = label_flags(stages, s["old"], conv.LabelFlags)
        out = Path(flags_dir) / batch
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"{cid}.npz", flags=flags, task_flags=stages, points=points)
        pts = first_steps(points)
        rec.update(status="ok", steps=s["T"], points=dict(zip(POINT_NAMES, pts)),
                   task_flag_steps=first_steps(stages), label_flag_steps=first_steps(flags),
                   stages_reached=int(stages[-1].sum()), old_part_ref=[float(v) for v in ref],
                   lid_joint=opts.lid_joint, wipe_contact_dist=opts.wipe_contact_dist)
        (out / f"{cid}.json").write_text(json.dumps(rec, indent=1))
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc(limit=3)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def load_converter(path):
    spec = importlib.util.spec_from_file_location("galbot_converter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def default_converter() -> Path:
    here = Path(__file__).resolve().parent
    for name in ("dataset_conversion.py", "convert_dataset.py"):
        if (here / name).is_file():
            return here / name
    raise SystemExit("converter not found next to this script; pass --converter PATH")


# =========================================================================== self-test
def synthetic_case(T=3000, seed=0):
    """A recording that does the whole task once, with the step each point should come on."""
    ref = np.array([-0.52, -0.26, 0.998], dtype=np.float32)
    lid = np.zeros(T)
    lid[200:220] = np.linspace(0, -np.pi / 2, 20)
    lid[220:2800] = -np.pi / 2
    lid[2800:2820] = np.linspace(-np.pi / 2, 0, 20)
    old = np.tile(ref, (T, 1))
    box = np.array([0.70, -0.42, 0.765], dtype=np.float32)
    up = ref + [0, 0, 0.2]
    old[400:450] = np.linspace(ref, up, 50)                      # lifted out
    old[450:600] = np.linspace(up, box + [0, 0, 0.2], 150)       # carried
    old[600:650] = np.linspace(box + [0, 0, 0.2], box, 50)       # lowered into the box
    old[650:] = box
    sponge0 = np.array([0.37, -0.50, 0.85], dtype=np.float32)
    sponge = np.tile(sponge0, (T, 1))
    sponge[750:800] = np.linspace(sponge0, ref + [0, 0, 0.1], 50)
    faces = [ref + [-0.30, 0, -0.05], ref + [0.30, 0, -0.05], ref + [0, -0.30, -0.05], ref + [0, 0.30, -0.05],
             ref + [0, 0, -0.15]]                                  # 4 cm from each wall, 3.3 cm above the floor
    k = 800
    for p in faces:
        sponge[k:k + 60] = p                                     # 1.2 s on each face
        k += 60
    sponge[k:k + 50] = np.linspace(faces[-1], sponge0, 50)
    sponge[k + 50:] = sponge0
    new0 = np.array([0.95, -0.42, 0.765], dtype=np.float32)
    new = np.tile(new0, (T, 1))
    new[1500:1600] = np.linspace(new0, ref + [0, 0, 0.2], 100)
    new[1600:1650] = np.linspace(ref + [0, 0, 0.2], ref + [0.01, 0, 0], 50)
    new[1650:] = ref + [0.01, 0, 0]
    return lid, old, new, sponge


def self_test(converter_path):
    lid, old, new, sponge = synthetic_case()
    points, stages, ref = task_points(lid, old, new, sponge)
    p = dict(zip(POINT_NAMES, first_steps(points)))
    s = first_steps(stages)
    # lid: 1 cm rise at angle asin(0.01/0.698) -> first step of the opening ramp past it; open and
    # still from step 220 (rate 0 from 221) -> 1 s later
    assert p["1a_lid_lifted_1cm"] == 201, p
    assert p["1b_lid_stable_open"] == 221 + 49, p
    assert p["2_part_lifted_1cm"] in (403, 404), p
    assert p["3_part_in_box_stable"] == 651 + 49, p            # still from 651, inside from ~649
    assert p["4a_sponge_picked_up"] is not None and 751 <= p["4a_sponge_picked_up"] <= 752, p
    for i, name in enumerate(POINT_NAMES[5:10]):               # 0.5 s (25 steps) into each 60-step face
        assert p[name] == 800 + 60 * i + 24, (name, p)
    assert p["5a_new_part_lifted_1cm"] == 1503, p      # 4.3 mm per step from step 1500
    assert p["5b_new_part_placed_stable"] == 1651 + 49, p
    assert p["6_lid_closed_stable"] == 2820 + 50, p   # closed and still from 2820, like 1b
    assert s == [270, p["2_part_lifted_1cm"], 700, 800 + 4 * 60 + 24, 1700, 2870], s
    assert stages[-1].all()
    # the label flags: flag 2 when the part leaves the cavity
    conv = load_converter(converter_path)
    lab = label_flags(stages, old, conv.LabelFlags)
    gone = np.flatnonzero((np.abs(old[:, 0] - ref[0]) > CHAMBER_INNER_HALF_X) | (np.abs(old[:, 1] - ref[1]) > CHAMBER_INNER_HALF_Y))
    assert first_steps(lab)[1] == int(gone[0]) and 450 < int(gone[0]) < 600, (first_steps(lab), gone[:3])
    assert [a for i, a in enumerate(first_steps(lab)) if i != 1] == [b for i, b in enumerate(s) if i != 1]
    # a wipe that hovers too far from the walls is not credited
    far = sponge.copy()
    far[800:1100] = ref + [0, 0, 0.05]                          # centre of the cavity
    _, st_far, _ = task_points(lid, old, new, far)
    assert not st_far[:, 3].any(), "hovering sponge credited"
    # missing stage 5: the new part is never placed -> stages 5 and 6 stay off
    lazy = np.tile(new[0], (len(new), 1))
    _, st_lazy, _ = task_points(lid, old, lazy, sponge)
    assert st_lazy[:, 3].any() and not st_lazy[:, 4].any() and not st_lazy[:, 5].any()
    # end to end on a case folder
    import h5py

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        case = d / "in" / "batch1" / "ChamberServiceTask_1"
        case.mkdir(parents=True)
        T = len(lid)
        q = np.zeros((T, 5))
        q[:, 0] = lid
        pose = lambda x: np.concatenate([x, np.tile([0, 0, 0, 1.0], (len(x), 1))], 1)
        with h5py.File(case / "galbot_success.hdf5", "w") as f:
            f.create_group("data/demo_0")
            g = f.create_group("data/demo_3")
            g["actions"] = np.zeros((T, 20))
            g[STATE_KEYS["lid"]] = q
            g[STATE_KEYS["old"]] = pose(old)
            g[STATE_KEYS["new"]] = pose(new)
            g[STATE_KEYS["sponge"]] = pose(sponge)
        opts = {"lid_joint": 0, "lid_span": LID_SPAN, "wipe_contact_dist": WIPE_CONTACT_DIST, "tmp": str(d)}
        r = label_case("batch1", "ChamberServiceTask_1", str(case), str(d / "flags"), opts, str(converter_path))
        assert r["status"] == "ok", r
        z = np.load(d / "flags" / "batch1" / "ChamberServiceTask_1.npz")
        assert z["flags"].shape == (T, 6) and np.array_equal(z["flags"], lab) and z["points"].shape == (T, 13)
    print("self-test passed")


# =========================================================================== main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=Path, nargs="?", help="folder with the batch folders")
    ap.add_argument("work_dir", type=Path, nargs="?", help="the converter's work folder (OUTPUT_DIR.work)")
    ap.add_argument("--flags-dir", type=Path, help="where to write the flags [WORK_DIR/flags]")
    ap.add_argument("--batches", nargs="+", help="only these batches")
    ap.add_argument("--limit", type=int, help="at most N cases per batch (quick test)")
    ap.add_argument("--workers", type=int, default=4, help="cases labelled in parallel [4]")
    ap.add_argument("--redo", action="store_true", help="label cases that already have flags again")
    ap.add_argument("--lid-joint", type=int, default=LID_JOINT, help="chamber joint of the lid [0]")
    ap.add_argument("--lid-span", type=float, default=LID_SPAN, help="lid joint range in rad [pi/2]")
    ap.add_argument("--wipe-contact-dist", type=float, default=WIPE_CONTACT_DIST,
                    help="sponge centre this close to a cavity face counts as contact, m [%(default)s]")
    ap.add_argument("--tmp", type=Path, help="temp dir for HDF5 files extracted from archives [WORK_DIR/tmp]")
    ap.add_argument("--converter", type=Path, help="dataset_conversion.py (for LabelFlags) [next to this script]")
    ap.add_argument("--self-test", action="store_true", help="check the port of the task's checks and exit")
    opts = ap.parse_args()
    converter = (opts.converter or default_converter()).resolve()
    if opts.self_test:
        self_test(converter)
        return 0
    if not (opts.input_dir and opts.work_dir):
        ap.error("INPUT_DIR and WORK_DIR are required")

    flags_dir = (opts.flags_dir or opts.work_dir / "flags").resolve()
    tmp = (opts.tmp or opts.work_dir / "tmp").resolve()
    flags_dir.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    cases = find_cases(opts.input_dir.resolve(), opts.batches)
    todo, per_batch = [], Counter()
    for b, cid, p in cases:
        if not opts.redo and (flags_dir / b / f"{cid}.npz").exists():
            continue
        if opts.limit and per_batch[b] >= opts.limit:
            continue
        per_batch[b] += 1
        todo.append((b, cid, p))
    print(f"{len(cases)} cases, {len(todo)} to label with {opts.workers} worker(s); flags -> {flags_dir}", flush=True)
    opts_d = {"lid_joint": opts.lid_joint, "lid_span": opts.lid_span, "wipe_contact_dist": opts.wipe_contact_dist,
              "tmp": str(tmp)}
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=opts.workers) as ex:
        futs = {ex.submit(label_case, b, cid, str(p), str(flags_dir), opts_d, str(converter)): (b, cid)
                for b, cid, p in todo}
        for fut in as_completed(futs):
            done += 1
            try:
                r = fut.result()
            except Exception as exc:
                b, cid = futs[fut]
                r = {"batch": b, "case": cid, "status": "failed", "error": repr(exc)}
            eta = (time.time() - t0) / done * (len(todo) - done) / 60
            msg = (f"{r['stages_reached']}/6 stages, flag steps {r['label_flag_steps']}" if r["status"] == "ok"
                   else f"FAILED: {r['error']}")
            print(f"[{done}/{len(todo)}] {r['batch']}/{r['case']}: {msg}  (eta {eta:.0f} min)", flush=True)

    # summary over every case that has a record
    recs = [json.loads(p.read_text()) for p in sorted(flags_dir.glob("*/*.json"))]
    reached = Counter(r.get("stages_reached") for r in recs if r.get("status") == "ok")
    missing = Counter()
    for r in recs:
        for name, step in (r.get("points") or {}).items():
            if step is None:
                missing[name] += 1
    summary = {"cases": len(recs), "all_six_stages": reached.get(6, 0),
               "stages_reached": {str(k): v for k, v in sorted(reached.items())},
               "points_never_reached": dict(missing), "lid_joint": opts.lid_joint,
               "wipe_contact_dist": opts.wipe_contact_dist}
    (flags_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n{summary['all_six_stages']} of {len(recs)} cases reach all six stages; stages reached: "
          f"{summary['stages_reached']}")
    if missing:
        print("points never reached (cases): " + ", ".join(f"{k} {v}" for k, v in missing.most_common()))
    print(f"next: dataset_conversion.py INPUT_DIR OUTPUT_DIR --merge-only --unlabeled full  (flags in {flags_dir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())