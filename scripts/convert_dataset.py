# /// script
# requires-python = ">=3.10"
# dependencies = ["h5py", "numpy", "pyarrow", "av"]
# ///
"""Convert the task dataset (one recorded case per archive or folder) into one LeRobot v2.1 dataset
for GR00T fine-tuning.

    uv run scripts/convert_dataset.py INPUT_DIR OUTPUT_DIR [--workers 4]
    uv run scripts/convert_dataset.py --self-test

INPUT_DIR   the folder that holds batch1/ batch2/ ... with one archive or folder per case
OUTPUT_DIR  the LeRobot v2.1 dataset to write (meta/ data/ videos/); created if missing.
            Work files go to OUTPUT_DIR.work next to it, so a rerun only converts new cases.

What goes in:
  * Source of truth is the cleaned galbot_success.hdf5 of every case.
  * Cases whose HDF5 holds the camera frames: they are encoded to H.264.
  * Cases without cameras in the HDF5: video comes from galbot.zip. Its rows are matched to HDF5
    steps by action (exact); unmatched rows (an extra block before the cleaned demo) are dropped,
    and the zip's observation.state is ignored (it is frozen).
  * One episode per case at 50 Hz: every HDF5 row is one control step (20 ms). The zip is labelled
    30 fps, which is wrong; convert_case checks the step length against the joint velocities.

observation.state (34 values; only what the policy also receives at evaluation time):
    0:7    left_arm_q       7 absolute joint angles of the left arm (joint_pos[LEFT_ARM_JOINTS])
    7:14   right_arm_q
   14:16   gripper          left, right gripper joint: 0 open ... 1.703 closed (joint_pos[GRIPPER_JOINTS])
   16:25   left_eef_world   obs/eef_pos[left] (world frame, xyz) + rot6d of obs/eef_quat[left]
   25:34   right_eef_world
  The arm angles say where the hands are relative to the robot. The two *_eef_world groups say
  where they are in the room, which is the only hint of where the robot stands; they are separate
  groups so that a model can be trained without them.
action (23):
    0:9    left_eef      target xyz + rot6d of the target quaternion (robot root frame, absolute)
    9:18   right_eef
   18:20   gripper       left, right: 0 open / 1 closed
   20:23   base_vel      vx, vy, wz in the body frame (m/s, m/s, rad/s), from the recorded wheel values
The same layout as src/astra/types.py; --self-test compares the two when that package is present.
rot6d = first two rows of the rotation matrix (GR00T's convention); no quaternion sign flips.
galbot_action_codec.py (written next to the dataset) turns model output back into the env's
20-value (or 19-value) action and builds the state from evaluation observations. It is generated from the
functions of this file, so training and evaluation use the same code.

Also written to OUTPUT_DIR: meta/modality.json, galbot_config.py (GR00T modality config),
galbot_action_codec.py, meta/conversion.json (case -> episode, checks) and
OUTPUT_DIR.work/report.md. GR00T computes meta/stats.json itself on the first training run.
meta/episodes_stats.jsonl (required by LeRobot v2.1 loaders and the v2.1 -> v3.0 converter) is
written at merge time: numeric columns over all frames, cameras per RGB channel on keyframes
(cached per case in WORK_DIR, so a rerun of --merge-only is fast).
"""
from __future__ import annotations

import argparse
import inspect
import io
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import traceback
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

FPS = 50   # one HDF5 row per control step: physics 200 Hz, decimation 4
CASE_RE = re.compile(r"ChamberServiceTask_\d+")
DEFAULT_TASK = ("Lift the chamber top, extract the large part, pack it into the box, wipe the chamber, "
                "unbox and place the replacement, then close and lock the chamber.")
CAMERAS = {  # LeRobot video key -> (HDF5 obs key, GR00T video key)
    "observation.images.first_person_camera_rgb": ("obs/first_person_camera_rgb", "head"),
    "observation.images.left_hand_camera_rgb": ("obs/left_hand_camera_rgb", "left_wrist"),
    "observation.images.right_hand_camera_rgb": ("obs/right_hand_camera_rgb", "right_wrist"),
}
N_JOINTS = 77
# joint_pos indices, measured in the sim. Arms: one arm moved at a time; joints 1-4 also matched to
# the default pose by name. Grippers: one closed at a time; left moves 65, 66, 67, 74 (to +1.703)
# and 68, 73, right 69, 70, 71, 76 and 72, 75.
LEFT_ARM_JOINTS = [50, 53, 55, 57, 59, 61, 63]    # left_arm_joint1..7
RIGHT_ARM_JOINTS = [51, 54, 56, 58, 60, 62, 64]   # right_arm_joint1..7
GRIPPER_JOINTS = {"left": 65, "right": 69}
STATE_GROUPS = {"left_arm_q": (0, 7), "right_arm_q": (7, 14), "gripper": (14, 16),
                "left_eef_world": (16, 25), "right_eef_world": (25, 34)}
STATE_DIM = 34
ACTION_GROUPS = {"left_eef": (0, 9), "right_eef": (9, 18), "gripper": (18, 20), "base_vel": (20, 23)}
ACTION_DIM = 23
# The env's own mapping between the four wheel values and the base velocity [vx, vy, wz].
WHEELS_FROM_BASE_VEL = [[1, -1, -1], [1, 1, 1], [1, 1, -1], [1, -1, 1]]
BASE_VEL_FROM_WHEELS = [[0.25, 0.25, 0.25, 0.25], [-0.25, 0.25, 0.25, -0.25], [-0.25, 0.25, -0.25, 0.25]]
RAW_GRIPPER_COLS = {"left": 14, "right": 15}   # in the recorded 20-value action
ACTION_QUAT_ORDER = "xyzw"   # galbot.py preprocess_device_action: arm_action[3:7] is quat_xyzw
STATE_QUAT_ORDER = "xyzw"    # FrameTransformer target_quat_w in the pinned Isaac Lab
STATE_TOL = 1e-4


# =========================================================================== math
def quat_to_mat(q: np.ndarray, order: str) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if order == "xyzw":
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    else:
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    n = np.where(n < 1e-12, 1.0, n)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def mat_to_rot6d(R: np.ndarray) -> np.ndarray:
    """GR00T convention (pose.py _matrix_to_rot6d): first two rows, flattened."""
    return R[..., :2, :].reshape(*R.shape[:-2], 6)


def pose_to_xyz_rot6d(pos: np.ndarray, quat: np.ndarray, order: str) -> np.ndarray:
    return np.concatenate([pos, mat_to_rot6d(quat_to_mat(quat, order))], axis=-1)


def rot6d_to_mat(r6: np.ndarray) -> np.ndarray:
    r6 = np.asarray(r6, dtype=np.float64).reshape(-1, 2, 3)
    a = r6[:, 0] / np.linalg.norm(r6[:, 0], axis=-1, keepdims=True)
    b = r6[:, 1] - (a * r6[:, 1]).sum(-1, keepdims=True) * a
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack([a, b, np.cross(a, b)], axis=1)


def mat_to_quat(R: np.ndarray, order: str) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = np.empty((len(R), 4))  # w, x, y, z
    for i, (m, t) in enumerate(zip(R, tr)):
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            q[i] = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q[i] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q[i] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q[i] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    return q[:, [1, 2, 3, 0]] if order == "xyzw" else q


def build_state(eef_pos, eef_quat, joint_pos) -> np.ndarray:
    """Observations of T steps (or of one step) -> (T, 34) state rows, as stored in the dataset.
    joint_pos holds the 77 ABSOLUTE joint positions."""
    pos = np.asarray(eef_pos, dtype=np.float64).reshape(-1, 2, 3)
    quat = np.asarray(eef_quat, dtype=np.float64).reshape(-1, 2, 4)
    q = np.asarray(joint_pos, dtype=np.float64).reshape(len(pos), -1)
    eef = [pose_to_xyz_rot6d(pos[:, i], quat[:, i], STATE_QUAT_ORDER) for i in range(2)]
    grip = q[:, [GRIPPER_JOINTS["left"], GRIPPER_JOINTS["right"]]]
    return np.concatenate([q[:, LEFT_ARM_JOINTS], q[:, RIGHT_ARM_JOINTS], grip] + eef, axis=1).astype(np.float32)


def build_action(a: np.ndarray) -> np.ndarray:
    """Recorded env actions (T, 20) -> dataset actions (T, 23)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 20)
    left = pose_to_xyz_rot6d(a[:, 0:3], a[:, 3:7], ACTION_QUAT_ORDER)
    right = pose_to_xyz_rot6d(a[:, 7:10], a[:, 10:14], ACTION_QUAT_ORDER)
    grip = (a[:, 14:16] < 0).astype(np.float64)   # the env closes a gripper below zero
    base = a[:, 16:20] @ np.asarray(BASE_VEL_FROM_WHEELS).T
    return np.concatenate([left, right, grip, base], axis=1).astype(np.float32)


def action_to_env(action, wheels: bool = True) -> np.ndarray:
    """Model output -> env actions: [left xyz quat | right xyz quat | lgrip | rgrip | base].
    wheels=True gives the 20-value form (four wheel values, which the env applies as is; the form of
    the recordings). wheels=False gives the 19-value form ([vx, vy, wz], which the env smooths).
    `action` is a dict of groups (each (T, d)) or a (T, 23) array in dataset order."""
    if isinstance(action, dict):
        action = np.concatenate([np.asarray(action[k], dtype=np.float64).reshape(len(np.asarray(action[k])), -1)
                                 for k in ("left_eef", "right_eef", "gripper", "base_vel")], 1)
    a = np.asarray(action, dtype=np.float64).reshape(-1, 23)
    out = []
    for p in (0, 9):
        out += [a[:, p:p + 3], mat_to_quat(rot6d_to_mat(a[:, p + 3:p + 9]), ACTION_QUAT_ORDER)]
    grip = np.where(a[:, 18:20] > 0.5, -1.0, 1.0)   # 1 = closed in the dataset; the env closes below zero
    base = a[:, 20:23] @ np.asarray(WHEELS_FROM_BASE_VEL).T if wheels else a[:, 20:23]
    return np.concatenate(out + [grip, base], 1).astype(np.float32)


# =========================================================================== reading a case
def find_cases(root: Path, batches: list[str] | None) -> list[tuple[str, str, Path]]:
    """[(batch, case_id, path)] for archives (.tar/.tar.gz/.tgz) or folders named ChamberServiceTask_<id>."""
    out = []
    for b in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if batches and b.name not in batches:
            continue
        seen = set()
        for p in sorted(b.iterdir()):
            m = CASE_RE.search(p.name)
            if not m or m.group(0) in seen:
                continue
            if p.is_dir() or p.name.endswith((".tar", ".tar.gz", ".tgz")):
                seen.add(m.group(0))
                out.append((b.name, m.group(0), p))
    return out


def read_case(path: Path, tmp: Path) -> dict:
    """One streaming pass: HDF5 -> temp file, galbot.zip -> bytes, running_args.json -> dict."""
    out = {"hdf5": None, "zip": None, "args": None}

    def take(name, opener):
        base = name.rsplit("/", 1)[-1]
        if base.endswith((".hdf5", ".h5")) and out["hdf5"] is None:
            dest = tmp / base
            with opener() as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst, 16 * 2**20)
            out["hdf5"] = dest
        elif base.endswith(".zip") and out["zip"] is None:
            with opener() as src:
                out["zip"] = src.read()
        elif base == "running_args.json":
            with opener() as src:
                out["args"] = json.loads(src.read())

    if path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.is_file():
                if p.suffix in (".hdf5", ".h5"):   # no copy needed for folders
                    out["hdf5"] = out["hdf5"] or p
                else:
                    take(p.name, lambda p=p: open(p, "rb"))
    else:
        with tarfile.open(path, "r|*") as tf:
            for m in tf:
                if m.isfile():
                    take(m.name, lambda m=m: tf.extractfile(m))
    return out


def load_hdf5(path: Path) -> dict:
    """Low-dimensional arrays of the demo with actions (the cleaned files hold one)."""
    import h5py

    with h5py.File(path, "r") as f:
        data = f["data"]
        demos = [k for k in data if "actions" in data[k]]
        if not demos:
            raise ValueError("no demo with actions")
        name = max(demos, key=lambda k: data[k]["actions"].shape[0])
        g = data[name]
        need = {"actions": "actions", "eef_pos": "obs/eef_pos", "eef_quat": "obs/eef_quat",
                "joint_pos": "states/articulation/robot/joint_position",
                "joint_vel": "states/articulation/robot/joint_velocity"}
        out = {"demo": name, "n_demos": len(demos)}
        for k, key in need.items():
            if key not in g:
                raise KeyError(f"{name}: missing {key}")
            out[k] = g[key][()].astype(np.float64)
        out["cameras"] = {lk: hk for lk, (hk, _) in CAMERAS.items() if hk in g}
        out["env_args"] = data.attrs.get("env_args")
    T = len(out["actions"])
    for k in ("eef_pos", "eef_quat", "joint_pos", "joint_vel"):
        if len(out[k]) != T:
            raise ValueError(f"{k} has {len(out[k])} rows, actions {T}")
    if out["joint_pos"].shape[1] != N_JOINTS or out["actions"].shape[1] != 20:
        raise ValueError(f"unexpected layout: joints {out['joint_pos'].shape[1]}, action {out['actions'].shape[1]}")
    return out


# --------------------------------------------------------------------------- batch2 zip
def read_zip(data: bytes) -> dict:
    """LeRobot v3/v2.1 dataset in a zip: actions in row order and the mp4 members per camera."""
    import pyarrow.parquet as pq

    z = zipfile.ZipFile(io.BytesIO(data))
    names = z.namelist()
    info_name = next(n for n in names if n.endswith("meta/info.json"))
    root = info_name[: -len("meta/info.json")]
    parts = []
    for n in sorted(x for x in names if x.startswith(root + "data/") and x.endswith(".parquet")):
        t = pq.read_table(io.BytesIO(z.read(n)), columns=["episode_index", "frame_index", "action"])
        col = t.column("action").combine_chunks()
        act = col.flatten().to_numpy(zero_copy_only=False).reshape(len(col), -1)
        parts.append((t.column("episode_index").to_numpy(), t.column("frame_index").to_numpy(), act))
    ep = np.concatenate([p[0] for p in parts])
    fr = np.concatenate([p[1] for p in parts])
    act = np.concatenate([p[2] for p in parts]).astype(np.float64)
    order = np.lexsort((fr, ep))          # rows in (episode, frame) order = video order
    videos = defaultdict(list)
    for n in sorted(names):
        if n.endswith(".mp4") and "videos/" in n:
            videos[n.split("videos/", 1)[1].split("/", 1)[0]].append(n)
    return {"zip": z, "actions": act[order], "episodes": int(len(np.unique(ep))), "videos": dict(videos)}


def align(lr: np.ndarray, h5: np.ndarray, tol: float = STATE_TOL) -> np.ndarray:
    """For each LeRobot row the HDF5 step with the same action (-1 if none); ties resolved
    towards the previous match + 1, then a backward pass."""
    A, B = lr.astype(np.float64), h5.astype(np.float64)
    b2 = (B * B).sum(1)
    match = np.full(len(A), -1, dtype=np.int64)
    ambiguous: dict[int, set] = {}
    prev = -1
    for i0 in range(0, len(A), 512):
        a = A[i0:i0 + 512]
        d2 = np.maximum((a * a).sum(1)[:, None] + b2[None, :] - 2.0 * a @ B.T, 0.0)
        for k, row in enumerate(d2):
            cands = np.flatnonzero(row <= tol * tol)
            if cands.size:
                j = int(cands[np.argmin(np.abs(cands - (prev + 1)))])
                match[i0 + k], prev = j, j
                if cands.size > 1:
                    ambiguous[i0 + k] = set(cands.tolist())
    for i in range(len(A) - 2, -1, -1):
        if i in ambiguous and match[i + 1] >= 0 and (match[i + 1] - 1) in ambiguous[i]:
            match[i] = match[i + 1] - 1
    return match


# =========================================================================== video
class Encoder:
    def __init__(self, path: Path, width: int, height: int, crf: int, preset: str, gop: int):
        import av

        self.c = av.open(str(path), "w")
        codec = "libx264" if "libx264" in av.codecs_available else "h264"
        self.s = self.c.add_stream(codec, rate=FPS)
        self.s.width, self.s.height, self.s.pix_fmt = width, height, "yuv420p"
        self.s.codec_context.gop_size = gop
        self.s.options = {"crf": str(crf), "preset": preset}
        self.n = 0

    def add(self, rgb: np.ndarray) -> None:
        import av

        f = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        for p in self.s.encode(f):
            self.c.mux(p)
        self.n += 1

    def close(self) -> int:
        for p in self.s.encode():
            self.c.mux(p)
        self.c.close()
        return self.n


class FrameStats:
    """Cheap per-frame features: blank frames, repeats of the previous frame, motion signal."""

    def __init__(self):
        self.prev = None
        self.motion, self.black, self.repeat = [], 0, 0

    def add(self, rgb: np.ndarray) -> None:
        k = max(1, min(rgb.shape[0], rgb.shape[1]) // 32)
        th = rgb[::k, ::k].astype(np.float32).mean(-1)
        if th.mean() < 3:
            self.black += 1
        if self.prev is not None:
            d = float(np.abs(th - self.prev).mean())
            self.motion.append(d)
            self.repeat += d < 0.25
        else:
            self.motion.append(np.nan)
        self.prev = th

    def summary(self, n: int) -> dict:
        return {"frames": n, "black": self.black, "repeated": self.repeat}


def encode_from_hdf5(path: Path, cam_key: str, out: Path, opts) -> tuple[int, dict, np.ndarray]:
    """Encode every frame of one camera of the demo, in step order."""
    import h5py

    st = FrameStats()
    with h5py.File(path, "r") as f:
        g = f["data"]
        demo = max((k for k in g if "actions" in g[k]), key=lambda k: g[k]["actions"].shape[0])
        ds = g[demo][cam_key]
        _, h, w, _ = ds.shape
        enc = Encoder(out, w, h, opts.crf, opts.preset, opts.gop)
        for b0 in range(0, len(ds), 128):
            for fr in ds[b0:b0 + 128]:
                enc.add(fr)
                st.add(fr)
        n = enc.close()
    return n, st.summary(n), np.array(st.motion)


def encode_from_zip(z: zipfile.ZipFile, members: list[str], rows: np.ndarray, out: Path, tmp: Path, opts
                    ) -> tuple[int, int, dict, np.ndarray]:
    """Decode the zip video and keep frames at `rows` (sorted). Returns (kept, total decoded, stats, motion)."""
    import av

    st = FrameStats()
    enc, total = None, 0
    want = set(rows.tolist())
    for m in members:
        src = tmp / Path(m).name
        src.write_bytes(z.read(m))
        try:
            with av.open(str(src)) as c:
                vs = c.streams.video[0]
                vs.thread_type = "AUTO"
                for fr in c.decode(vs):
                    if total in want:
                        rgb = fr.to_ndarray(format="rgb24")
                        if enc is None:
                            enc = Encoder(out, rgb.shape[1], rgb.shape[0], opts.crf, opts.preset, opts.gop)
                        enc.add(rgb)
                        st.add(rgb)
                    total += 1
        finally:
            src.unlink(missing_ok=True)
    n = enc.close() if enc else 0
    return n, total, st.summary(n), np.array(st.motion)


# =========================================================================== checks
def gripper_check(joint_pos: np.ndarray, raw_actions: np.ndarray, max_lag: int = 20) -> dict:
    """Per side: how often the gripper command toggles, and how well the joint in GRIPPER_JOINTS
    follows it (best |correlation| over a delay of 0..max_lag steps). A low value where the gripper
    does toggle means the joint order of this recording differs from the one the indices were
    measured on."""
    out = {}
    for side, col in RAW_GRIPPER_COLS.items():
        closed = (raw_actions[:, col] < 0).astype(np.float64)
        toggles = int(np.abs(np.diff(closed)).sum())
        q = joint_pos[:, GRIPPER_JOINTS[side]]
        corr = 0.0
        if toggles and q.std() > 1e-9:
            corr = max(abs(float(np.corrcoef(closed[: len(closed) - lag], q[lag:])[0, 1])) for lag in range(max_lag + 1))
        out[side] = {"toggles": toggles, "corr": corr}
    return out


def step_seconds(joint_pos: np.ndarray, joint_vel: np.ndarray) -> float | None:
    """Length of one recorded step, from the data itself: a joint moves by velocity x step."""
    dq = np.diff(joint_pos, axis=0)
    v = 0.5 * (joint_vel[1:] + joint_vel[:-1])
    moving = np.abs(v) > 0.2
    dt = dq[moving] / v[moving]
    dt = dt[(dt > 0.002) & (dt < 0.2)]
    return float(np.median(dt)) if dt.size > 500 else None


def sync_lag(motion: np.ndarray, signal: np.ndarray, max_lag: int = 15) -> tuple[int | None, float]:
    """Lag (frames) at which wrist-image motion best follows gripper joint motion. 0 = in sync."""
    n = min(len(motion), len(signal))
    m, s = motion[:n], signal[:n]
    best, best_c = None, -1.0
    for lag in range(-max_lag, max_lag + 1):
        a = m[max(0, lag): n + min(0, lag)]
        b = s[max(0, -lag): n - max(0, lag)]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 50 and a[ok].std() > 0 and b[ok].std() > 0:
            c = float(np.corrcoef(a[ok], b[ok])[0, 1])
            if c > best_c:
                best, best_c = lag, c
    return best, best_c


# =========================================================================== one case
def convert_case(batch: str, cid: str, path: str, work: str, opts_d: dict) -> dict:
    opts = argparse.Namespace(**opts_d)
    path, work = Path(path), Path(work)
    final = work / "cases" / batch / cid
    stage = work / "cases" / batch / f".{cid}.tmp"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"{cid}_", dir=opts.tmp or str(work / "tmp")))
    t0 = time.time()
    rec = {"batch": batch, "case": cid, "source": str(path), "source_size": _size(path), "status": "failed",
           "notes": [], "converter_version": VERSION}
    try:
        src = read_case(path, tmp)
        if src["hdf5"] is None:
            raise ValueError("no HDF5 in case")
        h = load_hdf5(src["hdf5"])
        rec["instance_id"] = (src["args"] or {}).get("instance_id")
        rec["hdf5_steps"] = T = len(h["actions"])
        rec["hdf5_demos_with_actions"] = h["n_demos"]

        if src["zip"] is not None and len(h["cameras"]) < len(CAMERAS):
            # batch2: video from the zip, rows matched to HDF5 steps by action
            zd = read_zip(src["zip"])
            match = align(zd["actions"], h["actions"])
            rows = np.flatnonzero(match >= 0)
            steps = match[rows]
            rec.update(video_source="galbot.zip", zip_rows=len(zd["actions"]), zip_episodes=zd["episodes"],
                       rows_dropped=int(len(zd["actions"]) - len(rows)))
            back = int((np.diff(steps) < 0).sum())
            if back:
                rec["notes"].append(f"{back} backward steps in the matched sequence")
            if len(rows) < 0.9 * T:
                raise ValueError(f"only {len(rows)} of {T} HDF5 steps found in galbot.zip")
            if not np.array_equal(zd["actions"][rows].astype(np.float32), h["actions"][steps].astype(np.float32)):
                raise ValueError("matched zip actions differ from HDF5 actions")
            missing = set(CAMERAS) - set(zd["videos"])
            if missing:
                raise ValueError(f"galbot.zip lacks videos {sorted(missing)}")
        elif len(h["cameras"]) == len(CAMERAS):
            rows = steps = np.arange(T)
            zd = None
            rec["video_source"] = "hdf5"
        else:
            raise ValueError(f"no complete camera set: HDF5 has {sorted(h['cameras'])}, zip: {src['zip'] is not None}")

        dt = step_seconds(h["joint_pos"], h["joint_vel"])
        rec["step_seconds"] = dt
        if dt is not None and abs(dt * FPS - 1.0) > 0.15:
            raise ValueError(f"one HDF5 step lasts {dt * 1000:.1f} ms, but the dataset is written at {FPS} fps")

        state = build_state(h["eef_pos"][steps], h["eef_quat"][steps], h["joint_pos"][steps])
        action = build_action(h["actions"][steps])
        raw = h["actions"][steps]
        if not (np.isfinite(state).all() and np.isfinite(action).all()):
            raise ValueError("non-finite values in state or action")
        n = len(steps)
        rec.update(frames=n, hdf5_steps_used=n, first_step=int(steps[0]), last_step=int(steps[-1]))

        # videos
        rec["video"] = {}
        motions = {}
        (stage / "videos").mkdir()
        for lk, (hk, gk) in CAMERAS.items():
            out = stage / "videos" / f"{lk}.mp4"
            if zd is None:
                got, stats, motion = encode_from_hdf5(src["hdf5"], hk, out, opts)
            else:
                got, total, stats, motion = encode_from_zip(zd["zip"], zd["videos"][lk], rows, out, tmp, opts)
                if total != len(zd["actions"]):
                    raise ValueError(f"{lk}: {total} video frames but {len(zd['actions'])} zip rows")
            if got != n:
                raise ValueError(f"{lk}: wrote {got} frames, expected {n}")
            if stats["black"] > 0.05 * n or stats["repeated"] > 0.98 * n:
                raise ValueError(f"{lk}: bad video ({stats})")
            rec["video"][lk] = stats
            motions[gk] = motion

        # gripper joints and video sync
        rec["gripper"] = gripper_check(h["joint_pos"][steps], raw)
        rec["sync"] = {}
        for side, cam in (("left", "left_wrist"), ("right", "right_wrist")):
            if rec["gripper"][side]["toggles"]:
                q = h["joint_pos"][steps, GRIPPER_JOINTS[side]]
                lag, corr = sync_lag(motions[cam], np.concatenate([[np.nan], np.abs(np.diff(q))]))
                rec["sync"][side] = {"lag": lag, "corr": corr, "toggles": rec["gripper"][side]["toggles"]}

        # per-column stats for the report
        rec["col_stats"] = {
            "state_min": state.min(0).tolist(), "state_max": state.max(0).tolist(),
            "action_min": action.min(0).tolist(), "action_max": action.max(0).tolist(),
            "raw_gripper_closed_share": {s: float((raw[:, c] < 0).mean()) for s, c in RAW_GRIPPER_COLS.items()},
        }

        _write_case_parquet(stage / "data.parquet", state, action)
        np.save(stage / "source_steps.npy", steps.astype(np.int64))
        rec["status"] = "ok"
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc(limit=4)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    rec["seconds"] = round(time.time() - t0, 1)
    (stage / "case.json").write_text(json.dumps(rec, indent=1))
    shutil.rmtree(final, ignore_errors=True)
    stage.rename(final)
    if rec["status"] != "ok":
        for p in ("data.parquet", "videos"):
            q = final / p
            shutil.rmtree(q, ignore_errors=True) if q.is_dir() else q.unlink(missing_ok=True)
    return {k: rec.get(k) for k in ("batch", "case", "status", "frames", "seconds", "error")}


def _size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size


def _write_case_parquet(path: Path, state: np.ndarray, action: np.ndarray) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = len(state)
    t = pa.table({
        "observation.state": pa.FixedSizeListArray.from_arrays(pa.array(state.reshape(-1), pa.float32()), STATE_DIM),
        "action": pa.FixedSizeListArray.from_arrays(pa.array(action.reshape(-1), pa.float32()), ACTION_DIM),
        "timestamp": pa.array((np.arange(n) / FPS).astype(np.float32)),
        "frame_index": pa.array(np.arange(n, dtype=np.int64)),
    })
    pq.write_table(t, path)


# =========================================================================== merge
def merge(work: Path, out: Path, opts) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    recs = []
    for cj in sorted((work / "cases").glob("*/*/case.json")):
        r = json.loads(cj.read_text())
        r["_dir"] = cj.parent
        recs.append(r)
    if opts.batches:
        recs = [r for r in recs if r["batch"] in opts.batches]
    ok = [r for r in recs if r["status"] == "ok"]
    ok.sort(key=lambda r: (r["batch"], r["case"]))
    if not ok:
        raise SystemExit("No converted cases to merge.")

    # clear previous output (only what this script writes)
    for d in ("data", "videos"):
        shutil.rmtree(out / d, ignore_errors=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    for f in ("info.json", "episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl", "modality.json",
              "conversion.json", "stats.json", "relative_stats.json"):
        (out / "meta" / f).unlink(missing_ok=True)

    chunks = 1000
    episodes, mapping, ep_stats = [], [], []
    index = 0
    for ep, r in enumerate(ok):
        chunk = ep // chunks
        t = pq.read_table(r["_dir"] / "data.parquet")
        n = t.num_rows
        t = t.append_column("episode_index", pa.array(np.full(n, ep, np.int64)))
        t = t.append_column("index", pa.array(np.arange(index, index + n, dtype=np.int64)))
        t = t.append_column("task_index", pa.array(np.zeros(n, np.int64)))
        t = t.append_column("annotation.human.task_description", pa.array(np.zeros(n, np.int64)))
        dp = out / f"data/chunk-{chunk:03d}/episode_{ep:06d}.parquet"
        dp.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, dp)
        for lk in CAMERAS:
            vp = out / f"videos/chunk-{chunk:03d}/{lk}/episode_{ep:06d}.mp4"
            vp.parent.mkdir(parents=True, exist_ok=True)
            src = r["_dir"] / "videos" / f"{lk}.mp4"
            try:
                os.link(src, vp)
            except OSError:
                shutil.copy2(src, vp)
        episodes.append({"episode_index": ep, "tasks": [opts.task], "length": n})
        ep_stats.append({"episode_index": ep, "stats": {**table_stats(t), **case_video_stats(r["_dir"])}})
        print(f"\rmerge: episode {ep + 1}/{len(ok)}", end="", flush=True)
        mapping.append({"episode_index": ep, "batch": r["batch"], "case": r["case"], "frames": n,
                        "video_source": r["video_source"], "first_hdf5_step": r["first_step"],
                        "last_hdf5_step": r["last_step"], "hdf5_steps": r["hdf5_steps"]})
        index += n

    # gripper joints: how well the fixed joint indices follow the command, over the cases where it toggles
    grip = {}
    for i, side in enumerate(GRIPPER_JOINTS):
        corrs = [r["gripper"][side]["corr"] for r in ok if r["gripper"][side]["toggles"]]
        grip[side] = {"state_col": STATE_GROUPS["gripper"][0] + i, "cases_toggling": len(corrs),
                      "mean_corr": float(np.mean(corrs)) if corrs else None}

    state_groups = {k: {"start": a, "end": b} for k, (a, b) in STATE_GROUPS.items()}
    modality = {
        "state": state_groups,
        "action": {k: {"start": a, "end": b} for k, (a, b) in ACTION_GROUPS.items()},
        "video": {gk: {"original_key": lk} for lk, (_, gk) in CAMERAS.items()},
        "annotation": {"human.task_description": {}},
    }
    # video shapes from the first episode
    import av
    shapes = {}
    for lk in CAMERAS:
        with av.open(str(out / f"videos/chunk-000/{lk}/episode_000000.mp4")) as c:
            s = c.streams.video[0]
            shapes[lk] = (s.codec_context.height, s.codec_context.width)

    total = index
    info = {
        "codebase_version": "v2.1", "robot_type": "Galbot-Mobile",
        "total_episodes": len(ok), "total_frames": total, "total_tasks": 1,
        "total_videos": len(ok) * len(CAMERAS), "total_chunks": (len(ok) - 1) // chunks + 1,
        "chunks_size": chunks, "fps": FPS, "splits": {"train": f"0:{len(ok)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [STATE_DIM], "names": state_names()},
            "action": {"dtype": "float32", "shape": [ACTION_DIM], "names": action_names()},
            **{lk: {"dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "channels"],
                    "info": {"video.height": h, "video.width": w, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                             "video.is_depth_map": False, "video.fps": FPS, "video.channels": 3, "has_audio": False}}
               for lk, (h, w) in shapes.items()},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "annotation.human.task_description": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    meta = out / "meta"
    (meta / "info.json").write_text(json.dumps(info, indent=4))
    (meta / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    (meta / "episodes_stats.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ep_stats))
    print()
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": opts.task}) + "\n")
    (meta / "modality.json").write_text(json.dumps(modality, indent=4))

    # constant columns over the whole dataset
    def span(key_min, key_max):
        lo = np.min([r["col_stats"][key_min] for r in ok], axis=0)
        hi = np.max([r["col_stats"][key_max] for r in ok], axis=0)
        return lo, hi
    a_lo, a_hi = span("action_min", "action_max")
    s_lo, s_hi = span("state_min", "state_max")
    const_action = [action_names()[i] for i in np.flatnonzero(a_hi - a_lo < 1e-6)]
    const_state = int((s_hi - s_lo < 1e-6).sum())

    summary = {
        "converter_version": VERSION, "episodes": len(ok), "frames": total, "hours": round(total / FPS / 3600, 2),
        "per_batch": {b: {"episodes": c, "frames": int(sum(m["frames"] for m in mapping if m["batch"] == b))}
                      for b, c in Counter(m["batch"] for m in mapping).items()},
        "failed_cases": [{"batch": r["batch"], "case": r["case"], "error": r.get("error")} for r in recs if r["status"] != "ok"],
        "gripper_columns": grip, "constant_action_columns": const_action, "constant_state_columns": const_state,
        "state_groups": state_groups, "action_groups": modality["action"], "task": opts.task,
        "quaternion_orders": {"action": ACTION_QUAT_ORDER, "state_eef": STATE_QUAT_ORDER},
        "episodes_map": mapping,
    }
    (meta / "conversion.json").write_text(json.dumps(summary, indent=1))
    write_config(out)
    write_codec(out, modality)
    write_report(work, summary, ok, recs)
    return summary


# --------------------------------------------------------------------------- episodes_stats.jsonl
def _stats(x: np.ndarray, axis, keepdims: bool = False) -> dict:
    x = x.astype(np.float64)
    return {"min": x.min(axis=axis, keepdims=keepdims), "max": x.max(axis=axis, keepdims=keepdims),
            "mean": x.mean(axis=axis, keepdims=keepdims), "std": x.std(axis=axis, keepdims=keepdims)}


def table_stats(t) -> dict:
    """LeRobot v2.1 stats for every numeric column of one episode table (per dimension)."""
    import pyarrow as pa

    out = {}
    for name in t.column_names:
        col = t.column(name).combine_chunks()
        if pa.types.is_fixed_size_list(col.type):
            arr = col.flatten().to_numpy(zero_copy_only=False).reshape(len(col), col.type.list_size)
        else:
            arr = col.to_numpy(zero_copy_only=False).reshape(-1, 1)
        if not np.issubdtype(arr.dtype, np.number):
            continue
        s = _stats(arr, axis=0)
        out[name] = {k: v.tolist() for k, v in s.items()} | {"count": [int(len(arr))]}
    return out


def _video_channel_stats(path: Path, max_samples: int = 100) -> dict:
    """Per-RGB-channel stats in [0, 1], shape (3, 1, 1), on keyframes only (fast: no full decode).
    Frames are downscaled to ~150 px like LeRobot's own stats sampling."""
    import av

    frames = []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.codec_context.skip_frame = "NONKEY"
        for fr in c.decode(s):
            frames.append(fr)
    if not frames:
        raise ValueError(f"no keyframes decoded from {path}")
    pick = np.linspace(0, len(frames) - 1, min(max_samples, len(frames))).round().astype(int)
    imgs = []
    for i in np.unique(pick):
        rgb = frames[i].to_ndarray(format="rgb24")
        k = max(1, min(rgb.shape[:2]) // 150)
        imgs.append(rgb[::k, ::k])
    x = np.stack(imgs).astype(np.float64).transpose(0, 3, 1, 2) / 255.0   # (N, 3, H, W)
    s = _stats(x, axis=(0, 2, 3), keepdims=True)
    return {k: v[0].tolist() for k, v in s.items()} | {"count": [int(len(imgs))]}


def case_video_stats(case_dir: Path) -> dict:
    """Camera stats of one converted case, cached in the case folder (rebuilt when a case is redone)."""
    cache = case_dir / "video_stats.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        if set(d) == set(CAMERAS):
            return d
    d = {lk: _video_channel_stats(case_dir / "videos" / f"{lk}.mp4") for lk in CAMERAS}
    cache.write_text(json.dumps(d))
    return d


def state_names() -> list[str]:
    n = [f"{side}_arm_q.{i}" for side in ("left", "right") for i in range(1, 8)] + ["gripper.left", "gripper.right"]
    for side in ("left", "right"):
        n += [f"{side}_eef_world.{a}" for a in ("x", "y", "z")] + [f"{side}_eef_world.rot6d_{i}" for i in range(6)]
    return n


def action_names() -> list[str]:
    n = []
    for side in ("left", "right"):
        n += [f"{side}_eef.{a}" for a in ("x", "y", "z")] + [f"{side}_eef.rot6d_{i}" for i in range(6)]
    return n + ["gripper.left", "gripper.right", "base_vel.vx", "base_vel.vy", "base_vel.wz"]


def write_config(out: Path) -> None:
    (out / "galbot_config.py").write_text(f'''"""GR00T modality config for the converted ChamberServiceTask dataset (convert_dataset.py).

Pass it to fine-tuning with --modality-config-path. Keys match meta/modality.json.
left_eef_world / right_eef_world tell the model where the hands are in the room. Remove them from
STATE_KEYS (no need to convert again) to train a model that does not rely on a fixed scene.
"""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (ActionConfig, ActionFormat, ActionRepresentation, ActionType,
                              ModalityConfig)

STATE_KEYS = {list(STATE_GROUPS)!r}
ACTION_KEYS = {list(ACTION_GROUPS)!r}
ACTION_HORIZON = 40  # GR00T N1.7 default action horizon

# Absolute actions everywhere: the arm targets are IK poses in the robot root frame, and the state
# holds no pose in that frame for them to be relative to. Rotations are xyz+rot6d
# stored as plain numbers (NON_EEF/DEFAULT): no conversion inside GR00T.
galbot_config = {{
    "video": ModalityConfig(delta_indices=[0], modality_keys={[gk for _, gk in CAMERAS.values()]!r}),
    "state": ModalityConfig(delta_indices=[0], modality_keys=STATE_KEYS),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=ACTION_KEYS,
        action_configs=[ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF,
                                     format=ActionFormat.DEFAULT) for _ in ACTION_KEYS],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}}
register_modality_config(galbot_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
''')


def write_codec(out: Path, modality: dict) -> None:
    """The codec is the source of the functions above, so the dataset and the evaluation share it."""
    shared = (quat_to_mat, mat_to_rot6d, pose_to_xyz_rot6d, rot6d_to_mat, mat_to_quat, build_state, action_to_env)
    (out / "galbot_action_codec.py").write_text(f'''"""Map between the Galbot env and the converted dataset (written by convert_dataset.py).

At evaluation:
    state  = build_state(eef_pos, eef_quat, joint_pos)[0]   # (34,)
    groups = split_state(state)                      # {{"left_eef": (9,), ...}} per meta/modality.json
    ...model predicts action groups...
    env_action = action_to_env(pred)                 # (T, 20) -> task_env.step(env_action[t]);
                                                     # wheels=False for an env that takes 19 values

Inputs come from the policy observation (policy + embodiment_general_obs groups):
    eef_pos (2,3) and eef_quat (2,4) from ee_frame (world frame),
    joint_pos = ABSOLUTE joint positions (embodiment_general_obs "joint_pos", not joint_pos_rel).
"""
import numpy as np

MODALITY = {json.dumps(modality)}
ACTION_QUAT_ORDER = "{ACTION_QUAT_ORDER}"   # quaternion order of the env's arm action
STATE_QUAT_ORDER = "{STATE_QUAT_ORDER}"     # order eef_quat is read in
LEFT_ARM_JOINTS = {LEFT_ARM_JOINTS}
RIGHT_ARM_JOINTS = {RIGHT_ARM_JOINTS}
GRIPPER_JOINTS = {GRIPPER_JOINTS}
WHEELS_FROM_BASE_VEL = {WHEELS_FROM_BASE_VEL}
BASE_VEL_FROM_WHEELS = {BASE_VEL_FROM_WHEELS}


{chr(10).join(inspect.getsource(f) + chr(10) for f in shared)}
def split_state(state):
    """State vector -> dict of groups, as in meta/modality.json."""
    return {{k: state[v["start"]:v["end"]] for k, v in MODALITY["state"].items()}}
''')


def write_report(work: Path, s: dict, ok: list, recs: list) -> None:
    L = ["# LeRobot v2.1 conversion report", "",
         f"Episodes: **{s['episodes']}**, frames {s['frames']} ({s['hours']} h at {FPS} fps)", ""]
    for b, v in sorted(s["per_batch"].items()):
        L.append(f"- {b}: {v['episodes']} episodes, {v['frames'] / FPS / 3600:.2f} h")
    if s["failed_cases"]:
        L += ["", f"## Failed cases ({len(s['failed_cases'])}) - not in the dataset"]
        L += [f"- {f['batch']}/{f['case']}: {f['error']}" for f in s["failed_cases"]]
    L += ["", "## Gripper joints (state group gripper)"]
    for side, g in s["gripper_columns"].items():
        if not g["cases_toggling"]:
            L.append(f"- {side}: state column {g['state_col']}; the {side} gripper never closes in the data")
        else:
            L.append(f"- {side}: state column {g['state_col']}; follows the command with mean |corr| "
                     f"{g['mean_corr']:.2f} in the {g['cases_toggling']} episodes where it toggles"
                     f"{'' if g['mean_corr'] > 0.5 else '  <-- LOW: check GRIPPER_JOINTS against this recording'}")
    L += ["", "## Constant columns over the whole dataset",
          f"- action: {s['constant_action_columns'] or 'none'}",
          f"- state: {s['constant_state_columns']} of {STATE_DIM}"]
    L += ["", "## Video sync (wrist image motion vs gripper joint motion; lag 0 = in sync)"]
    by = defaultdict(list)
    for r in ok:
        for side, v in r.get("sync", {}).items():
            if v["lag"] is not None:
                by[(r["batch"], r["video_source"])].append(v["lag"])
    for (b, vs), lags in sorted(by.items()):
        c = Counter(lags)
        L.append(f"- {b} ({vs}): {len(lags)} measurements, lags {dict(c.most_common(6))}")
    L += ["", "batch1/batch3 cameras come from the same HDF5 step as the state, so their lags show the",
          "physical delay between joint motion and image change. batch2 should show the same lags.", "",
          "## Per case", "", "| batch | case | frames | source | dropped zip rows | sec |", "|---|---|---|---|---|---|"]
    for r in recs:
        if r["status"] == "ok":
            L.append(f"| {r['batch']} | {r['case']} | {r['frames']} | {r['video_source']} | "
                     f"{r.get('rows_dropped', '')} | {r['seconds']} |")
        else:
            L.append(f"| {r['batch']} | {r['case']} | FAILED | {r.get('error', '')[:60]} | | {r['seconds']} |")
    (work / "report.md").write_text("\n".join(L) + "\n")


VERSION = "3"   # bump when the written values change: cases of an older version are converted again


# =========================================================================== main
def self_test() -> None:
    """Checks that need no data: the action round trip, the quaternion order and the row matching."""
    rng = np.random.default_rng(0)
    q = rng.normal(size=(200, 2, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    env = np.concatenate([rng.normal(size=(200, 3)), q[:, 0], rng.normal(size=(200, 3)), q[:, 1],
                          rng.choice([-1.0, 1.0], size=(200, 2)), rng.normal(size=(200, 4))], axis=1)
    env[:, 16:20] = rng.normal(size=(200, 3)) @ np.asarray(WHEELS_FROM_BASE_VEL).T   # wheels of a real base motion
    back = action_to_env(build_action(env))
    for a, b in ((0, 3), (7, 10), (14, 20)):
        assert np.allclose(back[:, a:b], env[:, a:b], atol=1e-5), "positions, grippers or wheels changed"
    for a in (3, 10):   # q and -q are the same rotation, so compare matrices
        assert np.allclose(quat_to_mat(back[:, a:a + 4], "xyzw"), quat_to_mat(env[:, a:a + 4], "xyzw"), atol=1e-5)
    assert np.allclose(np.asarray(BASE_VEL_FROM_WHEELS) @ np.asarray(WHEELS_FROM_BASE_VEL), np.eye(3))
    assert np.allclose(np.asarray(BASE_VEL_FROM_WHEELS) @ np.full(4, 0.4), [0.4, 0.0, 0.0]), "equal wheels drive straight"
    assert action_to_env(build_action(env), wheels=False).shape == (200, 19)
    assert np.allclose(quat_to_mat(np.array([0.0, 0.0, 0.0, 1.0]), "xyzw"), np.eye(3))
    turn_z = quat_to_mat(np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]), "xyzw")
    assert np.allclose(turn_z @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-9), "xyzw: 90 deg about z turns x into y"
    assert np.allclose(rot6d_to_mat(mat_to_rot6d(turn_z[None])), turn_z[None])
    h5 = rng.normal(size=(300, 20))
    lr = np.concatenate([rng.normal(size=(8, 20)), h5[:250]]).astype(np.float32)   # 8 extra rows, then the demo
    assert np.array_equal(align(lr, h5.astype(np.float32)), np.r_[np.full(8, -1), np.arange(250)])
    joints = rng.normal(size=(5, N_JOINTS))
    state = build_state(rng.normal(size=(5, 2, 3)), q[:5], joints)
    assert state.shape == (5, STATE_DIM) and build_action(env).shape == (200, ACTION_DIM)
    assert np.allclose(state[:, 14:16], joints[:, [65, 69]]) and np.allclose(state[:, 0:7], joints[:, LEFT_ARM_JOINTS])
    assert [b - a for a, b in STATE_GROUPS.values()] == [7, 7, 2, 9, 9] and STATE_GROUPS["right_eef_world"][1] == STATE_DIM
    # the runtime package defines the same layout; compare when it is in this checkout
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    try:
        from astra import types as rt
    except ImportError:
        print("self-test: src/astra not in this checkout, layout not compared with the runtime")
    else:
        mine = {"STATE_DIM": STATE_DIM, "MODEL_ACTION_DIM": ACTION_DIM,
                **{f"STATE_{k.upper()}": slice(*v) for k, v in STATE_GROUPS.items()},
                **{f"MODEL_{k.upper()}": slice(*v) for k, v in ACTION_GROUPS.items()}}
        differ = [f"{k}: here {v}, there {getattr(rt, k, 'missing')}" for k, v in mine.items() if getattr(rt, k, None) != v]
        if not (np.allclose(getattr(rt, "WHEELS_FROM_BASE_VEL", 0), WHEELS_FROM_BASE_VEL)
                and np.allclose(getattr(rt, "BASE_VEL_FROM_WHEELS", 0), BASE_VEL_FROM_WHEELS)):
            differ.append("the wheel <-> base velocity matrices")
        if differ:
            raise SystemExit("self-test: the layout differs from src/astra/types.py in this checkout:\n  " + "\n  ".join(differ))
        print("self-test: layout matches src/astra/types.py")
    print("self-test passed")


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=Path, help="folder that holds the batch folders")
    ap.add_argument("output_dir", type=Path, help="LeRobot v2.1 dataset to write")
    ap.add_argument("--batches", nargs="+", help="only these batches, e.g. --batches batch1 batch2 [all]")
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 4)),
                    help="cases converted in parallel; each needs ~3 GB temp disk for batch1/3 [%(default)s]")
    ap.add_argument("--limit", type=int, help="convert at most N new cases per batch (for a quick test)")
    ap.add_argument("--task", default=DEFAULT_TASK, help="language instruction for every episode")
    ap.add_argument("--crf", type=int, default=20, help="H.264 quality, lower = better/larger [20]")
    ap.add_argument("--preset", default="veryfast", help="x264 preset [veryfast]")
    ap.add_argument("--gop", type=int, default=FPS, help="keyframe interval in frames [one per second]")
    ap.add_argument("--work-dir", type=Path, help="per-case results [OUTPUT_DIR.work]")
    ap.add_argument("--tmp", type=Path, help="temp dir for extracted HDF5 files [WORK_DIR/tmp]")
    ap.add_argument("--retry-failed", action="store_true", help="convert failed cases again")
    ap.add_argument("--redo", action="store_true", help="convert every case again")
    ap.add_argument("--merge-only", action="store_true", help="only rebuild OUTPUT_DIR from converted cases")
    opts = ap.parse_args()

    inp = opts.input_dir.resolve()
    out = opts.output_dir.resolve()
    work = (opts.work_dir or out.with_name(out.name + ".work")).resolve()
    ours = {"meta", "data", "videos", "galbot_config.py", "galbot_action_codec.py", "__pycache__"}
    if out.exists() and {p.name for p in out.iterdir() if not p.name.startswith(".")} - ours:
        raise SystemExit(f"{out} holds files this script did not write; choose an empty or new OUTPUT_DIR.")
    for d in (out, work / "cases", work / "tmp"):
        d.mkdir(parents=True, exist_ok=True)
    if opts.tmp:
        opts.tmp.mkdir(parents=True, exist_ok=True)

    if not opts.merge_only:
        cases = find_cases(inp, opts.batches)
        if not cases:
            raise SystemExit(f"No cases found under {inp}")
        todo, per_batch = [], Counter()
        for b, cid, p in cases:
            cj = work / "cases" / b / cid / "case.json"
            if cj.exists() and not opts.redo:
                r = json.loads(cj.read_text())
                same = r.get("source_size") == _size(p) and r.get("converter_version") == VERSION
                if same and (r["status"] == "ok" or not opts.retry_failed):
                    continue
            if opts.limit and per_batch[b] >= opts.limit:
                continue
            per_batch[b] += 1
            todo.append((b, cid, p))
        print(f"{len(cases)} cases found in {inp}; {len(todo)} to convert with {opts.workers} worker(s); "
              f"work dir {work}", flush=True)
        opts_d = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(opts).items()}
        done = 0
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=opts.workers) as ex:
            futs = {ex.submit(convert_case, b, cid, str(p), str(work), opts_d): (b, cid) for b, cid, p in todo}
            for f in as_completed(futs):
                done += 1
                try:
                    r = f.result()
                except Exception as exc:  # worker crashed (e.g. out of memory)
                    b, cid = futs[f]
                    r = {"batch": b, "case": cid, "status": "failed", "error": repr(exc), "seconds": None}
                eta = (time.time() - t0) / done * (len(todo) - done)
                msg = f"{r['frames']} frames, {r['seconds']} s" if r["status"] == "ok" else f"FAILED: {r.get('error')}"
                print(f"[{done}/{len(todo)}] {r['batch']}/{r['case']}: {msg}  (eta {eta / 60:.0f} min)", flush=True)

    s = merge(work, out, opts)
    print(f"\nDataset: {out}\n  {s['episodes']} episodes, {s['frames']} frames ({s['hours']} h); per batch "
          f"{ {b: v['episodes'] for b, v in s['per_batch'].items()} }")
    if s["failed_cases"]:
        print(f"  {len(s['failed_cases'])} failed cases left out (see {work / 'report.md'})")
    for side, g in s["gripper_columns"].items():
        print(f"  {side} gripper: state column {g['state_col']}, toggles in {g['cases_toggling']} episode(s)")
    if s["constant_action_columns"]:
        print(f"  constant action columns: {s['constant_action_columns']}")
    print(f"  report: {work / 'report.md'}\n  GR00T config: {out / 'galbot_config.py'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
