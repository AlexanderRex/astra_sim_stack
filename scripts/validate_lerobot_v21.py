# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyarrow", "pandas"]
# ///
"""
Validate a dataset written by dataset_conversion.py: the LeRobotDataset v2.1 layout, the values in
it, the stage instructions, and the GR00T files next to it.

Usage:
    uv run scripts/validate_lerobot_v21.py /path/to/dataset [--check-videos] [--tol 1e-4]
                                           [--skip-gr00t] [--min-stage-frames 10]

What is checked:
  layout   meta/info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl; one parquet per episode
           and one video per episode and camera; totals, chunks, splits, feature names
  frames   frame_index, global index, timestamps at 1/fps, task_index known, vector lengths
  values   no NaN/inf in state or action; every xyz+rot6d group (*_eef, *_eef_world) holds a real
           rotation (unit, orthogonal rows); action gripper only 0/1; state gripper in range;
           action columns that never change
  stats    episodes_stats.jsonl: count = episode length; min, max and mean of every numeric column
           = the parquet; camera stats per RGB channel, inside [0, 1]
  stages   annotation.human.task_description (what GR00T reads) = task_index on every row; stage
           episodes start at stage 0 and move forward one stage at a time; full-task episodes keep
           one task; episodes.jsonl tasks = the tasks used; conversion.json stage starts = the
           parquet; short stages and episodes that stop before the last stage are warned about
  GR00T    modality.json against info.json; galbot_config.py keys and action horizon;
           galbot_action_codec.py: STAGE_TASKS = tasks.jsonl, MODALITY = modality.json, margins =
           conversion.json, needed functions present (the file is read, not run)
  videos   with --check-videos (ffprobe from ffmpeg, slower): frame count, size and fps of each video

Exit code 0 = no errors (warnings may still be printed), 1 = errors found.
"""
import argparse
import ast
import json
import math
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REQUIRED_INFO_KEYS = [
    "codebase_version", "fps", "total_episodes", "total_frames", "total_tasks",
    "chunks_size", "data_path", "features",
]
REQUIRED_COLUMNS = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
ANNOTATION = "annotation.human.task_description"
N_STAGES_DEFAULT = 6
ROT6D_TOL = 1e-3            # unit length and orthogonality of the two rot6d rows
STATE_GRIPPER_RANGE = (-0.1, 2.0)   # rad; 0 open ... 1.703 closed
CODEC_NEEDS = {"StageTracker", "build_state", "action_to_env", "split_state"}


class Report:
    def __init__(self):
        self.errors, self.warnings, self.notes = [], [], []

    def err(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)


def read_jsonl(path, rep):
    rows = []
    if not path.exists():
        rep.err(f"Missing {path.relative_to(path.parents[1])}")
        return rows
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            rep.err(f"{path.name} line {n}: invalid JSON ({e})")
    return rows


def read_json(path, rep, required=False):
    if not path.exists():
        (rep.err if required else rep.warn)(f"Missing {path.name}")
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        rep.err(f"{path.name}: invalid JSON ({e})")
        return None


def py_file(path):
    """Literal constants, top-level function/class names and the syntax tree of a Python file,
    without running it."""
    tree = ast.parse(path.read_text())
    consts, defs = {}, set()
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    try:
                        consts[t.id] = ast.literal_eval(n.value)
                    except (ValueError, TypeError, SyntaxError):
                        pass
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            defs.add(n.name)
    return consts, defs, tree


def ffprobe_video(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets,width,height,r_frame_rate", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    try:
        s = json.loads(out.stdout)["streams"][0]
        num, den = s["r_frame_rate"].split("/")
        return {"frames": int(s["nb_read_packets"]), "width": int(s["width"]), "height": int(s["height"]),
                "fps": float(num) / float(den)}
    except (ValueError, KeyError, IndexError, ZeroDivisionError):
        return None


def col_array(df, c):
    """(n, d) float64 array of one parquet column, or None if it is not numeric."""
    v = df[c].to_numpy()
    try:
        if len(v) and isinstance(v[0], (np.ndarray, list)):
            return np.stack(v).astype(np.float64)
        return v.astype(np.float64).reshape(-1, 1)
    except (TypeError, ValueError):
        return None


def rot_groups(modality, part):
    """Groups that hold xyz + rot6d (9 values): *_eef and *_eef_world."""
    return [(k, g["start"]) for k, g in modality.get(part, {}).items()
            if g.get("end", 0) - g.get("start", 0) == 9 and (k.endswith("_eef") or k.endswith("_eef_world"))]


# =========================================================================== main checks
def validate(root: Path, check_videos: bool, tol: float, min_stage_frames: int, gr00t: bool) -> Report:
    rep = Report()
    meta = root / "meta"

    # ---- meta/info.json ---------------------------------------------------
    info_path = meta / "info.json"
    if not info_path.exists():
        rep.err("Missing meta/info.json")
        return rep
    info = json.loads(info_path.read_text())

    for k in REQUIRED_INFO_KEYS:
        if k not in info:
            rep.err(f"info.json missing key '{k}'")
    if rep.errors:
        return rep

    if info["codebase_version"] != "v2.1":
        rep.err(f"codebase_version is '{info['codebase_version']}', expected 'v2.1'")
    if (meta / "stats.json").exists() and not (meta / "episodes_stats.jsonl").exists():
        rep.warn("Found stats.json but no episodes_stats.jsonl - this looks like v2.0, not v2.1")

    fps = info["fps"]
    chunks_size = info["chunks_size"]
    features = info["features"]
    video_keys = [k for k, f in features.items() if f.get("dtype") == "video"]
    data_cols = [k for k, f in features.items() if f.get("dtype") not in ("video",)]

    for col in REQUIRED_COLUMNS:
        if col not in features:
            rep.err(f"info.json features missing required '{col}'")
    for k, f in features.items():
        if "dtype" not in f or "shape" not in f:
            rep.err(f"feature '{k}' must have 'dtype' and 'shape'")
            continue
        names = f.get("names")
        if f["dtype"] != "video" and isinstance(names, list) and len(f["shape"]) == 1 and len(names) != f["shape"][0]:
            rep.err(f"feature '{k}': {len(names)} names for shape {f['shape']}")
        if f["dtype"] == "video":
            vfps = f.get("info", {}).get("video.fps")
            if vfps is not None and abs(vfps - fps) > 1e-6:
                rep.err(f"feature '{k}': video.fps={vfps}, but info.fps={fps}")
    if video_keys and "video_path" not in info:
        rep.err("Video features present but info.json has no 'video_path'")
    if info.get("total_videos") is not None and video_keys:
        expected = info["total_episodes"] * len(video_keys)
        if info["total_videos"] != expected:
            rep.warn(f"total_videos={info['total_videos']} but episodes*cameras={expected}")
    n_eps = info["total_episodes"]
    if "total_chunks" in info and n_eps and info["total_chunks"] != math.ceil(n_eps / chunks_size):
        rep.warn(f"total_chunks={info['total_chunks']}, expected {math.ceil(n_eps / chunks_size)}")
    if info.get("splits", {}).get("train") not in (None, f"0:{n_eps}"):
        rep.warn(f"splits.train is '{info['splits']['train']}', expected '0:{n_eps}'")

    # ---- meta/*.jsonl -----------------------------------------------------
    episodes = read_jsonl(meta / "episodes.jsonl", rep)
    tasks = read_jsonl(meta / "tasks.jsonl", rep)
    ep_stats = read_jsonl(meta / "episodes_stats.jsonl", rep)

    ep_ids = [e.get("episode_index") for e in episodes]
    if ep_ids != list(range(len(episodes))):
        rep.err("episodes.jsonl episode_index values are not 0..N-1 in order")
    if len(episodes) != n_eps:
        rep.err(f"episodes.jsonl has {len(episodes)} rows, info.total_episodes={n_eps}")
    for e in episodes:
        if not {"episode_index", "tasks", "length"} <= e.keys():
            rep.err(f"episodes.jsonl row missing keys: {e}")

    task_ids = {t.get("task_index") for t in tasks}
    if task_ids != set(range(len(tasks))):
        rep.err("tasks.jsonl task_index values are not 0..N-1")
    if len(tasks) != info["total_tasks"]:
        rep.err(f"tasks.jsonl has {len(tasks)} rows, info.total_tasks={info['total_tasks']}")
    task_by_index = {t.get("task_index"): t.get("task") for t in tasks}
    dup = [s for s, c in Counter(task_by_index.values()).items() if c > 1]
    if dup:
        rep.err(f"tasks.jsonl lists the same instruction more than once: {dup}")
    task_strings = set(task_by_index.values())
    for e in episodes:
        for t in e.get("tasks", []):
            if t not in task_strings:
                rep.err(f"episode {e.get('episode_index')} uses task '{t}' not in tasks.jsonl")

    stats_ids = sorted(s.get("episode_index") for s in ep_stats)
    if stats_ids != list(range(len(episodes))):
        rep.err("episodes_stats.jsonl does not have exactly one entry per episode")
    for s in ep_stats:
        for feat, st in s.get("stats", {}).items():
            missing = {"min", "max", "mean", "std", "count"} - st.keys()
            if missing:
                rep.err(f"episodes_stats ep {s.get('episode_index')} '{feat}' missing {sorted(missing)}")
                break
    stats_by_ep = {s.get("episode_index"): s.get("stats", {}) for s in ep_stats}

    # ---- GR00T side files used by the content checks -------------------------
    modality = read_json(meta / "modality.json", rep) if gr00t else None
    modality = modality or {}
    conv = read_json(meta / "conversion.json", rep)
    codec_path = root / "galbot_action_codec.py"
    codec = py_file(codec_path) if codec_path.exists() else None
    stage_tasks = codec[0].get("STAGE_TASKS") if codec else None
    if not (isinstance(stage_tasks, list) and stage_tasks):
        stage_tasks = None

    single_task = len(tasks) == 1
    n_stage = len(stage_tasks) if stage_tasks else N_STAGES_DEFAULT
    if not single_task and not stage_tasks:
        rep.warn(f"galbot_action_codec.py has no STAGE_TASKS; assuming tasks 0..{n_stage - 1} are the stages")
    full_index = n_stage if n_stage in task_by_index else None
    if not single_task and len(tasks) > n_stage + 1:
        rep.err(f"tasks.jsonl has {len(tasks)} tasks; expected {n_stage} stages and at most one full-task instruction")
    conv_map = (conv or {}).get("episodes_map") or []

    ann_in_features = ANNOTATION in features
    if not ann_in_features:
        (rep.err if gr00t else rep.warn)(f"no '{ANNOTATION}' column: GR00T would get no instruction")

    # ---- per-episode parquet + video files ---------------------------------
    total_frames = 0
    expected_global_index = 0
    task_frames, task_episodes, kinds = Counter(), Counter(), Counter()
    col_lo, col_hi = {}, {}
    lengths = []
    for e in episodes:
        ep = e["episode_index"]
        chunk = ep // chunks_size
        length = e["length"]
        lengths.append(length)
        pq_path = root / info["data_path"].format(episode_chunk=chunk, episode_index=ep)
        if not pq_path.exists():
            rep.err(f"Missing parquet for episode {ep}: {pq_path.relative_to(root)}")
            continue

        df = pq.read_table(pq_path).to_pandas()
        total_frames += len(df)
        if len(df) != length:
            rep.err(f"ep {ep}: parquet has {len(df)} rows, episodes.jsonl length={length}")
        if not len(df):
            rep.err(f"ep {ep}: empty episode")
            continue

        missing_cols = [c for c in data_cols if c not in df.columns]
        if missing_cols:
            rep.err(f"ep {ep}: parquet missing columns {missing_cols}")
            continue
        extra = [c for c in df.columns if c not in features]
        if extra:
            rep.warn(f"ep {ep}: parquet has columns not in info.features: {extra}")

        if not (df["episode_index"] == ep).all():
            rep.err(f"ep {ep}: episode_index column has wrong values")
        if not np.array_equal(df["frame_index"].to_numpy(), np.arange(len(df))):
            rep.err(f"ep {ep}: frame_index is not 0..len-1")
        idx = df["index"].to_numpy()
        if not np.array_equal(idx, np.arange(expected_global_index, expected_global_index + len(df))):
            rep.err(f"ep {ep}: global 'index' not contiguous (expected start {expected_global_index})")
        expected_global_index += len(df)
        if not df["task_index"].isin(task_ids).all():
            rep.err(f"ep {ep}: task_index refers to unknown task")

        ts = df["timestamp"].to_numpy(dtype=np.float64)
        if len(ts) > 1:
            dt = np.diff(ts)
            bad = np.abs(dt - 1.0 / fps) > tol
            if bad.any():
                rep.err(f"ep {ep}: {bad.sum()} timestamp gaps differ from 1/fps={1/fps:.5f}s "
                        f"(min dt={dt.min():.5f}, max dt={dt.max():.5f})")
        if len(ts) and abs(ts[0]) > tol:
            rep.warn(f"ep {ep}: first timestamp is {ts[0]}, usually 0.0")

        arrays = {}
        for c in data_cols:
            a = col_array(df, c)
            if a is None:
                continue
            arrays[c] = a
            shape = features[c]["shape"]
            if len(shape) == 1 and a.shape[1] != shape[0] and not (shape[0] == 1 and a.shape[1] == 1):
                rep.err(f"ep {ep}: '{c}' length {a.shape[1]} != declared shape {shape}")
                arrays.pop(c)

        check_values(ep, arrays, modality, rep, col_lo, col_hi)
        check_stats(ep, arrays, length, stats_by_ep.get(ep), video_keys, rep)
        used = check_stages(ep, df, e, task_by_index, n_stage, full_index, single_task, min_stage_frames,
                            conv_map, rep, kinds)
        for k in used:
            task_episodes[k] += 1
        for k, c in Counter(df["task_index"].to_numpy().tolist()).items():
            task_frames[k] += c

        for vk in video_keys:
            vp = root / info["video_path"].format(episode_chunk=chunk, video_key=vk, episode_index=ep)
            if not vp.exists():
                rep.err(f"Missing video for ep {ep}, {vk}: {vp.relative_to(root)}")
            elif check_videos:
                v = ffprobe_video(vp)
                if v is None:
                    rep.warn(f"ep {ep} {vk}: could not read the video with ffprobe")
                    continue
                if abs(v["frames"] - length) > 1:
                    rep.err(f"ep {ep} {vk}: video has {v['frames']} frames, episode length={length}")
                h, w = features[vk]["shape"][:2]
                if (v["height"], v["width"]) != (h, w):
                    rep.err(f"ep {ep} {vk}: video is {v['width']}x{v['height']}, info.json says {w}x{h}")
                if abs(v["fps"] - fps) > 0.01:
                    rep.err(f"ep {ep} {vk}: video runs at {v['fps']:.3f} fps, info.fps={fps}")

    if total_frames != info["total_frames"]:
        rep.err(f"Sum of parquet rows={total_frames}, info.total_frames={info['total_frames']}")

    if "action" in col_lo:
        const = np.flatnonzero(col_hi["action"] - col_lo["action"] < 1e-6)
        if const.size:
            names = features["action"].get("names") or []
            rep.warn("action columns that never change in the whole dataset: "
                     + ", ".join(str(names[i]) if i < len(names) else str(i) for i in const))

    # ---- summary ------------------------------------------------------------
    rep.notes.append(f"{len(episodes)} episodes, {total_frames} frames ({total_frames / fps / 3600:.2f} h at {fps} fps)")
    if not single_task:
        rep.notes.append(f"stage-labelled episodes: {kinds['stages']}, full-task episodes: {kinds['full']}")
    for k in sorted(task_by_index):
        rep.notes.append(f"  task {k}: {task_frames[k]:>9} frames {task_frames[k] / fps / 3600:7.2f} h "
                         f"in {task_episodes[k]:>5} episodes  {task_by_index[k]!r}")

    if gr00t:
        validate_gr00t(root, info, modality, task_by_index, conv, codec, min(lengths, default=0), rep)
    return rep


def check_values(ep, arrays, modality, rep, col_lo, col_hi):
    for col, part in (("observation.state", "state"), ("action", "action")):
        x = arrays.get(col)
        if x is None:
            continue
        bad = ~np.isfinite(x)
        if bad.any():
            rep.err(f"ep {ep}: {int(bad.any(1).sum())} rows of {col} hold NaN or inf")
            continue
        col_lo[col] = np.minimum(col_lo.get(col, x.min(0)), x.min(0))
        col_hi[col] = np.maximum(col_hi.get(col, x.max(0)), x.max(0))
        for name, s in rot_groups(modality, part):
            if s + 9 > x.shape[1]:
                continue
            r = x[:, s + 3:s + 9].reshape(-1, 2, 3)
            worst = max(float(np.abs(np.linalg.norm(r, axis=2) - 1).max()),
                        float(np.abs((r[:, 0] * r[:, 1]).sum(1)).max()))
            if worst > ROT6D_TOL:
                rep.err(f"ep {ep}: {part}.{name} rot6d is not a rotation (off by {worst:.2g})")
        g = modality.get(part, {}).get("gripper")
        if g and g["end"] <= x.shape[1]:
            v = x[:, g["start"]:g["end"]]
            if part == "action" and not np.isin(v, (0.0, 1.0)).all():
                rep.err(f"ep {ep}: action.gripper holds values other than 0 (open) and 1 (closed)")
            if part == "state" and (v.min() < STATE_GRIPPER_RANGE[0] or v.max() > STATE_GRIPPER_RANGE[1]):
                rep.warn(f"ep {ep}: state.gripper spans [{v.min():.3f}, {v.max():.3f}], "
                         f"expected about 0 (open) .. 1.7 (closed)")


def check_stats(ep, arrays, length, st, video_keys, rep):
    if st is None:
        return
    for col, x in arrays.items():
        s = st.get(col)
        if s is None:
            rep.warn(f"ep {ep}: episodes_stats has no entry for '{col}'")
            continue
        if list(s.get("count", [])) != [length]:
            rep.err(f"ep {ep}: episodes_stats '{col}' count={s.get('count')}, episode length={length}")
        for k, f in (("min", np.min), ("max", np.max), ("mean", np.mean)):
            ref = np.asarray(s.get(k, []), dtype=np.float64).reshape(-1)
            got = f(x, axis=0)
            if ref.shape != got.shape or not np.allclose(ref, got, rtol=1e-5, atol=1e-5):
                rep.err(f"ep {ep}: episodes_stats '{col}' {k} does not match the parquet")
                break
    for vk in video_keys:
        s = st.get(vk)
        if s is None:
            rep.err(f"ep {ep}: episodes_stats has no entry for camera '{vk}'")
            continue
        ok = True
        for k in ("min", "max", "mean", "std"):
            a = np.asarray(s.get(k, []), dtype=np.float64)
            ok &= a.shape == (3, 1, 1) and bool(((a >= 0) & (a <= 1)).all())
        if not ok:
            rep.err(f"ep {ep}: episodes_stats '{vk}' is not per-channel (3, 1, 1) values in [0, 1]")


def check_stages(ep, df, e, task_by_index, n_stage, full_index, single_task, min_stage_frames, conv_map,
                 rep, kinds):
    """Returns the task indices this episode uses."""
    ti = df["task_index"].to_numpy()
    if ANNOTATION in df.columns:
        ann = df[ANNOTATION].to_numpy()
        mism = int((ann != ti).sum())
        if mism:
            rep.err(f"ep {ep}: {ANNOTATION} differs from task_index on {mism} rows "
                    f"(first at frame {int(np.argmax(ann != ti))}); GR00T reads the annotation column")
    used = sorted(set(ti.tolist()))
    if sorted(set(e.get("tasks", []))) != sorted(task_by_index.get(k) for k in used if k in task_by_index):
        rep.err(f"ep {ep}: episodes.jsonl tasks differ from the task_index values in the parquet")
    if single_task:
        kinds["single"] += 1
        return used

    if full_index is not None and (ti == full_index).any():
        kinds["full"] += 1
        if not (ti == full_index).all():
            rep.err(f"ep {ep}: mixes the full-task instruction with stage instructions")
        return used
    kinds["stages"] += 1
    if ti[0] != 0:
        rep.err(f"ep {ep}: starts in stage {ti[0] + 1}, not in stage 1 ({task_by_index.get(0)!r})")
    d = np.diff(ti)
    if (d < 0).any():
        rep.err(f"ep {ep}: goes back a stage {int((d < 0).sum())} time(s), first at frame {int(np.argmax(d < 0)) + 1}")
    if (d > 1).any():
        rep.err(f"ep {ep}: skips a stage at frame {int(np.argmax(d > 1)) + 1} "
                f"({ti[int(np.argmax(d > 1))] + 1} -> {ti[int(np.argmax(d > 1)) + 1] + 1})")
    if ti[-1] != n_stage - 1:
        rep.warn(f"ep {ep}: ends in stage {ti[-1] + 1} of {n_stage}")
    counts = np.bincount(ti, minlength=n_stage)
    short = [f"{k + 1} ({counts[k]} frames)" for k in used if counts[k] < min_stage_frames]
    if short:
        rep.warn(f"ep {ep}: very short stage(s): {', '.join(short)}")

    if ep < len(conv_map):
        m = conv_map[ep]
        if m.get("episode_index") != ep:
            rep.err(f"conversion.json episodes_map[{ep}] is for episode {m.get('episode_index')}")
        starts = m.get("stage_starts")
        if starts is not None:
            got = {task_by_index[k]: int(np.argmax(ti == k)) for k in used if k in task_by_index}
            if starts != got:
                rep.err(f"ep {ep}: stage starts in conversion.json differ from the parquet")
    return used


# =========================================================================== GR00T files
def validate_gr00t(root, info, modality, task_by_index, conv, codec, min_len, rep):
    """modality.json against info.json; galbot_config.py and galbot_action_codec.py against both."""
    feats = info["features"]
    if not modality:
        rep.err("meta/modality.json is missing or empty (rerun dataset_conversion.py with --merge-only)")
        return
    for part in ("state", "action", "video", "annotation"):
        if part not in modality:
            rep.err(f"modality.json has no '{part}' section")
    for part, col in (("state", "observation.state"), ("action", "action")):
        if part not in modality or col not in feats:
            continue
        dim = feats[col]["shape"][0]
        covered = set()
        for k, g in modality[part].items():
            s, e = g.get("start"), g.get("end")
            if not (isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= dim):
                rep.err(f"{part}.{k}: [{s}, {e}) is not a valid slice of {col} (dim {dim})")
            else:
                covered.update(range(s, e))
        if len(covered) != dim:
            rep.warn(f"{part} columns not in any group: {sorted(set(range(dim)) - covered)}")
    for k, v in modality.get("video", {}).items():
        if feats.get(v.get("original_key"), {}).get("dtype") != "video":
            rep.err(f"video.{k}: '{v.get('original_key')}' is not a video feature in info.json")
    if "human.task_description" not in modality.get("annotation", {}):
        rep.err("modality.json annotation has no 'human.task_description'")
    for k, v in modality.get("annotation", {}).items():
        col = v.get("original_key", f"annotation.{k}")
        if col not in feats:
            rep.err(f"annotation.{k}: column '{col}' not in info.json features")

    # galbot_config.py
    cp = root / "galbot_config.py"
    if not cp.exists():
        rep.warn("galbot_config.py not found next to the dataset (needed for --modality-config-path)")
    else:
        consts, _, tree = py_file(cp)
        for name, part in (("STATE_KEYS", "state"), ("ACTION_KEYS", "action")):
            keys = consts.get(name)
            if keys is None:
                rep.warn(f"galbot_config.py: could not read {name}; check its keys by hand")
                continue
            for k in keys:
                if k not in modality.get(part, {}):
                    rep.err(f"galbot_config.py {name} uses '{k}', which is not in modality.json '{part}'")
        known = set(modality.get("video", {})) | {f"annotation.{k}" for k in modality.get("annotation", {})}
        literal_keys = []
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "modality_keys" and isinstance(node.value, ast.List):
                try:
                    literal_keys += ast.literal_eval(node.value)
                except ValueError:
                    pass
        for k in literal_keys:
            if k not in known:
                rep.err(f"galbot_config.py uses modality key '{k}', not a video or annotation key of modality.json")
        if ANNOTATION not in literal_keys:
            rep.warn(f"galbot_config.py does not list '{ANNOTATION}' as its language key")
        horizon = consts.get("ACTION_HORIZON")
        if isinstance(horizon, int) and min_len and min_len < horizon:
            rep.warn(f"shortest episode has {min_len} frames, less than ACTION_HORIZON={horizon}")

    # galbot_action_codec.py
    if codec is None:
        rep.warn("galbot_action_codec.py not found next to the dataset (the evaluation needs it)")
    else:
        consts, defs, _ = codec
        missing = CODEC_NEEDS - defs
        if missing:
            rep.err(f"galbot_action_codec.py lacks {sorted(missing)}")
        st = consts.get("STAGE_TASKS")
        if isinstance(st, list) and len(task_by_index) > 1:
            for i, s in enumerate(st):
                if task_by_index.get(i) != s:
                    rep.err(f"galbot_action_codec.py STAGE_TASKS[{i}] = {s!r}, tasks.jsonl task {i} = "
                            f"{task_by_index.get(i)!r}; the evaluation would send instructions never trained on")
        if consts.get("MODALITY") is not None and consts["MODALITY"] != modality:
            rep.err("galbot_action_codec.py MODALITY differs from meta/modality.json")
        sl = (conv or {}).get("stage_labels") or {}
        for name, key in (("RELEASE_MARGIN", "release_margin"), ("MAX_WAIT", "max_wait")):
            if key in sl and consts.get(name) != sl[key]:
                rep.err(f"galbot_action_codec.py {name}={consts.get(name)}, but the labels used {sl[key]}")

    # conversion.json
    if conv:
        if conv.get("episodes") != info["total_episodes"]:
            rep.err(f"conversion.json episodes={conv.get('episodes')}, info.total_episodes={info['total_episodes']}")
        if "tasks" in conv and conv["tasks"] != [task_by_index.get(i) for i in range(len(task_by_index))]:
            rep.err("conversion.json tasks differ from tasks.jsonl")
        unl = (conv.get("stage_labels") or {}).get("unlabeled_cases") or []
        if unl:
            rep.warn(f"{len(unl)} case(s) had no usable stage flags "
                     f"({conv['stage_labels'].get('unlabeled_policy')}); see the conversion report")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="Dataset folder (contains meta/, data/, videos/)")
    ap.add_argument("--check-videos", action="store_true", help="Compare video frames, size and fps via ffprobe")
    ap.add_argument("--tol", type=float, default=1e-4, help="Timestamp tolerance in seconds")
    ap.add_argument("--skip-gr00t", action="store_true", help="Leave out the GR00T files and modality checks")
    ap.add_argument("--min-stage-frames", type=int, default=10, help="Warn about stages shorter than this [10]")
    args = ap.parse_args()

    if args.check_videos and not shutil.which("ffprobe"):
        sys.exit("ffprobe not found; install ffmpeg or drop --check-videos")

    rep = validate(args.root, args.check_videos, args.tol, args.min_stage_frames, not args.skip_gr00t)
    for n in rep.notes:
        print(n)
    if rep.notes:
        print()
    for w in rep.warnings:
        print(f"WARNING: {w}")
    for e in rep.errors:
        print(f"ERROR:   {e}")
    if rep.errors:
        print(f"\nFAILED: {len(rep.errors)} error(s), {len(rep.warnings)} warning(s)")
        sys.exit(1)
    print(f"\nOK: valid dataset ({len(rep.warnings)} warning(s))")


if __name__ == "__main__":
    main()
