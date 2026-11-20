# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyarrow", "pandas"]
# ///
"""
Validate that a local dataset folder follows the LeRobotDataset v2.1 layout, and that the GR00T
files written by convert_dataset.py (meta/modality.json, galbot_config.py) agree with it.

Usage:
    uv run scripts/validate_lerobot_v21.py /path/to/dataset [--check-videos] [--tol 1e-4] [--skip-gr00t]

--check-videos uses ffprobe (from ffmpeg) to compare each video's frame count
with the episode length. It is slower, so it is optional.

Exit code 0 = no errors (warnings may still be printed), 1 = errors found.
"""
import argparse
import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REQUIRED_INFO_KEYS = [
    "codebase_version", "fps", "total_episodes", "total_frames", "total_tasks",
    "chunks_size", "data_path", "features",
]
REQUIRED_COLUMNS = ["timestamp", "frame_index", "episode_index", "index", "task_index"]


class Report:
    def __init__(self):
        self.errors, self.warnings = [], []

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


def ffprobe_frames(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return int(out.stdout.strip().split(",")[0])
    except ValueError:
        return None


def validate(root: Path, check_videos: bool, tol: float) -> Report:
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
    if video_keys and "video_path" not in info:
        rep.err("Video features present but info.json has no 'video_path'")
    if info.get("total_videos") is not None and video_keys:
        expected = info["total_episodes"] * len(video_keys)
        if info["total_videos"] != expected:
            rep.warn(f"total_videos={info['total_videos']} but episodes*cameras={expected}")

    # ---- meta/*.jsonl -----------------------------------------------------
    episodes = read_jsonl(meta / "episodes.jsonl", rep)
    tasks = read_jsonl(meta / "tasks.jsonl", rep)
    ep_stats = read_jsonl(meta / "episodes_stats.jsonl", rep)

    ep_ids = [e.get("episode_index") for e in episodes]
    if ep_ids != list(range(len(episodes))):
        rep.err("episodes.jsonl episode_index values are not 0..N-1 in order")
    if len(episodes) != info["total_episodes"]:
        rep.err(f"episodes.jsonl has {len(episodes)} rows, info.total_episodes={info['total_episodes']}")
    for e in episodes:
        if not {"episode_index", "tasks", "length"} <= e.keys():
            rep.err(f"episodes.jsonl row missing keys: {e}")

    task_ids = {t.get("task_index") for t in tasks}
    if task_ids != set(range(len(tasks))):
        rep.err("tasks.jsonl task_index values are not 0..N-1")
    if len(tasks) != info["total_tasks"]:
        rep.err(f"tasks.jsonl has {len(tasks)} rows, info.total_tasks={info['total_tasks']}")
    task_strings = {t.get("task") for t in tasks}
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

    # ---- per-episode parquet + video files ---------------------------------
    total_frames = 0
    expected_global_index = 0
    for e in episodes:
        ep = e["episode_index"]
        chunk = ep // chunks_size
        length = e["length"]
        pq_path = root / info["data_path"].format(episode_chunk=chunk, episode_index=ep)
        if not pq_path.exists():
            rep.err(f"Missing parquet for episode {ep}: {pq_path.relative_to(root)}")
            continue

        df = pq.read_table(pq_path).to_pandas()
        total_frames += len(df)
        if len(df) != length:
            rep.err(f"ep {ep}: parquet has {len(df)} rows, episodes.jsonl length={length}")

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

        # vector shape check on first row
        for c in data_cols:
            shape = features[c]["shape"]
            if c in df.columns and len(df) and len(shape) == 1 and shape[0] > 1:
                val = df[c].iloc[0]
                if hasattr(val, "__len__") and len(val) != shape[0]:
                    rep.err(f"ep {ep}: '{c}' length {len(val)} != declared shape {shape}")

        for vk in video_keys:
            vp = root / info["video_path"].format(episode_chunk=chunk, video_key=vk, episode_index=ep)
            if not vp.exists():
                rep.err(f"Missing video for ep {ep}, {vk}: {vp.relative_to(root)}")
            elif check_videos:
                n = ffprobe_frames(vp)
                if n is None:
                    rep.warn(f"ep {ep} {vk}: could not read frame count")
                elif abs(n - length) > 1:
                    rep.err(f"ep {ep} {vk}: video has {n} frames, episode length={length}")

    if total_frames != info["total_frames"]:
        rep.err(f"Sum of parquet rows={total_frames}, info.total_frames={info['total_frames']}")

    return rep


def validate_gr00t(root: Path, rep: Report) -> None:
    """meta/modality.json against info.json, and the keys of galbot_config.py against modality.json."""
    feats = json.loads((root / "meta" / "info.json").read_text())["features"]
    mp = root / "meta" / "modality.json"
    if not mp.exists():
        rep.err("meta/modality.json is missing (rerun convert_dataset.py with --merge-only)")
        return
    modality = json.loads(mp.read_text())
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
        if part == "action" and len(covered) != dim:
            rep.warn(f"action columns not in any group: {sorted(set(range(dim)) - covered)}")
    for k, v in modality.get("video", {}).items():
        if feats.get(v.get("original_key"), {}).get("dtype") != "video":
            rep.err(f"video.{k}: '{v.get('original_key')}' is not a video feature in info.json")
    for k, v in modality.get("annotation", {}).items():
        col = v.get("original_key", f"annotation.{k}")
        if col not in feats:
            rep.err(f"annotation.{k}: column '{col}' not in info.json features")

    cp = root / "galbot_config.py"
    if not cp.exists():
        rep.warn("galbot_config.py not found next to the dataset (needed for --modality-config-path)")
        return
    consts = {t.id: ast.literal_eval(n.value) for n in ast.parse(cp.read_text()).body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and isinstance(n.value, (ast.List, ast.Constant))}
    for name, part in (("STATE_KEYS", "state"), ("ACTION_KEYS", "action")):
        keys = consts.get(name)
        if keys is None:
            rep.warn(f"galbot_config.py: could not read {name}; check its keys by hand")
            continue
        for k in keys:
            if k not in modality.get(part, {}):
                rep.err(f"galbot_config.py {name} uses '{k}', which is not in modality.json '{part}'")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="Dataset folder (contains meta/, data/, videos/)")
    ap.add_argument("--check-videos", action="store_true", help="Compare video frame counts via ffprobe")
    ap.add_argument("--tol", type=float, default=1e-4, help="Timestamp tolerance in seconds")
    ap.add_argument("--skip-gr00t", action="store_true", help="Only the LeRobot v2.1 checks")
    args = ap.parse_args()

    if args.check_videos and not shutil.which("ffprobe"):
        sys.exit("ffprobe not found; install ffmpeg or drop --check-videos")

    rep = validate(args.root, args.check_videos, args.tol)
    if not args.skip_gr00t and (args.root / "meta" / "info.json").exists():
        validate_gr00t(args.root, rep)
    for w in rep.warnings:
        print(f"WARNING: {w}")
    for e in rep.errors:
        print(f"ERROR:   {e}")
    if rep.errors:
        print(f"\nFAILED: {len(rep.errors)} error(s), {len(rep.warnings)} warning(s)")
        sys.exit(1)
    print(f"\nOK: valid LeRobot v2.1 layout ({len(rep.warnings)} warning(s))")


if __name__ == "__main__":
    main()
