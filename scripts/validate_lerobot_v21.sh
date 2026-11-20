#!/usr/bin/env bash
# Validate a dataset written by convert_demos.sh (LeRobot v2.1 for GR00T). Runs on the host; no containers.
# Wraps scripts/validate_lerobot_v21.py:
#   1. makes sure the Python environment has numpy, pyarrow and pandas (adds what is missing, never wipes it)
#   2. checks the dataset folder and prints its size, episodes and cameras
#   3. validates the LeRobot v2.1 layout (meta, indices, timestamps, files, stats; optionally video frames)
#   4. checks the GR00T files: meta/modality.json against info.json, and galbot_config.py keys
# Exit code 0 = valid, 1 = errors found, 2 = usage error.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
validator="${repo_root}/scripts/validate_lerobot_v21.py"
VENV_DIR="${VENV_DIR:-${repo_root}/.venvs}"
venv="${VENV_DIR}/dataset"

usage() {
    cat <<USAGE
Usage: scripts/validate_lerobot_v21.sh [options] DATASET_DIR

  DATASET_DIR    LeRobot v2.1 dataset written by convert_demos.sh (contains meta/ data/ videos/)

Options:
  --check-videos   also compare each video's frame count with its episode length (needs ffprobe; slower)
  --tol S          timestamp tolerance in seconds                                   [1e-4]
  --skip-gr00t     only the LeRobot v2.1 checks, not modality.json / galbot_config.py
  -h, --help       show this help

Environment:
  VENV_DIR         host Python environments                                         [<repo>/.venvs]

Examples:
  scripts/validate_lerobot_v21.sh /ds/galbot_gr00t_v21
  scripts/validate_lerobot_v21.sh --check-videos /ds/galbot_gr00t_v21
USAGE
}

fail() {
    echo "$*" >&2
    exit 1
}

step() {
    echo
    echo "==> $*"
}

passthrough=()
positional=()
skip_gr00t=0
while (($# > 0)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --check-videos) passthrough+=("$1") ;;
        --tol)
            [[ $# -ge 2 ]] || fail "--tol needs a value"
            passthrough+=("$1" "$2"); shift ;;
        --skip-gr00t) skip_gr00t=1 ;;
        -*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
        *) positional+=("$1") ;;
    esac
    shift
done
((${#positional[@]} == 1)) || { usage >&2; exit 2; }

dataset="$(realpath -m "${positional[0]}")"
[[ -d "${dataset}" ]] || fail "DATASET_DIR does not exist: ${dataset}"
[[ -f "${validator}" ]] || fail "Missing ${validator}"
[[ -f "${dataset}/meta/info.json" ]] || fail "No meta/info.json in ${dataset} - not a LeRobot dataset (or the conversion did not finish)"
command -v uv >/dev/null 2>&1 || fail "Required command is missing: uv (https://docs.astral.sh/uv/)"
if [[ " ${passthrough[*]-} " == *" --check-videos "* ]]; then
    command -v ffprobe >/dev/null 2>&1 || fail "--check-videos needs ffprobe (install ffmpeg)"
fi

# --- 1. Python environment ---------------------------------------------------------------
step "Python environment ${venv}"
[[ -x "${venv}/bin/python" ]] || uv venv --python 3.12 "${venv}"
if ! "${venv}/bin/python" -c 'import numpy, pyarrow, pandas' 2>/dev/null; then
    # add packages only: the venv may also hold lerobot / GR00T tools
    uv pip install --python "${venv}/bin/python" numpy pyarrow pandas
fi
"${venv}/bin/python" -c 'import numpy, pyarrow, pandas; print(f"numpy {numpy.__version__}, pyarrow {pyarrow.__version__}, pandas {pandas.__version__}")'

# --- 2. Dataset --------------------------------------------------------------------------
step "Dataset ${dataset}"
"${venv}/bin/python" - "${dataset}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
info = json.loads((root / "meta" / "info.json").read_text())
cams = [k for k, f in info["features"].items() if f.get("dtype") == "video"]
print(f"  codebase_version {info.get('codebase_version')}, robot {info.get('robot_type')}, fps {info.get('fps')}")
print(f"  {info.get('total_episodes')} episodes, {info.get('total_frames')} frames "
      f"({info.get('total_frames', 0) / max(info.get('fps', 1), 1) / 3600:.2f} h)")
for k in ("observation.state", "action"):
    if k in info["features"]:
        print(f"  {k}: {info['features'][k]['shape']}")
for c in cams:
    print(f"  camera {c}: {info['features'][c]['shape']}")
meta = sorted(p.name for p in (root / "meta").iterdir())
print(f"  meta/: {', '.join(meta)}")
PY
echo "  size: $(du -sh "${dataset}" 2>/dev/null | cut -f1)"

status=0

# --- 3. LeRobot v2.1 layout --------------------------------------------------------------
step "LeRobot v2.1 checks${passthrough[*]:+ (${passthrough[*]})}"
"${venv}/bin/python" "${validator}" "${dataset}" "${passthrough[@]}" || status=1

# --- 4. GR00T files ----------------------------------------------------------------------
if ((skip_gr00t == 0)); then
    step "GR00T checks (meta/modality.json, galbot_config.py)"
    "${venv}/bin/python" - "${dataset}" <<'PY' || status=1
import ast, json, sys
from pathlib import Path

root = Path(sys.argv[1])
info = json.loads((root / "meta" / "info.json").read_text())
feats = info["features"]
errors, warnings = [], []

mp = root / "meta" / "modality.json"
if not mp.exists():
    errors.append("meta/modality.json is missing (rerun convert_demos.sh with --merge-only)")
    modality = None
else:
    modality = json.loads(mp.read_text())
    for part in ("state", "action", "video", "annotation"):
        if part not in modality:
            errors.append(f"modality.json has no '{part}' section")
    for part, col in (("state", "observation.state"), ("action", "action")):
        if part not in modality or col not in feats:
            continue
        dim = feats[col]["shape"][0]
        covered = set()
        for k, g in modality[part].items():
            s, e = g.get("start"), g.get("end")
            if not (isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= dim):
                errors.append(f"{part}.{k}: [{s}, {e}) is not a valid slice of {col} (dim {dim})")
            else:
                covered.update(range(s, e))
        if part == "action" and len(covered) != dim:
            warnings.append(f"action columns not in any group: {sorted(set(range(dim)) - covered)}")
        groups = ", ".join("%s[%s:%s]" % (k, g.get("start"), g.get("end")) for k, g in modality[part].items())
        print(f"  {part}: {groups}")
    for k, v in modality.get("video", {}).items():
        ok = feats.get(v.get("original_key"), {}).get("dtype") == "video"
        print(f"  video: {k} <- {v.get('original_key')}")
        if not ok:
            errors.append(f"video.{k}: '{v.get('original_key')}' is not a video feature in info.json")
    for k, v in modality.get("annotation", {}).items():
        col = v.get("original_key", f"annotation.{k}")
        print(f"  annotation: {k} <- {col}")
        if col not in feats:
            errors.append(f"annotation.{k}: column '{col}' not in info.json features")

cp = root / "galbot_config.py"
if not cp.exists():
    warnings.append("galbot_config.py not found next to the dataset (needed for --modality-config-path)")
elif modality is not None:
    tree = ast.parse(cp.read_text())
    consts = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and isinstance(n.value, (ast.List, ast.Constant))}
    for name, part in (("STATE_KEYS", "state"), ("ACTION_KEYS", "action")):
        keys = consts.get(name)
        if keys is None:
            warnings.append(f"galbot_config.py: could not read {name}; check its keys by hand")
            continue
        print(f"  config {name}: {keys}")
        for k in keys:
            if k not in modality.get(part, {}):
                errors.append(f"galbot_config.py {name} uses '{k}', which is not in modality.json '{part}'")

for w in warnings:
    print(f"WARNING: {w}")
for e in errors:
    print(f"ERROR:   {e}")
print(f"\n{'FAILED' if errors else 'OK'}: GR00T files ({len(errors)} error(s), {len(warnings)} warning(s))")
sys.exit(1 if errors else 0)
PY
fi

# --- Result ------------------------------------------------------------------------------
step "Result"
if ((status == 0)); then
    echo "VALID: ${dataset}"
    echo "Next: open it in Rerun (rerun ${dataset}) or start a short GR00T fine-tuning run."
else
    echo "INVALID: ${dataset} - see the errors above."
fi
exit "${status}"
