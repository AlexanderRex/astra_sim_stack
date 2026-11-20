#!/usr/bin/env bash
# Convert the task repo's demonstration dataset into one LeRobot v2.1 dataset for GR00T.
# Runs on the host; no containers. Wraps scripts/convert_demos.py:
#   1. creates a Python environment with h5py, numpy, pyarrow and PyAV (once, reused)
#   2. checks the input, free disk space and the output folder
#   3. converts every case (resumable: a rerun only converts new or changed cases) and builds the dataset
#   4. prints the report
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
converter="${repo_root}/scripts/convert_demos.py"
VENV_DIR="${VENV_DIR:-${repo_root}/.venvs}"
venv="${VENV_DIR}/dataset"

usage() {
    cat <<USAGE
Usage: scripts/convert_demos.sh [options] INPUT_DIR OUTPUT_DIR

  INPUT_DIR    the dataset folder (batch1/ batch2/ batch3/), or the dataset root containing it
  OUTPUT_DIR   LeRobot v2.1 dataset to write; work files go to OUTPUT_DIR.work next to it

Options (passed to convert_demos.py):
  --batches B1,B2      only these batches, e.g. --batches batch1,batch2      [all]
  --workers N          cases converted in parallel (~3 GB temp disk each)    [auto]
  --limit N            at most N new cases per batch, for a quick test
  --tmp DIR            temp dir for extracted HDF5 files                     [OUTPUT_DIR.work/tmp]
  --retry-failed       convert failed cases again
  --merge-only         only rebuild OUTPUT_DIR from cases already converted
  --task TEXT          language instruction for every episode
  -h, --help           show this help

Environment:
  VENV_DIR             host Python environments                              [<repo>/.venvs]

Examples:
  scripts/convert_demos.sh --limit 1 /ds/raw /ds/galbot_gr00t_v21
  scripts/convert_demos.sh --workers 4 /ds/raw /ds/galbot_gr00t_v21
  scripts/convert_demos.sh --batches batch3 /ds/raw /ds/galbot_gr00t_v21
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
while (($# > 0)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --retry-failed|--merge-only|--redo) passthrough+=("$1") ;;
        --workers|--limit|--tmp|--task|--crf|--preset|--gop|--work-dir)
            [[ $# -ge 2 ]] || fail "$1 needs a value"
            passthrough+=("$1" "$2"); shift ;;
        --batches)
            [[ $# -ge 2 ]] || fail "--batches needs a value, e.g. batch1,batch2"
            IFS=',' read -r -a batch_list <<< "$2"
            passthrough+=(--batches "${batch_list[@]}"); shift ;;
        -*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
        *) positional+=("$1") ;;
    esac
    shift
done
((${#positional[@]} == 2)) || { usage >&2; exit 2; }

input_dir="$(realpath -m "${positional[0]}")"
output_dir="$(realpath -m "${positional[1]}")"
[[ -d "${input_dir}" ]] || fail "INPUT_DIR does not exist: ${input_dir}"
[[ -f "${converter}" ]] || fail "Missing ${converter}"
[[ -d "${input_dir}/Roco_ChamberTask" ]] && input_dir="${input_dir}/Roco_ChamberTask"
compgen -G "${input_dir}/batch*" >/dev/null || fail "No batch folders in ${input_dir}"

command -v uv >/dev/null 2>&1 || fail "Required command is missing: uv (https://docs.astral.sh/uv/)"

# --- 1. Python environment ---------------------------------------------------------------
step "Python environment ${venv}"
if ! "${venv}/bin/python" -c 'import h5py, numpy, pyarrow, av' 2>/dev/null; then
    uv venv --clear --python 3.12 "${venv}"
    uv pip install --python "${venv}/bin/python" h5py numpy pyarrow av
fi
"${venv}/bin/python" -c 'import av, sys; sys.exit(0 if "libx264" in av.codecs_available or "h264" in av.codecs_available else 1)' \
    || fail "PyAV in ${venv} has no H.264 encoder."
"${venv}/bin/python" -c 'import h5py, pyarrow, av; print(f"h5py {h5py.__version__}, pyarrow {pyarrow.__version__}, av {av.__version__}")'

# --- 2. Checks ---------------------------------------------------------------------------
step "Input ${input_dir}"
for b in "${input_dir}"/batch*/; do
    n="$(find "${b}" -maxdepth 1 \( -name 'ChamberServiceTask_*.tar' -o -name 'ChamberServiceTask_*.tar.gz' \
         -o -name 'ChamberServiceTask_*.tgz' -o \( -type d -name 'ChamberServiceTask_*' \) \) | wc -l)"
    echo "  $(basename "${b}"): ${n} cases, $(du -sh "${b}" 2>/dev/null | cut -f1)"
done

mkdir -p "${output_dir}"
work_parent="$(dirname "${output_dir}")"
free_gb="$(df -Pk "${work_parent}" | awk 'NR==2 {printf "%d", $4 / 1048576}')"
echo "  free disk at ${work_parent}: ${free_gb} GB"
((free_gb >= 20)) || fail "Less than 20 GB free at ${work_parent}; the output needs a few % of the input plus ~3 GB per worker."

# --- 3. Convert --------------------------------------------------------------------------
step "Converting (rerun the same command to resume or to add new cases)"
"${venv}/bin/python" "${converter}" "${input_dir}" "${output_dir}" "${passthrough[@]}"

# --- 4. Report ---------------------------------------------------------------------------
report="${output_dir}.work/report.md"
if [[ -f "${report}" ]]; then
    step "Report ${report}"
    sed -n '1,/^## Per case/p' "${report}" | sed '$d'
fi
step "Done"
du -sh "${output_dir}"
echo "Dataset for GR00T:   ${output_dir}"
echo "Modality config:     ${output_dir}/galbot_config.py   (use with --modality-config-path)"
echo "Upload only ${output_dir} to the training machine, not ${output_dir}.work."
