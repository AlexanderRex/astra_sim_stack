#!/usr/bin/env bash
# Steps per second of the headless env server for each given CPU set, to choose SIM_CPUSET.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat <<'USAGE'
Usage: scripts/bench_sim.sh CPUSET...

For each CPU set, runs scripts/run_headless.sh with src/smoke.yml (zero actions
for 30 s, three cameras) with the env server pinned to that set, and prints the
server's steps per second. "all" runs it unpinned. Each run takes 1 to 2 minutes.

Example: scripts/bench_sim.sh all 0-3 0-7
Then export the fastest set as SIM_CPUSET for scripts/start_sim.sh and
scripts/run_headless.sh.
USAGE
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    "") usage >&2; exit 2 ;;
esac

for set in "$@"; do
    cpuset="${set}"
    [[ "${set}" == all ]] && cpuset=""
    log="$(mktemp --tmpdir bench_sim.XXXXXX.log)"
    if SIM_CPUSET="${cpuset}" "${repo_root}/scripts/run_headless.sh" "${repo_root}/src/smoke.yml" >"${log}" 2>&1 \
        && result="$(grep -o 'smoke policy: [0-9]* steps in [0-9]* s' "${log}" | tail -n 1)" \
        && [[ -n "${result}" ]]; then
        read -r steps seconds < <(awk '{print $3, $6}' <<<"${result}")
        awk -v s="${set}" -v n="${steps}" -v t="${seconds}" 'BEGIN { printf "%-12s %5.1f steps/s\n", s, n / t }'
        rm -f "${log}"
    else
        echo "${set}: run failed, log in ${log}"
    fi
done
