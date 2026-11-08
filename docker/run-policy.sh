#!/usr/bin/env bash
set -euo pipefail

policy_config="${POLICY_CONFIG:-/workspace/policy_submission/chamber_policy.yml}"
if [[ ! -f "$policy_config" ]]; then
    echo "Policy config not found: $policy_config" >&2
    echo "Copy policy_submission/chamber_policy.example.yml to chamber_policy.yml and edit it." >&2
    exit 2
fi

exec bash scripts/run_policy_eval.sh "$policy_config" "$@"
