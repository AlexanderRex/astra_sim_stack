# Local policy container

This container runs the repository's current local policy client. It is a development workflow, not the official challenge submission service: the upstream repository describes that service as planned and has not published its transport or message schemas.

## Prepare a policy

1. Copy `chamber_policy.example.yml` to `chamber_policy.yml`.
2. Put your policy module and checkpoint in this directory. Update `policy_name` and `checkpoint` in the YAML.
3. Add model-only Python dependencies to `docker/policy-requirements.txt` and rebuild with `scripts/run_policy.sh`.

The policy class must implement the methods used by the local evaluator in `third_party/ecm2027_sim_track/policy/base.py`.

## Run local evaluation

Start the simulator IPC server in one terminal:

```bash
./scripts/launch.sh -- bash scripts/run_eval_env.sh
```

Then run the policy container from another host terminal:

```bash
./scripts/run_policy.sh
```

Both services use Linux host networking so the existing local IPC client at `127.0.0.1:50000` can reach the simulator server. They also share `host-ipc` as `TMPDIR`; PyTorch's CPU-tensor transfer uses a Unix socket there. Results are written to `host-eval_result/eval_results.json`.

To smoke-test the IPC/client/container path without a model, copy
`chamber_policy.smoke.yml` to `chamber_policy.yml` instead of the example above.
The smoke policy sends one zero action and is not intended to solve the task.
