"""Independent development policy for point-to-point base movement."""

import json
from pathlib import Path

from policy.base import BasePolicy

from base_drive import run_episode


class BaseDrivePolicy(BasePolicy):
    def get_model(self, usr_args):
        self.reports = []
        return None

    def reset_model(self):
        self.last_report = None

    def get_action(self):
        raise RuntimeError("BaseDrivePolicy owns its rollout in eval()")

    def eval(self, task_env, observation, usr_args, video_writer):
        if int(usr_args.get("env_cfg", {}).get("num_envs", 1)) != 1:
            raise ValueError("BaseDrivePolicy currently supports one environment")
        if int(usr_args["actions_dim"]) != 19:
            raise ValueError("BaseDrivePolicy requires the native 19-D body-velocity action interface")
        self.last_report = run_episode(task_env.step, observation, usr_args.get("base_drive", {}))
        self.reports.append(self.last_report)
        output = Path(usr_args["save_path"]) / "base_drive.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.last_report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print("base drive: " + json.dumps({k: v for k, v in self.last_report.items() if k != "trace"}), flush=True)
        return self.last_report["success"]

    def format_eval_results(self, results, usr_args, episodes_done):
        summary = super().format_eval_results(results, usr_args, episodes_done)
        summary["base_drive"] = self.reports
        return summary
