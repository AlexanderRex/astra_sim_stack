"""Zero-action policy that checks the client-server path. Not a solution to the task."""

import os
import time

import numpy as np
import torch
from policy.base import BasePolicy

SECONDS = float(os.environ.get("SMOKE_SECONDS", "30"))


class SmokePolicy(BasePolicy):
    def get_model(self, usr_args):
        return None

    def get_action(self):
        return None

    def reset_model(self):
        pass

    def eval(self, task_env, observation, usr_args, video_writer):
        num_envs = int(usr_args.get("env_cfg", {}).get("num_envs", 1))
        action = torch.zeros((num_envs, int(usr_args["actions_dim"])), dtype=torch.float32)
        deadline = time.monotonic() + SECONDS
        steps = 0
        while time.monotonic() < deadline:
            _, _, terminated, truncated, _ = task_env.step(action)
            steps += 1
            if np.asarray(terminated).any() or np.asarray(truncated).any():
                break
        print(f"smoke policy: {steps} steps in {SECONDS:.0f} s", flush=True)
        return False
